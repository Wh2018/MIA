"""Evaluate ESConv predictions with an LLM-as-a-judge protocol.

Default input layout:
    data/training/predict/esc/<MODEL_NAME>/generated_predictions.jsonl

Each prediction line is expected to contain:
    {"prompt": "...", "predict": "...", "label": "..."}

Outputs:
    llm_judge/gpt-5-mini/per_sample.jsonl
    llm_judge/gpt-5-mini/summary.json

Each API call returns all five dimension scores for one sample.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_client import LLMClient  # noqa: E402


DEFAULT_DIMS = ("empathy", "identification", "informativeness", "guidance", "coherence")
CHATML_USER_RE = re.compile(r"<\|im_start\|>user\n(?P<user>.*?)(?:<\|im_end\|>)", re.S)
TAG_RE = re.compile(r"<(?P<tag>Facts|Causes|Results|Beliefs|Intentions|Desires|Emotions)>(?P<body>.*?)</(?P=tag)>", re.S)


@dataclass(frozen=True)
class Sample:
    index: int
    prompt: str
    context: str
    prediction: str
    reference: str


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_prompt_template(prompt_dir: Path) -> str:
    path = prompt_dir / "all_dimensions.md"
    if not path.is_file():
        raise FileNotFoundError(f"missing combined prompt template: {path}")
    return path.read_text(encoding="utf-8")


def extract_user_payload(prompt: str) -> str:
    matches = list(CHATML_USER_RE.finditer(prompt))
    if not matches:
        return prompt.strip()
    return matches[-1].group("user").strip()


def parse_context(prompt: str) -> str:
    user_payload = extract_user_payload(prompt)
    factors: dict[str, str] = {}
    for m in TAG_RE.finditer(user_payload):
        factors[m.group("tag")] = " ".join(m.group("body").split())

    utterance = TAG_RE.sub("", user_payload)
    utterance = re.sub(r"\s+", " ", utterance).strip()

    lines = []
    if utterance:
        lines.append(f"Current seeker utterance: {utterance}")

    factor_order = [
        ("Facts", "Factual facts"),
        ("Causes", "Factual causes"),
        ("Results", "Possible factual results"),
        ("Beliefs", "Personal beliefs"),
        ("Intentions", "Personal intentions"),
        ("Desires", "Personal desires"),
        ("Emotions", "Personal emotions"),
    ]
    for key, label in factor_order:
        value = factors.get(key, "").strip()
        if value:
            lines.append(f"{label}: {value}")

    return "\n".join(lines).strip()


def load_samples(prediction_file: Path, *, limit: int | None = None, start_index: int = 0) -> list[Sample]:
    samples: list[Sample] = []
    with prediction_file.open("r", encoding="utf-8") as f:
        for raw_index, line in enumerate(f):
            if raw_index < start_index:
                continue
            if limit is not None and len(samples) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = str(obj.get("prompt", ""))
            samples.append(
                Sample(
                    index=raw_index,
                    prompt=prompt,
                    context=parse_context(prompt),
                    prediction=str(obj.get("predict", "")).strip(),
                    reference=str(obj.get("label", "")).strip(),
                )
            )
    return samples


def render_prompt(template: str, sample: Sample) -> str:
    return (
        template.replace("{{CONVERSATION_HISTORY}}", sample.context)
        .replace("{{REFERENCE_ANSWER}}", sample.reference)
        .replace("{{GENERATED_RESPONSE}}", sample.prediction)
    )


def parse_judge_scores(text: str, dimensions: list[str]) -> dict[str, int]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.S)
        if not match:
            raise ValueError(f"judge response is not JSON: {text[:300]}")
        data = json.loads(match.group(0))

    scores: dict[str, int] = {}
    for dim in dimensions:
        if dim not in data:
            raise ValueError(f"judge response missing {dim}: {text[:300]}")
        score = int(data[dim])
        if score not in {0, 1, 2, 3}:
            raise ValueError(f"judge score for {dim} out of range 0..3: {score}")
        scores[dim] = score
    return scores


def write_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_done_keys(path: Path) -> set[tuple[int, str]]:
    done: set[tuple[int, str]] = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "index" in obj and "dimension" in obj and "error" not in obj:
                done.add((int(obj["index"]), str(obj["dimension"])))
    return done


def read_done_samples(path: Path, dimensions: list[str]) -> set[int]:
    done: set[int] = set()
    if not path.is_file():
        return done
    required = set(dimensions)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in obj or "index" not in obj:
                continue
            scores = obj.get("scores")
            if isinstance(scores, dict) and required.issubset(scores):
                done.add(int(obj["index"]))
                continue
            if all(dim in obj for dim in dimensions):
                done.add(int(obj["index"]))
    return done


def summarize(results_file: Path, summary_file: Path, dimensions: list[str]) -> dict[str, Any]:
    by_dim: dict[str, list[int]] = {dim: [] for dim in dimensions}
    per_sample: dict[int, dict[str, int]] = {}
    errors = 0

    if results_file.is_file():
        with results_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if "error" in obj:
                    errors += 1
                    continue
                index = int(obj["index"])
                if isinstance(obj.get("scores"), dict):
                    for dim in dimensions:
                        if dim in obj["scores"]:
                            score = int(obj["scores"][dim])
                            by_dim.setdefault(dim, []).append(score)
                            per_sample.setdefault(index, {})[dim] = score
                    continue
                if "dimension" in obj and "score" in obj:
                    dim = str(obj["dimension"])
                    score = int(obj["score"])
                    by_dim.setdefault(dim, []).append(score)
                    per_sample.setdefault(index, {})[dim] = score

    complete_avgs = []
    for dim_scores in per_sample.values():
        if all(dim in dim_scores for dim in dimensions):
            complete_avgs.append(mean(dim_scores[dim] for dim in dimensions))

    average_scores = {
        dim: mean(scores) if scores else None
        for dim, scores in by_dim.items()
    }
    valid_dimension_means = [v for v in average_scores.values() if v is not None]

    summary = {
        "num_records": sum(len(v) for v in by_dim.values()),
        "num_complete_samples": len(complete_avgs),
        "num_errors": errors,
        "average_scores": average_scores,
        "overall_average": mean(valid_dimension_means) if valid_dimension_means else None,
        "overall_average_of_complete_samples": mean(complete_avgs) if complete_avgs else None,
        "dimensions": {
            dim: {
                "count": len(scores),
                "mean": mean(scores) if scores else None,
                "score_histogram": {str(i): scores.count(i) for i in range(4)},
            }
            for dim, scores in by_dim.items()
        },
    }
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input-dir",
        default="data/training/predict/esc/MIA",
        help="Directory containing generated_predictions.jsonl.",
    )
    ap.add_argument("--prediction-file", default=None, help="Override input JSONL path.")
    ap.add_argument(
        "--prompt-dir",
        default="config/evaluation_prompts/esc",
        help="Directory containing all_dimensions.md.",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: <input-dir>/llm_judge/<model_judge>.",
    )
    ap.add_argument("--config", default=None, help="Optional config/llm_api.yaml override path.")
    ap.add_argument("--dimensions", nargs="+", default=list(DEFAULT_DIMS), choices=list(DEFAULT_DIMS))
    ap.add_argument("--limit", type=int, default=None, help="Evaluate only the first N selected samples.")
    ap.add_argument("--start-index", type=int, default=0, help="Skip samples before this raw JSONL index.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=5120)
    ap.add_argument("--overwrite", action="store_true", help="Delete previous output files before running.")
    ap.add_argument("--dry-run", action="store_true", help="Parse inputs and render one prompt without API calls.")
    ap.add_argument("--keep-errors", action="store_true", help="Write failed judge calls to output JSONL.")
    args = ap.parse_args()

    input_dir = project_path(args.input_dir)
    prediction_file = project_path(args.prediction_file) if args.prediction_file else input_dir / "generated_predictions.jsonl"
    prompt_dir = project_path(args.prompt_dir)

    if not prediction_file.is_file():
        raise FileNotFoundError(f"missing prediction file: {prediction_file}")

    client = None if args.dry_run else LLMClient(args.config)
    judge_model = "dry-run"
    if client is not None:
        judge_model = client._model_for("eval").replace("/", "_").replace(":", "_")

    output_dir = project_path(args.output_dir) if args.output_dir else input_dir / "llm_judge" / judge_model
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "per_sample.jsonl"
    summary_file = output_dir / "summary.json"

    if args.overwrite:
        results_file.unlink(missing_ok=True)
        summary_file.unlink(missing_ok=True)

    samples = load_samples(prediction_file, limit=args.limit, start_index=args.start_index)
    template = load_prompt_template(prompt_dir)

    if args.dry_run:
        print(f"Loaded {len(samples)} samples from {prediction_file}")
        if samples:
            sample = samples[0]
            print("\n--- Parsed context ---")
            print(sample.context)
            print("\n--- Rendered prompt preview ---")
            print(render_prompt(template, sample)[:3000])
        return

    done_samples = read_done_samples(results_file, args.dimensions)
    selected_indices = {sample.index for sample in samples}
    done_in_selection = done_samples & selected_indices
    tasks = [sample for sample in samples if sample.index not in done_samples]
    total_expected = len(samples)

    print(
        f"Loaded {len(samples)} samples from {prediction_file}\n"
        f"Judgments: {total_expected} total = one API call per sample, returning {len(args.dimensions)} dimensions\n"
        f"Resume state: {len(done_in_selection)} already completed, {len(tasks)} remaining\n"
        f"Output: {results_file}"
    )

    system = (
        "You are a strict but fair evaluator for emotional support conversations. "
        "Return only valid JSON exactly as requested by the user prompt."
    )

    with tqdm(
        total=total_expected,
        initial=len(done_in_selection),
        desc=f"LLM judge ({judge_model})",
        unit="sample",
        dynamic_ncols=True,
    ) as pbar:
        for sample in tasks:
            pbar.set_postfix(sample=sample.index, refresh=False)
            user_prompt = render_prompt(template, sample)
            try:
                raw = client.chat(
                    system,
                    user_prompt,
                    role="eval",
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    top_p=1.0,
                )
                scores = parse_judge_scores(raw, args.dimensions)
                record = {
                    "index": sample.index,
                    "scores": scores,
                    "raw_response": raw,
                    "prediction": sample.prediction,
                    "reference": sample.reference,
                }
            except Exception as exc:
                if not args.keep_errors:
                    raise
                record = {
                    "index": sample.index,
                    "error": repr(exc),
                    "prediction": sample.prediction,
                    "reference": sample.reference,
                }
            write_jsonl_record(results_file, record)
            pbar.update(1)

    summary = summarize(results_file, summary_file, args.dimensions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved per-sample judgments to {results_file}")
    print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
