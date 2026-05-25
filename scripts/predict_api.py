"""Generate ESConv or CPsyCounD test predictions with a hosted LLM API.

The API endpoint, key, and model are read from config/llm_api.yaml:
    predict_base_url
    predict_api_key
    model_predict

Input defaults:
    data/instruction/esc/test.json
    data/instruction/cpsy/test.json

Output layout:
    data/training/predict/<dataset>/<model_name>/generated_predictions.jsonl

The output JSONL keeps the same core fields as local baseline prediction files:
    {"prompt": "...", "predict": "...", "label": "..."}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_client import LLMClient  # noqa: E402

try:
    from predict_esc_baseline import (  # type: ignore
        build_eval_prompt as build_esc_eval_prompt,
        build_generation_prompt as build_esc_generation_prompt,
        clean_response,
        load_records,
        project_path,
        read_done_indices,
    )
except ModuleNotFoundError:
    from scripts.predict_esc_baseline import (
        build_eval_prompt as build_esc_eval_prompt,
        build_generation_prompt as build_esc_generation_prompt,
        clean_response,
        load_records,
        project_path,
        read_done_indices,
    )

try:
    from predict_cpsy import (  # type: ignore
        build_eval_prompt as build_cpsy_eval_prompt,
        build_generation_prompt as build_cpsy_generation_prompt,
    )
except ModuleNotFoundError:
    from scripts.predict_cpsy import (
        build_eval_prompt as build_cpsy_eval_prompt,
        build_generation_prompt as build_cpsy_generation_prompt,
    )


def normalize_dataset(value: str) -> str:
    name = value.lower()
    aliases = {
        "esc": "esc",
        "esconv": "esc",
        "cpsy": "cpsy",
        "cpsycound": "cpsy",
        "cpsycoun": "cpsy",
        "cpsycound": "cpsy",
    }
    if name not in aliases:
        raise ValueError(f"unsupported dataset {value!r}; use esc or cpsy")
    return aliases[name]


def safe_model_dir_name(model_name: str) -> str:
    name = model_name.strip().replace("/", "_").replace(":", "_")
    name = re.sub(r"\s+", "_", name)
    return name or "api_model"


def default_input_file(dataset: str) -> str:
    return f"data/instruction/{dataset}/test.json"


def default_language(dataset: str) -> str:
    return "English" if dataset == "esc" else "中文"


def build_prompts(
    dataset: str,
    record: dict[str, Any],
    *,
    include_factors: bool,
    language: str,
) -> tuple[str, str]:
    if dataset == "esc":
        return (
            build_esc_generation_prompt(
                record,
                model_type="api",
                include_factors=include_factors,
                language=language,
            ),
            build_esc_eval_prompt(record),
        )
    if dataset == "cpsy":
        return (
            build_cpsy_generation_prompt(
                record,
                model_type="api",
                include_factors=include_factors,
                language=language,
            ),
            build_cpsy_eval_prompt(record),
        )
    raise ValueError(f"unsupported dataset: {dataset}")


def system_prompt_for(dataset: str) -> str:
    if dataset == "esc":
        return (
            "You are a professional emotional support counselor. "
            "Generate only the next supporter response."
        )
    return (
        "你是一名专业、审慎、温暖的心理咨询师。"
        "只生成咨询师的下一轮回复，不要输出解释。"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="Dataset name: esc/esconv or cpsy/cpsycound.")
    ap.add_argument(
        "--model-name",
        default=None,
        help="Name used for output folder. Default: model_predict from llm_api.yaml.",
    )
    ap.add_argument("--input-file", default=None, help="Default: data/instruction/<dataset>/test.json.")
    ap.add_argument("--output-dir", default=None, help="Default: data/training/predict/<dataset>/<model-name>.")
    ap.add_argument("--config", default=None, help="Optional config/llm_api.yaml override path.")
    ap.add_argument("--include-factors", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--language", default=None, help="Default: English for ESConv, 中文 for CPsyCounD.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Build prompts but do not call the API.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=2560)
    ap.add_argument("--keep-errors", action="store_true", help="Write failed API calls as JSONL error rows.")
    args = ap.parse_args()

    dataset = normalize_dataset(args.dataset)
    client = LLMClient(args.config)
    api_model_name = client._model_for("predict")
    model_name = safe_model_dir_name(args.model_name or api_model_name)
    language = args.language or default_language(dataset)

    input_file = project_path(args.input_file or default_input_file(dataset))
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else ROOT / "data" / "training" / "predict" / dataset / model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "generated_predictions.jsonl"
    meta_file = output_dir / "generation_config.json"

    if args.overwrite:
        output_file.unlink(missing_ok=True)
        meta_file.unlink(missing_ok=True)

    records = load_records(input_file)
    selected = [
        (idx, record)
        for idx, record in enumerate(records)
        if idx >= args.start_index and (args.limit is None or idx < args.start_index + args.limit)
    ]

    print(f"Loaded {len(records)} records from {input_file}")
    print(f"Selected {len(selected)} records starting at index {args.start_index}")
    print(f"Dataset: {dataset}")
    print(f"API model: {api_model_name}")
    print(f"Output model folder: {model_name}")
    print(f"Output: {output_file}")

    if args.dry_run:
        if selected:
            idx, record = selected[0]
            model_prompt, eval_prompt = build_prompts(
                dataset,
                record,
                include_factors=args.include_factors,
                language=language,
            )
            print(f"\n--- Generation prompt preview for index {idx} ---")
            print(model_prompt[:3000])
            print("\n--- Stored eval prompt preview ---")
            print(eval_prompt[:2000])
        return

    done = read_done_indices(output_file)
    selected_indices = {idx for idx, _ in selected}
    remaining = [(idx, record) for idx, record in selected if idx not in done]
    print(f"Resume state: {len(done & selected_indices)} completed, {len(remaining)} remaining")

    meta = {
        "dataset": dataset,
        "model_name": model_name,
        "api_model_name": api_model_name,
        "input_file": str(input_file),
        "include_factors": args.include_factors,
        "language": language,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "config": args.config or "config/llm_api.yaml",
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    system = system_prompt_for(dataset)
    with output_file.open("a", encoding="utf-8") as out:
        with tqdm(
            total=len(selected),
            initial=len(done & selected_indices),
            desc=f"API predict {dataset}/{model_name}",
            unit="sample",
            dynamic_ncols=True,
        ) as pbar:
            for idx, record in remaining:
                pbar.set_postfix(index=idx, refresh=False)
                model_prompt, eval_prompt = build_prompts(
                    dataset,
                    record,
                    include_factors=args.include_factors,
                    language=language,
                )
                try:
                    raw = client.chat(
                        system,
                        model_prompt,
                        role="predict",
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        top_p=args.top_p,
                    )
                    pred = clean_response(raw)
                    row = {
                        "index": idx,
                        "prompt": eval_prompt,
                        "model_prompt": model_prompt,
                        "predict": pred,
                        "label": str(record.get("output", "")).strip(),
                        "api_model_name": api_model_name,
                    }
                except Exception as exc:
                    if not args.keep_errors:
                        raise
                    row = {
                        "index": idx,
                        "error": repr(exc),
                        "prompt": eval_prompt,
                        "model_prompt": model_prompt,
                        "predict": "",
                        "label": str(record.get("output", "")).strip(),
                        "api_model_name": api_model_name,
                    }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                pbar.update(1)

    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    main()
