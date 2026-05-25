"""Generate CPsyCounD test predictions with local psychological-counseling baselines.

Input:
    data/instruction/cpsy/test.json

Output layout:
    data/training/predict/cpsy/<MODEL_NAME>/generated_predictions.jsonl

The output JSONL keeps the same core fields as LLaMA-Factory prediction files:
    {"prompt": "...", "predict": "...", "label": "..."}

Supported model types:
    mechat          models/MeChat-style AutoModel.chat()
    psyllm          Qwen3/PsyLLM AutoModelForCausalLM with optional thinking
    soulchat_qwen2  SoulChat2.0-Qwen2-7B / Qwen2 chat-template causal LM
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from predict_esc_baseline import (  # type: ignore
        ROOT,
        infer_default_type,
        load_records,
        make_generator,
        project_path,
        read_done_indices,
    )
except ModuleNotFoundError:
    from scripts.predict_esc_baseline import (
        ROOT,
        infer_default_type,
        load_records,
        make_generator,
        project_path,
        read_done_indices,
    )


TAG_RE = re.compile(
    r"<(?P<tag>Facts|Causes|Results|Beliefs|Intentions|Desires|Emotions|Approach|Explanation)>"
    r"(?P<body>.*?)</(?P=tag)>",
    re.S,
)


def parse_instruction_tags(instruction: str) -> dict[str, str]:
    return {
        match.group("tag"): " ".join(match.group("body").split())
        for match in TAG_RE.finditer(instruction or "")
    }


def build_eval_prompt(record: dict[str, Any]) -> str:
    """Store a ChatML-style prompt so the CPsy judge script can parse context."""
    instruction = str(record.get("instruction", "")).strip()
    user_input = str(record.get("input", "")).strip()
    return (
        "<|im_start|>system\n"
        "You are a helpful psychological counseling assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_readable_context(record: dict[str, Any]) -> str:
    factors = parse_instruction_tags(str(record.get("instruction", "")))
    labels = [
        ("Facts", "事实信息"),
        ("Causes", "可能成因"),
        ("Results", "可能结果"),
        ("Beliefs", "信念"),
        ("Intentions", "意图"),
        ("Desires", "需求"),
        ("Emotions", "情绪"),
        ("Approach", "建议回应策略"),
        ("Explanation", "策略解释"),
    ]
    lines = []
    for key, label in labels:
        if factors.get(key):
            lines.append(f"{label}: {factors[key]}")
    return "\n".join(lines)


def build_generation_prompt(record: dict[str, Any], *, model_type: str, include_factors: bool, language: str) -> str:
    user_input = str(record.get("input", "")).strip()
    context = build_readable_context(record) if include_factors else ""
    language = language.strip() or "中文"

    language_rule = f"请使用{language}生成咨询师的下一轮回复。"
    context_block = f"\n已知背景与心理状态推断：\n{context}\n" if context else ""

    if model_type == "mechat":
        return f"""现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如共情回应、澄清、开放式提问、动机访谈和问题探索。{language_rule}回复应温暖、自然、专业，并适合多轮心理咨询场景。避免空泛赞美、机械说教、过度诊断和泄露隐私。只输出咨询师的下一句回复，不要输出解释。
{context_block}
来访者：{user_input}
咨询师："""

    return f"""你是一名专业、审慎、温暖的心理咨询师。
{language_rule}
根据来访者当前发言给出下一轮咨询师回复。可以使用已知背景和心理状态推断，但不要提及 XML 标签、字段名或隐藏标注。回复需要体现理解、尊重和专业引导，避免空泛赞美、机械说教、过度诊断和隐私泄露。只输出咨询师的下一句回复，不要输出解释。
{context_block}
来访者：{user_input}
咨询师："""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", required=True, help="Name used for output folder, e.g. MeChat.")
    ap.add_argument("--model-path", default=None, help="Local model directory. Default: models/<model-name>.")
    ap.add_argument("--model-type", choices=["auto", "mechat", "psyllm", "soulchat_qwen2"], default="auto")
    ap.add_argument("--input-file", default="data/instruction/cpsy/test.json")
    ap.add_argument("--output-dir", default=None, help="Default: data/training/predict/cpsy/<model-name>.")
    ap.add_argument("--include-factors", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--language", default="中文", help="Response language instruction, default: 中文.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Build prompts but do not load the model.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda:0", help="Only used by MeChat AutoModel.chat().")
    ap.add_argument("--device-map", default="auto", help="Used by AutoModelForCausalLM baselines.")
    ap.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    ap.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    args = ap.parse_args()

    if args.model_type == "auto":
        args.model_type = infer_default_type(args.model_name)
    if args.model_path is None:
        args.model_path = str(Path("models") / args.model_name)

    input_file = project_path(args.input_file)
    output_dir = (
        project_path(args.output_dir)
        if args.output_dir
        else ROOT / "data" / "training" / "predict" / "cpsy" / args.model_name
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
    print(f"Model: {args.model_name} ({args.model_type}) from {project_path(args.model_path)}")
    print(f"Output: {output_file}")

    if args.dry_run:
        if selected:
            idx, record = selected[0]
            print(f"\n--- Generation prompt preview for index {idx} ---")
            print(
                build_generation_prompt(
                    record,
                    model_type=args.model_type,
                    include_factors=args.include_factors,
                    language=args.language,
                )[:3000]
            )
            print("\n--- Stored eval prompt preview ---")
            print(build_eval_prompt(record)[:2000])
        return

    generator = make_generator(args)
    done = read_done_indices(output_file)
    selected_indices = {idx for idx, _ in selected}
    remaining = [(idx, record) for idx, record in selected if idx not in done]
    print(f"Resume state: {len(done & selected_indices)} completed, {len(remaining)} remaining")

    meta = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "model_path": str(project_path(args.model_path)),
        "input_file": str(input_file),
        "include_factors": args.include_factors,
        "language": args.language,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with output_file.open("a", encoding="utf-8") as out:
        with tqdm(
            total=len(selected),
            initial=len(done & selected_indices),
            desc=f"Predict CPsy {args.model_name}",
            unit="sample",
            dynamic_ncols=True,
        ) as pbar:
            for idx, record in remaining:
                pbar.set_postfix(index=idx, refresh=False)
                prompt = build_generation_prompt(
                    record,
                    model_type=args.model_type,
                    include_factors=args.include_factors,
                    language=args.language,
                )
                pred = generator.generate(
                    prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                )
                row = {
                    "index": idx,
                    "prompt": build_eval_prompt(record),
                    "model_prompt": prompt,
                    "predict": pred,
                    "label": str(record.get("output", "")).strip(),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                pbar.update(1)

    print(f"Saved predictions to {output_file}")


if __name__ == "__main__":
    main()
