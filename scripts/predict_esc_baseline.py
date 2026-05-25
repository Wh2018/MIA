"""Generate ESConv test predictions with local emotional-support baselines.

Input:
    data/instruction/esc/test.json

Output layout:
    data/training/predict/esc/<MODEL_NAME>/generated_predictions.jsonl

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
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
TAG_RE = re.compile(r"<(?P<tag>Facts|Causes|Results|Beliefs|Intentions|Desires|Emotions)>(?P<body>.*?)</(?P=tag)>", re.S)


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    return data


def parse_factor_text(instruction: str) -> dict[str, str]:
    return {
        m.group("tag"): " ".join(m.group("body").split())
        for m in TAG_RE.finditer(instruction or "")
    }


def build_eval_prompt(record: dict[str, Any]) -> str:
    """Store a ChatML-style prompt so the judge script can parse context."""
    instruction = str(record.get("instruction", "")).strip()
    user_input = str(record.get("input", "")).strip()
    return (
        "<|im_start|>system\n"
        "You are a helpful emotional support assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}\n{user_input}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_readable_context(record: dict[str, Any]) -> str:
    factors = parse_factor_text(str(record.get("instruction", "")))
    labels = [
        ("Facts", "Facts"),
        ("Causes", "Causes"),
        ("Results", "Possible results"),
        ("Beliefs", "Beliefs"),
        ("Intentions", "Intentions"),
        ("Desires", "Desires"),
        ("Emotions", "Emotions"),
    ]
    lines = []
    for key, label in labels:
        if factors.get(key):
            lines.append(f"{label}: {factors[key]}")
    return "\n".join(lines)


def build_generation_prompt(record: dict[str, Any], *, model_type: str, include_factors: bool, language: str) -> str:
    user_input = str(record.get("input", "")).strip()
    context = build_readable_context(record) if include_factors else ""

    if language.lower() == "english":
        language_rule = "Generate the next counselor/supporter response in English."
    else:
        language_rule = f"Generate the next counselor/supporter response in {language}."

    if model_type == "mechat":
        context_block = f"\n已知背景：\n{context}\n" if context else ""
        return f"""现在你扮演一位专业的心理咨询师，你具备丰富的心理学和心理健康知识。你擅长运用多种心理咨询技巧，例如共情回应、动机访谈技巧和问题探索。请以温暖亲切的语气，展现出对来访者感受的理解，避免空泛赞美或教导式回应。{language_rule} 只输出咨询师的下一句回复，不要输出解释。
{context_block}
对话：
来访者：{user_input}
咨询师："""

    context_block = f"\nContext inferred from previous turns:\n{context}\n" if context else ""
    return f"""You are a professional emotional support counselor.
{language_rule}
Use the provided context only when it helps you respond more accurately. Do not mention XML tags, factor names, or hidden annotations. Keep the response natural, supportive, concise, and suitable for a multi-turn emotional support conversation.
{context_block}
Seeker: {user_input}
Supporter:"""


def clean_response(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    text = re.sub(r"^(咨询师|Counselor|Supporter|Assistant)\s*[:：]\s*", "", text, flags=re.I).strip()
    stop_markers = [
        "\n来访者：",
        "\n来访者:",
        "\nSeeker:",
        "\nUser:",
        "\nClient:",
        "\n咨询师：",
        "\nCounselor:",
        "\nSupporter:",
    ]
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker, 1)[0].strip()
    return text


class MeChatGenerator:
    def __init__(self, model_path: Path, *, device: str, trust_remote_code: bool):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)
        if device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.half().to(device)
        else:
            self.model = self.model.to("cpu")
        self.model.eval()

    def generate(self, prompt: str, *, temperature: float, top_p: float, max_new_tokens: int) -> str:
        kwargs = {"temperature": temperature, "top_p": top_p}
        try:
            response, _ = self.model.chat(self.tokenizer, prompt, history=[], max_length=max_new_tokens, **kwargs)
        except TypeError:
            response, _ = self.model.chat(self.tokenizer, prompt, history=[], **kwargs)
        return clean_response(response)


class CausalChatGenerator:
    def __init__(
        self,
        model_path: Path,
        *,
        model_type: str,
        torch_dtype: str,
        device_map: str,
        trust_remote_code: bool,
        enable_thinking: bool,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_type = model_type
        self.enable_thinking = enable_thinking
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=trust_remote_code)

        dtype = torch_dtype
        if dtype == "float16":
            dtype = torch.float16
        elif dtype == "bfloat16":
            dtype = torch.bfloat16
        elif dtype == "float32":
            dtype = torch.float32
        elif dtype == "auto":
            dtype = "auto"
        else:
            raise ValueError(f"unsupported torch dtype: {torch_dtype}")

        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _format_chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    def _decode_output(self, output_ids: list[int]) -> str:
        if self.model_type == "psyllm" and 151668 in output_ids:
            # PsyLLM/Qwen3 thinking separator; keep only final response content.
            split_at = len(output_ids) - output_ids[::-1].index(151668)
            output_ids = output_ids[split_at:]
        text = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return clean_response(text)

    def generate(self, prompt: str, *, temperature: float, top_p: float, max_new_tokens: int) -> str:
        import torch

        text = self._format_chat(prompt)
        inputs = self.tokenizer([text], return_tensors="pt")
        try:
            inputs = inputs.to(self.model.device)
        except Exception:
            pass

        do_sample = temperature > 0
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "top_p": top_p,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            generated = self.model.generate(**inputs, **gen_kwargs)
        output_ids = generated[0][len(inputs.input_ids[0]):].tolist()
        return self._decode_output(output_ids)


def make_generator(args):
    model_path = project_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"missing model path: {model_path}")

    if args.model_type == "mechat":
        return MeChatGenerator(model_path, device=args.device, trust_remote_code=args.trust_remote_code)

    if args.model_type in {"psyllm", "soulchat_qwen2"}:
        return CausalChatGenerator(
            model_path,
            model_type=args.model_type,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            enable_thinking=args.enable_thinking,
        )

    raise ValueError(f"unsupported model type: {args.model_type}")


def infer_default_type(model_name: str) -> str:
    name = model_name.lower()
    if "mechat" in name:
        return "mechat"
    if "psyllm" in name:
        return "psyllm"
    if "soulchat" in name:
        return "soulchat_qwen2"
    raise ValueError("could not infer --model-type; pass it explicitly")


def read_done_indices(output_file: Path) -> set[int]:
    done: set[int] = set()
    if not output_file.is_file():
        return done
    with output_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                done.add(int(obj.get("index", line_no)))
            except Exception:
                done.add(line_no)
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-name", required=True, help="Name used for output folder, e.g. MeChat.")
    ap.add_argument("--model-path", default=None, help="Local model directory. Default: models/<model-name>.")
    ap.add_argument("--model-type", choices=["auto", "mechat", "psyllm", "soulchat_qwen2"], default="auto")
    ap.add_argument("--input-file", default="data/instruction/esc/test.json")
    ap.add_argument("--output-dir", default=None, help="Default: data/training/predict/esc/<model-name>.")
    ap.add_argument("--include-factors", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--language", default="English", help="Response language instruction, default: English.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="Build prompts but do not load the model.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-new-tokens", type=int, default=160)
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
    output_dir = project_path(args.output_dir) if args.output_dir else ROOT / "data" / "training" / "predict" / "esc" / args.model_name
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
            print(build_generation_prompt(record, model_type=args.model_type, include_factors=args.include_factors, language=args.language)[:3000])
            print("\n--- Stored eval prompt preview ---")
            print(build_eval_prompt(record)[:2000])
        return

    generator = make_generator(args)
    done = read_done_indices(output_file)
    remaining = [(idx, record) for idx, record in selected if idx not in done]
    print(f"Resume state: {len(done & {idx for idx, _ in selected})} completed, {len(remaining)} remaining")

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
        with tqdm(total=len(selected), initial=len(done & {idx for idx, _ in selected}), desc=f"Predict {args.model_name}", unit="sample", dynamic_ncols=True) as pbar:
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
