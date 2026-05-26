"""Parse per-conversation LLM output files into a single 7-factor JSON.

Input  : directory of `Qwen{idx}.txt` files produced by gen_seven_{esc,cpsy}.
Output : a single JSON keyed by file name, each value a list of turn dicts:
         {Count, utterance: {seeker, supporter}, Belief, Intention, Desire,
          Emotion, Cause, Fact, Result}.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from src.utils.paths import project_root


_PUNCT_MAP = {
    "，": ",", "。": ".", "；": ";", "（": "(", "）": ")",
    "：": ":", "“": '"', "”": '"', "、": ",",
    "Supporter:": "supporter:", "Seeker:": "seeker:",
}


def normalize_punctuation(content: str) -> str:
    for src, dst in _PUNCT_MAP.items():
        content = content.replace(src, dst)
    return content


def clean_text(text: str) -> str:
    return text.strip().strip('"')


_FACTOR_KEYS = ("Belief", "Intention", "Desire", "Emotion", "Cause", "Fact", "Result")


def _flush(processed_data, file_number, current_dialogue, factors):
    processed_data.append({
        "Count": f"{file_number},{len(processed_data)}",
        "utterance": {
            "seeker": clean_text(current_dialogue.get("seeker", "")),
            "supporter": clean_text(current_dialogue.get("supporter", "")),
        },
        **{k: factors.get(k, "") for k in _FACTOR_KEYS},
    })


def process_qwen_file(file_path: str, file_name: str):
    m = re.search(r"Qwen(\d+)", file_name)
    file_number = int(m.group(1)) if m else 0

    with open(file_path, "r", encoding="utf-8") as f:
        content = normalize_punctuation(f.read())

    processed_data = []
    current_dialogue: dict = {}
    factors = {k: "" for k in _FACTOR_KEYS}

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("seeker"):
            if "seeker" in current_dialogue:
                _flush(processed_data, file_number, current_dialogue, factors)
                factors = {k: "" for k in _FACTOR_KEYS}
            current_dialogue = {"seeker": line.split("seeker:", 1)[1].strip()}
        elif line.startswith("supporter:"):
            current_dialogue["supporter"] = line.split("supporter:", 1)[1].strip()
        else:
            for k in _FACTOR_KEYS:
                if line.startswith(k):
                    factors[k] = line.split(f"{k}:", 1)[1].strip()
                    break

    if "seeker" in current_dialogue:
        _flush(processed_data, file_number, current_dialogue, factors)

    return file_number, processed_data


def main():
    ap = argparse.ArgumentParser(description="Parse Qwen*.txt into 7-factor JSON.")
    ap.add_argument("--input-dir", required=True,
                    help="Directory of Qwen*.txt files.")
    ap.add_argument("--output-file", required=True,
                    help="Output JSON path (relative to project root).")
    args = ap.parse_args()

    root = project_root()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    output_file = Path(args.output_file)
    if not output_file.is_absolute():
        output_file = root / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    file_data_list = []
    for fname in os.listdir(input_dir):
        if fname.startswith("Qwen") and fname.endswith(".txt"):
            num, data = process_qwen_file(str(input_dir / fname), fname)
            file_data_list.append((num, fname, data))
    file_data_list.sort(key=lambda x: x[0])

    output_json = {fname: data for _, fname, data in file_data_list}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
    print(f"Wrote {output_file} ({len(output_json)} files).")


if __name__ == "__main__":
    main()
