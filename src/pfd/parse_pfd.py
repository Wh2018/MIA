"""Merge PFD (Approach / Explanation) outputs back into 7-factor JSON.

Two modes:
  --style esc   : Each Qwen{N}.txt holds full re-emitted dialogue blocks with
                  Belief/.../Result + Approach + Explanation per turn.
                  Re-parses everything into JSON, like ER_pure_mind_2_gen_ESC.

  --style cpsy  : Each Qwen{N}.txt holds only "[ Approach: ... Explanation: ... ]"
                  blocks. The dialogue + 7 factors come from a base JSON; we
                  merge the bracketed blocks into that base in order.

Output is a single JSON keyed by file name.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from src.etom.parse_seven import (
    _FACTOR_KEYS, _flush, normalize_punctuation, clean_text,
)
from src.utils.paths import project_root


# ---- ESC style ------------------------------------------------------------

def _parse_esc_file(file_path: str, file_name: str):
    m = re.search(r"Qwen(\d+)", file_name)
    file_number = int(m.group(1)) if m else 0

    with open(file_path, "r", encoding="utf-8") as f:
        content = normalize_punctuation(f.read())

    processed: List[dict] = []
    current_dialogue: dict = {}
    factors = {k: "" for k in _FACTOR_KEYS}
    approach = ""
    explanation = ""

    def _flush_one():
        processed.append({
            "Count": f"{file_number},{len(processed)}",
            "utterance": {
                "seeker": clean_text(current_dialogue.get("seeker", "")),
                "supporter": clean_text(current_dialogue.get("supporter", "")),
            },
            **{k: factors.get(k, "") for k in _FACTOR_KEYS},
            "Approach": approach,
            "Explanation": explanation,
        })

    for line in content.splitlines():
        line = line.strip()
        if line.startswith("seeker"):
            if "seeker" in current_dialogue:
                _flush_one()
                factors = {k: "" for k in _FACTOR_KEYS}
                approach = ""
                explanation = ""
            current_dialogue = {"seeker": line.split("seeker:", 1)[1].strip()}
        elif line.startswith("supporter:"):
            current_dialogue["supporter"] = line.split("supporter:", 1)[1].strip()
        elif line.startswith("Approach"):
            approach = line.split("Approach:", 1)[1].strip()
        elif line.startswith("Explanation"):
            explanation = line.split("Explanation:", 1)[1].strip()
        else:
            for k in _FACTOR_KEYS:
                if line.startswith(k):
                    factors[k] = line.split(f"{k}:", 1)[1].strip()
                    break

    if "seeker" in current_dialogue:
        _flush_one()
    return file_number, processed


# ---- CPsy style -----------------------------------------------------------

def _extract_approach_explanation(lines):
    """Extract a list of [Approach, Explanation] blocks (CPsy output style)."""
    out = []
    approach, explanation = "None", "None"
    for raw in lines:
        line = raw.strip()
        if line.startswith("Approach:"):
            approach = line.replace("Approach:", "", 1).strip()
        elif line.startswith("Explanation:"):
            explanation = line.replace("Explanation:", "", 1).strip()
        elif line == "]":
            out.append({"Approach": approach, "Explanation": explanation})
            approach, explanation = "None", "None"
    if approach != "None" or explanation != "None":
        out.append({"Approach": approach, "Explanation": explanation})
    return out


def _merge_cpsy(base_json: Dict[str, list], txt_dir: Path) -> Dict[str, list]:
    for fname, turns in base_json.items():
        txt_path = txt_dir / fname
        if not txt_path.exists():
            continue
        with txt_path.open("r", encoding="utf-8") as f:
            blocks = _extract_approach_explanation(f.readlines())
        for i, turn in enumerate(turns):
            if i < len(blocks):
                turn["Approach"] = blocks[i]["Approach"]
                turn["Explanation"] = blocks[i]["Explanation"]
            else:
                turn["Approach"] = "None"
                turn["Explanation"] = "None"
    return base_json


# ---- CLI ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse PFD output txt files into JSON.")
    ap.add_argument("--style", choices=["esc", "cpsy"], required=True)
    ap.add_argument("--input-dir", required=True,
                    help="Directory of Qwen*.txt PFD outputs.")
    ap.add_argument("--output-file", required=True,
                    help="Output JSON path.")
    ap.add_argument("--base-json", default=None,
                    help="[cpsy only] 7-factor JSON to merge Approach/Explanation into.")
    args = ap.parse_args()

    root = project_root()
    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    output_file = Path(args.output_file)
    if not output_file.is_absolute():
        output_file = root / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.style == "esc":
        merged = {}
        rows = []
        for fname in os.listdir(input_dir):
            if fname.startswith("Qwen") and fname.endswith(".txt"):
                num, data = _parse_esc_file(str(input_dir / fname), fname)
                rows.append((num, fname, data))
        rows.sort(key=lambda x: x[0])
        merged = {fname: data for _, fname, data in rows}
    else:
        if not args.base_json:
            ap.error("--base-json is required when --style=cpsy")
        base_path = Path(args.base_json)
        if not base_path.is_absolute():
            base_path = root / base_path
        with base_path.open("r", encoding="utf-8") as f:
            base = json.load(f)
        merged = _merge_cpsy(base, input_dir)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)
    print(f"Wrote {output_file} ({len(merged)} files).")


if __name__ == "__main__":
    main()
