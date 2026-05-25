"""Legacy 3-factor pure-factual raw-output parser.

Preserved from IJCAI/Update/dataset2inst/gen2merge.py. Parses Qwen*.txt or
Qwen*.json (Chinese-mode) files containing Observable Events / Past
Experiences / Potential Behaviors blocks and emits a single JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from src.utils.paths import project_root


def process_esc_file(file_path: str, file_name: str):
    m = re.search(r"Qwen(\d+)", file_name)
    file_number = int(m.group(1)) if m else 0
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    dialogues = re.findall(r"\{(.*?)\}", content, re.DOTALL)
    out = []
    for i, d in enumerate(dialogues):
        seeker = re.search(r"seeker:\s*\"(.*?)\"", d)
        supporter = re.search(r"supporter:\s*\"(.*?)\"", d)
        obs = re.search(r"Observable Events:\s*(.*?)(?:\n|$)", d)
        past = re.search(r"Past Experiences:\s*(.*?)(?:\n|$)", d)
        beh = re.search(r"Potential Behaviors:\s*(.*?)(?:\n|$)", d)
        out.append({
            "Count": f"{file_number},{i}",
            "utterance": {
                "seeker": seeker.group(1) if seeker else "",
                "supporter": supporter.group(1) if supporter else "",
            },
            "Observable Events": obs.group(1).strip() if obs else "",
            "Past Experiences": past.group(1).strip() if past else "",
            "Potential Behaviors": beh.group(1).strip() if beh else "",
        })
    return file_number, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-file", required=True)
    args = ap.parse_args()

    root = project_root()
    in_dir = Path(args.input_dir)
    if not in_dir.is_absolute():
        in_dir = root / in_dir
    out_p = Path(args.output_file)
    if not out_p.is_absolute():
        out_p = root / out_p
    out_p.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for fname in os.listdir(in_dir):
        if fname.startswith("Qwen") and fname.endswith(".txt"):
            num, data = process_esc_file(str(in_dir / fname), fname)
            rows.append((num, fname, data))
    rows.sort(key=lambda x: x[0])
    output = {fname: data for _, fname, data in rows}
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    print(f"Wrote {out_p}.")


if __name__ == "__main__":
    main()
