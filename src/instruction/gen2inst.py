"""Legacy 3-factor pure-factual gen → instruction converter.

Preserved from IJCAI/Update/dataset2inst/gen2inst.py for backwards-compat
with the older pipeline that used (Observable Events / Past Experiences /
Potential Behaviors) instead of the full 7-factor EToM block.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from src.utils.paths import project_root


def select_entries(entries, choose):
    if choose == "all":
        return entries
    if choose == "partial":
        n = len(entries)
        idx = sorted({math.floor(n * 0.3), math.floor(n * 0.5), math.floor(n * 0.8)})
        return [entries[i] for i in idx if 0 <= i < n]
    if choose == "random":
        if len(entries) <= 3:
            return entries
        return random.sample(entries, 3)
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--choose", choices=["all", "partial", "random"], default="all")
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = project_root()

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    with _resolve(args.input).open("r", encoding="utf-8") as f:
        data = json.load(f)

    txt_files = list(data.keys())
    random.seed(args.seed)
    test_size = max(1, int(len(txt_files) * args.test_ratio))
    test_files = set(random.sample(txt_files, test_size))

    transformed = []
    for txt_file, entries in data.items():
        for entry in select_entries(entries, args.choose):
            instruction = (
                f"<Observable Events>{entry['Observable Events']}</Observable Events>"
                f"<Past Experiences>{entry['Past Experiences']}</Past Experiences>"
                f"<Potential Behaviors>{entry['Potential Behaviors']}</Potential Behaviors>"
            )
            transformed.append((txt_file, {
                "instruction": instruction,
                "input": entry["utterance"]["seeker"],
                "output": entry["utterance"]["supporter"],
            }))

    train_data = [e for tf, e in transformed if tf not in test_files]
    test_data = [e for tf, e in transformed if tf in test_files]

    for p, payload in [(args.train_out, train_data), (args.test_out, test_data)]:
        out_p = _resolve(p)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
    print(f"Train: {len(train_data)}  Test: {len(test_data)}")


if __name__ == "__main__":
    main()
