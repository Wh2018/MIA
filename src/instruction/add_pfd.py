"""Merge PFD (Approach / Explanation) labels into updated-memory JSON.

he updated-memory JSON is a
flat list keyed by item_ids[0] and turn_ids[0]; the PFD JSON is keyed by
`Qwen{N}.txt`. We look up each memory entry's matching PFD turn and copy
its `Approach` and `Explanation` fields onto the memory entry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.paths import project_root


def merge(memory_data, pfd_data):
    result = []
    for obj in memory_data:
        item_id = obj["item_ids"][0]
        turn_id = obj["turn_ids"][0]
        key = f"Qwen{item_id}.txt"
        if key in pfd_data and len(pfd_data[key]) > turn_id:
            count = pfd_data[key][turn_id]
            obj["Approach"] = count.get("Approach", "")
            obj["Explanation"] = count.get("Explanation", "")
        result.append(obj)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True,
                    help="Updated-memory JSON (list of per-turn states).")
    ap.add_argument("--pfd", required=True,
                    help="PFD-annotated JSON keyed by Qwen{N}.txt.")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    root = project_root()

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    with _resolve(args.memory).open("r", encoding="utf-8") as f:
        memory_data = json.load(f)
    with _resolve(args.pfd).open("r", encoding="utf-8") as f:
        pfd_data = json.load(f)

    merged = merge(memory_data, pfd_data)
    out_p = _resolve(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with out_p.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)
    print(f"Wrote {out_p} ({len(merged)} entries).")


if __name__ == "__main__":
    main()
