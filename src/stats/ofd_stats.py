"""Cross-reference ORM "outdated" log against the 7-factor gen file.

Replaces IJCAI/Statistics/statistics_OFD.py. For every memory entry that the
ORM judged outdated, find the turn in the gen file where that text first
appeared, and tally the gap turn_id_ofd − turn_id_gen.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.utils.paths import project_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ofd", required=True,
                    help="ORM outdated-log JSON written by run_update.py.")
    ap.add_argument("--gen", required=True, help="7-factor JSON (parse_seven output).")
    args = ap.parse_args()

    root = project_root()

    def _resolve(p):
        p = Path(p)
        return p if p.is_absolute() else root / p

    with _resolve(args.ofd).open("r", encoding="utf-8") as f:
        ofd_data = json.load(f)
    with _resolve(args.gen).open("r", encoding="utf-8") as f:
        gen_data = json.load(f)

    diff_count = Counter()
    for ofd in ofd_data:
        item_id = ofd.get("item_id")
        marked = ofd.get("marked_memory")
        turn_ofd = ofd.get("turn_id")
        if marked is None or item_id is None:
            continue
        found = False
        for _, entries in gen_data.items():
            for entry in entries:
                count = entry.get("Count")
                if not count:
                    continue
                item_gen, turn_gen = (int(x) for x in count.split(","))
                if item_gen != item_id:
                    continue
                for v in entry.values():
                    if isinstance(v, str) and marked in v:
                        diff_count[turn_ofd - turn_gen] += 1
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            print(f"Warning: item_id {item_id} not found in gen file.")

    for diff, count in sorted(diff_count.items()):
        print(f"diff={diff:+d}  count={count}")


if __name__ == "__main__":
    main()
