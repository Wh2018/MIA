"""Write a LLaMA-Factory `dataset_info.json` for the MIA instruction sets.

Run from the project root (or with --root /path/to/code_new). It only
references files that already exist on disk, so partial pipelines don't
break LLaMA-Factory's dataset loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


_COLUMNS = {"prompt": "instruction", "query": "input", "response": "output"}

_CANDIDATES = [
    ("ESC_train",  "esc/train.json"),
    ("ESC_test",   "esc/test.json"),
    ("CPsy_train", "cpsy/train.json"),
    ("CPsy_test",  "cpsy/test.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="code_new project root.")
    ap.add_argument("--dataset-dir", default="data/instruction",
                    help="Where dataset_info.json should live (rel. to --root).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    dataset_dir = (root / args.dataset_dir).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    info = {}
    for name, rel in _CANDIDATES:
        if (dataset_dir / rel).exists():
            info[name] = {
                "file_name": rel,
                "formatting": "alpaca",
                "columns": _COLUMNS,
            }
        else:
            print(f"[skip] {name}: {dataset_dir / rel} not found yet.")

    out = dataset_dir / "dataset_info.json"
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out} with {len(info)} datasets: {list(info)}")


if __name__ == "__main__":
    main()
