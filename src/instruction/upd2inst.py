"""Convert updated-memory JSON → LLaMA-Factory instruction format (no PFD).

Replaces IJCAI/Update/dataset2inst/upd2inst.py. Baseline that always pastes
all 7 factors into the prompt — used for the "no-PFD" ablation.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.utils.paths import project_root


def _join(xs):
    return "; ".join(xs) if isinstance(xs, list) else (xs or "")


def extract_inst_entries(data):
    out = []
    for entry in data:
        instruction = (
            f"<Facts>{_join(entry.get('events', []))}</Facts>"
            f"<Causes>{_join(entry.get('backgrounds', []))}</Causes>"
            f"<Results>{_join(entry.get('predictions', []))}</Results>"
            f"<Beliefs>{_join(entry.get('beliefs', []))}</Beliefs>"
            f"<Intentions>{_join(entry.get('intentions', []))}</Intentions>"
            f"<Desires>{_join(entry.get('desires', []))}</Desires>"
            f"<Emotions>{_join(entry.get('emotions', []))}</Emotions>"
        )
        utt = entry.get("utterances") or []
        rsp = entry.get("responses") or []
        out.append({
            "instruction": instruction,
            "input": utt[0] if utt else "",
            "output": rsp[0] if rsp else "",
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--test-out", required=True)
    ap.add_argument("--test-ratio", type=float, default=0.1,
                    help="Fraction of items reserved for the test set (default 9:1).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = project_root()
    in_p = Path(args.input)
    if not in_p.is_absolute():
        in_p = root / in_p

    with in_p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(args.seed)
    test_size = max(1, int(len(data) * args.test_ratio))
    test_idx = set(random.sample(range(len(data)), test_size))

    inst = extract_inst_entries(data)
    train = [inst[i] for i in range(len(inst)) if i not in test_idx]
    test = [inst[i] for i in test_idx]

    for p, payload in [(args.train_out, train), (args.test_out, test)]:
        out_p = Path(p) if Path(p).is_absolute() else root / p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
    print(f"Train: {len(train)}  Test: {len(test)}")


if __name__ == "__main__":
    main()
