"""Convert PFD-labeled updated-memory JSON → instruction format.

Merges and supersedes:
  - IJCAI/Update/dataset2inst/upd_ER2inst.py     (had try/except cruft)
  - IJCAI/Update/dataset2inst/upd_newER2inst.py  (cleaner)

Bug fix: the legacy `elif approach == "both emotional and rational" or "" or "None"
or "None.":` was a no-op chain (empty string and bare literals are always
falsy / truthy independently of the comparison). Replaced with explicit
`in (...)` membership.

Branches on the `Approach` field:
  - emotional-focused           → Personal block only (Beliefs/Intentions/...).
  - rational-focused            → Factual block only (Facts/Causes/Results).
  - both / None / missing       → Both blocks pasted.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.utils.paths import project_root


_PERSONAL_KEYS = ("beliefs", "intentions", "desires", "emotions")
_FACTUAL_KEYS = ("events", "backgrounds", "predictions")

_PERSONAL_TAGS = (("Beliefs", "beliefs"), ("Intentions", "intentions"),
                  ("Desires", "desires"), ("Emotions", "emotions"))
_FACTUAL_TAGS = (("Facts", "events"), ("Causes", "backgrounds"),
                 ("Results", "predictions"))

_BOTH_VALUES = {"both emotional and rational", "", "None", "None.", "lose get"}


def _tag(label, key, entry):
    vals = entry.get(key, []) or []
    return f"<{label}>{'; '.join(vals) if isinstance(vals, list) else vals}</{label}>"


def _personal_block(entry):
    return "".join(_tag(lbl, key, entry) for lbl, key in _PERSONAL_TAGS)


def _factual_block(entry):
    return "".join(_tag(lbl, key, entry) for lbl, key in _FACTUAL_TAGS)


def _approach_suffix(approach, explanation):
    return (
        f"<Approach>{approach}</Approach>"
        f"<Explanation>{explanation}</Explanation>"
    )


def extract_inst_entries(data):
    out = []
    for entry in data:
        approach = entry.get("Approach", "")
        explanation = entry.get("Explanation", "")

        if approach == "emotional-focused":
            body = _personal_block(entry)
        elif approach == "rational-focused":
            body = _factual_block(entry)
        elif approach in _BOTH_VALUES:
            body = _factual_block(entry) + _personal_block(entry)
        else:
            # Unknown label — fall back to both-block but record explicitly.
            body = _factual_block(entry) + _personal_block(entry)

        instruction = body + _approach_suffix(approach, explanation)

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
    ap.add_argument("--full-out", default=None,
                    help="Optional: write the full (un-split) instruction list here.")
    ap.add_argument("--test-ratio", type=float, default=0.1)
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

    targets = [(args.train_out, train), (args.test_out, test)]
    if args.full_out:
        targets.append((args.full_out, inst))
    for p, payload in targets:
        out_p = Path(p) if Path(p).is_absolute() else root / p
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=4)
    print(f"Train: {len(train)}  Test: {len(test)}")


if __name__ == "__main__":
    main()
