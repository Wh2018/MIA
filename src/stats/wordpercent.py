"""Per-factor average words/sentence and sentences/turn.

Replaces IJCAI/Statistics/statistics_wordpersent.py. The original code had
hard-coded counts; this version recomputes them from a 7-factor JSON so the
chart stays in sync with the data.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import project_root


_FACTOR_KEYS = ("Belief", "Intention", "Desire", "Emotion",
                "Cause", "Fact", "Result")


def _count(text: str):
    """Return (#words, #sentences) for one factor entry."""
    sentences = [s for s in re.split(r"[.;。；]", text) if s.strip()]
    words = sum(len(s.strip().split()) for s in sentences)
    return words, len(sentences)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="7-factor JSON.")
    ap.add_argument("--output-dir", default="data/stats/wordpercent")
    args = ap.parse_args()

    root = project_root()
    in_p = Path(args.input)
    if not in_p.is_absolute():
        in_p = root / in_p
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with in_p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    totals = {k: [0, 0] for k in _FACTOR_KEYS}
    total_turns = 0
    for _, dialogues in data.items():
        for d in dialogues:
            total_turns += 1
            for k in _FACTOR_KEYS:
                w, s = _count(d.get(k, ""))
                totals[k][0] += w
                totals[k][1] += s

    words = np.array([totals[k][0] for k in _FACTOR_KEYS])
    sents = np.array([totals[k][1] for k in _FACTOR_KEYS])
    wps = np.divide(words, sents, out=np.zeros_like(words, dtype=float), where=sents > 0)
    spt = sents / max(total_turns, 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    axs[0].bar(_FACTOR_KEYS, wps, color="skyblue")
    axs[0].set_title("Words per Sentence")
    axs[0].tick_params(axis="x", rotation=45)
    axs[1].bar(_FACTOR_KEYS, spt, color="lightcoral")
    axs[1].set_title("Sentences per Turn")
    axs[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(out_dir / "words_per_sentence.png")
    fig.savefig(out_dir / "sentences_per_turn.png")
    plt.close(fig)
    print(f"Total turns: {total_turns}")
    for k in _FACTOR_KEYS:
        print(f"  {k:10s}  words={totals[k][0]:6d}  sentences={totals[k][1]:6d}")


if __name__ == "__main__":
    main()
