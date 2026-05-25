"""Top-20 word frequency bar charts per EToM factor.

Replaces IJCAI/Statistics/statistics_wordfreq.py.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from src.stats._common import STOP_WORDS_EN, collect_factors
from src.utils.paths import project_root


def _plot(words, file_path: Path):
    counts = Counter(words)
    for w in STOP_WORDS_EN:
        counts.pop(w, None)
    if not counts:
        return
    top = counts.most_common(20)
    words_, freqs = zip(*top)
    plt.figure(figsize=(10, 6))
    plt.bar(words_, freqs, color="skyblue")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    plt.title("Top 20 Words Frequency (Excluding Stop Words)")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="data/stats/wordfreq")
    args = ap.parse_args()

    root = project_root()
    in_p = Path(args.input)
    if not in_p.is_absolute():
        in_p = root / in_p
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for factor, text in collect_factors(in_p).items():
        _plot(text.split(), out_dir / f"{factor}.png")
    print(f"Wrote frequency plots to {out_dir}.")


if __name__ == "__main__":
    main()
