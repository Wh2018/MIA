"""Word-cloud visualization for each EToM factor.

Replaces IJCAI/Statistics/statistics_wordcloud.py. Pass --font to point at
your TTF (Times by default).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import wordcloud

from src.stats._common import STOP_WORDS_EN, collect_factors
from src.utils.paths import project_root


def _plot(wc, fname: Path):
    plt.figure()
    plt.axis("off")
    plt.gcf().set_size_inches(10, 10)
    plt.imshow(wc, interpolation="bilinear")
    plt.savefig(fname.with_suffix(".png"), dpi=700, format="png")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="7-factor JSON path.")
    ap.add_argument("--output-dir", default="data/stats/wordclouds")
    ap.add_argument("--font", default=None,
                    help="Path to a TTF font (default: matplotlib's DejaVu).")
    args = ap.parse_args()

    root = project_root()
    in_p = Path(args.input)
    if not in_p.is_absolute():
        in_p = root / in_p
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = collect_factors(in_p)
    wc_kwargs = dict(width=1000, height=700, background_color="white",
                     max_words=100, stopwords=set(STOP_WORDS_EN))
    if args.font:
        wc_kwargs["font_path"] = args.font

    for factor, text in buckets.items():
        if not text.strip():
            continue
        wc = wordcloud.WordCloud(**wc_kwargs).generate(text)
        _plot(wc, out_dir / f"all_{factor.lower()}s")
    print(f"Wrote word clouds to {out_dir}.")


if __name__ == "__main__":
    main()
