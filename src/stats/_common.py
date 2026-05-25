"""Shared helpers for the stats scripts (word lists + stopword set)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


STOP_WORDS_EN = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn", "The", "A", "An", "seeker", "supporter", "None",
    "supporter's", "seeker's", "may",
}


_FACTOR_KEYS = ("Belief", "Intention", "Desire", "Emotion",
                "Fact", "Cause", "Result")


def collect_factors(path: Path) -> Dict[str, str]:
    """Read a 7-factor JSON keyed by Qwen{N}.txt and return one big string
    per factor (lower-cased and punctuation-stripped)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    buckets: Dict[str, List[str]] = {k: [] for k in _FACTOR_KEYS}
    for _, dialogues in data.items():
        for d in dialogues:
            for k in _FACTOR_KEYS:
                buckets[k].append(d.get(k, ""))

    def _clean(joined: str) -> str:
        return joined.replace(".", "").replace(",", "").replace(";", "")

    return {k: _clean(" ".join(v)) for k, v in buckets.items()}
