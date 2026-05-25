"""SBERT-based cosine similarity (Memory module helper).

Model path and device are read from config/llm_api.yaml (keys:
sbert_model_path, sbert_device). The model is loaded once and reused.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances

from src.llm_client import load_config
from src.utils.paths import project_root


logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


@lru_cache(maxsize=1)
def _get_model():
    cfg = load_config()
    rel = cfg.get("sbert_model_path", "models/sbert-base-chinese-nli")
    device = cfg.get("sbert_device", "cuda:0")
    path = rel if os.path.isabs(rel) else str(project_root() / rel)
    return SentenceTransformer(path, device=device)


def get_similarity(sentence1: str, sentence2: str):
    model = _get_model()
    emb = model.encode([sentence1, sentence2])
    return 1 - paired_cosine_distances([emb[0]], [emb[1]])
