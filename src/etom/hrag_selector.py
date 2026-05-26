"""Hierarchical Retrieval-Augmented exemplar Selector for EToM generation.


Pipeline:
  1. Load an exemplar bank (JSON list, each entry = one labeled dialogue turn
     with the 7 EToM factors).
  2. Embed every exemplar's `seeker + supporter` utterance with SBERT once and
     cache to disk.
  3. At query time, embed the current (seeker, supporter) pair and return the
     top-k exemplars by cosine similarity.

The bank can be hierarchically organised (e.g. one per dataset) by simply
passing different files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.llm_client import load_config
from src.utils.paths import project_root


@dataclass
class Exemplar:
    seeker: str
    supporter: str
    Belief: str
    Intention: str
    Desire: str
    Emotion: str
    Fact: str
    Cause: str
    Result: str

    @classmethod
    def from_dict(cls, d: dict) -> "Exemplar":
        return cls(
            seeker=d.get("seeker", ""),
            supporter=d.get("supporter", ""),
            Belief=d.get("Belief", "None."),
            Intention=d.get("Intention", "None."),
            Desire=d.get("Desire", "None."),
            Emotion=d.get("Emotion", "None."),
            Fact=d.get("Fact", "None."),
            Cause=d.get("Cause", "None."),
            Result=d.get("Result", "None."),
        )

    def join_text(self) -> str:
        return f"seeker: {self.seeker} supporter: {self.supporter}"

    def render_block(self) -> str:
        return (
            "{\n"
            f"    Seeker: \"{self.seeker}\"\n"
            f"    Supporter: \"{self.supporter}\"\n\n"
            f"    Belief: {self.Belief}\n"
            f"    Intention: {self.Intention}\n"
            f"    Desire: {self.Desire}\n"
            f"    Emotion: {self.Emotion}\n"
            f"    Fact: {self.Fact}\n"
            f"    Cause: {self.Cause}\n"
            f"    Result: {self.Result}\n"
            "}"
        )


class HRAGSelector:
    def __init__(
        self,
        bank_path: str | os.PathLike,
        *,
        sbert_path: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str | os.PathLike] = None,
    ):
        with open(bank_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.exemplars: List[Exemplar] = [Exemplar.from_dict(d) for d in raw]
        self.model = None
        self._query_prompt_name = None

        if not self.exemplars:
            self.embeddings = np.empty((0, 0), dtype=np.float32)
            return

        cfg = load_config()
        self.device = device or cfg.get("sbert_device", "cuda:0")
        sbert_path = sbert_path or cfg.get("sbert_model_path", "models/sbert-base-chinese-nli")
        if not sbert_path:
            raise ValueError(
                "sbert_model_path is empty. Fill config/llm_api.yaml or provide "
                "an empty exemplar bank to run without retrieval examples."
            )
        if os.path.isabs(sbert_path):
            sbert_full = sbert_path
        else:
            local_candidate = project_root() / sbert_path
            sbert_full = str(local_candidate) if local_candidate.exists() else sbert_path

        self._model_name = sbert_full
        self.model = SentenceTransformer(sbert_full, device=self.device)
        self._query_prompt_name = cfg.get("embedding_query_prompt_name") or None

        cache_dir = Path(cache_dir) if cache_dir else project_root() / "data" / "etom" / "_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        bank_name = Path(bank_path).stem
        self.cache_file = cache_dir / f"{bank_name}.embeddings.npy"
        self.meta_file = cache_dir / f"{bank_name}.embeddings.meta.json"

        self.embeddings = self._load_or_rebuild_cache()

    def _expected_meta(self, dim: int) -> dict:
        return {
            "model": self._model_name,
            "n": len(self.exemplars),
            "dim": int(dim),
        }

    def _load_or_rebuild_cache(self) -> np.ndarray:
        if self.cache_file.exists() and self.meta_file.exists():
            try:
                cached_meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
                cached = np.load(self.cache_file)
                if (cached_meta.get("model") == self._model_name
                        and cached_meta.get("n") == len(self.exemplars)
                        and cached.shape[0] == len(self.exemplars)
                        and cached.shape[1] == cached_meta.get("dim")):
                    return cached
            except Exception:
                pass  # fall through to rebuild
        emb = self._encode_bank()
        np.save(self.cache_file, emb)
        self.meta_file.write_text(
            json.dumps(self._expected_meta(emb.shape[1]), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return emb

    def _encode_bank(self) -> np.ndarray:
        texts = [e.join_text() for e in self.exemplars]
        with torch.no_grad():
            emb = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return emb

    def _encode_query(self, text: str) -> np.ndarray:
        kwargs = dict(
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if self._query_prompt_name:
            kwargs["prompt_name"] = self._query_prompt_name
        with torch.no_grad():
            return self.model.encode([text], **kwargs)[0]

    def select(self, seeker: str, supporter: str, k: int = 3) -> List[Exemplar]:
        if not self.exemplars:
            return []
        query = f"seeker: {seeker} supporter: {supporter}"
        q_emb = self._encode_query(query)
        sims = self.embeddings @ q_emb
        top_idx = np.argsort(-sims)[: min(k, len(self.exemplars))]
        return [self.exemplars[i] for i in top_idx]

    def render_prefix(self, seeker: str, supporter: str, k: int = 3) -> str:
        picks = self.select(seeker, supporter, k=k)
        return "\n".join(p.render_block() for p in picks)
