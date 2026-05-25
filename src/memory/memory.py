"""Mental_Memory — incremental memory updater for the MIA framework.

Implements the four update operators from the paper:
    - append       (Eq.5)  on FT_ATTR + MIND_ATTR
    - overwrite    (Eq.6)  on OBJ_ATTR + EMO_ATTR
    - similarity   (Eq.7)  SBERT threshold 0.8 — used for FT_ATTR
    - ORM          (Eq.8)  SBERT threshold 0.3 + LLM judgment — selective
                          deletion across FT_ATTR + MIND_ATTR

ORM is **enabled by default** in `whole_updater()`. The previous codebase had
it commented out — see README "Paper vs. code reconciliation" for why this
matters.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import List, Optional

from src.memory.outdated import discriminate_outdated
from src.memory.similarity import get_similarity
from src.memory.state import Mental_State
from src.memory.str_format import str_delete_format


ALL_ATTR = ["utterance", "response", "item_id", "turn_id",
            "event", "belief", "intention", "desire", "emotion",
            "background", "prediction"]
FT_ATTR = ["event", "background", "prediction"]
EMO_ATTR = ["emotion"]
MIND_ATTR = ["belief", "intention", "desire"]
OBJ_ATTR = ["utterance", "response", "item_id", "turn_id"]


def _append_jsonlog(path: Optional[Path], obj):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        existing = []
    existing.append(obj)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


class Mental_Memory:
    """Memory bank with per-attribute update operators.

    Optional log paths:
      similar_log_path  — high-similarity events recorded by the SBERT op (Eq.7).
      outdated_log_path — ORM judgments (Eq.8): kept/removed entries.
      sim_score_path    — flat list of all similarity scores from ORM scans.
    """

    def __init__(
        self,
        *,
        similar_log_path: Optional[Path] = None,
        outdated_log_path: Optional[Path] = None,
        sim_score_path: Optional[Path] = None,
    ):
        for attr in ALL_ATTR:
            setattr(self, attr + "s", [])
        self.history: List[dict] = []
        self.similar_log_path = Path(similar_log_path) if similar_log_path else None
        self.outdated_log_path = Path(outdated_log_path) if outdated_log_path else None
        self.sim_score_path = Path(sim_score_path) if sim_score_path else None

    # --- I/O ---------------------------------------------------------------

    def add_memory(self, ms: Mental_State):
        """Replace memory with a single Mental_State (used at item boundaries)."""
        self.utterances = ms.utterance or []
        self.responses = ms.response or []
        self.item_ids = ms.item_id or []
        self.turn_ids = ms.turn_id or []
        self.events = ms.event or []
        self.beliefs = ms.belief or []
        self.intentions = ms.intention or []
        self.desires = ms.desire or []
        self.emotions = ms.emotion or []
        self.backgrounds = ms.background or []
        self.predictions = ms.prediction or []

    def __str__(self) -> str:
        out = ["---------Memory SAT---------"]
        for attr in self.__dict__:
            values = getattr(self, attr)
            if isinstance(values, list) and values:
                for i, v in enumerate(values, 1):
                    out.append(f"  {attr[:-1]} {i}: {v}")
        out.append("---------Memory END---------")
        return "\n".join(out)

    def clear_memory(self):
        for attr in ALL_ATTR:
            setattr(self, attr + "s", [])
        self.history = []

    # --- low-level helpers -------------------------------------------------

    def _update_memory(self, index, **kwargs):
        if index == -1:
            for key, value in kwargs.items():
                getattr(self, key + "s").append(value)
        elif 0 <= index < len(self.utterances):
            for key, value in kwargs.items():
                getattr(self, key + "s")[index] = value
        else:
            raise IndexError("Memory index out of range")

    # --- four operators ----------------------------------------------------

    def _updater_similarity(self, attrs, new_state, SIM_SCORE=0.8):
        """Eq.(7): high-similarity gated overwrite, used on factual attributes."""
        for key in attrs:
            state_value = getattr(new_state, key, None)
            if not state_value:
                continue
            list_attrs = getattr(self, key + "s")
            ini_list_attrs = deepcopy(list_attrs)

            for single in state_value:
                update_flag = True
                high_score = SIM_SCORE
                high_content = None
                for mem in ini_list_attrs:
                    if not isinstance(mem, str) or not isinstance(single, str):
                        raise TypeError("similarity inputs must be strings")
                    sim = get_similarity(mem, single)
                    if sim >= high_score:
                        update_flag = False
                        high_score = sim
                        high_content = mem
                        _append_jsonlog(self.similar_log_path, {
                            "similar_score": float(sim[0]) if hasattr(sim, "__len__") else float(sim),
                            "mem": mem, "incoming": single,
                        })
                if update_flag:
                    self._update_memory(-1, **{key: single})
                else:
                    try:
                        list_attrs.remove(high_content)
                        ini_list_attrs.remove(high_content)
                    except ValueError:
                        self._update_memory(-1, **{key: single})

    def _updater_overwrite(self, attrs, new_state):
        """Eq.(6): replace the entire attribute list."""
        for key in attrs:
            value = getattr(new_state, key, None)
            if value is not None:
                setattr(self, key + "s", value)

    def _updater_append(self, attrs, new_state):
        """Eq.(5): append every incoming item."""
        for key in attrs:
            value = getattr(new_state, key, None)
            if value is None:
                continue
            if not isinstance(value, list):
                raise TypeError(f"{key!r} must be a list, got {type(value)}")
            for single in value:
                if not isinstance(single, str):
                    raise TypeError(f"{key!r} items must be strings")
                self._update_memory(-1, **{key: single})

    def _add_history(self, new_state):
        if new_state.turn_id and new_state.turn_id[0] == 1:
            self.history = []
        self.history.append({
            "seeker": new_state.utterance,
            "supporter": new_state.response,
        })

    def _updater_ObsolescenceRemoval(self, attrs, new_state, SIM_SCORE=0.3):
        """Eq.(8): for each (memory, new) pair whose similarity is below the
        threshold, ask the judge LLM whether the memory is outdated, and drop
        it from the attribute list if so."""
        current_mind_state = {
            "Belief": new_state.belief,
            "Intention": new_state.intention,
            "Desire": new_state.desire,
            "Emotion": new_state.emotion,
            "Fact": new_state.event,
            "Cause": new_state.background,
            "Result": new_state.prediction,
        }

        for key in attrs:
            state_value = getattr(new_state, key, None)
            if not state_value:
                continue
            ini_list_attrs = deepcopy(getattr(self, key + "s", []))
            for single in state_value:
                for mem in list(ini_list_attrs):
                    if not isinstance(mem, str) or not isinstance(single, str):
                        raise TypeError("ORM inputs must be strings")
                    sim = float(get_similarity(
                        str_delete_format(mem), str_delete_format(single)
                    )[0])

                    _append_jsonlog(self.sim_score_path, sim)

                    if sim < SIM_SCORE:
                        prev_hist = self.history[:-1] if len(self.history) >= 2 else []
                        cur_utt = getattr(new_state, "utterance", "")
                        verdict = discriminate_outdated(
                            mem, str(prev_hist), str(cur_utt), str(current_mind_state)
                        )
                        outdated = ("Outdated" in verdict) or ("outdated" in verdict)
                        log_entry = {
                            "turn_id": self.turn_ids[0] if self.turn_ids else None,
                            "item_id": self.item_ids[0] if self.item_ids else None,
                            "marked_memory": mem,
                            "similar_score": sim,
                            "outdate": outdated,
                        }
                        _append_jsonlog(self.outdated_log_path, log_entry)
                        if outdated:
                            try:
                                ini_list_attrs.remove(mem)
                            except ValueError:
                                pass
            setattr(self, key + "s", ini_list_attrs)

    # --- public driver -----------------------------------------------------

    def whole_updater(
        self,
        new_state,
        *,
        sim_score: float = 0.8,
        orm_threshold: float = 0.3,
        use_similarity: bool = True,
        use_orm: bool = True,
    ):
        """Run the full update pipeline for one new Mental_State.

        ORM is ON by default (it was OFF in the legacy code). Disable via
        `use_orm=False` if you want a strict-append baseline.
        """
        self._updater_overwrite(attrs=OBJ_ATTR, new_state=new_state)
        self._add_history(new_state=new_state)

        if use_similarity:
            self._updater_similarity(attrs=FT_ATTR, new_state=new_state, SIM_SCORE=sim_score)
        if use_orm:
            self._updater_ObsolescenceRemoval(
                attrs=FT_ATTR + MIND_ATTR, new_state=new_state, SIM_SCORE=orm_threshold,
            )
        self._updater_append(attrs=FT_ATTR + MIND_ATTR, new_state=new_state)
        self._updater_overwrite(attrs=EMO_ATTR, new_state=new_state)
