"""Mental_State data container.

The 7 EToM factors per the paper map onto the original attribute names:
    Belief    -> belief
    Intention -> intention
    Desire    -> desire
    Emotion   -> emotion
    Fact      -> event
    Cause     -> background
    Result    -> prediction

(The legacy names — event/background/prediction — are kept so existing JSON
files load unchanged.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


@dataclass
class Mental_State:
    utterance: List[str] = field(default_factory=list)
    response: List[str] = field(default_factory=list)
    turn_id: List[int] = field(default_factory=list)
    item_id: List[int] = field(default_factory=list)
    event: List[str] = field(default_factory=list)        # Fact
    belief: List[str] = field(default_factory=list)
    intention: List[str] = field(default_factory=list)
    desire: List[str] = field(default_factory=list)
    emotion: List[str] = field(default_factory=list)
    background: List[str] = field(default_factory=list)   # Cause
    prediction: List[str] = field(default_factory=list)   # Result

    def __str__(self) -> str:
        return (
            f"Utterance: {self.utterance}\n"
            f"Belief: {self.belief}\n"
            f"Intention: {self.intention}\n"
            f"Desire: {self.desire}\n"
            f"Emotion: {self.emotion}\n"
            f"Fact: {self.event}\n"
            f"Cause: {self.background}\n"
            f"Result: {self.prediction}"
        )

    @classmethod
    def from_turn(cls, turn: dict) -> "Mental_State":
        """Build from a single turn dict produced by parse_seven / parse_pfd."""
        count = turn.get("Count", "0,0")
        item_id, turn_id = (int(x) for x in str(count).split(",")[:2])
        utt = turn.get("utterance", {}).get("seeker", "")
        rsp = turn.get("utterance", {}).get("supporter", "")

        def _clean(values):
            v = _as_list(values)
            return [] if v in ([""], ["None."], ["None"]) else v

        return cls(
            utterance=_as_list(utt),
            response=_as_list(rsp),
            item_id=[item_id],
            turn_id=[turn_id],
            event=_clean(turn.get("Fact", [])),
            background=_clean(turn.get("Cause", [])),
            prediction=_clean(turn.get("Result", [])),
            belief=_clean(turn.get("Belief", [])),
            intention=_clean(turn.get("Intention", [])),
            desire=_clean(turn.get("Desire", [])),
            emotion=_clean(turn.get("Emotion", [])),
        )
