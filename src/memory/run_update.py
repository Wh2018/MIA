"""Drive Mental_Memory across an entire dataset.

Replaces IJCAI/Update/updater.py. Input is a 7-factor JSON keyed by file name
(produced by src/etom/parse_seven.py). For every turn we instantiate a
Mental_State and call `memory.whole_updater()`; at item boundaries we flush
the memory contents to an output JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from src.memory.memory import ALL_ATTR, Mental_Memory
from src.memory.state import Mental_State
from src.utils.paths import project_root


def _load_states(input_file: Path):
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    states = []
    for _, dialogue in data.items():
        for turn in dialogue:
            states.append(Mental_State.from_turn(turn))
    return states


def run(
    input_file: str,
    output_file: str,
    *,
    similar_log: str | None = None,
    outdated_log: str | None = None,
    sim_score_log: str | None = None,
    start_id: int = 0,
    sim_score: float = 0.8,
    orm_threshold: float = 0.3,
    use_similarity: bool = True,
    use_orm: bool = True,
):
    root = project_root()

    def _resolve(p):
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else root / p

    in_p = _resolve(input_file)
    out_p = _resolve(output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    states = _load_states(in_p)

    memory = Mental_Memory(
        similar_log_path=_resolve(similar_log),
        outdated_log_path=_resolve(outdated_log),
        sim_score_path=_resolve(sim_score_log),
    )

    if start_id == 0:
        out_p.write_text("[", encoding="utf-8")
        now_item = 0
    else:
        now_item = start_id - 1

    end_n = len(states)
    result_entry: dict = {}

    with out_p.open("a", encoding="utf-8") as fout:
        for turn_number, ms in tqdm(enumerate(states), total=end_n):
            if not ms.item_id or ms.item_id[0] < start_id:
                continue

            if ms.item_id[0] != now_item:
                if now_item != 0 and result_entry:
                    json.dump(result_entry, fout, ensure_ascii=False)
                    if turn_number != end_n - 1:
                        fout.write(",\n")
                now_item += 1
                memory.clear_memory()
                memory.add_memory(ms)
            else:
                memory.whole_updater(
                    ms,
                    sim_score=sim_score,
                    orm_threshold=orm_threshold,
                    use_similarity=use_similarity,
                    use_orm=use_orm,
                )

            result_entry = {
                attr + "s": getattr(memory, attr + "s", None) for attr in ALL_ATTR
            }
            json.dump(result_entry, fout, ensure_ascii=False)
            if turn_number != end_n - 1:
                fout.write(",\n")
        fout.write("]")


def main():
    ap = argparse.ArgumentParser(description="Run Mental_Memory across a dataset.")
    ap.add_argument("--input", required=True,
                    help="7-factor JSON (output of parse_seven.py or parse_pfd.py).")
    ap.add_argument("--output", required=True, help="Updated-memory JSON output.")
    ap.add_argument("--similar-log", default=None)
    ap.add_argument("--outdated-log", default=None)
    ap.add_argument("--sim-score-log", default=None)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--sim-score", type=float, default=0.8)
    ap.add_argument("--orm-threshold", type=float, default=0.3)
    ap.add_argument("--no-similarity", action="store_true")
    ap.add_argument("--no-orm", action="store_true",
                    help="Disable ORM (Eq.8). On by default — paper claims it.")
    args = ap.parse_args()
    run(
        args.input, args.output,
        similar_log=args.similar_log,
        outdated_log=args.outdated_log,
        sim_score_log=args.sim_score_log,
        start_id=args.start_id,
        sim_score=args.sim_score,
        orm_threshold=args.orm_threshold,
        use_similarity=not args.no_similarity,
        use_orm=not args.no_orm,
    )


if __name__ == "__main__":
    main()
