"""Generate EToM 7-factor annotations for CPsyCounD-style dialogues.


Reuses the same prompt template and HRAG selector as the ESC generator, but
defaults to the Chinese exemplar bank so retrieved few-shot examples match
the Chinese dialogue language of CPsyCounD.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.etom.hrag_selector import HRAGSelector
from src.etom.prompts import BASIC_PROMPT, build_query_block
from src.llm_client import LLMClient
from src.utils.paths import project_root


def _iter_histories(data):
    """Yield (id, [(seeker, supporter), ...]) tuples from a CPsy JSON.

    Supports two shapes:
      - list[ {"history": [["seeker","supporter"], ...]} ]   (ESC-style)
      - dict[str, list[ {"utterance": {"seeker":..., "supporter":...}} ] ]
        (CPsy already-parsed style, useful for re-generating on a subset).
    """
    if isinstance(data, list):
        for i, entry in enumerate(data):
            turns = [(t[0], t[1]) for t in entry["history"]]
            yield i, turns
    elif isinstance(data, dict):
        for i, (key, dialogues) in enumerate(data.items()):
            turns = [
                (d["utterance"]["seeker"], d["utterance"]["supporter"])
                for d in dialogues
            ]
            yield i, turns
    else:
        raise ValueError(f"Unsupported input shape: {type(data)}")


def run(
    input_file: str,
    output_dir: str,
    *,
    exemplar_bank: Optional[str] = None,
    k: int = 3,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
):
    root = project_root()
    output_dir_p = Path(output_dir)
    if not output_dir_p.is_absolute():
        output_dir_p = root / output_dir
    output_dir_p.mkdir(parents=True, exist_ok=True)

    bank = exemplar_bank or str(root / "config" / "etom_exemplars_cpsy.json")
    selector = HRAGSelector(bank)
    client = LLMClient()

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = list(_iter_histories(data))
    end_idx = end_idx if end_idx is not None else len(items)

    for index, turns in tqdm(items[start_idx:end_idx], total=end_idx - start_idx):
        try:
            seeker0, supporter0 = turns[0]
            exemplar_text = "\n".join(
                e.render_block() for e in selector.select(seeker0, supporter0, k=k)
            )
            query_blocks = "\n".join(build_query_block(s, t) for s, t in turns)
            prompt = BASIC_PROMPT.format(
                exemplars=exemplar_text, query_blocks=query_blocks
            )
            content = client.chat(
                system="You are a careful psychological-state annotator.",
                user=prompt,
                role="gen",
            )
            formatted = content.replace("\\n", "\n")
            (output_dir_p / f"Qwen{index}.txt").write_text(formatted, encoding="utf-8")
        except Exception as e:
            with (output_dir_p / "error_log.txt").open("a", encoding="utf-8") as fe:
                fe.write(f"Error at prompt {index}: {e}\n")
            continue


def main():
    ap = argparse.ArgumentParser(description="EToM 7-factor generation (CPsyCounD).")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="data/etom/cpsy_raw")
    ap.add_argument("--exemplar-bank", default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()
    run(
        args.input,
        args.output_dir,
        exemplar_bank=args.exemplar_bank,
        k=args.top_k,
        start_idx=args.start,
        end_idx=args.end,
    )


if __name__ == "__main__":
    main()
