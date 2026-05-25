"""Generate EToM 7-factor annotations for ESConv-style dialogues.

Replaces IJCAI/prompt/src/Seven_ESC.py:
  - Uses the unified LLMClient (config-driven endpoint/model).
  - Uses HRAGSelector to pick top-k exemplars per turn (paper's H-RAG claim),
    instead of a fixed static prefix.
  - All paths are relative; configurable via CLI flags.

Input JSON shape (matching the original):
    [
      {"history": [["seeker turn", "supporter turn"], ...]},
      ...
    ]

Output: per-conversation `.txt` files containing the LLM's 7-factor analysis
in the same format the downstream parser (parse_seven.py) expects.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from src.etom.hrag_selector import HRAGSelector
from src.etom.prompts import BASIC_PROMPT, build_query_block
from src.llm_client import LLMClient
from src.utils.paths import project_root


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

    bank = exemplar_bank or str(root / "config" / "etom_exemplars_esc.json")
    selector = HRAGSelector(bank)
    client = LLMClient()

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if end_idx is None:
        end_idx = len(data)

    for index, entry in enumerate(tqdm(data[start_idx:end_idx]), start=start_idx):
        turns = entry["history"]
        try:
            # Choose exemplars based on the first turn of this conversation
            # (one retrieval per conversation, matching the static-prefix idea
            # but now dynamic per dialogue).
            seeker0, supporter0 = turns[0][0], turns[0][1]
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
            out_file = output_dir_p / f"Qwen{index}.txt"
            out_file.write_text(formatted, encoding="utf-8")
        except Exception as e:
            err_file = output_dir_p / "error_log.txt"
            with err_file.open("a", encoding="utf-8") as fe:
                fe.write(f"Error at prompt {index}: {e}\n")
            continue


def main():
    ap = argparse.ArgumentParser(description="EToM 7-factor generation (ESConv).")
    ap.add_argument("--input", required=True, help="Formatted ESC JSON input.")
    ap.add_argument(
        "--output-dir",
        default="data/etom/esc_raw",
        help="Directory to write per-conversation .txt files into.",
    )
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
