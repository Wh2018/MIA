"""PFD (Personal-Factual Discriminator) generation for ESConv.

Replaces IJCAI/prompt/src/ER_ESC_pure_mind.py.

Reads:
  - Formatted ESC JSON  (list[{"history": [["seeker","supporter"], ...]}])
  - Directory of Qwen{idx}.txt files holding 4-factor mind info per turn
    (Belief / Intention / Desire / Emotion) — same format produced by the
    EToM step in the legacy "pure_mind" branch.

Writes Qwen{idx}.txt files containing the dialogue plus inferred Approach
and Explanation per turn. Run parse_pfd.py to fold these back into JSON.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from src.llm_client import LLMClient
from src.pfd.prompts import (
    PFD_PROMPT_ESC, DEFAULT_ESC_EXEMPLARS,
    build_esc_exemplar, build_esc_query,
)
from src.utils.paths import project_root


def _read_mind_lines(file_path: Path) -> List[dict]:
    """Parse a Qwen*.txt of 4-factor mind blocks into a list of dicts."""
    if not file_path.exists():
        return []
    titles = ("Belief", "Intention", "Desire", "Emotion")
    out: List[dict] = []
    current = {t: "None" for t in titles}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            for t in titles:
                if line.startswith(f"{t}:"):
                    current[t] = line.split(":", 1)[1].strip()
                    if all(v != "None" for v in current.values()):
                        out.append(current)
                        current = {t: "None" for t in titles}
                    break
    if any(v != "None" for v in current.values()):
        out.append(current)
    return out


def run(
    input_file: str,
    mind_dir: str,
    output_dir: str,
    *,
    start_idx: int = 0,
    end_idx: int | None = None,
):
    root = project_root()
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = root / input_path
    mind_dir_p = Path(mind_dir)
    if not mind_dir_p.is_absolute():
        mind_dir_p = root / mind_dir
    output_dir_p = Path(output_dir)
    if not output_dir_p.is_absolute():
        output_dir_p = root / output_dir
    output_dir_p.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if end_idx is None:
        end_idx = len(data)

    exemplar_block = "\n".join(
        build_esc_exemplar(
            e["seeker"], e["supporter"], e["mind"], e["approach"], e["explanation"]
        )
        for e in DEFAULT_ESC_EXEMPLARS
    )

    client = LLMClient()

    for index, entry in enumerate(tqdm(data[start_idx:end_idx]), start=start_idx):
        try:
            mind_list = _read_mind_lines(mind_dir_p / f"Qwen{index}.txt")
            query_blocks: List[str] = []
            for i, conv in enumerate(entry["history"]):
                mind = mind_list[i] if i < len(mind_list) else {
                    "Belief": "None", "Intention": "None",
                    "Desire": "None", "Emotion": "None",
                }
                query_blocks.append(build_esc_query(conv[0], conv[1], mind))
            prompt = PFD_PROMPT_ESC.format(
                exemplars=exemplar_block, query_blocks="".join(query_blocks)
            )
            content = client.chat(
                system="You are a careful PFD labeler.",
                user=prompt,
                role="gen",
            )
            (output_dir_p / f"Qwen{index}.txt").write_text(
                content.replace("\\n", "\n"), encoding="utf-8"
            )
        except Exception as e:
            with (output_dir_p / "error_log.txt").open("a", encoding="utf-8") as fe:
                fe.write(f"Error at prompt {index}: {e}\n")
            continue


def main():
    ap = argparse.ArgumentParser(description="PFD (Approach) generation — ESConv.")
    ap.add_argument("--input", required=True, help="Formatted ESC JSON.")
    ap.add_argument("--mind-dir", required=True,
                    help="Directory of Qwen*.txt 4-factor mind files.")
    ap.add_argument("--output-dir", default="data/pfd/esc_raw")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    args = ap.parse_args()
    run(
        args.input, args.mind_dir, args.output_dir,
        start_idx=args.start, end_idx=args.end,
    )


if __name__ == "__main__":
    main()
