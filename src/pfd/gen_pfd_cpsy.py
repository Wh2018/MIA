"""PFD (Personal-Factual Discriminator) generation for CPsyCounD.

Replaces IJCAI/prompt/src/ER_CPsy_Seven.py.

Input shapes accepted:
  (A) Original CPsy turn-list (one element per turn, grouped by Item[0]):
      [ { "Item": [id, ...], "utterance": [seeker, supporter],
          "Belief": [..], "Intension": [..], "Desire": [..],
          "Emotion": [..], "Event": [..], "Background": [..],
          "Prediction": [..] }, ... ]
      (Note the original key 'Intension' — sic — is preserved for compat.)
  (B) Parsed 7-factor JSON from src/etom/parse_seven.py:
      { "Qwen{N}.txt": [ {"utterance": {...}, "Belief": "...", ...}, ... ] }

For each conversation we emit a Qwen{N}.txt file of bracketed
[Approach / Explanation] blocks, one per turn.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from src.llm_client import LLMClient
from src.pfd.prompts import (
    PFD_PROMPT_CPSY, DEFAULT_CPSY_EXEMPLARS,
    build_cpsy_exemplar, build_cpsy_query,
)
from src.utils.paths import project_root


def _group_turns(data) -> "OrderedDict[int, List[Tuple[str,str,Dict[str,str]]]]":
    """Return mapping conv_id → list of (seeker, supporter, mind7) tuples."""
    grouped: "OrderedDict[int, list]" = OrderedDict()

    if isinstance(data, list):
        # Shape (A)
        for turn in data:
            item_id = turn["Item"][0] if isinstance(turn.get("Item"), list) else turn["Item"]
            seeker = turn["utterance"][0]
            supporter = turn["utterance"][1]
            mind = {
                "Belief": "；".join(turn.get("Belief", [])) or "None",
                "Intention": "；".join(turn.get("Intension", turn.get("Intention", []))) or "None",
                "Desire": "；".join(turn.get("Desire", [])) or "None",
                "Emotion": "；".join(turn.get("Emotion", [])) or "None",
                "Fact": "；".join(turn.get("Event", turn.get("Fact", []))) or "None",
                "Cause": "；".join(turn.get("Background", turn.get("Cause", []))) or "None",
                "Result": "；".join(turn.get("Prediction", turn.get("Result", []))) or "None",
            }
            grouped.setdefault(item_id, []).append((seeker, supporter, mind))

    elif isinstance(data, dict):
        # Shape (B): keyed by filename like "Qwen5.txt"
        for fname, dialogues in data.items():
            m = re.search(r"Qwen(\d+)", fname)
            cid = int(m.group(1)) if m else len(grouped)
            for d in dialogues:
                seeker = d["utterance"]["seeker"]
                supporter = d["utterance"]["supporter"]
                mind = {k: d.get(k, "None") for k in
                        ("Belief", "Intention", "Desire", "Emotion",
                         "Fact", "Cause", "Result")}
                grouped.setdefault(cid, []).append((seeker, supporter, mind))
    else:
        raise ValueError(f"Unsupported input shape: {type(data)}")

    return grouped


def run(
    input_file: str,
    output_dir: str,
    *,
    start_idx: int | None = None,
    end_idx: int | None = None,
):
    root = project_root()
    input_path = Path(input_file)
    if not input_path.is_absolute():
        input_path = root / input_path
    output_dir_p = Path(output_dir)
    if not output_dir_p.is_absolute():
        output_dir_p = root / output_dir
    output_dir_p.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    grouped = _group_turns(data)

    exemplar_block = "\n".join(
        build_cpsy_exemplar(
            e["seeker"], e["supporter"], e["mind"], e["approach"], e["explanation"]
        )
        for e in DEFAULT_CPSY_EXEMPLARS
    )

    client = LLMClient()

    items = list(grouped.items())
    for item_id, turns in tqdm(items):
        if start_idx is not None and item_id < start_idx:
            continue
        if end_idx is not None and item_id > end_idx:
            continue
        try:
            query_blocks = "".join(
                build_cpsy_query(s, sup, mind) for s, sup, mind in turns
            )
            prompt = PFD_PROMPT_CPSY.format(
                exemplars=exemplar_block, query_blocks=query_blocks
            )
            content = client.chat(
                system="You are a careful PFD labeler.",
                user=prompt,
                role="gen",
            )
            (output_dir_p / f"Qwen{item_id}.txt").write_text(
                content.replace("\\n", "\n"), encoding="utf-8"
            )
        except Exception as e:
            with (output_dir_p / "error_log.txt").open("a", encoding="utf-8") as fe:
                fe.write(f"Error at item_id {item_id}: {e}\n")
            continue


def main():
    ap = argparse.ArgumentParser(description="PFD (Approach) generation — CPsyCounD.")
    ap.add_argument("--input", required=True,
                    help="CPsy 7-factor JSON (either turn-list or parsed dict shape).")
    ap.add_argument("--output-dir", default="data/pfd/cpsy_raw")
    ap.add_argument("--start", type=int, default=None,
                    help="Start item_id (inclusive); only used for list-shape input.")
    ap.add_argument("--end", type=int, default=None,
                    help="End item_id (inclusive).")
    args = ap.parse_args()
    run(args.input, args.output_dir, start_idx=args.start, end_idx=args.end)


if __name__ == "__main__":
    main()
