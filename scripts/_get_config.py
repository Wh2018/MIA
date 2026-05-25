"""Tiny YAML reader used by shell scripts.

Usage:
    python scripts/_get_config.py <config_file> <key> [--root PATH] [--resolve-path]

Reads `config/<config_file>` (path relative to --root, default cwd) and
prints the value at `key` to stdout. With --resolve-path, treats the value
as a filesystem path and prints its absolute form (relative paths resolve
against --root). Exits non-zero with an empty stdout if the key is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_file", help="e.g. training.yaml or llm_api.yaml")
    ap.add_argument("key")
    ap.add_argument("--root", default=".", help="Project root (default: cwd).")
    ap.add_argument("--resolve-path", action="store_true",
                    help="Treat the value as a path; print its absolute form.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cfg_path = root / "config" / args.config_file
    if not cfg_path.exists():
        print(f"missing config: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if args.key not in data:
        print(f"missing key '{args.key}' in {cfg_path}", file=sys.stderr)
        sys.exit(3)

    value = data[args.key]
    if args.resolve_path and isinstance(value, str):
        p = Path(value)
        if not p.is_absolute():
            p = (root / p).resolve()
        print(str(p))
    else:
        print(value)


if __name__ == "__main__":
    main()
