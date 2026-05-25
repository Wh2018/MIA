"""Project-root resolution and data-path helpers.

All file I/O in code_new/ should resolve paths through `project_root()` so the
code is portable across machines. The project root is the directory containing
`config/llm_api.yaml`.
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    here = Path(__file__).resolve().parent
    for p in [here, *here.parents]:
        if (p / "config" / "llm_api.yaml").is_file():
            return p
    raise FileNotFoundError("project root (containing config/llm_api.yaml) not found")


def data_dir(*parts: str) -> Path:
    """Return project_root()/data/<parts...>, creating intermediate dirs."""
    p = project_root().joinpath("data", *parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
