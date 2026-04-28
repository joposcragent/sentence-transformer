"""REST service wrapping sentence-transformers (BGE-M3)."""

from __future__ import annotations

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_DIST_NAME = "st-service"


def _version_from_pyproject() -> str:
    root = Path(__file__).resolve().parents[2]
    with (root / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    return str(data["project"]["version"])


try:
    __version__ = version(_DIST_NAME)
except PackageNotFoundError:
    __version__ = _version_from_pyproject()
