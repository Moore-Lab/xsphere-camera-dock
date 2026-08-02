"""Data directories, anchored to the dock repo root — never the CWD.

The parent DAQ launches with CWD = its own repo; standalone runs launch with
CWD = the dock. Anchoring here makes both write the same tree (recordings/,
captures/, presets/ under the dock repo), which is where all existing data
already lives.
"""

from __future__ import annotations

import os

DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def data_dir(name: str) -> str:
    """Absolute path of a data subdirectory (created on demand)."""
    d = os.path.join(DATA_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def resolve(path: str) -> str:
    """Absolute path for a possibly-relative user-supplied path (e.g. the
    timelapse save folder) — relative paths land under the dock repo root."""
    path = os.path.expanduser(str(path))
    if not os.path.isabs(path):
        path = os.path.join(DATA_ROOT, path)
    return path
