"""Per-camera pixel -> micrometre calibration.

A microscope camera's scale depends on the optics in front of it, so each camera
keeps a small document of *named* calibrations (one per objective, say "10x" and
"40x") with one marked active. The active scale is what the UI uses to annotate
measurements in micrometres alongside pixels, and it is recorded into every video
sidecar so downstream analysis can convert without guessing.

Scale is stored as **micrometres per SENSOR pixel**, deliberately not per image
pixel: binning changes the image pixel size but not the sensor geometry, so a
calibration stays valid when binning changes.

One JSON file per camera at ``<repo>/calibrations/<camera>.json``::

    {"active": "10x",
     "entries": {"10x": {"um_per_px": 0.345, "label": "Nikon 10x",
                         "notes": "stage micrometer 2026-08-02", "saved": "..."}}}

Hand-editable on purpose — a lab notebook value can be typed straight in.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from .paths import data_dir


def _safe(name: str) -> str:
    return "".join(c for c in str(name) if c.isalnum() or c in "-_.") or "cal"


def _path(camera: str) -> str:
    return os.path.join(data_dir("calibrations"), f"{_safe(camera)}.json")


def read(camera: str) -> dict:
    """The camera's calibration document (empty skeleton if none exists)."""
    try:
        with open(_path(camera)) as f:
            doc = json.load(f)
        if not isinstance(doc, dict):
            raise ValueError("calibration file is not an object")
        doc.setdefault("active", None)
        doc.setdefault("entries", {})
        return doc
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return {"active": None, "entries": {}}


def write(camera: str, doc: dict) -> str:
    path = _path(camera)
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    return path


def save_entry(camera: str, name: str, um_per_px: float, *, label: str = "",
               notes: str = "", make_active: bool = True) -> dict:
    """Add/replace a named calibration; by default it becomes the active one."""
    um_per_px = float(um_per_px)
    if not (um_per_px > 0) or um_per_px == float("inf"):
        raise ValueError("um_per_px must be a positive number")
    name = _safe(name)
    doc = read(camera)
    doc["entries"][name] = {
        "um_per_px": um_per_px,
        "label": label or name,
        "notes": notes,
        "saved": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if make_active:
        doc["active"] = name
    write(camera, doc)
    return doc


def set_active(camera: str, name: Optional[str]) -> dict:
    doc = read(camera)
    if name is not None:
        name = _safe(name)
        if name not in doc["entries"]:
            raise FileNotFoundError(f"no calibration '{name}'")
    doc["active"] = name
    write(camera, doc)
    return doc


def delete_entry(camera: str, name: str) -> dict:
    name = _safe(name)
    doc = read(camera)
    doc["entries"].pop(name, None)
    if doc.get("active") == name:
        doc["active"] = None
    write(camera, doc)
    return doc


def active(camera: str) -> Optional[dict]:
    """The active calibration entry (with its name), or None."""
    doc = read(camera)
    name = doc.get("active")
    entry = doc.get("entries", {}).get(name) if name else None
    if entry is None:
        return None
    return {"name": name, **entry}


def um_per_px(camera: str) -> Optional[float]:
    entry = active(camera)
    return float(entry["um_per_px"]) if entry else None


def document(camera: str) -> dict:
    """The API view: active entry + every saved entry."""
    doc = read(camera)
    return {"active": doc.get("active"), "entry": active(camera),
            "entries": doc.get("entries", {})}
