"""Camera orientation — rotate in 90° steps and mirror, applied to every frame.

Cameras get mounted sideways, or look through a mirror, and the operator wants
the picture upright. The Thorlabs TSI SDK has no readout flip or rotation (none
of TLCamera's attributes offers one), so this is done in software, once, in the
acquisition engine right after the grab. Everything downstream — the live
stream, recordings, timelapse stills, snapshots, auto-exposure — therefore sees
the same oriented frames: what you see is what gets saved.

What does NOT rotate is the sensor. ROI and binning are hardware operations in
SENSOR coordinates and stay that way (presets store them unrotated), so the
camera page maps between the displayed, oriented image and sensor pixels.

Model
-----
An orientation is ``(rotate, mirror)``: ``mirror`` flips left-right in the
sensor's own frame FIRST, then the result is turned ``rotate`` degrees
clockwise (0/90/180/270). Those 8 combinations are every way to lay a rectangle
on a screen, and each has exactly one representation. In numpy::

    np.rot90(np.fliplr(a) if mirror else a, -rotate // 90)

The buttons act on the picture AS DISPLAYED ("flip what I see left-right"), so
they compose onto the current orientation instead of setting it:

    cw      rotate + 90
    ccw     rotate - 90
    flip_h  rotate -> -rotate,      mirror -> not mirror
    flip_v  rotate -> 180 - rotate, mirror -> not mirror

(Reflecting a rotated image equals mirroring first and rotating the other way;
flip_v is flip_h plus a half turn. tests/test_orientation.py checks all 32
state x operation combinations against numpy.)

Persisted per camera at ``<repo>/orientations/<camera>.json``, because it
describes how the camera is MOUNTED: it must survive a restart, unlike a preset
you choose to load.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .paths import DATA_ROOT, data_dir

OPS = ("cw", "ccw", "flip_h", "flip_v", "reset")

_DIR = "orientations"


@dataclass(frozen=True)
class Orientation:
    rotate: int = 0          # degrees clockwise: 0, 90, 180 or 270
    mirror: bool = False     # left-right flip, applied BEFORE the rotation

    def __post_init__(self) -> None:
        r = int(self.rotate)
        if r % 90:
            raise ValueError("rotate must be a multiple of 90 degrees")
        object.__setattr__(self, "rotate", r % 360)
        object.__setattr__(self, "mirror", bool(self.mirror))

    # --- properties ----------------------------------------------------------
    @property
    def is_identity(self) -> bool:
        return self.rotate == 0 and not self.mirror

    @property
    def swaps_axes(self) -> bool:
        """True when displayed x runs along sensor y (a quarter turn)."""
        return self.rotate in (90, 270)

    # --- frames --------------------------------------------------------------
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Orient a native frame. Identity returns the SAME array (zero cost);
        anything else returns a new C-contiguous array, because cv2 and the
        recorder's raw writer both need one (rot90/fliplr alone give views)."""
        if self.is_identity:
            return data
        a = data[:, ::-1] if self.mirror else data
        if self.rotate:
            a = np.rot90(a, -(self.rotate // 90), axes=(0, 1))
        return np.ascontiguousarray(a)

    def size(self, w: int, h: int) -> Tuple[int, int]:
        """Oriented (width, height) of a native (width, height) image."""
        return (h, w) if self.swaps_axes else (w, h)

    # --- composition ---------------------------------------------------------
    def compose(self, op: str) -> "Orientation":
        """The orientation after applying a display-space operation."""
        if op == "cw":
            return Orientation(self.rotate + 90, self.mirror)
        if op == "ccw":
            return Orientation(self.rotate - 90, self.mirror)
        if op == "flip_h":
            return Orientation(-self.rotate, not self.mirror)
        if op == "flip_v":
            return Orientation(180 - self.rotate, not self.mirror)
        if op == "reset":
            return Orientation()
        raise ValueError(f"unknown orientation op '{op}' (use one of {', '.join(OPS)})")

    # --- description ---------------------------------------------------------
    def describe(self) -> str:
        """Plain words for the operator and for a sidecar reader."""
        turns = {0: "", 90: "rotated 90° clockwise", 180: "rotated 180°",
                 270: "rotated 90° counter-clockwise"}
        if not self.mirror:
            return turns[self.rotate] or "as the sensor reads it"
        if self.rotate == 0:
            return "flipped left–right"
        if self.rotate == 180:
            return "flipped top–bottom"
        return f"flipped left–right, then {turns[self.rotate]}"

    def as_dict(self) -> dict:
        return {"rotate": self.rotate, "mirror": self.mirror,
                "label": self.describe()}

    @classmethod
    def from_dict(cls, d) -> "Orientation":
        """Tolerant parse: anything malformed reads as upright."""
        try:
            return cls(int(d.get("rotate", 0)), bool(d.get("mirror", False)))
        except (AttributeError, TypeError, ValueError):
            return cls()


# --- persistence (per camera) ---------------------------------------------------
def _safe(name: str) -> str:
    return "".join(c for c in str(name) if c.isalnum() or c in "-_.") or "camera"


def load(camera: str) -> Orientation:
    """The camera's saved orientation; upright if none is saved or it is unreadable.
    Reads without creating the directory, so an unrotated rig leaves no trace."""
    path = os.path.join(DATA_ROOT, _DIR, f"{_safe(camera)}.json")
    try:
        with open(path) as f:
            return Orientation.from_dict(json.load(f))
    except (OSError, ValueError):
        return Orientation()


def save(camera: str, orientation: Orientation) -> str:
    path = os.path.join(data_dir(_DIR), f"{_safe(camera)}.json")
    with open(path, "w") as f:
        json.dump({"rotate": orientation.rotate, "mirror": orientation.mirror}, f,
                  indent=2)
    return path
