"""Standalone test GUI for Hayear (ToupTek-OEM) cameras.

Thin shell over the shared harness in ``camera_dock.preview`` (same as the Basler
and Zelux shells) — all GUI logic lives once in the dock. Run with::

    python -m hayear.gui

STATUS: depends on the unvalidated hycam driver + the hycam Python SDK; will run
once a camera + ``hycam.py`` are present.
"""

from __future__ import annotations

import os
import sys

# Shared harness lives in the parent dock repo; this package sits at
# .../xsphere-camera-dock/hayear/hayear, so the dock root is three dirs up.
_DOCK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _DOCK_ROOT not in sys.path:
    sys.path.insert(0, _DOCK_ROOT)

from .camera import HayearCamera

FPS_CAP = 60.0   # nominal; refine once measured against hardware


def main() -> None:
    try:
        from camera_dock.preview import run
    except ImportError as exc:
        raise SystemExit(
            "Could not import the shared GUI (camera_dock.preview). Run this from "
            "within the xsphere-camera-dock checkout. Original error: " + str(exc))
    run(HayearCamera(), fps_cap=FPS_CAP)


if __name__ == "__main__":
    main()
