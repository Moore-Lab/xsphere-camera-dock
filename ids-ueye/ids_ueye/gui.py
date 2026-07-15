"""Standalone test GUI for IDS uEye cameras.

Thin shell over the shared, camera-agnostic harness in ``camera_dock.preview`` —
so the test GUI, the dock, and the DAQ share one implementation (no duplication).
All the real logic (threaded acquisition, decoupled preview, hybrid recording at
full data rate, snapshot) lives in the dock and is driven here with the uEye
driver. Frames are Mono8, so no bit-depth scaling is needed.

Run with::

    python -m ids_ueye.gui

Controls: exposure/fps trackbars, ``s`` snapshot, ``r`` record, ``q``/ESC quit.
"""

from __future__ import annotations

import os
import sys

# Shared harness lives in the parent dock repo; this package sits at
# .../xsphere-camera-dock/ids-ueye/ids_ueye, so the dock root is three
# directories up. Add it so ``camera_dock`` imports inside the dock checkout. The
# driver itself never imports the dock — only this test shell does.
_DOCK_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _DOCK_ROOT not in sys.path:
    sys.path.insert(0, _DOCK_ROOT)

from .camera import IDSUEye

FPS_CAP = 250.0   # fps-slider top; achievable rate depends on pixel clock + AOI


def main() -> None:
    try:
        from camera_dock.preview import run
    except ImportError as exc:
        raise SystemExit(
            "Could not import the shared GUI (camera_dock.preview). Run this from "
            "within the xsphere-camera-dock checkout. Original error: " + str(exc))
    run(IDSUEye(), fps_cap=FPS_CAP)


if __name__ == "__main__":
    main()
