"""Shared OpenCV test-GUI shell — reused by every camera and the dock.

This is the camera-agnostic preview/record harness that each per-camera ``gui.py``
launches with its own driver (``run(BaslerACA1440())`` etc.). It is built entirely
on :class:`~camera_dock.base.CameraBase` plus the shared
:class:`~camera_dock.engine.AcquisitionEngine` and
:class:`~camera_dock.recorder.HybridRecorder`, so the same code drives the test
GUIs today and the dock's preview panes tomorrow.

Key behaviour (vs. a naive single-loop GUI): acquisition runs on its own thread at
the camera's full rate; the window just samples the latest frame at ~30 fps. The
overlay shows **two** rates honestly — the true acquisition fps and the preview
fps. Recording captures every frame at the acquisition rate (no drops), then
encodes to a lossless video after you press stop.

Controls
--------
    exposure / fps : trackbars at the top
    s              : snapshot   -> captures/   (full bit depth, TIFF)
    r              : record on/off -> recordings/   (encodes on stop)
    q or ESC       : quit
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from time import perf_counter

import cv2
import numpy as np

from .engine import AcquisitionEngine
from .recorder import HybridRecorder

STEPS = 1000              # trackbar integer resolution
PREVIEW_FPS = 30.0        # how often the window refreshes (independent of capture)


def _geom(lo: float, hi: float, frac: float) -> float:
    lo = max(lo, 1e-6)
    return lo * (hi / lo) ** max(0.0, min(1.0, frac))


def _geom_frac(lo: float, hi: float, value: float) -> float:
    lo = max(lo, 1e-6)
    return math.log(max(value, lo) / lo) / math.log(hi / lo)


def _make_to_8bit(bit_depth: int):
    shift = max(bit_depth - 8, 0)
    if shift == 0:
        return lambda f: f if f.dtype == np.uint8 else f.astype(np.uint8)
    return lambda f: (f >> shift).astype(np.uint8)


def run(camera, *, title: str | None = None, fps_cap: float | None = None,
        exp_cap_us: float = 100_000.0) -> None:
    """Launch the shared test GUI against ``camera`` (a connected-or-not driver).

    Parameters
    ----------
    camera:    any :class:`~camera_dock.base.CameraBase` driver instance.
    title:     window title; defaults to the camera model.
    fps_cap:   upper end of the fps slider; defaults to the camera's max.
    exp_cap_us: upper end of the exposure slider.
    """
    try:
        camera.connect()
    except Exception as exc:
        print(f"Could not connect to camera: {exc}")
        print("Is another client (ThorCam / Pylon Viewer) holding the camera?")
        return

    info = camera.device_info
    bit_depth = int(getattr(camera, "bit_depth", 8))
    to_8bit = _make_to_8bit(bit_depth)
    window = title or f"{info.get('model', 'camera')} - test GUI"

    exp_lo, exp_hi = camera.exposure_range()
    fps_lo, fps_hi = camera.frame_rate_range()
    exp_cap = min(exp_hi, exp_cap_us)
    has_fps = fps_hi > 0
    fps_top = min(fps_hi, fps_cap) if (has_fps and fps_cap) else fps_hi

    camera.set_exposure(min(5000.0, exp_hi))
    if has_fps:
        camera.set_frame_rate(min(fps_top, fps_hi))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 760)

    cv2.createTrackbar("exposure", window,
                       int(_geom_frac(exp_lo, exp_cap, camera.get_exposure()) * STEPS),
                       STEPS, lambda v: camera.set_exposure(_geom(exp_lo, exp_cap, v / STEPS)))
    if has_fps:
        cv2.createTrackbar("fps", window,
                           int((camera.get_frame_rate() - fps_lo) / (fps_top - fps_lo) * STEPS),
                           STEPS,
                           lambda v: camera.set_frame_rate(fps_lo + (fps_top - fps_lo) * v / STEPS))

    engine = AcquisitionEngine(camera)
    engine.start()
    recorder: HybridRecorder | None = None
    rec_path = ""

    w, h = camera.sensor_size()
    print(f"Live: {info.get('model')} s/n {info.get('serial')}  ({w}x{h}, {bit_depth}-bit).  "
          f"s=snapshot  r=record  q/ESC=quit")

    prev_n, prev_t0, preview_fps = 0, perf_counter(), 0.0
    last_index = -1
    frame8 = None
    try:
        while True:
            frame, index = engine.latest()
            if frame is not None and index != last_index:
                last_index = index
                frame8 = to_8bit(frame)

            if frame8 is None:
                if cv2.waitKey(20) & 0xFF in (27, ord("q")):
                    break
                continue

            bgr = cv2.cvtColor(frame8, cv2.COLOR_GRAY2BGR)

            prev_n += 1
            now = perf_counter()
            if now - prev_t0 >= 0.5:
                preview_fps, prev_n, prev_t0 = prev_n / (now - prev_t0), 0, now

            target = f"{camera.get_frame_rate():.1f}" if has_fps else "n/a"
            status = (f"acq {engine.acquisition_fps:5.1f} fps   preview {preview_fps:4.1f} fps   "
                      f"exp {camera.get_exposure():.0f} us   target {target}")
            cv2.putText(bgr, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
            if recorder is not None:
                cv2.putText(bgr, f"REC {recorder._captured}", (bgr.shape[1] - 150, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.imshow(window, bgr)

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(max(1, int(1000 / PREVIEW_FPS))) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("s"):
                os.makedirs("captures", exist_ok=True)
                name = datetime.now().strftime(f"captures/{_slug(info)}_%Y%m%d_%H%M%S_%f.tiff")
                cv2.imwrite(name, frame)   # native full bit depth
                print(f"snapshot -> {name}")
            elif key == ord("r"):
                if recorder is None:
                    recorder = HybridRecorder()
                    engine.set_sink(recorder.submit)
                    os.makedirs("recordings", exist_ok=True)
                    rec_path = datetime.now().strftime(f"recordings/{_slug(info)}_%Y%m%d_%H%M%S.avi")
                    print(f"recording... (capturing at full rate -> {rec_path})")
                else:
                    engine.set_sink(None)
                    fps = engine.acquisition_fps or camera.get_frame_rate() or 30.0
                    print("encoding...")
                    stats = recorder.stop_and_encode(rec_path, fps, to_8bit)
                    recorder = None
                    print(f"recording saved -> {stats['path']}  "
                          f"({stats['encoded']} frames @ {stats['capture_fps']} fps capture, "
                          f"dropped {stats['dropped']}, encode {stats['encode_seconds']}s)")
    finally:
        engine.set_sink(None)
        engine.stop()
        camera.disconnect()
        cv2.destroyAllWindows()


def _slug(info: dict) -> str:
    model = str(info.get("model", "camera")).replace(" ", "").replace("/", "-")
    return model.lower()
