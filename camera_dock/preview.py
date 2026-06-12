"""Shared OpenCV test-GUI shell — reused by every camera and the dock.

This is the camera-agnostic preview/record harness that each per-camera ``gui.py``
launches with its own driver (``run(BaslerACA1440())`` etc.). It is built entirely
on :class:`~camera_dock.base.CameraBase` plus the shared
:class:`~camera_dock.engine.AcquisitionEngine`,
:class:`~camera_dock.recorder.HybridRecorder`, and the helpers in
:mod:`camera_dock.imaging` — so the same code drives the test GUIs today and the
dock's preview panes tomorrow.

Acquisition runs on its own thread at the camera's full rate; the window just
samples the latest frame at ~30 fps. The overlay shows the true acquisition fps
*and* the preview fps. Recording captures every frame at the acquisition rate (no
drops), then encodes to a lossless video after you press stop.

Controls
--------
    exposure / fps / gain : trackbars at the top
    e / t / g             : type an exact exposure / fps / gain (digits, Enter)
    a                     : one-shot auto-exposure
    h                     : toggle histogram
    f                     : cycle snapshot format (tiff/png/npy)
    s                     : snapshot     -> captures/
    r                     : record on/off -> recordings/ (encodes on stop)
    q or ESC              : quit
"""

from __future__ import annotations

import math
import os
from datetime import datetime
from time import perf_counter

import cv2
import numpy as np

from . import imaging
from .engine import AcquisitionEngine
from .recorder import HybridRecorder

STEPS = 1000              # trackbar integer resolution
PREVIEW_FPS = 30.0        # window refresh rate (independent of capture)


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


def _clamp_pos(pos: float) -> int:
    return int(max(0, min(STEPS, pos)))


def run(camera, *, title: str | None = None, fps_cap: float | None = None,
        exp_cap_us: float = 100_000.0) -> None:
    """Launch the shared test GUI against ``camera`` (a connected-or-not driver)."""
    try:
        camera.connect()
    except Exception as exc:
        print(f"Could not connect to camera: {exc}")
        print("Is another client (ThorCam / Pylon Viewer) holding the camera?")
        return

    info = camera.device_info
    bit_depth = int(getattr(camera, "bit_depth", 8))
    max_value = imaging.max_value_for(bit_depth)
    to_8bit = _make_to_8bit(bit_depth)
    window = title or f"{info.get('model', 'camera')} - test GUI"

    exp_lo, exp_hi = camera.exposure_range()
    fps_lo, fps_hi = camera.frame_rate_range()
    gain_lo, gain_hi = camera.gain_range()
    exp_cap = min(exp_hi, exp_cap_us)
    has_exposure = exp_hi > exp_lo
    has_fps = fps_hi > 0
    fps_top = min(fps_hi, fps_cap) if (has_fps and fps_cap) else fps_hi
    has_gain = gain_hi > gain_lo

    try:
        camera.roi_range()
        has_roi = True
    except Exception:
        has_roi = False
    try:
        bx_max, by_max = camera.binning_range()
    except Exception:
        bx_max = by_max = 1
    has_binning = max(bx_max, by_max) > 1

    if has_exposure:
        camera.set_exposure(min(5000.0, exp_hi))
    if has_fps:
        camera.set_frame_rate(min(fps_top, fps_hi))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 780)

    if has_exposure:
        cv2.createTrackbar("exposure", window,
                           _clamp_pos(_geom_frac(exp_lo, exp_cap, camera.get_exposure()) * STEPS),
                           STEPS, lambda v: camera.set_exposure(_geom(exp_lo, exp_cap, v / STEPS)))
    if has_fps:
        cv2.createTrackbar("fps", window,
                           _clamp_pos((camera.get_frame_rate() - fps_lo) / (fps_top - fps_lo) * STEPS),
                           STEPS,
                           lambda v: camera.set_frame_rate(fps_lo + (fps_top - fps_lo) * v / STEPS))
    if has_gain:
        cv2.createTrackbar("gain", window,
                           _clamp_pos((camera.get_gain() - gain_lo) / (gain_hi - gain_lo) * STEPS),
                           STEPS, lambda v: camera.set_gain(gain_lo + (gain_hi - gain_lo) * v / STEPS))

    # --- trackbar sync (after numeric entry / auto-exposure set a value) ---
    def sync_exposure():
        if has_exposure:
            cv2.setTrackbarPos("exposure", window,
                               _clamp_pos(_geom_frac(exp_lo, exp_cap, camera.get_exposure()) * STEPS))

    def sync_fps():
        if has_fps:
            cv2.setTrackbarPos("fps", window,
                               _clamp_pos((camera.get_frame_rate() - fps_lo) / (fps_top - fps_lo) * STEPS))

    def sync_gain():
        if has_gain:
            cv2.setTrackbarPos("gain", window,
                               _clamp_pos((camera.get_gain() - gain_lo) / (gain_hi - gain_lo) * STEPS))

    engine = AcquisitionEngine(camera)
    engine.start()
    recorder: HybridRecorder | None = None
    rec_path = ""

    # fresh-frame grab for auto-exposure: wait a few frames after an exposure change
    def grab_fresh():
        _, start = engine.latest()
        t0 = perf_counter()
        while perf_counter() - t0 < 2.0:
            frame, idx = engine.latest()
            if frame is not None and idx >= start + 4:
                return frame
            cv2.waitKey(5)
        return engine.latest()[0]

    snap_fmt_i = 0
    show_hist = False
    entry_field: str | None = None
    entry_buf = ""

    def apply_entry():
        nonlocal entry_field, entry_buf
        try:
            val = float(entry_buf)
        except ValueError:
            val = None
        if val is not None:
            if entry_field == "exposure":
                camera.set_exposure(val); sync_exposure()
            elif entry_field == "gain" and has_gain:
                camera.set_gain(val); sync_gain()
            elif entry_field == "fps" and has_fps:
                camera.set_frame_rate(val); sync_fps()
        entry_field, entry_buf = None, ""

    w, h = camera.sensor_size()
    region_keys = (" o=ROI O=reset" if has_roi else "") + (" b=bin" if has_binning else "")
    print(f"Live: {info.get('model')} s/n {info.get('serial')}  ({w}x{h}, {bit_depth}-bit).  "
          f"e/t/g=type  a=auto-exp  h=hist  f=fmt  s=snap  r=rec{region_keys}  q/ESC=quit")

    prev_n, prev_t0, preview_fps = 0, perf_counter(), 0.0
    last_index = -1
    frame = frame8 = None

    def _restart(action) -> None:
        """Stop the engine, change the sensor region, restart. Blocked while recording."""
        nonlocal last_index, frame8
        if recorder is not None:
            print("stop recording before changing ROI / binning")
            return
        engine.stop()
        try:
            action()
        except Exception as exc:
            print(f"ROI/binning change failed: {exc}")
        engine.start()
        last_index, frame8 = -1, None

    def do_roi_select() -> None:
        camera.reset_roi()                      # select on the full frame for absolute coords
        camera.start(False)
        f = camera.grab()
        camera.stop()
        sel = cv2.selectROI(window, to_8bit(f), showCrosshair=False, fromCenter=False)
        x, y, wd, ht = sel
        if wd > 0 and ht > 0:
            camera.set_roi(int(x), int(y), int(wd), int(ht))
            print(f"ROI -> {camera.get_roi()}")
        else:
            print("ROI select cancelled (full frame)")

    def do_binning_cycle() -> None:
        bx, _ = camera.get_binning()
        nxt = {1: 2, 2: 4, 4: 1}.get(bx, 1)
        if nxt > max(bx_max, by_max):
            nxt = 1
        camera.set_binning(nxt, nxt)
        print(f"binning -> {camera.get_binning()}, sensor {camera.sensor_size()}")

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
            if show_hist:
                imaging.histogram_overlay(bgr, frame8)

            prev_n += 1
            now = perf_counter()
            if now - prev_t0 >= 0.5:
                preview_fps, prev_n, prev_t0 = prev_n / (now - prev_t0), 0, now

            sat = imaging.saturation_fraction(frame, max_value) * 100.0
            target = f"{camera.get_frame_rate():.1f}" if has_fps else "n/a"
            gain_txt = f"   gain {camera.get_gain():.1f}" if has_gain else ""
            exp_txt = f"exp {camera.get_exposure():.0f}us  " if has_exposure else ""
            status = (f"acq {engine.acquisition_fps:5.1f}  prev {preview_fps:4.1f}  "
                      f"{exp_txt}fps {target}{gain_txt}  sat {sat:4.1f}%")
            cv2.putText(bgr, status, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 1, cv2.LINE_AA)

            line2 = None
            if entry_field is not None:
                line2 = f"{entry_field} = {entry_buf}_  (Enter=set  ESC=cancel)"
            else:
                line2 = f"fmt={imaging.SNAPSHOT_FORMATS[snap_fmt_i]}"
            cv2.putText(bgr, line2, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)

            if recorder is not None:
                cv2.putText(bgr, f"REC {recorder._captured}", (bgr.shape[1] - 150, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            imaging.draw_timestamp(bgr, imaging.eastern_now())   # New Haven time (live)
            cv2.imshow(window, bgr)

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(max(1, int(1000 / PREVIEW_FPS))) & 0xFF

            # --- numeric entry mode captures all keys until Enter/ESC ---
            if entry_field is not None:
                if key in (13, 10):
                    apply_entry()
                elif key == 27:
                    entry_field, entry_buf = None, ""
                elif key == 8:
                    entry_buf = entry_buf[:-1]
                elif key != 255 and chr(key) in "0123456789.":
                    entry_buf += chr(key)
                continue

            if key in (27, ord("q")):
                break
            elif key == ord("e") and has_exposure:
                entry_field, entry_buf = "exposure", ""
            elif key == ord("t") and has_fps:
                entry_field, entry_buf = "fps", ""
            elif key == ord("g") and has_gain:
                entry_field, entry_buf = "gain", ""
            elif key == ord("a") and has_exposure:
                print("auto-exposing...")
                exp = imaging.auto_expose(camera, grab_fresh)
                sync_exposure()
                print(f"auto-exposure -> {exp:.0f} us")
            elif key == ord("h"):
                show_hist = not show_hist
            elif key == ord("o") and has_roi:
                _restart(do_roi_select)
            elif key == ord("O") and has_roi:
                _restart(camera.reset_roi)
            elif key == ord("b") and has_binning:
                _restart(do_binning_cycle)
            elif key == ord("f"):
                snap_fmt_i = (snap_fmt_i + 1) % len(imaging.SNAPSHOT_FORMATS)
                print(f"snapshot format -> {imaging.SNAPSHOT_FORMATS[snap_fmt_i]}")
            elif key == ord("s"):
                os.makedirs("captures", exist_ok=True)
                base = datetime.now().strftime(f"captures/{_slug(info)}_%Y%m%d_%H%M%S_%f")
                path = imaging.save_snapshot(frame, base, imaging.SNAPSHOT_FORMATS[snap_fmt_i])
                print(f"snapshot -> {path}")
            elif key == ord("r"):
                if recorder is None:
                    recorder = HybridRecorder(clock=imaging.eastern_now)
                    engine.set_sink(recorder.submit)
                    os.makedirs("recordings", exist_ok=True)
                    rec_path = datetime.now().strftime(f"recordings/{_slug(info)}_%Y%m%d_%H%M%S.avi")
                    print(f"recording... (capturing at full rate -> {rec_path})")
                else:
                    engine.set_sink(None)
                    fps = engine.acquisition_fps or camera.get_frame_rate() or 30.0
                    print("encoding...")
                    stats = recorder.stop_and_encode(rec_path, fps, to_8bit,
                                                     stamp=imaging.draw_timestamp)
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
