"""Simulated camera — the dock's hardware-free test fixture.

Implements the full standardized interface (exposure/gain/fps controls, ROI,
binning, 10-bit frames with hardware-style metadata) so the eval server, the
drag-ROI UI, and the recording pipeline can be exercised without any hardware
or the DAQ. Frames show a moving crosshair + noise, brightness follows
exposure×gain, and the delivered rate follows the fps control — so "shrink the
ROI, watch the fps" behaves like a real sensor (rate rises as area drops).
"""

from __future__ import annotations

import threading
from time import perf_counter, sleep
from typing import Optional, Tuple

import numpy as np

from ..base import CameraDriver, ControlSpec, Frame, FrameMeta

_SW, _SH = 1440, 1080          # simulated sensor
_BIT_DEPTH = 10
_BASE_FPS_FULL = 35.0          # full-frame max rate; scales with 1/area like USB3


class SimCamera(CameraDriver):
    def __init__(self, serial: Optional[str] = None) -> None:
        super().__init__()
        self.serial = serial or "SIM0001"
        self._connected = False
        self._grabbing = False
        self._exp_us = 5000.0
        self._gain = 0.0
        self._fps = 30.0
        self._roi = (0, 0, _SW, _SH)
        self._bin = (1, 1)
        self._hw_count = 0
        self._t_last = 0.0
        self._geom_lock = threading.Lock()

    # --- lifecycle ---
    def connect(self) -> None:
        self._connected = True
        self._register(ControlSpec(
            name="exposure", units="us", kind="int", scale="log", step=1,
            get_range=lambda: (64.0, 5_000_000.0),
            getter=lambda: self._exp_us,
            setter=lambda v: setattr(self, "_exp_us", float(v)),
            display={"label": "exposure", "unit_display": "ms", "unit_scale": 1e-3}))
        self._register(ControlSpec(
            name="gain", units="index", kind="int", step=1,
            get_range=lambda: (0.0, 480.0),
            getter=lambda: self._gain,
            setter=lambda v: setattr(self, "_gain", float(v)),
            display={"label": "gain", "db_lo": 0.0, "db_hi": 48.0}))
        self._register(ControlSpec(
            name="fps", units="fps", kind="float",
            get_range=self._fps_range,
            getter=lambda: self._fps,
            setter=lambda v: setattr(self, "_fps", float(v)),
            display={"label": "target fps"}))

    def _fps_range(self) -> Tuple[float, float]:
        _, _, w, h = self._roi
        area = (w // self._bin[0]) * (h // self._bin[1])
        return (1.0, _BASE_FPS_FULL * (_SW * _SH) / max(area, 1))

    def disconnect(self) -> None:
        self._grabbing = False
        self._connected = False
        for n in list(self._controls):
            self._unregister(n)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def device_info(self) -> dict:
        return {"model": "SimCam", "serial": self.serial, "vendor": "xsphere",
                "sdk": "builtin"}

    @property
    def bit_depth(self) -> int:
        return _BIT_DEPTH

    # --- geometry ---
    def sensor_size(self) -> Tuple[int, int]:
        return (_SW, _SH)

    def image_size(self) -> Tuple[int, int]:
        _, _, w, h = self._roi
        return (w // self._bin[0], h // self._bin[1])

    def roi_range(self) -> dict:
        return {"w_min": 80, "h_min": 4, "w_max": _SW, "h_max": _SH,
                "w_inc": 2, "h_inc": 2, "x_inc": 2, "y_inc": 2,
                "min_lrx": 79, "min_lry": 3,
                "max_ulx": _SW - 80, "max_uly": _SH - 4}

    def set_roi(self, x, y, w, h) -> Tuple[int, int, int, int]:
        with self._geom_lock:
            r = self.roi_range()
            w = max(int(w) - int(w) % r["w_inc"], r["w_min"])
            h = max(int(h) - int(h) % r["h_inc"], r["h_min"])
            x = min(max(int(x) - int(x) % r["x_inc"], 0), _SW - w)
            y = min(max(int(y) - int(y) % r["y_inc"], 0), _SH - h)
            self._roi = (x, y, w, h)
            self._fps = min(self._fps, self._fps_range()[1])
            return self._roi

    def get_roi(self) -> Tuple[int, int, int, int]:
        return self._roi

    def reset_roi(self) -> Tuple[int, int, int, int]:
        return self.set_roi(0, 0, _SW, _SH)

    def binning_range(self) -> Tuple[int, int]:
        return (4, 4)

    def set_binning(self, bx, by) -> Tuple[int, int]:
        with self._geom_lock:
            self._bin = (max(1, min(int(bx), 4)), max(1, min(int(by), 4)))
            return self._bin

    def get_binning(self) -> Tuple[int, int]:
        return self._bin

    # --- acquisition ---
    def start(self, max_throughput: bool = False) -> None:
        self._grabbing = True
        self._t_last = perf_counter()

    def stop(self) -> None:
        self._grabbing = False

    @property
    def is_grabbing(self) -> bool:
        return self._grabbing

    def grab(self, timeout_ms: int = 5000) -> Frame:
        if not self._grabbing:
            raise RuntimeError("acquisition stopped")
        period = 1.0 / max(self._fps, 0.1)
        wait = self._t_last + period - perf_counter()
        if wait > 0:
            sleep(min(wait, timeout_ms / 1000.0))
        self._t_last = perf_counter()

        with self._geom_lock:
            x, y, w, h = self._roi
            bx, by = self._bin
        iw, ih = w // bx, h // by
        t = self._t_last

        # signal: gradient background + moving crosshair, in full-sensor coords
        max_val = (1 << _BIT_DEPTH) - 1
        level = min(1.0, (self._exp_us / 50_000.0) * (1.0 + self._gain / 96.0))
        yy = (np.arange(ih) * by + y)[:, None]
        xx = (np.arange(iw) * bx + x)[None, :]
        img = ((xx + yy) % 512) / 512.0 * 0.5 * level * max_val
        cx = int((_SW / 2) + (_SW / 3) * np.sin(t * 0.9))
        cy = int((_SH / 2) + (_SH / 3) * np.cos(t * 0.7))
        img += (np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 40.0 ** 2))
                * level * max_val * 0.8)
        img += np.random.default_rng(self._hw_count).normal(0, 4 + self._gain / 24, img.shape)
        data = np.clip(img, 0, max_val).astype(np.uint16)

        self._hw_count += 1
        return Frame(data=data, meta=FrameMeta(
            t_sw=perf_counter(), hw_count=self._hw_count,
            hw_timestamp_ns=t * 1e9))
