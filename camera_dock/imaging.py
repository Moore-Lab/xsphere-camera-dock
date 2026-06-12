"""Camera-agnostic imaging helpers — shared by the GUIs, the dock, and the DAQ.

These operate on plain ``numpy`` frames (and, for auto-exposure, a ``CameraBase``),
so the same code backs the test GUIs today and the dock/DAQ UI tomorrow. Covers:
histogram + saturation, software one-shot auto-exposure, and multi-format snapshots.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Optional

import cv2
import numpy as np

try:
    from zoneinfo import ZoneInfo
    _EASTERN: Optional["ZoneInfo"] = ZoneInfo("America/New_York")  # New Haven, CT (DST-aware)
except Exception:
    _EASTERN = None  # tz database unavailable (see camera_dock requirements: tzdata)

# Snapshot formats: tiff/png are lossless and preserve bit depth; npy is the raw
# native array (exact sensor values, no image-format quantisation).
SNAPSHOT_FORMATS = ("tiff", "png", "npy")


def max_value_for(bit_depth: int) -> int:
    """Largest pixel value for a given bit depth (e.g. 255 for 8-bit, 1023 for 10)."""
    return (1 << int(bit_depth)) - 1


def saturation_fraction(frame: np.ndarray, max_value: int) -> float:
    """Fraction of pixels at/above the sensor max — i.e. clipped/saturated [0..1]."""
    if frame.size == 0:
        return 0.0
    return float(np.count_nonzero(frame >= max_value)) / frame.size


def histogram_overlay(bgr: np.ndarray, frame8: np.ndarray, *, height: int = 90,
                      bins: int = 128, color=(0, 255, 0)) -> np.ndarray:
    """Draw a luminance histogram (of ``frame8``) into the bottom-left of ``bgr``.

    Mutates and returns ``bgr``. Log-scaled bars so faint detail is visible.
    """
    h, w = bgr.shape[:2]
    hist = cv2.calcHist([frame8], [0], None, [bins], [0, 256]).flatten()
    hist = np.log1p(hist)
    peak = hist.max()
    if peak > 0:
        hist = hist / peak

    pad = 10
    bw = max(1, (w // 4) // bins)
    panel_w = bw * bins
    x0, y0 = pad, h - pad

    backdrop = bgr.copy()
    cv2.rectangle(backdrop, (x0 - 4, y0 - height - 4), (x0 + panel_w + 4, y0 + 4), (0, 0, 0), -1)
    cv2.addWeighted(backdrop, 0.45, bgr, 0.55, 0, bgr)
    for i, v in enumerate(hist):
        bh = int(v * height)
        if bh:
            cv2.rectangle(bgr, (x0 + i * bw, y0), (x0 + i * bw + bw - 1, y0 - bh), color, -1)
    return bgr


def auto_expose(camera, grab_fresh: Callable[[], Optional[np.ndarray]], *,
                target: float = 0.9, percentile: float = 99.0,
                max_iters: int = 8, tol: float = 0.04) -> float:
    """Software one-shot auto-exposure.

    Iteratively scales exposure so the ``percentile`` brightness reaches ``target``
    times the sensor max. ``grab_fresh`` must return a *fresh* native frame captured
    at the current exposure (the caller is responsible for freshness — e.g. waiting
    for the acquisition engine's frame index to advance). Returns the final exposure
    in microseconds.
    """
    max_value = max_value_for(int(getattr(camera, "bit_depth", 8)))
    lo, hi = camera.exposure_range()
    exp = camera.get_exposure()
    for _ in range(max_iters):
        frame = grab_fresh()
        if frame is None:
            break
        level = float(np.percentile(frame, percentile)) / max_value if max_value else 0.0
        if level <= 1e-4:
            exp = min(exp * 4.0, hi)            # pitch black: open up fast
        else:
            factor = max(0.25, min(4.0, target / level))   # damped multiplicative step
            exp = min(max(exp * factor, lo), hi)
        camera.set_exposure(exp)
        if abs(level - target) <= tol:
            break
    return camera.get_exposure()


def eastern_now() -> datetime:
    """Current wall-clock time in New Haven, CT (America/New_York, DST-aware).

    Falls back to the system local time zone if the IANA tz database is missing
    (install ``tzdata`` on Windows — see the dock requirements).
    """
    if _EASTERN is not None:
        return datetime.now(_EASTERN)
    return datetime.now().astimezone()


def format_timestamp(dt: datetime) -> str:
    """e.g. ``2026-06-12 14:30:45.123 EDT`` (millisecond precision + tz label)."""
    tz = dt.tzname() or ""
    return (dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
            + (f" {tz}" if tz else ""))


def draw_timestamp(bgr: np.ndarray, dt: datetime, *, scale: float | None = None) -> np.ndarray:
    """Burn a New-Haven timestamp into the bottom-right of a BGR image (mutates it).

    Used both for the live preview and, at encode time, for each recorded frame —
    so the date/time is permanently in the movie. Font scales with frame width.
    """
    text = format_timestamp(dt)
    h, w = bgr.shape[:2]
    fs = scale if scale is not None else max(0.4, w / 1600.0)
    th = max(1, int(round(fs * 2)))
    (tw, tht), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = 6
    x = max(0, w - tw - 2 * pad)
    y = h - pad
    backdrop = bgr.copy()
    cv2.rectangle(backdrop, (x, y - tht - base - pad), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(backdrop, 0.5, bgr, 0.5, 0, bgr)
    cv2.putText(bgr, text, (x + pad, y - base), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, 255, 0), th, cv2.LINE_AA)
    return bgr


def save_snapshot(frame: np.ndarray, path_base: str, fmt: str = "tiff") -> str:
    """Save a native frame as ``path_base.<ext>``; returns the full path.

    tiff/png keep full bit depth (lossless); npy stores the exact raw array.
    """
    fmt = fmt.lower()
    if fmt == "npy":
        path = path_base + ".npy"
        np.save(path, frame)
    elif fmt == "png":
        path = path_base + ".png"
        cv2.imwrite(path, frame)
    else:
        path = path_base + ".tiff"
        cv2.imwrite(path, frame)
    return path
