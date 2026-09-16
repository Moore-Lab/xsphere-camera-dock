"""Video discovery and frame access for droplab.

Lists recordings (with an "in progress" heuristic tuned to the dock's
HybridRecorder), probes properties (preferring the metadata sidecars the dock
writes over the container's nominal fps), and serves random-access frames
through a small per-video reader cache so scrubbing in the browser stays fast.

In-progress heuristic: the dock recorder only creates the video file at
stop/encode time and writes the .json sidecar AFTER encoding finishes. So a
dock recording without its sidecar is still encoding. Files from other sources
(no sidecar ever) fall back to "modified in the last few seconds".

Reader lifecycle: cv2.VideoCapture is not thread-safe and holds a Windows file
lock, so ReaderCache hands out readers only through lease() — a context
manager that refcounts use. Eviction (LRU over the open cap, plus an idle TTL
so recordings can be deleted) only ever closes readers nobody is leasing.
Long jobs (detection, sum views) should open their own private VideoReader
instead of hogging a cached one.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from typing import Optional

import cv2
import numpy as np

from .paths import RECORDINGS_DIR, VIDEO_EXTS, video_basename

# A sidecar-less file whose mtime is newer than this is presumed still writing.
IN_PROGRESS_MTIME_S = 5.0


# ── discovery ───────────────────────────────────────────────────────────────

def sidecar_meta(video_path: str) -> Optional[dict]:
    """Load the dock's <stem>.json sidecar, or None."""
    p = os.path.splitext(video_path)[0] + ".json"
    if not os.path.isfile(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def sidecar_binning(meta: Optional[dict]) -> tuple[float, float]:
    """(bx, by) from the sidecar, defaulting to 1 for anything unusable."""
    bx = by = 1.0
    binning = (meta or {}).get("binning")
    if isinstance(binning, (list, tuple)) and binning:
        try:
            bx = float(binning[0]) or 1.0
        except (TypeError, ValueError):
            bx = 1.0
        try:
            by = float(binning[1]) or 1.0 if len(binning) > 1 else bx
        except (TypeError, ValueError):
            by = bx
    return bx, by


def sidecar_image_binning(meta: Optional[dict]) -> tuple[float, float]:
    """(horizontal, vertical) binning of the RECORDED image's own axes.

    ``binning`` in the sidecar is (bx, by) in SENSOR axes. The dock orients
    frames after binning (camera_dock/orientation.py), and a quarter turn puts
    sensor y along the image's horizontal axis — so the factors swap. A mirror or
    a half turn keeps the axes. Sidecars from before orientation existed carry no
    ``orientation`` and read as upright.
    """
    bx, by = sidecar_binning(meta)
    o = (meta or {}).get("orientation")
    try:
        rotate = int(o.get("rotate", 0)) % 360 if isinstance(o, dict) else 0
    except (TypeError, ValueError):
        rotate = 0
    return (by, bx) if rotate in (90, 270) else (bx, by)


def sidecar_pixels_per_um(meta: Optional[dict]) -> Optional[float]:
    """Image pixels per micrometre (**horizontal**), from the dock's sidecar.

    The dock stores ``um_per_px`` as micrometres per *sensor* pixel, deliberately,
    so the number survives binning and ROI changes. droplab (and the upstream
    count-droplets scripts) measure in *image* pixels, and binning maps ``b_h``
    sensor pixels onto one image pixel horizontally::

        um per image px = um_per_px * b_h
        pixels_per_um   = 1 / (um_per_px * b_h)

    ``b_h`` is the binning along the IMAGE's horizontal axis
    (``sidecar_image_binning``): bx for an upright recording, by after the dock
    turned the camera a quarter turn.

    ROI is a crop, not a rescale, so it does not enter the conversion.

    **This is the horizontal scale**, and every consumer downstream applies it
    isotropically — one ``mm_per_px`` for both ROI axes, one ``pixels_per_m`` for
    both ``vx`` and ``vy``. It is therefore correct in both axes only when
    ``bx == by``. Callers must check ``probe()['anisotropic_binning']`` and warn;
    returning None instead would be worse, since the caller would then fall back
    to a *different rig's* constant — wrong in both axes rather than one.

    Returns None when the recording carries no calibration, so the caller falls
    back to the upstream default *and can say that it did* — a wrong scale that
    looks authoritative is worse than an honest one.
    """
    if not meta:
        return None
    try:
        um_per_px = float(meta.get("um_per_px") or 0)
    except (TypeError, ValueError):
        return None
    if um_per_px <= 0:
        return None
    b_h, _ = sidecar_image_binning(meta)
    scale = um_per_px * b_h
    return (1.0 / scale) if scale > 0 else None


def list_videos(dirs: list[str]) -> list[dict]:
    """One dict per video across *dirs*, newest first."""
    out = []
    now = time.time()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            if not name.lower().endswith(VIDEO_EXTS):
                continue
            path = os.path.join(d, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            meta = sidecar_meta(path)
            in_progress = (meta is None
                           and now - st.st_mtime < IN_PROGRESS_MTIME_S)
            out.append({
                "name": name,
                "basename": video_basename(path),
                "dir": d,
                "path": path,
                "size_bytes": st.st_size,
                "mtime": st.st_mtime,
                "has_sidecar": meta is not None,
                "in_progress": in_progress,
                "sidecar_fps": _sidecar_fps(meta),
                "n_frames_sidecar": (meta or {}).get("n_frames"),
            })
    out.sort(key=lambda v: v["mtime"], reverse=True)
    return out


def _sidecar_fps(meta: Optional[dict]) -> Optional[float]:
    """True capture fps from the sidecar.

    Preference: sw_measured_fps > measured_fps > n_frames/duration_s >
    hw_measured_fps. The software-clock rate is the most trustworthy across
    cameras — e.g. Zelux sidecars report an hw_measured_fps in a different
    tick unit (3.05 "fps" for a real 30 fps clip) while sw_measured_fps is
    correct.
    """
    if not meta:
        return None
    candidates = [meta.get("sw_measured_fps"), meta.get("measured_fps")]
    n, dur = meta.get("n_frames"), meta.get("duration_s")
    if n and dur:
        try:
            candidates.append(float(n) / float(dur))
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    candidates.append(meta.get("hw_measured_fps"))
    for fps in candidates:
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            continue
        if fps > 0:
            return fps
    return None


def resolve_video(name: str, dirs: list[str]) -> Optional[str]:
    """Map a bare filename back to a full path inside the allowed dirs."""
    base = os.path.basename(name)          # refuse path traversal
    if not base.lower().endswith(VIDEO_EXTS):
        return None
    for d in dirs:
        p = os.path.join(d, base)
        if os.path.isfile(p):
            return p
    return None


# ── frame access ────────────────────────────────────────────────────────────

class VideoReader:
    """Random access into one video with cheap sequential stepping.

    OpenCV re-seek is expensive; like the dock's viewers we only set
    CAP_PROP_POS_FRAMES when the request is not the next sequential frame.
    A lock serializes access — cv2.VideoCapture is not thread-safe and FastAPI
    handlers run on a thread pool.
    """

    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        self.lock = threading.Lock()
        self.last_idx = -2
        self.last_used = time.time()
        self.busy = 0                      # lease count; owned by ReaderCache
        try:
            self._size = os.path.getsize(path)
        except OSError:
            self._size = -1

        self.container_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0) or None
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def _reopen_if_grown_locked(self) -> None:
        """OpenCV caches container metadata at open; a file still being
        encoded by the dock grows underneath us. Reopen when the size moved."""
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return
        if size == self._size:
            return
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            return                          # keep the old handle; try later
        self.cap.release()
        self.cap = cap
        self._size = size
        self.last_idx = -2
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self, idx: int) -> Optional[np.ndarray]:
        """Return BGR frame *idx* or None."""
        with self.lock:
            self.last_used = time.time()
            idx = max(0, min(idx, max(self.total_frames - 1, 0)))
            if idx != self.last_idx + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok:
                # A recording may have grown or the seek failed; retry once cold.
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = self.cap.read()
            self.last_idx = idx if ok else -2
            return frame if ok else None

    def refresh_length(self) -> int:
        """Re-read frame count, reopening if the file grew since open."""
        with self.lock:
            self._reopen_if_grown_locked()
            n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if n:
                self.total_frames = n
            return self.total_frames

    def close(self) -> None:
        with self.lock:
            self.cap.release()


class ReaderCache:
    """Bounded set of open VideoReaders, handed out via refcounted leases."""

    def __init__(self, max_open: int = 4, idle_ttl_s: float = 120.0):
        self.max_open = max_open
        self.idle_ttl_s = idle_ttl_s
        self._readers: dict[str, VideoReader] = {}
        self._lock = threading.Lock()
        # Janitor: release idle Windows file locks even when no requests
        # arrive to trigger a sweep (so recordings stay deletable).
        t = threading.Thread(target=self._janitor, name="reader-janitor",
                             daemon=True)
        t.start()

    def _janitor(self) -> None:
        while True:
            time.sleep(60.0)
            with self._lock:
                self._sweep_locked()

    @contextlib.contextmanager
    def lease(self, path: str):
        """Yield a VideoReader; it will not be evicted while leased."""
        with self._lock:
            r = self._readers.get(path)
            if r is None:
                r = VideoReader(path)       # RuntimeError if unopenable
                self._readers[path] = r
            r.busy += 1
            r.last_used = time.time()
            self._sweep_locked()
        try:
            yield r
        finally:
            with self._lock:
                r.busy -= 1
                r.last_used = time.time()

    def _sweep_locked(self) -> None:
        """Close idle readers past the TTL, then LRU-evict over the cap.
        Only readers with no active lease are ever closed."""
        now = time.time()
        doomed = [p for p, r in self._readers.items()
                  if r.busy == 0 and now - r.last_used > self.idle_ttl_s]
        idle = sorted((p for p, r in self._readers.items()
                       if r.busy == 0 and p not in doomed),
                      key=lambda p: self._readers[p].last_used)
        overflow = len(self._readers) - len(doomed) - self.max_open
        doomed += idle[:max(overflow, 0)]
        for p in doomed:
            self._readers.pop(p).close()

    def drop(self, path: str) -> None:
        with self._lock:
            r = self._readers.get(path)
            if r is not None and r.busy == 0:
                self._readers.pop(path).close()

    def close_all(self) -> None:
        with self._lock:
            readers, self._readers = list(self._readers.values()), {}
        for r in readers:
            r.close()


def probe(path: str) -> dict:
    """One-shot properties for a video (opens and closes its own capture)."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open {path}")
        meta = sidecar_meta(path)
        container_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or None
        fps = _sidecar_fps(meta) or container_fps or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return {
            "path": path,
            "name": os.path.basename(path),
            "basename": video_basename(path),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "total_frames": total,
            "container_fps": container_fps,
            "sidecar_fps": _sidecar_fps(meta),
            "fps": fps,                       # best available
            "duration_s": (total / fps) if fps else None,
            "has_sidecar": meta is not None,
            # Calibration as recorded by the dock — the scale this rig actually
            # has, rather than the upstream constant from a different one.
            "pixels_per_um": sidecar_pixels_per_um(meta),
            "calibration": (meta or {}).get("calibration"),
            "um_per_px": (meta or {}).get("um_per_px"),
            "binning": (meta or {}).get("binning"),
            "image_binning": list(sidecar_image_binning(meta)),   # [horizontal, vertical]
            # pixels_per_um is the horizontal scale and is applied isotropically
            # downstream — so it is only valid in y when the axes bin equally.
            "anisotropic_binning": sidecar_binning(meta)[0] != sidecar_binning(meta)[1],
        }
    finally:
        cap.release()
