"""Hybrid recorder — capture at the camera's full data rate, encode afterwards.

Recording at a camera's maximum rate is bottlenecked by per-frame *encoding* if
you encode in the acquisition hot path. This recorder removes encoding from that
path entirely:

1. **Capture** (hot path, acquisition thread): each frame is appended to a RAM
   buffer, its metadata (host time, hardware frame counter/timestamp) to
   parallel index-aligned lists. No encode, no disk.
2. **Spill** (long clips only): past a RAM cap, frames stream to a raw file via
   a background writer with a bounded queue; overflow is *counted*, never
   silently lost, and never blocks acquisition.
3. **Encode** (after stop): everything is written to a lossless video, and the
   metadata sidecars (timestamps / hwclock / json — see metadata.py) are written
   next to it. This is minutes of work for long clips — the web app runs it on a
   dedicated non-daemon thread, never in the capture path or a request handler.

Optional limits (``max_frames`` / ``max_seconds``): when reached the recorder
just stops *accepting* frames and raises its ``limit_reached`` flag — no
cross-thread finalization; the UI poll sees the flag and calls stop.
"""

from __future__ import annotations

import os
import queue
import threading
from datetime import datetime, timedelta
from time import perf_counter, time as unix_time
from typing import Callable, Optional

import cv2
import numpy as np

from . import metadata
from .base import Frame

# Convert a native mono frame (uint8/uint16) to a uint8 2-D array for video.
To8Bit = Callable[[np.ndarray], np.ndarray]
# Burn an overlay (e.g. a timestamp) onto a BGR frame, given its wall time.
Stamp = Callable[[np.ndarray, datetime], None]


class HybridRecorder:
    """Buffer frames at full rate, spill past a RAM cap, encode + sidecars on stop."""

    def __init__(
        self,
        ram_cap_bytes: int = 2_000_000_000,
        queue_frames: int = 256,
        clock: Optional[Callable[[], datetime]] = None,
        max_frames: int = 0,             # 0 = unlimited
        max_seconds: float = 0.0,        # 0 = unlimited
    ) -> None:
        self.ram_cap_bytes = int(ram_cap_bytes)
        self.max_frames = int(max_frames)
        self.max_seconds = float(max_seconds)
        self.limit_reached = False
        self._clock = clock

        # Parallel, index-aligned per-frame metadata (RAM + spilled frames).
        self._times: list[float] = []
        self._hw_counts: list[Optional[int]] = []
        self._hw_ts: list[Optional[float]] = []

        self._t0_perf: Optional[float] = None
        self._t0_wall: Optional[datetime] = None
        self._start_unix: Optional[float] = None
        self._ram: list[np.ndarray] = []
        self._ram_bytes = 0

        self._spilling = False
        self._spill_path: Optional[str] = None
        self._spill_file = None
        self._queue: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=queue_frames)
        self._writer_thread: Optional[threading.Thread] = None
        self._spilled = 0
        self._dropped = 0

        self._shape: Optional[tuple] = None
        self._dtype = None
        self._captured = 0
        self._t_first: Optional[float] = None
        self._t_last: Optional[float] = None
        self._closed = False

    @property
    def captured(self) -> int:
        return self._captured

    @property
    def elapsed_s(self) -> float:
        if self._t_first is None:
            return 0.0
        return (self._t_last or self._t_first) - self._t_first

    # --- hot path (acquisition thread) ------------------------------------
    def submit(self, frame: Frame, index: int, t: float) -> None:
        """Accept one frame. Cheap: RAM append, or non-blocking enqueue if spilling."""
        if self._closed or self.limit_reached:
            return
        data, meta = frame.data, frame.meta
        if self._shape is None:
            self._shape, self._dtype = data.shape, data.dtype
            self._t_first = t
            self._t0_perf = t
            self._start_unix = unix_time()
            if self._clock is not None:
                self._t0_wall = self._clock()   # wall time anchored to first frame
        elif data.shape != self._shape:
            return                              # stale-geometry frame — never record
        self._t_last = t
        self._times.append(t)
        self._hw_counts.append(meta.hw_count)
        self._hw_ts.append(meta.hw_timestamp_ns)
        self._captured += 1

        if ((self.max_frames and self._captured >= self.max_frames)
                or (self.max_seconds and t - self._t_first >= self.max_seconds)):
            self.limit_reached = True           # stop accepting; UI poll finalizes

        if not self._spilling and self._ram_bytes < self.ram_cap_bytes:
            self._ram.append(data)
            self._ram_bytes += data.nbytes
            return

        if not self._spilling:
            self._begin_spill()
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            self._dropped += 1

    # --- spill writer ------------------------------------------------------
    def _begin_spill(self) -> None:
        self._spilling = True
        self._spill_path = os.path.join(
            os.environ.get("TEMP", "."), f"_xsphere_spill_{id(self)}.raw")
        self._spill_file = open(self._spill_path, "wb")
        self._writer_thread = threading.Thread(target=self._writer_loop,
                                               name="spill-writer", daemon=True)
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            self._spill_file.write(np.ascontiguousarray(item).tobytes())
            self._spilled += 1

    # --- finalize (dedicated encode thread; never the capture path) --------
    def stop_and_encode(self, path: str, fps: float, to_8bit: To8Bit,
                        stamp: Optional[Stamp] = None,
                        camera: Optional[dict] = None,
                        extra: Optional[dict] = None) -> dict:
        """Stop accepting frames, encode to a lossless video, write sidecars.

        Call only after the engine's sink has been detached (set_sink(None) is a
        barrier — no submit is in flight afterwards).
        """
        self._closed = True
        stop_unix = unix_time()

        if self._spilling:
            self._queue.put(None)
            if self._writer_thread is not None:
                self._writer_thread.join(timeout=30.0)
            if self._spill_file is not None:
                self._spill_file.close()

        capture_seconds = (self._t_last - self._t_first) if self._t_first is not None else 0.0
        capture_fps = (self._captured / capture_seconds) if capture_seconds > 0 else 0.0

        encode_t0 = perf_counter()
        encoded = 0
        ok = False
        if self._shape is not None:
            h, w = self._shape
            writer = _open_writer(path, fps, (w, h))
            if writer is not None:
                try:
                    for data in self._ram:
                        self._write_frame(writer, data, to_8bit, stamp, encoded)
                        encoded += 1
                    encoded += self._encode_spill(writer, to_8bit, stamp, encoded)
                    ok = True
                finally:
                    writer.release()

        if self._spill_path and os.path.exists(self._spill_path):
            try:
                os.remove(self._spill_path)
            except OSError:
                pass

        sidecar = None
        if ok and self._shape is not None:
            h, w = self._shape
            sidecar = metadata.write_sidecars(
                path, t_sw=self._times, hw_counts=self._hw_counts,
                hw_ts_ns=self._hw_ts, camera=camera, width=w, height=h,
                nominal_fps=fps, start_unix=self._start_unix, stop_unix=stop_unix,
                extra={"ram_frames": len(self._ram), "spilled": self._spilled,
                       **(extra or {})})

        return {
            "ok": ok,
            "path": path if ok else None,
            "sidecar": sidecar,
            "captured": self._captured,
            "encoded": encoded,
            "dropped": self._dropped,
            "spilled": self._spilled,
            "limit_reached": self.limit_reached,
            "capture_fps": round(capture_fps, 1),
            "capture_seconds": round(capture_seconds, 3),
            "encode_seconds": round(perf_counter() - encode_t0, 3),
        }

    def _write_frame(self, writer, data, to_8bit: To8Bit, stamp: Optional[Stamp],
                     idx: int) -> None:
        bgr = cv2.cvtColor(to_8bit(data), cv2.COLOR_GRAY2BGR)
        if stamp is not None and self._t0_wall is not None and idx < len(self._times):
            dt = self._t0_wall + timedelta(seconds=self._times[idx] - self._t0_perf)
            stamp(bgr, dt)
        writer.write(bgr)

    def _encode_spill(self, writer, to_8bit: To8Bit, stamp: Optional[Stamp],
                      start_idx: int) -> int:
        if not self._spilling or not self._spill_path or not os.path.exists(self._spill_path):
            return 0
        n = 0
        frame_bytes = int(np.prod(self._shape)) * np.dtype(self._dtype).itemsize
        with open(self._spill_path, "rb") as f:
            while True:
                raw = f.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                data = np.frombuffer(raw, dtype=self._dtype).reshape(self._shape)
                self._write_frame(writer, data, to_8bit, stamp, start_idx + n)
                n += 1
        return n


def _open_writer(path: str, fps: float, size):
    """Open a VideoWriter, preferring lossless FFV1, falling back to MJPG."""
    for codec in ("FFV1", "MJPG"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), max(fps, 1.0), size, True)
        if w.isOpened():
            return w
    return None
