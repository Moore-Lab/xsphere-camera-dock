"""Hybrid recorder — capture at the camera's full data rate, encode afterwards.

Recording video at a camera's maximum frame rate is bottlenecked by per-frame
*encoding* (FFV1/MJPG on the CPU) if you encode in the acquisition hot path. This
recorder removes encoding from that path entirely:

1. **Capture** (hot path, called from the acquisition thread): each frame is just
   appended to a RAM buffer. No encode, no disk — so acquisition runs at the
   sensor's full rate with no dropped frames.
2. **Spill** (only for long clips): once the RAM buffer passes a size cap, further
   frames stream to a raw file via a background writer thread with a bounded queue.
   The queue depth is the "few seconds of latency" that absorbs disk jitter; if it
   ever overflows (disk can't keep up at all), the overflow is *counted*, not
   silently lost.
3. **Encode** (after you press stop): the buffered + spilled frames are written to
   a lossless video file. This is where the few-seconds delay is spent — off the
   capture path, so it never costs you frame rate.

Camera-agnostic: it stores whatever 2-D ``numpy`` frames the engine feeds it and
uses a caller-supplied ``to_8bit`` to convert for the 8-bit video at encode time
(so 16-bit cameras work too). Built for reuse by the test GUIs, the dock, and the
DAQ.
"""

from __future__ import annotations

import os
import queue
import threading
from datetime import datetime, timedelta
from time import perf_counter
from typing import Callable, Optional

import cv2
import numpy as np

# Convert a native mono frame (uint8 or uint16) to a uint8 2-D array for video.
To8Bit = Callable[[np.ndarray], np.ndarray]
# Burn an overlay (e.g. a timestamp) onto a BGR frame, given that frame's wall time.
Stamp = Callable[[np.ndarray, datetime], None]


class HybridRecorder:
    """Buffer frames at full rate, optionally spilling to disk, encode on stop.

    Parameters
    ----------
    ram_cap_bytes:
        Soft cap on RAM held before spilling subsequent frames to a raw file.
    queue_frames:
        Bounded spill-queue depth (the latency buffer). Overflow is counted.
    """

    def __init__(
        self,
        ram_cap_bytes: int = 2_000_000_000,
        queue_frames: int = 256,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.ram_cap_bytes = int(ram_cap_bytes)
        self._clock = clock          # returns the current wall time (e.g. New Haven)
        self._times: list[float] = []  # per-frame perf timestamps, in capture order
        self._t0_perf: Optional[float] = None
        self._t0_wall: Optional[datetime] = None
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

    # --- hot path (acquisition thread) ------------------------------------
    def submit(self, frame: np.ndarray, index: int, t: float) -> None:
        """Accept one frame. Cheap: RAM append, or non-blocking enqueue if spilling."""
        if self._closed:
            return
        if self._shape is None:
            self._shape, self._dtype = frame.shape, frame.dtype
            self._t_first = t
            self._t0_perf = t
            if self._clock is not None:
                self._t0_wall = self._clock()   # anchor wall time to this first frame
        self._t_last = t
        self._times.append(t)                   # tiny; kept for every frame even if spilled
        self._captured += 1

        if not self._spilling and self._ram_bytes < self.ram_cap_bytes:
            self._ram.append(frame)
            self._ram_bytes += frame.nbytes
            return

        # Over the RAM cap: stream to disk via the writer thread.
        if not self._spilling:
            self._begin_spill()
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._dropped += 1

    # --- spill writer ------------------------------------------------------
    def _begin_spill(self, path: Optional[str] = None) -> None:
        self._spilling = True
        self._spill_path = path or os.path.join(
            os.environ.get("TEMP", "."), f"_xsphere_spill_{id(self)}.raw"
        )
        self._spill_file = open(self._spill_path, "wb")
        self._writer_thread = threading.Thread(target=self._writer_loop, name="spill-writer", daemon=True)
        self._writer_thread.start()

    def _writer_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                break
            self._spill_file.write(np.ascontiguousarray(item).tobytes())
            self._spilled += 1

    # --- finalize ----------------------------------------------------------
    def stop_and_encode(self, path: str, fps: float, to_8bit: To8Bit,
                        stamp: Optional[Stamp] = None) -> dict:
        """Stop accepting frames and encode everything to a lossless video.

        Call only after the engine's sink has been detached (no more ``submit``).
        If ``stamp`` is given, each frame is annotated with its own capture-time
        wall clock (anchored at the first frame) — burning the timestamp into the
        movie. Returns a stats dict. ``path`` should end in ``.avi``.
        """
        self._closed = True

        # Drain and close the spill writer, if any.
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
        if self._shape is not None:
            h, w = self._shape
            writer = _open_writer(path, fps, (w, h))
            if writer is None:
                return self._stats(path, capture_fps, capture_seconds, 0.0, encoded, ok=False)
            try:
                for frame in self._ram:
                    self._write_frame(writer, frame, to_8bit, stamp, encoded)
                    encoded += 1
                encoded += self._encode_spill(writer, to_8bit, stamp, encoded)
            finally:
                writer.release()

        # Clean up the raw spill file.
        if self._spill_path and os.path.exists(self._spill_path):
            try:
                os.remove(self._spill_path)
            except OSError:
                pass

        return self._stats(path, capture_fps, capture_seconds, perf_counter() - encode_t0, encoded, ok=True)

    def _write_frame(self, writer, frame, to_8bit: To8Bit, stamp: Optional[Stamp], idx: int) -> None:
        bgr = cv2.cvtColor(to_8bit(frame), cv2.COLOR_GRAY2BGR)
        if stamp is not None and self._t0_wall is not None and idx < len(self._times):
            dt = self._t0_wall + timedelta(seconds=self._times[idx] - self._t0_perf)
            stamp(bgr, dt)
        writer.write(bgr)

    def _encode_spill(self, writer, to_8bit: To8Bit, stamp: Optional[Stamp], start_idx: int) -> int:
        if not self._spilling or not self._spill_path or not os.path.exists(self._spill_path):
            return 0
        n = 0
        frame_bytes = int(np.prod(self._shape)) * np.dtype(self._dtype).itemsize
        with open(self._spill_path, "rb") as f:
            while True:
                raw = f.read(frame_bytes)
                if len(raw) < frame_bytes:
                    break
                frame = np.frombuffer(raw, dtype=self._dtype).reshape(self._shape)
                self._write_frame(writer, frame, to_8bit, stamp, start_idx + n)
                n += 1
        return n

    def _stats(self, path, capture_fps, capture_seconds, encode_seconds, encoded, ok) -> dict:
        return {
            "ok": ok,
            "path": path if ok else None,
            "captured": self._captured,
            "encoded": encoded,
            "dropped": self._dropped,
            "spilled": self._spilled,
            "capture_fps": round(capture_fps, 1),
            "capture_seconds": round(capture_seconds, 3),
            "encode_seconds": round(encode_seconds, 3),
        }


def _open_writer(path: str, fps: float, size):
    """Open a VideoWriter, preferring lossless FFV1, falling back to MJPG."""
    for codec in ("FFV1", "MJPG"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), max(fps, 1.0), size, True)
        if w.isOpened():
            return w
    return None
