"""Threaded acquisition engine — shared by the dock web app and the DAQ.

Decouples frame *acquisition* from *display* and *recording*. A single producer
thread pulls every :class:`~camera_dock.base.Frame` from a driver at the camera's
full rate and distributes each frame to:

* a **latest-frame slot** the UI samples at its own, slower, preview rate — so
  slow rendering never throttles capture; and
* an optional **sink** (e.g. the recorder) that receives *every* frame.

Concurrency contracts (docs/DESIGN.md §2.3):

* ``_frame_index`` is monotonic across restarts — index-based freshness guards
  (auto-exposure, timelapse stale detection) survive region changes.
* ``stop()`` clears the latest slot, so a stale-geometry frame is never served
  after a restart; it also stops the camera BEFORE joining (a disarm aborts a
  grab blocked inside the SDK), and reports if the thread failed to join —
  callers must not touch geometry while a producer might still be alive.
* The sink is invoked under a dedicated ``_sink_lock`` and ``set_sink`` takes the
  same lock, so ``set_sink(None)`` returning is a barrier: no ``submit`` is in
  flight afterwards and none will start.
* Frames are ORIENTED here, right after the grab (see orientation.py), so the
  latest slot and the sink only ever see oriented frames — stream, recording,
  timelapse and snapshots cannot disagree. ``set_orientation`` clears the latest
  slot and bumps a generation; a frame transformed under the previous
  orientation is dropped rather than published.
"""

from __future__ import annotations

import threading
from time import perf_counter, sleep
from typing import Callable, Optional, Tuple

from .base import Frame
from .orientation import Orientation

# A sink receives (frame, frame_index, t_sw). It runs on the acquisition thread
# and must be cheap (append/enqueue) — it must never block.
Sink = Callable[[Frame, int, float], None]


class AcquisitionEngine:
    """Runs a driver's acquisition on a background thread."""

    def __init__(self, camera, orientation: Optional[Orientation] = None) -> None:
        self._cam = camera
        self._orientation = orientation or Orientation()
        self._orient_gen = 0           # bumped by set_orientation
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._sink_lock = threading.Lock()

        self._latest: Optional[Frame] = None
        self._frame_index = 0          # monotonic across restarts (never reset)
        self._sink: Optional[Sink] = None

        self._acq_fps = 0.0
        self._errors = 0

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Begin acquisition on a background thread (no-op if running)."""
        if self._running:
            return
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("previous acquisition thread still alive")
        self._cam.start(max_throughput=True)
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="acquisition", daemon=True)
        self._thread.start()

    def stop(self) -> bool:
        """Stop acquisition; return True if the producer thread exited cleanly.

        Camera stopped BEFORE the join: stopping aborts a grab blocked inside
        the SDK so the thread exits promptly. A False return means the thread is
        wedged inside the SDK — callers must not proceed to geometry changes.
        """
        self._running = False
        try:
            self._cam.stop()
        except Exception:
            pass
        joined = True
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            joined = not self._thread.is_alive()
            if joined:
                self._thread = None
        with self._lock:
            self._latest = None       # never serve a stale-geometry frame
        return joined

    def _loop(self) -> None:
        window_n, window_t0 = 0, perf_counter()
        while self._running:
            try:
                frame = self._cam.grab(timeout_ms=2000)
            except Exception:
                if not self._running:
                    break
                self._errors += 1
                sleep(0.05)   # a fast-failing camera must not busy-spin
                continue

            # Orient outside the frame lock (a quarter turn of a full Zelux frame
            # is ~1 ms), then publish only if nobody re-oriented meanwhile.
            with self._lock:
                orient, gen = self._orientation, self._orient_gen
            if not orient.is_identity:
                frame = Frame(orient.apply(frame.data), frame.meta)

            t = frame.meta.t_sw
            with self._lock:
                if gen != self._orient_gen:
                    continue           # transformed under the old orientation
                self._latest = frame
                self._frame_index += 1
                index = self._frame_index

            # Sink outside the frame lock but under the sink lock — set_sink(None)
            # is then a barrier (no submit in flight after it returns).
            with self._sink_lock:
                # Re-check the orientation generation here too: set_orientation
                # can land between publishing above and this point, and a
                # recorder attached right after it would latch this stale shape
                # and then silently drop every correctly oriented frame. (An
                # unlocked int read is enough: the bump happens-before any
                # record_start that follows the re-orientation.)
                if self._sink is not None and gen == self._orient_gen:
                    self._sink(frame, index, t)

            window_n += 1
            dt = t - window_t0
            if dt >= 0.5:
                self._acq_fps = window_n / dt
                window_n, window_t0 = 0, t

    # --- consumer-facing ---------------------------------------------------
    def latest(self) -> Tuple[Optional[Frame], int]:
        """Most recent ``(frame, frame_index)``; frame is None until one arrives."""
        with self._lock:
            return self._latest, self._frame_index

    @property
    def acquisition_fps(self) -> float:
        """Measured true acquisition rate (from host-side frame deltas — the
        SDK's own measurement returns 0.0 on some units)."""
        return self._acq_fps

    @property
    def error_count(self) -> int:
        return self._errors

    @property
    def is_running(self) -> bool:
        return self._running

    def set_sink(self, sink: Optional[Sink]) -> None:
        """Attach (or clear with None) the per-frame sink. Clearing is a barrier."""
        with self._sink_lock:
            self._sink = sink

    @property
    def orientation(self) -> Orientation:
        return self._orientation

    def set_orientation(self, orientation: Orientation) -> None:
        """Orient subsequent frames. Clears the latest slot so a frame from the
        previous orientation is never served. Callers must not change it while a
        recording sink is attached — the recorder latches the frame shape."""
        with self._lock:
            self._orientation = orientation
            self._orient_gen += 1
            self._latest = None
