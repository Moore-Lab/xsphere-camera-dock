"""Threaded acquisition engine — shared by the test GUIs, the dock, and the DAQ.

Decouples frame *acquisition* from *display* and *recording*. A single producer
thread pulls every frame from a :class:`~camera_dock.base.CameraBase` at the
camera's full (no-drop) rate and distributes each frame to:

* a **latest-frame slot** that the UI samples at its own, slower, preview rate —
  so slow rendering never throttles capture; and
* an optional **sink** (e.g. a :class:`~camera_dock.recorder.HybridRecorder`)
  that receives *every* frame for recording at the full data rate.

This is the piece that makes "record at the camera's maximum data rate while the
live view runs at a comfortable ~30 fps" possible. It depends only on the
``CameraBase`` surface, so it works for any camera and is reused unchanged by the
dock and the DAQ.
"""

from __future__ import annotations

import threading
from time import perf_counter
from typing import Callable, Optional, Tuple

import numpy as np

# A sink receives (frame, frame_index, timestamp_s). It must be cheap and
# thread-safe — it runs on the acquisition thread and must not block it.
Sink = Callable[[np.ndarray, int, float], None]


class AcquisitionEngine:
    """Runs a camera's acquisition on a background thread.

    Parameters
    ----------
    camera:
        Any object satisfying :class:`~camera_dock.base.CameraBase`.
    """

    def __init__(self, camera) -> None:
        self._cam = camera
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._latest: Optional[np.ndarray] = None
        self._frame_index = 0          # total frames acquired since start()
        self._sink: Optional[Sink] = None

        self._acq_fps = 0.0
        self._errors = 0

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Begin no-drop acquisition on a background thread."""
        if self._running:
            return
        self._cam.start(max_throughput=True)
        self._running = True
        self._frame_index = 0
        self._thread = threading.Thread(target=self._loop, name="acquisition", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop acquisition and join the background thread.

        The camera is stopped *before* the join (mirroring the tested reference
        GUIs' disarm-first discipline): stopping aborts a grab blocked inside
        the SDK, so the thread exits within one poll chunk instead of waiting
        out a full frame period — without this, a slow frame rate makes the
        join time out and the thread outlive the engine.
        """
        self._running = False
        try:
            self._cam.stop()
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _loop(self) -> None:
        window_n, window_t0 = 0, perf_counter()
        while self._running:
            try:
                frame = self._cam.grab()
            except Exception:
                # A grab can fail transiently (timeout) or during shutdown.
                if not self._running:
                    break
                self._errors += 1
                continue

            t = perf_counter()
            with self._lock:
                self._latest = frame
                self._frame_index += 1
                index = self._frame_index
                sink = self._sink

            # Deliver to the recorder (if any) OUTSIDE the lock. The sink must be
            # cheap (enqueue/append) so it never throttles acquisition.
            if sink is not None:
                sink(frame, index, t)

            window_n += 1
            dt = t - window_t0
            if dt >= 0.5:
                self._acq_fps = window_n / dt
                window_n, window_t0 = 0, t

    # --- consumer-facing ---------------------------------------------------
    def latest(self) -> Tuple[Optional[np.ndarray], int]:
        """Return the most recent ``(frame, frame_index)`` for preview.

        ``frame`` is ``None`` until the first frame arrives. Frames are fresh
        copies from the driver, so the returned array is safe to read without
        further locking.
        """
        with self._lock:
            return self._latest, self._frame_index

    @property
    def acquisition_fps(self) -> float:
        """Measured true acquisition rate (frames actually pulled per second)."""
        return self._acq_fps

    @property
    def is_running(self) -> bool:
        return self._running

    def set_sink(self, sink: Optional[Sink]) -> None:
        """Attach (or clear with ``None``) the per-frame recording sink."""
        with self._lock:
            self._sink = sink
