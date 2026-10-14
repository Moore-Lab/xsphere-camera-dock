"""Driver/API for the Hayear HY-1136 camera (UVC / OpenCV).

The connected Hayear (HY-1136) is a **UVC webcam-class colour camera**, not a
ToupTek-OEM / hycam-SDK device: the hycam SDK enumerates 0 devices, but the camera
streams as a standard DirectShow/UVC device. So this driver uses ``cv2.VideoCapture``
rather than the hycam SDK.

Characteristics (measured on the connected unit):
- Colour (BGR), modes 640×480 / 1280×720 / 1920×1080.
- **No manual exposure / gain / frame-rate / ROI control** over UVC (auto-only;
  DSHOW and MSMF both refuse). Those are reported as unsupported so the dock gates
  the corresponding controls off.

It satisfies the dock's :class:`~camera_dock.base.CameraBase` surface, so it inherits
the shared engine, recorder, streaming, snapshot, and timestamp. Frames are converted
to mono ``uint8`` to match the (currently mono) dock pipeline.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


class HayearCamera:
    """UVC driver for the Hayear HY-1136 via OpenCV.

    Parameters
    ----------
    index:      DirectShow device index (HY-1136 enumerates as index 0).
    resolution: requested (width, height); snapped to a supported mode by the camera.
    backend:    OpenCV capture backend (DirectShow by default).
    """

    def __init__(self, index: int = 0, resolution: Tuple[int, int] = (1920, 1080),
                 backend: int = cv2.CAP_DSHOW) -> None:
        self.index = index
        self._req_res = resolution
        self.backend = backend
        self._cap: Optional[cv2.VideoCapture] = None
        self._w = self._h = 0
        self._grabbing = False

    # --- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        cap = cv2.VideoCapture(self.index, self.backend)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(
                f"Could not open Hayear UVC camera at index {self.index}. "
                "Is it connected and not held by another app (HAYEAR / a browser)?")
        w, h = self._req_res
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._cap = cap

    def disconnect(self) -> None:
        if self._cap is not None:
            self._grabbing = False
            self._cap.release()
            self._cap = None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None

    def _require(self) -> cv2.VideoCapture:
        if self._cap is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")
        return self._cap

    # --- device info -------------------------------------------------------
    @property
    def device_info(self) -> dict:
        return {"model": "HY-1136", "serial": str(self.index), "vendor": "Hayear"}

    def sensor_size(self) -> Tuple[int, int]:
        return (int(self._w), int(self._h))

    @property
    def bit_depth(self) -> int:
        return 8

    # --- configuration (all unsupported on this UVC unit) ------------------
    # exposure / gain / frame rate / ROI controls are not available over UVC; they
    # report empty ranges so the dock feature-detects and gates the controls off.
    def set_exposure(self, microseconds: float) -> None:
        return

    def get_exposure(self) -> float:
        return 0.0

    def exposure_range(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    def set_gain(self, value: float) -> None:
        return

    def get_gain(self) -> float:
        return 0.0

    def gain_range(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    def set_frame_rate(self, fps: float) -> None:
        return

    def get_frame_rate(self) -> float:
        return 0.0

    def frame_rate_range(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    def resulting_frame_rate(self) -> float:
        return 0.0

    # --- ROI / binning (unsupported) --------------------------------------
    def roi_range(self) -> dict:
        raise NotImplementedError("HY-1136 (UVC) has no ROI control")

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        raise NotImplementedError("HY-1136 (UVC) has no ROI control")

    def get_roi(self) -> Tuple[int, int, int, int]:
        return (0, 0, int(self._w), int(self._h))

    def reset_roi(self) -> None:
        return

    def binning_range(self) -> Tuple[int, int]:
        return (1, 1)

    def set_binning(self, bx: int, by: int) -> None:
        return

    def get_binning(self) -> Tuple[int, int]:
        return (1, 1)

    # --- acquisition -------------------------------------------------------
    def start(self, max_throughput: bool = False) -> None:
        # cv2.VideoCapture streams continuously once opened; just track state.
        self._require()
        self._grabbing = True

    def stop(self) -> None:
        self._grabbing = False

    @property
    def is_grabbing(self) -> bool:
        return self._grabbing

    def grab(self, timeout_ms: int = 5000) -> np.ndarray:
        """Grab one frame as a 2-D ``uint8`` mono array (BGR is grayscaled)."""
        cap = self._require()
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("Hayear frame grab failed.")
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def frames(self, timeout_ms: int = 5000) -> Iterator[np.ndarray]:
        self.start()
        while self._grabbing:
            yield self.grab(timeout_ms)

    # --- context manager ---------------------------------------------------
    def __enter__(self) -> "HayearCamera":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
