"""Shared camera interface expected by the dock.

Each per-camera submodule provides a driver that satisfies this protocol. The dock
(and the shared acquisition engine, recorder, and test GUIs built on top of it)
depend only on this surface, never on a specific camera SDK, so cameras can be
added or swapped without touching the integration layer.

This is a structural ``Protocol`` (duck-typed): camera drivers do **not** need to
import or subclass anything from the dock, which keeps the dependency arrow pointing
one way (dock -> cameras) and avoids a circular submodule dependency.

Pixel data
----------
``grab`` / ``frames`` return a 2-D ``numpy`` array (mono). The dtype is
camera-dependent — ``uint8`` for 8-bit sensors (Basler acA1440), ``uint16`` for
deeper sensors (Zelux CS165MU, 10 significant bits). Use :attr:`bit_depth` to scale
to 8 bits for display.

Throughput
----------
``start(max_throughput=True)`` selects a **no-drop** acquisition mode (every frame is
queued and delivered in order) for recording at the camera's full data rate.
``max_throughput=False`` favours low preview latency (latest-frame-only), where
frames produced faster than they are consumed may be dropped. The shared
:class:`~camera_dock.engine.AcquisitionEngine` always uses ``max_throughput=True``
and handles the preview/record split itself.
"""

from __future__ import annotations

from typing import Iterator, Protocol, Tuple, runtime_checkable


@runtime_checkable
class CameraBase(Protocol):
    """Minimal interface a camera driver must expose to be driven by the dock."""

    # --- lifecycle ---
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    # --- device info ---
    @property
    def device_info(self) -> dict: ...

    def sensor_size(self) -> Tuple[int, int]: ...

    # --- configuration ---
    def set_exposure(self, microseconds: float) -> None: ...

    def get_exposure(self) -> float: ...

    def exposure_range(self) -> Tuple[float, float]: ...

    def set_frame_rate(self, fps: float) -> None: ...

    def get_frame_rate(self) -> float: ...

    def frame_rate_range(self) -> Tuple[float, float]: ...

    def resulting_frame_rate(self) -> float: ...

    # --- gain (optional capability) ---
    # Drivers expose these; a camera without gain returns (0, 0) from
    # ``gain_range`` and makes ``set_gain`` a no-op, so callers can feature-detect
    # by checking the range (max > min) rather than catching exceptions.
    def set_gain(self, value: float) -> None: ...

    def get_gain(self) -> float: ...

    def gain_range(self) -> Tuple[float, float]: ...

    # --- region of interest / binning (optional capability) ---
    # ROI is uniform across cameras as (offset_x, offset_y, width, height). Must be
    # set while acquisition is stopped; the dock stops the engine, sets, restarts.
    # ``roi_range`` returns a dict: w_min/w_max/w_inc/h_min/h_max/h_inc/x_inc/y_inc.
    def roi_range(self) -> dict: ...

    def set_roi(self, x: int, y: int, w: int, h: int) -> None: ...

    def get_roi(self) -> Tuple[int, int, int, int]: ...

    def reset_roi(self) -> None: ...

    def binning_range(self) -> Tuple[int, int]: ...

    def set_binning(self, bx: int, by: int) -> None: ...

    def get_binning(self) -> Tuple[int, int]: ...

    # --- acquisition ---
    def start(self, max_throughput: bool = False) -> None: ...

    def stop(self) -> None: ...

    def grab(self): ...

    def frames(self) -> Iterator: ...
