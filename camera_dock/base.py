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

    # --- acquisition ---
    def start(self, max_throughput: bool = False) -> None: ...

    def stop(self) -> None: ...

    def grab(self): ...

    def frames(self) -> Iterator: ...
