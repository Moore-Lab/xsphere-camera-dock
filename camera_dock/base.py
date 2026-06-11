"""Shared camera interface expected by the dock.

Each per-camera submodule provides a driver that satisfies this protocol. The dock
depends only on this surface, never on a specific camera SDK, so cameras can be added
or swapped without touching the integration layer.

This is a structural ``Protocol`` (duck-typed): camera drivers do **not** need to
import or subclass anything from the dock, which keeps the dependency arrow pointing
one way (dock -> cameras) and avoids a circular submodule dependency.
"""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class CameraBase(Protocol):
    """Minimal interface a camera driver must expose to be driven by the dock."""

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def set_exposure(self, microseconds: float) -> None: ...

    def set_frame_rate(self, fps: float) -> None: ...

    def grab(self): ...

    def frames(self) -> Iterator: ...
