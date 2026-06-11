"""xsphere-camera-dock: combine and control multiple cameras.

The shared interface every camera module is expected to implement lives in
``camera_dock.base``. Concrete drivers ship in the per-camera submodules
(``basler-acA1440``, ``zelux-cs165mu``).
"""

from .base import CameraBase

__all__ = ["CameraBase"]
