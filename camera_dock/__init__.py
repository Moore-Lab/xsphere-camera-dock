"""xsphere-camera-dock: standardized camera control + eval web server.

The standardized driver interface lives in ``camera_dock.base`` (CameraDriver,
ControlSpec, ControlValue, Frame). Concrete drivers live in
``camera_dock.drivers`` and are built via ``camera_dock.drivers.make_camera``
(lazy imports — a broken driver never poisons this package import). The eval
web server is ``camera_dock.webapp``; the parent DAQ mounts it.
"""

from .base import CameraDriver, CameraBase, ControlSpec, ControlValue, Frame, FrameMeta
from .engine import AcquisitionEngine
from .recorder import HybridRecorder

__all__ = ["CameraDriver", "CameraBase", "ControlSpec", "ControlValue", "Frame",
           "FrameMeta", "AcquisitionEngine", "HybridRecorder"]
