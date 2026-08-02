"""Camera driver registry — token -> driver instance, contractually never-raise.

``make_camera("zelux")`` or ``make_camera("zelux:12345")`` (serial selection).
Any failure — unknown token, paused module, missing SDK, constructor error —
returns an :class:`~camera_dock.base.UnavailableCamera` stub instead of raising:
the parent DAQ panel constructs sessions before its server starts, and one bad
token must not take down the working cameras.

Driver modules import lazily inside ``make_camera`` so a broken driver file can
never poison ``import camera_dock``.
"""

from __future__ import annotations

from ..base import CameraDriver, UnavailableCamera

# token -> (module name under camera_dock.drivers, class name)
_REGISTRY = {
    "zelux": ("zelux", "ZeluxCamera"),
    "sim": ("sim", "SimCamera"),     # hardware-free test fixture
}

# Tokens the old dock knew; development on them is paused during the rebuild.
_PAUSED = {"basler", "hayear", "ids", "dcc"}


def make_camera(token: str) -> CameraDriver:
    """Build a driver for ``token`` ("model" or "model:SERIAL"). Never raises."""
    base, _, serial = str(token).partition(":")
    entry = _REGISTRY.get(base)
    if entry is None:
        if base in _PAUSED:
            reason = (f"camera '{base}' is paused during the zelux-first rebuild "
                      f"(see docs/DESIGN.md)")
        else:
            reason = (f"unknown camera '{base}' — registered: "
                      f"{', '.join(sorted(_REGISTRY))}")
        return UnavailableCamera(token, reason)
    mod_name, cls_name = entry
    try:
        module = __import__(f"camera_dock.drivers.{mod_name}", fromlist=[cls_name])
        cls = getattr(module, cls_name)
        return cls(serial) if serial else cls()
    except Exception as exc:
        return UnavailableCamera(token, f"driver '{base}' failed to load: {exc}")
