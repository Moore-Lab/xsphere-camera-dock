"""Camera settings presets — capture / apply / save / load, camera-agnostic.

Lets the dock reproduce a camera setup across restarts: capture the current
exposure / gain / frame rate / ROI / binning, save it to a JSON file, and apply it
later. Built only on the :class:`~camera_dock.base.CameraBase` surface, so it works
for any camera and is reused by the web app (and the DAQ).

Presets are stored as ``presets/<camera>__<name>.json`` under the working directory.
A preset named ``default`` is auto-applied when a camera session starts, so the rig
comes up configured.
"""

from __future__ import annotations

import json
import os
from typing import List


def capture(camera, *, has_roi: bool) -> dict:
    """Snapshot the current settings of ``camera`` into a plain dict."""
    s = {
        "exposure": float(camera.get_exposure()),
        "gain": float(camera.get_gain()),
        "fps": float(camera.get_frame_rate()),
    }
    if has_roi:
        try:
            s["roi"] = list(camera.get_roi())
        except Exception:
            pass
    try:
        s["binning"] = list(camera.get_binning())
    except Exception:
        pass
    return s


def apply(camera, settings: dict, *, set_roi: bool = True) -> None:
    """Apply a settings dict to ``camera`` (best-effort; unsupported keys ignored).

    ``set_roi=False`` skips the ROI (the caller may need to stop acquisition first;
    see the web app's region-change path). Binning is applied before ROI.
    """
    if "exposure" in settings:
        try:
            camera.set_exposure(float(settings["exposure"]))
        except Exception:
            pass
    if "gain" in settings:
        try:
            camera.set_gain(float(settings["gain"]))
        except Exception:
            pass
    if "fps" in settings:
        try:
            camera.set_frame_rate(float(settings["fps"]))
        except Exception:
            pass
    if settings.get("binning"):
        try:
            camera.set_binning(int(settings["binning"][0]), int(settings["binning"][1]))
        except Exception:
            pass
    if set_roi and settings.get("roi"):
        try:
            x, y, w, h = (int(v) for v in settings["roi"])
            camera.set_roi(x, y, w, h)
        except Exception:
            pass


def _dir() -> str:
    d = os.path.join(os.getcwd(), "presets")
    os.makedirs(d, exist_ok=True)
    return d


def _safe(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in "-_") or "preset"


def _path(camera_name: str, preset: str) -> str:
    return os.path.join(_dir(), f"{_safe(camera_name)}__{_safe(preset)}.json")


def save(camera_name: str, preset: str, settings: dict) -> str:
    path = _path(camera_name, preset)
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)
    return path


def load(camera_name: str, preset: str) -> dict:
    with open(_path(camera_name, preset)) as f:
        return json.load(f)


def exists(camera_name: str, preset: str) -> bool:
    return os.path.isfile(_path(camera_name, preset))


def list_presets(camera_name: str) -> List[str]:
    prefix = f"{_safe(camera_name)}__"
    try:
        names = os.listdir(_dir())
    except OSError:
        return []
    return sorted(f[len(prefix):-5] for f in names
                  if f.startswith(prefix) and f.endswith(".json"))
