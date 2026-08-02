"""Standardized camera interface — the one command set every driver speaks.

The dock (engine, recorder, web app, DAQ) depends only on this surface. A concrete
driver subclasses :class:`CameraDriver` and *registers* the scalar controls its
hardware actually has (exposure, gain, fps, ...); the base class owns normalization
(0..1 across the native range), clamping, int casting, and graceful-unsupported
handling. A camera without gain simply never registers "gain" — setting it returns
``ControlValue(ok=False, supported=False)``; nothing raises, nothing crashes.

Scalar controls go through the generic ``get``/``set`` funnel. Geometry (ROI,
binning) gets dedicated verbs because it changes readout size, invalidates
in-flight frames, and shifts other controls' ranges — the dock orchestrates
stop → apply → re-apply fps/exposure → restart around it (see webapp).

Contracts that matter (from the 2026-08 design review, docs/DESIGN.md):

* ``get``/``set`` NEVER raise — failures come back inside :class:`ControlValue`.
  The web layer converts to HTTP status; driver code stays exception-free.
* Setters clamp against ranges read from hardware AT WRITE TIME (ranges shift with
  ROI/binning); the ``ControlSpec`` ranges in :meth:`capabilities` are UI hints,
  refreshed on every call.
* ``set`` reads the applied value BACK from hardware — cameras snap silently.
* Frames are :class:`Frame` objects: a numpy array copied out of the SDK plus
  per-frame metadata (host time, hardware frame counter, hardware timestamp).
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple


# --------------------------------------------------------------------------- #
# frames
# --------------------------------------------------------------------------- #

@dataclass
class FrameMeta:
    """Per-frame metadata captured at the poll site.

    ``hw_count``/``hw_timestamp_ns`` come from the camera when it provides them
    (the Zelux carries both on every frame; ``hw_timestamp_ns`` may still be
    ``None`` on units whose timestamp clock reads 0). ``hw_count`` re-baselines
    at every arm — per-recording drop detection is valid because geometry changes
    are blocked while recording.
    """
    t_sw: float                          # host perf_counter at dequeue
    hw_count: Optional[int] = None       # hardware frame counter
    hw_timestamp_ns: Optional[float] = None  # device timestamp (relative; deltas only)


@dataclass
class Frame:
    data: "np.ndarray"                   # 2-D mono; uint8 or uint16 (low-bit-packed)
    meta: FrameMeta


# --------------------------------------------------------------------------- #
# scalar-control model
# --------------------------------------------------------------------------- #

Getter = Callable[[], float]
Setter = Callable[[float], None]
RangeGetter = Callable[[], Tuple[float, float]]


@dataclass
class ControlSpec:
    """What a control *is*. Registered once by the driver; ranges refresh on read."""
    name: str
    units: str                           # native units ("us", "index", "fps", "ADU")
    get_range: RangeGetter               # fresh (lo, hi) from hardware
    getter: Getter
    setter: Optional[Setter] = None      # None -> read-only
    step: float = 0.0                    # increment hint (0 = continuous)
    kind: str = "float"                  # "int" | "float" — int casts before write
    scale: str = "linear"                # "linear" | "log" — normalized mapping
    display: dict = field(default_factory=dict)

    # --- normalization (guards per DESIGN 2.1) ---
    def to_native(self, frac: float, lo: float, hi: float) -> float:
        frac = min(max(float(frac), 0.0), 1.0)
        if hi <= lo:
            return lo
        if self.scale == "log" and lo > 0:
            return lo * math.exp(frac * math.log(hi / lo))
        return lo + frac * (hi - lo)

    def to_norm(self, value: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return 0.0
        value = min(max(float(value), lo), hi)
        if self.scale == "log" and lo > 0:
            return math.log(value / lo) / math.log(hi / lo)
        return (value - lo) / (hi - lo)

    def describe(self) -> dict:
        """Capability-document entry (ranges read fresh; never raises)."""
        try:
            lo, hi = self.get_range()
        except Exception:
            lo, hi = 0.0, 0.0
        return {
            "name": self.name, "units": self.units, "lo": lo, "hi": hi,
            "step": self.step, "kind": self.kind, "scale": self.scale,
            "display": self.display, "read_only": self.setter is None,
        }


@dataclass
class ControlValue:
    """Result of any funnel get/set — the graceful-failure envelope."""
    ok: bool
    supported: bool
    value: Optional[float] = None        # applied native value (read back)
    normalized: Optional[float] = None
    error: Optional[str] = None
    warning: Optional[str] = None
    side_effects: dict = field(default_factory=dict)   # {name: ControlValue.as_dict()}

    def as_dict(self) -> dict:
        d = {"ok": self.ok, "supported": self.supported, "value": self.value,
             "normalized": self.normalized}
        if self.error:
            d["error"] = self.error
        if self.warning:
            d["warning"] = self.warning
        if self.side_effects:
            d["side_effects"] = self.side_effects
        return d


def _unsupported(name: str, model: str = "") -> ControlValue:
    where = f" by {model}" if model else ""
    return ControlValue(ok=False, supported=False,
                        error=f"control '{name}' not supported{where}")


# --------------------------------------------------------------------------- #
# the driver ABC
# --------------------------------------------------------------------------- #

class CameraDriver:
    """Base class for per-camera drivers.

    Subclasses implement lifecycle/geometry/acquisition and call
    :meth:`_register` (typically during ``connect``) for each scalar control the
    hardware supports. Everything above the funnel — normalization, clamping,
    casting, unsupported handling — lives here and is identical for all cameras.
    """

    def __init__(self) -> None:
        self._controls: Dict[str, ControlSpec] = {}
        # Last-commanded values, keyed by control name. The region-change flow
        # re-applies THESE after a re-arm (hardware forgets), clamped against
        # fresh ranges at write time.
        self._commanded: Dict[str, float] = {}
        self._ctl_lock = threading.Lock()   # registry/commanded-store consistency

    # --- registration (driver-side) ----------------------------------------
    def _register(self, spec: ControlSpec) -> None:
        if spec.scale == "log":
            try:
                lo, _ = spec.get_range()
            except Exception:
                lo = 0.0
            if lo <= 0:
                spec.scale = "linear"    # log needs lo > 0 (DESIGN 2.1 guard)
        with self._ctl_lock:
            self._controls[spec.name] = spec

    def _unregister(self, name: str) -> None:
        with self._ctl_lock:
            self._controls.pop(name, None)
            self._commanded.pop(name, None)

    # --- the funnel (dock-side) --------------------------------------------
    def capabilities(self) -> dict:
        """Full capability document: every registered control + current value."""
        with self._ctl_lock:
            specs = list(self._controls.values())
        out = {}
        for spec in specs:
            entry = spec.describe()
            try:
                v = float(spec.getter())
                entry["value"] = v
                entry["normalized"] = spec.to_norm(v, entry["lo"], entry["hi"])
            except Exception as exc:
                entry["value"] = None
                entry["normalized"] = None
                entry["error"] = str(exc)
            out[spec.name] = entry
        return out

    def get(self, name: str) -> ControlValue:
        spec = self._controls.get(name)
        if spec is None:
            return _unsupported(name, self._model_for_errors())
        try:
            lo, hi = spec.get_range()
            v = float(spec.getter())
            return ControlValue(ok=True, supported=True, value=v,
                                normalized=spec.to_norm(v, lo, hi))
        except Exception as exc:
            return ControlValue(ok=False, supported=True, error=str(exc))

    def set(self, name: str, value: float, *, normalized: bool = False) -> ControlValue:
        spec = self._controls.get(name)
        if spec is None:
            return _unsupported(name, self._model_for_errors())
        if spec.setter is None:
            return ControlValue(ok=False, supported=True,
                                error=f"control '{name}' is read-only")
        try:
            lo, hi = spec.get_range()        # fresh hardware ranges at write time
            v = spec.to_native(value, lo, hi) if normalized else float(value)
            v = min(max(v, lo), hi)
            if spec.step:
                v = lo + round((v - lo) / spec.step) * spec.step
                v = min(max(v, lo), hi)
            if spec.kind == "int":
                v = int(round(v))
            spec.setter(v)
            applied = float(spec.getter())   # read back — hardware snaps silently
            with self._ctl_lock:
                self._commanded[name] = applied
            return ControlValue(ok=True, supported=True, value=applied,
                                normalized=spec.to_norm(applied, lo, hi))
        except Exception as exc:
            return ControlValue(ok=False, supported=True, error=str(exc))

    def commanded(self, name: str) -> Optional[float]:
        """Last successfully applied value (for re-apply after re-arm)."""
        with self._ctl_lock:
            return self._commanded.get(name)

    def _model_for_errors(self) -> str:
        try:
            return str(self.device_info.get("model", ""))
        except Exception:
            return ""

    # --- lifecycle (subclass responsibility) --------------------------------
    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        raise NotImplementedError

    @property
    def device_info(self) -> dict:
        raise NotImplementedError

    @property
    def bit_depth(self) -> int:
        return 8

    # --- geometry (subclass responsibility; absent capability -> raise
    #     NotImplementedError and the webapp treats ROI/binning as unsupported) --
    def sensor_size(self) -> Tuple[int, int]:
        raise NotImplementedError

    def image_size(self) -> Tuple[int, int]:
        raise NotImplementedError

    def roi_range(self) -> dict:
        raise NotImplementedError

    def set_roi(self, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        raise NotImplementedError

    def get_roi(self) -> Tuple[int, int, int, int]:
        raise NotImplementedError

    def reset_roi(self) -> Tuple[int, int, int, int]:
        raise NotImplementedError

    def binning_range(self) -> Tuple[int, int]:
        return (1, 1)

    def set_binning(self, bx: int, by: int) -> Tuple[int, int]:
        raise NotImplementedError

    def get_binning(self) -> Tuple[int, int]:
        return (1, 1)

    # --- acquisition (subclass responsibility) ------------------------------
    def start(self, max_throughput: bool = False) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    @property
    def is_grabbing(self) -> bool:
        raise NotImplementedError

    def grab(self, timeout_ms: int = 5000) -> Frame:
        raise NotImplementedError

    # --- legacy-surface shims ------------------------------------------------
    # presets.py and imaging.auto_expose were written against the old CameraBase
    # surface; these thin wrappers keep them (and their on-disk preset JSON)
    # working unchanged. New code should use the funnel directly.
    def get_exposure(self) -> float:
        return self.get("exposure").value or 0.0

    def set_exposure(self, microseconds: float) -> None:
        self.set("exposure", microseconds)

    def exposure_range(self) -> Tuple[float, float]:
        spec = self._controls.get("exposure")
        return spec.get_range() if spec else (0.0, 0.0)

    def get_gain(self) -> float:
        return self.get("gain").value or 0.0

    def set_gain(self, value: float) -> None:
        self.set("gain", value)

    def gain_range(self) -> Tuple[float, float]:
        spec = self._controls.get("gain")
        return spec.get_range() if spec else (0.0, 0.0)

    def get_frame_rate(self) -> float:
        return self.get("fps").value or 0.0

    def set_frame_rate(self, fps: float) -> None:
        self.set("fps", fps)

    def frame_rate_range(self) -> Tuple[float, float]:
        spec = self._controls.get("fps")
        return spec.get_range() if spec else (0.0, 0.0)

    def resulting_frame_rate(self) -> float:
        return self.get_frame_rate()

    # --- context manager -----------------------------------------------------
    def __enter__(self) -> "CameraDriver":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()


class UnavailableCamera(CameraDriver):
    """Stub returned by ``make_camera`` for unknown/paused/broken tokens.

    ``connect()`` raises with the reason, which ``CameraSession.start()`` catches
    into ``ok=False`` — the app boots, the dashboard shows the camera as
    unavailable, and every other camera is unaffected. This is why
    ``make_camera`` can be contractually never-raise.
    """

    def __init__(self, token: str, reason: str) -> None:
        super().__init__()
        self.token = token
        self.reason = reason

    def connect(self) -> None:
        raise RuntimeError(self.reason)

    def disconnect(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return False

    @property
    def device_info(self) -> dict:
        return {"model": self.token, "serial": "", "vendor": "",
                "unavailable": self.reason}


# Back-compat name: external code (old GUIs) may still import CameraBase.
CameraBase = CameraDriver
