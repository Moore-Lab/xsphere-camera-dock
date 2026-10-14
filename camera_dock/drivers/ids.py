"""IDS uEye driver — including Thorlabs DCC/DCx rebrands — behind the standardized interface.

Thorlabs DCC/DCx cameras (DCC1545M, DCC1240, DCC3240, ...) are rebranded IDS uEye
hardware: ThorCam's "DCx" stack ships the same SDK with the DLL renamed
``uc480(_64).dll`` and identical ``is_*`` exports. ``pyueye`` hardcodes the IDS DLL
name, so :func:`_import_ueye` re-binds it to uc480 when the IDS suite is absent.
Set ``THORLABS_UC480_DIR`` if ThorCam lives somewhere unusual.

The capture core is ported from ``ids-ueye/ids_ueye/camera.py`` (itself from the
200 fps-tested ``IDSCamera`` in ``reference/dualcam_fast.py``): freerun capture into
a 12-buffer ring, event-driven frame waits, pitch-stripped Mono8 frames, and the
free/AOI/realloc sequence for ROI changes. What is load-bearing here (measured on a
DCC1545M-class C1285R12M through uc480, 2026-10-14; see docs/DESIGN.md §3b):

* Two locks, and what matters is what each is held ACROSS. ``self._lock`` (RLock)
  wraps every SDK sequence on this camera. ``self._wait_lock`` is held by
  :meth:`grab` across ONE bounded event wait, and by whoever destroys the frame
  event (stop/disconnect/ROI) — closing a handle under a pending wait is undefined
  on Windows. Order is always ``_wait_lock -> _lock``; setters never touch
  ``_wait_lock``, so a waiting grab never starves them. ``_UEYE_LOCK`` serializes
  the process-wide calls (enumerate, init, exit) across cameras.
* ``is_GetImageInfo`` must be passed the uc480 struct size (80 B). pyueye's
  ``UEYEIMAGEINFO`` is the newer 96 B layout and uc480 rejects it with error 125
  (the legacy driver also passed the sequence number instead of the memory ID,
  so its per-frame metadata never worked). With both fixed, every frame carries
  the hardware frame counter and a 100 ns device timestamp.
* fps is read back from the hardware (``IS_GET_FRAMERATE``). The legacy driver
  echoed the commanded value, so a 30 fps request "succeeded" at 13.96.
* The uEye exposure ceiling IS the current frame period (unlike the TSI SDK). To
  give the DESIGN §2.1 semantics — fps wins when fps is set, exposure wins when
  exposure is set — the advertised exposure range runs to the period at the
  SLOWEST frame rate, and a long exposure lowers the frame rate to fit. The
  operator's fps request is remembered separately, so shortening the exposure
  again (or shrinking the ROI) restores it.
* Pixel clock (MHz) is a funnel control: it sets the frame-rate ceiling and
  shifts both other ranges. Above roughly 30 MHz at full frame this USB 2 link
  stops delivering (35 MHz: 13.9 of 20.4 fps arrive; 43 MHz: none), so raising it
  past the camera default comes back with a warning.
"""

from __future__ import annotations

import ctypes
import importlib.util
import math
import os
import sys
import threading
from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np

from ..base import CameraDriver, ControlSpec, ControlValue, Frame, FrameMeta

# Ring depth, hardware-tested at 12 in the reference GUI (absorbs host-side
# dequeue latency at 200 fps without dropping frames).
_RING_BUFFERS = 12

# One grab() event wait is at most this long, so stop()/ROI/disconnect wait at
# most this long for the frame event to be free (the uc480 path wakes the waiter
# immediately; the IDS is_WaitEvent path has no wake and relies on this bound).
_WAIT_CHUNK_MS = 100

# sizeof(UC480IMAGEINFO) in the DCx SDK's uc480.h (ends at dwHostProcessTime).
_UC480_IMAGEINFO_SIZE = 80

# A long exposure lowers the frame rate to this fraction of 1/exposure, leaving
# the sensor its readout margin (5 fps -> 199.92 ms max exposure, measured).
_EXPOSURE_FPS_FIT = 0.99

_IS_USE_DEVICE_ID = 0x8000

_DEFAULT_THORCAM_DIR = r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam"

# Process-wide uEye calls (enumerate / init / exit) — one at a time across cameras.
_UEYE_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# SDK binding
# --------------------------------------------------------------------------- #

def _import_ueye():
    """Import ``pyueye.ueye``, falling back to the Thorlabs uc480 DLL.

    pyueye resolves its DLL in a module-level ``load_dll`` call with the IDS names
    hardcoded, so a ThorCam-only machine fails the plain import even though
    uc480 exports the identical API. On that failure: load ``pyueye.dll``
    standalone, widen its loader to also try ``uc480_64``/``uc480`` (plus the
    ThorCam dir and ``THORLABS_UC480_DIR``), then run the package init against the
    patched loader. A genuine IDS install is always preferred.
    """
    if "pyueye.ueye" in sys.modules:
        return sys.modules["pyueye.ueye"]
    try:
        from pyueye import ueye
        return ueye
    except ImportError as first_exc:
        if importlib.util.find_spec("pyueye") is None:
            raise RuntimeError("pyueye is not installed (pip install pyueye).") from first_exc
        for mod in [m for m in sys.modules if m == "pyueye" or m.startswith("pyueye.")]:
            del sys.modules[mod]

        spec = importlib.util.find_spec("pyueye")
        pkg_dir = spec.submodule_search_locations[0]
        dll_spec = importlib.util.spec_from_file_location(
            "pyueye.dll", os.path.join(pkg_dir, "dll.py"))
        dll_mod = importlib.util.module_from_spec(dll_spec)
        sys.modules["pyueye.dll"] = dll_mod
        dll_spec.loader.exec_module(dll_mod)

        def load_dll_uc480(libinfo, libnames, envname=None):
            names = list(libnames) + ["uc480_64", "uc480"]
            parts = []
            env = os.getenv(envname) if envname else None
            if env:
                parts.append(env)
            thorcam = os.environ.get("THORLABS_UC480_DIR", _DEFAULT_THORCAM_DIR)
            if os.path.isdir(thorcam):
                parts.append(thorcam)
            try:
                dll = dll_mod.DLL(libinfo, names, os.pathsep.join(parts) or None)
            except RuntimeError as exc:
                raise ImportError(exc)
            return dll.libfile, dll.bind_function

        dll_mod.load_dll = load_dll_uc480
        pkg = importlib.util.module_from_spec(spec)
        sys.modules["pyueye"] = pkg
        try:
            spec.loader.exec_module(pkg)           # runs `from . import ueye`
        except BaseException:
            for mod in [m for m in sys.modules if m == "pyueye" or m.startswith("pyueye.")]:
                del sys.modules[mod]
            raise RuntimeError(
                "Could not load a uEye SDK DLL. For IDS cameras install the IDS "
                "Software Suite (ueye_api.dll); for Thorlabs DCC/DCx cameras install "
                "ThorCam with the DCx driver (uc480_64.dll) and, if it is in a "
                "non-standard place, set THORLABS_UC480_DIR. "
                f"Original error: {first_exc}") from first_exc
        return sys.modules["pyueye.ueye"]


def _decode(raw) -> str:
    return bytes(raw).decode("ascii", errors="replace").rstrip("\x00")


def _enumerate(ue) -> List[dict]:
    n = ctypes.c_int(0)
    if ue.is_GetNumberOfCameras(n) != ue.IS_SUCCESS or n.value <= 0:
        return []
    cam_list = ue.UEYE_CAMERA_LIST(ue.UEYE_CAMERA_INFO * n.value)
    cam_list.dwCount = n.value
    if ue.is_GetCameraList(cam_list) != ue.IS_SUCCESS:
        return []
    out = []
    for i in range(n.value):
        info = cam_list.uci[i]
        out.append({"device_id": int(info.dwDeviceID),
                    "model": _decode(info.FullModelName or info.Model),
                    "serial": _decode(info.SerNo),
                    "in_use": bool(info.dwInUse)})
    return out


def list_devices() -> List[dict]:
    """Attached uEye/DCx cameras as ``{device_id, model, serial, in_use}`` (never raises)."""
    try:
        ue = _import_ueye()
        with _UEYE_LOCK:
            return _enumerate(ue)
    except Exception:
        return []


def clamp_roi(x, y, w, h, sensor_w, sensor_h, step_x=1, step_y=1, min_w=1, min_h=1):
    """Align an ROI to the AOI steps and fit it on the sensor (from the reference GUI).

    The origin is pulled back so the minimum size always fits — otherwise a box
    near the right/bottom edge clamps below the SDK minimum.
    """
    if int(w) <= 0 or int(h) <= 0:
        raise ValueError("ROI width and height must be positive")
    x = max(0, (int(x) // step_x) * step_x)
    y = max(0, (int(y) // step_y) * step_y)
    x = min(x, ((sensor_w - min_w) // step_x) * step_x)
    y = min(y, ((sensor_h - min_h) // step_y) * step_y)
    w = max(min_w, (int(w) // step_x) * step_x)
    h = max(min_h, (int(h) // step_y) * step_y)
    return x, y, min(w, sensor_w - x), min(h, sensor_h - y)


class _FrameEvent:
    """Frame-ready event for both uEye driver generations.

    The IDS ``ueye_api`` DLL waits with ``is_WaitEvent``. The Thorlabs ``uc480``
    DLL predates it: the application registers its own Windows event with
    ``is_InitEvent`` and waits with ``WaitForSingleObject`` — which also lets
    :meth:`wake` release a waiter instantly. ``disable`` destroys the event, so a
    signal left from before a stop can never satisfy the first wait after a
    restart.
    """

    _WAIT_OBJECT_0 = 0x0
    _WAIT_TIMEOUT = 0x102

    def __init__(self, ue, hCam) -> None:
        self._ue = ue
        self._hCam = hCam
        self._use_wait_event = getattr(ue, "_is_WaitEvent", True) is not None
        self._hEvent = None
        self.enabled = False
        if not self._use_wait_event:
            k32 = ctypes.windll.kernel32
            k32.CreateEventW.restype = ctypes.c_void_p
            k32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_void_p]
            k32.WaitForSingleObject.restype = ctypes.c_uint32
            k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            k32.SetEvent.argtypes = [ctypes.c_void_p]
            k32.CloseHandle.argtypes = [ctypes.c_void_p]

    def enable(self) -> None:
        if self.enabled:
            return
        ue = self._ue
        if not self._use_wait_event:
            k32 = ctypes.windll.kernel32
            hEvent = k32.CreateEventW(None, 0, 0, None)    # auto-reset, unsignaled
            if not hEvent:
                raise RuntimeError("CreateEvent failed for the uc480 frame event")
            ret = ue.is_InitEvent(self._hCam, ctypes.c_void_p(hEvent), ue.IS_SET_EVENT_FRAME)
            if ret != ue.IS_SUCCESS:
                k32.CloseHandle(hEvent)
                raise RuntimeError(f"is_InitEvent(FRAME) failed: {ret}")
            self._hEvent = hEvent
        ret = ue.is_EnableEvent(self._hCam, ue.IS_SET_EVENT_FRAME)
        if ret != ue.IS_SUCCESS:
            self._exit_win_event()
            raise RuntimeError(f"is_EnableEvent(FRAME) failed: {ret}")
        self.enabled = True

    def disable(self) -> None:
        if not self.enabled:
            return
        try:
            self._ue.is_DisableEvent(self._hCam, self._ue.IS_SET_EVENT_FRAME)
        finally:
            self._exit_win_event()
            self.enabled = False

    def _exit_win_event(self) -> None:
        if self._hEvent is not None:
            try:
                self._ue.is_ExitEvent(self._hCam, self._ue.IS_SET_EVENT_FRAME)
            except Exception:
                pass
            ctypes.windll.kernel32.CloseHandle(self._hEvent)
            self._hEvent = None

    def wake(self) -> None:
        """Release a pending :meth:`wait` now (uc480 path; the IDS path times out)."""
        h = self._hEvent
        if h is not None:
            ctypes.windll.kernel32.SetEvent(h)

    def wait(self, timeout_ms: int) -> bool:
        """``True`` when the frame event fired, ``False`` on timeout."""
        ue = self._ue
        if self._use_wait_event:
            ret = ue.is_WaitEvent(self._hCam, ue.IS_SET_EVENT_FRAME, int(timeout_ms))
            if ret == ue.IS_TIMED_OUT:
                return False
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_WaitEvent(FRAME) error: {ret}")
            return True
        r = ctypes.windll.kernel32.WaitForSingleObject(self._hEvent, int(timeout_ms))
        if r == self._WAIT_TIMEOUT:
            return False
        if r != self._WAIT_OBJECT_0:
            raise RuntimeError(f"WaitForSingleObject(frame event) error: {r}")
        return True


# --------------------------------------------------------------------------- #
# the driver
# --------------------------------------------------------------------------- #

class IDSCamera(CameraDriver):
    """One IDS uEye or Thorlabs DCC/DCx camera (mono, Mono8 frames).

    Select a unit by serial (``ids:4103060588``). With several uEye cameras
    attached, an unqualified open is refused rather than guessed.
    """

    def __init__(self, serial: Optional[str] = None) -> None:
        super().__init__()
        self.serial = serial
        self._ue = None
        self._hCam = None
        self._info: Optional[dict] = None        # cached identity; no DLL per poll
        self._sensor = (0, 0)
        self._aoi = (0, 0, 0, 0)                 # applied AOI (read back), sensor px
        self._roi_range: Optional[dict] = None   # constant per camera; cached
        self._mem: List[Tuple[object, ctypes.c_int]] = []
        self._mem_by_addr: dict = {}             # buffer address -> memory id
        self._pitch = 0
        self._event: Optional[_FrameEvent] = None
        self._live = False                       # hardware capture state (under _lock)
        self._capturing = False                  # intent flag read by grab() unlocked
        self._last_key = None                    # dedup of re-signalled frames
        self._fps_request: Optional[float] = None  # operator's fps intent (see module doc)
        self._pclk_default: Optional[int] = None
        self._pclk_values: List[int] = []
        self._lock = threading.RLock()
        self._wait_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        if self._hCam is not None:
            return                               # idempotent: a retry must not re-open
        try:
            ue = _import_ueye()
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                "Could not load the uEye SDK (pyueye). Install pyueye plus either the "
                "IDS Software Suite or, for Thorlabs DCC cameras, ThorCam with the DCx "
                f"driver. Original error: {exc}") from exc

        with _UEYE_LOCK:
            device_id = self._pick_device(ue)
            hCam = ctypes.c_uint(device_id | _IS_USE_DEVICE_ID)
            ret = ue.is_InitCamera(hCam, None)
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_InitCamera failed (device {device_id}): {ret}")
        self._ue, self._hCam = ue, hCam
        try:
            with self._lock:
                self._open_locked()
        except Exception:
            self.disconnect()                    # never leak the handle
            raise

    def _pick_device(self, ue) -> int:
        devices = _enumerate(ue)
        if self.serial:
            for d in devices:
                if d["serial"] == self.serial:
                    if d["in_use"]:
                        raise RuntimeError(
                            f"uEye camera {self.serial} is in use (close ThorCam / "
                            "uEye Cockpit, or another dock session has it).")
                    return d["device_id"]
            seen = ", ".join(d["serial"] for d in devices) or "none"
            raise RuntimeError(f"uEye camera {self.serial} not found (attached: {seen}).")
        free = [d for d in devices if not d["in_use"]]
        if not free:
            if devices:
                raise RuntimeError("All uEye cameras are in use (close ThorCam / "
                                   "uEye Cockpit first).")
            raise RuntimeError("No uEye / Thorlabs DCx cameras found.")
        if len(free) > 1:
            choices = ", ".join(f"{d['model']} ({d['serial']})" for d in free)
            raise RuntimeError(f"Several uEye cameras attached; pick one by serial, "
                               f"e.g. ids:{free[0]['serial']}. Attached: {choices}")
        return free[0]["device_id"]

    def _open_locked(self) -> None:
        ue, h = self._ue, self._hCam
        sinfo = ue.SENSORINFO()
        ret = ue.is_GetSensorInfo(h, sinfo)
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_GetSensorInfo failed: {ret}")
        self._sensor = (int(sinfo.nMaxWidth), int(sinfo.nMaxHeight))
        binfo = ue.BOARDINFO()
        serial = (_decode(binfo.SerNo)
                  if ue.is_GetCameraInfo(h, binfo) == ue.IS_SUCCESS else "unknown")
        dll = os.path.basename(str(getattr(ue, "get_dll_file", "")))
        self._info = {
            "model": _decode(sinfo.strSensorName), "serial": serial,
            "vendor": "Thorlabs (uc480)" if "uc480" in dll.lower() else "IDS",
            "sdk": f"pyueye / {dll}",
        }

        ret = ue.is_SetColorMode(h, ue.IS_CM_MONO8)
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_SetColorMode(MONO8) failed: {ret}")
        self._roi_range = self._query_roi_range()
        self._aoi = self._read_aoi()
        self._alloc_ring(self._aoi[2], self._aoi[3])
        self._event = _FrameEvent(ue, h)
        self._register_controls(bool(sinfo.bMasterGain))

    def _wake_waiter(self) -> None:
        # Under _lock: the event is only ever destroyed with _lock held, so the
        # handle cannot be closed (and its value reused) under this SetEvent.
        with self._lock:
            if self._event is not None:
                self._event.wake()

    def disconnect(self) -> None:
        """Release the camera. Contractually NEVER raises."""
        self._capturing = False
        try:
            self._wake_waiter()
        except Exception:
            pass
        with self._wait_lock, self._lock:
            ue, h = self._ue, self._hCam
            if ue is not None and h is not None:
                for step in (self._stop_locked, self._free_ring):
                    try:
                        step()
                    except Exception as exc:
                        print(f"IDSCamera.disconnect {step.__name__}: {exc}")
                try:
                    with _UEYE_LOCK:
                        ue.is_ExitCamera(h)
                except Exception as exc:
                    print(f"IDSCamera.disconnect is_ExitCamera: {exc}")
            self._hCam = None
            self._event = None
            self._live = False
        for name in list(self._controls):
            self._unregister(name)

    @property
    def is_connected(self) -> bool:
        return self._hCam is not None

    def _require(self):
        if self._hCam is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")
        return self._ue

    @property
    def device_info(self) -> dict:
        self._require()
        return dict(self._info or {})

    @property
    def bit_depth(self) -> int:
        return 8

    # ------------------------------------------------------------------ #
    # scalar controls
    # ------------------------------------------------------------------ #
    def _register_controls(self, has_master_gain: bool) -> None:
        """Probe and register what this unit has. Lock held, camera open, not live."""
        ue, h = self._ue, self._hCam

        self._register(ControlSpec(
            name="exposure", units="us", kind="float", scale="log",
            get_range=self._exposure_range_advertised,
            getter=self._get_exposure,
            setter=self._set_exposure_hw,
            display={"label": "exposure", "unit_display": "ms", "unit_scale": 1e-3},
        ))

        if self._frame_time_range() is not None:
            self._register(ControlSpec(
                name="fps", units="fps", kind="float",
                get_range=self._fps_range,
                getter=self._get_fps,
                setter=self._set_fps_hw,
                display={"label": "target fps"},
            ))

        if has_master_gain:
            # Master gain is an index 0..100; the SDK reports the matching
            # amplification x100, so the dB endpoints are exact, not assumed.
            try:
                f_hi = ue.is_SetHWGainFactor(h, ue.IS_INQUIRE_MASTER_GAIN_FACTOR, 100)
                db_hi = 20.0 * math.log10(f_hi / 100.0) if f_hi > 0 else 0.0
            except Exception:
                db_hi = 0.0
            self._register(ControlSpec(
                name="gain", units="index", kind="int", step=1,
                get_range=lambda: (0.0, 100.0),
                getter=self._get_gain,
                setter=self._set_gain_hw,
                display={"label": "gain", "db_lo": 0.0, "db_hi": db_hi},
            ))

        rng = (ctypes.c_uint * 3)()
        if ue.is_PixelClock(h, ue.IS_PIXELCLOCK_CMD_GET_RANGE, rng,
                            ctypes.sizeof(rng)) == ue.IS_SUCCESS and rng[1] > rng[0]:
            lo, hi, inc = int(rng[0]), int(rng[1]), int(rng[2])
            if inc > 0:
                self._pclk_values = list(range(lo, hi + 1, inc))
            else:                                # discrete clocks only
                n = ctypes.c_uint()
                ue.is_PixelClock(h, ue.IS_PIXELCLOCK_CMD_GET_NUMBER, n, ctypes.sizeof(n))
                arr = (ctypes.c_uint * max(n.value, 1))()
                ue.is_PixelClock(h, ue.IS_PIXELCLOCK_CMD_GET_LIST, arr, ctypes.sizeof(arr))
                self._pclk_values = sorted(set(int(v) for v in arr[:n.value])) or [lo, hi]
            d = ctypes.c_uint()
            if ue.is_PixelClock(h, ue.IS_PIXELCLOCK_CMD_GET_DEFAULT, d,
                                ctypes.sizeof(d)) == ue.IS_SUCCESS:
                self._pclk_default = int(d.value)
            self._register(ControlSpec(
                name="pixel_clock", units="MHz", kind="int", step=max(inc, 0),
                get_range=lambda: (float(self._pclk_values[0]), float(self._pclk_values[-1])),
                getter=self._get_pclk,
                setter=self._set_pclk_hw,
                display={"label": "pixel clock"},
            ))

    def _locked(self, fn):
        with self._lock:
            self._require()
            return fn()

    # --- raw hardware accessors (the funnel's getters/setters) ---
    def _exposure_range_hw(self) -> Tuple[float, float]:
        """Exposure limits (us) at the CURRENT frame period."""
        def q():
            ue = self._ue
            rng = (ctypes.c_double * 3)()        # min, max, increment (ms)
            ret = ue.is_Exposure(self._hCam, ue.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE,
                                 rng, ctypes.sizeof(rng))
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_Exposure(GET_RANGE) failed: {ret}")
            return rng[0] * 1000.0, rng[1] * 1000.0
        return self._locked(q)

    def _exposure_range_advertised(self) -> Tuple[float, float]:
        """Hardware minimum .. the longest exposure reachable at the SLOWEST rate.

        A request above the current frame period lowers the frame rate (see
        _set_exposure_coupled); the hardware read-back is still authoritative.
        """
        lo, hi = self._exposure_range_hw()
        ft = self._frame_time_range()
        if ft is not None:
            hi = max(hi, ft[1] * 1e6)
        return lo, hi

    def _get_exposure(self) -> float:
        def q():
            ue = self._ue
            ms = ctypes.c_double()
            ret = ue.is_Exposure(self._hCam, ue.IS_EXPOSURE_CMD_GET_EXPOSURE,
                                 ms, ctypes.sizeof(ms))
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_Exposure(GET) failed: {ret}")
            return ms.value * 1000.0
        return self._locked(q)

    def _set_exposure_hw(self, us: float) -> None:
        def w():
            ue = self._ue
            ms = ctypes.c_double(float(us) / 1000.0)
            ret = ue.is_Exposure(self._hCam, ue.IS_EXPOSURE_CMD_SET_EXPOSURE,
                                 ms, ctypes.sizeof(ms))
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_Exposure(SET {us:g} us) failed: {ret}")
        self._locked(w)

    def _frame_time_range(self) -> Optional[Tuple[float, float]]:
        """(shortest, longest) frame period in seconds, or None."""
        def q():
            t_min, t_max, t_inc = ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
            ret = self._ue.is_GetFrameTimeRange(self._hCam, t_min, t_max, t_inc)
            if ret != self._ue.IS_SUCCESS or t_min.value <= 0 or t_max.value <= 0:
                return None
            return t_min.value, t_max.value
        return self._locked(q)

    def _fps_range(self) -> Tuple[float, float]:
        ft = self._frame_time_range()
        if ft is None:
            raise RuntimeError("frame-rate range unavailable")
        return 1.0 / ft[1], 1.0 / ft[0]

    def _get_fps(self) -> float:
        def q():
            ue = self._ue
            fps = ctypes.c_double()
            ret = ue.is_SetFrameRate(self._hCam, ue.IS_GET_FRAMERATE, fps)
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_SetFrameRate(GET) failed: {ret}")
            return fps.value
        return self._locked(q)

    def _set_fps_hw(self, fps: float) -> None:
        def w():
            new = ctypes.c_double()
            ret = self._ue.is_SetFrameRate(self._hCam, ctypes.c_double(float(fps)), new)
            if ret != self._ue.IS_SUCCESS:
                raise RuntimeError(f"is_SetFrameRate({fps:g}) failed: {ret}")
        self._locked(w)

    def _get_gain(self) -> float:
        def q():
            ue = self._ue
            ign = ue.IS_IGNORE_PARAMETER
            return float(ue.is_SetHardwareGain(self._hCam, ue.IS_GET_MASTER_GAIN, ign, ign, ign))
        return self._locked(q)

    def _set_gain_hw(self, v: float) -> None:
        def w():
            ue = self._ue
            ign = ue.IS_IGNORE_PARAMETER
            ret = ue.is_SetHardwareGain(self._hCam, int(v), ign, ign, ign)
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_SetHardwareGain({int(v)}) failed: {ret}")
        self._locked(w)

    def _get_pclk(self) -> float:
        def q():
            ue = self._ue
            val = ctypes.c_uint()
            ret = ue.is_PixelClock(self._hCam, ue.IS_PIXELCLOCK_CMD_GET, val, ctypes.sizeof(val))
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_PixelClock(GET) failed: {ret}")
            return float(val.value)
        return self._locked(q)

    def _set_pclk_hw(self, mhz: float) -> None:
        mhz = min(self._pclk_values, key=lambda c: abs(c - mhz))   # nearest supported
        def w():
            ue = self._ue
            val = ctypes.c_uint(int(mhz))
            ret = ue.is_PixelClock(self._hCam, ue.IS_PIXELCLOCK_CMD_SET, val, ctypes.sizeof(val))
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_PixelClock(SET {mhz} MHz) failed: {ret}")
        self._locked(w)

    # --- fps/exposure/pixel-clock coupling (DESIGN §2.1 semantics) ---
    def set(self, name: str, value: float, *, normalized: bool = False) -> ControlValue:
        # Lock held across range read + write + read-back + coupling, so no other
        # setter lands in between. RLock: the funnel's accessors re-enter freely.
        with self._lock:
            if name == "fps":
                return self._set_fps_coupled(value, normalized)
            if name == "exposure":
                return self._set_exposure_coupled(value, normalized)
            if name == "pixel_clock":
                return self._set_pclk_coupled(value, normalized)
            return super().set(name, value, normalized=normalized)

    def _remember(self, name: str, value: Optional[float]) -> None:
        if value is not None:
            with self._ctl_lock:
                self._commanded[name] = float(value)

    def _set_fps_coupled(self, value: float, normalized: bool) -> ControlValue:
        """fps wins: the camera shortens an exposure that no longer fits the period."""
        before = self.get("exposure")
        result = super().set("fps", value, normalized=normalized)
        if not result.ok:
            return result
        # A typed/default value is remembered RAW: 30 fps clamps to 13.96 at full
        # frame but is reached once the ROI shrinks. A slider move is its value.
        self._fps_request = float(result.value if normalized else value)
        after = self.get("exposure")
        if before.ok and after.ok and after.value < before.value * (1 - 1e-3):
            self._remember("exposure", after.value)   # re-arm must not undo this
            result.warning = (f"exposure reduced to {after.value / 1000:.2f} ms to fit "
                              f"{result.value:.1f} fps")
            result.side_effects["exposure"] = after.as_dict()
        return result

    def _set_exposure_coupled(self, value: float, normalized: bool) -> ControlValue:
        """Exposure wins: the frame rate drops to fit it, and comes back when it can."""
        spec = self._controls.get("exposure")
        if spec is None or "fps" not in self._controls:
            return super().set("exposure", value, normalized=normalized)
        try:
            lo, hi = spec.get_range()
            target = spec.to_native(value, lo, hi) if normalized else float(value)
            target = min(max(target, lo), hi)
            want = self._fps_request if self._fps_request is not None else self._get_fps()
            fps_lo, fps_hi = self._fps_range()
            cap = min(want, fps_hi)              # what the rate would be anyway
            fit = _EXPOSURE_FPS_FIT * 1e6 / max(target, 1.0)
            new_fps = max(min(cap, fit), fps_lo)
            exposure_binds = fit < cap * (1 - 1e-3)
            cur_fps = self._get_fps()
        except Exception as exc:
            return ControlValue(ok=False, supported=True, error=str(exc))
        side, warning = {}, None
        if abs(new_fps - cur_fps) > 1e-3 * max(cur_fps, 1.0):
            # fps first: the exposure ceiling is the frame period.
            r = super().set("fps", new_fps)
            if r.ok:
                side["fps"] = r.as_dict()
                if exposure_binds:
                    warning = (f"{target / 1000:.1f} ms exposure caps the frame rate at "
                               f"{r.value:.2f} fps")
        result = super().set("exposure", target)
        result.side_effects.update(side)
        if result.ok and warning:
            result.warning = warning
        return result

    def _set_pclk_coupled(self, value: float, normalized: bool) -> ControlValue:
        """The pixel clock moves both other ranges; re-apply the operator's intent."""
        result = super().set("pixel_clock", value, normalized=normalized)
        if not result.ok:
            return result
        self._reapply()
        for n in ("fps", "exposure"):
            if n in self._controls:
                result.side_effects[n] = self.get(n).as_dict()
        if self._pclk_default and result.value > self._pclk_default:
            result.warning = (f"above the {self._pclk_default} MHz default; if the stream "
                              "stalls, the USB link cannot carry this rate; lower the pixel "
                              "clock or shrink the ROI")
        return result

    def _reapply(self) -> None:
        """Restore fps then exposure intent after anything that resets or re-ranges them
        (capture start, AOI change, pixel clock). Clamps against fresh ranges."""
        exp = self.commanded("exposure")
        fps = self._fps_request if self._fps_request is not None else self.commanded("fps")
        if fps is not None and "fps" in self._controls:
            super().set("fps", fps)
        if exp is not None:
            self._set_exposure_coupled(exp, False)

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def sensor_size(self) -> Tuple[int, int]:
        self._require()
        return self._sensor

    def image_size(self) -> Tuple[int, int]:
        self._require()
        return self._aoi[2], self._aoi[3]

    def _query_roi_range(self) -> dict:
        ue, h = self._ue, self._hCam
        size_min, size_inc, pos_inc = ue.IS_SIZE_2D(), ue.IS_SIZE_2D(), ue.IS_POINT_2D()
        w_min, h_min, w_inc, h_inc, x_inc, y_inc = 32, 4, 4, 2, 4, 2   # DCC1545M values
        if ue.is_AOI(h, ue.IS_AOI_IMAGE_GET_SIZE_MIN, size_min,
                     ctypes.sizeof(size_min)) == ue.IS_SUCCESS:
            w_min, h_min = int(size_min.s32Width), int(size_min.s32Height)
        if ue.is_AOI(h, ue.IS_AOI_IMAGE_GET_SIZE_INC, size_inc,
                     ctypes.sizeof(size_inc)) == ue.IS_SUCCESS:
            w_inc, h_inc = int(size_inc.s32Width), int(size_inc.s32Height)
        if ue.is_AOI(h, ue.IS_AOI_IMAGE_GET_POS_INC, pos_inc,
                     ctypes.sizeof(pos_inc)) == ue.IS_SUCCESS:
            x_inc, y_inc = int(pos_inc.s32X), int(pos_inc.s32Y)
        sw, sh = self._sensor
        return {"w_min": w_min, "h_min": h_min, "w_max": sw, "h_max": sh,
                "w_inc": w_inc, "h_inc": h_inc, "x_inc": x_inc, "y_inc": y_inc,
                "min_lrx": w_min - 1, "min_lry": h_min - 1,
                "max_ulx": sw - w_min, "max_uly": sh - h_min}

    def roi_range(self) -> dict:
        self._require()
        return dict(self._roi_range)

    def _read_aoi(self) -> Tuple[int, int, int, int]:
        ue = self._ue
        rect = ue.IS_RECT()
        ret = ue.is_AOI(self._hCam, ue.IS_AOI_IMAGE_GET_AOI, rect, ctypes.sizeof(rect))
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_AOI(GET) failed: {ret}")
        return (int(rect.s32X), int(rect.s32Y), int(rect.s32Width), int(rect.s32Height))

    def _apply_aoi_locked(self, x: int, y: int, w: int, h: int) -> None:
        ue = self._ue
        self._free_ring()
        rect = ue.IS_RECT()
        rect.s32X, rect.s32Y, rect.s32Width, rect.s32Height = x, y, w, h
        ret = ue.is_AOI(self._hCam, ue.IS_AOI_IMAGE_SET_AOI, rect, ctypes.sizeof(rect))
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_AOI(SET {x},{y} {w}x{h}) failed: {ret}")
        self._aoi = self._read_aoi()               # the camera may snap
        self._alloc_ring(self._aoi[2], self._aoi[3])
        self._last_key = None

    def set_roi(self, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        """Apply a sensor-coordinate ROI; return the APPLIED (read-back) ROI.

        The ring is sized to the AOI, so it is freed and re-allocated — never
        with capture live. The dock stops the engine first; a direct caller with
        capture running gets it stopped and restarted around the change.
        """
        self._require()
        r = self._roi_range
        x, y, w, h = clamp_roi(x, y, w, h, *self._sensor,
                               step_x=max(r["x_inc"], r["w_inc"]),
                               step_y=max(r["y_inc"], r["h_inc"]),
                               min_w=r["w_min"], min_h=r["h_min"])
        was = self.is_grabbing
        if was:
            self.stop()
        try:
            with self._wait_lock, self._lock:
                try:
                    self._apply_aoi_locked(x, y, w, h)
                except Exception:
                    # Leave the camera operable at full sensor; the original
                    # error stays primary.
                    try:
                        self._apply_aoi_locked(0, 0, *self._sensor)
                    except Exception as exc:
                        print(f"IDSCamera: full-sensor restore after failed ROI: {exc}")
                    raise
                return self._aoi
        finally:
            if was:
                self.start()

    def get_roi(self) -> Tuple[int, int, int, int]:
        self._require()
        return self._aoi

    def reset_roi(self) -> Tuple[int, int, int, int]:
        return self.set_roi(0, 0, *self._sensor)

    # binning: not exposed on uEye/DCx (base defaults report (1, 1))

    # ------------------------------------------------------------------ #
    # ring buffer
    # ------------------------------------------------------------------ #
    def _alloc_ring(self, w: int, h: int) -> None:
        """Allocate + queue the Mono8 ring and index it by address. All-or-nothing."""
        ue, hc = self._ue, self._hCam
        try:
            for i in range(_RING_BUFFERS):
                ptr, mid = ue.c_mem_p(), ctypes.c_int()
                ret = ue.is_AllocImageMem(hc, w, h, 8, ptr, mid)
                if ret != ue.IS_SUCCESS:
                    raise RuntimeError(f"is_AllocImageMem failed on buffer {i}: {ret}")
                ret = ue.is_AddToSequence(hc, ptr, mid)
                if ret != ue.IS_SUCCESS:
                    ue.is_FreeImageMem(hc, ptr, mid)
                    raise RuntimeError(f"is_AddToSequence failed on buffer {i}: {ret}")
                self._mem.append((ptr, mid))
                self._mem_by_addr[ptr.value] = mid.value
        except Exception:
            self._free_ring()
            raise
        pitch = ctypes.c_int()
        ue.is_GetImageMemPitch(hc, pitch)
        self._pitch = pitch.value

    def _free_ring(self) -> None:
        ue, hc = self._ue, self._hCam
        ue.is_ClearSequence(hc)
        for ptr, mid in self._mem:
            ue.is_FreeImageMem(hc, ptr, mid)
        self._mem.clear()
        self._mem_by_addr.clear()

    # ------------------------------------------------------------------ #
    # acquisition
    # ------------------------------------------------------------------ #
    def start(self, max_throughput: bool = False) -> None:
        """Freerun capture (trigger off + live video). The ring is already sized
        for full-rate capture, so ``max_throughput`` changes nothing here."""
        ue = self._require()
        with self._lock:
            if self._live:
                return
            self._event.enable()
            ret = ue.is_SetExternalTrigger(self._hCam, ue.IS_SET_TRIGGER_OFF)
            if ret != ue.IS_SUCCESS:
                self._event.disable()
                raise RuntimeError(f"is_SetExternalTrigger(OFF) failed: {ret}")
            ret = ue.is_CaptureVideo(self._hCam, ue.IS_DONT_WAIT)
            if ret != ue.IS_SUCCESS:
                self._event.disable()
                raise RuntimeError(f"is_CaptureVideo failed: {ret}")
            self._live = True
            self._capturing = True
            self._last_key = None
            self._reapply()

    def _stop_locked(self) -> None:
        if self._live:
            self._live = False
            try:
                self._ue.is_StopLiveVideo(self._hCam, self._ue.IS_FORCE_VIDEO_STOP)
            finally:
                if self._event is not None:
                    self._event.disable()

    def stop(self) -> None:
        if self._hCam is None:
            return
        self._capturing = False              # grab() gives up after its current wait
        self._wake_waiter()                  # ... which on uc480 ends right now
        with self._wait_lock, self._lock:    # event is free: safe to destroy it
            self._stop_locked()

    @property
    def is_grabbing(self) -> bool:
        return self._hCam is not None and self._live

    def grab(self, timeout_ms: int = 5000) -> Frame:
        """Next frame from the running stream (event wait + copy + hw metadata).

        The event wait holds only ``_wait_lock`` (setters stay free); the buffer
        lookup and copy hold ``_lock``, which every ring teardown also takes.
        """
        deadline = perf_counter() + timeout_ms / 1000.0
        while True:
            if not self._capturing:
                raise RuntimeError("acquisition stopped")
            remaining_ms = int((deadline - perf_counter()) * 1000)
            with self._wait_lock:
                ev = self._event
                if not self._capturing or ev is None or not ev.enabled:
                    raise RuntimeError("acquisition stopped")
                fired = ev.wait(max(1, min(_WAIT_CHUNK_MS, remaining_ms)))
            if fired:
                frame = self._take_frame()
                if frame is not None:
                    return frame
            if perf_counter() >= deadline:
                raise TimeoutError("Timed out waiting for a frame from the uEye camera.")

    def _take_frame(self) -> Optional[Frame]:
        with self._lock:
            ue, hc = self._ue, self._hCam
            if hc is None or not self._live:
                return None
            n, cur, last = ctypes.c_int(), ue.c_mem_p(), ue.c_mem_p()
            if ue.is_GetActSeqBuf(hc, n, cur, last) != ue.IS_SUCCESS or not last.value:
                return None                  # nothing transferred yet (fresh sequence)
            hw_count = hw_ts = None
            mid = self._mem_by_addr.get(last.value)
            if mid is not None:
                info = ue.UEYEIMAGEINFO()
                if ue.is_GetImageInfo(hc, ctypes.c_int(mid), info,
                                      _UC480_IMAGEINFO_SIZE) == ue.IS_SUCCESS:
                    hw_count = int(info.u64FrameNumber)
                    hw_ts = float(info.u64TimestampDevice) * 100.0   # 100 ns ticks
            # A stale or doubled signal points at the frame already delivered.
            key = hw_count if hw_count is not None else (n.value, last.value)
            if key == self._last_key:
                return None
            self._last_key = key
            _, _, w, h = self._aoi
            raw = (ctypes.c_ubyte * (h * self._pitch)).from_address(last.value)
            data = np.frombuffer(raw, dtype=np.uint8).reshape(h, self._pitch)[:, :w].copy()
        return Frame(data=data, meta=FrameMeta(t_sw=perf_counter(), hw_count=hw_count,
                                               hw_timestamp_ns=hw_ts))
