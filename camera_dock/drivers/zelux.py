"""Thorlabs Zelux CS165MU driver — TSI SDK 0.0.8 behind the standardized interface.

Translates the dock's command set (camera_dock.base.CameraDriver) to the Thorlabs
Scientific Camera SDK. Everything here follows docs/DESIGN.md §3; the non-obvious
choices are all load-bearing:

* One internal RLock serializes EVERY SDK touch — the TSI DLL is not safe for
  concurrent calls (dualcam_fast needed a global SDK lock for the same reason).
  The poll path uses a non-blocking poll (image_poll_timeout_ms=0) plus a ~1 ms
  sleep so the producer thread never starves setter threads waiting on the lock.
* Continuous video is SOFTWARE_TRIGGERED + frames_per_trigger_zero_for_unlimited=0
  (re-set before EVERY arm) + exactly one issue_software_trigger() after arm —
  omit the trigger and polling returns None forever, with no error.
* ROI is disarm-only, corner-coordinate (inclusive), and the hardware snaps
  requests silently — set_roi holds the lock across the whole
  disarm→set→read-back→arm→trigger→re-apply sequence and returns the APPLIED roi.
* fps and exposure do not survive re-arm; set_roi re-applies the last-commanded
  values through the funnel, which clamps against fresh post-arm ranges.
* Frame.image_buffer is a zero-copy view over the SDK ring — np.copy at the poll
  site. SDK 0.0.8 already shapes it (h, w) from arm-time dims; no reshape here.
"""

from __future__ import annotations

import os
import threading
from time import perf_counter, sleep
from typing import Optional, Tuple

import numpy as np

from ..base import CameraDriver, ControlSpec, ControlValue, Frame, FrameMeta

# Default location of the Thorlabs native DLLs on a standard ThorCam install.
_DEFAULT_DLL_DIR = r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam"

# ROI increment hints. The SDK exposes no increment API; 2 matches observed
# CS165MU behavior. Hardware read-back is authoritative regardless.
_ROI_INC = 2

# Exposure may occupy at most this fraction of the frame period before it starts
# capping the achievable rate (the SDK applies such settings silently).
_EXPOSURE_DUTY_MAX = 0.98


def _add_dll_directory() -> None:
    """Put the Thorlabs native-DLL dir on the search path (before TLCameraSDK()).

    BOTH steps are required: os.add_dll_directory lets ctypes find the main DLL;
    the PATH prepend lets that DLL's own dependencies resolve.
    """
    dll_dir = os.environ.get("THORLABS_TSI_DLL_DIR", _DEFAULT_DLL_DIR)
    if os.path.isdir(dll_dir):
        try:
            os.add_dll_directory(dll_dir)
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")


class ZeluxCamera(CameraDriver):
    """One Thorlabs Zelux CS165MU (USB3, mono, 1440x1080, 10-bit).

    The TSI SDK allows one TLCameraSDK instance per process; this driver owns it
    between connect() and disconnect(). Don't hold two connected instances.
    """

    def __init__(self, serial: Optional[str] = None) -> None:
        super().__init__()
        self.serial = serial
        self._sdk = None
        self._cam = None
        self._lock = threading.RLock()   # serializes every SDK call
        self._max_throughput = False

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def connect(self) -> None:
        _add_dll_directory()
        try:
            from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
            from thorlabs_tsi_sdk.tl_camera_enums import OPERATION_MODE
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Could not load the Thorlabs SDK (thorlabs_tsi_sdk). Install the "
                "wheel from the Scientific Camera SDK and ensure the native DLLs "
                "are discoverable (THORLABS_TSI_DLL_DIR or a ThorCam install). "
                "Original error: " + str(exc)) from exc

        with self._lock:
            try:
                self._sdk = TLCameraSDK()
            except Exception:
                # A prior failed dispose leaves the singleton flag set without a
                # working SDK (0.0.8: dispose() raises WITHOUT clearing the flag).
                # Force-reset and retry once.
                if getattr(TLCameraSDK, "_is_sdk_open", False):
                    TLCameraSDK._is_sdk_open = False
                    self._sdk = TLCameraSDK()
                else:
                    raise
            try:
                serials = self._sdk.discover_available_cameras()
                if not serials:
                    raise RuntimeError("No Thorlabs cameras found.")
                target = str(self.serial) if self.serial else serials[0]
                if target not in serials:
                    raise RuntimeError(
                        f"Camera {target} not found. Available: {', '.join(serials)}")
                self._cam = self._sdk.open_camera(target)
            except Exception:
                self._dispose_sdk()
                raise

            cam = self._cam
            cam.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
            cam.frames_per_trigger_zero_for_unlimited = 0
            # Non-blocking polls: the poll loop sleeps between attempts so the
            # SDK lock is never held across a blocking wait (DESIGN §3).
            cam.image_poll_timeout_ms = 0

            self._register_controls()

    def _register_controls(self) -> None:
        """Probe the hardware and register the scalar controls it supports.

        Called with the lock held, camera open and disarmed.
        """
        cam = self._cam

        # exposure — integer microseconds, live-settable, log-scale slider.
        self._register(ControlSpec(
            name="exposure", units="us", kind="int", scale="log", step=1,
            get_range=lambda: self._locked(lambda: (
                float(cam.exposure_time_range_us.min),
                float(cam.exposure_time_range_us.max))),
            getter=lambda: self._locked(lambda: float(cam.exposure_time_us)),
            setter=lambda v: self._locked(lambda: setattr(cam, "exposure_time_us", int(v))),
            display={"label": "exposure", "unit_display": "ms", "unit_scale": 1e-3},
        ))

        # gain — native integer index; display hints carry dB endpoints.
        try:
            g = cam.gain_range
            if g.max > g.min:
                try:
                    db_lo = float(cam.convert_gain_to_decibels(int(g.min)))
                    db_hi = float(cam.convert_gain_to_decibels(int(g.max)))
                except Exception:
                    db_lo, db_hi = 0.0, 0.0
                self._register(ControlSpec(
                    name="gain", units="index", kind="int", step=1,
                    get_range=lambda: self._locked(lambda: (
                        float(cam.gain_range.min), float(cam.gain_range.max))),
                    getter=lambda: self._locked(lambda: float(cam.gain)),
                    setter=lambda v: self._locked(lambda: setattr(cam, "gain", int(v))),
                    display={"label": "gain", "db_lo": db_lo, "db_hi": db_hi},
                ))
        except Exception:
            pass

        # fps — frame-rate control. Enable exactly ONCE here (disarmed); if the
        # enable fails or reads back False, don't register fps at all. The live
        # set path writes only the value, never the enable flag.
        try:
            rng = cam.frame_rate_control_value_range
            if rng.max > 0:
                cam.is_frame_rate_control_enabled = True
                if bool(cam.is_frame_rate_control_enabled):
                    self._register(ControlSpec(
                        name="fps", units="fps", kind="float",
                        get_range=lambda: self._locked(lambda: (
                            max(float(cam.frame_rate_control_value_range.min), 1.0),
                            float(cam.frame_rate_control_value_range.max))),
                        getter=lambda: self._locked(
                            lambda: float(cam.frame_rate_control_value)),
                        setter=self._set_fps_native,
                        display={"label": "target fps"},
                    ))
        except Exception:
            pass

        # black level — SDK signals absence via range max == 0 (error 1002 is
        # swallowed into Range(0,0) inside the SDK; there is no exception).
        try:
            bl = cam.black_level_range
            if bl.max > 0:
                self._register(ControlSpec(
                    name="black_level", units="ADU", kind="int", step=1,
                    get_range=lambda: self._locked(lambda: (
                        float(cam.black_level_range.min),
                        float(cam.black_level_range.max))),
                    getter=lambda: self._locked(lambda: float(cam.black_level)),
                    setter=lambda v: self._locked(lambda: setattr(cam, "black_level", int(v))),
                    display={"label": "black level"},
                ))
        except Exception:
            pass

    def _set_fps_native(self, v: float) -> None:
        with self._lock:
            self._cam.frame_rate_control_value = float(v)

    def _locked(self, fn):
        with self._lock:
            return fn()

    def disconnect(self) -> None:
        with self._lock:
            try:
                if self._cam is not None:
                    try:
                        if self._cam.is_armed:
                            self._cam.disarm()
                    except Exception:
                        pass
                    self._cam.dispose()
            finally:
                self._cam = None
                self._dispose_sdk()
                for name in list(self._controls):
                    self._unregister(name)

    def _dispose_sdk(self) -> None:
        if self._sdk is None:
            return
        try:
            self._sdk.dispose()
        except Exception:
            # 0.0.8: dispose() raises without clearing the singleton flag — clear
            # it ourselves so the next connect() can recreate the SDK.
            try:
                type(self._sdk)._is_sdk_open = False
            except Exception:
                pass
        finally:
            self._sdk = None

    @property
    def is_connected(self) -> bool:
        return self._cam is not None

    def _require(self):
        if self._cam is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")
        return self._cam

    # ------------------------------------------------------------------ #
    # device info
    # ------------------------------------------------------------------ #
    @property
    def device_info(self) -> dict:
        cam = self._require()
        with self._lock:
            return {"model": cam.model, "serial": cam.serial_number,
                    "vendor": "Thorlabs", "sdk": "thorlabs_tsi_sdk 0.0.8"}

    @property
    def bit_depth(self) -> int:
        with self._lock:
            return int(self._require().bit_depth)

    def sensor_size(self) -> Tuple[int, int]:
        cam = self._require()
        with self._lock:
            return int(cam.sensor_width_pixels), int(cam.sensor_height_pixels)

    def image_size(self) -> Tuple[int, int]:
        cam = self._require()
        with self._lock:
            return int(cam.image_width_pixels), int(cam.image_height_pixels)

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def roi_range(self) -> dict:
        cam = self._require()
        with self._lock:
            rr = cam.roi_range
            bx, by = int(cam.binx), int(cam.biny)
            # The SDK's ROIRange has no w/h minimums; the image-size ranges do
            # (post-binning px -> multiply back into sensor coords).
            try:
                w_min = int(cam.image_width_range_pixels.min) * bx
                h_min = int(cam.image_height_range_pixels.min) * by
            except Exception:
                w_min, h_min = 80, 4
            return {
                "w_min": w_min, "h_min": h_min,
                "w_max": int(cam.sensor_width_pixels),
                "h_max": int(cam.sensor_height_pixels),
                "w_inc": _ROI_INC, "h_inc": _ROI_INC,
                "x_inc": _ROI_INC, "y_inc": _ROI_INC,
                "min_lrx": int(rr.lower_right_x_pixels_min),
                "min_lry": int(rr.lower_right_y_pixels_min),
                "max_ulx": int(rr.upper_left_x_pixels_max),
                "max_uly": int(rr.upper_left_y_pixels_max),
            }

    def _clamp_roi(self, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        """Best-effort pre-clamp to the advertised rules (read-back still wins).

        Precedence per DESIGN §2.2: offsets by ul maxes -> corners by lr mins ->
        enforce w/h minimums (pulling the origin back so the minimum fits).
        """
        r = self.roi_range()
        sw, sh = r["w_max"], r["h_max"]
        w = max(int(w), r["w_min"])
        h = max(int(h), r["h_min"])
        x = min(max(int(x), 0), min(r["max_ulx"], sw - w))
        y = min(max(int(y), 0), min(r["max_uly"], sh - h))
        w = min(w, sw - x)
        h = min(h, sh - y)
        # inclusive corners
        ulx = x - (x % r["x_inc"])
        uly = y - (y % r["y_inc"])
        lrx = min(ulx + w - 1, sw - 1)
        lry = min(uly + h - 1, sh - 1)
        lrx = max(lrx, r["min_lrx"])
        lry = max(lry, r["min_lry"])
        return ulx, uly, lrx, lry

    def set_roi(self, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
        """Apply a sensor-coordinate ROI atomically; return the APPLIED roi.

        Holds the SDK lock across disarm -> set -> read-back -> re-arm ->
        re-trigger -> re-apply(fps, exposure), so no getter/setter can land in
        the half-configured window. Restores streaming if it was running.
        """
        cam = self._require()
        with self._lock:
            was_armed = bool(cam.is_armed)
            if was_armed:
                cam.disarm()
            ulx, uly, lrx, lry = self._clamp_roi(x, y, w, h)
            roi_type = type(cam.roi)
            cam.roi = roi_type(ulx, uly, lrx, lry)
            applied = self.get_roi()
            if was_armed:
                self._arm_and_trigger()
                self._reapply_after_rearm()
            return applied

    def get_roi(self) -> Tuple[int, int, int, int]:
        cam = self._require()
        with self._lock:
            r = cam.roi
            return (int(r.upper_left_x_pixels), int(r.upper_left_y_pixels),
                    int(r.lower_right_x_pixels - r.upper_left_x_pixels + 1),
                    int(r.lower_right_y_pixels - r.upper_left_y_pixels + 1))

    def reset_roi(self) -> Tuple[int, int, int, int]:
        sw, sh = self.sensor_size()
        # Explicit full-sensor rect — roi=None wedges the camera until reconnect.
        return self.set_roi(0, 0, sw, sh)

    def binning_range(self) -> Tuple[int, int]:
        cam = self._require()
        with self._lock:
            return int(cam.binx_range.max), int(cam.biny_range.max)

    def set_binning(self, bx: int, by: int) -> Tuple[int, int]:
        cam = self._require()
        with self._lock:
            was_armed = bool(cam.is_armed)
            if was_armed:
                cam.disarm()
            cam.binx = int(min(max(bx, cam.binx_range.min), cam.binx_range.max))
            cam.biny = int(min(max(by, cam.biny_range.min), cam.biny_range.max))
            applied = self.get_binning()
            if was_armed:
                self._arm_and_trigger()
                self._reapply_after_rearm()
            return applied

    def get_binning(self) -> Tuple[int, int]:
        cam = self._require()
        with self._lock:
            return int(cam.binx), int(cam.biny)

    def _reapply_after_rearm(self) -> None:
        """fps and exposure do not survive re-arm; re-issue the last-commanded
        values through the funnel (clamps against fresh post-arm ranges)."""
        for name in ("fps", "exposure"):     # fps first — exposure duty depends on it
            v = self.commanded(name)
            if v is not None:
                self.set(name, v)

    # ------------------------------------------------------------------ #
    # fps/exposure coupling — the sole authority (DESIGN §2.1)
    # ------------------------------------------------------------------ #
    def set(self, name: str, value: float, *, normalized: bool = False) -> ControlValue:
        # Hold the SDK lock across the whole clamp+write+read-back (and the
        # coupling adjustments below) so concurrent setters can't interleave
        # between the range read, the write, and the echo. RLock — the funnel's
        # internal getter/setter calls re-enter freely.
        with self._lock:
            return self._set_coupled(name, value, normalized)

    def _set_coupled(self, name: str, value: float, normalized: bool) -> ControlValue:
        result = super().set(name, value, normalized=normalized)
        if not result.ok:
            return result
        if name == "fps":
            # Exposure must fit inside the new frame period or it silently caps
            # the achievable rate — reduce it and surface the change.
            fps = result.value or 0.0
            exp = self.get("exposure").value
            if fps > 0 and exp:
                limit_us = _EXPOSURE_DUTY_MAX * 1e6 / fps
                if exp > limit_us:
                    eff = super().set("exposure", limit_us)
                    result.warning = (f"exposure reduced to "
                                      f"{(eff.value or limit_us)/1000:.2f} ms to fit "
                                      f"{fps:.1f} fps")
                    result.side_effects["exposure"] = eff.as_dict()
        elif name == "exposure":
            fps = self.get("fps").value
            exp = result.value or 0.0
            if fps and exp > _EXPOSURE_DUTY_MAX * 1e6 / fps:
                result.warning = (f"exposure {exp/1000:.2f} ms exceeds the "
                                  f"{fps:.1f} fps frame period — achieved rate will drop")
        return result

    # ------------------------------------------------------------------ #
    # acquisition
    # ------------------------------------------------------------------ #
    def _arm_and_trigger(self) -> None:
        cam = self._cam
        cam.frames_per_trigger_zero_for_unlimited = 0   # re-set before EVERY arm
        cam.arm(20 if self._max_throughput else 2)
        cam.issue_software_trigger()

    def start(self, max_throughput: bool = False) -> None:
        cam = self._require()
        with self._lock:
            self._max_throughput = bool(max_throughput)
            if not cam.is_armed:
                self._arm_and_trigger()
                self._reapply_after_rearm()

    def stop(self) -> None:
        cam = self._require()
        with self._lock:
            if cam.is_armed:
                cam.disarm()

    @property
    def is_grabbing(self) -> bool:
        cam = self._cam
        if cam is None:
            return False
        with self._lock:
            return bool(cam.is_armed)

    def grab(self, timeout_ms: int = 5000) -> Frame:
        """Return the next frame from the running stream (poll + copy + meta).

        Non-blocking SDK polls under the lock, ~1 ms sleeps outside it, so
        control threads interleave freely with acquisition.
        """
        cam = self._require()
        deadline = perf_counter() + timeout_ms / 1000.0
        while True:
            with self._lock:
                if self._cam is None:
                    raise RuntimeError("camera disconnected")
                f = cam.get_pending_frame_or_null()
                if f is not None:
                    data = np.copy(f.image_buffer)      # SDK-owned; copy before escape
                    hw_count = int(f.frame_count)
                    ts = f.time_stamp_relative_ns_or_null
                    hw_ts = float(ts) if ts is not None else None
            if f is not None:
                return Frame(data=data,
                             meta=FrameMeta(t_sw=perf_counter(),
                                            hw_count=hw_count,
                                            hw_timestamp_ns=hw_ts))
            if perf_counter() >= deadline:
                raise TimeoutError("Timed out waiting for a frame from the Zelux.")
            if not self.is_grabbing:
                raise RuntimeError("acquisition stopped")
            sleep(0.001)
