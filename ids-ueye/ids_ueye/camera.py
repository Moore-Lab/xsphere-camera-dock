"""Driver/API for IDS uEye cameras (e.g. DCC1545M-GL, USB2, mono).

Wraps the IDS uEye SDK (``pyueye``) behind the same small, GUI-friendly interface
the other camera modules expose, so that ``xsphere-camera-dock`` can drive every
camera through identical calls (``connect`` / ``disconnect`` / ``set_exposure`` /
``set_frame_rate`` / ``grab`` / ``frames``). The public surface here mirrors
``ZeluxCS165MU`` / ``BaslerACA1440``.

The capture core is a port of the hardware-tested ``IDSCamera`` adapter from
``reference/dualcam_fast.py`` (validated at 200 fps in the PyQt dual-camera GUI):
freerun capture with a 12-buffer ring, event-driven frame waits
(``is_WaitEvent``), sequence-number dedup, pitch-stripped Mono8 frames, and the
full stop/free/AOI/realloc/restart sequence for ROI changes. What changed for the
dock is only the lifecycle: the dock's engine owns start/stop, so ``set_roi``
reconfigures without restarting capture itself, and ``start()`` re-applies the
cached frame rate (``is_SetFrameRate`` does **not** survive a re-arm).

Backend notes
-------------
* ``pyueye`` is on PyPI, but it is only bindings: the IDS uEye driver stack
  (IDS Software Suite) must be installed for ``ueye_api.dll`` to load. The import
  is deferred until :meth:`connect`, so ``import ids_ueye`` works everywhere.
* Frames are Mono8 ``uint8`` (the dock's ``bit_depth`` handling treats this like
  the Basler).

Frame delivery semantics
------------------------
``is_GetActSeqBuf`` hands back the *most recently filled* ring buffer. With the
event-driven wait this delivers every frame as long as the consumer keeps up with
the frame period (the dock's acquisition thread does nothing but wait + copy);
if the host ever stalls longer than one period, intermediate frames are skipped
rather than queued. This matches the tested reference design, which accounted
for genuine drops via the hardware frame counter (``is_GetImageInfo``, exposed
here through ``capture_hw_info`` / ``last_frame_meta``) instead of a strict FIFO.

Exposure/frame-rate coupling
----------------------------
The uEye driver clamps exposure to the current frame period internally: set the
frame rate first, then exposure (the dock's defaults/presets/preview apply in
that order for this reason). Setting a long exposure caps the achievable rate.
"""

from __future__ import annotations

import ctypes
from time import perf_counter
from typing import Iterator, List, Optional, Tuple

import numpy as np

# uEye image-queue ring depth. Hardware-tested at 12 in the reference GUI: deep
# enough to absorb host-side dequeue latency at 200 fps without dropping frames.
IDS_RING_BUFFERS = 12


def clamp_roi(x, y, w, h, sensor_w, sensor_h, step_x=1, step_y=1,
              min_w=1, min_h=1):
    """Validate and align ROI parameters to sensor bounds and hardware steps.

    Ported from the tested reference GUI (``reference/dualcam_fast.py``):
    step_x=4, step_y=2 for the IDS DCC1545M-GL. Returns a clamped
    ``(x, y, w, h)`` tuple; raises ``ValueError`` if the result is degenerate.
    """
    if int(w) <= 0 or int(h) <= 0:
        raise ValueError("ROI width and height must be positive")
    if sensor_w < min_w or sensor_h < min_h:
        raise ValueError("Sensor smaller than the minimum ROI size")
    x = max(0, (int(x) // step_x) * step_x)
    y = max(0, (int(y) // step_y) * step_y)
    # Pull the origin back so the minimum size always fits inside the sensor —
    # otherwise a request near the right/bottom edge clamps below the SDK minimum.
    x = min(x, ((sensor_w - min_w) // step_x) * step_x)
    y = min(y, ((sensor_h - min_h) // step_y) * step_y)
    w = max(min_w, (int(w) // step_x) * step_x)
    h = max(min_h, (int(h) // step_y) * step_y)
    w = min(w, sensor_w - x)
    h = min(h, sensor_h - y)
    if w < min_w or h < min_h:
        raise ValueError("ROI is outside sensor bounds")
    return x, y, w, h


def reshape_ids_frame(raw_bytes, width, height, pitch):
    """Strip row-padding from a uEye frame buffer; return (height, width) uint8."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    if arr.size != height * pitch:
        raise ValueError(
            f"Buffer size {arr.size} does not match height*pitch ({height}*{pitch}={height * pitch})")
    return arr.reshape(height, pitch)[:, :width].copy()


def list_devices() -> List[dict]:
    """Enumerate connected uEye cameras without opening any.

    Returns a list of ``{"device_id", "model", "serial", "in_use"}`` dicts
    (empty if the SDK is unavailable or no cameras are connected).
    """
    try:
        from pyueye import ueye
    except Exception:
        return []
    n = ctypes.c_int(0)
    if ueye.is_GetNumberOfCameras(n) != ueye.IS_SUCCESS or n.value <= 0:
        return []
    cam_list = ueye.UEYE_CAMERA_LIST(ueye.UEYE_CAMERA_INFO * n.value)
    cam_list.dwCount = n.value
    if ueye.is_GetCameraList(cam_list) != ueye.IS_SUCCESS:
        return []
    devices = []
    for i in range(n.value):
        info = cam_list.uci[i]
        devices.append({
            "device_id": int(info.dwDeviceID),
            "model": (info.FullModelName or info.Model).decode(
                "ascii", errors="replace").rstrip("\x00"),
            "serial": info.SerNo.decode("ascii", errors="replace").rstrip("\x00"),
            "in_use": bool(info.dwInUse),
        })
    return devices


class IDSUEye:
    """Thin driver around a single IDS uEye camera.

    Parameters
    ----------
    device_id:
        uEye device ID of the target camera (see :func:`list_devices`). If
        ``None``, the first camera not already in use is opened.
    """

    IS_USE_DEVICE_ID = 0x8000
    bit_depth = 8  # Mono8; class attribute like the frames' dtype, fixed

    def __init__(self, device_id: Optional[int] = None,
                 n_buffers: int = IDS_RING_BUFFERS) -> None:
        self.device_id = device_id
        self._ue = None                    # pyueye.ueye module (set on connect)
        self._hCam = None
        self._n_buffers = max(3, int(n_buffers))
        self._mem_ptrs: list = []
        self._mem_ids: list = []
        self._pitch = 0
        self._current_w = 0
        self._current_h = 0
        self.sensor_width_pixels = 0
        self.sensor_height_pixels = 0
        self.model = ""
        self.serial_number = ""
        self._event_enabled = False
        self._capturing = False
        self._last_seq_num = -1
        # Last commanded frame rate; re-applied on every start() because
        # is_SetFrameRate does not survive a capture re-arm. None = never set.
        self._frame_rate: Optional[float] = None
        # When True, capture per-frame hardware info (is_GetImageInfo) into
        # last_frame_meta — enabled only for recording so the free-run path
        # pays no extra per-frame cost.
        self.capture_hw_info = False
        self.last_frame_meta: Optional[Tuple] = None  # (frame_number, timestamp_ns, buffers_in_use)

    # --- lifecycle ---------------------------------------------------------
    def connect(self) -> None:
        """Open the camera, set Mono8, allocate the ring, enable the frame event."""
        try:
            from pyueye import ueye
        except Exception as exc:
            raise RuntimeError(
                "Could not load the IDS uEye SDK (pyueye). Install pyueye "
                "(pip install pyueye) and the IDS Software Suite (uEye driver, "
                "provides ueye_api.dll). Original error: " + str(exc)
            ) from exc
        if self._hCam is not None:
            self.disconnect()   # re-entry safe: a /connect retry starts clean
        self._ue = ueye
        self._last_seq_num = -1

        device_id = self.device_id
        if device_id is None:
            devices = list_devices()
            free = [d for d in devices if not d["in_use"]]
            if not free:
                if devices:
                    raise RuntimeError(
                        "All uEye cameras are in use (close IDS Camera Manager / "
                        "uEye Cockpit first).")
                raise RuntimeError("No IDS uEye cameras found.")
            device_id = free[0]["device_id"]

        self._hCam = ctypes.c_uint(int(device_id) | self.IS_USE_DEVICE_ID)
        ret = ueye.is_InitCamera(self._hCam, None)
        if ret != ueye.IS_SUCCESS:
            self._hCam = None
            raise RuntimeError(f"is_InitCamera failed (device {device_id}): {ret}")
        self.device_id = int(device_id)

        try:
            sinfo = ueye.SENSORINFO()
            ueye.is_GetSensorInfo(self._hCam, sinfo)
            self.sensor_width_pixels = int(sinfo.nMaxWidth)
            self.sensor_height_pixels = int(sinfo.nMaxHeight)
            self.model = sinfo.strSensorName.decode("ascii", errors="replace").rstrip("\x00")

            binfo = ueye.BOARDINFO()
            if ueye.is_GetCameraInfo(self._hCam, binfo) == ueye.IS_SUCCESS:
                self.serial_number = binfo.SerNo.decode("ascii", errors="replace").rstrip("\x00")
            else:
                self.serial_number = "unknown"

            ret = ueye.is_SetColorMode(self._hCam, ueye.IS_CM_MONO8)
            if ret != ueye.IS_SUCCESS:
                raise RuntimeError(f"is_SetColorMode(MONO8) failed: {ret}")

            self._current_w = self.sensor_width_pixels
            self._current_h = self.sensor_height_pixels
            self._alloc_ring(self._current_w, self._current_h)

            ret = ueye.is_EnableEvent(self._hCam, ueye.IS_SET_EVENT_FRAME)
            if ret != ueye.IS_SUCCESS:
                raise RuntimeError(f"is_EnableEvent(FRAME) failed: {ret}")
            self._event_enabled = True
        except Exception:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        """Stop capture and release the camera. Best-effort; never raises."""
        ue, h = self._ue, self._hCam
        if ue is None or h is None:
            self._hCam = None
            return
        try:
            if self._capturing:
                ue.is_StopLiveVideo(h, ue.IS_FORCE_VIDEO_STOP)
        except Exception as exc:
            print(f"IDSUEye.disconnect StopLiveVideo: {exc}")
        self._capturing = False
        try:
            if self._event_enabled:
                ue.is_DisableEvent(h, ue.IS_SET_EVENT_FRAME)
        except Exception as exc:
            print(f"IDSUEye.disconnect DisableEvent: {exc}")
        self._event_enabled = False
        try:
            self._free_ring()
        except Exception as exc:
            print(f"IDSUEye.disconnect FreeImageMem: {exc}")
        try:
            ue.is_ExitCamera(h)
        except Exception as exc:
            print(f"IDSUEye.disconnect ExitCamera: {exc}")
        self._hCam = None

    @property
    def is_connected(self) -> bool:
        return self._hCam is not None

    def _require(self):
        if self._hCam is None:
            raise RuntimeError("Camera is not connected. Call connect() first.")
        return self._ue

    # --- ring buffer management ---------------------------------------------
    def _alloc_ring(self, w: int, h: int) -> None:
        """Allocate ``_n_buffers`` Mono8 image buffers, queue them, cache pitch.

        Atomic: on any failure the buffers already allocated are freed again, so
        the ring is never left partially built.
        """
        ue = self._ue
        try:
            for i in range(self._n_buffers):
                mem_ptr = ue.c_mem_p()
                mem_id = ctypes.c_int()
                ret = ue.is_AllocImageMem(self._hCam, w, h, 8, mem_ptr, mem_id)
                if ret != ue.IS_SUCCESS:
                    raise RuntimeError(f"is_AllocImageMem failed on buffer {i}: {ret}")
                ret = ue.is_AddToSequence(self._hCam, mem_ptr, mem_id)
                if ret != ue.IS_SUCCESS:
                    ue.is_FreeImageMem(self._hCam, mem_ptr, mem_id)
                    raise RuntimeError(f"is_AddToSequence failed on buffer {i}: {ret}")
                self._mem_ptrs.append(mem_ptr)
                self._mem_ids.append(mem_id)
        except Exception:
            self._free_ring()
            raise
        pitch = ctypes.c_int()
        ue.is_GetImageMemPitch(self._hCam, pitch)
        self._pitch = pitch.value

    def _free_ring(self) -> None:
        """Dequeue and free all image buffers."""
        ue = self._ue
        ue.is_ClearSequence(self._hCam)
        for mem_ptr, mem_id in zip(self._mem_ptrs, self._mem_ids):
            ue.is_FreeImageMem(self._hCam, mem_ptr, mem_id)
        self._mem_ptrs.clear()
        self._mem_ids.clear()

    # --- device info ---------------------------------------------------------
    @property
    def device_info(self) -> dict:
        self._require()
        return {"model": self.model, "serial": self.serial_number, "vendor": "IDS"}

    def sensor_size(self) -> Tuple[int, int]:
        """Return ``(width, height)`` of the current image region (AOI)."""
        self._require()
        return self._current_w, self._current_h

    # --- exposure (protocol unit: microseconds; SDK unit: milliseconds) -----
    def set_exposure(self, microseconds: float) -> None:
        """Set exposure time in microseconds (clamped to the current valid range).

        Note the uEye range depends on the current frame rate and pixel clock;
        the driver additionally clamps internally to fit the frame period.
        """
        ue = self._require()
        lo, hi = self.exposure_range()
        if hi > lo:
            microseconds = min(max(microseconds, lo), hi)
        exp_ms = ctypes.c_double(microseconds / 1000.0)
        ret = ue.is_Exposure(self._hCam, ue.IS_EXPOSURE_CMD_SET_EXPOSURE,
                             exp_ms, ctypes.sizeof(exp_ms))
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_Exposure(SET, {microseconds:g}us) failed: {ret}")

    def get_exposure(self) -> float:
        ue = self._require()
        exp_ms = ctypes.c_double()
        ret = ue.is_Exposure(self._hCam, ue.IS_EXPOSURE_CMD_GET_EXPOSURE,
                             exp_ms, ctypes.sizeof(exp_ms))
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_Exposure(GET) failed: {ret}")
        return exp_ms.value * 1000.0

    def exposure_range(self) -> Tuple[float, float]:
        """``(min, max)`` exposure in microseconds at the current frame rate."""
        ue = self._require()
        rng = (ctypes.c_double * 3)()  # min, max, increment (ms)
        ret = ue.is_Exposure(self._hCam, ue.IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE,
                             rng, ctypes.sizeof(rng))
        if ret != ue.IS_SUCCESS:
            return 0.0, 0.0
        return rng[0] * 1000.0, rng[1] * 1000.0

    # --- frame rate ----------------------------------------------------------
    def set_frame_rate(self, fps: float) -> None:
        """Command a freerun frame rate (the SDK snaps to the nearest achievable).

        The commanded value is cached and re-applied on every :meth:`start`,
        because the uEye freerun rate resets on re-arm.
        """
        ue = self._require()
        # Cache the RAW commanded value (matching the tested reference): the
        # achievable range varies with AOI/pixel clock, so clamping the cache
        # would permanently degrade the request after a full-sensor excursion.
        self._frame_rate = float(fps)
        applied = float(fps)
        lo, hi = self.frame_rate_range()
        if hi > 0:
            applied = min(max(applied, lo), hi)
        new_fps = ctypes.c_double()
        ret = ue.is_SetFrameRate(self._hCam, ctypes.c_double(applied), new_fps)
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_SetFrameRate({applied:g}) failed: {ret}")

    def get_frame_rate(self) -> float:
        """Target frame rate (what we asked for; SDK-reported if never set)."""
        ue = self._require()
        if self._frame_rate is not None:
            return self._frame_rate
        fps = ctypes.c_double()
        ret = ue.is_SetFrameRate(self._hCam, ue.IS_GET_FRAMERATE, fps)
        return fps.value if ret == ue.IS_SUCCESS else 0.0

    def frame_rate_range(self) -> Tuple[float, float]:
        """``(fps_min, fps_max)`` achievable at the current pixel clock / AOI.

        ``is_GetFrameTimeRange`` returns frame *times* (seconds), so the fps
        bounds are the inverse. Returns ``(0.0, 0.0)`` on failure (callers treat
        ``fps_max <= 0`` as "no usable range").
        """
        ue = self._require()
        t_min = ctypes.c_double()
        t_max = ctypes.c_double()
        t_int = ctypes.c_double()
        ret = ue.is_GetFrameTimeRange(self._hCam, t_min, t_max, t_int)
        if ret != ue.IS_SUCCESS or t_min.value <= 0 or t_max.value <= 0:
            return 0.0, 0.0
        return 1.0 / t_max.value, 1.0 / t_min.value

    def resulting_frame_rate(self) -> float:
        """Live measured frame rate; falls back to the target when not streaming."""
        ue = self._require()
        fps = ctypes.c_double()
        ret = ue.is_GetFramesPerSecond(self._hCam, fps)
        if ret == ue.IS_SUCCESS and fps.value > 0:
            return fps.value
        return self.get_frame_rate()

    # --- gain (not exposed for uEye; feature-detected off via the range) ----
    def set_gain(self, value: float) -> None:
        pass

    def get_gain(self) -> float:
        return 0.0

    def gain_range(self) -> Tuple[float, float]:
        return 0.0, 0.0

    # --- region of interest / binning ----------------------------------------
    def roi_range(self) -> dict:
        """AOI limits/steps queried from the SDK (reference DCC1545M values as fallback)."""
        ue = self._require()
        size_min = ue.IS_SIZE_2D()
        size_inc = ue.IS_SIZE_2D()
        pos_inc = ue.IS_POINT_2D()
        w_min, h_min, w_inc, h_inc, x_inc, y_inc = 32, 4, 4, 2, 4, 2
        if ue.is_AOI(self._hCam, ue.IS_AOI_IMAGE_GET_SIZE_MIN,
                     size_min, ctypes.sizeof(size_min)) == ue.IS_SUCCESS:
            w_min, h_min = int(size_min.s32Width), int(size_min.s32Height)
        if ue.is_AOI(self._hCam, ue.IS_AOI_IMAGE_GET_SIZE_INC,
                     size_inc, ctypes.sizeof(size_inc)) == ue.IS_SUCCESS:
            w_inc, h_inc = int(size_inc.s32Width), int(size_inc.s32Height)
        if ue.is_AOI(self._hCam, ue.IS_AOI_IMAGE_GET_POS_INC,
                     pos_inc, ctypes.sizeof(pos_inc)) == ue.IS_SUCCESS:
            x_inc, y_inc = int(pos_inc.s32X), int(pos_inc.s32Y)
        return {
            "w_min": w_min, "w_max": self.sensor_width_pixels, "w_inc": w_inc,
            "h_min": h_min, "h_max": self.sensor_height_pixels, "h_inc": h_inc,
            "x_inc": x_inc, "y_inc": y_inc,
        }

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        """Set the AOI: stop capture, free the ring, set AOI, realloc, restore.

        The dock calls this with acquisition stopped (engine stopped); if capture
        happens to be running it is stopped and restarted around the change. The
        commanded frame rate is re-applied by :meth:`start` after the re-arm.
        """
        ue = self._require()
        rng = self.roi_range()
        x, y, w, h = clamp_roi(
            x, y, w, h, self.sensor_width_pixels, self.sensor_height_pixels,
            step_x=max(rng["x_inc"], rng["w_inc"]), step_y=max(rng["y_inc"], rng["h_inc"]),
            min_w=rng["w_min"], min_h=rng["h_min"])

        was_capturing = self._capturing
        if was_capturing:
            self.stop()

        self._free_ring()
        try:
            rect = ue.IS_RECT()
            rect.s32X, rect.s32Y, rect.s32Width, rect.s32Height = x, y, w, h
            ret = ue.is_AOI(self._hCam, ue.IS_AOI_IMAGE_SET_AOI, rect, ctypes.sizeof(rect))
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_AOI(SET {x},{y},{w}x{h}) failed: {ret}")
            self._alloc_ring(w, h)
            self._current_w, self._current_h = w, h
        except Exception:
            # Restore full-sensor AOI + a usable ring so the camera stays
            # operable; best-effort (an already-raised error stays primary).
            try:
                full = ue.IS_RECT()
                full.s32X, full.s32Y = 0, 0
                full.s32Width, full.s32Height = self.sensor_width_pixels, self.sensor_height_pixels
                ue.is_AOI(self._hCam, ue.IS_AOI_IMAGE_SET_AOI, full, ctypes.sizeof(full))
                self._alloc_ring(self.sensor_width_pixels, self.sensor_height_pixels)
                self._current_w, self._current_h = (self.sensor_width_pixels,
                                                    self.sensor_height_pixels)
                if was_capturing:
                    self.start()
            except Exception as exc:
                print(f"IDSUEye: full-sensor restore after failed ROI change: {exc}")
            raise

        self._last_seq_num = -1
        if was_capturing:
            self.start()

    def get_roi(self) -> Tuple[int, int, int, int]:
        ue = self._require()
        rect = ue.IS_RECT()
        ret = ue.is_AOI(self._hCam, ue.IS_AOI_IMAGE_GET_AOI, rect, ctypes.sizeof(rect))
        if ret != ue.IS_SUCCESS:
            return 0, 0, self._current_w, self._current_h
        return int(rect.s32X), int(rect.s32Y), int(rect.s32Width), int(rect.s32Height)

    def reset_roi(self) -> None:
        """Restore the full-sensor AOI."""
        self.set_roi(0, 0, self.sensor_width_pixels, self.sensor_height_pixels)

    def binning_range(self) -> Tuple[int, int]:
        return 1, 1  # not exposed for uEye (use ROI + pixel clock for rate)

    def set_binning(self, bx: int, by: int) -> None:
        pass

    def get_binning(self) -> Tuple[int, int]:
        return 1, 1

    # --- pixel clock (uEye-specific optional capability) ---------------------
    def get_pixel_clock_list(self) -> List[int]:
        """Sorted list of supported pixel clock values (MHz)."""
        ue = self._require()
        n = ctypes.c_uint()
        ue.is_PixelClock(self._hCam, ue.IS_PIXELCLOCK_CMD_GET_NUMBER,
                         n, ctypes.sizeof(n))
        if n.value == 0:
            return []
        arr = (ctypes.c_uint * n.value)()
        ue.is_PixelClock(self._hCam, ue.IS_PIXELCLOCK_CMD_GET_LIST,
                         arr, n.value * ctypes.sizeof(ctypes.c_uint()))
        return sorted(set(int(v) for v in arr))

    def get_pixel_clock(self) -> int:
        ue = self._require()
        val = ctypes.c_uint()
        ue.is_PixelClock(self._hCam, ue.IS_PIXELCLOCK_CMD_GET, val, ctypes.sizeof(val))
        return int(val.value)

    def set_pixel_clock(self, mhz: int) -> None:
        """Set the pixel clock (MHz). Changes the achievable frame-rate range."""
        ue = self._require()
        val = ctypes.c_uint(int(mhz))
        ret = ue.is_PixelClock(self._hCam, ue.IS_PIXELCLOCK_CMD_SET,
                               val, ctypes.sizeof(val))
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_PixelClock SET {mhz} MHz failed: {ret}")

    # --- acquisition ----------------------------------------------------------
    def start(self, max_throughput: bool = False) -> None:
        """Begin freerun capture (trigger off + live video).

        ``max_throughput`` is accepted for protocol compatibility; the ring is
        already sized (12 buffers) for full-rate no-stall capture, matching the
        hardware-tested reference configuration.
        """
        ue = self._require()
        if self._capturing:
            return
        if not self._event_enabled:
            ret = ue.is_EnableEvent(self._hCam, ue.IS_SET_EVENT_FRAME)
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_EnableEvent(FRAME) failed: {ret}")
            self._event_enabled = True
        ret = ue.is_SetExternalTrigger(self._hCam, ue.IS_SET_TRIGGER_OFF)
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_SetExternalTrigger(OFF/freerun) failed: {ret}")
        ret = ue.is_CaptureVideo(self._hCam, ue.IS_DONT_WAIT)
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_CaptureVideo failed: {ret}")
        self._capturing = True
        self._last_seq_num = -1
        # Re-apply the commanded frame rate — the freerun rate resets on re-arm.
        if self._frame_rate is not None:
            try:
                self.set_frame_rate(self._frame_rate)
            except Exception as exc:
                print(f"IDSUEye: frame-rate re-apply after start failed: {exc}")

    def stop(self) -> None:
        """Stop freerun capture (the camera stays configured and connected).

        Also disables the frame event (mirroring the reference's disarm): a frame
        completing between the consumer's last wait and the stop leaves the event
        signaled, and a stale signal surviving into the next start would make the
        first wait fire before any frame exists in the (possibly re-allocated)
        ring. Disable/re-enable destroys and recreates the SDK event unsignaled.
        """
        ue = self._require()
        if not self._capturing:
            return
        self._capturing = False   # aborts any in-flight _poll within one chunk
        ue.is_StopLiveVideo(self._hCam, ue.IS_FORCE_VIDEO_STOP)
        if self._event_enabled:
            ue.is_DisableEvent(self._hCam, ue.IS_SET_EVENT_FRAME)
            self._event_enabled = False

    @property
    def is_grabbing(self) -> bool:
        return self._hCam is not None and self._capturing

    def _wait_frame(self, chunk_ms: int = 200) -> Optional[np.ndarray]:
        """Wait up to ``chunk_ms`` for the next frame event; return it or ``None``.

        ``None`` means either the wait timed out or the event fired for a frame
        we already delivered (sequence-number dedup) — callers just wait again.
        """
        ue = self._ue
        ret = ue.is_WaitEvent(self._hCam, ue.IS_SET_EVENT_FRAME, chunk_ms)
        if ret == ue.IS_TIMED_OUT:
            return None
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_WaitEvent(FRAME) error: {ret}")

        nNum = ctypes.c_int()
        pcMem = ue.c_mem_p()
        pcMemLast = ue.c_mem_p()
        ret = ue.is_GetActSeqBuf(self._hCam, nNum, pcMem, pcMemLast)
        if ret != ue.IS_SUCCESS:
            return None
        if not pcMemLast.value:
            return None   # no frame transferred yet (fresh sequence)
        if nNum.value == self._last_seq_num:
            return None
        self._last_seq_num = nNum.value

        w, h = self._current_w, self._current_h
        raw = ue.get_data(pcMemLast, w, h, 8, self._pitch, copy=True)
        frame = reshape_ids_frame(raw, w, h, self._pitch)
        self.last_frame_meta = self._read_image_info(nNum.value)
        ue.is_UnlockSeqBuf(self._hCam, nNum, pcMemLast)
        return frame

    def _read_image_info(self, mem_id) -> Optional[Tuple[int, int, int]]:
        """Per-frame ``(hw_frame_number, hw_timestamp_ns, buffers_in_use)``.

        Queried only when :attr:`capture_hw_info` is set (recording) so the
        free-run path pays no per-frame cost. Never raises; ``None`` on failure.
        """
        if not self.capture_hw_info:
            return None
        ue = self._ue
        try:
            info = ue.UEYEIMAGEINFO()
            ret = ue.is_GetImageInfo(self._hCam, ctypes.c_int(mem_id),
                                     info, ctypes.sizeof(info))
            if ret != ue.IS_SUCCESS:
                return None
            # u64TimestampDevice is in 100 ns ticks (uEye convention) -> ns.
            return (int(info.u64FrameNumber),
                    int(info.u64TimestampDevice) * 100,
                    int(info.dwImageBuffersInUse))
        except Exception:
            return None

    def grab(self, timeout_ms: int = 5000) -> np.ndarray:
        """Grab and return one frame as a 2-D ``uint8`` array.

        Works whether or not continuous capture is running: if stopped, capture
        is briefly started for a one-shot (mirrors the other dock drivers).
        """
        self._require()
        if self._capturing:
            return self._poll(timeout_ms)
        self.start()
        try:
            return self._poll(timeout_ms)
        finally:
            self.stop()

    def _poll(self, timeout_ms: int) -> np.ndarray:
        deadline = perf_counter() + timeout_ms / 1000.0
        while True:
            # Abort promptly if stop() lands from another thread (e.g. the dock
            # engine stopping for an ROI change) — never outlive the capture.
            if not self._capturing:
                raise RuntimeError("Capture stopped while waiting for a frame.")
            frame = self._wait_frame()
            if frame is not None:
                return frame
            if perf_counter() >= deadline:
                raise RuntimeError("Timed out waiting for a frame from the uEye camera.")

    def frames(self, timeout_ms: int = 5000) -> Iterator[np.ndarray]:
        """Yield frames continuously until :meth:`stop` is called.

        ``timeout_ms`` bounds the wait *between* frames; a camera that stalls
        longer raises ``RuntimeError`` (a plain :meth:`stop` ends the iterator
        cleanly).
        """
        self._require()
        self.start()
        deadline = perf_counter() + timeout_ms / 1000.0
        while self._capturing:
            frame = self._wait_frame()
            if frame is not None:
                deadline = perf_counter() + timeout_ms / 1000.0
                yield frame
            elif perf_counter() >= deadline:
                raise RuntimeError("Timed out waiting for a frame from the uEye camera.")

    # --- context manager -----------------------------------------------------
    def __enter__(self) -> "IDSUEye":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.disconnect()
