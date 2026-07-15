#!/usr/bin/env python3

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
import traceback
import ctypes
import threading
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton,
                            QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QSlider,
                            QSpinBox, QDoubleSpinBox, QFileDialog, QGroupBox,
                            QComboBox, QSplitter, QMessageBox, QCheckBox,
                            QTableWidget, QTableWidgetItem, QSizePolicy,
                            QHeaderView, QAbstractItemView, QLineEdit)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QIntValidator

# Import Thorlabs SDK
try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera
    from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE
    print("Successfully imported Thorlabs SDK modules")
except ImportError as e:
    print(f"Error: Thorlabs TSI SDK not found: {e}")
    print("Please install it using:")
    print("pip install thorlabs_tsi_sdk")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error importing Thorlabs SDK: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from pyueye import ueye as _ueye_mod
    _IDS_AVAILABLE = True
    print("pyueye (IDS uEye) loaded successfully")
except ImportError as e:
    _IDS_AVAILABLE = False
    _ueye_mod = None
    print(f"pyueye not available; IDS cameras disabled: {e}")

# JSON video sidecar writer (stdlib-only, best-effort; never raises).
try:
    import video_metadata
except ImportError:
    from cameras import video_metadata

class CameraLabel(QLabel):
    """QLabel subclass that supports click-and-drag repositioning of markup overlays."""

    def __init__(self, cam_id, app, parent=None):
        super().__init__(parent)
        self.cam_id = cam_id
        self.app = app
        self._drag_idx = -1
        self.setMouseTracking(True)

    @property
    def _cam(self):
        return self.app.cameras[self.cam_id]

    def _scale_info(self):
        """Return (scale, x_offset, y_offset) mapping image pixels → label pixels."""
        cam = self._cam
        iw = getattr(cam, 'image_width', None)
        ih = getattr(cam, 'image_height', None)
        if not iw or not ih:
            return None
        lw, lh = self.width(), self.height()
        scale = min(lw / iw, lh / ih)
        x_off = (lw - iw * scale) / 2
        y_off = (lh - ih * scale) / 2
        return scale, x_off, y_off

    def _to_image(self, lx, ly):
        """Convert label pixel coords to image pixel coords."""
        si = self._scale_info()
        if si is None:
            return None, None
        scale, x_off, y_off = si
        return round((lx - x_off) / scale), round((ly - y_off) / scale)

    def _find_overlay(self, lx, ly, threshold=10):
        """Return index of the closest overlay within *threshold* label-pixels, or -1."""
        si = self._scale_info()
        if si is None:
            return -1
        scale, x_off, y_off = si
        best, best_dist = -1, threshold
        for i, ov in enumerate(self._cam.overlays):
            if ov['type'] == 'hline':
                d = abs(ly - (ov['pos'] * scale + y_off))
            elif ov['type'] == 'vline':
                d = abs(lx - (ov['pos'] * scale + x_off))
            elif ov['type'] == 'circle':
                cx_l = ov['center'][0] * scale + x_off
                cy_l = ov['center'][1] * scale + y_off
                r_l  = ov['radius'] * scale
                d = abs(((lx - cx_l) ** 2 + (ly - cy_l) ** 2) ** 0.5 - r_l)
            else:
                continue
            if d < best_dist:
                best_dist, best = d, i
        return best

    def _set_hover_cursor(self, lx, ly):
        idx = self._find_overlay(lx, ly)
        if idx < 0:
            self.setCursor(Qt.CrossCursor)
            return
        ov = self._cam.overlays[idx]
        if ov['type'] == 'hline':
            self.setCursor(Qt.SizeVerCursor)
        elif ov['type'] == 'vline':
            self.setCursor(Qt.SizeHorCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_idx = self._find_overlay(event.x(), event.y())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_idx >= 0 and (event.buttons() & Qt.LeftButton):
            ix, iy = self._to_image(event.x(), event.y())
            if ix is None:
                return
            cam = self._cam
            ov = cam.overlays[self._drag_idx]
            if ov['type'] == 'hline':
                ov['pos'] = max(0, iy)
            elif ov['type'] == 'vline':
                ov['pos'] = max(0, ix)
            elif ov['type'] == 'circle':
                ov['center'] = (max(0, ix), max(0, iy))
            self.app.sync_overlay_to_table_row(cam, self._drag_idx)
        else:
            self._set_hover_cursor(event.x(), event.y())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_idx = -1
        super().mouseReleaseEvent(event)


# BGR color tuples for markup overlays
MARKUP_COLORS = {
    "White":   (255, 255, 255),
    "Red":     (0,   0,   255),
    "Green":   (0,   255, 0),
    "Blue":    (255, 0,   0),
    "Yellow":  (0,   255, 255),
    "Cyan":    (255, 255, 0),
    "Magenta": (255, 0,   255),
}

# ── Pure helper functions (no SDK / Qt dependency; testable without hardware) ──

def clamp_roi(x, y, w, h, sensor_w, sensor_h, step_x=1, step_y=1,
              min_lrx=0, min_lry=0):
    """Validate and align ROI parameters to sensor bounds and hardware step constraints.

    step_x=4, step_y=2 for IDS DCC1545M-GL; step_x=1, step_y=1 for Thorlabs.
    min_lrx/min_lry: SDK minimum lower-right pixel coordinate (e.g. Thorlabs CS165MU
    has lower_right_x_pixels_min=79, lower_right_y_pixels_min=3).
    Returns (x, y, w, h) clamped tuple. Raises ValueError if result is degenerate.
    """
    if int(w) <= 0 or int(h) <= 0:
        raise ValueError("ROI width and height must be positive")
    x = max(0, (int(x) // step_x) * step_x)
    y = max(0, (int(y) // step_y) * step_y)
    w = max(step_x, (int(w) // step_x) * step_x)
    h = max(step_y, (int(h) // step_y) * step_y)
    w = min(w, sensor_w - x)
    h = min(h, sensor_h - y)
    if w <= 0 or h <= 0:
        raise ValueError("ROI is outside sensor bounds")
    # Enforce SDK minimum lower-right coordinate (e.g. Thorlabs lower_right_x_pixels_min)
    if x + w - 1 < min_lrx:
        w = min_lrx - x + 1
        w = ((w + step_x - 1) // step_x) * step_x   # round up to step alignment
        w = min(w, sensor_w - x)
    if y + h - 1 < min_lry:
        h = min_lry - y + 1
        h = ((h + step_y - 1) // step_y) * step_y
        h = min(h, sensor_h - y)
    if w <= 0 or h <= 0:
        raise ValueError("ROI is outside sensor bounds")
    return x, y, w, h


def compute_pacing_delay(next_trigger_time):
    """Return seconds to sleep before the next trigger (≥0)."""
    return max(0.0, next_trigger_time - time.monotonic())


def reshape_ids_frame(raw_bytes, width, height, pitch):
    """Strip row-padding from IDS frame buffer and return (height, width) uint8 array."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8)
    if arr.size != height * pitch:
        raise ValueError(
            f"Buffer size {arr.size} does not match height*pitch ({height}*{pitch}={height*pitch})")
    return arr.reshape(height, pitch)[:, :width].copy()


def build_ids_rect(x, y, w, h):
    """Construct an IS_RECT ctypes struct for use with is_AOI."""
    from pyueye import ueye as _ue
    rect = _ue.IS_RECT()
    rect.s32X = x
    rect.s32Y = y
    rect.s32Width = w
    rect.s32Height = h
    return rect


# Fraction of the frame period the exposure may occupy before it must be reduced
# to let the requested frame rate be achievable (leaves headroom for readout).
_EXPOSURE_DUTY_MAX = 0.98


def clamp_fps_exposure(fps, exposure_ms, fps_min, fps_max):
    """Reconcile a requested frame rate and exposure for a camera.

    Encodes two coupled hardware constraints in one testable unit:
      1. Range clamp: fps is clamped to [fps_min, fps_max]. If fps_max <= 0 the
         camera does not report a usable range (e.g. model without frame-rate
         control), so the requested fps is passed through unchanged.
      2. Exposure/period coupling: exposure must fit inside the frame period or
         it caps the achievable rate. If exposure_ms exceeds
         _EXPOSURE_DUTY_MAX * (1000 / fps), it is reduced to that bound.

    Returns (fps, exposure_ms, warn) where warn is a human-readable string when a
    value was changed, else None.
    """
    warns = []
    fps = float(fps)
    exposure_ms = float(exposure_ms)

    if fps_max and fps_max > 0:
        clamped = min(max(fps, fps_min), fps_max)
        if clamped != fps:
            warns.append(f"fps {fps:g} clamped to [{fps_min:g}, {fps_max:g}] "
                         f"-> {clamped:g}")
            fps = clamped

    if fps > 0:
        max_exposure_ms = _EXPOSURE_DUTY_MAX * (1000.0 / fps)
        if exposure_ms > max_exposure_ms:
            warns.append(f"exposure {exposure_ms:g}ms reduced to "
                         f"{max_exposure_ms:.3g}ms to fit {fps:g} fps")
            exposure_ms = max_exposure_ms

    return fps, exposure_ms, ("; ".join(warns) if warns else None)


def summarize_frame_info(hw_frame_count, hw_timestamp_ns):
    """Derive drop count and a leakage-free frame rate from hardware frame info.

    hw_frame_count : per-frame hardware frame numbers (monotonically increasing,
        one per delivered frame). Gaps > 1 indicate frames the camera captured
        but the host never received — GENUINE drops, distinct from late dequeues.
    hw_timestamp_ns : per-frame hardware timestamps in nanoseconds (or None/NaN
        entries where the model does not report them).

    Returns dict: {dropped, n_frames, hw_fps, hw_span_s}. hw_fps is computed from
    the hardware timestamps (mean interval) and is the leakage-free clock; it is
    None if fewer than 2 valid timestamps are available.
    """
    fc = np.asarray(hw_frame_count, dtype=np.float64)
    fc = fc[np.isfinite(fc)]
    dropped = 0
    if fc.size >= 2:
        diffs = np.diff(fc)
        gaps = diffs[diffs > 1]
        dropped = int(np.sum(gaps - 1))

    ts = np.asarray(hw_timestamp_ns, dtype=np.float64)
    ts = ts[np.isfinite(ts)]
    ts = ts[ts >= 0]  # sentinel for "unsupported" frames
    hw_fps = None
    hw_span_s = None
    if ts.size >= 2:
        span_ns = ts[-1] - ts[0]
        if span_ns > 0:
            hw_span_s = span_ns / 1e9
            hw_fps = (ts.size - 1) / hw_span_s

    return {
        "dropped": dropped,
        "n_frames": int(np.asarray(hw_frame_count).size),
        "hw_fps": hw_fps,
        "hw_span_s": hw_span_s,
    }


def write_recording_sidecars(rec_file, sw_timestamps, hw_frames, hw_timestamps_ns,
                             ring_in_use, *, start_unix, stop_unix, camera,
                             width, height, nominal_fps):
    """Write the three recording sidecars and return summary stats (or None).

    Writes (next to <rec_file>):
      * <stem>_timestamps.npy  — software monotonic grab-times (back-compat)
      * <stem>_hwclock.npz     — hardware frame clock + software clock + ring trace
      * <stem>.json            — video_metadata sidecar; measured_fps prefers the
                                 leakage-free HARDWARE clock, plus drop count.

    Returns {n, sw_fps, hw_fps, dropped, ring_peak} or None if < 2 frames.
    Never raises on I/O — file errors are printed and swallowed (like write_sidecar).
    """
    ts = list(sw_timestamps)
    n = len(ts)
    if n < 2:
        return None

    span = ts[-1] - ts[0]
    sw_fps = (n - 1) / span if span > 0 else 0.0

    hw_frames = np.asarray(hw_frames, dtype=np.float64)
    hw_ts_ns = np.asarray(hw_timestamps_ns, dtype=np.float64)
    ring = np.asarray(ring_in_use, dtype=np.float64)
    summary = summarize_frame_info(hw_frames, hw_ts_ns)
    hw_fps = summary["hw_fps"]
    dropped = summary["dropped"]
    ring_peak = int(np.nanmax(ring)) if ring.size else 0

    print(f"{camera}: recorded {n} frames over {span:.3f}s "
          f"| sw_fps={sw_fps:.1f} hw_fps={hw_fps if hw_fps else float('nan'):.1f} "
          f"| genuine drops={dropped} | ring peak={ring_peak}")

    try:
        if rec_file:
            stem = os.path.splitext(rec_file)[0]
            np.save(stem + '_timestamps.npy', np.array(ts))
            np.savez(stem + '_hwclock.npz',
                     sw_monotonic=np.array(ts),
                     hw_frame_count=hw_frames,
                     hw_timestamp_ns=hw_ts_ns,
                     ring_in_use=ring)
            video_metadata.write_sidecar(
                rec_file, n, start_unix=start_unix, stop_unix=stop_unix,
                camera=camera, width=width, height=height, nominal_fps=nominal_fps,
                extra={
                    "hw_measured_fps": hw_fps,
                    "sw_measured_fps": sw_fps,
                    "dropped_frames": dropped,
                    "ring_peak_in_use": ring_peak,
                    "schema": "mastqg.video_sidecar.v2",
                })
    except Exception as e:
        print(f"Error saving recording sidecars: {e}")

    return {"n": n, "sw_fps": sw_fps, "hw_fps": hw_fps,
            "dropped": dropped, "ring_peak": ring_peak}


class CameraInstance:
    """Class to store state and controls for each camera"""
    def __init__(self, name="Camera"):
        self.camera = None
        self.camera_id = ""
        self.usb_port = ""
        self.name = name
        self.recording = False
        self.video_writer = None
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 30
        self.exposure_ms = 10.0
        self.record_duration_limit = 0
        self.record_frame_limit = 0
        self.recorded_frame_count = 0
        self.recording_start_time = 0
        self.last_frame = None
        self.image_width = None   # set from live frames; used for drag coordinate mapping
        self.image_height = None
        # Actual acquisition FPS tracking (in acquisition thread)
        self.acq_frame_count = 0
        self.acq_last_time = time.time()
        # Markup overlays: list of dicts with keys 'type', 'pos'/'center'/'radius', 'color'
        self.overlays = []
        self.panel_widget = None   # QWidget in the splitter; set in init_ui
        self.controls_splitter = None  # QSplitter(Vertical) per camera; set in init_ui
        self._splitter_sizes = None    # saved sizes for toggle restore
        self.cam_type = None           # 'thorlabs' or 'ids_ueye'; set on connect
        self.consecutive_frame_errors = 0  # stale-connection detector
        # Add locks for thread safety
        self.camera_lock = threading.Lock()
        self.bit_depth = 8             # cached at connect; display thread reads this without lock
        self.sensor_width = 0          # cached at connect; fallback when frame lacks dimensions
        self.sensor_height = 0
        self.acq_thread = None
        self.acq_stop_event = threading.Event()
        self.pending_frame = None      # latest frame from acquisition thread
        self.frame_lock = threading.Lock()
        self.new_frame_available = False
        self.stale_detected = threading.Event()  # set by acq thread; polled by display timer
        self.video_lock = threading.Lock()       # protects video_writer from concurrent access
        self.rotation = 0                          # display rotation in degrees: 0, 90, 180, 270
        # ROI state (None = full sensor)
        self.roi = None          # (x, y, w, h) or None
        self.current_frame_w = 0  # active capture width; updated on connect and ROI change
        self.current_frame_h = 0  # active capture height; updated on connect and ROI change
        # Recording timestamps (monotonic); appended per frame; reset on each new recording
        self._rec_timestamps = []
        # Hardware frame clock captured per recorded frame (leakage-free; detects
        # genuine drops vs late software dequeues). Reset on each new recording.
        self._rec_hw_frames = []      # hardware frame counter per frame
        self._rec_hw_timestamps = []  # hardware device timestamp (ns) per frame
        self._rec_ring_in_use = []    # IDS ring occupancy per frame (backpressure)
        self._rec_filename = ''  # path of current/last recording file

class IDSFrame:
    """Duck-type match for Thorlabs frame; holds captured pixel data.

    hw_frame_count / hw_timestamp_ns / buffers_in_use are the hardware frame
    number, device timestamp (ns) and ring occupancy captured from
    is_GetImageInfo. They default to None so the duck-type stays compatible with
    the Thorlabs Frame (which exposes frame_count / time_stamp_relative_ns_or_null
    directly). Populated only during recording (see IDSCamera capture-attach).
    """
    def __init__(self, data: np.ndarray, width: int, height: int,
                 hw_frame_count=None, hw_timestamp_ns=None, buffers_in_use=None):
        self.image_buffer = data                              # 1-D uint8 numpy array
        self.image_buffer_size_pixels_horizontal = width
        self.image_buffer_size_pixels_vertical   = height
        self.hw_frame_count  = hw_frame_count
        self.hw_timestamp_ns = hw_timestamp_ns
        self.buffers_in_use  = buffers_in_use


# IDS uEye image-queue ring depth. The original code hardcoded 3, which under
# USB bandwidth pressure is a plausible source of the isolated frame stalls seen
# in issue #4 (vs the Thorlabs 30-buffer pool). A deeper ring absorbs host-side
# dequeue latency without dropping frames. Named so it is trivial to tune live.
IDS_RING_BUFFERS = 12

# The display refresh is decoupled from capture: the sensor may free-run at
# 200 fps but the GUI only needs ~30 fps of on-screen updates. Repainting at the
# full capture rate wastes CPU and adds display-thread contention that itself
# worsens the software dequeue jitter in the acquisition loop. The recording
# path still writes EVERY captured frame regardless of this cap.
DISPLAY_FPS_CAP = 30


class IDSCamera:
    """Adapter around pyueye; presents same duck-type API as Thorlabs TLCamera."""

    IS_USE_DEVICE_ID = 0x8000

    def __init__(self, device_id: int, n_buffers: int = IDS_RING_BUFFERS):
        from pyueye import ueye
        self._ue = ueye
        self._pending_frame = None
        self._last_seq_num = -1  # Track frame sequence to avoid counting duplicates
        self._n_buffers = max(3, int(n_buffers))
        # Cache of the last commanded frame rate so it can be re-applied after an
        # AOI change (is_SetFrameRate does not survive re-arm). None = not set.
        self._frame_rate = None
        # When True, capture per-frame hardware info (is_GetImageInfo) — enabled
        # only during recording so the free-run display path pays no extra cost.
        self.capture_hw_info = False

        # Open camera by physical device ID
        self._hCam = ctypes.c_uint(device_id | self.IS_USE_DEVICE_ID)
        ret = ueye.is_InitCamera(self._hCam, None)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"is_InitCamera failed (device {device_id}): {ret}")

        # Sensor dimensions
        sinfo = ueye.SENSORINFO()
        ueye.is_GetSensorInfo(self._hCam, sinfo)
        self.sensor_width_pixels  = int(sinfo.nMaxWidth)
        self.sensor_height_pixels = int(sinfo.nMaxHeight)
        self.model = sinfo.strSensorName.decode('ascii', errors='replace').rstrip('\x00')

        # Serial number from BOARDINFO
        binfo = ueye.BOARDINFO()
        if ueye.is_GetCameraInfo(self._hCam, binfo) == ueye.IS_SUCCESS:
            self.serial_number = binfo.SerNo.decode('ascii', errors='replace').rstrip('\x00')
        else:
            self.serial_number = "unknown"
        self.firmware_version = "N/A"
        self.bit_depth = 8

        # 8-bit mono colour mode
        ret = ueye.is_SetColorMode(self._hCam, ueye.IS_CM_MONO8)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"is_SetColorMode(MONO8) failed: {ret}")

        # Allocate multiple image buffers for queue mode (allows non-blocking polling)
        self._mem_ptrs = []
        self._mem_ids = []
        for i in range(self._n_buffers):  # deeper ring absorbs host dequeue latency
            mem_ptr = ueye.c_mem_p()
            mem_id = ctypes.c_int()
            ret = ueye.is_AllocImageMem(
                self._hCam,
                self.sensor_width_pixels, self.sensor_height_pixels,
                8, mem_ptr, mem_id)
            if ret != ueye.IS_SUCCESS:
                raise RuntimeError(f"is_AllocImageMem failed on buffer {i}: {ret}")

            # Add buffer to image queue
            ret = ueye.is_AddToSequence(self._hCam, mem_ptr, mem_id)
            if ret != ueye.IS_SUCCESS:
                raise RuntimeError(f"is_AddToSequence failed on buffer {i}: {ret}")

            self._mem_ptrs.append(mem_ptr)
            self._mem_ids.append(mem_id)

        # Keep first buffer reference for compatibility
        self._mem_ptr = self._mem_ptrs[0]
        self._mem_id = self._mem_ids[0]

        # Cache pitch (bytes per row, may include padding)
        pitch = ctypes.c_int()
        ueye.is_GetImageMemPitch(self._hCam, pitch)
        self._pitch = pitch.value

        # Active AOI dimensions (updated by set_aoi; start at full sensor)
        self._current_w = self.sensor_width_pixels
        self._current_h = self.sensor_height_pixels

        # Enable frame-ready event for blocking wait_for_frame()
        ret = ueye.is_EnableEvent(self._hCam, ueye.IS_SET_EVENT_FRAME)
        if ret != ueye.IS_SUCCESS:
            raise RuntimeError(f"is_EnableEvent(FRAME) failed: {ret}")
        self._event_enabled = True

        # No-op properties for API compatibility
        self.frames_per_trigger_zero_for_unlimited = 0
        self.image_poll_timeout_ms = 1000

    # ── Thorlabs-compatible API ──────────────────────────────────────────

    @property
    def exposure_time_us(self):
        return None  # getter unused by app

    @exposure_time_us.setter
    def exposure_time_us(self, value_us: int):
        exp_ms = ctypes.c_double(value_us / 1000.0)
        self._ue.is_Exposure(
            self._hCam,
            self._ue.EXPOSURE_CMD.IS_EXPOSURE_CMD_SET_EXPOSURE,
            exp_ms,
            ctypes.sizeof(exp_ms))

    def arm(self, n_buffers=None):
        # Freerun mode: camera delivers frames continuously; event-driven wait in acquisition loop.
        # n_buffers, when given and different from the current ring depth, re-allocates
        # the image queue (was previously ignored — the ring was fixed at 3).
        if n_buffers is not None and max(3, int(n_buffers)) != self._n_buffers:
            self._realloc_buffers(max(3, int(n_buffers)))

        ret = self._ue.is_SetExternalTrigger(self._hCam, self._ue.IS_SET_TRIGGER_OFF)
        if ret != self._ue.IS_SUCCESS:
            raise RuntimeError(f"is_SetExternalTrigger(OFF/freerun) failed: {ret}")

        # Start live video; each frame gated by is_ForceTrigger
        ret = self._ue.is_CaptureVideo(self._hCam, self._ue.IS_DONT_WAIT)
        if ret != self._ue.IS_SUCCESS:
            raise RuntimeError(f"is_CaptureVideo failed: {ret}")

        # Re-apply the commanded frame rate — freerun rate resets on re-arm.
        if self._frame_rate:
            self.set_frame_rate(self._frame_rate)

    def _realloc_buffers(self, n_buffers):
        """Free and re-allocate the image queue to `n_buffers` at the current AOI.

        Caller must ensure live video is stopped. Used to honor a changed ring
        depth from arm().
        """
        ue = self._ue
        ue.is_StopLiveVideo(self._hCam, ue.IS_FORCE_VIDEO_STOP)
        ue.is_ClearSequence(self._hCam)
        for mem_ptr, mem_id in zip(self._mem_ptrs, self._mem_ids):
            ue.is_FreeImageMem(self._hCam, mem_ptr, mem_id)
        self._mem_ptrs.clear()
        self._mem_ids.clear()
        for i in range(n_buffers):
            mem_ptr = ue.c_mem_p()
            mem_id = ctypes.c_int()
            ret = ue.is_AllocImageMem(self._hCam, self._current_w, self._current_h,
                                      8, mem_ptr, mem_id)
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_AllocImageMem (realloc) failed on buffer {i}: {ret}")
            ret = ue.is_AddToSequence(self._hCam, mem_ptr, mem_id)
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_AddToSequence (realloc) failed on buffer {i}: {ret}")
            self._mem_ptrs.append(mem_ptr)
            self._mem_ids.append(mem_id)
        self._mem_ptr = self._mem_ptrs[0]
        self._mem_id = self._mem_ids[0]
        self._n_buffers = n_buffers

    def disarm(self):
        self._ue.is_StopLiveVideo(self._hCam, self._ue.IS_FORCE_VIDEO_STOP)
        if getattr(self, '_event_enabled', False):
            self._ue.is_DisableEvent(self._hCam, self._ue.IS_SET_EVENT_FRAME)
            self._event_enabled = False

    def dispose(self):
        try:
            # Free all allocated image buffers
            for mem_ptr, mem_id in zip(self._mem_ptrs, self._mem_ids):
                self._ue.is_FreeImageMem(self._hCam, mem_ptr, mem_id)
        except Exception as e:
            print(f"IDSCamera.dispose FreeImageMem: {e}")
        try:
            self._ue.is_ExitCamera(self._hCam)
        except Exception as e:
            print(f"IDSCamera.dispose ExitCamera: {e}")

    def issue_software_trigger(self):
        """Poll for next available frame from image queue (non-blocking)."""
        # Get the most recent frame from the sequence (non-blocking query)
        nNum = ctypes.c_int()
        pcMem = self._ue.c_mem_p()
        pcMemLast = self._ue.c_mem_p()

        ret = self._ue.is_GetActSeqBuf(self._hCam, nNum, pcMem, pcMemLast)

        if ret != self._ue.IS_SUCCESS:
            # No frame available yet - this is normal, just return
            self._pending_frame = None
            return

        # Check if this is a new frame by comparing sequence number
        seq_num = nNum.value
        if seq_num == self._last_seq_num:
            # Same frame as last poll - don't return it again
            self._pending_frame = None
            return

        # New frame - update sequence number and copy frame data
        self._last_seq_num = seq_num
        w, h = self._current_w, self._current_h
        raw = self._ue.get_data(pcMemLast, w, h, 8, self._pitch, copy=True)
        frame_np = reshape_ids_frame(raw, w, h, self._pitch)
        hw_fc, hw_ts, in_use = self._read_image_info(nNum.value)
        self._pending_frame = IDSFrame(frame_np.ravel(), w, h,
                                       hw_frame_count=hw_fc, hw_timestamp_ns=hw_ts,
                                       buffers_in_use=in_use)

        # Unlock the buffer so it can be reused
        self._ue.is_UnlockSeqBuf(self._hCam, nNum, pcMemLast)

    def _read_image_info(self, mem_id):
        """Return (hw_frame_count, hw_timestamp_ns, buffers_in_use) for a buffer.

        Uses is_GetImageInfo -> UEYEIMAGEINFO. Only queried when capture_hw_info is
        set (recording), otherwise returns (None, None, None) so the free-run
        display path pays no cost. Never raises — returns Nones on any failure.
        """
        if not self.capture_hw_info:
            return None, None, None
        try:
            info = self._ue.UEYEIMAGEINFO()
            ret = self._ue.is_GetImageInfo(self._hCam, ctypes.c_int(mem_id),
                                           info, ctypes.sizeof(info))
            if ret != self._ue.IS_SUCCESS:
                return None, None, None
            # u64TimestampDevice is in 100ns ticks (uEye convention) -> ns.
            ts_ns = int(info.u64TimestampDevice) * 100
            return int(info.u64FrameNumber), ts_ns, int(info.dwImageBuffersInUse)
        except Exception:
            return None, None, None

    def get_pending_frame_or_null(self):
        frame, self._pending_frame = self._pending_frame, None
        return frame

    def issue_trigger(self):
        """Issue one software trigger; camera captures exactly one frame."""
        ret = self._ue.is_ForceTrigger(self._hCam)
        # IS_TRIGGER_ACTIVATED means a prior trigger is still processing — not an error
        if ret not in (self._ue.IS_SUCCESS, self._ue.IS_TRIGGER_ACTIVATED):
            print(f"IDS is_ForceTrigger returned {ret}")

    def wait_for_frame(self, timeout_ms=200):
        """Block until frame ready (event-driven). Returns IDSFrame or None on timeout."""
        ret = self._ue.is_WaitEvent(self._hCam, self._ue.IS_SET_EVENT_FRAME, timeout_ms)
        if ret == self._ue.IS_TIMED_OUT:
            return None
        if ret != self._ue.IS_SUCCESS:
            print(f"IDS is_WaitEvent error: {ret}")
            return None

        nNum = ctypes.c_int()
        pcMem = self._ue.c_mem_p()
        pcMemLast = self._ue.c_mem_p()
        ret = self._ue.is_GetActSeqBuf(self._hCam, nNum, pcMem, pcMemLast)
        if ret != self._ue.IS_SUCCESS:
            return None

        seq_num = nNum.value
        if seq_num == self._last_seq_num:
            return None
        self._last_seq_num = seq_num

        w, h = self._current_w, self._current_h
        raw = self._ue.get_data(pcMemLast, w, h, 8, self._pitch, copy=True)
        frame_np = reshape_ids_frame(raw, w, h, self._pitch)
        hw_fc, hw_ts, in_use = self._read_image_info(nNum.value)
        frame = IDSFrame(frame_np.ravel(), w, h,
                         hw_frame_count=hw_fc, hw_timestamp_ns=hw_ts,
                         buffers_in_use=in_use)
        self._ue.is_UnlockSeqBuf(self._hCam, nNum, pcMemLast)
        return frame

    def set_aoi(self, x, y, w, h):
        """Stop capture, free buffers, set AOI, reallocate buffers, restart.

        Caller must ensure the acquisition thread is stopped before calling.
        """
        ue = self._ue

        # 1. Stop live video
        ue.is_StopLiveVideo(self._hCam, ue.IS_FORCE_VIDEO_STOP)

        # 2. Disable frame event temporarily
        if getattr(self, '_event_enabled', False):
            ue.is_DisableEvent(self._hCam, ue.IS_SET_EVENT_FRAME)
            self._event_enabled = False

        # 3. Clear sequence (dequeues all buffers before freeing)
        ue.is_ClearSequence(self._hCam)

        # 4. Free existing buffers
        for mem_ptr, mem_id in zip(self._mem_ptrs, self._mem_ids):
            ue.is_FreeImageMem(self._hCam, mem_ptr, mem_id)
        self._mem_ptrs.clear()
        self._mem_ids.clear()

        # 5. Set AOI; restore full-sensor on failure before raising
        aoi_rect = build_ids_rect(x, y, w, h)
        ret = ue.is_AOI(self._hCam, ue.IS_AOI_IMAGE_SET_AOI,
                        aoi_rect, ctypes.sizeof(aoi_rect))
        if ret != ue.IS_SUCCESS:
            full_rect = build_ids_rect(0, 0, self.sensor_width_pixels, self.sensor_height_pixels)
            ue.is_AOI(self._hCam, ue.IS_AOI_IMAGE_SET_AOI,
                      full_rect, ctypes.sizeof(full_rect))
            raise RuntimeError(f"is_AOI(SET) failed: {ret}")

        # 6. Update cached dimensions
        self._current_w = w
        self._current_h = h

        # 7. Allocate new buffers sized to AOI (same ring depth as at init)
        for i in range(self._n_buffers):
            mem_ptr = ue.c_mem_p()
            mem_id = ctypes.c_int()
            ret = ue.is_AllocImageMem(self._hCam, w, h, 8, mem_ptr, mem_id)
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_AllocImageMem (AOI) failed on buffer {i}: {ret}")
            ret = ue.is_AddToSequence(self._hCam, mem_ptr, mem_id)
            if ret != ue.IS_SUCCESS:
                raise RuntimeError(f"is_AddToSequence (AOI) failed on buffer {i}: {ret}")
            self._mem_ptrs.append(mem_ptr)
            self._mem_ids.append(mem_id)
        self._mem_ptr = self._mem_ptrs[0]
        self._mem_id = self._mem_ids[0]

        # 8. Re-read pitch for new AOI width
        pitch = ctypes.c_int()
        ue.is_GetImageMemPitch(self._hCam, pitch)
        self._pitch = pitch.value

        # 9. Re-enable frame event
        ret = ue.is_EnableEvent(self._hCam, ue.IS_SET_EVENT_FRAME)
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_EnableEvent(FRAME) after AOI failed: {ret}")
        self._event_enabled = True

        # 10. Restart capture in freerun mode
        ret = ue.is_SetExternalTrigger(self._hCam, ue.IS_SET_TRIGGER_OFF)
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_SetExternalTrigger(OFF/freerun) after AOI failed: {ret}")
        ret = ue.is_CaptureVideo(self._hCam, ue.IS_DONT_WAIT)
        if ret != ue.IS_SUCCESS:
            raise RuntimeError(f"is_CaptureVideo after AOI failed: {ret}")

        # 11. Re-apply commanded frame rate — freerun rate is reset by re-arm.
        if self._frame_rate:
            self.set_frame_rate(self._frame_rate)

        self._last_seq_num = -1

    def reset_aoi(self):
        """Restore full-sensor AOI."""
        self.set_aoi(0, 0, self.sensor_width_pixels, self.sensor_height_pixels)

    def get_pixel_clock_list(self):
        """Return sorted list of supported pixel clock values (MHz)."""
        n = ctypes.c_uint()
        self._ue.is_PixelClock(
            self._hCam, self._ue.IS_PIXELCLOCK_CMD_GET_NUMBER, n, ctypes.sizeof(n))
        count = n.value
        if count == 0:
            return []
        arr = (ctypes.c_uint * count)()
        self._ue.is_PixelClock(
            self._hCam, self._ue.IS_PIXELCLOCK_CMD_GET_LIST,
            arr, count * ctypes.sizeof(ctypes.c_uint()))
        return sorted(set(arr))

    def get_pixel_clock(self):
        val = ctypes.c_uint()
        self._ue.is_PixelClock(
            self._hCam, self._ue.IS_PIXELCLOCK_CMD_GET, val, ctypes.sizeof(val))
        return val.value

    def set_pixel_clock(self, mhz: int):
        val = ctypes.c_uint(mhz)
        ret = self._ue.is_PixelClock(
            self._hCam, self._ue.IS_PIXELCLOCK_CMD_SET, val, ctypes.sizeof(val))
        if ret != self._ue.IS_SUCCESS:
            raise RuntimeError(f"is_PixelClock SET {mhz} MHz failed: {ret}")

    # ── Frame-rate control (freerun) ─────────────────────────────────────
    # In freerun (trigger OFF + capture on — the state arm()/set_aoi() leave the
    # camera in) the frame rate can be commanded with is_SetFrameRate. It must be
    # re-issued after any re-arm (arm/set_aoi do this via self._frame_rate).

    def get_frame_rate_range(self):
        """Return (fps_min, fps_max) achievable at the current pixel-clock/AOI.

        is_GetFrameTimeRange returns frame *times* (seconds), so the fps bounds
        are the inverse: fps_max = 1/t_min, fps_min = 1/t_max. Returns (0.0, 0.0)
        on failure (caller treats fps_max<=0 as "no usable range").
        """
        t_min = ctypes.c_double()
        t_max = ctypes.c_double()
        t_intv = ctypes.c_double()
        ret = self._ue.is_GetFrameTimeRange(self._hCam, t_min, t_max, t_intv)
        if ret != self._ue.IS_SUCCESS or t_min.value <= 0 or t_max.value <= 0:
            return 0.0, 0.0
        return 1.0 / t_max.value, 1.0 / t_min.value

    def set_frame_rate(self, fps):
        """Command a fixed freerun frame rate; return the achieved rate.

        The SDK snaps to the nearest achievable rate and writes it into newFPS.
        Caches the request so arm()/set_aoi() can re-apply it after re-arm.
        """
        self._frame_rate = float(fps)
        new_fps = ctypes.c_double()
        ret = self._ue.is_SetFrameRate(self._hCam, ctypes.c_double(float(fps)), new_fps)
        if ret != self._ue.IS_SUCCESS:
            raise RuntimeError(f"is_SetFrameRate({fps}) failed: {ret}")
        return new_fps.value

    def get_measured_frame_rate_fps(self):
        """Live measured frame rate (duck-type match for Thorlabs)."""
        fps = ctypes.c_double()
        ret = self._ue.is_GetFramesPerSecond(self._hCam, fps)
        if ret != self._ue.IS_SUCCESS:
            return 0.0
        return fps.value


class ThorlabsCameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.sdk = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.sdk_lock = threading.Lock()  # Add lock for SDK access
        
        # Create two camera instances
        self.cameras = {
            "cam1": CameraInstance("Camera 1"),
            "cam2": CameraInstance("Camera 2")
        }
        
        # Flag to track if we're currently refreshing cameras
        self.refreshing_cameras = False

        # Per-camera acquisition threads are started when cameras connect
        # (reverted from unified thread to allow independent pacing)

        # Add dedicated lock for Thorlabs SDK to prevent simultaneous calls
        self.thorlabs_sdk_lock = threading.Lock()

        self.init_ui()
        # Delay SDK initialization to prevent segfaults during startup
        QTimer.singleShot(500, self.init_sdk)
        
    def init_sdk(self):
        """Initialize the SDK and discover available cameras"""
        # Make sure any previous SDK instance is disposed
        with self.sdk_lock:
            if self.sdk is not None:
                try:
                    print("Disposing of existing SDK instance")
                    self.sdk.dispose()
                    self.sdk = None
                    # Small delay to ensure SDK is fully disposed
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error disposing SDK: {e}")
                    traceback.print_exc()
        
        try:
            print("Initializing SDK...")
            with self.sdk_lock:
                self.sdk = TLCameraSDK()
                print(f"SDK initialized: {self.sdk}")
                
                # Add a small delay before discovering cameras to prevent race conditions
                time.sleep(0.2)
                
                available_cameras = self.sdk.discover_available_cameras()
                # Filter out invalid camera IDs to prevent segfaults
                valid_cameras = []
                for cam_id in available_cameras:
                    # Check if ID is valid for processing
                    if self.is_valid_camera_id(cam_id):
                        valid_cameras.append(cam_id)
                    else:
                        print(f"Skipping invalid camera ID: {repr(cam_id)}")
                
                # Check if we have multiple cameras and display a warning
                if len(valid_cameras) > 1:
                    print("NOTICE: Multiple cameras detected. Some Thorlabs camera models may")
                    print("not support simultaneous operation. If you encounter issues, try")
                    print("disconnecting one camera before connecting another.")
                
                print(f"Valid cameras: {[repr(c) for c in valid_cameras]}")
                
                # Store camera count for later use
                self.available_camera_count = len(valid_cameras)
            
            if not valid_cameras:
                print("No valid cameras found during Thorlabs discovery")
                self.statusBar().showMessage("No Thorlabs cameras found; checking IDS cameras...")
            
            # Populate the camera selection dropdown
            self.camera_selector.clear()
            
            for cam_id in valid_cameras:
                try:
                    # Safely handle camera info retrieval
                    with self.sdk_lock:
                        # Open camera briefly to get info - use repr for safer printing
                        print(f"Getting info for camera {repr(cam_id)}")
                        temp_camera = self.sdk.open_camera(cam_id)
                        
                        # Safely get camera name and port info, handle potential encoding issues
                        try:
                            camera_name = str(temp_camera.name)
                            print(f"Camera name: {camera_name}")
                        except (UnicodeDecodeError, AttributeError):
                            # Fallback: use part of ID as name
                            try:
                                if isinstance(cam_id, str):
                                    camera_name = f"Camera {cam_id[-6:]}"
                                else:
                                    camera_name = f"Camera {cam_id}"
                            except:
                                camera_name = "Unknown Camera"
                            print(f"Using fallback name: {camera_name}")
                            
                        usb_port = self.get_camera_usb_port(temp_camera)
                        print(f"USB port: {usb_port}")
                        
                        display_text = f"{camera_name} ({usb_port})"
                        # Store camera ID with USB port info for later use
                        self.camera_selector.addItem(display_text, ('thorlabs', cam_id, usb_port))
                        
                        temp_camera.dispose()
                        print(f"Successfully added camera to selector: {display_text}")
                        
                        # Add a small delay between camera operations
                        time.sleep(0.1)
                        
                except Exception as e:
                    print(f"Error getting info for camera {repr(cam_id)}: {e}")
                    traceback.print_exc()
                    # Still try to add the camera with minimal information
                    try:
                        id_part = str(cam_id)[-6:] if isinstance(cam_id, str) and len(str(cam_id)) >= 6 else "unknown"
                        self.camera_selector.addItem(f"Camera {id_part}", ('thorlabs', cam_id, "Unknown"))
                    except Exception as e2:
                        print(f"Failed to add camera to selector: {e2}")
            
            msg = f"Found {len(valid_cameras)} valid camera(s)"
            print(msg)
            self.statusBar().showMessage(msg)

        except Exception as e:
            print(f"SDK Initialization Error: {e}")
            traceback.print_exc()
            self.show_error("SDK Initialization Error", str(e))
            self.statusBar().showMessage(f"Error initializing SDK: {str(e)}")

        # ── IDS uEye discovery ─────────────────────────────────────────
        if _IDS_AVAILABLE:
            try:
                n = ctypes.c_int(0)
                if (_ueye_mod.is_GetNumberOfCameras(n) == _ueye_mod.IS_SUCCESS
                        and n.value > 0):
                    count = n.value
                    cam_list = _ueye_mod.UEYE_CAMERA_LIST(
                        _ueye_mod.UEYE_CAMERA_INFO * count)
                    cam_list.dwCount = count
                    if _ueye_mod.is_GetCameraList(cam_list) == _ueye_mod.IS_SUCCESS:
                        for i in range(count):
                            info = cam_list.uci[i]
                            dev_id  = int(info.dwDeviceID)
                            name    = (info.FullModelName or info.Model).decode(
                                        'ascii', errors='replace').rstrip('\x00')
                            serno   = info.SerNo.decode('ascii', errors='replace').rstrip('\x00')
                            in_use  = bool(info.dwInUse)
                            label   = f"[IDS] {name} (SN:{serno})"
                            if in_use:
                                label += " [IN USE — close IDS Camera Manager first]"
                            self.camera_selector.addItem(
                                label, ('ids_ueye', dev_id, f"{name}|{serno}"))
                            print(f"IDS camera: {label}, device_id={dev_id}")
            except Exception as e:
                print(f"IDS discovery error: {e}")
                traceback.print_exc()

        if self.camera_selector.count() == 0:
            self.show_error("No cameras found!", "Make sure cameras are connected and powered on.")
            self.statusBar().showMessage("No cameras found!")
    
    def is_valid_camera_id(self, camera_id):
        """Check if the camera ID is valid and safe to use"""
        try:
            # Check if camera_id is a string and can be safely printed
            if not isinstance(camera_id, str):
                print(f"Non-string camera ID: {repr(camera_id)}")
                return False
                
            # Check if the string contains only printable characters
            if not all(c.isprintable() for c in camera_id):
                print(f"Camera ID contains non-printable characters: {repr(camera_id)}")
                return False
                
            # Check if the string is empty
            if not camera_id.strip():
                print("Empty camera ID")
                return False
                
            # If it's a single control character, it's likely invalid
            if len(camera_id) == 1 and ord(camera_id) < 32:
                print(f"Camera ID is a control character: {repr(camera_id)}")
                return False
                
            return True
        except Exception as e:
            print(f"Error validating camera ID: {e}")
            return False
            
    def get_camera_usb_port(self, camera):
        """Attempts to get USB port information for the camera"""
        try:
            # Try several approaches to get meaningful camera identifier info
            if hasattr(camera, "usb_port") and camera.usb_port:
                return str(camera.usb_port)
            elif hasattr(camera, "serial_number") and camera.serial_number:
                return str(camera.serial_number)
            elif hasattr(camera, "model") and camera.model:
                return str(camera.model)
            else:
                # Create a unique identifier from the camera id
                try:
                    camera_id = str(camera.camera_id) if hasattr(camera, "camera_id") else "Unknown"
                    # Make sure we only use printable characters
                    camera_id = ''.join(c for c in camera_id if c.isprintable())
                    return f"ID-{camera_id[-6:] if len(camera_id) >= 6 else camera_id}"
                except:
                    return "Unknown ID"
        except Exception as e:
            print(f"Error getting camera USB port: {e}")
            return "Unknown port"
            
    def init_ui(self):
        # Main window setup
        self.setWindowTitle("Dual Camera Fast (200 fps) — dualcam_fast")
        self.setGeometry(100, 100, 1200, 800)
        self.setMinimumSize(400, 300)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Camera connection controls
        connection_layout = QHBoxLayout()
        main_layout.addLayout(connection_layout)
        
        # Camera selector dropdown
        connection_layout.addWidget(QLabel("Available Cameras:"))
        self.camera_selector = QComboBox()
        connection_layout.addWidget(self.camera_selector)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.on_connect_btn_clicked)
        connection_layout.addWidget(self.connect_btn)
        self.camera_selector.currentIndexChanged.connect(self.update_connect_btn)
        
        # Refresh camera list button
        self.refresh_btn = QPushButton("Refresh Camera List")
        self.refresh_btn.clicked.connect(self.refresh_camera_list)
        connection_layout.addWidget(self.refresh_btn)
        
        # Debug mode checkbox
        self.debug_checkbox = QCheckBox("Debug Mode")
        connection_layout.addWidget(self.debug_checkbox)
        
        # Camera displays - side-by-side panels; panels appear when cameras connect
        self.camera_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.camera_splitter)

        # Create panel for each camera
        for cam_id, cam_instance in self.cameras.items():
            cam_panel = QWidget()
            cam_layout = QVBoxLayout()
            cam_panel.setLayout(cam_layout)
            
            # Vertical splitter: image on top, controls on bottom (drag to resize)
            cam_instance.controls_splitter = QSplitter(Qt.Vertical)

            # Top pane: live image feed
            cam_instance.image_label = CameraLabel(cam_id, self)
            cam_instance.image_label.setAlignment(Qt.AlignCenter)
            cam_instance.image_label.setMinimumSize(1, 1)
            cam_instance.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            cam_instance.image_label.setText(f"Connect to {cam_instance.name} to view feed")
            cam_instance.controls_splitter.addWidget(cam_instance.image_label)

            # Bottom pane: toggle button + collapsible controls
            controls_bottom = QWidget()
            controls_bottom.setMinimumHeight(18)   # always show toggle btn
            controls_bottom_layout = QVBoxLayout()
            controls_bottom_layout.setContentsMargins(0, 0, 0, 0)
            controls_bottom_layout.setSpacing(0)
            controls_bottom.setLayout(controls_bottom_layout)

            cam_instance.toggle_controls_btn = QPushButton("▲ Hide Controls")
            cam_instance.toggle_controls_btn.setFixedHeight(18)
            cam_instance.toggle_controls_btn.clicked.connect(
                lambda _, c=cam_id: self.toggle_controls(c))
            controls_bottom_layout.addWidget(cam_instance.toggle_controls_btn)

            # ── Controls panel (collapsible) ──────────────────────────────
            cam_instance.controls_widget = QWidget()
            controls_layout = QHBoxLayout()
            controls_layout.setContentsMargins(0, 0, 0, 0)
            cam_instance.controls_widget.setLayout(controls_layout)

            # --- Acquisition (Exposure + Framerate merged) ---
            acq_group = QGroupBox("Acquisition")
            acq_layout = QGridLayout()
            acq_group.setLayout(acq_layout)

            acq_layout.addWidget(QLabel("Exp (ms):"), 0, 0)
            cam_instance.exposure_value = QDoubleSpinBox()
            cam_instance.exposure_value.setRange(0.1, 1000.0)
            cam_instance.exposure_value.setValue(10.0)
            cam_instance.exposure_value.setSingleStep(1.0)
            cam_instance.exposure_value.valueChanged.connect(
                lambda value, c=cam_id: self.set_exposure(c, value))
            acq_layout.addWidget(cam_instance.exposure_value, 0, 1)

            acq_layout.addWidget(QLabel("FPS:"), 0, 2)
            cam_instance.framerate_value = QSpinBox()
            cam_instance.framerate_value.setRange(1, 500)
            cam_instance.framerate_value.setValue(30)
            cam_instance.framerate_value.valueChanged.connect(
                lambda value, c=cam_id: self.set_framerate(c, value))
            acq_layout.addWidget(cam_instance.framerate_value, 0, 3)

            cam_instance.exposure_slider = QSlider(Qt.Horizontal)
            cam_instance.exposure_slider.setRange(1, 10000)
            cam_instance.exposure_slider.setValue(100)
            cam_instance.exposure_slider.valueChanged.connect(
                lambda value, c=cam_id: self.exposure_slider_changed(c, value))
            acq_layout.addWidget(cam_instance.exposure_slider, 1, 0, 1, 2)

            cam_instance.framerate_slider = QSlider(Qt.Horizontal)
            cam_instance.framerate_slider.setRange(1, 500)
            cam_instance.framerate_slider.setValue(30)
            cam_instance.framerate_slider.valueChanged.connect(
                lambda value, c=cam_id: self.framerate_slider_changed(c, value))
            acq_layout.addWidget(cam_instance.framerate_slider, 1, 2, 1, 2)

            acq_layout.addWidget(QLabel("FPS:"), 2, 0)
            cam_instance.fps_label = QLabel("0")
            acq_layout.addWidget(cam_instance.fps_label, 2, 1)

            cam_instance.rotate_btn = QPushButton("Rotate 90°")
            cam_instance.rotate_btn.clicked.connect(
                lambda _, c=cam_id: self.rotate_camera(c))
            acq_layout.addWidget(cam_instance.rotate_btn, 2, 2, 1, 2)

            cam_instance.pixel_clock_label = QLabel("Clock (MHz):")
            acq_layout.addWidget(cam_instance.pixel_clock_label, 3, 0)
            cam_instance.pixel_clock_combo = QComboBox()
            cam_instance.pixel_clock_combo.currentTextChanged.connect(
                lambda val, c=cam_id: self.set_pixel_clock(c, val))
            acq_layout.addWidget(cam_instance.pixel_clock_combo, 3, 1, 1, 3)
            cam_instance.pixel_clock_label.setVisible(False)
            cam_instance.pixel_clock_combo.setVisible(False)

            # --- ROI controls ---
            acq_layout.addWidget(QLabel("ROI X,Y,W,H:"), 4, 0)
            roi_coords_layout = QHBoxLayout()
            for attr, placeholder in (('roi_x', 'X'), ('roi_y', 'Y'),
                                       ('roi_w', 'W'), ('roi_h', 'H')):
                le = QLineEdit("0")
                le.setFixedWidth(55)
                le.setPlaceholderText(placeholder)
                le.setValidator(QIntValidator(0, 99999))
                setattr(cam_instance, attr, le)
                roi_coords_layout.addWidget(le)
            acq_layout.addLayout(roi_coords_layout, 4, 1, 1, 3)

            cam_instance.roi_apply_btn = QPushButton("Apply ROI")
            cam_instance.roi_apply_btn.setEnabled(False)
            cam_instance.roi_apply_btn.clicked.connect(
                lambda _, c=cam_id: self.apply_roi(c))
            acq_layout.addWidget(cam_instance.roi_apply_btn, 5, 0, 1, 2)

            cam_instance.roi_reset_btn = QPushButton("Reset ROI")
            cam_instance.roi_reset_btn.setEnabled(False)
            cam_instance.roi_reset_btn.clicked.connect(
                lambda _, c=cam_id: self.reset_roi(c))
            acq_layout.addWidget(cam_instance.roi_reset_btn, 5, 2, 1, 2)

            # --- Recording Controls ---
            recording_group = QGroupBox("Recording")
            recording_layout = QVBoxLayout()
            recording_group.setLayout(recording_layout)

            cam_instance.record_button = QPushButton("Start Recording")
            cam_instance.record_button.clicked.connect(
                lambda _, c=cam_id: self.toggle_recording(c))
            recording_layout.addWidget(cam_instance.record_button)

            cam_instance.recording_label = QLabel("Not Recording")
            recording_layout.addWidget(cam_instance.recording_label)

            duration_layout = QHBoxLayout()
            duration_layout.addWidget(QLabel("Duration (s):"))
            cam_instance.duration_spinbox = QDoubleSpinBox()
            cam_instance.duration_spinbox.setRange(0, 3600)
            cam_instance.duration_spinbox.setValue(0)
            duration_layout.addWidget(cam_instance.duration_spinbox)
            recording_layout.addLayout(duration_layout)

            frames_layout = QHBoxLayout()
            frames_layout.addWidget(QLabel("Frames:"))
            cam_instance.framecount_spinbox = QSpinBox()
            cam_instance.framecount_spinbox.setRange(0, 1000000)
            cam_instance.framecount_spinbox.setValue(0)
            frames_layout.addWidget(cam_instance.framecount_spinbox)
            recording_layout.addLayout(frames_layout)

            # --- Markup Overlay Controls ---
            markup_group = QGroupBox("Markup Overlays")
            markup_layout = QVBoxLayout()
            markup_group.setLayout(markup_layout)

            # Table: Type | X/CX | Y/CY | Radius | Thickness | Color
            cam_instance.markup_table = QTableWidget(0, 6)
            cam_instance.markup_table.setHorizontalHeaderLabels(
                ["Type", "X / CX", "Y / CY", "Radius", "Thickness", "Color"])
            cam_instance.markup_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            cam_instance.markup_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            cam_instance.markup_table.verticalHeader().setVisible(False)
            cam_instance.markup_table.cellChanged.connect(
                lambda row, col, c=cam_id: self._on_markup_cell_changed(c, row, col))
            markup_layout.addWidget(cam_instance.markup_table)

            color_row = QHBoxLayout()
            color_row.addWidget(QLabel("Color:"))
            cam_instance.markup_color = QComboBox()
            for color_name in MARKUP_COLORS:
                cam_instance.markup_color.addItem(color_name)
            color_row.addWidget(cam_instance.markup_color)
            markup_layout.addLayout(color_row)

            markup_btn_row = QHBoxLayout()
            add_hline_btn = QPushButton("Add H Line")
            add_hline_btn.clicked.connect(lambda _, c=cam_id: self.add_hline_overlay(c))
            markup_btn_row.addWidget(add_hline_btn)
            add_vline_btn = QPushButton("Add V Line")
            add_vline_btn.clicked.connect(lambda _, c=cam_id: self.add_vline_overlay(c))
            markup_btn_row.addWidget(add_vline_btn)
            add_circle_btn = QPushButton("Add Circle")
            add_circle_btn.clicked.connect(lambda _, c=cam_id: self.add_circle_overlay(c))
            markup_btn_row.addWidget(add_circle_btn)
            markup_layout.addLayout(markup_btn_row)

            remove_markup_btn = QPushButton("Remove Selected")
            remove_markup_btn.clicked.connect(
                lambda _, c=cam_id: self.remove_selected_overlay(c))
            markup_layout.addWidget(remove_markup_btn)

            # Proportional widths: acq 2 : recording 2 : markup 5
            controls_layout.addWidget(acq_group, 2)
            controls_layout.addWidget(recording_group, 2)
            controls_layout.addWidget(markup_group, 5)

            controls_bottom_layout.addWidget(cam_instance.controls_widget)

            cam_instance.controls_splitter.addWidget(controls_bottom)
            # Bottom pane non-collapsible via drag (toggle btn handles full hide)
            cam_instance.controls_splitter.setCollapsible(1, False)
            cam_instance.controls_splitter.setSizes([500, 150])

            cam_layout.addWidget(cam_instance.controls_splitter, 1)  # stretch=1
            
            # Status display for this camera
            cam_instance.status_label = QLabel(f"{cam_instance.name} not connected")
            cam_layout.addWidget(cam_instance.status_label)
            
            # Debug info area
            cam_instance.debug_label = QLabel("Debug info: None")
            cam_layout.addWidget(cam_instance.debug_label)
            cam_instance.debug_label.setVisible(False)
            
            self.camera_splitter.addWidget(cam_panel)
            cam_panel.setVisible(False)          # hidden until camera connects
            cam_instance.panel_widget = cam_panel
        
        # Status bar for global information
        self.statusBar().showMessage("Ready. Please connect cameras.")
        
        # Connect debug checkbox to update debug visibility
        self.debug_checkbox.stateChanged.connect(self.toggle_debug_mode)
    
    def update_connect_btn(self):
        """Set button text to Connect or Disconnect based on the selected camera's state."""
        idx = self.camera_selector.currentIndex()
        if idx < 0:
            self.connect_btn.setText("Connect")
            return
        camera_info = self.camera_selector.itemData(idx)
        if not camera_info:
            self.connect_btn.setText("Connect")
            return
        device_id = camera_info[1]
        for cam_instance in self.cameras.values():
            if cam_instance.camera_id == device_id:
                self.connect_btn.setText("Disconnect")
                return
        self.connect_btn.setText("Connect")

    def on_connect_btn_clicked(self):
        """Connect or disconnect whichever camera is currently selected in the dropdown."""
        idx = self.camera_selector.currentIndex()
        if idx < 0:
            return
        camera_info = self.camera_selector.itemData(idx)
        if not camera_info:
            return
        device_id = camera_info[1]
        # If already connected, disconnect it
        for cam_id, cam_instance in self.cameras.items():
            if cam_instance.camera_id == device_id:
                self.disconnect_camera(cam_id)
                return
        # Not connected — assign to the first free slot
        for cam_id in ("cam1", "cam2"):
            if not self.cameras[cam_id].camera:
                self.connect_camera(cam_id)
                return
        self.show_error("No Slot Available",
                        "Both camera slots are already occupied. Disconnect one first.")

    def toggle_controls(self, cam_id):
        """Show or hide the controls panel for a camera tab."""
        cam_instance = self.cameras[cam_id]
        splitter = cam_instance.controls_splitter
        visible = cam_instance.controls_widget.isVisible()
        if visible:
            # Save sizes, collapse controls area to just the toggle button
            cam_instance._splitter_sizes = splitter.sizes()
            total = sum(splitter.sizes())
            splitter.setSizes([total - 18, 18])
            cam_instance.controls_widget.setVisible(False)
            cam_instance.toggle_controls_btn.setText("▼ Show Controls")
        else:
            cam_instance.controls_widget.setVisible(True)
            if cam_instance._splitter_sizes:
                splitter.setSizes(cam_instance._splitter_sizes)
            cam_instance.toggle_controls_btn.setText("▲ Hide Controls")

    def toggle_debug_mode(self, state):
        """Toggle visibility of debug information"""
        is_visible = state == Qt.Checked
        for cam_id, cam_instance in self.cameras.items():
            cam_instance.debug_label.setVisible(is_visible)
            
    # ------------------------------------------------------------------ #
    # Markup overlay helpers                                               #
    # ------------------------------------------------------------------ #

    def _markup_color(self, cam_instance):
        return MARKUP_COLORS.get(cam_instance.markup_color.currentText(), (255, 255, 255))

    # --- table helpers ---

    def _make_cell(self, text, editable=True):
        item = QTableWidgetItem(text)
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item

    def _add_overlay_table_row(self, cam_instance, ov, color_name):
        """Append one row to the markup table for the given overlay dict."""
        table = cam_instance.markup_table
        table.blockSignals(True)
        row = table.rowCount()
        table.insertRow(row)
        labels = {'hline': 'H Line', 'vline': 'V Line', 'circle': 'Circle'}
        table.setItem(row, 0, self._make_cell(labels[ov['type']], editable=False))
        if ov['type'] == 'hline':
            table.setItem(row, 1, self._make_cell("—", editable=False))
            table.setItem(row, 2, self._make_cell(str(ov['pos'])))
            table.setItem(row, 3, self._make_cell("—", editable=False))
        elif ov['type'] == 'vline':
            table.setItem(row, 1, self._make_cell(str(ov['pos'])))
            table.setItem(row, 2, self._make_cell("—", editable=False))
            table.setItem(row, 3, self._make_cell("—", editable=False))
        elif ov['type'] == 'circle':
            table.setItem(row, 1, self._make_cell(str(ov['center'][0])))
            table.setItem(row, 2, self._make_cell(str(ov['center'][1])))
            table.setItem(row, 3, self._make_cell(str(ov['radius'])))
        table.setItem(row, 4, self._make_cell(str(ov.get('thickness', 1))))
        table.blockSignals(False)
        # Color column: always-visible QComboBox; not affected by blockSignals
        color_combo = QComboBox()
        for cname in MARKUP_COLORS:
            color_combo.addItem(cname)
        color_combo.setCurrentText(color_name)
        color_combo.currentTextChanged.connect(
            lambda _, ci=cam_instance, cb=color_combo: self._on_color_combo_changed(ci, cb))
        table.setCellWidget(row, 5, color_combo)

    def _on_color_combo_changed(self, cam_instance, combo):
        """Update overlay color when the user changes the color combobox."""
        table = cam_instance.markup_table
        for row in range(table.rowCount()):
            if table.cellWidget(row, 5) is combo:
                if row < len(cam_instance.overlays):
                    cam_instance.overlays[row]['color'] = MARKUP_COLORS.get(
                        combo.currentText(), (255, 255, 255))
                break

    def sync_overlay_to_table_row(self, cam_instance, row):
        """Push overlay dict values back into the table row (blocks signals to avoid loops)."""
        table = cam_instance.markup_table
        ov = cam_instance.overlays[row]
        table.blockSignals(True)
        if ov['type'] == 'hline':
            table.item(row, 2).setText(str(ov['pos']))
        elif ov['type'] == 'vline':
            table.item(row, 1).setText(str(ov['pos']))
        elif ov['type'] == 'circle':
            table.item(row, 1).setText(str(ov['center'][0]))
            table.item(row, 2).setText(str(ov['center'][1]))
            table.item(row, 3).setText(str(ov['radius']))
        table.item(row, 4).setText(str(ov.get('thickness', 1)))
        table.blockSignals(False)
        combo = table.cellWidget(row, 5)
        if combo:
            combo.blockSignals(True)
            color_name = next((n for n, c in MARKUP_COLORS.items() if c == ov['color']), 'White')
            combo.setCurrentText(color_name)
            combo.blockSignals(False)

    def _on_markup_cell_changed(self, cam_id, row, col):
        """Update overlay dict when the user edits a table cell."""
        cam_instance = self.cameras[cam_id]
        if row >= len(cam_instance.overlays):
            return
        ov = cam_instance.overlays[row]
        item = cam_instance.markup_table.item(row, col)
        if item is None:
            return
        try:
            val = int(item.text())
        except ValueError:
            self.sync_overlay_to_table_row(cam_instance, row)
            return
        if ov['type'] == 'hline' and col == 2:
            ov['pos'] = max(0, val)
        elif ov['type'] == 'vline' and col == 1:
            ov['pos'] = max(0, val)
        elif ov['type'] == 'circle':
            cx, cy = ov['center']
            if col == 1:
                ov['center'] = (max(0, val), cy)
            elif col == 2:
                ov['center'] = (cx, max(0, val))
            elif col == 3:
                ov['radius'] = max(1, val)
        if col == 4:
            ov['thickness'] = max(1, val)
        # Re-sync to clamp any out-of-range values the user may have typed
        self.sync_overlay_to_table_row(cam_instance, row)

    # --- add / remove ---

    def add_hline_overlay(self, cam_id):
        cam_instance = self.cameras[cam_id]
        y = (cam_instance.image_height // 2) if cam_instance.image_height else 0
        color_name = cam_instance.markup_color.currentText()
        ov = {'type': 'hline', 'pos': y, 'color': self._markup_color(cam_instance), 'thickness': 1}
        cam_instance.overlays.append(ov)
        self._add_overlay_table_row(cam_instance, ov, color_name)

    def add_vline_overlay(self, cam_id):
        cam_instance = self.cameras[cam_id]
        x = (cam_instance.image_width // 2) if cam_instance.image_width else 0
        color_name = cam_instance.markup_color.currentText()
        ov = {'type': 'vline', 'pos': x, 'color': self._markup_color(cam_instance), 'thickness': 1}
        cam_instance.overlays.append(ov)
        self._add_overlay_table_row(cam_instance, ov, color_name)

    def add_circle_overlay(self, cam_id):
        cam_instance = self.cameras[cam_id]
        cx = (cam_instance.image_width  // 2) if cam_instance.image_width  else 0
        cy = (cam_instance.image_height // 2) if cam_instance.image_height else 0
        r  = (min(cam_instance.image_width, cam_instance.image_height) // 8
              if cam_instance.image_width and cam_instance.image_height else 50)
        color_name = cam_instance.markup_color.currentText()
        ov = {'type': 'circle', 'center': (cx, cy), 'radius': r,
              'color': self._markup_color(cam_instance), 'thickness': 1}
        cam_instance.overlays.append(ov)
        self._add_overlay_table_row(cam_instance, ov, color_name)

    def remove_selected_overlay(self, cam_id):
        cam_instance = self.cameras[cam_id]
        table = cam_instance.markup_table
        rows = sorted({idx.row() for idx in table.selectedIndexes()}, reverse=True)
        for row in rows:
            table.removeRow(row)
            cam_instance.overlays.pop(row)

    def apply_overlays(self, cam_instance, image):
        """Draw markup overlays onto a grayscale image.

        Returns a BGR numpy array with overlays drawn, or None if no overlays
        are defined (so the caller can skip the conversion cost).
        """
        if not cam_instance.overlays:
            return None
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = out.shape[:2]
        for ov in cam_instance.overlays:
            color = ov['color']
            t = ov.get('thickness', 1)
            if ov['type'] == 'hline':
                cv2.line(out, (0, ov['pos']), (w - 1, ov['pos']), color, t)
            elif ov['type'] == 'vline':
                cv2.line(out, (ov['pos'], 0), (ov['pos'], h - 1), color, t)
            elif ov['type'] == 'circle':
                cv2.circle(out, ov['center'], ov['radius'], color, t)
        return out

    def refresh_camera_list(self):
        """Safely refresh the camera list"""
        if self.refreshing_cameras:
            print("Camera refresh already in progress")
            return
            
        self.refreshing_cameras = True
        self.refresh_btn.setEnabled(False)
        self.statusBar().showMessage("Refreshing camera list...")
        
        # Dispose current cameras first
        for cam_id, cam_instance in self.cameras.items():
            if cam_instance.camera:
                self.disconnect_camera(cam_id)
        
        # Run SDK reinitialization after a brief delay
        QTimer.singleShot(500, self.delayed_refresh)
    
    def delayed_refresh(self):
        """Second part of refresh after cameras are disconnected"""
        try:
            self.init_sdk()
        finally:
            self.refreshing_cameras = False
            self.refresh_btn.setEnabled(True)

    def _handle_stale_camera(self, cam_id):
        """Called after too many consecutive frame errors.

        Force-disconnects without crashing on SDK errors (error 1004 etc.),
        leaving the slot clean so the user can click 'Refresh Camera List'
        and then Connect — no app restart needed.
        """
        cam_instance = self.cameras[cam_id]
        self.disconnect_camera(cam_id)
        self.statusBar().showMessage(
            f"{cam_instance.name}: connection lost — "
            "click 'Refresh Camera List' then reconnect")

    def _acquisition_loop(self, cam_id):
        """Full-rate acquisition thread.

        Runs at full hardware speed: Thorlabs uses continuous mode (frames_per_trigger=0)
        with blocking get_pending_frame_or_null(timeout=200ms); IDS uses freerun +
        is_WaitEvent(IS_SET_EVENT_FRAME, 200ms). No in-loop pacing or triggering.
        Display timer reads frames at ~30 fps; recording writes every frame with timestamps.
        """
        cam_instance = self.cameras[cam_id]
        frame_count = 0
        last_report = time.monotonic()

        print(f"Starting full-rate acquisition for {cam_id} (type={cam_instance.cam_type})")

        while not cam_instance.acq_stop_event.is_set():
            cam = cam_instance.camera
            if cam is None:
                break
            try:
                # --- Blocking wait for next frame ---
                frame = None
                if cam_instance.cam_type == 'thorlabs':
                    # image_poll_timeout_ms=200 makes this a blocking 200ms wait
                    with self.thorlabs_sdk_lock:
                        frame = cam.get_pending_frame_or_null()
                elif cam_instance.cam_type == 'ids_ueye':
                    frame = cam.wait_for_frame(timeout_ms=200)
                else:
                    break

                if cam_instance.acq_stop_event.is_set():
                    break

                if frame is None:
                    continue  # timeout; retry immediately

                frame_count += 1

                # --- Store latest frame for display (~30 fps display timer reads this) ---
                with cam_instance.frame_lock:
                    cam_instance.pending_frame = frame
                    cam_instance.new_frame_available = True
                    cam_instance.acq_frame_count += 1

                # --- Record every frame at full acquisition rate ---
                with cam_instance.video_lock:
                    if cam_instance.recording and cam_instance.video_writer:
                        try:
                            w = cam_instance.current_frame_w or cam_instance.sensor_width
                            h = cam_instance.current_frame_h or cam_instance.sensor_height
                            raw = frame.image_buffer
                            bit_depth = cam_instance.bit_depth
                            if bit_depth <= 8:
                                img = np.frombuffer(raw, dtype=np.uint8).reshape(h, w)
                            else:
                                img = np.frombuffer(raw, dtype=np.uint16).reshape(h, w)
                                img = cv2.normalize(img, None, 0, 255,
                                                    cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            cam_instance.video_writer.write(bgr)
                            cam_instance.recorded_frame_count += 1
                            cam_instance._rec_timestamps.append(time.monotonic())
                            # Hardware frame clock: Thorlabs Frame carries
                            # frame_count / time_stamp_relative_ns_or_null directly;
                            # IDSFrame carries hw_frame_count / hw_timestamp_ns from
                            # is_GetImageInfo. Missing values -> NaN sentinel.
                            fc = getattr(frame, 'hw_frame_count', None)
                            if fc is None:
                                fc = getattr(frame, 'frame_count', None)
                            ts = getattr(frame, 'hw_timestamp_ns', None)
                            if ts is None:
                                ts = getattr(frame, 'time_stamp_relative_ns_or_null', None)
                            cam_instance._rec_hw_frames.append(
                                float('nan') if fc is None else fc)
                            cam_instance._rec_hw_timestamps.append(
                                float('nan') if ts is None else ts)
                            cam_instance._rec_ring_in_use.append(
                                getattr(frame, 'buffers_in_use', None) or 0)
                            duration = time.monotonic() - cam_instance.recording_start_time
                            if ((cam_instance.record_duration_limit > 0 and
                                     duration >= cam_instance.record_duration_limit) or
                                    (cam_instance.record_frame_limit > 0 and
                                     cam_instance.recorded_frame_count >=
                                     cam_instance.record_frame_limit)):
                                cam_instance.recording = False  # display timer picks this up
                        except Exception as e:
                            print(f"Record write error for {cam_id}: {e}")

                cam_instance.consecutive_frame_errors = 0

                # --- Periodic console report every 5 seconds ---
                now = time.monotonic()
                if now - last_report >= 5.0:
                    elapsed = now - last_report
                    print(f"{cam_id} ({cam_instance.cam_type}): "
                          f"{frame_count} frames in {elapsed:.1f}s "
                          f"= {frame_count/elapsed:.1f} fps")
                    frame_count = 0
                    last_report = now

            except Exception as e:
                if cam_instance.acq_stop_event.is_set():
                    break
                cam_instance.consecutive_frame_errors += 1
                print(f"Acquisition error for {cam_instance.name}: {e}")
                if cam_instance.consecutive_frame_errors >= 10:
                    cam_instance.stale_detected.set()
                    break
                time.sleep(0.05)

        print(f"Acquisition thread for {cam_id} exiting")

    def connect_camera(self, cam_id):
        """Connect to selected camera and assign to the specified camera slot"""
        if self.camera_selector.count() == 0:
            self.show_error("No Cameras Available", "No cameras were detected. Please check connections and refresh.")
            return
            
        # Get the selected camera ID and USB port
        selected_idx = self.camera_selector.currentIndex()
        if selected_idx < 0:
            self.show_error("No Camera Selected", "Please select a camera from the dropdown.")
            return
            
        camera_info = self.camera_selector.itemData(selected_idx)
        if not camera_info:
            self.show_error("Invalid Camera Selection", "Could not retrieve camera information.")
            return
            
        cam_instance = self.cameras[cam_id]
        cam_type, *_rest = camera_info
        if cam_type == 'thorlabs':
            device_id, usb_port = _rest
        elif cam_type == 'ids_ueye':
            device_id, model_serno = _rest
            usb_port = model_serno          # used only for display
        else:
            self.show_error("Unknown camera type", repr(cam_type))
            return
        cam_instance.cam_type = cam_type

        # Check if another camera instance is already using this camera
        for other_id, other_cam in self.cameras.items():
            if other_id != cam_id and other_cam.camera_id == device_id:
                self.show_error("Camera Already In Use", 
                                f"This camera is already connected as {other_cam.name}.")
                return
        
        # Get count of currently connected cameras
        connected_count = sum(1 for c in self.cameras.values() if c.camera is not None)
        
        # Check if we're trying to connect multiple cameras
        if connected_count > 0:
            # Show warning if connecting second camera
            if getattr(self, 'available_camera_count', 0) > 1:
                msg = "You are connecting to a second camera while another is active.\n\n"
                msg += "Some Thorlabs camera models may not support simultaneous operation.\n"
                msg += "If this fails, try disconnecting the first camera before connecting another."
                
                warning_box = QMessageBox()
                warning_box.setIcon(QMessageBox.Warning)
                warning_box.setWindowTitle("Multiple Camera Warning")
                warning_box.setText(msg)
                warning_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
                
                if warning_box.exec_() == QMessageBox.Cancel:
                    return
        
        # Clean up existing camera if there is one
        if cam_instance.camera:
            try:
                with cam_instance.camera_lock:
                    cam_instance.camera.disarm()
                    cam_instance.camera.dispose()
                    # Add delay after disposing camera
                    time.sleep(0.2)
            except Exception as e:
                error_msg = f"Error disconnecting from {cam_instance.name}: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                cam_instance.debug_label.setText(f"Error: {error_msg}")
                self.show_error(f"Error disconnecting from {cam_instance.name}", str(e))
        
        # Stop the timer temporarily while connecting to camera
        timer_was_active = self.timer.isActive()
        if timer_was_active:
            self.timer.stop()
            
        # Use a separate try-except block for each major step to provide better error reporting
        try:
            # Connect to the camera
            print(f"Connecting to camera {device_id} for {cam_id}")
            self.statusBar().showMessage(f"Connecting to camera {device_id}...")
            
            with self.sdk_lock:
                if cam_type == 'thorlabs':
                    if not self.sdk:
                        raise RuntimeError("SDK is no longer valid")

                    # Add additional delay to ensure system stability
                    time.sleep(0.5)

                    # Try to connect with retry mechanism
                    max_retries = 3
                    retry_count = 0
                    last_error = None

                    while retry_count < max_retries:
                        try:
                            cam_instance.camera = self.sdk.open_camera(device_id)
                            cam_instance.camera_id = device_id
                            cam_instance.usb_port = usb_port
                            break  # Successfully connected
                        except Exception as e:
                            last_error = e
                            retry_count += 1
                            print(f"Connection attempt {retry_count} failed: {e}")
                            time.sleep(1.0)  # Wait before retrying

                    if retry_count == max_retries:
                        raise RuntimeError(f"Failed to connect after {max_retries} attempts: {last_error}")

                elif cam_type == 'ids_ueye':
                    cam_instance.camera    = IDSCamera(device_id)
                    cam_instance.camera_id = device_id
                    cam_instance.usb_port  = usb_port
        
            # Configure camera step
            try:
                # Configure camera with lock to ensure thread safety
                with cam_instance.camera_lock:
                    print(f"Configuring camera {device_id}")
                    if cam_type == 'thorlabs':
                        # Continuous mode: SDK delivers frames as fast as hardware allows.
                        # get_pending_frame_or_null() with 200ms timeout blocks until frame ready.
                        cam_instance.camera.frames_per_trigger_zero_for_unlimited = 0
                        cam_instance.camera.image_poll_timeout_ms = 200
                        print("Thorlabs: frames_per_trigger=0 (continuous), poll_timeout=200ms")
                    else:
                        # IDS: arm() will set software trigger mode
                        cam_instance.camera.frames_per_trigger_zero_for_unlimited = 0
                    cam_instance.camera.exposure_time_us = int(cam_instance.exposure_ms * 1000)  # Convert ms to μs
            except Exception as e:
                error_msg = f"Error configuring camera: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                # Clean up the camera connection
                with self.sdk_lock:
                    try:
                        if cam_instance.camera:
                            cam_instance.camera.dispose()
                            cam_instance.camera = None
                    except:
                        pass
                raise RuntimeError(error_msg)
            
            # Arm camera step
            try:
                # Start the camera
                with cam_instance.camera_lock:
                    print(f"Arming camera {device_id}")
                    # Thorlabs: large SDK pool. IDS: already allocated IDS_RING_BUFFERS
                    # at construction — pass None so arm() keeps that ring (don't shrink it).
                    num_buffers = 30 if cam_type == 'thorlabs' else None
                    cam_instance.camera.arm(num_buffers)
                    print(f"Armed with {num_buffers if num_buffers else IDS_RING_BUFFERS} buffers")
                    time.sleep(0.2)  # Short delay after arming
                    # Continuous mode: one software trigger starts the stream; acquisition
                    # loop consumes frames with get_pending_frame_or_null() — no re-triggering.
                    if cam_type == 'thorlabs':
                        cam_instance.camera.issue_software_trigger()
                        print("Thorlabs: issued initial trigger to start continuous stream")
            except Exception as e:
                error_msg = f"Error arming camera: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                # Clean up the camera connection
                with self.sdk_lock:
                    try:
                        if cam_instance.camera:
                            cam_instance.camera.dispose()
                            cam_instance.camera = None
                    except:
                        pass
                raise RuntimeError(error_msg)

            # Cache sensor info for thread-safe display (no lock needed — not yet started)
            cam_instance.bit_depth     = cam_instance.camera.bit_depth
            cam_instance.sensor_width  = cam_instance.camera.sensor_width_pixels
            cam_instance.sensor_height = cam_instance.camera.sensor_height_pixels
            # Active frame dimensions start at full sensor; updated by apply_roi/reset_roi
            cam_instance.roi = None
            cam_instance.current_frame_w = cam_instance.sensor_width
            cam_instance.current_frame_h = cam_instance.sensor_height
            # Pre-fill ROI width/height fields with full sensor size
            cam_instance.roi_w.setText(str(cam_instance.sensor_width))
            cam_instance.roi_h.setText(str(cam_instance.sensor_height))

            # --- IDS pixel clock UI ---
            if cam_type == 'ids_ueye':
                try:
                    clocks = cam_instance.camera.get_pixel_clock_list()
                    current = cam_instance.camera.get_pixel_clock()
                    cam_instance.pixel_clock_combo.blockSignals(True)
                    cam_instance.pixel_clock_combo.clear()
                    for c in clocks:
                        cam_instance.pixel_clock_combo.addItem(str(c))
                    cam_instance.pixel_clock_combo.setCurrentText(str(current))
                    cam_instance.pixel_clock_combo.blockSignals(False)
                    cam_instance.pixel_clock_label.setVisible(True)
                    cam_instance.pixel_clock_combo.setVisible(True)
                except Exception as e:
                    print(f"Could not read pixel clock info for {cam_instance.name}: {e}")

            # Command the initial hardware frame rate (default fps) now that the
            # camera is armed — otherwise it free-runs at its ROI-dependent max.
            with cam_instance.camera_lock:
                try:
                    self._apply_fps_exposure(cam_instance, cam_instance.fps)
                except Exception as e:
                    print(f"Initial frame-rate apply failed for {cam_instance.name}: {e}")

            # Start per-camera acquisition thread
            cam_instance.stale_detected.clear()
            cam_instance.acq_stop_event.clear()
            cam_instance.new_frame_available = False
            cam_instance.acq_thread = threading.Thread(
                target=self._acquisition_loop, args=(cam_id,),
                daemon=True, name=f"acq-{cam_id}")
            cam_instance.acq_thread.start()
            print(f"Started acquisition thread for {cam_id}")

            # Enable ROI controls now that camera is connected
            cam_instance.roi_apply_btn.setEnabled(True)
            cam_instance.roi_reset_btn.setEnabled(True)

            # Update UI
            self.update_connect_btn()
            
            cam_instance.status_label.setText(f"Connected to {cam_instance.name} ({usb_port})")
            
            # Update debug info
            try:
                camera_info = f"Model: {cam_instance.camera.model}, "
                camera_info += f"SN: {cam_instance.camera.serial_number}, "
                camera_info += f"Firmware: {cam_instance.camera.firmware_version}"
                cam_instance.debug_label.setText(camera_info)
            except Exception as e:
                cam_instance.debug_label.setText(f"Camera info error: {str(e)}")
            
            # Start the display refresh timer (decoupled from capture rate).
            self._ensure_display_timer()
            cam_instance.panel_widget.setVisible(True)

            msg = f"{cam_instance.name} connected successfully"
            print(msg)
            self.statusBar().showMessage(msg)
            
        except Exception as e:
            error_msg = f"Failed to connect to camera: {str(e)}"
            print(f"Error connecting to {cam_instance.name}: {error_msg}")
            traceback.print_exc()
            cam_instance.debug_label.setText(f"Connection error: {error_msg}")
            self.show_error(f"Error connecting to {cam_instance.name}", error_msg)
            self.statusBar().showMessage(f"Error connecting to {cam_instance.name}: {error_msg}")
            
            # Restore timer if it was active
            if timer_was_active and not self.timer.isActive():
                self._ensure_display_timer()
                
    def safe_camera_operation(self, func, *args, **kwargs):
        """Execute a function with proper locking to ensure thread safety"""
        with self.sdk_lock:
            return func(*args, **kwargs)

    def disconnect_camera(self, cam_id):
        """Disconnect the specified camera"""
        cam_instance = self.cameras[cam_id]
        
        if cam_instance.recording:
            self.toggle_recording(cam_id)  # Stop recording if active
        
        if cam_instance.camera:
            print(f"Disconnecting camera {cam_id}")
            # Signal the acquisition thread to stop
            cam_instance.acq_stop_event.set()
            # disarm() interrupts any blocking FreezeVideo / get_pending_frame in the thread.
            try:
                with cam_instance.camera_lock:
                    cam_instance.camera.disarm()
            except Exception as e:
                print(f"Warning: disarm() error for {cam_instance.name} (stale handle?): {e}")

            # Join the acquisition thread BEFORE dispose — the thread may still be
            # inside an SDK call; disposing the handle while it runs causes a segfault.
            if cam_instance.acq_thread and cam_instance.acq_thread.is_alive():
                cam_instance.acq_thread.join(timeout=2.0)
            cam_instance.acq_thread = None
            cam_instance.new_frame_available = False

            # Now safe to dispose — the acquisition thread is guaranteed done.
            dispose_failed = False
            try:
                with cam_instance.camera_lock:
                    cam_instance.camera.dispose()
            except Exception as e:
                dispose_failed = True
                print(f"Warning: dispose() error for {cam_instance.name} (stale handle?): {e}")

            # If Thorlabs dispose failed, the SDK's internal handle table is corrupted.
            # Reinitialize the SDK so the next open_camera() call doesn't segfault.
            if dispose_failed and cam_instance.cam_type == 'thorlabs':
                print(f"Reinitializing Thorlabs SDK after failed dispose")
                with self.sdk_lock:
                    try:
                        self.sdk.dispose()
                    except Exception:
                        pass
                    time.sleep(0.3)
                    self.sdk = TLCameraSDK()

            # Always clean up local state regardless of SDK errors above
            cam_instance.camera = None
            cam_instance.camera_id = ""
            cam_instance.consecutive_frame_errors = 0
            cam_instance.roi = None
            cam_instance.current_frame_w = 0
            cam_instance.current_frame_h = 0
            cam_instance.roi_apply_btn.setEnabled(False)
            cam_instance.roi_reset_btn.setEnabled(False)
            cam_instance.pixel_clock_label.setVisible(False)
            cam_instance.pixel_clock_combo.setVisible(False)
            cam_instance.panel_widget.setVisible(False)

            self.update_connect_btn()

            cam_instance.status_label.setText(f"{cam_instance.name} not connected")
            cam_instance.image_label.setText(f"Connect to {cam_instance.name} to view feed")
            cam_instance.image_label.setPixmap(QPixmap())
            cam_instance.debug_label.setText("Debug info: Disconnected")

            msg = f"{cam_instance.name} disconnected"
            print(msg)
            self.statusBar().showMessage(msg)

            if not self.cameras["cam1"].camera and not self.cameras["cam2"].camera:
                print("Stopping timer - no cameras connected")
                self.timer.stop()
    
    def update_frames(self):
        for cam_id, cam_instance in self.cameras.items():
            if cam_instance.stale_detected.is_set():
                cam_instance.stale_detected.clear()
                self._handle_stale_camera(cam_id)
            elif cam_instance.camera:
                self.update_camera_frame(cam_id)
    
    def update_camera_frame(self, cam_id):
        """Read the latest frame from the acquisition thread and display it."""
        cam_instance = self.cameras[cam_id]

        with cam_instance.frame_lock:
            if not cam_instance.new_frame_available:
                return
            frame = cam_instance.pending_frame
            cam_instance.new_frame_available = False

        if frame is None:
            return

        try:
            width  = frame.image_buffer_size_pixels_horizontal
            height = frame.image_buffer_size_pixels_vertical
        except AttributeError:
            # Thorlabs FrameAndMetadata does not carry per-frame dimensions;
            # fall back to active frame dimensions (ROI or full sensor)
            width  = cam_instance.current_frame_w or cam_instance.sensor_width
            height = cam_instance.current_frame_h or cam_instance.sensor_height
        if not width or not height:
            return

        bit_depth = cam_instance.bit_depth
        image_data = frame.image_buffer
        if bit_depth <= 8:
            arr = np.frombuffer(image_data, dtype=np.uint8)
        else:
            arr = np.frombuffer(image_data, dtype=np.uint16)
        if arr.size != width * height:
            # Stale frame from before an ROI change (buffer size doesn't match
            # current dimensions) — discard silently and wait for next frame.
            return
        if bit_depth <= 8:
            image = arr.reshape(height, width)
        else:
            image = cv2.normalize(arr.reshape(height, width), None, 0, 255,
                                  cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Apply display rotation
        rot = cam_instance.rotation
        if rot == 90:
            image = np.rot90(image, k=3)  # rot90 k=3 is clockwise 90°
        elif rot == 180:
            image = np.rot90(image, k=2)
        elif rot == 270:
            image = np.rot90(image, k=1)  # rot90 k=1 is counter-clockwise 90° = clockwise 270°
        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]

        cam_instance.last_frame  = image
        cam_instance.image_width  = width
        cam_instance.image_height = height

        overlay_image = self.apply_overlays(cam_instance, image)

        # Update recording label (frame writes happen in acquisition thread)
        auto_stopped = False
        with cam_instance.video_lock:
            if cam_instance.recording and cam_instance.video_writer:
                duration = time.time() - cam_instance.recording_start_time
                cam_instance.recording_label.setText(
                    f"Recording: {duration:.1f}s, "
                    f"Frames: {cam_instance.recorded_frame_count}")
            elif not cam_instance.recording and cam_instance.video_writer:
                # Auto-stop was triggered by acquisition thread (duration/frame limit)
                cam_instance.video_writer.release()
                cam_instance.video_writer = None
                cam_instance.record_button.setText("Start Recording")
                auto_stopped = True
        if auto_stopped:
            # Disable IDS hw-info capture and write the same sidecars as a manual stop.
            if cam_instance.cam_type == 'ids_ueye' and cam_instance.camera:
                try:
                    cam_instance.camera.capture_hw_info = False
                except Exception:
                    pass
            status_msg = self._finalize_recording(cam_instance, time.time())
            cam_instance.status_label.setText(
                status_msg or f"Recording stopped - {cam_instance.name}")

        # Display
        if overlay_image is not None:
            display_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            q_image = QImage(display_image.data, width, height, width * 3, QImage.Format_RGB888)
        else:
            q_image = QImage(image.data, width, height, width, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        cam_instance.image_label.setPixmap(pixmap.scaled(
            cam_instance.image_label.width(),
            cam_instance.image_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation))

        # FPS counter (actual acquisition rate from acquisition thread)
        elapsed = time.time() - cam_instance.acq_last_time
        if elapsed >= 1.0:
            actual_fps = cam_instance.acq_frame_count / elapsed
            cam_instance.fps_label.setText(f"{actual_fps:.1f}")
            print(f"Display FPS update for {cam_id}: {actual_fps:.1f} fps ({cam_instance.acq_frame_count} frames in {elapsed:.1f}s)")
            cam_instance.acq_frame_count = 0
            cam_instance.acq_last_time = time.time()
    
    def exposure_slider_changed(self, cam_id, value):
        """Handle exposure slider change for a specific camera"""
        cam_instance = self.cameras[cam_id]
        # Convert slider value (which is integer) to actual exposure in ms
        exposure_ms = value / 10.0
        cam_instance.exposure_value.blockSignals(True)
        cam_instance.exposure_value.setValue(exposure_ms)
        cam_instance.exposure_value.blockSignals(False)
        self.set_exposure(cam_id, exposure_ms)
    
    def set_exposure(self, cam_id, value_ms):
        """Set exposure for a specific camera"""
        cam_instance = self.cameras[cam_id]
        cam_instance.exposure_ms = value_ms
        
        if cam_instance.camera:
            try:
                # Convert from ms to μs for the camera
                with cam_instance.camera_lock:
                    cam_instance.camera.exposure_time_us = int(value_ms * 1000)
                cam_instance.status_label.setText(f"Exposure set to {value_ms} ms")
                # Update slider if value was changed directly
                slider_value = int(value_ms * 10)
                if cam_instance.exposure_slider.value() != slider_value:
                    cam_instance.exposure_slider.blockSignals(True)
                    cam_instance.exposure_slider.setValue(slider_value)
                    cam_instance.exposure_slider.blockSignals(False)
            except Exception as e:
                error_msg = f"Error setting exposure: {str(e)}"
                print(error_msg)
                if self.debug_checkbox.isChecked():
                    traceback.print_exc()
                    cam_instance.debug_label.setText(f"Exposure error: {error_msg}")
    
    def set_pixel_clock(self, cam_id, val_str):
        cam_instance = self.cameras[cam_id]
        if not cam_instance.camera or cam_instance.cam_type != 'ids_ueye':
            return
        try:
            cam_instance.camera.set_pixel_clock(int(val_str))
            print(f"{cam_instance.name}: pixel clock → {val_str} MHz")
        except Exception as e:
            print(f"Error setting pixel clock for {cam_instance.name}: {e}")

    def rotate_camera(self, cam_id):
        """Cycle display rotation by 90° (0 → 90 → 180 → 270 → 0)."""
        cam_instance = self.cameras[cam_id]
        cam_instance.rotation = (cam_instance.rotation + 90) % 360
        label = f"{cam_instance.rotation}°" if cam_instance.rotation else "0°"
        cam_instance.rotate_btn.setText(f"Rotate 90° ({label})")

    def framerate_slider_changed(self, cam_id, value):
        """Handle framerate slider change for a specific camera"""
        cam_instance = self.cameras[cam_id]
        fps = value
        cam_instance.framerate_value.blockSignals(True)
        cam_instance.framerate_value.setValue(fps)
        cam_instance.framerate_value.blockSignals(False)
        self.set_framerate(cam_id, fps)
    
    def _ensure_display_timer(self):
        """Run the display refresh timer at DISPLAY_FPS_CAP while any camera is
        connected, else stop it. Independent of capture fps (see DISPLAY_FPS_CAP)."""
        any_connected = any(c.camera for c in self.cameras.values())
        if any_connected:
            interval = int(1000 / DISPLAY_FPS_CAP)
            if self.timer.isActive():
                self.timer.stop()
            self.timer.start(interval)
        elif self.timer.isActive():
            self.timer.stop()

    def _thorlabs_set_frame_rate(self, cam, fps):
        """Enable + set Thorlabs hardware frame-rate control; return achieved fps.

        No-op (returns None) on models that don't support it (range.max <= 0).
        """
        try:
            rng = cam.frame_rate_control_value_range
        except Exception as e:
            print(f"Thorlabs frame_rate_control_value_range unavailable: {e}")
            return None
        fmax = getattr(rng, 'max', 0) or 0
        fmin = getattr(rng, 'min', 0) or 0
        if fmax <= 0:
            print("Thorlabs: model does not support frame-rate control; leaving free-run")
            return None
        target = min(max(float(fps), fmin), fmax)
        cam.is_frame_rate_control_enabled = True
        cam.frame_rate_control_value = target
        return target

    def _apply_fps_exposure(self, cam_instance, fps):
        """Command a fixed hardware frame rate on a connected camera and reconcile
        exposure to fit the frame period. Returns (achieved_fps, warn_or_None).

        Dispatches by cam_type. Caller must hold camera_lock. Safe to call for
        either camera type; a model without frame-rate support is left free-run.
        """
        cam = cam_instance.camera
        if cam is None:
            return None, None

        # Query the achievable range so we can clamp fps + exposure sensibly.
        fps_min, fps_max = 1.0, 0.0
        if cam_instance.cam_type == 'ids_ueye':
            try:
                fps_min, fps_max = cam.get_frame_rate_range()
            except Exception:
                fps_min, fps_max = 1.0, 0.0
        elif cam_instance.cam_type == 'thorlabs':
            try:
                rng = cam.frame_rate_control_value_range
                fps_min = getattr(rng, 'min', 1.0) or 1.0
                fps_max = getattr(rng, 'max', 0.0) or 0.0
            except Exception:
                fps_min, fps_max = 1.0, 0.0

        fps, exposure_ms, warn = clamp_fps_exposure(
            fps, cam_instance.exposure_ms, fps_min, fps_max)

        # Apply frame rate first (defines the period), then exposure to fit it.
        achieved = None
        if cam_instance.cam_type == 'ids_ueye':
            achieved = cam.set_frame_rate(fps)
        elif cam_instance.cam_type == 'thorlabs':
            achieved = self._thorlabs_set_frame_rate(cam, fps)

        # Reconcile exposure (both SDKs: exposure setter takes μs here).
        if exposure_ms != cam_instance.exposure_ms:
            cam_instance.exposure_ms = exposure_ms
        try:
            cam.exposure_time_us = int(exposure_ms * 1000)
        except Exception as e:
            print(f"Exposure re-apply failed: {e}")

        return achieved, warn

    def _reapply_camera_settings(self, cam_instance):
        """Re-apply {frame rate, exposure} after a re-arm so they survive ROI changes.

        is_SetFrameRate / Thorlabs frame-rate control do NOT persist across the
        disarm+arm cycle an ROI change performs, so any commanded fps must be
        re-issued afterwards. Caller must hold camera_lock. Best-effort.
        """
        if cam_instance.camera is None:
            return
        try:
            self._apply_fps_exposure(cam_instance, cam_instance.fps)
        except Exception as e:
            print(f"Re-apply camera settings failed for {cam_instance.name}: {e}")

    def set_framerate(self, cam_id, fps):
        """Command a fixed hardware frame rate for a camera (not just the display).

        Previously this only restarted the display timer; the sensor free-ran at
        its ROI-dependent max. Now it issues the real SDK frame-rate command and
        reconciles exposure, keeping the display timer decoupled at DISPLAY_FPS_CAP.
        """
        cam_instance = self.cameras[cam_id]
        cam_instance.fps = fps

        if cam_instance.camera:
            try:
                with cam_instance.camera_lock:
                    achieved, warn = self._apply_fps_exposure(cam_instance, fps)
                if achieved is not None:
                    msg = f"Frame rate → {achieved:.1f} fps (requested {fps})"
                else:
                    msg = f"Frame rate {fps} fps requested (hardware control unavailable)"
                if warn:
                    msg += f" [{warn}]"
                cam_instance.status_label.setText(msg)
                print(f"{cam_instance.name}: {msg}")
            except Exception as e:
                error_msg = f"Error setting frame rate: {e}"
                print(error_msg)
                if self.debug_checkbox.isChecked():
                    traceback.print_exc()
                cam_instance.status_label.setText(error_msg)
        else:
            cam_instance.status_label.setText(f"Frame rate set to {fps} FPS (not connected)")

        # Display refresh is decoupled from capture rate.
        self._ensure_display_timer()

        # Update slider if value was changed directly
        if cam_instance.framerate_slider.value() != fps:
            cam_instance.framerate_slider.blockSignals(True)
            cam_instance.framerate_slider.setValue(fps)
            cam_instance.framerate_slider.blockSignals(False)

    def apply_roi(self, cam_id):
        """Stop acquisition, apply ROI to camera, restart acquisition thread."""
        cam_instance = self.cameras[cam_id]
        if not cam_instance.camera:
            return

        try:
            x = int(cam_instance.roi_x.text())
            y = int(cam_instance.roi_y.text())
            w = int(cam_instance.roi_w.text())
            h = int(cam_instance.roi_h.text())
        except ValueError:
            self.show_error("ROI Error", "ROI fields must be integers.")
            return

        try:
            if cam_instance.cam_type == 'ids_ueye':
                x, y, w, h = clamp_roi(x, y, w, h,
                                        cam_instance.sensor_width,
                                        cam_instance.sensor_height,
                                        step_x=4, step_y=2)
            else:
                # Enforce Thorlabs SDK minimum lower-right coordinate to prevent crash
                rr = cam_instance.camera.roi_range
                x, y, w, h = clamp_roi(x, y, w, h,
                                        cam_instance.sensor_width,
                                        cam_instance.sensor_height,
                                        step_x=1, step_y=1,
                                        min_lrx=rr.lower_right_x_pixels_min,
                                        min_lry=rr.lower_right_y_pixels_min)
        except ValueError as e:
            self.show_error("ROI Error", str(e))
            return

        # Stop acquisition thread (join must succeed before touching camera handle)
        cam_instance.acq_stop_event.set()
        if cam_instance.acq_thread and cam_instance.acq_thread.is_alive():
            cam_instance.acq_thread.join(timeout=3.0)
        cam_instance.acq_stop_event.clear()

        try:
            with cam_instance.camera_lock:
                if cam_instance.cam_type == 'ids_ueye':
                    cam_instance.camera.set_aoi(x, y, w, h)
                elif cam_instance.cam_type == 'thorlabs':
                    from thorlabs_tsi_sdk.tl_camera import ROI
                    cam_instance.camera.disarm()
                    cam_instance.camera.roi = ROI(x, y, x + w - 1, y + h - 1)
                    cam_instance.camera.frames_per_trigger_zero_for_unlimited = 0
                    cam_instance.camera.image_poll_timeout_ms = 200
                    cam_instance.camera.arm(30)
                    cam_instance.camera.issue_software_trigger()  # start continuous stream
                    # Read back what the SDK actually applied — it may silently expand
                    # the ROI to satisfy hardware alignment / minimum constraints.
                    applied = cam_instance.camera.roi
                    w = applied.lower_right_x_pixels - applied.upper_left_x_pixels + 1
                    h = applied.lower_right_y_pixels - applied.upper_left_y_pixels + 1
                    x, y = applied.upper_left_x_pixels, applied.upper_left_y_pixels
                    print(f"Thorlabs ROI applied by SDK: {w}×{h} at ({x},{y})")

                # Re-apply commanded fps/exposure — the re-arm above resets them.
                self._reapply_camera_settings(cam_instance)

            # Flush stale frame before updating dimensions — prevents display thread from
            # reshaping an old full-frame buffer using the new smaller ROI dimensions.
            with cam_instance.frame_lock:
                cam_instance.pending_frame = None
                cam_instance.new_frame_available = False

            cam_instance.roi = (x, y, w, h)
            cam_instance.current_frame_w = w
            cam_instance.current_frame_h = h

            # Reflect actual SDK-applied values back into UI fields
            cam_instance.roi_x.setText(str(x))
            cam_instance.roi_y.setText(str(y))
            cam_instance.roi_w.setText(str(w))
            cam_instance.roi_h.setText(str(h))

            # Restart acquisition thread
            cam_instance.acq_thread = threading.Thread(
                target=self._acquisition_loop, args=(cam_id,),
                daemon=True, name=f"acq-{cam_id}")
            cam_instance.acq_thread.start()

            cam_instance.status_label.setText(
                f"ROI applied: {w}×{h} at ({x},{y})")

        except Exception as e:
            self.show_error("ROI Apply Error", str(e))
            traceback.print_exc()
            cam_instance.status_label.setText(
                f"ROI apply failed — try Refresh & reconnect: {e}")

    def reset_roi(self, cam_id):
        """Restore full-sensor ROI and restart acquisition thread."""
        cam_instance = self.cameras[cam_id]
        if not cam_instance.camera:
            return

        cam_instance.acq_stop_event.set()
        if cam_instance.acq_thread and cam_instance.acq_thread.is_alive():
            cam_instance.acq_thread.join(timeout=3.0)
        cam_instance.acq_stop_event.clear()

        try:
            with cam_instance.camera_lock:
                if cam_instance.cam_type == 'ids_ueye':
                    cam_instance.camera.reset_aoi()
                elif cam_instance.cam_type == 'thorlabs':
                    # Reset via an EXPLICIT full-sensor ROI, not roi=None: this SDK's
                    # roi setter unpacks its argument ("cannot unpack non-iterable
                    # NoneType" + wedged camera on reconnect). Mirrors the working
                    # apply_roi path.
                    from thorlabs_tsi_sdk.tl_camera import ROI
                    sw = cam_instance.camera.sensor_width_pixels
                    sh = cam_instance.camera.sensor_height_pixels
                    cam_instance.camera.disarm()
                    cam_instance.camera.roi = ROI(0, 0, sw - 1, sh - 1)
                    cam_instance.camera.frames_per_trigger_zero_for_unlimited = 0
                    cam_instance.camera.image_poll_timeout_ms = 200
                    cam_instance.camera.arm(30)
                    cam_instance.camera.issue_software_trigger()  # start continuous stream

                # Re-apply commanded fps/exposure — the re-arm above resets them.
                self._reapply_camera_settings(cam_instance)

            with cam_instance.frame_lock:
                cam_instance.pending_frame = None
                cam_instance.new_frame_available = False

            cam_instance.roi = None
            cam_instance.current_frame_w = cam_instance.sensor_width
            cam_instance.current_frame_h = cam_instance.sensor_height

            cam_instance.roi_x.setText("0")
            cam_instance.roi_y.setText("0")
            cam_instance.roi_w.setText(str(cam_instance.sensor_width))
            cam_instance.roi_h.setText(str(cam_instance.sensor_height))

            cam_instance.acq_thread = threading.Thread(
                target=self._acquisition_loop, args=(cam_id,),
                daemon=True, name=f"acq-{cam_id}")
            cam_instance.acq_thread.start()

            cam_instance.status_label.setText("ROI reset to full sensor")

        except Exception as e:
            self.show_error("ROI Reset Error", str(e))
            traceback.print_exc()

    def toggle_recording(self, cam_id):
        """Toggle recording for a specific camera"""
        cam_instance = self.cameras[cam_id]
        
        if not cam_instance.camera:
            self.show_error("Camera Not Connected", f"{cam_instance.name} is not connected.")
            return
            
        if not cam_instance.recording:
            # Start recording
            try:
                filename, _ = QFileDialog.getSaveFileName(
                    self, f"Save Video - {cam_instance.name}", 
                    f"{cam_instance.name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    "Video Files (*.mp4)"
                )
                if filename:
                    # Use current ROI dimensions (full sensor if no ROI applied)
                    width  = cam_instance.current_frame_w or cam_instance.sensor_width
                    height = cam_instance.current_frame_h or cam_instance.sensor_height
                    
                    # Initialize video writer with MJPG codec which has good compatibility with MP4
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    # Ensure filename has .mp4 extension
                    if not filename.lower().endswith('.mp4'):
                        filename = filename + '.mp4'
                    
                    # Container fps: prefer the camera's live measured rate so the
                    # mp4 plays back at the correct SPEED even when the achievable
                    # rate differs from the commanded fps (e.g. IDS min-fps floor at
                    # small ROI). The TRUE per-frame timing still lives in the
                    # hardware-clock sidecar; this only sets casual-playback speed.
                    container_fps = cam_instance.fps
                    try:
                        with cam_instance.camera_lock:
                            measured = cam_instance.camera.get_measured_frame_rate_fps()
                        if measured and measured > 0:
                            container_fps = measured
                    except Exception:
                        pass

                    with cam_instance.video_lock:
                        cam_instance.video_writer = cv2.VideoWriter(
                            filename, fourcc, container_fps, (width, height)
                        )
                        if cam_instance.video_writer.isOpened():
                            cam_instance.recording = True
                            cam_instance.recording_start_time = time.time()
                            cam_instance._rec_filename = filename  # saved for timestamps file on stop
                            # Initialize recording limits
                            cam_instance.record_duration_limit = cam_instance.duration_spinbox.value()
                            cam_instance.record_frame_limit = cam_instance.framecount_spinbox.value()
                            cam_instance.recorded_frame_count = 0
                            cam_instance._rec_timestamps = []  # reset for new recording
                            cam_instance._rec_hw_frames = []
                            cam_instance._rec_hw_timestamps = []
                            cam_instance._rec_ring_in_use = []
                            # Enable per-frame hardware-info capture on IDS while recording
                            # (Thorlabs Frame carries it for free; IDS pays a small cost).
                            if cam_instance.cam_type == 'ids_ueye' and cam_instance.camera:
                                cam_instance.camera.capture_hw_info = True
                        else:
                            cam_instance.video_writer = None
                    if cam_instance.recording:
                        cam_instance.record_button.setText("Stop Recording")
                        cam_instance.recording_label.setText("Recording started")
                        cam_instance.status_label.setText(f"Recording to {os.path.basename(filename)}")
                    else:
                        self.show_error("Recording Error", f"Failed to create video writer for {cam_instance.name}")
                        print(f"Failed to open video writer for {filename}")
            except Exception as e:
                error_msg = f"Recording error for {cam_instance.name}: {str(e)}"
                print(error_msg)
                traceback.print_exc()
                cam_instance.debug_label.setText(f"Recording error: {error_msg}")
                self.show_error(f"Recording Error - {cam_instance.name}", str(e))
                cam_instance.video_writer = None
        else:
            # Stop recording
            with cam_instance.video_lock:
                if cam_instance.video_writer:
                    try:
                        cam_instance.video_writer.release()
                    except Exception as e:
                        print(f"Error releasing video writer: {e}")
                    cam_instance.video_writer = None
                cam_instance.recording = False
            # Stop the extra per-frame hardware-info capture on IDS.
            if cam_instance.cam_type == 'ids_ueye' and cam_instance.camera:
                try:
                    cam_instance.camera.capture_hw_info = False
                except Exception:
                    pass
            cam_instance.record_button.setText("Start Recording")
            stop_unix = time.time()
            status_msg = self._finalize_recording(cam_instance, stop_unix)
            if status_msg:
                cam_instance.status_label.setText(status_msg)
            else:
                cam_instance.recording_label.setText("Not Recording")
                cam_instance.status_label.setText(f"Recording stopped - {cam_instance.name}")

    def _finalize_recording(self, cam_instance, stop_unix):
        """Write recording sidecars and update UI labels. Returns a status string,
        or None if nothing was recorded. Delegates the file I/O + stats to the
        module-level write_recording_sidecars (which is unit-testable without Qt)."""
        stats = write_recording_sidecars(
            getattr(cam_instance, '_rec_filename', ''),
            cam_instance._rec_timestamps,
            cam_instance._rec_hw_frames,
            cam_instance._rec_hw_timestamps,
            cam_instance._rec_ring_in_use,
            start_unix=cam_instance.recording_start_time,
            stop_unix=stop_unix,
            camera=cam_instance.name,
            width=cam_instance.current_frame_w or cam_instance.sensor_width,
            height=cam_instance.current_frame_h or cam_instance.sensor_height,
            nominal_fps=cam_instance.fps)
        if stats is None:
            return None
        fps_for_report = stats["hw_fps"] or stats["sw_fps"]
        cam_instance.recording_label.setText(
            f"Recorded: {stats['n']} frames at {fps_for_report:.1f} fps "
            f"({stats['dropped']} drops)")
        return (f"Saved {stats['n']} frames at {fps_for_report:.1f} fps "
                f"(drops={stats['dropped']}, ring peak={stats['ring_peak']})")
    
    def show_error(self, title, message):
        """Display an error dialog with the given title and message"""
        print(f"ERROR: {title} - {message}")
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Critical)
        error_box.setWindowTitle(title)
        error_box.setText(message)
        error_box.setStandardButtons(QMessageBox.Ok)
        error_box.exec_()
    
    def closeEvent(self, event):
        # Cleanup when application is closed
        print("Application closing, cleaning up...")
        
        # Stop the timer first
        if self.timer.isActive():
            self.timer.stop()
        
        for cam_id, cam_instance in self.cameras.items():
            if cam_instance.recording and cam_instance.video_writer:
                try:
                    print(f"Releasing video writer for {cam_id}")
                    cam_instance.video_writer.release()
                except Exception as e:
                    print(f"Error releasing video writer for {cam_id}: {e}")
            
            if cam_instance.camera:
                print(f"Disposing camera {cam_id}")
                # Stop the acquisition thread before touching the camera handle.
                cam_instance.acq_stop_event.set()
                try:
                    cam_instance.camera.disarm()
                except Exception as e:
                    print(f"Error disarming camera {cam_id}: {e}")
                if cam_instance.acq_thread and cam_instance.acq_thread.is_alive():
                    cam_instance.acq_thread.join(timeout=2.0)
                cam_instance.acq_thread = None
                try:
                    cam_instance.camera.dispose()
                except Exception as e:
                    print(f"Error disposing camera {cam_id}: {e}")
        
        # Add a small delay to ensure cameras are properly disposed
        time.sleep(0.5)
        
        if self.sdk:
            try:
                print("Disposing SDK")
                self.sdk.dispose()
            except Exception as e:
                print(f"Error disposing SDK: {e}")
        
        print("Cleanup complete")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThorlabsCameraApp()
    window.show()
    sys.exit(app.exec_())