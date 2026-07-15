"""End-to-end logic test of ids_ueye.IDSUEye against the mock pyueye SDK.

Runs WITHOUT hardware or the IDS driver stack: the ``pyueye`` package in this
directory is a mock that emulates the SDK subset the driver uses (one
DCC1545M-like camera with real-time frame pacing, AOI alignment asserts, and
event semantics — including the stale-signal and re-arm-resets-fps behaviors
the review fixes guard against). It shadows any real pyueye on sys.path.

Run with::

    python tests/test_driver_mock.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))        # ids-ueye/tests
PKG_DIR = os.path.dirname(HERE)                          # ids-ueye
DOCK = os.path.dirname(PKG_DIR)                          # dock root
sys.path.insert(0, PKG_DIR)
sys.path.insert(0, DOCK)
sys.path.insert(0, HERE)                                 # mock pyueye wins

import numpy as np
from pyueye import ueye as mock                    # the mock
from ids_ueye import IDSUEye, list_devices
from camera_dock.base import CameraBase

failures = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


print("1. enumeration")
devs = list_devices()
check("list_devices finds mock camera", len(devs) == 1 and devs[0]["model"] == "DCC1545M-GL-MOCK", str(devs))

print("2. connect / protocol conformance")
cam = IDSUEye()
check("CameraBase runtime protocol", isinstance(cam, CameraBase))
cam.connect()
check("connected", cam.is_connected)
check("device_info", cam.device_info["model"] == "DCC1545M-GL-MOCK" and cam.device_info["serial"] == "4103000000")
check("sensor_size full", cam.sensor_size() == (1280, 1024))
check("bit_depth 8", cam.bit_depth == 8)
check("ring allocated (12)", len(mock.S.buffers) == 12 and len(mock.S.sequence) == 12)
check("mono8 set", mock.S.color_mode == mock.IS_CM_MONO8)

print("3. ranges / capability gating")
el, eh = cam.exposure_range()
check("exposure range sane (us)", 0 < el < eh, f"{el}..{eh}")
fl, fh = cam.frame_rate_range()
check("fps range sane", 0 < fl < fh, f"{fl}..{fh}")
check("gain off", cam.gain_range() == (0.0, 0.0))
check("binning off", cam.binning_range() == (1, 1))
check("pixel clock list", cam.get_pixel_clock_list() == [5, 10, 20, 30, 43])

print("4. webapp defaults order (fps then... exposure first like _apply_defaults)")
cam.set_exposure(5000.0)   # webapp sets exposure before fps
cam.set_frame_rate(30.0)
check("fps commanded", abs(cam.get_frame_rate() - 30.0) < 1e-6)
check("exposure readable", cam.get_exposure() > 0)

print("5. capture: grab under engine semantics")
cam.set_pixel_clock(43)          # raise the achievable range so 60 fps is in-range
cam.set_frame_rate(60.0)
target = cam.get_frame_rate()
check("60 fps in range after clock raise", abs(target - 60.0) < 1e-6, f"target={target}")
cam.start(max_throughput=True)
f1 = cam.grab(timeout_ms=3000)
check("frame shape full sensor", f1.shape == (1024, 1280), str(f1.shape))
check("frame dtype uint8", f1.dtype == np.uint8)
check("pitch padding stripped", not (f1 == 255).all(axis=0).any(), "found a padding column of 255s")
n0 = time.perf_counter()
frames = [cam.grab(timeout_ms=3000) for _ in range(12)]
dt = time.perf_counter() - n0
rate = 12 / dt
check("paced near 60 fps", 30 < rate < 90, f"measured {rate:.1f} fps")
check("dedup: consecutive frames differ", any(not np.array_equal(a, b) for a, b in zip(frames, frames[1:])))
check("resulting_frame_rate live", abs(cam.resulting_frame_rate() - target) < 1e-6)

print("6. fps survives stop/start (re-apply on re-arm)")
cam.stop()
cam.start()
check("fps re-applied after re-arm", abs(mock.S.fps - target) < 1e-6, f"mock fps={mock.S.fps} (uEye resets to 25 on re-arm)")

print("7. ROI: engine-stopped change, misaligned request")
cam.stop()
cam.set_roi(101, 51, 321, 241)   # misaligned + odd sizes -> must clamp/align
x, y, w, h = cam.get_roi()
check("ROI aligned by clamp_roi", x % 4 == 0 and y % 2 == 0 and w % 4 == 0 and h % 2 == 0, f"{(x, y, w, h)}")
check("sensor_size tracks AOI", cam.sensor_size() == (w, h))
check("ring reallocated at AOI size", all(dims == (w, h) for dims in mock.S.buffers.values()))
cam.start()
f2 = cam.grab(timeout_ms=3000)
check("post-ROI frame shape", f2.shape == (h, w), str(f2.shape))
fl2, fh2 = cam.frame_rate_range()
check("fps range rose with smaller AOI", fh2 > fh, f"{fh2} vs {fh}")

print("8. one-shot grab while stopped (preview.py do_roi_select path)")
cam.stop()
f3 = cam.grab(timeout_ms=3000)
check("one-shot grab works while stopped", f3.shape == (h, w))
check("one-shot left capture stopped", not cam.is_grabbing)

print("9. reset_roi + hw metadata gating")
cam.reset_roi()
check("reset_roi full sensor", cam.get_roi() == (0, 0, 1280, 1024))
cam.start()
cam.grab(timeout_ms=3000)
check("hw meta off by default", cam.last_frame_meta is None)
cam.capture_hw_info = True
cam.grab(timeout_ms=3000)
check("hw meta captured when enabled", cam.last_frame_meta is not None and len(cam.last_frame_meta) == 3)

print("10. frames() generator + stop from consumer side")
cam.capture_hw_info = False
gen = cam.frames()
got = [next(gen) for _ in range(3)]
check("frames() yields", len(got) == 3 and got[0].shape == (1024, 1280))
cam.stop()
check("frames() terminates after stop", len(list(gen)) == 0)

print("11. disconnect (no buffer leaks, camera closed)")
cam.disconnect()
check("disconnected", not cam.is_connected)
check("all buffers freed", len(mock.S.buffers) == 0)
check("camera closed", not mock.S.open)
cam.disconnect()  # idempotent
check("disconnect idempotent", True)

print("12a. review fixes: event lifecycle across stop/start")
cam12 = IDSUEye()
cam12.connect()
check("event enabled after connect", mock.S.event_enabled)
cam12.start()
cam12.grab(timeout_ms=3000)
cam12.stop()
check("event disabled by stop (stale-signal guard)", not mock.S.event_enabled)
cam12.start()
check("event re-enabled by start", mock.S.event_enabled)
cam12.grab(timeout_ms=3000)

print("12b. review fixes: raw fps cache survives full-sensor excursion")
cam12.stop()
cam12.set_pixel_clock(43)
cam12.set_roi(0, 0, 320, 240)
cam12.set_frame_rate(200.0)
check("200 fps applied at small ROI", abs(mock.S.fps - 200.0) < 1e-6, f"{mock.S.fps}")
cam12.reset_roi()                       # range shrinks; applied value clamps
cam12.start()
applied_full = mock.S.fps
check("clamped at full sensor", applied_full < 200.0, f"{applied_full}")
cam12.stop()
cam12.set_roi(0, 0, 320, 240)           # range re-widens
cam12.start()
check("commanded 200 restored at small ROI", abs(mock.S.fps - 200.0) < 1e-6,
      f"{mock.S.fps} (raw command must survive the full-sensor clamp)")

print("12c. review fixes: edge-of-sensor ROI keeps SDK minimum")
cam12.stop()
cam12.set_roi(1272, 1020, 100, 100)     # near bottom-right corner
x, y, w, h = cam12.get_roi()
check("min size preserved at edge", w >= 32 and h >= 4, f"{(x, y, w, h)}")
check("still inside sensor", x + w <= 1280 and y + h <= 1024, f"{(x, y, w, h)}")

print("12d. review fixes: failed AOI set restores a working full-sensor camera")
mock.S.fail_aoi_set = True
try:
    cam12.set_roi(0, 0, 320, 240)
    check("set_roi raises on SDK failure", False)
except RuntimeError:
    check("set_roi raises on SDK failure", True)
check("full-sensor restored", cam12.sensor_size() == (1280, 1024))
check("ring restored (12 buffers)", len(mock.S.buffers) == 12)
cam12.start()
f12 = cam12.grab(timeout_ms=3000)
check("camera still grabs after failed ROI", f12.shape == (1024, 1280))

print("12e. review fixes: stop() aborts an in-flight grab from another thread")
import threading
mock.S.stall = True                     # no frames arrive
result = {}


def _blocked_grab():
    try:
        cam12.grab(timeout_ms=5000)
        result["outcome"] = "returned"
    except RuntimeError as e:
        result["outcome"] = "runtime-error"
    except Exception as e:
        result["outcome"] = f"other: {type(e).__name__}"


t = threading.Thread(target=_blocked_grab)
t.start()
time.sleep(0.3)                          # let it settle into the wait loop
t0 = time.perf_counter()
cam12.stop()
t.join(timeout=2.0)
abort_dt = time.perf_counter() - t0
check("grab aborted promptly (not 5 s)", not t.is_alive() and abort_dt < 1.0, f"{abort_dt:.2f}s")
check("abort surfaced as RuntimeError", result.get("outcome") == "runtime-error", str(result))
mock.S.stall = False

print("12f. review fixes: frames() honors its timeout on a stalled camera")
cam12.start()
mock.S.stall = True
gen12 = cam12.frames(timeout_ms=500)
try:
    next(gen12)
    check("frames() raises on stall", False)
except RuntimeError:
    check("frames() raises on stall", True)
mock.S.stall = False
cam12.stop()

print("12g. review fixes: connect() re-entry does not leak")
buf_before = len(mock.S.buffers)
cam12.connect()                          # second connect on a live instance
check("reconnect keeps exactly one ring", len(mock.S.buffers) == 12, f"{len(mock.S.buffers)}")
check("reconnect leaves camera usable", cam12.grab(timeout_ms=3000) is not None)
cam12.disconnect()

print("13. error path: grab timeout while stopped-and-broken")
cam2 = IDSUEye()
cam2.connect()
mock.S.capturing = False


def _no_video(hCam, mode):
    return -1


orig = mock.is_CaptureVideo
mock.is_CaptureVideo = _no_video
try:
    cam2.grab(timeout_ms=300)
    check("grab raises when capture cannot start", False)
except RuntimeError:
    check("grab raises when capture cannot start", True)
mock.is_CaptureVideo = orig
cam2.disconnect()

print()
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
