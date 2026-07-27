# ids-ueye

Driver + test GUI for IDS uEye cameras — **including the older Thorlabs
DCC/DCx-series cameras (DCC1545M, DCC1240, DCC3240, ...), which are rebranded
IDS uEye hardware** — for the [xsphere-camera-dock](../README.md).

The capture core is a port of the hardware-tested `IDSCamera` adapter from the
lab's PyQt dual-camera GUI (`reference/dualcam_fast.py`, validated at 200 fps):
freerun capture with a 12-buffer ring, event-driven frame waits, and the full
stop/free/AOI/realloc sequence for ROI changes — re-wrapped behind the dock's
`CameraBase` surface so the shared acquisition engine, recorder, preview GUI,
and web app drive it like any other camera.

## Install

1. Install ONE driver stack:
   - **IDS cameras**: the [IDS Software Suite](https://en.ids-imaging.com/downloads.html)
     (provides `ueye_api.dll`), or
   - **Thorlabs DCC/DCx cameras**: ThorCam with "DCx Camera Support"
     (provides `uc480_64.dll` — same SDK, renamed). The driver detects which
     one is present automatically; set `THORLABS_UC480_DIR` only if ThorCam is
     installed somewhere non-standard.
2. `pip install -r requirements.txt`

The Thorlabs uc480 DLL predates `is_WaitEvent`, so on that stack the driver
automatically switches to the classic event mechanism (`is_InitEvent` + a
Windows event). Which DLL a session is using is reported in `device_info`
(`sdk_dll`, and `vendor` shows "Thorlabs (uc480)").

## Use

```bash
python smoke_test.py          # enumerate, connect, grab a burst, report rates
python -m ids_ueye.gui        # shared OpenCV test GUI (from the dock checkout)
python -m camera_dock.webapp ids   # web app (from the dock root)
python -m camera_dock.webapp dcc   # same driver, named for the Thorlabs DCC
python tests/test_driver_mock.py            # no-hardware regression (IDS mode)
UC480_SIM=1 python tests/test_driver_mock.py   # same, Thorlabs DCC event path
python tests/test_uc480_real.py             # binds the real ThorCam uc480 DLL
```

## Notes

- Frames are Mono8 `uint8`.
- Gain and binning are not exposed (the dock feature-detects them off via the
  ranges); use ROI + pixel clock to trade field of view for frame rate.
- `get_pixel_clock_list()` / `set_pixel_clock()` are available as uEye-specific
  extras (a higher pixel clock raises the achievable frame-rate range).
- The commanded frame rate is re-applied automatically on every capture start —
  the uEye freerun rate does not survive a re-arm (ROI changes re-arm).
- Per-frame hardware metadata (frame number, device timestamp) is available via
  `capture_hw_info` / `last_frame_meta` for drop accounting during recording;
  it is off by default so the free-run path pays no per-frame cost.
