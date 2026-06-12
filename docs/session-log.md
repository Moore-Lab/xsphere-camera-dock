# xsphere-camera-dock — Development Session Log

Running, chronological record of development for **this repo only**
(`xsphere-camera-dock`). Changes inside a camera submodule are logged in *that
submodule's* own `docs/session-log.md` — write to the log of the repo you're modifying.

Scope of this log: the shared camera interface (`camera_dock/`), the dock integration
layer (combined feeds, unified controls, snapshot/record across cameras), camera
submodule pointer bumps, and (eventually) the dock's web UI.

Newest entries first. Keep entries short and factual; convert relative dates to absolute.
See the [README](../README.md) for this repo's roadmap.

---

## 2026-06-12 — Shared acquisition engine + hybrid recorder (Stage 1)

**Context.** Began the reusable integration layer. Driving requirement: record video
at the camera's *true* max data rate (the old per-camera GUIs recorded at render speed
and dropped frames via latest-image-only grabbing). User chose: keep per-camera GUIs
(as thin shells), hybrid/auto recording. Logic lives here so the dock + DAQ reuse it.

**New shared modules (all on `CameraBase`, camera-agnostic):**

- `camera_dock/engine.py` — `AcquisitionEngine`: a producer thread pulls every frame
  at the full no-drop rate (`camera.start(max_throughput=True)`) and distributes to a
  latest-frame slot (preview samples it at its own rate) + an optional per-frame sink
  (recorder). Slow rendering can't throttle capture. Reports true acquisition fps.
- `camera_dock/recorder.py` — `HybridRecorder`: hot path just appends frames to RAM
  (no encode); past a RAM cap, frames spill to a raw file via a bounded-queue writer
  thread (overflow counted, not lost); on stop, encodes everything to lossless video
  (FFV1→MJPG). Encoding is the post-stop "few seconds delay", off the capture path.
  `to_8bit` callback handles 16-bit cameras.
- `camera_dock/preview.py` — `run(camera)`: the shared OpenCV test-GUI shell (threaded
  preview, dual fps overlay [acquisition vs preview], exposure/fps sliders, snapshot at
  full bit depth, hybrid record). This is what each camera's `gui.py` now launches.
- `camera_dock/base.py` — `CameraBase` expanded to the full shared surface the engine
  relies on (info, exposure/fps get/set/range, `start(max_throughput)`, stop, grab,
  frames). Also fixes the old em-dash window-title mojibake (plain hyphen now).

**Validated on hardware (headless — engine + recorder directly, GUI window needs a
desktop):**

- Basler @ full rate: **227.8 fps** acquisition during record, 456/456 frames, **0
  dropped**, AVI readback 456 frames (12.8 MB FFV1), encode 7.3 s.
- Basler spill path (tiny RAM cap, 30 fps): 60 captured = 4 RAM + 56 spilled, 0 drops,
  all encoded + readback-verified.
- Zelux @ full rate: **34.8 fps**, 70/70, 0 drops, uint16→8-bit encode correct.

This is the headline win: recording now runs at the camera's real ceiling (227 fps on
the Basler) instead of the old ~60 fps render-coupled rate, with no frame drops.

**Next (Stage 2):** discrete feature widgets on top of the engine — gain, ROI/binning,
numeric entry, histogram/saturation, auto-exposure, snapshot formats — added to the
shared engine + `CameraBase` so both cameras and the dock inherit them.

## 2026-06-11 — Session log started; baseline

**Context.** First logged session. Established per-repo session logs across the project.

**State of this repo:**

- `camera_dock/base.py` — `CameraBase`, a `runtime_checkable` structural `Protocol`
  (`connect` / `disconnect` / `set_exposure` / `set_frame_rate` / `grab` / `frames`).
  Dependency arrow points dock → cameras (cameras don't import the dock), avoiding a
  circular submodule dependency. **Defined; no integration layer yet.**
- Submodules: `basler-acA1440` (driver + test GUI implemented — see its log),
  `zelux-cs165mu` (scaffolding only).

**Next (dock):**

1. Once both camera drivers are validated in isolation, build the integration layer over
   `CameraBase`: combined live feeds, unified frame-rate / exposure controls, synchronized
   snapshot and recording.
2. Surface the dock as a web app feeding up into the top-level `xsphere-daq` panel.
