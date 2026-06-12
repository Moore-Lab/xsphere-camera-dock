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

## 2026-06-12 — Dock web app: settings presets

**Context.** Make camera setups reproducible across restarts (important for the
experiment) and bring the rig up pre-configured.

- `camera_dock/presets.py` — camera-agnostic capture/apply/save/load of exposure,
  gain, frame rate, ROI, binning to `presets/<camera>__<name>.json`. Built only on
  `CameraBase`.
- `webapp.py`: per-camera endpoints `GET /cam/{name}/presets`,
  `POST /cam/{name}/presets/save?preset=`, `POST /cam/{name}/presets/load?preset=`
  (ROI applied via the stop→set→restart region change; 409 while recording, 404 if
  missing). `CameraSession.start()` **auto-applies a `default` preset** if present, so
  the rig comes up configured. Control page gains a preset save/load row (with a
  datalist of existing presets).
- `.gitignore`: ignore `presets/` (local rig state).

**Validated on hardware (Basler, server + curl):** set exp 2500 / gain 4 / ROI
200,150,800,600 → save `expt1` → change everything away → load `expt1` → all three
restored exactly; missing preset → 404; saved a `default`, restarted the server, and
it **auto-loaded on startup** (exp/gain/ROI all applied).

**Next (dock):** ROI box-draw on the live stream, and promote the dock up into the
top-level `xsphere-daq` control panel.

## 2026-06-12 — Dock web app: multi-camera streaming

**Context.** One server driving several cameras at once.

- `camera_dock/webapp.py` refactored around a `CameraSession` (camera + engine +
  recording state + control ops). `create_app(sessions)` manages a dict of sessions;
  all routes namespaced `/cam/{name}/...`. `/` is an overview page (grid of every
  camera's live stream, each linking to its full control page at `/cam/{name}`).
  `/cameras` lists them. Startup is resilient — one camera that fails to connect is
  marked unavailable (503) without sinking the others.
- `python -m camera_dock.webapp basler zelux [--host 0.0.0.0 --port 8000]`.

**Two fixes found via concurrent testing:**

- JPEG encode moved off the event loop (`asyncio.to_thread`) so concurrent MJPEG
  streams interleave fairly (before: one stream starved the other).
- `CameraSession._apply_defaults()` sets a sane exposure (5 ms) + frame rate (30 fps)
  on start (mirrors the preview GUI). The Zelux otherwise streamed at ~2.5 fps because
  frame-rate control was enabled but unset (near its minimum).

**Validated on hardware (Basler + Zelux simultaneously):** both connect; `/cameras` and
the overview show both; controls are independent (Basler exp 1500 us vs Zelux 7969 us);
both streams run concurrently at **30 fps engine** (Basler 63 / Zelux 44 frames in 3 s).

**Next (dock):** persist settings / config, ROI box-draw on the stream, and promote the
dock up into the top-level `xsphere-daq` control panel.

## 2026-06-12 — Dock web app: live controls

**Context.** Drive the camera from the browser (next after streaming).

- `camera_dock/webapp.py`: added control endpoints over the shared engine —
  `GET /controls` (values + ranges + capability flags + rec state),
  `POST /controls/exposure|gain|fps|auto_exposure`, `POST /controls/roi` +
  `/controls/roi/reset` (stop engine → set region → restart; **409 while recording**),
  `POST /record/start` + `/record/stop` (→ encode stats via HybridRecorder),
  `POST /snapshot/save`. Blocking ops are sync `def` so FastAPI runs them off the event
  loop; ROI/record guarded by a lock. The HTML page now has a control panel (geometric
  exposure slider, gain/fps sliders, auto-exposure, snapshot, ROI x,y,w,h set/reset,
  record toggle with live stats) — all reusing the same `CameraBase` surface as the GUIs.

**Validated on hardware (Basler, server + curl):** exposure 3000 us, gain 6 dB,
auto-exposure → 57 ms (converged), ROI 720×540 then reset to 1456×1088, record →
35 frames 0 dropped encoded. All endpoints return expected JSON.

**Next:** multi-camera streaming (pick/stream several cameras at once).

## 2026-06-12 — Dock web app: stream the camera feed (first web UI)

**Context.** Start of the dock's web UI — "a webpage that streams the camera feed."

- `camera_dock/webapp.py` — a FastAPI server that runs any `CameraBase` camera through
  the shared `AcquisitionEngine` and streams it as **MJPEG** (a plain `<img>` tag).
  Camera-agnostic: `create_app(camera)` works for Basler/Zelux/Hayear unchanged — the
  same engine that backs the test GUIs now feeds the browser. Endpoints: `/` (HTML
  page), `/stream` (multipart MJPEG), `/snapshot` (one JPEG), `/info` (JSON). Frames
  carry the acq-fps/exposure overlay + the New Haven timestamp. Run:
  `python -m camera_dock.webapp basler --host 0.0.0.0 --port 8000`.
- `requirements.txt`: enabled `fastapi` + `uvicorn`.

**Validated on hardware (Basler, headless server + curl):** `/info` →
`acA1440-220um 1456x1088 @ 30.1 fps`; `/` serves the page; `/snapshot` → 33 KB JPEG
(magic FFD8); `/stream` → 62 MJPEG frames in 3 s. Streams at ~30 fps to the browser
while the engine runs the camera underneath (display rate decoupled from capture).

**Hayear scaffold (separate, uncommitted).** Added `hayear/` (hycam/ToupCam driver +
GUI shell) as an **unvalidated** local scaffold — imports cleanly, conforms to
`CameraBase`, fails `connect()` with a clear "install hycam.py" message. Left untracked
pending its own Moore-Lab repo; see memory `hayear-camera-support-pending`. TODO needs a
connected camera + the hycam wrapper to validate (driver acquisition/no-drop/color).

**Next:** grow the web UI — camera picker / multi-stream, and live controls
(exposure/gain/ROI/record) over the engine, reusing the same `CameraBase` surface.

## 2026-06-12 — New Haven timestamp burned into preview + movie

**Context.** Request: show the New Haven date/time in the window *and* have it recorded
in the movie. Key subtlety — the recorder stores **clean raw frames** (the preview's
status overlay is not in the video), so the timestamp has to be burned into the recorded
frames separately, not just shown live.

- `camera_dock/imaging.py`: `eastern_now()` (America/New_York, DST-aware via `zoneinfo`),
  `format_timestamp` (`YYYY-MM-DD HH:MM:SS.mmm EDT/EST`), `draw_timestamp` (burns it into
  a BGR frame, bottom-right, font scales with width).
- `camera_dock/recorder.py`: `HybridRecorder(clock=...)` anchors a wall time to the first
  frame and keeps a per-frame perf timestamp for **every** frame (even spilled ones).
  `stop_and_encode(..., stamp=draw_timestamp)` annotates each frame with **its own
  capture time** at encode — so the movie shows true per-frame New Haven time, not the
  encode time.
- `camera_dock/preview.py`: draws the live timestamp every frame, creates the recorder
  with `clock=eastern_now`, and passes `stamp=draw_timestamp` on stop.
- `requirements.txt`: added `opencv-python` (the dock modules use cv2) and `tzdata`
  (IANA tz db for `zoneinfo` on Windows).

**Validated:** tz resolves to EDT; a recorded clip has a non-blank bottom-right stamp
region that changes frame-to-frame as time advances (per-frame timestamps confirmed),
45/45 frames, 0 dropped.

## 2026-06-12 — Stage 2 Round B: ROI / binning

**Context.** The last Stage 2 feature — held back because it must stop acquisition to
change the sensor region and is camera-specific.

**`CameraBase`** gains optional ROI/binning: `roi_range` (dict: w/h min/max/inc + x/y
inc), `set_roi(x,y,w,h)` / `get_roi` / `reset_roi`, `binning_range` /
`set_binning(bx,by)` / `get_binning`. ROI is uniform across cameras as
`(offset_x, offset_y, width, height)`; drivers translate (Zelux uses corner coords).

**`preview.py`** wiring: `o` = draw a region with `cv2.selectROI` (selects on a full
frame for absolute coords), `O` = reset to full, `b` = cycle binning 1→2→4. Each runs
through a `_restart` helper that stops the engine, changes the region, and restarts —
**blocked while recording** (the video frame size is fixed at record start).

**Validated on hardware (both cameras):**

- Basler: full sensor is actually **1456×1088** (the "1440" is nominal); `set_roi`
  exact to pixel increments; `reset_roi` restores full; binning to 4×4.
- Zelux: ROI snaps to even boundaries (720×540 @ y270 → 720×544 @ y268); binning to
  16×16.
- Engine restart through region changes: full → 2×2 bin (728×544) → ROI (400×300),
  frames grab at the new size each time. Only the interactive `selectROI` box-draw
  needs the window (Remote Desktop).

**Stage 2 complete.** Gain, histogram/saturation, auto-exposure, snapshot formats,
numeric entry (Round A) + ROI/binning (Round B) — all on the shared engine + drivers
behind `CameraBase`, reused by the GUIs and ready for the dock/DAQ.

## 2026-06-12 — Stage 2 Round A: gain, histogram, auto-exp, snapshots, numeric entry

**Context.** Discrete feature widgets on top of the Stage 1 engine. Camera-agnostic
logic lives in the dock; camera-specific capability (gain) goes in the drivers behind a
uniform `CameraBase` surface. ROI/binning deferred to Round B (it restarts acquisition
and is camera-specific — wants its own validated pass).

**New shared module `camera_dock/imaging.py`** (operates on plain numpy frames; reused
by GUIs + dock + DAQ):

- `saturation_fraction` + `histogram_overlay` (log-scaled luminance histogram drawn into
  the preview).
- `auto_expose` — software one-shot: scales exposure so the 99th-percentile brightness
  hits a target fraction of the sensor max. Caller supplies a fresh-frame grabber.
- `save_snapshot` + `SNAPSHOT_FORMATS` = tiff / png / npy (tiff/png lossless full-depth,
  npy = exact raw array).

**`camera_dock/preview.py`** rewired with: gain slider (when supported), histogram toggle
(`h`), saturation % in the overlay, one-shot auto-exposure (`a`), snapshot-format cycle
(`f`), and in-window **numeric entry** (`e`/`t`/`g` → type digits → Enter) for precise
exposure/fps/gain — sliders stay in sync.

**`camera_dock/base.py`** — `CameraBase` gains optional `set_gain`/`get_gain`/`gain_range`
(feature-detect via range max > min; no exceptions for unsupported cameras).

**Validated on hardware (headless — logic, not the window):**

- Gain: Basler 0–36 dB, Zelux 0–480 index (0–48 dB) — set/get/range correct on both.
- Auto-exposure: Basler 100 µs → 114 ms, 99th-pct level converged to **0.90** (= target).
- Snapshots: tiff/png/npy all write; npy round-trips bit-identical.
- Histogram overlay + saturation %: correct.

**Next (Stage 2 Round B):** ROI / binning — set while acquisition is stopped, snap to
camera increments, restart engine; both drivers + `CameraBase` + preview.

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
