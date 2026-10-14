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

## 2026-10-14 — IDS uEye / Thorlabs DCx camera joins the dock (`ids`, `dcc`)

**Context.** An older Thorlabs DCx camera (C1285R12M, s/n 4103060588; IDS uEye hardware behind
ThorCam's `uc480_64.dll`) was connected. `ids`/`dcc` had been paused stubs since the rebuild.

- **New `camera_dock/drivers/ids.py`** on the `CameraDriver` ABC, ported from the legacy
  `ids-ueye` module; registered as `ids` and `dcc`, removed from `_PAUSED`. Controls: exposure,
  fps, gain (master gain), pixel_clock. ROI yes, binning no. Details in DESIGN §3b.
- **Legacy bugs fixed in the port:** per-frame metadata never worked (wrong buffer ID, and
  pyueye's struct is too big for uc480) and now delivers frame counter and device timestamp; fps
  getter returned the request instead of the hardware rate; `sensor_size` returned the AOI.
- **fps/exposure coupling** per §2.1: long exposure lowers fps, raising fps shortens exposure,
  and the operator's fps request survives re-arms. `webapp._apply_defaults` now asks for 30 fps
  outright (the funnel clamps; identical for Zelux/sim).
- **Bandwidth limit found:** 43 MHz pixel clock at full frame delivers no frames on this USB 2
  link; sets above the 24 MHz default warn.
- **Verified on hardware** (scratch test, 30 checks; then `camera-dock-ids` preview on :8001):
  14.0 fps full frame, 296×200 ROI → 30 fps, 150 ms exposure → 6.6 fps and back, live pixel
  clock 24→30→24 without a stall, stop releases a waiting grab in 1 ms, 3 s recording 43/43
  frames with consecutive hw counts and 13.954 fps hw-clock rate, unsupported control → 422.

---

## 2026-09-15 — Camera orientation: rotate 90° at a time, mirror, applied to saved data

**Context.** Cameras get mounted sideways or look through a mirror; the owner wants the picture
upright, and wants recordings, stills and snapshots to come out the way the stream looks. The
TSI SDK offers no readout flip or rotation (none of `TLCamera`'s 106 attributes), so it is
software.

- **One choke point.** `AcquisitionEngine._loop` orients each frame right after `grab()`, so
  the latest slot and the recorder sink both receive oriented frames. Measured 0.6–0.9 ms per
  full 1440×1080 frame (~2–3 % of a core at 35 fps); upright returns the same array at zero
  cost. A generation counter drops a frame transformed under the previous orientation, and
  `set_orientation` clears the latest slot.
- **New `camera_dock/orientation.py`:** `(rotate cw, mirror)` with mirror applied first — 8
  states, one representation each. Buttons compose in display space (`cw`, `ccw`, `flip_h`,
  `flip_v`, `reset`). Saved per camera in `orientations/<camera>.json` (gitignored) because it
  describes the MOUNTING and must survive a restart.
- **The trap.** `fresh_frame` drops any frame whose shape differs from `expected_shape`, which
  came from the native `image_size()`. A quarter turn swaps the shape, so without updating it
  every frame reads as stale geometry and the stream goes black. `expected_shape` and `/info`
  width/height are now oriented; `capabilities.image_size` stays native (the ROI/binning grid)
  with a new `display_size` beside it.
- **The sensor does not rotate.** ROI, binning, presets and annotations stay in sensor pixels.
  cam.js had two separate display→sensor bridges — the measure layer's
  `sensorToDisplay`/`displayToSensor` and the ROI drag's own inline corner maths — and both now
  go through `orientFwd`/`orientInv`. The staleness gate compares against `display_size`.
  measure.js gained an optional `transform.swapsAxes()` so a rectangle's "w × h" reads as seen
  (NOTE[live->offline]; the offline host is untouched).
- Refused (409) while recording, encoding or timelapsing, like a region change. Video sidecars
  record `orientation` next to `roi`/`binning`.

**Verified.** `tests/test_orientation.py` (76 tests): all 8 orientations against numpy, all 32
state × button compositions, the cam.js maps mirrored in Python and pinned against real frames
for every pixel, dragged boxes crop exactly the covered pixels, and end-to-end with the sim
camera (fresh frames, `/info`, capabilities, sidecar extra, 409 while recording, persistence
across a new session, the HTTP route). Parent panel tests still pass. On the running panel after
a real ⟳ click: stream, `/info`, a server snapshot, a saved snapshot TIFF, a timelapse still and
a recording's video frames were all 1080×1440, with the sidecar carrying `orientation` beside
the unrotated sensor ROI. A drag over the displayed top-left posted `x=0 y=540 w=360 h=540` —
the sensor's bottom-left, as a clockwise turn predicts.

**Adversarial review** (10 agents; 4 of 8 findings confirmed, all fixed):
- *Stale frame into a new recording.* The generation check guarded the latest slot but not
  the sink, so a record_start landing right after a re-orientation could hand the recorder one
  old-shape frame; the recorder latches that shape and silently drops the rest of the take.
  The sink now re-checks the generation. A deterministic test reproduces the window and fails
  against the pre-fix loop (first delivered shape `(3, 5)`, then `(5, 3)`).
- *Re-orienting while disconnected* returned no capabilities, so the page kept the old
  `display_size`; after another client reconnected, ROI drags were refused forever and
  measurements were scaled 25–33 % wrong. The page now drops its size on that path, and
  `pollInfo` refetches capabilities whenever `/info`'s delivered width/height differ from what it
  holds (which also reconciles a binning change made in another tab).
- *Scale readers used the sensor's bx after a quarter turn.* droplab's
  `sidecar_pixels_per_um` and measure.html's sidecar loader now take the binning of the
  recording's own horizontal axis (`sidecar_image_binning`); droplab's anisotropy caption names
  image axes. Only matters with bx ≠ by, which the Zelux driver allows.
- *The measurement table* kept the pre-turn "w × h" while the overlay swapped; it is rebuilt
  when the quarter-turn state changes.

---

## 2026-08-03 — Several Zelux in one process: the SDK becomes process-owned

**Context.** The rig has **three** Zelux CS165MU (serials 32943, 29559, 35969) and the
owner wants them all in one panel. "One Zelux per process" was documented as a hard SDK
limit; it was not. `ZeluxCamera.connect()` built its own `TLCameraSDK()` — a per-process
singleton — so a second camera either failed or tripped a recovery path that force-cleared
`_is_sdk_open` out from under the first. Proved on the rig before writing any code: one
shared SDK opens and streams all three.

- **New `camera_dock/drivers/tsi.py`** owns the SDK for the process. Three facts drive it,
  each verified: (1) `open_camera` returns `TLCamera._create(self._sdk, handle)` — the raw
  cdll, *not* the wrapper — so an open camera holds no reference back, and since
  `TLCameraSDK.__del__` disposes, letting the SDK become garbage closes it under live
  cameras. The module global is the only strong reference. (2) `dispose()` closes first and
  clears `_is_sdk_open` only after the error check, so one bad close wedges the flag
  permanently — so we never dispose on refcount zero, only at teardown, and *neuter* rather
  than close if a handle is still open. (3) `discover_available_cameras()` must run before
  any `open_camera()` or it fails with 1004 "Serial number not found"; it is safe to call
  while other cameras stream (measured).
- **Two-tier locking**, and the rule is what each lock is held *across*: the driver's RLock
  wraps a sequence on one camera (a whole `set_roi`, the 3.1 MB frame copy);
  `tsi.TSI_LOCK` wraps a single native call, process-wide. Every acquisition is bounded, so
  a wedged DLL costs one camera and some 503s rather than freezing every handler.
- **`TsiBusy` in the poll loop is a retry, not an error.** Caught by testing: an ROI change
  on one camera (~1 s of disarm/set/arm) made the *other two* accrue acquisition errors,
  which the panel's new liveness indicator would have painted as "no frames". Now zero
  errors across three ROI changes.
- **`is_grabbing` is mirrored, not probed** — it used to be a real `tl_camera_get_is_armed`
  round-trip on every idle poll iteration, ~1 kHz per camera of DLL traffic for information
  the driver already had. `device_info` is likewise cached at connect (every browser polls
  `/cameras` every 2 s).
- **Session-level handle leak fixed.** `start()` wrapped connect + defaults + preset +
  `engine.start()` in one `except` that set `ok=False` without disconnecting, and
  `_stop_locked` then early-returned on `not self.ok` — so a failure *after* a successful
  connect left the camera open forever. Under a shared SDK that makes the serial
  permanently unloadable. `start()` now disconnects on failure and `_stop_locked` releases
  whenever the driver is still connected.
- **`engine.stop()`'s return value is honoured.** It returns False when the producer is
  wedged inside the SDK; `_stop_locked` discarded it and disconnected anyway — a segfault
  that, with a shared SDK, would take the other cameras with it. Now it leaves the handle
  open and says so.

**Measured on the rig, three cameras through the real stack:** 30.0 / 18.0 / 18.0 fps, each
identical to its solo rate, so the global lock costs nothing; a control write during
streaming took 17 ms; three ROI changes on one camera produced zero errors on the others;
`/cameras` reports 3/3, all three panes stream 1440×1080, and teardown releases the SDK
cleanly. (The 18 fps pair is state persisted on those cameras, not a dock behaviour.)

**Still to come:** runtime discovery/add/remove (no more fixed token list), serial-keyed
persistence with editable labels, and the UI for both.

## 2026-08-02 — Three data-loss paths closed (recording finalize, teardown, calibration)

**Context.** A multi-agent review of the DAQ panel + dock + droplab (run from the parent
repo) surfaced three defects that cost experimental data rather than convenience. All
three were verified in the source before being fixed, and live-tested after.

- **Limited recordings finalized server-side.** `max_seconds`/`max_frames` made the
  recorder stop *accepting* frames and raise `limit_reached`; the only caller of stop was
  the browser poll (`static/cam.js:463`). The DAQ panel mounts camera panes lazily, so a
  limited recording only finalized while its tab happened to be open — an F5, a tab
  switch or a closed pop-out stranded the session in `recording` forever, holding RAM and
  an un-encoded `_xsphere_spill_*.raw` in `%TEMP%`. New `_start_limit_watchdog()` spawns a
  daemon thread per *limited* recording (unlimited ones start nothing) that polls at 5 Hz
  and finalizes. The browser call stays as harmless redundancy — it 409s once encoding.
- **Two shutdown modes** (revised later the same session — see below). Teardown first
  became encode-on-shutdown, then was reverted to an instant kill at the owner's request,
  with an explicit graceful path added instead.
- **Shared finalize path.** All three endings (operator stop, limit, teardown) funnel
  through `_begin_encode_locked()`. `_recording_extra()` snapshots calibration/ROI/binning
  on the *calling* thread, because teardown disconnects the camera immediately after —
  reading them on the encode thread would have silently emptied the sidecar exactly when a
  recording was being rescued. Its three probes are now independent, so a missing
  calibration no longer costs the ROI and binning too.
- **droplab uses the recording's own calibration.** `store.DEFAULT_PIXELS_PER_UM = 0.1878`
  is a count-droplets constant from a *different* rig, while every dock sidecar carries the
  real `um_per_px` and `video.py:sidecar_meta()` already loaded and discarded it. New
  `video.sidecar_pixels_per_um()`: `pixels_per_um = 1 / (um_per_px * binning[0])` —
  `um_per_px` is per **sensor** pixel, droplab measures **image** pixels, ROI is a crop and
  does not enter. `probe()` returns it, `video_info()` prefers it, and the lab page shows
  provenance (green = this recording's calibration, amber = borrowed default). On the sim
  10x calibration that is 0.7216 vs 0.1878 — a **3.84x** error on every length, and so on
  volume and mass. Existing `cs165mu_*` recordings have no calibration in their sidecars,
  so they correctly show amber.
- **Longitudinal honours the density field.** `lab.js` sent `density_kg_m3` on the
  distributions job but not the longitudinal one, which the server also reads
  (`server.py:473`) — so changing the density silently moved one stage and not the other.

**Verified live** (sim camera, panel on :8032): a 3 s limited recording with *no* browser
attached went recording → `limit_reached` → encoding → idle with 52/52 frames encoded and
sidecars written; `stop_all()` during an unlimited recording produced a 22 MB video with
45/45 frames, all three sidecars, and no leftover spill file; the lab page prefills 0.7216
green for a calibrated recording and 0.1878 amber for an uncalibrated one. Conversion
helper unit-checked against 14 edge cases (missing/zero/negative/junk `um_per_px`,
missing/null/empty/zero/junk binning) — all degrade to the honest default.

**Adversarial review of the above** (4 lenses — locking/lifetime, teardown, regressions,
the physics of the conversion — each finding attacked by 2 independent skeptics). Six
findings survived; all six were confirmed by hand against the code and fixed:

- **`stop_all` was a regression** and the only finding that cost data in the shipped
  three-camera config. Teardown now hands recordings to an encode thread and `stop()`
  joins it, but `stop_all` was still one sequential pass — so camera 1's encode blocked
  while cameras 2..N kept acquiring, recording and spilling. Split `stop()` into
  `begin_stop()` + `join_encode()` and made `stop_all` **two passes**: every camera stops
  acquiring, *then* the files are waited on. Measured with two sim cameras: single-pass
  gave a **1.878 s** capture-window skew (exactly camera A's 1.885 s encode) and cost the
  sum of the encodes; two-pass gives **0.003 s** skew and the longest single encode.
  Before FIX 2 this loop was harmless — `_stop_locked` returned in milliseconds.
- **`sidecar_pixels_per_um` is the horizontal scale only.** `bx`/`by` are independently
  settable and every consumer applies the scale isotropically. Rather than returning None
  on `bx != by` (which would fall back to *another rig's* constant — wrong in both axes
  instead of one), `probe()` now reports `anisotropic_binning` and the provenance label
  goes amber and names the problem. Latent today: nothing in any UI posts `/binning`.
- **`measure.html` had the same class of bug, pre-existing.** It fed the per-**sensor**-px
  `um_per_px` straight into the offline measure layer, which works in **image** px
  (`measure.js` `defaultTransform`) — the live page is unaffected because `cam.js`'s
  transform already works in sensor px. At 2×2 binning, measure and droplab would quote
  the same calibration name and disagree by exactly `binning[0]`. Now multiplied up, with
  the working shown in the status line. `measure.js` itself untouched (single-copy rule).
- **`analyze()` preferred the foreign constant** over the sidecar value already in scope
  17 lines above — this is the layer that *writes results*, so its default is what lands
  in the log. Now falls back to the recording's own calibration, plus a positivity check.
- **The provenance caption was write-once at boot**, so it kept claiming "recording
  calibration" after anyone typed in the field or loaded a saved log. Now tracks the
  field: green only while it still equals the sidecar prefill, amber for a hand-entered
  value, red for a non-number. Verified in the browser across all four states.
- **`stop()` could skip the join if `_stop_locked` raised** after handing off. Closed
  structurally by the `try/finally` in the new `stop()` plus the two-pass `stop_all`.

**Resolved — Ctrl+C is an instant kill, graceful shutdown is a button.** The owner's call:
the rig must always come down promptly, and finishing a recording should be a deliberate
act, not something shutdown does for you.

- `_stop_locked` discards a running recording again (~0.1 s for two cameras, measured),
  but no longer *silently*: it prints `ABORTED recording on shutdown — N captured frame(s)
  discarded, no file written (<name>.avi)` per camera, and names any orphaned
  `_xsphere_spill_*.raw` left in `%TEMP%` (`stop_and_encode`, which would have removed it,
  never runs). Silence here is what made this a data-loss bug in the first place.
- New `webapp.finalize_recordings(sessions)`: stop + encode every running recording, then
  wait — two passes, so every camera stops capturing before any encode is waited on.
  This is the only path that writes files at shutdown.
- `stop_all` stays two-pass for the narrower case of an encode already in flight when
  shutdown arrives. Such an encode still completes: the thread is non-daemon on purpose,
  because killing it mid-write leaves a corrupt video.

Measured both modes with two sim cameras recording: instant kill **0.10 s**, no files
written; graceful **1.97 s** including both encodes (which overlap), two good files,
capture windows co-terminal to **6 ms**.

## 2026-08-02 — Rebuild: standardized driver interface + drag-ROI eval server

**Context.** Zelux-first redesign per `docs/DESIGN.md` (spec written and adversarially
reviewed this session — four-lens review: SDK reality, web UX, concurrency, DAQ
contract). Goal: one common command set for all cameras, per-camera drivers translating
native controls, graceful unsupported handling, and a standalone eval server whose
features later fold into the DAQ.

- **`base.py` rewritten**: `CameraDriver` ABC + generic scalar-control funnel.
  Drivers *register* `ControlSpec`s (units, fresh-range getter, int/float kind,
  linear/log scale); the base owns normalization (0..1 across the native range),
  clamping against write-time hardware ranges, casting, read-back, and the
  never-raise `ControlValue` envelope. Unsupported control ⇒ absent from
  capabilities, `supported=False` on set (HTTP 422) — one convention, no exceptions.
  Legacy shims (`get_exposure`/`set_frame_rate`/…) keep `presets.py` and
  `imaging.auto_expose` unchanged.
- **Drivers moved into the dock** (`camera_dock/drivers/`): `zelux.py` (rebuilt on
  TSI SDK 0.0.8: internal RLock around every SDK call with non-blocking polls;
  atomic disarm→ROI→read-back→re-arm→re-trigger; fps-then-exposure re-apply after
  every re-arm; fps/exposure duty coupling with warnings + side_effects; hw
  frame-count/timestamp captured per frame) and `sim.py` (hardware-free fixture).
  `make_camera(token)` is contractually never-raise (UnavailableCamera stub) —
  the panel boots with the bat's `basler zelux hayear` and paused tokens degrade.
- **Engine**: monotonic frame index across restarts, latest-slot cleared on stop,
  sink under a dedicated lock (`set_sink(None)` is a barrier), Frame objects with
  per-frame hw metadata.
- **Recorder**: sidecars per recording (`_timestamps.npy`, `_hwclock.npz` with NaN
  for missing hw timestamps, `.json` with sw/hw measured fps + genuine drops from
  hw-counter gaps), duration/frame limits (flag-only; UI poll finalizes),
  background encode via a recording state machine (idle→recording→encoding).
- **Web app rebuilt**: generic control panel rendered from `/capabilities`
  (normalized 0..1000 sliders, debounced, echo-resync, side-effect refresh);
  **drag-ROI zoom** on the live stream (exact CSS-px transform, staleness gates,
  applied-ROI adoption, stream reconnect per change), right-click = uniform undo
  via a zoom stack, editable coordinate fields; recording UI with limits;
  timelapse carried forward (incl. state restore). Pages in `camera_dock/static/`
  derive their base from `window.location` (works standalone and mounted).
  Data dirs anchored to the dock repo root (`paths.py`), never CWD.
  `preview.py` removed (superseded; git history has it).
- **Hardware-verified on CS165MU s/n 32943** [verified-run]: full frame 30 fps →
  drag-zoom to 292×220 → fps range re-published 1–34.8 ⇒ 3.9–**149.1**, measured
  149.1 fps; recorded 600 frames @ 149.3 fps, 0 drops, hw-clock sidecar valid
  (6.708 ms mean Δ); right-click restored full frame with fps re-clamped to 34.8;
  coupling clamped 100 ms exposure → 6.57 ms on fps=149 with warning; black_level
  auto-registered (0–511); `pixel_clock` → clean 422; duration-limit auto-stop OK.
  Panel contract verified against unmodified `xsphere_daq/panel.py`.

**Next.** Port basler/ids/hayear drivers onto `CameraDriver` when they're needed
again (feature matrix + porting notes in DESIGN.md); binning UI; integrate the
eval server's features into the DAQ panel.

## 2026-06-12 — Web app: per-camera connect / disconnect at runtime

**Context.** User needs to release a camera from the dock at runtime (to hand it to the
GUI / ThorCam / Pylon Viewer) and re-acquire it, without restarting the server.

- `CameraSession.start()` is now idempotent (no-op if connected) and `stop()` fully
  releases (stops engine + `camera.disconnect()` + clears recorder + `ok=False`), so both
  can be driven at runtime.
- Endpoints `POST /cam/{name}/connect` and `POST /cam/{name}/disconnect` (disconnect is
  409 while recording). While released, the camera's other endpoints 503; `/info` and
  `/connect`/`/disconnect` still work.
- Control page: a **connect/disconnect button** in the header (label tracks live state
  via `/info`); on toggle it POSTs and reloads to re-establish the stream/controls. The
  page degrades gracefully when the camera is released (controls skip binding).

**Validated on the Zelux:** disconnect → `ok=False`, `/controls` returns 503 (camera
released); connect → `ok=True`, stream resumes. Confirms a camera can be freed for
another client and re-acquired live.

**Incident:** the Hayear HY-1136 (cheap UVC, RDP-redirected) dropped off the USB bus
during rapid connect/disconnect/force-kill test cycles — it stopped enumerating in
DirectShow/PnP and needs a physical replug. Lesson: UVC cameras don't tolerate fast
open/close churn well; the SDK cameras (Basler/Zelux) do.

## 2026-06-12 — Hayear HY-1136 (UVC) + has_exposure gating

**Context.** The connected Hayear (HY-1136) turned out to be a **UVC webcam-class colour
camera**, not a ToupCam-SDK device (the hycam SDK enumerates 0; it streams via
DirectShow). Added it as a UVC camera and taught the dock to handle a camera with no
exposure control.

- `hayear/` driver rewritten from the (wrong) hycam-SDK scaffold to **UVC/OpenCV**:
  `HayearCamera` over `cv2.VideoCapture` (DSHOW index 0), modes 640×480/720p/1080p,
  BGR→mono uint8. Exposure/gain/frame-rate/ROI are unsupported (auto-only over UVC) and
  report empty ranges. (Still an **untracked** local module pending its own Moore-Lab
  repo; `_make_camera` already maps `hayear`.)
- **`has_exposure` capability gating** added to `preview.py` and `webapp.py` (parallel to
  has_gain/has_fps/has_roi): controls report it; the test GUI skips the exposure
  trackbar / numeric entry / auto-exposure when unsupported (was crashing on a 0-range
  exposure slider); the web control page hides the exposure row + auto-exposure button.
  `controls()` now includes `has_exposure`.

**Validated on the connected HY-1136:** driver conforms to `CameraBase`; streams mono
1080p through the engine at 30 fps (0 dropped recording); in the dock via
`python -m xsphere_daq.panel hayear` — info/stream/snapshot/record all work and
`controls` reports has_exposure/gain/fps/roi all false (controls correctly gated off).

## 2026-06-12 — webapp: mountable as a sub-application

**Context.** Make the camera app mountable under a path prefix so the top-level
`xsphere_daq` panel can host it at `/cameras` (see the top repo's log).

- `create_app(sessions, manage_lifecycle=True)` — `False` skips the lifespan so a parent
  app manages startup/shutdown. Added `start_all` / `stop_all` helpers for the parent.
- HTML links are now **root_path-aware**: the overview cards, the control page `BASE`,
  and the "all cameras" back-link prefix with `request.scope["root_path"]`, so the app
  works both standalone (prefix "") and mounted (e.g. "/cameras").
- **Bug fixed:** moved the FastAPI imports to module top. With `from __future__ import
  annotations`, the `request: Request` annotation is a string FastAPI resolves against
  module globals — where `Request` wasn't imported (it was local to `create_app`), so
  FastAPI treated `request` as a required query param and the overview/control pages
  500'd. Module-level import fixes resolution.

**Validated** via the top-level panel (mounted under /cameras): overview cards, control
page BASE, info/controls/stream all correctly prefixed and working.

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
