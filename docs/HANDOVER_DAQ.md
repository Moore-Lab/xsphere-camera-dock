# Camera dock → DAQ handover

*2026-08-02, from the camera-dock session. Read this before changing anything that
touches `camera_dock/`. Deeper detail lives in [DESIGN.md](DESIGN.md) (architecture +
driver contract) and [MEASURE_INTEGRATION.md](MEASURE_INTEGRATION.md) (the measurement
layer shared with the analysis tab).*

## State: the dock was rebuilt

The dock is no longer a thin wrapper over per-camera submodules. As of commit `4409dc2`:

| path | what it is |
|---|---|
| `camera_dock/base.py` | `CameraDriver` — the standardized command set every camera speaks |
| `camera_dock/drivers/` | drivers live **in-repo** now: `zelux.py` (Thorlabs CS165MU), `sim.py` (hardware-free fixture). `make_camera(token)` is the registry |
| `camera_dock/engine.py` | acquisition thread, latest-frame slot, per-frame sink |
| `camera_dock/recorder.py`, `metadata.py` | RAM→spill→encode recording + timing sidecars |
| `camera_dock/calibration.py` | per-camera µm/pixel scales |
| `camera_dock/webapp.py` | the eval server / mounted camera app |
| `camera_dock/static/` | UI, incl. `measure.js` shared with analysis |

The old `basler-acA1440/`, `zelux-cs165mu/`, `ids-ueye/`, `hayear/` submodule drivers are
**reference only** — nothing imports them. Only `zelux` and `sim` are registered; every
other token yields an `UnavailableCamera` stub.

**Key idea, if you read nothing else:** controls are a generic funnel. A driver *registers*
the controls its hardware has (`exposure`, `gain`, `fps`, `black_level`, …); the base class
owns normalisation (0–1 across the native range), clamping against hardware ranges read at
write time, read-back, and graceful "unsupported". Nothing raises — `get`/`set` return a
`ControlValue`. So the panel/UI never needs per-camera special cases: render whatever
`/capabilities` lists.

## The contract you consume (unchanged, and you're already using it correctly)

`xsphere_daq/panel.py` currently does exactly the right thing. For the record, these are
the guarantees the dock commits to keeping:

- `webapp.create_app(sessions, manage_lifecycle=False)` → a FastAPI app with **no lifespan**,
  safe to mount.
- `webapp.start_all(sessions)` / `stop_all(sessions)` — **never raise**; per-camera failure
  is isolated into `session.ok` / `session.error`.
- `webapp.CameraSession(name, camera)` — construction never touches hardware.
- `webapp.make_camera(token)` — **never raises**. Unknown/paused/broken → stub whose
  `connect()` raises, so the camera shows as unavailable and the panel still boots.
  (`_make_camera` remains as an alias.)
- Pages derive their API base from `window.location`, so everything works mounted under any
  prefix. No `root_path` plumbing needed.

Verified this session against your panel: `python -m xsphere_daq.panel sim sim:2 basler
--port 8030` serves `/`, `/cameras/`, `/cameras/cam/sim`, `/cameras/cam/sim/calibration`,
`/cameras/static/measure.{html,js}` — all 200, two sim cameras live, `basler` cleanly
unavailable.

If you need to break any of the above, say so and I'll change the dock side rather than
have you work around it.

## What's new since you last looked

1. **Landing page connect/disconnect.** Each camera card on `/cameras/` has a
   connect/disconnect button + live status (model, size, fps, `RECORDING`/`TIMELAPSE`).
   A camera released to ThorCam can be re-acquired without opening its page.
2. **Measurement layer.** Toolbar with ROI + line/circle/rect annotations, dimensions in
   pixels and µm. The engine (`static/measure.js`) is shared **unmodified** with the offline
   page (`/cameras/static/measure.html`) and, shortly, the analysis tab you mount at
   `/analysis`. Both mounted subsystems will reference the same file — please don't let a
   copy of it appear anywhere. Governed by `MEASURE_INTEGRATION.md`, including the
   `NOTE[live->offline]:` comment convention.
3. **Calibration.** Per camera, several named entries (one per objective), one active,
   stored as µm per **sensor** pixel (survives binning/ROI). Files:
   `calibrations/<camera>.json` (gitignored — lab-local data). API under
   `/cam/{name}/calibration`.
4. **Recording sidecars.** Every recording writes `<stem>.json` (measured sw/hw fps, genuine
   dropped frames from hardware frame-counter gaps, `um_per_px`, `calibration`, `roi`,
   `binning`), `<stem>_timestamps.npy`, `<stem>_hwclock.npz`. **The video container's fps is
   a playback speed only — real timing lives in the sidecars.** This is the natural
   handoff into analysis.
5. **Static files served `no-cache`** (they're read from disk per request and edited
   constantly; browsers were silently serving stale pages).

## Constraints worth knowing

- **One Zelux per process.** The Thorlabs TSI SDK is a hard per-process singleton. A panel
  process can drive at most one `zelux` token. Two Zelux cameras ⇒ two processes on two
  ports (or a shared-SDK restructure — `reference/dualcam_fast.py` shows the pattern).
  `sim` has no such limit, hence `sim sim:2` for multi-camera panel work.
- **Data directories are anchored to the dock repo root**, never the CWD
  (`camera_dock/paths.py`): `recordings/`, `captures/`, `presets/`, `calibrations/`. This
  is deliberate — the panel launches with a different CWD and used to split the data.
- **ROI changes stop and restart acquisition**, and are refused (409) while recording or
  timelapsing. They also re-scale fps/exposure ranges, so any UI must re-read capabilities
  after one.
- **Paused cameras.** basler / ids / hayear tokens are accepted and shown as unavailable.
  Porting one is a contained job: write `camera_dock/drivers/<name>.py` against
  `CameraDriver` and add a registry row. DESIGN.md has the cross-camera feature matrix.
- **Frames are `Frame` objects** (`data` + `meta` with host time, hardware frame counter and
  device timestamp), not bare arrays. Anything consuming `engine.latest()` or a sink wants
  `frame.data`.

## Running it

```bash
python -m xsphere_daq.panel zelux --port 8000
```

Standalone dock (no DAQ), useful for camera work:

```bash
python -m camera_dock.webapp zelux --port 8000
```

Hardware-free, for UI/panel work:

```bash
python -m camera_dock.webapp sim --port 8010
```

The desktop shortcut "xSphere Camera Dock" runs `launch_dock.bat` (standalone dock, zelux,
port 8000). If the panel becomes the normal entry point, point that bat at
`xsphere_daq.panel` instead — one line.

## Open items I'd pick up next

- Port the paused drivers onto `CameraDriver` when those cameras are needed again.
- Binning is implemented in the driver but not surfaced in the UI.
- Trigger modes (hardware/BULB) are deliberately deferred; the Zelux SDK supports them and
  `get_is_operation_mode_supported` is the probe.
- Annotations are display-only — never burned into snapshots or recordings.
