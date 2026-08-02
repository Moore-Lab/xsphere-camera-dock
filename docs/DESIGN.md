# Camera dock redesign — standardized driver interface + eval server

*v1.1, 2026-08-02. Supersedes the `CameraBase` Protocol in the pre-rebuild `camera_dock/base.py`.
Grounded in the 2026-08-01 six-subsystem study (Zelux driver, TSI SDK 0.0.8 surface, dualcam_fast
reference, webapp, parent DAQ, paused drivers), then amended by a four-lens adversarial design
review (SDK reality, web UX, concurrency, DAQ contract/scope). Zelux CS165MU is the only active
camera; the other modules (basler, ids-ueye, hayear) are paused but shaped the command set.*

## 1. Architecture

```
camera_dock/
  base.py        CameraDriver ABC + ControlSpec/ControlValue + Frame/FrameMeta + errors
  drivers/
    __init__.py  make_camera(token) registry ("zelux", "zelux:SERIAL", ...) — NEVER raises
    zelux.py     Thorlabs Zelux CS165MU driver (TSI SDK), rebuilt in-repo
  engine.py      AcquisitionEngine — producer thread, latest-Frame slot, per-frame sink
  recorder.py    HybridRecorder — RAM→spill→encode + metadata sidecars + frame/duration limits
  metadata.py    sidecar writing (adopted from reference/video_metadata.py, never raises)
  imaging.py     histogram / auto-expose / timestamp burn / snapshots (carried forward)
  presets.py     capture/apply/save/load (carried forward; DATA_ROOT-anchored)
  webapp.py      eval server: FastAPI app, MJPEG streams, generic control UI, drag-ROI
  static/        cam.html/js/css + index.html — pages derive their API base from
                 window.location, so they work standalone and mounted (no .replace hacks)
```

Decisions:

- **Drivers move into the dock package.** The submodule-per-camera indirection (sys.path hacks,
  cross-repo commits per interface change) cost more than it bought. A driver is a thin translation
  layer; it lives at `camera_dock/drivers/<name>.py`. Old submodules stay as reference.
- **The dock stays multi-camera.** One server, N `CameraSession`s at `/cam/{name}/…`.
- **`make_camera(token)` never raises.** Unknown token, missing SDK, constructor error → an
  `UnavailableCamera(token, reason)` stub whose `connect()` raises `RuntimeError(reason)`;
  `CameraSession.start()`'s try/except then yields `ok=False` with the reason. This matters because
  the parent panel constructs sessions *itself* before uvicorn starts (`panel.py:86`) with the bat
  file's token set `basler zelux hayear` — the app must boot with only zelux registered.
- **Parent-DAQ contract preserved**: package dir `camera_dock/` at repo root, module `webapp`,
  symbols `create_app(sessions, manage_lifecycle=False)` (no lifespan when False), `start_all`
  (never raises), `stop_all` (per-session try/except — one bad disconnect must not skip the rest),
  `CameraSession(name, camera)`, `make_camera` + `_make_camera` alias. `camera_dock/__init__.py` is
  in scope: reduced to light exports (no `preview`, no driver imports — drivers import lazily
  inside `make_camera` so a broken driver file can never poison `import camera_dock`).
- **`DATA_ROOT` = the dock repo root** (`dirname(dirname(abspath(webapp.__file__)))`). Used by
  `presets`, `recordings/`, `captures/`, and relative timelapse dirs — the bat launch (CWD=parent)
  and standalone runs write the same tree. (Verified safe: parent `presets/` is empty; parent
  `recordings/` holds only write-only old AVIs.)
- **Legacy shims**: `CameraDriver` base provides `get_exposure/set_exposure/exposure_range`,
  `get_gain/set_gain/gain_range`, `get_frame_rate/set_frame_rate/frame_rate_range`,
  `resulting_frame_rate`, `get_binning/set_binning` as thin wrappers over the funnel, so
  `presets.py` and `imaging.auto_expose` carry forward unchanged (same on-disk preset JSON keys).

## 2. The standardized command set (dock ↔ driver)

One funnel for scalar controls; dedicated verbs for geometry and acquisition. The base class owns
normalization and graceful-unsupported handling; a concrete driver *registers* what its hardware
has. **Driver-level contract: `get`/`set` never raise** — they return `ControlValue` with
`ok=False` on failure; the web layer converts to HTTP status.

### 2.1 Scalar controls — the generic funnel

```python
class ControlSpec:
    name: str            # "exposure", "gain", "fps", "black_level", ...
    units: str           # native units: "us", "index", "fps", "ADU"
    lo, hi: float        # native range — UI hint only; setters re-read hardware ranges
    step: float          # increment hint (0 = continuous)
    kind: str            # "int" | "float" — base casts int(round(v)) before writing
    scale: str           # "linear" | "log" — normalized 0..1 mapping
    display: dict        # hints: {"label": "exposure", "unit_display": "ms", "db_lo":.., "db_hi":..}
    getter, setter       # bound callables registered by the driver (setter=None → read-only)

class ControlValue:
    ok: bool; supported: bool
    value: float | None       # applied native value, READ BACK from hardware
    normalized: float | None
    error: str | None         # failure reason
    warning: str | None       # applied-but-heads-up (e.g. exposure auto-reduced)
    side_effects: dict        # {other_control: ControlValue} when a set changed another control
```

- `capabilities() -> {name: ControlSpec-as-dict + current value}` — one document; the web UI
  renders its control panel generically from it. Unsupported controls are **absent** (single
  convention). Ranges in the document are refreshed on every call.
- `set(name, value, *, normalized=False) -> ControlValue`: clamp **against ranges read from
  hardware at write time** (never the cached spec — ranges shift with geometry), snap to step,
  cast per `kind`, write, read back, return applied value. Unknown/unregistered name →
  `ControlValue(ok=False, supported=False, error=...)`.
- **Normalization**: 0..1 across the *current* native range via `scale`. Guards: `scale="log"`
  requires `lo > 0` (else registration falls back to linear); `hi == lo` → normalized 0; values
  clamped into `[lo, hi]` before converting. Exposure is log-scale; gain/fps linear.
- **fps/exposure coupling** (`clamp_fps_exposure`, ported from dualcam): setting fps reduces
  exposure to ≤ 0.98× the frame period when needed (reported in `side_effects` + `warning`);
  setting exposure beyond the current frame period is applied with a `warning` that achievable
  fps will drop. This helper is the *sole* coupling authority — the SDK's
  `exposure_time_range_us` does **not** reflect the frame period, and measured fps must come from
  the engine's own t_sw deltas (the SDK's measured-fps call returns 0.0 on this unit).

### 2.2 Geometry — ROI and binning (not in the funnel)

```python
roi_range() -> {w_min, h_min, w_inc, h_inc, x_inc, y_inc,
                min_lrx, min_lry, max_ulx, max_uly}       # sensor coords
set_roi(x, y, w, h) -> (x, y, w, h)   # FULL-SENSOR coords in, APPLIED roi out (read back)
get_roi() / reset_roi()               # reset = explicit full-sensor rect (never None)
set_binning(bx, by) / get_binning() / binning_range()
sensor_size() -> (w, h)               # PHYSICAL sensor, constant
image_size() -> (w, h)                # current readout dims (post ROI+binning)
```

`roi_range` sourcing on the Zelux (the SDK's `ROIRange` has **no increments and no direct
w/h minimums**): `w_min/h_min` come from `image_width_range_pixels.min` /
`image_height_range_pixels.min` (× binx/biny to convert to sensor coords when binning > 1);
`min_lrx/min_lry/max_ulx/max_uly` come from `ROIRange`; the four increments are driver constants
(2) documented as hints. Clamp precedence: offsets by ul maxes → corners by lr mins → enforce
w_min/h_min → **hardware read-back is authoritative** (the camera snaps beyond all of this).

Region-change flow (dock side, under the session lock):
1. Read the driver's **last-commanded** fps and exposure (a small commanded-value store in the
   driver, updated by every successful set — not a pre-change hardware read).
2. Stop engine (engine.stop() clears the latest-frame slot; abort with 503 if the producer thread
   fails to join — never touch geometry with a live producer).
3. Driver applies geometry inside its SDK lock (single atomic disarm → set → read-back → re-arm →
   re-trigger sequence).
4. Re-apply fps then exposure through the funnel — which clamps against **fresh** hardware ranges.
5. Restart engine; read refreshed `capabilities()` **after** re-arm (ranges depend on the armed
   state); return `{roi, image_size, capabilities}`.
Blocked with 409 while recording **or timelapsing**.

### 2.3 Acquisition and frames

```python
start(max_throughput=False); stop(); is_grabbing
grab(timeout_ms) -> Frame             # requires streaming (engine owns start/stop)
class Frame:
    data: np.ndarray                  # 2-D mono, np.copy'd at the poll site (never SDK-owned)
    meta: FrameMeta(t_sw,             # host perf_counter at dequeue
                    hw_count,         # hardware frame counter (int on TSI; None elsewhere)
                    hw_timestamp_ns)  # device timestamp or None — None is COMMON (unit-dependent)
```

- Engine: `latest() -> (Frame | None, index)`; sink signature `sink(frame: Frame, index, t)`.
  `_frame_index` is **monotonic across restarts** (never reset — index-based freshness guards in
  auto-exposure and timelapse depend on it); `stop()` clears `_latest`. Sink invocation happens
  under a dedicated `_sink_lock`, and `set_sink` acquires it — so `set_sink(None)` returning is a
  barrier: no `submit` is in flight afterwards.
- The TSI SDK 0.0.8 already returns `image_buffer` shaped `(h, w)` from arm-time cached dims — the
  driver just `np.copy`s it (no reshape from live properties; that was the geometry race).
- hw_count re-baselines at every arm; drop detection per recording is safe because ROI changes are
  blocked while recording.
- Stale-geometry guard: webapp session tracks expected `image_size`; the MJPEG/snapshot path skips
  frames whose `data.shape` mismatches.

Lifecycle: `connect()` / `disconnect()` / `is_connected` / `device_info` / `bit_depth`.

## 3. Zelux driver specifics (`drivers/zelux.py`)

- **One internal `threading.RLock` around every SDK touch** (the TSI DLL is not concurrency-safe;
  dualcam needed a global SDK lock for the same reason). The poll loop must *not* hold the lock
  across a blocking poll: set `image_poll_timeout_ms = 0` (non-blocking) and loop
  `{with lock: get_pending_frame_or_null()} + sleep(~1 ms)` so setter threads never starve.
  `set()` holds the lock across clamp+write+read-back; `set_roi` holds it across the entire
  disarm → set → read-back → `frames_per_trigger=0` → arm → `issue_software_trigger` → re-apply
  sequence (RLock so internal re-applies can call `set()`).
- DLL bootstrap before `TLCameraSDK()`: `THORLABS_TSI_DLL_DIR` env or ThorCam default; **both**
  `os.add_dll_directory` and PATH prepend. SDK import deferred into `connect()`.
- Dispose order: Frame refs → `camera.dispose()` → `sdk.dispose()` in `finally`. On dispose
  failure, force-reset `TLCameraSDK._is_sdk_open = False` before the next connect (documented
  0.0.8 workaround — `dispose()` raises *without* clearing the singleton flag).
- Continuous video triple: `SOFTWARE_TRIGGERED` + `frames_per_trigger_zero_for_unlimited=0`
  (re-set before **every** arm) + exactly one `issue_software_trigger()` after arm.
- fps: enable `is_frame_rate_control_enabled` **once, at connect, while disarmed**; if the enable
  fails or reads back False, don't register the fps control at all. The live set path writes
  `frame_rate_control_value` only. Registered `lo` floored at `max(range.min, 1.0)` (SDK can
  report 0.0).
- Gain: integer index (0–480 ↔ 0–48 dB on CS165MU); display hints carry converted dB endpoints
  (`convert_gain_to_decibels` at lo and hi — not assumed linear).
- black_level: register iff `black_level_range.max > 0` (the SDK swallows error 1002 into
  `Range(0,0)` — there is no exception to catch).
- Exposure: integer µs (`c_longlong`); all int controls cast `int(round(v))` before the write.
- 10-bit data in the LOW bits of uint16; display shift by `bit_depth − 8` (absolute levels — never
  per-frame MINMAX). Snapshots stay 16-bit.
- Optional extras probed at connect, registered only if present: black_level, hot-pixel
  correction, LED. Trigger modes deferred (SW-triggered pinned). All scalar controls on the Zelux
  are live-settable while armed — no disarm-wrapper machinery.

## 4. Eval web server (`webapp.py`)

Carried forward: session model, MJPEG generator (asyncio pacing, JPEG off the event loop),
timelapse (drift-free schedule, stale guard via monotonic engine index, **UI state restore on page
load** incl. the already-running and server-restarted paths), presets (auto-`default` at connect,
before the engine starts so ROI applies without a region cycle), runtime connect/disconnect (409
while recording/timelapsing; page reload re-establishes stream+controls), per-camera failure
isolation, auto-exposure.

Locking (from the concurrency review): session lock required by control POSTs, auto-exposure,
preset load, connect (**`start()` now takes the lock and re-checks `ok`**), disconnect, record
lifecycle, timelapse lifecycle, region changes. NOT required by: MJPEG/snapshot (read
`engine.latest()` only — **the render path never touches the driver**; status overlay uses
session-cached values), `/info` and capability GETs (driver RLock suffices). `Timelapse.stop`
detaches under the lock, joins outside it.

New/changed:

- **Generic control panel.** `GET /cam/{n}/capabilities` → ControlSpec document + values; JS
  builds one row per control (normalized 0..1000 slider + native-value box + unit label).
  `POST /cam/{n}/control/{name}?value=&normalized=` → applied ControlValue JSON; errors: 404
  unknown control, 422 supported=False, 409 state conflict. Sliders debounce (~80 ms), re-sync to
  the echoed applied value, and carry a client `capsRev` — echoes from a superseded revision are
  discarded; a response with non-empty `side_effects` triggers a capabilities refetch (the
  fps→exposure clamp is visible immediately).
- **Drag-ROI zoom.** Exact client transform (all CSS px; DPR-independent): inputs captured at
  pointerdown — applied ROI `(Rx,Ry,Rw,Rh)` (last server echo, sensor px), binning `(bx,by)`,
  server-reported image size `(Iw,Ih)` (authoritative — `naturalWidth` is only a *gate*, not the
  source), displayed content size excluding borders. Box corners:
  `ix0 = clamp(floor(min(u0,u1)·Iw/Dw), 0, Iw−1)`, `ix1 = clamp(ceil(max(u0,u1)·Iw/Dw)−1, ix0,
  Iw−1)` (same for y); sensor rect `x = Rx + ix0·bx`, `w = (ix1−ix0+1)·bx`, etc. POST; **adopt the
  echoed applied ROI** (hardware snaps). Drag is *gated* on `img.naturalWidth === Iw` (stale-frame
  protection) and cancelled if any captured input changed by pointerup. Browser plumbing:
  `draggable=false` + `dragstart`/`contextmenu` preventDefault (scoped to the stream), Pointer
  Events with `setPointerCapture`, `user-select:none`, absolutely-positioned overlay div for the
  box. Min drag 8 display px. After every successful ROI change the client resets
  `img.src = stream?e=<epoch>` (an `<img>` never auto-reconnects a stalled MJPEG response — this
  is also the reconnect path), cancels pending slider debounces, and re-renders sliders from the
  returned capabilities (updates applied LAST). Display CSS gives the stream a **fixed width**
  (small ROIs upscale — zooming must zoom).
- **Zoom stack semantics**: every *successful* ROI change (drag, typed Apply, preset-with-ROI)
  pushes the *previous applied* ROI after the 2xx lands; a pop never pushes; the full-frame button
  clears the stack; right-click with empty stack at full frame is a no-op; pushes are skipped when
  applied == current. On error (409 toast) stack and fields stay untouched. The 1 s status poll
  carries the current ROI — if it differs from the client's last echo (another tab changed it),
  the stack resets and fields redraw.
- **Recording state machine**: `rec_state ∈ {idle, recording, encoding}`, all transitions under
  the session lock. `record_start` (409 unless idle): makedirs + `%f`-timestamped path first,
  create recorder, assign state, `set_sink` **last** (cleanup on any failure). `record_stop`
  (409 unless recording): `set_sink(None)` (barrier), state=encoding, spawn **non-daemon** encode
  thread; encode runs without the lock (never touches the camera); on completion stores stats,
  state=idle. `/record/status` polls state+stats. Disconnect/shutdown joins the encode thread.
  Optional frame/duration limits: the recorder just stops *accepting* at the limit and flags
  `limit_reached`; the UI poll sees it and calls stop — no cross-thread finalization.
  Sidecars (best-effort, never raise): `<stem>_timestamps.npy` (sw monotonic, float64),
  `<stem>_hwclock.npz` (hw_count int64, hw_timestamp_ns **float64 with NaN** for missing — the
  Zelux may return None on every frame), `<stem>.json` (metadata.py: sw/hw measured fps, genuine
  drops from hw-counter gaps when available). Container fps = engine-measured rate; encode BGR
  (the tested path). `Cache-Control: no-store` on /stream and /snapshot.
- **Status line** per camera: model, serial, image size, acq fps (engine-measured), saturation %,
  REC state — all from session/engine caches, no driver calls.

## 5. Graceful failure rules (summary)

| Condition | Behavior |
|---|---|
| Control not supported / unknown | absent from capabilities; `set`/`get` → `supported=False`; HTTP 422/404; UI hides row |
| Value out of range | clamped + snapped against fresh hardware ranges, applied value echoed |
| ROI while recording/timelapsing | 409 with reason; UI toast; zoom stack untouched |
| Camera fails to connect | session `ok=False` + reason; app and other cameras unaffected |
| Unknown/paused camera token | `UnavailableCamera` stub; session `ok=False`; dashboard shows unavailable |
| Frame with stale geometry | shape-checked against expected image size, skipped |
| Producer thread won't join | region change aborted 503; geometry never touched with a live producer |
| Sidecar write failure | swallowed; noted in stats |
| SDK dispose failure | singleton flag force-reset; next connect recreates the SDK |

## 6. Definition of done (tonight)

1. `python -m camera_dock.webapp zelux` streams the live Zelux at ~30 fps preview.
2. Drag a box → sensor ROI shrinks → measured acquisition fps rises (status line); right-click
   steps back out (uniform undo); coordinate fields track and accept typed values. [verified-run]
3. Exposure/gain sliders drive normalized 0..1 across native ranges; ranges re-scale after ROI;
   fps→exposure clamp surfaces as a side-effect refresh. [verified-run]
4. Record start/stop → playable video + three sidecars; hwclock arrays handle None timestamps
   (NaN) without dying. [verified-artifact]
5. Unsupported-control path proven (e.g. `pixel_clock` on the Zelux → clean 422). [verified-run]
6. Parent contract: `python -m xsphere_daq.panel basler zelux hayear` (the bat's exact token set,
   CWD = parent repo) boots, zelux works under `/cameras`, paused tokens show unavailable.
   [verified-run]
