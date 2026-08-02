# Measurement layer — shared between live and offline analysis

*For whoever is building the analysis tab. Written 2026-08-02 by the camera-dock
side. Short version: **don't rewrite the measuring — import it.***

## What exists

| file | what it is |
|---|---|
| `camera_dock/static/measure.js` | the whole measurement engine: tools, shapes, labels, measurements table, calibration formatting, persistence, JSON import/export. **Zero dependencies, no build step, no imports.** |
| `camera_dock/static/measure.css` | its styles (toolbar, overlay, table). Self-contained. |
| `camera_dock/static/measure.html` | **offline host** — open a still or video, measure it. No server needed. This is the worked example to integrate. |
| `camera_dock/static/cam.js` | **live host** — the camera page. Same engine, one extra transform. |

Both hosts run the *same* `measure.js`. It is the single source of truth; please
extend it rather than forking it.

## The one idea you need

**Annotations are stored in native pixel coordinates**, never in screen
coordinates:

* offline — pixels of the image/video **file**
* live — pixels of the camera **sensor**

Screen positions are derived on every redraw through a `transform`. That's why a
measurement doesn't change when the window resizes, and why (live) it stays
pinned to the scene when the sensor ROI is cropped.

**The transform is the entire online/offline bridge.** Offline passes nothing and
gets `MeasureLayer.defaultTransform` (file pixels → displayed box). Live passes
one that also shifts by the ROI origin and scales by binning. Nothing else about
the two hosts differs.

## Using it (offline)

```html
<link rel="stylesheet" href=".../measure.css">
<div class="measure-stage" id="stage">
  <img id="media"><svg id="overlay"></svg>
</div>
<div class="toolbar" id="toolbar"></div>
<table class="measure-tbl"><tbody id="rows"></tbody></table>
<script src=".../measure.js"></script>
<script>
  const M = new MeasureLayer({
    media: document.getElementById('media'),
    svg: document.getElementById('overlay'),
    storageKey: 'myapp.' + filename,      // optional autosave
    onChange: list => console.log(list.length + ' annotations'),
  });
  M.renderList(document.getElementById('rows'));
  M.buildToolbar(document.getElementById('toolbar'));
  M.setScale(0.34, '10x');                 // µm per pixel, or null for px only
</script>
```

That's the complete integration. `measure.html` does exactly this plus file
loading, video controls and calibration entry — copy from it freely.

Useful members: `M.annots`, `M.lines`, `M.byId(id)`, `M.add/remove/undo/clear`,
`M.export()` / `M.import(json)`, `M.setScale(umPerPx, label)`, `M.setTool(name)`,
`M.describe(a)`, and the statics `MeasureLayer.lineLength(a)`,
`MeasureLayer.contentRect(media)`, `MeasureLayer.num(v)`.

### Where the code should live

`analysis/` is inside this repo, so the analysis app can serve or reference
`camera_dock/static/measure.js` directly (e.g. mount `camera_dock/static` as a
static route, or copy it in at build/startup). Prefer referencing over copying —
a stale copy is how the two sides drift apart.

## Calibration flows between the two halves

Scale is **µm per native (sensor) pixel**, deliberately not per image pixel, so
it survives binning and ROI changes.

* Live: stored server-side per camera, `calibrations/<camera>.json`, several
  named entries (one per objective) with one active. API under
  `/cam/{name}/calibration`.
* **Every dock recording writes `<stem>.json` next to the video containing
  `um_per_px`, `calibration` (the entry name), `roi` and `binning`.** Load that
  sidecar offline and you measure at exactly the scale the camera was calibrated
  at — `measure.html` has a file input that does this.
* Offline-only material: enter µm/px directly, or draw a line across something of
  known size and let it compute the scale.

`<stem>_timestamps.npy` and `<stem>_hwclock.npz` sit alongside with per-frame
host timestamps, hardware frame counters and device timestamps — use those for
timing/rate work rather than the video container's fps, which is only a playback
speed. Sidecar details: `camera_dock/metadata.py`.

## Notes convention

We're editing this file and `measure.js` from both sides. When you change
something the other side must know about, leave an inline comment:

```js
// NOTE[offline->live]: ...   analysis dev  -> camera dock dev
// NOTE[live->offline]: ...   camera dock dev -> analysis dev
```

Keep `measure.js` generic — camera-specific code belongs in `cam.js`,
file/video-specific code in the offline host. If you need a new hook, add an
option with a default so the other host keeps working untouched.

## Known limits (deliberate, not bugs)

* Shapes are line / circle / rectangle, drawn once; there is no move-or-resize
  after the fact (delete and redraw). Editing handles are the obvious next step.
* Annotations are display-only — they are never burned into snapshots or
  recordings. Export the JSON if you need them downstream.
* Browsers can't decode the dock's FFV1 `.avi` or 16-bit TIFFs. Offline video
  measuring needs a browser-playable transcode, or feed frames in as PNGs. If the
  analysis app decodes frames itself (server-side or via canvas), point
  `MeasureLayer` at that `<canvas>` — it accepts `<img>`, `<video>` and
  `<canvas>` identically.
* Video annotations are not per-frame; they persist across the whole clip. If you
  need time-scoped annotations, add a `t` field and filter in `redraw()` — please
  do it in `measure.js` behind an option so live keeps working.
