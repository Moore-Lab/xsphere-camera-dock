# droplab — browser droplet analysis for dock recordings

> **Handing this over?** Read `../../../docs/HANDOVER-panel-and-analysis.md`
> (parent repo, `xsphere-daq/docs/`) first — it covers this app *and* the tabbed
> DAQ panel that mounts it, plus the contracts between them.

A web UI around the [Moore-Lab/count-droplets](https://github.com/Moore-Lab/count-droplets)
pipeline, living inside the camera dock so you can **analyze recordings while
the next one is being captured** in another browser window.

```
python -m analysis.droplab.server            # http://127.0.0.1:8020
```

(or start the `droplet-analysis` entry in `.claude/launch.json`.)

**In the DAQ**, the parent panel (`python -m xsphere_daq.panel …`) mounts this app
at `/analysis` as its Analysis tab — same code, no second process. Pages derive
their API base from `window.location`, so every URL below works standalone *and*
under the mount. Keep it that way, and keep `vendor_api` (matplotlib + scipy)
imported lazily so the panel starts fast whether or not anyone opens the tab.

## Workflow

1. **`/`** lists everything in `recordings/` (newest first, refreshes every
   5 s — new recordings appear as the dock finishes encoding them).
2. **`/lab/{video}`** is the analysis page:
   - scrub frames; **drag on the image** to set the ROI (right-click = full frame)
   - live view modes: raw / **droplets** (detection preview) / **processed**
     (CLAHE→blur→threshold→morphology binary) / **sum** (persistence view)
   - tune threshold / min-max area / time window / fps override / exposure
   - **run detection** → background job with progress; writes
     count-droplets-compatible outputs (below) + a reloadable parameter log
   - then **track** (greedy endpoint linking, optional auto distance scan),
     **velocity fig** (6-panel anisotropy/flow analysis), **distributions**
     (Poisson/Rayleigh/half-normal fits figure), **longitudinal**
     (chunked time series from the saved log)
   - results and all generated figures render at the bottom of the page.

## Output tree (count-droplets compatible)

```
analysis/
  data/<basename>/<basename>_droplet_data.npy     frame,id,x,y,length,width,angle (in-ROI)
  data/<basename>/<basename>_tracking_data.npy    + streak endpoint coords
  data/<basename>/<basename>_track_data.npz       linked tracks + velocities
  data/<basename>/<basename>_test_results.json    stats + calibration
  data/<basename>/longitudinal_analysis_*.npy     per-chunk time series
  logs/<basename>_<timestamp>.json                reloadable parameter logs
  plots/<basename>/*.png                          all generated figures
```

The upstream scripts can be pointed at `analysis/` as their project root and
will consume these files unchanged.

## Layout

| file | role |
|------|------|
| `server.py` | FastAPI app + routes (`python -m analysis.droplab.server`) |
| `detect.py` | native port of the count-droplets detection algorithm (headless, cancellable) |
| `store.py` | statistics + byte-compatible npy/json/log writers |
| `video.py` | recording discovery (in-progress heuristic), sidecar fps, frame reader cache |
| `jobs.py` | background job manager (2 workers, cooperative cancel) |
| `vendor_api.py` | headless facade over the vendored lab scripts |
| `vendor/` | verbatim count-droplets scripts (see `vendor/VENDOR_README.md`) |
| `static/` | index + lab pages (vanilla JS, dock styling) |

Notes:
- fps prefills from the recording's metadata sidecar (`sw_measured_fps`) when
  it disagrees with the container fps — mp4/avi headers often carry a nominal
  rate, not the true capture rate.
- a video without its `.json` sidecar that was modified in the last few
  seconds is flagged "writing" (the dock writes sidecars after encoding).
- detection runs on full frames; the ROI tags droplets (`in_roi`) for
  filtering/density normalization — same semantics as upstream.
