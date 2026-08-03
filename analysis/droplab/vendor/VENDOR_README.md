# Vendored count-droplets scripts

Verbatim copies from https://github.com/Moore-Lab/count-droplets
at commit `43e0d9984e2a4b35a0c2fbf76bf80be4e36f6775` (fetched 2026-08-02).

These are the lab's canonical analysis algorithms — droplab calls their pure
functions headlessly through `../vendor_api.py` (Agg backend, no plt.show(),
project-root monkeypatched to `analysis/`). DO NOT edit these files; to update,
re-copy from the upstream repo and bump the commit hash here.

| file | used by droplab for |
|------|--------------------|
| track_droplets.py | `link_droplets_by_endpoints`, `save_track_data`, `build_streamlines` |
| set_distance_threshold.py | auto linking-distance selection (`plot_droplets_vs_distance`) |
| analyze_distributions.py | `load_droplet_data`, `analyze_independent` (4x3 fits figure) |
| analyze_tracks.py | `load_track_data`, `analyze_tracks` (6-panel velocity figure) |
| longitudinal_analysis.py | `run_longitudinal_analysis` (chunked time series) |
| plot_longitudinal.py | `plot_longitudinal` (overview figure) |

Not vendored (out of scope for the dock): environmental_correlations.py,
query_db.py (Yale InfluxDB access), analyze_droplets.py (its detection
algorithm is ported natively in `../detect.py` + `../store.py` so the web UI
can preview per-frame with progress/cancellation; outputs are byte-compatible).

Known upstream quirks droplab works around (see vendor_api.py):
- `track_droplets.main()` auto-scan appends the endpoints dict instead of a
  track count (crash) and re-runs saves per scan step — droplab reimplements
  the scan loop calling `link_droplets_by_endpoints` directly.
- `analyze_tracks.analyze_tracks()` calls `plt.show()` unconditionally — a
  no-op under the Agg backend.
- `plt.cm.get_cmap(name, N)` in track_droplets' plot functions is removed in
  matplotlib >= 3.9 — droplab never calls those plot functions.
- `longitudinal_analysis` anchors data/, plots/ to its own file location —
  droplab monkeypatches `_get_project_root` to `analysis/`.
