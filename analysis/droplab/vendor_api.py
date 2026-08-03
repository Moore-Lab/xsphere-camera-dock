"""Headless facade over the vendored count-droplets scripts.

Importing this module pulls in matplotlib + scipy + the vendored scripts
(~1.5 s, ~150 MB), so :mod:`analysis.droplab.server` imports it LAZILY — only
when an analysis stage actually runs. Keep it that way: the detection and
preview paths must stay light enough to mount inside the DAQ panel.

The vendor/ modules are verbatim lab code (flat scripts importing each other
by bare name), so this module:

  * forces the matplotlib Agg backend BEFORE any vendor import (their
    plt.show() calls then no-op),
  * puts vendor/ on sys.path so `import analyze_distributions` etc. resolve,
  * monkeypatches longitudinal_analysis._get_project_root to analysis/ so its
    data/ and plots/ outputs land in droplab's tree,
  * wraps each stage in a clean function returning JSON-safe dicts,
  * reimplements the auto linking-distance scan (upstream main()'s scan
    appends a dict where it means a count and re-saves outputs per step).
"""

from __future__ import annotations

import os
import sys
import threading

# The vendored scripts print unicode arrows ('→') in their progress output;
# on Windows a cp1252 stdout raises UnicodeEncodeError and kills the stage.
# Replace unencodable characters instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(errors="replace")
        except (ValueError, OSError):
            pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .paths import ANALYSIS_ROOT, data_dir_for, plots_dir_for, video_basename
from .store import json_safe  # noqa: F401  (re-exported for callers)

_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import analyze_distributions            # noqa: E402
import analyze_tracks as _analyze_tracks  # noqa: E402
import longitudinal_analysis            # noqa: E402
import plot_longitudinal as _plot_longitudinal  # noqa: E402
import set_distance_threshold           # noqa: E402
import track_droplets                   # noqa: E402

# Longitudinal outputs (data/<basename>/, plots/<basename>/) anchor to the
# script's own directory upstream; re-anchor to analysis/.
longitudinal_analysis._get_project_root = lambda: ANALYSIS_ROOT

# pyplot keeps GLOBAL figure state and the job pool runs 2 workers — every
# figure-producing stage must hold this lock (detection never plots, so it
# still runs concurrently with a plotting stage).
_MPL_LOCK = threading.Lock()


# ── tracking ────────────────────────────────────────────────────────────────

def auto_distance_scan(droplet_npy: str, *, scan_max: int = 75,
                       k_sigma: float = 2.0, fit_mode: str = "joint",
                       progress=None, save_plot_to: str | None = None):
    """Fixed port of upstream's auto-threshold scan.

    Links at every max_distance in [0, scan_max) counting the resulting number
    of tracks, saves <stem>_droplets_vs_distance.npy, runs the vendored
    line+exponential threshold picker, and (optionally) saves its diagnostic
    figure. Returns (best_distance, n_tracks_at_best, curve list).
    """
    data = track_droplets.load_tracking_data(droplet_npy)
    curve = []
    for i in range(scan_max):
        tracks = track_droplets.link_droplets_by_endpoints(data, max_distance=float(i))
        curve.append(len(tracks))
        if progress is not None:
            progress(i + 1, scan_max)
    curve_arr = np.array(curve, dtype=np.int32)
    stem = droplet_npy[:-len("_droplet_data.npy")] if droplet_npy.endswith(
        "_droplet_data.npy") else os.path.splitext(droplet_npy)[0]
    np.save(stem + "_droplets_vs_distance.npy", curve_arr)

    with _MPL_LOCK:
        plt.close("all")
        best, n_at = set_distance_threshold.plot_droplets_vs_distance(
            data=curve_arr, show_plot=bool(save_plot_to), fit_mode=fit_mode,
            k_sigma=k_sigma, two_sided=False)
        if save_plot_to:
            for num in plt.get_fignums():    # show() no-ops on Agg; grab figures
                plt.figure(num).savefig(save_plot_to, dpi=120, bbox_inches="tight")
                break
        plt.close("all")
    return int(best), int(n_at), curve


def run_tracking(video_path: str, *, max_distance=24.0, min_track_length: int = 1,
                 fps: float = 30.0, pixels_per_um: float = 0.1878,
                 auto_distance: bool = False, progress=None) -> dict:
    """Link detections into tracks and write <basename>_track_data.npz.

    Reads data/<basename>/<basename>_droplet_data.npy (written by the
    detection stage). With auto_distance=True the linking radius comes from
    the vendored threshold scan. Returns a JSON-safe summary.
    """
    basename = video_basename(video_path)
    ddir = data_dir_for(video_path, create=True)
    droplet_npy = os.path.join(ddir, f"{basename}_droplet_data.npy")
    if not os.path.isfile(droplet_npy):
        raise FileNotFoundError(f"run detection first — missing {droplet_npy}")

    scan = None
    if auto_distance:
        pdir = plots_dir_for(video_path, create=True)
        best, n_at, curve = auto_distance_scan(
            droplet_npy, progress=progress,
            save_plot_to=os.path.join(pdir, f"{basename}_distance_scan.png"))
        max_distance = float(best)
        scan = {"best_distance_px": best, "n_tracks_at_best": n_at, "curve": curve}

    data = track_droplets.load_tracking_data(droplet_npy)
    tracks = track_droplets.link_droplets_by_endpoints(data, max_distance=float(max_distance))
    if min_track_length and min_track_length > 1:
        tracks = {tid: recs for tid, recs in tracks.items()
                  if len(recs) >= min_track_length}

    npz_path = os.path.join(ddir, f"{basename}_track_data.npz")
    track_droplets.save_track_data(
        tracks, filepath=npz_path, fps=fps, pixels_per_um=pixels_per_um,
        max_distance=max_distance, min_track_length=min_track_length)

    lengths = [len(recs) for recs in tracks.values()]
    return json_safe({
        "npz_path": npz_path,
        "max_distance_px": max_distance,
        "min_track_length": min_track_length,
        "n_detections": int(len(data)),
        "n_tracks": len(tracks),
        "n_multi_frame_tracks": sum(1 for n in lengths if n >= 2),
        "longest_track_frames": max(lengths) if lengths else 0,
        "auto_scan": scan,
    })


# ── distributions ───────────────────────────────────────────────────────────

def run_distributions(video_path: str, *, density_kg_m3: float = 3520.0,
                      r_mm=None, z_mm=None) -> dict:
    """Vendored analyze_independent: 4x3 fits figure + JSON-safe results."""
    basename = video_basename(video_path)
    ddir = data_dir_for(video_path)
    droplet_npy = os.path.join(ddir, f"{basename}_droplet_data.npy")
    results_json = os.path.join(ddir, f"{basename}_test_results.json")
    if not os.path.isfile(droplet_npy):
        raise FileNotFoundError(f"run detection first — missing {droplet_npy}")

    per_droplet = analyze_distributions.load_droplet_data(
        droplet_npy, results_json=results_json if os.path.isfile(results_json) else None)

    pdir = plots_dir_for(video_path, create=True)
    fig_path = os.path.join(pdir, f"{basename}_distributions.png")
    with _MPL_LOCK:
        plt.close("all")
        step1 = analyze_distributions.analyze_independent(
            per_droplet, density_kg_m3=density_kg_m3, save_path=fig_path,
            r_mm=r_mm, z_mm=z_mm)
        combined = analyze_distributions.analyze_combined(step1, per_droplet)
        plt.close("all")
    return json_safe({"figure": fig_path, "independent": step1,
                      "combined": combined.get("combined")})


def run_track_analysis(video_path: str, *, n_angle_bins: int = 24,
                       n_time_windows: int = 5, spatial_bins: int = 8) -> dict:
    """Vendored analyze_tracks: 6-panel velocity/anisotropy figure + stats."""
    basename = video_basename(video_path)
    npz_path = os.path.join(data_dir_for(video_path), f"{basename}_track_data.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"run tracking first — missing {npz_path}")

    pdir = plots_dir_for(video_path, create=True)
    fig_path = os.path.join(pdir, f"{basename}_track_analysis.png")
    with _MPL_LOCK:
        plt.close("all")
        td = _analyze_tracks.load_track_data(npz_path)
        results = _analyze_tracks.analyze_tracks(
            td, save_path=fig_path, n_angle_bins=n_angle_bins,
            n_time_windows=n_time_windows, spatial_bins=spatial_bins)
        plt.close("all")                   # their plt.show() no-ops on Agg
    out = json_safe(results)
    if isinstance(out, dict):
        out["figure"] = fig_path if os.path.isfile(fig_path) else None
    return out


# ── longitudinal ────────────────────────────────────────────────────────────

def run_longitudinal(log_path: str, video_path: str, *,
                     chunk_size_frames=None, chunk_duration_s=None,
                     start_utc=None, density_kg_m3: float = 3520.0) -> dict:
    """Vendored chunked time-series analysis from a saved parameter log."""
    with _MPL_LOCK:
        plt.close("all")
        arr, out_path = longitudinal_analysis.run_longitudinal_analysis(
            log_path,
            chunk_size_frames=chunk_size_frames,
            chunk_duration_s=chunk_duration_s,
            start_utc=start_utc,
            density_kg_m3=density_kg_m3)
        basename = video_basename(video_path)
        pdir = plots_dir_for(video_path, create=True)
        overview = os.path.join(pdir, f"{basename}_longitudinal_overview.png")
        _plot_longitudinal.plot_longitudinal(arr, title_prefix=basename,
                                             save_path=overview)
        plt.close("all")
    return json_safe({
        "npy_path": out_path,
        "overview_figure": overview,
        "n_chunks": int(len(arr)),
        "fields": list(arr.dtype.names),
    })
