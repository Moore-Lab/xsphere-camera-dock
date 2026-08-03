"""Statistics + on-disk outputs — port of count-droplets result/log writers.

File formats are byte-compatible with the originals (same npy dtypes, same
JSON keys) so track_droplets.py / analyze_distributions.py / longitudinal
tooling can consume droplab output unchanged. Layout (see paths.py):

    data/<basename>/<basename>_droplet_data.npy
    data/<basename>/<basename>_tracking_data.npy
    data/<basename>/<basename>_test_results.json
    logs/<basename>_<YYYY-MM-DD_HH-MM-SS>.json
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re

import numpy as np

from .paths import data_dir_for, logs_dir, video_basename

DEFAULT_PIXELS_PER_UM = 0.1878        # count-droplets calibration constant

DROPLET_DTYPE = [('frame', 'i4'), ('droplet_id', 'i4'), ('x', 'f4'), ('y', 'f4'),
                 ('length', 'f4'), ('width', 'f4'), ('angle', 'f4')]

TRACKING_DTYPE = DROPLET_DTYPE + [
    ('length_start_x', 'f4'), ('length_start_y', 'f4'),
    ('length_end_x', 'f4'), ('length_end_y', 'f4'),
    ('width_start_x', 'f4'), ('width_start_y', 'f4'),
    ('width_end_x', 'f4'), ('width_end_y', 'f4'),
]


def calculate_statistics(frame_data, roi, frame_width, frame_height):
    """Dual-normalized count/density statistics — port of calculate_statistics()."""
    if not frame_data:
        raise ValueError("No data to analyze.")
    x_start, x_stop, y_start, y_stop = roi

    roi_width = x_stop - x_start
    roi_height = y_stop - y_start
    roi_area_px2 = roi_width * roi_height
    roi_beam_radius_px = roi_width / 2
    roi_beam_cross_section_px2 = np.pi * roi_beam_radius_px ** 2

    full_frame_area_px2 = frame_width * frame_height
    full_beam_radius_px = frame_width / 2
    full_beam_cross_section_px2 = np.pi * full_beam_radius_px ** 2

    counts_total = np.array([f['droplet_count'] for f in frame_data])
    counts_in_roi = np.array([f.get('droplet_count_in_roi', f['droplet_count'])
                              for f in frame_data])

    mean_count_total = np.mean(counts_total)
    std_count_total = np.std(counts_total)
    mean_count_roi = np.mean(counts_in_roi)
    std_count_roi = np.std(counts_in_roi)

    mean_density_roi = mean_count_roi / roi_beam_cross_section_px2
    std_density_roi = std_count_roi / roi_beam_cross_section_px2
    mean_density_full = mean_count_total / full_beam_cross_section_px2
    std_density_full = std_count_total / full_beam_cross_section_px2

    return {
        'frames_analyzed': len(frame_data),

        'frame_width': frame_width,
        'frame_height': frame_height,
        'frame_area_pixels': full_frame_area_px2,
        'full_beam_radius_pixels': full_beam_radius_px,
        'full_beam_cross_section_pixels': full_beam_cross_section_px2,

        'roi_width': roi_width,
        'roi_height': roi_height,
        'roi_area_pixels': roi_area_px2,
        'roi_beam_radius_pixels': roi_beam_radius_px,
        'roi_beam_cross_section_pixels': roi_beam_cross_section_px2,

        'mean_droplet_count_total': float(mean_count_total),
        'std_droplet_count_total': float(std_count_total),
        'min_droplet_count_total': int(np.min(counts_total)),
        'max_droplet_count_total': int(np.max(counts_total)),

        'mean_droplet_count_roi': float(mean_count_roi),
        'std_droplet_count_roi': float(std_count_roi),
        'min_droplet_count_roi': int(np.min(counts_in_roi)),
        'max_droplet_count_roi': int(np.max(counts_in_roi)),

        'mean_density_roi_per_pixel': float(mean_density_roi),
        'std_density_roi_per_pixel': float(std_density_roi),
        'mean_density_full_per_pixel': float(mean_density_full),
        'std_density_full_per_pixel': float(std_density_full),

        # Legacy fields kept for downstream compatibility
        'beam_radius_pixels': roi_beam_radius_px,
        'beam_cross_section_pixels': roi_beam_cross_section_px2,
        'mean_droplet_count': float(mean_count_roi),
        'std_droplet_count': float(std_count_roi),
        'min_droplet_count': int(np.min(counts_in_roi)),
        'max_droplet_count': int(np.max(counts_in_roi)),
        'mean_density_per_pixel': float(mean_density_roi),
        'std_density_per_pixel': float(std_density_roi),
        'mean_density_per_1000px': float(mean_density_roi * 1000),
        'std_density_per_1000px': float(std_density_roi * 1000),
    }


def _iter_droplets(frame_data):
    for frame_info in frame_data:
        for droplet_id, d in enumerate(frame_info.get('droplets', [])):
            yield frame_info['frame_number'], droplet_id, d


def save_droplet_data(frame_data, video_path):
    """Write <basename>_droplet_data.npy (in-ROI droplets only). Returns (path, array)."""
    if not frame_data:
        # Upstream raises before writing — never clobber a previous good run
        # with an empty array.
        raise ValueError("No data to save.")
    basename = video_basename(video_path)
    out = os.path.join(data_dir_for(video_path, create=True),
                       f"{basename}_droplet_data.npy")
    rows = [(fn, did, d['x'], d['y'], d['length'], d['width'], d['angle'])
            for fn, did, d in _iter_droplets(frame_data) if d.get('in_roi', True)]
    arr = np.array(rows, dtype=DROPLET_DTYPE)
    np.save(out, arr)
    return out, arr


def save_tracking_data(frame_data, video_path):
    """Write <basename>_tracking_data.npy (with endpoint coords). Returns (path, array)."""
    if not frame_data:
        raise ValueError("No data to save.")
    basename = video_basename(video_path)
    out = os.path.join(data_dir_for(video_path, create=True),
                       f"{basename}_tracking_data.npy")
    rows = []
    for fn, did, d in _iter_droplets(frame_data):
        if not d.get('in_roi', True):
            continue
        ls = d.get('length_start', (d['x'], d['y']))
        le = d.get('length_end', (d['x'], d['y']))
        ws = d.get('width_start', (d['x'], d['y']))
        we = d.get('width_end', (d['x'], d['y']))
        rows.append((fn, did, d['x'], d['y'], d['length'], d['width'], d['angle'],
                     ls[0], ls[1], le[0], le[1], ws[0], ws[1], we[0], we[1]))
    arr = np.array(rows, dtype=TRACKING_DTYPE)
    np.save(out, arr)
    return out, arr


def save_results(frame_data, video_path, params, frame_width, frame_height):
    """Write <basename>_test_results.json. Returns (path, results dict).

    *params* carries start_time, end_time, roi, fps, fps_override,
    exposure_time_s (missing keys are written as None, like the original).
    """
    basename = video_basename(video_path)
    out = os.path.join(data_dir_for(video_path, create=True),
                       f"{basename}_test_results.json")
    roi = (params['x_start'], params['x_stop'], params['y_start'], params['y_stop'])
    results = calculate_statistics(frame_data, roi, frame_width, frame_height)
    full = {
        'video_path': str(video_path),
        'analysis_parameters': {
            'start_time': params.get('start_time'),
            'end_time': params.get('end_time'),
            'x_start': params['x_start'],
            'x_stop': params['x_stop'],
            'y_start': params['y_start'],
            'y_stop': params['y_stop'],
            'frame_width': frame_width,
            'frame_height': frame_height,
            'pixels_per_um': params.get('pixels_per_um', DEFAULT_PIXELS_PER_UM),
            'fps': params.get('fps'),
            'fps_override': params.get('fps_override'),
            'exposure_time_s': params.get('exposure_time_s'),
        },
        'results': results,
    }
    with open(out, 'w') as f:
        json.dump(full, f, indent=2)
    return out, results


def save_log(video_path, params):
    """Write logs/<basename>_<timestamp>.json — the reloadable parameter log."""
    basename = video_basename(video_path)
    data_dir = data_dir_for(video_path, create=True)
    now = datetime.datetime.now()
    out = os.path.join(logs_dir(create=True),
                       f"{basename}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json")
    log = {
        'video_path': str(video_path),
        'video_basename': basename,
        'analysis_datetime': now.isoformat(),
        'analysis_parameters': {
            'start_time': params.get('start_time'),
            'end_time': params.get('end_time'),
            'x_start': params['x_start'],
            'x_stop': params['x_stop'],
            'y_start': params['y_start'],
            'y_stop': params['y_stop'],
            'threshold': params.get('threshold'),
            'min_area': params.get('min_area'),
            'max_area': params.get('max_area'),
            'fps': params.get('fps'),
            'fps_override': params.get('fps_override'),
            'exposure_time_s': params.get('exposure_time_s'),
            'pixels_per_um': params.get('pixels_per_um', DEFAULT_PIXELS_PER_UM),
        },
        'data_dir': data_dir,
        'droplet_data_path': os.path.join(data_dir, f"{basename}_droplet_data.npy"),
        'tracking_data_path': os.path.join(data_dir, f"{basename}_tracking_data.npy"),
        'results_json_path': os.path.join(data_dir, f"{basename}_test_results.json"),
    }
    with open(out, 'w') as f:
        json.dump(log, f, indent=2)
    return out


def list_logs(basename: str | None = None):
    """Newest-first list of saved parameter logs (optionally for one video).

    Log names are <basename>_<YYYY-MM-DD_HH-MM-SS>.json; the timestamp is
    matched exactly so video "run1" never picks up logs of "run1_dark".
    """
    d = logs_dir()
    if not os.path.isdir(d):
        return []
    pattern = (re.compile(re.escape(basename)
                          + r'_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.json')
               if basename else None)
    out = []
    for name in os.listdir(d):
        if not name.endswith('.json'):
            continue
        if pattern and not pattern.fullmatch(name):
            continue
        p = os.path.join(d, name)
        try:
            out.append({'name': name, 'path': p, 'mtime': os.path.getmtime(p)})
        except OSError:
            continue
    out.sort(key=lambda x: x['mtime'], reverse=True)
    return out


def load_log(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def json_safe(obj):
    """Recursively convert numpy scalars/arrays and NaN/inf for JSON.

    Lives here (not in vendor_api) so the light paths — detection, the track
    overlay — never drag in matplotlib/scipy.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def load_track_polylines(video_path, *, max_tracks: int = 200) -> dict:
    """Track polylines for the frame viewer overlay, straight from the npz.

    Pure numpy — deliberately not routed through the vendored analyze_tracks
    (which pulls in matplotlib) so toggling the overlay stays cheap.
    """
    basename = video_basename(video_path)
    npz_path = os.path.join(data_dir_for(video_path), f"{basename}_track_data.npz")
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(npz_path)
    z = np.load(npz_path, allow_pickle=False)
    obs = z["track_observations"]
    try:
        meta = json.loads(str(z["metadata"]))
    except (KeyError, ValueError):
        meta = {}
    # Track ids are assigned in ascending start-frame order upstream, so a
    # head slice would draw only the opening seconds of a long recording.
    # Keep the LONGEST tracks instead — they are the streamlines the overlay
    # exists to show — and tell the client what was left out.
    ids, counts = np.unique(obs["track_id"], return_counts=True)
    keep = ids[np.argsort(-counts, kind="stable")[:max_tracks]]
    out = []
    for tid in keep:
        rows = obs[obs["track_id"] == tid]
        pts = []
        for r in rows:
            pts.append([float(r["start_x"]), float(r["start_y"])])
            pts.append([float(r["end_x"]), float(r["end_y"])])
        out.append({"id": int(tid), "frames": [int(f) for f in rows["frame"]],
                    "points": pts})
    return {"tracks": out, "metadata": json_safe(meta),
            "n_tracks_total": int(len(ids)),
            "truncated": bool(len(ids) > len(keep))}
