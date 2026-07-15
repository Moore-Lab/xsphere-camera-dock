"""
Sidecar metadata for recorded videos.
======================================
The camera GUIs write a fixed nominal fps into the mp4 container (an acquisition
*setting*, not the true rate), so the recorded frame rate cannot be recovered from
the file itself. This module writes a small JSON sidecar next to each recording with
the TRUE measured rate (frames actually written / wall-clock duration) so downstream
analysis can length-calibrate correctly.

Deliberately dependency-free (stdlib only) and best-effort: ``write_sidecar`` never
raises — a metadata-write failure must never interrupt recording or streaming.
"""

import json
import os


def sidecar_path(video_path: str) -> str:
    """Return the sidecar path for a video: '<stem>.json' next to the video."""
    base, _ext = os.path.splitext(str(video_path))
    return base + ".json"


def build_metadata(video_path, n_frames, start_unix, stop_unix, *,
                   camera=None, width=None, height=None, nominal_fps=None,
                   extra=None) -> dict:
    """Assemble the sidecar dict. ``measured_fps`` = n_frames / duration (the true
    average rate); guarded against a zero/negative duration."""
    duration = None
    measured_fps = None
    try:
        duration = float(stop_unix) - float(start_unix)
        if duration > 0 and n_frames and n_frames > 1:
            # n_frames-1 intervals span the duration; both conventions are within
            # 1/n_frames of each other, use frames/duration for a stable estimate.
            measured_fps = float(n_frames) / duration
    except (TypeError, ValueError):
        pass
    meta = {
        "video_file": os.path.basename(str(video_path)),
        "n_frames": int(n_frames) if n_frames is not None else None,
        "start_unix": float(start_unix) if start_unix is not None else None,
        "stop_unix": float(stop_unix) if stop_unix is not None else None,
        "duration_s": duration,
        "measured_fps": measured_fps,
        "nominal_fps": float(nominal_fps) if nominal_fps is not None else None,
        "camera": camera,
        "width": int(width) if width is not None else None,
        "height": int(height) if height is not None else None,
        "schema": "mastqg.video_sidecar.v1",
    }
    if extra:
        meta.update(extra)
    return meta


def write_sidecar(video_path, n_frames, start_unix, stop_unix, **kwargs) -> str | None:
    """Best-effort write of the JSON sidecar. Returns the path on success, else None.

    NEVER raises: any failure (bad path, permissions, serialization) is swallowed and
    reported to stdout, so recording/streaming is never interrupted by metadata I/O.
    """
    try:
        if not video_path:
            return None
        meta = build_metadata(video_path, n_frames, start_unix, stop_unix, **kwargs)
        path = sidecar_path(video_path)
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"Video sidecar written: {path}  (measured_fps={meta['measured_fps']})")
        return path
    except Exception as exc:  # noqa: BLE001 — must never propagate
        print(f"WARNING: failed to write video sidecar for {video_path}: {exc}")
        return None
