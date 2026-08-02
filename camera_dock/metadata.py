"""Recording metadata sidecars — the truth about timing lives here, not in the
video container.

The container fps is a playback-speed setting; the true rate, per-frame host
timestamps, and the hardware frame clock are written next to each recording:

* ``<stem>_timestamps.npy`` — float64 host perf_counter per captured frame
* ``<stem>_hwclock.npz``    — ``hw_count`` (int64, -1 for missing) and
  ``hw_timestamp_ns`` (float64, NaN for missing — some units never report one)
* ``<stem>.json``           — summary: sw/hw measured fps, genuine drops
  (hardware frame-counter gaps), camera identity, geometry

Everything here is best-effort and NEVER raises — a metadata failure must not
kill a recording. Adapted from reference/video_metadata.py (dualcam_fast).
"""

from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np


def summarize_hw_clock(hw_counts: Sequence[Optional[int]],
                       hw_ts_ns: Sequence[Optional[float]]) -> dict:
    """Genuine-drop count from hw-counter gaps + leakage-free hw fps from device
    timestamps. Tolerates missing values (None) anywhere."""
    out = {"dropped_frames": None, "hw_measured_fps": None}
    counts = [c for c in hw_counts if c is not None]
    if len(counts) >= 2:
        gaps = sum(max(0, b - a - 1) for a, b in zip(counts, counts[1:]) if b >= a)
        out["dropped_frames"] = int(gaps)
    ts = [t for t in hw_ts_ns if t is not None]
    if len(ts) >= 2 and ts[-1] > ts[0]:
        out["hw_measured_fps"] = (len(ts) - 1) / ((ts[-1] - ts[0]) * 1e-9)
    return out


def write_sidecars(video_path: str, *, t_sw: Sequence[float],
                   hw_counts: Sequence[Optional[int]],
                   hw_ts_ns: Sequence[Optional[float]],
                   camera: Optional[dict] = None,
                   width: Optional[int] = None, height: Optional[int] = None,
                   nominal_fps: Optional[float] = None,
                   start_unix: Optional[float] = None,
                   stop_unix: Optional[float] = None,
                   extra: Optional[dict] = None) -> Optional[str]:
    """Write all three sidecars next to ``video_path``. Never raises; returns
    the JSON path on success, else None."""
    try:
        stem, _ = os.path.splitext(str(video_path))
        n = len(t_sw)

        try:
            np.save(stem + "_timestamps.npy", np.asarray(t_sw, dtype=np.float64))
        except Exception as exc:
            print(f"WARNING: timestamps sidecar failed: {exc}")

        try:
            counts = np.asarray([c if c is not None else -1 for c in hw_counts],
                                dtype=np.int64)
            ts = np.asarray([t if t is not None else np.nan for t in hw_ts_ns],
                            dtype=np.float64)
            np.savez(stem + "_hwclock.npz", sw_monotonic=np.asarray(t_sw, dtype=np.float64),
                     hw_count=counts, hw_timestamp_ns=ts)
        except Exception as exc:
            print(f"WARNING: hwclock sidecar failed: {exc}")

        sw_fps = None
        if n >= 2 and t_sw[-1] > t_sw[0]:
            sw_fps = (n - 1) / (t_sw[-1] - t_sw[0])
        meta = {
            "video_file": os.path.basename(str(video_path)),
            "n_frames": n,
            "start_unix": start_unix, "stop_unix": stop_unix,
            "duration_s": (stop_unix - start_unix)
                          if (start_unix is not None and stop_unix is not None) else None,
            "sw_measured_fps": sw_fps,
            "nominal_fps": nominal_fps,
            "camera": camera, "width": width, "height": height,
            "schema": "mastqg.video_sidecar.v2",
        }
        meta.update(summarize_hw_clock(hw_counts, hw_ts_ns))
        if extra:
            meta.update(extra)
        path = stem + ".json"
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=2)
        return path
    except Exception as exc:  # noqa: BLE001 — must never propagate
        print(f"WARNING: failed to write sidecars for {video_path}: {exc}")
        return None
