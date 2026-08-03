#!/usr/bin/env python3
"""
analyze_frames.py — frame-level analysis for dualcam_fast recordings.
====================================================================
Reads the recording bundles produced by ``reference/dualcam_fast.py`` (an
``<stem>.mp4`` plus its ``.json`` / ``_hwclock.npz`` / ``_timestamps.npy``
sidecars) and reports:

  * Timing   — true frame rate from the HARDWARE clock, inter-frame interval
               mean/jitter, and genuine dropped-frame count. The mp4 container
               only stores a nominal fps, so the real rate lives in the sidecars.
  * Intensity— per-frame mean/std/min/max/percentiles + saturated & dark pixel
               fractions (over the whole frame or an optional ROI).
  * Motion   — mean absolute difference between consecutive frames, a cheap
               proxy for how much the scene changed frame-to-frame.

Outputs a printed summary by default; optionally a per-frame CSV (``--csv``),
PNG plots (``--plot``, needs matplotlib), and extracted PNG frames
(``--extract``).

``--view`` opens an interactive PyQt5 window instead: pick a recording, scrub or
step through it (±1 and ±100 frame buttons, plus ←/→ and PgUp/PgDn), play/pause
at the true recorded rate, and zoom in/out. The 📏 Measure tool lets you click two
points to read the pixel distance between them; measurements stay on screen so you
can lay down several and compare (uncalibrated — pixels only).

Only numpy + opencv are required (same as the recorder); matplotlib is used
lazily and only for ``--plot``.

Examples
--------
    # One recording, printed summary
    python analysis/analyze_frames.py recordings/camera_1_20260715_120000.mp4

    # Whole folder, write a CSV + plots next to each video
    python analysis/analyze_frames.py recordings/ --csv --plot

    # Restrict stats to an ROI and dump the 5 brightest/darkest frames
    python analysis/analyze_frames.py rec.mp4 --roi 200,150,300,300 --extract bright,dark
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

# Directories/extensions we treat as "a video".
VIDEO_EXTS = (".mp4", ".avi")

# The repo's recordings/ folder, resolved from THIS file's location (repo root is
# the parent of analysis/). Anchoring to __file__ — not the process cwd — means
# the default works no matter where you launch from (e.g. an IDE "Run" button that
# runs with the analysis/ folder as the working directory).
DEFAULT_RECORDINGS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "recordings"))


# ── Recording discovery & sidecar loading ──────────────────────────────────

def find_videos(path):
    """Return a sorted list of video files for a file or directory *path*."""
    if os.path.isdir(path):
        out = []
        for ext in VIDEO_EXTS:
            out.extend(glob.glob(os.path.join(path, "*" + ext)))
        return sorted(out)
    if os.path.isfile(path):
        return [path]
    # allow a glob pattern
    return sorted(glob.glob(path))


def load_sidecars(video_path):
    """Load the JSON / npz / npy sidecars for *video_path* (missing ones -> None)."""
    stem = os.path.splitext(video_path)[0]
    meta = None
    hwclock = None
    sw_ts = None

    json_path = stem + ".json"
    if os.path.isfile(json_path):
        try:
            with open(json_path) as fh:
                meta = json.load(fh)
        except (OSError, ValueError) as e:
            print(f"  ! could not read {os.path.basename(json_path)}: {e}")

    npz_path = stem + "_hwclock.npz"
    if os.path.isfile(npz_path):
        try:
            hwclock = dict(np.load(npz_path))
        except (OSError, ValueError) as e:
            print(f"  ! could not read {os.path.basename(npz_path)}: {e}")

    npy_path = stem + "_timestamps.npy"
    if os.path.isfile(npy_path):
        try:
            sw_ts = np.load(npy_path)
        except (OSError, ValueError) as e:
            print(f"  ! could not read {os.path.basename(npy_path)}: {e}")

    return meta, hwclock, sw_ts


# ── Timing analysis (from the hardware clock sidecar) ───────────────────────

def analyze_timing(hwclock, sw_ts, meta):
    """Derive true fps, inter-frame jitter and genuine drops from the sidecars.

    Prefers the leakage-free HARDWARE timestamps (``hw_timestamp_ns``); falls
    back to the software monotonic clock, then to the JSON ``measured_fps``.
    Returns a dict of scalars (values may be None when a source is absent).
    """
    out = {
        "source": None, "fps": None, "interval_ms_mean": None,
        "interval_ms_std": None, "interval_ms_min": None, "interval_ms_max": None,
        "jitter_pct": None, "dropped": None, "ring_peak": None, "span_s": None,
    }

    # Genuine drops + ring backpressure come from the hw frame counter / ring trace.
    if hwclock is not None:
        fc = np.asarray(hwclock.get("hw_frame_count", []), dtype=np.float64)
        fc = fc[np.isfinite(fc)]
        if fc.size >= 2:
            diffs = np.diff(fc)
            out["dropped"] = int(np.sum(diffs[diffs > 1] - 1))
        ring = np.asarray(hwclock.get("ring_in_use", []), dtype=np.float64)
        ring = ring[np.isfinite(ring)]
        if ring.size:
            out["ring_peak"] = int(np.nanmax(ring))

    # Pick the best available per-frame time base (seconds).
    times_s = None
    if hwclock is not None:
        ts = np.asarray(hwclock.get("hw_timestamp_ns", []), dtype=np.float64)
        ts = ts[np.isfinite(ts)]
        ts = ts[ts >= 0]  # -1/negative is the "unsupported" sentinel
        if ts.size >= 2:
            times_s = ts / 1e9
            out["source"] = "hardware clock"
        else:
            sw = np.asarray(hwclock.get("sw_monotonic", []), dtype=np.float64)
            sw = sw[np.isfinite(sw)]
            if sw.size >= 2:
                times_s = sw
                out["source"] = "software clock (hwclock.npz)"
    if times_s is None and sw_ts is not None:
        sw = np.asarray(sw_ts, dtype=np.float64)
        sw = sw[np.isfinite(sw)]
        if sw.size >= 2:
            times_s = sw
            out["source"] = "software clock (timestamps.npy)"

    if times_s is not None:
        intervals_ms = np.diff(np.sort(times_s)) * 1e3
        intervals_ms = intervals_ms[intervals_ms >= 0]
        if intervals_ms.size:
            span = float(times_s.max() - times_s.min())
            out["span_s"] = span
            out["fps"] = (times_s.size - 1) / span if span > 0 else None
            out["interval_ms_mean"] = float(np.mean(intervals_ms))
            out["interval_ms_std"] = float(np.std(intervals_ms))
            out["interval_ms_min"] = float(np.min(intervals_ms))
            out["interval_ms_max"] = float(np.max(intervals_ms))
            if out["interval_ms_mean"]:
                out["jitter_pct"] = 100.0 * out["interval_ms_std"] / out["interval_ms_mean"]

    # Last-resort fps from the JSON sidecar.
    if out["fps"] is None and meta:
        out["fps"] = meta.get("hw_measured_fps") or meta.get("measured_fps")
        if out["fps"] is not None and out["source"] is None:
            out["source"] = "json sidecar"

    return out


# ── Per-frame image statistics ─────────────────────────────────────────────

def _to_gray(frame):
    """Recorded frames are grayscale replicated to BGR; collapse back to 1 channel."""
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def iter_frames(video_path, roi=None, step=1):
    """Yield (index, gray_roi_frame) for each decoded frame.

    *roi* is an (x, y, w, h) tuple or None (whole frame). *step* decimates the
    stream (every Nth frame) for a quick look at long recordings.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                gray = _to_gray(frame)
                if roi is not None:
                    x, y, w, h = roi
                    gray = gray[y:y + h, x:x + w]
                yield idx, gray
            idx += 1
    finally:
        cap.release()


def frame_stats(gray, prev_gray):
    """Return the per-frame statistics dict for one grayscale frame."""
    flat = gray.reshape(-1)
    stats = {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": int(flat.min()),
        "max": int(flat.max()),
        "p1": float(np.percentile(flat, 1)),
        "p99": float(np.percentile(flat, 99)),
        "sat_frac": float(np.count_nonzero(flat == 255)) / flat.size,
        "dark_frac": float(np.count_nonzero(flat == 0)) / flat.size,
        "motion": None,
    }
    if prev_gray is not None and prev_gray.shape == gray.shape:
        stats["motion"] = float(np.mean(np.abs(gray.astype(np.int16)
                                                - prev_gray.astype(np.int16))))
    return stats


def analyze_frames(video_path, roi=None, step=1):
    """Decode *video_path* and return (list of per-frame stat dicts, dimensions)."""
    rows = []
    prev = None
    dims = None
    for idx, gray in iter_frames(video_path, roi=roi, step=step):
        if dims is None:
            dims = (gray.shape[1], gray.shape[0])  # (w, h) of the analyzed region
        s = frame_stats(gray, prev)
        s["frame"] = idx
        rows.append(s)
        prev = gray
    return rows, dims


# ── Aggregation / reporting ────────────────────────────────────────────────

def _col(rows, key):
    return np.array([r[key] for r in rows if r[key] is not None], dtype=np.float64)


def summarize(rows):
    """Roll per-frame rows up into a small dict of aggregate intensity/motion stats."""
    if not rows:
        return {}
    mean = _col(rows, "mean")
    motion = _col(rows, "motion")
    sat = _col(rows, "sat_frac")
    return {
        "n_decoded": len(rows),
        "intensity_mean": float(mean.mean()),
        "intensity_range": (float(mean.min()), float(mean.max())),
        "brightest_frame": int(rows[int(np.argmax([r["mean"] for r in rows]))]["frame"]),
        "darkest_frame": int(rows[int(np.argmin([r["mean"] for r in rows]))]["frame"]),
        "max_sat_frac": float(sat.max()) if sat.size else 0.0,
        "motion_mean": float(motion.mean()) if motion.size else None,
        "motion_peak": float(motion.max()) if motion.size else None,
        "motion_peak_frame": (int(rows[1 + int(np.argmax([r["motion"]
                              for r in rows[1:]]))]["frame"]) if len(rows) > 1 else None),
    }


def _fmt(v, spec="{:.3f}"):
    return "n/a" if v is None else spec.format(v)


def print_report(video_path, meta, timing, summary, roi, step):
    name = os.path.basename(video_path)
    print("=" * 72)
    print(f"{name}")
    print("-" * 72)

    if meta:
        cam = meta.get("camera")
        w, h = meta.get("width"), meta.get("height")
        print(f"  camera      : {cam}   sensor/ROI in sidecar: {w}x{h}")
        print(f"  frames (rec): {meta.get('n_frames')}   "
              f"nominal_fps={_fmt(meta.get('nominal_fps'), '{:.1f}')}")
    if roi:
        print(f"  analysis ROI: x={roi[0]} y={roi[1]} w={roi[2]} h={roi[3]}"
              + ("" if step == 1 else f"   (every {step}th frame)"))
    elif step != 1:
        print(f"  decimation  : every {step}th frame")

    print("  timing:")
    print(f"    source        : {timing['source']}")
    print(f"    true fps      : {_fmt(timing['fps'], '{:.2f}')}"
          + (f"   over {_fmt(timing['span_s'], '{:.2f}')}s" if timing['span_s'] else ""))
    print(f"    frame interval: {_fmt(timing['interval_ms_mean'])} ms "
          f"(std {_fmt(timing['interval_ms_std'])} ms, "
          f"jitter {_fmt(timing['jitter_pct'], '{:.1f}')}%)")
    print(f"    interval range: {_fmt(timing['interval_ms_min'])} .. "
          f"{_fmt(timing['interval_ms_max'])} ms")
    print(f"    dropped frames: {timing['dropped']}    ring peak: {timing['ring_peak']}")

    if summary:
        lo, hi = summary["intensity_range"]
        print("  intensity (0-255):")
        print(f"    mean          : {summary['intensity_mean']:.2f}  "
              f"(per-frame mean ranges {lo:.1f}..{hi:.1f})")
        print(f"    brightest/dark: frame {summary['brightest_frame']} / "
              f"{summary['darkest_frame']}")
        print(f"    max saturated : {100 * summary['max_sat_frac']:.3f}% of pixels")
        print("  motion (mean abs consecutive-frame diff):")
        print(f"    mean / peak   : {_fmt(summary['motion_mean'])} / "
              f"{_fmt(summary['motion_peak'])}"
              + (f"  at frame {summary['motion_peak_frame']}"
                 if summary['motion_peak_frame'] is not None else ""))
    print()


# ── Optional outputs: CSV, plots, extracted frames ─────────────────────────

def write_csv(rows, out_path):
    import csv
    cols = ["frame", "mean", "std", "min", "max", "p1", "p99",
            "sat_frac", "dark_frac", "motion"]
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    print(f"  -> wrote {out_path}")


def write_plots(rows, timing, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ! matplotlib not installed; skipping --plot "
              "(pip install matplotlib)")
        return

    frames = [r["frame"] for r in rows]
    means = [r["mean"] for r in rows]
    motion = [r["motion"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(frames, means, lw=0.8)
    axes[0].set_ylabel("mean intensity")
    axes[0].set_title(os.path.basename(out_path).replace("_analysis.png", ""))
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames[1:], motion[1:], lw=0.8, color="tab:red")
    axes[1].set_ylabel("frame-to-frame motion")
    axes[1].set_xlabel("frame index")
    axes[1].grid(True, alpha=0.3)

    fps = timing.get("fps")
    if fps:
        fig.suptitle(f"true fps={fps:.2f}  drops={timing.get('dropped')}",
                     x=0.99, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"  -> wrote {out_path}")


def extract_frames(video_path, rows, which, out_dir, n=5, roi=None):
    """Save selected frames as PNGs. *which* is a comma list of:
    first, last, bright, dark, motion, or explicit integer indices."""
    if not rows:
        return
    targets = []
    order_mean = sorted(rows, key=lambda r: r["mean"])
    order_motion = sorted([r for r in rows if r["motion"] is not None],
                          key=lambda r: r["motion"])
    for token in which.split(","):
        token = token.strip().lower()
        if token == "first":
            targets.append(rows[0]["frame"])
        elif token == "last":
            targets.append(rows[-1]["frame"])
        elif token == "bright":
            targets += [r["frame"] for r in order_mean[-n:]]
        elif token == "dark":
            targets += [r["frame"] for r in order_mean[:n]]
        elif token == "motion":
            targets += [r["frame"] for r in order_motion[-n:]]
        elif token.isdigit():
            targets.append(int(token))
    targets = sorted(set(targets))
    if not targets:
        return

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    saved = 0
    try:
        for idx in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            if roi is not None:
                x, y, w, h = roi
                frame = frame[y:y + h, x:x + w]
            out = os.path.join(out_dir, f"{stem}_frame{idx:06d}.png")
            cv2.imwrite(out, frame)
            saved += 1
    finally:
        cap.release()
    print(f"  -> extracted {saved} frame(s) to {out_dir}")


# ── Interactive frame viewer (PyQt5) ───────────────────────────────────────
# Launched with --view. Lets you open a recording, scrub/step through frames
# (±1 and ±100), play/pause at the true recorded rate, and zoom in/out. PyQt5 is
# imported lazily inside launch_viewer so the batch CLI has no Qt dependency.

def launch_viewer(initial_path=None, start_dir="recordings"):
    """Open the interactive frame viewer. Returns the process exit code."""
    from PyQt5 import QtCore, QtGui, QtWidgets

    class FrameViewer(QtWidgets.QMainWindow):
        def __init__(self, path=None):
            super().__init__()
            self.setWindowTitle("Frame Viewer — dualcam_fast recordings")
            self.resize(1000, 780)

            self.cap = None
            self.video_path = None
            self.total = 0
            self.cur_idx = -1
            self.cur_frame = None       # current frame as BGR ndarray
            self._prev_shown = None     # last displayed frame (for Δ readout)
            self.zoom = 1.0
            self.fit = True             # auto-fit until the user zooms manually
            self.playback_fps = 30.0

            # Measuring tool: click two points to record a pixel-distance line.
            # Measurements persist (drawn every render) so several can be compared;
            # all points are stored in IMAGE coordinates so they track zoom/scroll.
            self.measure_mode = False
            self.measurements = []      # list of ((x1,y1),(x2,y2)) in image coords
            self.pending_point = None   # first clicked point awaiting the second
            self.cursor_img = None      # live cursor pos (image coords) for preview

            self.play_timer = QtCore.QTimer(self)
            self.play_timer.timeout.connect(self._on_tick)

            self._build_ui()
            if path:
                self._load(path)

        # ── UI construction ────────────────────────────────────────────
        def _build_ui(self):
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            root = QtWidgets.QVBoxLayout(central)

            # Top bar: open + filename
            top = QtWidgets.QHBoxLayout()
            open_btn = QtWidgets.QPushButton("Open recording…")
            open_btn.clicked.connect(self._choose_file)
            top.addWidget(open_btn)
            self.file_label = QtWidgets.QLabel("No file loaded")
            self.file_label.setStyleSheet("color: #666;")
            top.addWidget(self.file_label, 1)
            root.addLayout(top)

            # Image area (scrollable so you can pan when zoomed in)
            self.scroll = QtWidgets.QScrollArea()
            self.scroll.setWidgetResizable(False)
            self.scroll.setAlignment(QtCore.Qt.AlignCenter)
            self.scroll.setStyleSheet("background:#202020;")
            self.image_label = QtWidgets.QLabel("Open a recording to begin")
            self.image_label.setAlignment(QtCore.Qt.AlignCenter)
            self.image_label.setStyleSheet("color:#aaa;")
            self.image_label.setMouseTracking(True)      # for the live measure preview
            self.image_label.installEventFilter(self)    # capture clicks for measuring
            self.scroll.setWidget(self.image_label)
            root.addWidget(self.scroll, 1)

            # Scrubber
            slider_row = QtWidgets.QHBoxLayout()
            self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.slider.setEnabled(False)
            self.slider.valueChanged.connect(self._on_slider)
            slider_row.addWidget(self.slider, 1)
            self.frame_spin = QtWidgets.QSpinBox()
            self.frame_spin.setEnabled(False)
            self.frame_spin.valueChanged.connect(self._on_spin)
            slider_row.addWidget(self.frame_spin)
            root.addLayout(slider_row)

            # Transport buttons
            btn_row = QtWidgets.QHBoxLayout()
            self.b_back100 = QtWidgets.QPushButton("◀◀ −100")
            self.b_back1 = QtWidgets.QPushButton("◀ −1")
            self.b_play = QtWidgets.QPushButton("▶ Play")
            self.b_fwd1 = QtWidgets.QPushButton("+1 ▶")
            self.b_fwd100 = QtWidgets.QPushButton("+100 ▶▶")
            self.b_back100.clicked.connect(lambda: self.step(-100))
            self.b_back1.clicked.connect(lambda: self.step(-1))
            self.b_play.clicked.connect(self.toggle_play)
            self.b_fwd1.clicked.connect(lambda: self.step(1))
            self.b_fwd100.clicked.connect(lambda: self.step(100))
            for b in (self.b_back100, self.b_back1, self.b_play,
                      self.b_fwd1, self.b_fwd100):
                b.setEnabled(False)
                btn_row.addWidget(b)
            btn_row.addStretch(1)

            # Zoom controls
            btn_row.addWidget(QtWidgets.QLabel("Zoom:"))
            self.b_zoom_out = QtWidgets.QPushButton("−")
            self.b_zoom_in = QtWidgets.QPushButton("+")
            self.b_fit = QtWidgets.QPushButton("Fit")
            self.b_1to1 = QtWidgets.QPushButton("1:1")
            self.b_zoom_out.clicked.connect(lambda: self.set_zoom(self.zoom / 1.25))
            self.b_zoom_in.clicked.connect(lambda: self.set_zoom(self.zoom * 1.25))
            self.b_fit.clicked.connect(self.zoom_fit)
            self.b_1to1.clicked.connect(lambda: self.set_zoom(1.0))
            for b in (self.b_zoom_out, self.b_zoom_in, self.b_fit, self.b_1to1):
                b.setEnabled(False)
                btn_row.addWidget(b)
            self.zoom_label = QtWidgets.QLabel("—")
            btn_row.addWidget(self.zoom_label)
            root.addLayout(btn_row)

            # Measure tool row: toggle, undo last, clear all, and a live readout.
            measure_row = QtWidgets.QHBoxLayout()
            self.b_measure = QtWidgets.QPushButton("📏 Measure: off")
            self.b_measure.setCheckable(True)
            self.b_measure.toggled.connect(self._set_measure_mode)
            self.b_undo_measure = QtWidgets.QPushButton("Undo")
            self.b_undo_measure.clicked.connect(self._undo_measurement)
            self.b_clear_measure = QtWidgets.QPushButton("Clear")
            self.b_clear_measure.clicked.connect(self._clear_measurements)
            for b in (self.b_measure, self.b_undo_measure, self.b_clear_measure):
                b.setEnabled(False)
                measure_row.addWidget(b)
            self.measure_label = QtWidgets.QLabel("")
            self.measure_label.setStyleSheet("font-family: monospace; color:#e0e0e0;")
            measure_row.addWidget(self.measure_label, 1)
            root.addLayout(measure_row)

            # Info line
            self.info_label = QtWidgets.QLabel("")
            self.info_label.setStyleSheet("font-family: monospace;")
            root.addWidget(self.info_label)

            self.statusBar().showMessage(
                "Space: play/pause   ← → : ±1   PgUp/PgDn: ±100   "
                "Ctrl++/Ctrl+-: zoom   F: fit   M: measure")

            # Keyboard shortcuts
            def sc(seq, fn):
                s = QtWidgets.QShortcut(QtGui.QKeySequence(seq), self)
                s.activated.connect(fn)
            sc(QtCore.Qt.Key_Space, self.toggle_play)
            sc(QtCore.Qt.Key_Left, lambda: self.step(-1))
            sc(QtCore.Qt.Key_Right, lambda: self.step(1))
            sc(QtCore.Qt.Key_PageDown, lambda: self.step(-100))
            sc(QtCore.Qt.Key_PageUp, lambda: self.step(100))
            sc(QtCore.Qt.Key_Home, lambda: self.goto(0))
            sc(QtCore.Qt.Key_End, lambda: self.goto(self.total - 1))
            sc("Ctrl++", lambda: self.set_zoom(self.zoom * 1.25))
            sc("Ctrl+=", lambda: self.set_zoom(self.zoom * 1.25))
            sc("Ctrl+-", lambda: self.set_zoom(self.zoom / 1.25))
            sc("F", self.zoom_fit)
            sc("M", self.b_measure.toggle)

        # ── File loading ───────────────────────────────────────────────
        def _choose_file(self):
            start = start_dir if os.path.isdir(start_dir) else os.getcwd()
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Open recording", start,
                "Videos (*.mp4 *.avi);;All files (*)")
            if path:
                self._load(path)

        def _load(self, path):
            self.pause()
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                QtWidgets.QMessageBox.critical(
                    self, "Open failed", f"OpenCV could not open:\n{path}")
                return
            if self.cap is not None:
                self.cap.release()
            self.cap = cap
            self.video_path = path

            # Frame count: trust the sidecar first, else the container.
            meta, hwclock, sw_ts = load_sidecars(path)
            timing = analyze_timing(hwclock, sw_ts, meta)
            n = (meta or {}).get("n_frames")
            if not n:
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.total = max(1, int(n))
            self.playback_fps = float(
                (timing.get("fps") or (meta or {}).get("nominal_fps") or 30.0))
            if self.playback_fps <= 0:
                self.playback_fps = 30.0

            self.file_label.setText(
                f"{os.path.basename(path)}   —   {self.total} frames "
                f"@ {self.playback_fps:.1f} fps  "
                f"(drops={timing.get('dropped')})")

            for w in (self.slider, self.frame_spin, self.b_back100, self.b_back1,
                      self.b_play, self.b_fwd1, self.b_fwd100, self.b_zoom_out,
                      self.b_zoom_in, self.b_fit, self.b_1to1,
                      self.b_measure, self.b_undo_measure, self.b_clear_measure):
                w.setEnabled(True)
            self._clear_measurements()   # a new video may differ in size
            self.slider.blockSignals(True)
            self.slider.setRange(0, self.total - 1)
            self.slider.blockSignals(False)
            self.frame_spin.blockSignals(True)
            self.frame_spin.setRange(0, self.total - 1)
            self.frame_spin.blockSignals(False)

            self.cur_idx = -1
            self._prev_shown = None
            self.fit = True
            self.goto(0)

        # ── Frame access ───────────────────────────────────────────────
        def _read_at(self, idx):
            """Decode and return the BGR frame at *idx*, or None."""
            if self.cap is None:
                return None
            idx = max(0, min(idx, self.total - 1))
            # Sequential forward reads avoid a costly re-seek every step.
            if idx != self.cur_idx + 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = self.cap.read()
            if not ok:
                return None
            return frame

        def goto(self, idx):
            idx = max(0, min(int(idx), self.total - 1))
            frame = self._read_at(idx)
            if frame is None:
                return
            self._prev_shown = self.cur_frame
            self.cur_frame = frame
            self.cur_idx = idx
            if self.fit:
                self._compute_fit_zoom()
            self._render()
            self._sync_controls()
            self._update_info()

        def step(self, delta):
            self.goto(self.cur_idx + delta)

        def _on_slider(self, val):
            if val != self.cur_idx:
                self.goto(val)

        def _on_spin(self, val):
            if val != self.cur_idx:
                self.goto(val)

        def _sync_controls(self):
            for w in (self.slider, self.frame_spin):
                w.blockSignals(True)
            self.slider.setValue(self.cur_idx)
            self.frame_spin.setValue(self.cur_idx)
            for w in (self.slider, self.frame_spin):
                w.blockSignals(False)

        # ── Playback ───────────────────────────────────────────────────
        def toggle_play(self):
            if self.cap is None:
                return
            if self.play_timer.isActive():
                self.pause()
            else:
                self.play()

        def play(self):
            if self.cur_idx >= self.total - 1:
                self.goto(0)
            self.play_timer.start(max(1, int(1000.0 / self.playback_fps)))
            self.b_play.setText("❚❚ Pause")

        def pause(self):
            self.play_timer.stop()
            self.b_play.setText("▶ Play")

        def _on_tick(self):
            if self.cur_idx >= self.total - 1:
                self.pause()
                return
            self.step(1)

        # ── Zoom / rendering ───────────────────────────────────────────
        def set_zoom(self, factor):
            self.fit = False
            self.zoom = max(0.05, min(float(factor), 32.0))
            self._render()
            self._update_zoom_label()

        def zoom_fit(self):
            self.fit = True
            self._compute_fit_zoom()
            self._render()
            self._update_zoom_label()

        def _compute_fit_zoom(self):
            if self.cur_frame is None:
                return
            h, w = self.cur_frame.shape[:2]
            vp = self.scroll.viewport().size()
            if w and h:
                self.zoom = max(0.05, min(vp.width() / w, vp.height() / h))

        def _update_zoom_label(self):
            self.zoom_label.setText(f"{self.zoom * 100:.0f}%"
                                    + ("  (fit)" if self.fit else ""))

        def _render(self):
            if self.cur_frame is None:
                return
            frame = self.cur_frame
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            qimg = QtGui.QImage(rgb.data, w, h, 3 * w,
                                QtGui.QImage.Format_RGB888)
            pix = QtGui.QPixmap.fromImage(qimg)
            zw, zh = max(1, int(w * self.zoom)), max(1, int(h * self.zoom))
            # Nearest-neighbour when zoomed in so pixels stay crisp.
            mode = (QtCore.Qt.FastTransformation if self.zoom >= 1.0
                    else QtCore.Qt.SmoothTransformation)
            scaled = pix.scaled(zw, zh, QtCore.Qt.KeepAspectRatio, mode)
            if self.measurements or self.pending_point is not None:
                self._draw_measurements(scaled, w, h)
            self.image_label.setPixmap(scaled)
            self.image_label.resize(scaled.size())
            self._update_zoom_label()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            if self.fit:
                self._compute_fit_zoom()
                self._render()

        # ── Info readout ───────────────────────────────────────────────
        def _update_info(self):
            if self.cur_frame is None:
                return
            gray = _to_gray(self.cur_frame)
            t = self.cur_idx / self.playback_fps if self.playback_fps else 0.0
            motion = ""
            if (self._prev_shown is not None
                    and self._prev_shown.shape == self.cur_frame.shape):
                d = float(np.mean(np.abs(_to_gray(self.cur_frame).astype(np.int16)
                                         - _to_gray(self._prev_shown).astype(np.int16))))
                motion = f"  Δview={d:5.2f}"
            self.info_label.setText(
                f"frame {self.cur_idx:>6d} / {self.total - 1}   "
                f"t≈{t:7.3f}s   mean={gray.mean():6.2f}   "
                f"min={int(gray.min()):3d} max={int(gray.max()):3d}"
                f"{motion}")

        # ── Measuring tool ─────────────────────────────────────────────
        @staticmethod
        def _dist(p1, p2):
            return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

        def _measure_color(self, i):
            palette = [(255, 80, 80), (80, 200, 255), (120, 255, 120),
                       (255, 210, 70), (220, 120, 255), (255, 150, 60)]
            r, g, b = palette[i % len(palette)]
            return QtGui.QColor(r, g, b)

        def _set_measure_mode(self, on):
            self.measure_mode = bool(on)
            self.b_measure.setText("📏 Measure: ON — click 2 points" if on
                                   else "📏 Measure: off")
            if not on:                       # drop a half-finished measurement
                self.pending_point = None
                self.cursor_img = None
                self._render()
            self.image_label.setCursor(
                QtCore.Qt.CrossCursor if on else QtCore.Qt.ArrowCursor)

        def _label_to_image(self, lx, ly):
            """Map a click in image_label coords to (x, y) image-pixel coords."""
            pm = self.image_label.pixmap()
            if pm is None or self.cur_frame is None:
                return None
            h, w = self.cur_frame.shape[:2]
            pmw, pmh = pm.width(), pm.height()
            if pmw <= 0 or pmh <= 0:
                return None
            offx = max(0, (self.image_label.width() - pmw) / 2)
            offy = max(0, (self.image_label.height() - pmh) / 2)
            x = (lx - offx) * w / pmw
            y = (ly - offy) * h / pmh
            if x < 0 or y < 0 or x > w - 1 or y > h - 1:
                return None
            return (x, y)

        def eventFilter(self, obj, event):
            if (obj is self.image_label and self.measure_mode
                    and self.cur_frame is not None):
                et = event.type()
                if (et == QtCore.QEvent.MouseButtonPress
                        and event.button() == QtCore.Qt.LeftButton):
                    pt = self._label_to_image(event.x(), event.y())
                    if pt is not None:
                        self._add_measure_point(pt)
                    return True
                if et == QtCore.QEvent.MouseMove and self.pending_point is not None:
                    pt = self._label_to_image(event.x(), event.y())
                    if pt is not None:
                        self.cursor_img = pt
                        self._render()
            return super().eventFilter(obj, event)

        def _add_measure_point(self, pt):
            if self.pending_point is None:
                self.pending_point = pt
                self.cursor_img = pt
            else:
                self.measurements.append((self.pending_point, pt))
                self.pending_point = None
                self.cursor_img = None
            self._render()
            self._update_measure_readout()

        def _undo_measurement(self):
            if self.pending_point is not None:
                self.pending_point = None
                self.cursor_img = None
            elif self.measurements:
                self.measurements.pop()
            self._render()
            self._update_measure_readout()

        def _clear_measurements(self):
            self.measurements = []
            self.pending_point = None
            self.cursor_img = None
            if self.cur_frame is not None:
                self._render()
            self._update_measure_readout()

        def _update_measure_readout(self):
            parts = [f"M{i + 1}={self._dist(a, b):.1f}px"
                     for i, (a, b) in enumerate(self.measurements)]
            if self.pending_point is not None:
                parts.append(f"M{len(self.measurements) + 1}: pick 2nd point…")
            if len(self.measurements) >= 2:
                ds = [self._dist(a, b) for a, b in self.measurements]
                if ds[-2] > 0:
                    parts.append(f"|  M{len(ds)}/M{len(ds) - 1}={ds[-1] / ds[-2]:.2f}×")
            self.measure_label.setText("   ".join(parts))

        def _draw_measurements(self, pixmap, img_w, img_h):
            sx = pixmap.width() / img_w
            sy = pixmap.height() / img_h

            def L(pt):
                return QtCore.QPointF(pt[0] * sx, pt[1] * sy)

            painter = QtGui.QPainter(pixmap)
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            font = painter.font()
            font.setPointSize(10)
            font.setBold(True)
            painter.setFont(font)

            for i, (p1, p2) in enumerate(self.measurements):
                self._draw_segment(painter, L(p1), L(p2), self._measure_color(i),
                                   f"M{i + 1}: {self._dist(p1, p2):.1f}px", solid=True)
            if self.pending_point is not None:
                col = self._measure_color(len(self.measurements))
                a = L(self.pending_point)
                painter.setPen(QtGui.QPen(col, 1))
                painter.setBrush(col)
                painter.drawEllipse(a, 3, 3)
                if self.cursor_img is not None:
                    self._draw_segment(
                        painter, a, L(self.cursor_img), col,
                        f"{self._dist(self.pending_point, self.cursor_img):.1f}px",
                        solid=False)
            painter.end()

        def _draw_segment(self, painter, a, b, col, text, solid):
            style = QtCore.Qt.SolidLine if solid else QtCore.Qt.DashLine
            painter.setPen(QtGui.QPen(col, 2, style))
            painter.drawLine(a, b)
            painter.setBrush(col)
            painter.setPen(QtGui.QPen(col, 1))
            for pt in (a, b):
                painter.drawEllipse(pt, 3, 3)
            mid = QtCore.QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2)
            self._draw_text(painter, mid, text, col)

        def _draw_text(self, painter, pt, text, col):
            fm = painter.fontMetrics()
            rect = fm.boundingRect(text).adjusted(-4, -2, 4, 2)
            rect.moveCenter(QtCore.QPoint(int(pt.x()), int(pt.y()) - 12))
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(0, 0, 0, 180))
            painter.drawRoundedRect(rect, 3, 3)
            painter.setPen(QtGui.QPen(col))
            painter.drawText(rect, QtCore.Qt.AlignCenter, text)

        def closeEvent(self, event):
            self.pause()
            if self.cap is not None:
                self.cap.release()
            super().closeEvent(event)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    win = FrameViewer(initial_path)
    win.show()
    return app.exec_()


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_roi(text):
    if not text:
        return None
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("--roi must be x,y,w,h")
    return tuple(parts)


def build_parser():
    p = argparse.ArgumentParser(
        description="Analyze frames of dualcam_fast recordings.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", default=DEFAULT_RECORDINGS_DIR,
                   help="a video file, a directory, or a glob "
                        "(default: the repo's recordings/ folder)")
    p.add_argument("--roi", type=parse_roi, metavar="X,Y,W,H",
                   help="restrict intensity/motion stats to this region")
    p.add_argument("--step", type=int, default=1, metavar="N",
                   help="analyze every Nth frame (quick look at long clips)")
    p.add_argument("--csv", action="store_true",
                   help="write <stem>_frames.csv of per-frame stats")
    p.add_argument("--plot", action="store_true",
                   help="write <stem>_analysis.png (needs matplotlib)")
    p.add_argument("--extract", metavar="WHICH",
                   help="save PNG frames: comma list of "
                        "first,last,bright,dark,motion or frame indices")
    p.add_argument("--extract-n", type=int, default=5, metavar="N",
                   help="how many frames for bright/dark/motion (default 5)")
    p.add_argument("--outdir", metavar="DIR",
                   help="output directory for csv/plot/frames "
                        "(default: next to each video)")
    p.add_argument("--view", action="store_true",
                   help="open the interactive frame viewer (step/scrub/zoom) "
                        "instead of printing a batch report. Also the default when "
                        "the script is run with no arguments.")
    return p


def main(argv=None):
    raw = (sys.argv[1:] if argv is None else list(argv))
    args = build_parser().parse_args(argv)

    # Interactive viewer. Opened by --view, OR by a bare run with no arguments
    # (e.g. the IDE "Run" button) — the common "I just want to look at my
    # recordings frame by frame" case. Any explicit path/flag uses the batch CLI
    # below. When pointed at a folder, the newest recording is auto-loaded.
    if args.view or not raw:
        if os.path.isfile(args.path):
            initial = args.path
            start_dir = os.path.dirname(args.path) or DEFAULT_RECORDINGS_DIR
        else:
            start_dir = args.path if os.path.isdir(args.path) else DEFAULT_RECORDINGS_DIR
            vids = find_videos(start_dir)
            initial = vids[-1] if vids else None   # newest (names are timestamped)
        return launch_viewer(initial_path=initial, start_dir=start_dir)

    videos = find_videos(args.path)
    if not videos:
        # Printed to stdout (not just stderr) so it is visible in an IDE run.
        print(f"No videos found in: {args.path}")
        print("Record something with reference/dualcam_fast.py first (it now saves "
              "into recordings/ by default), then run this again.")
        print("Or point it at a file/folder:  "
              "python analysis/analyze_frames.py <path-to-video.mp4>")
        print("Interactive viewer:            "
              "python analysis/analyze_frames.py --view")
        return 1

    for video in videos:
        meta, hwclock, sw_ts = load_sidecars(video)
        timing = analyze_timing(hwclock, sw_ts, meta)
        try:
            rows, _dims = analyze_frames(video, roi=args.roi, step=args.step)
        except RuntimeError as e:
            print(f"! {os.path.basename(video)}: {e}", file=sys.stderr)
            continue
        summary = summarize(rows)
        print_report(video, meta, timing, summary, args.roi, args.step)

        out_dir = args.outdir or os.path.dirname(os.path.abspath(video))
        stem = os.path.splitext(os.path.basename(video))[0]
        if (args.csv or args.plot) and out_dir:
            os.makedirs(out_dir, exist_ok=True)
        if args.csv:
            write_csv(rows, os.path.join(out_dir, stem + "_frames.csv"))
        if args.plot:
            write_plots(rows, timing, os.path.join(out_dir, stem + "_analysis.png"))
        if args.extract:
            extract_frames(video, rows, args.extract,
                           os.path.join(out_dir, stem + "_frames"),
                           n=args.extract_n, roi=args.roi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
