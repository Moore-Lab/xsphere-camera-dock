#!/usr/bin/env python3
"""
compile_timelapse.py — compile a timelapse capture folder into an annotated video.
=================================================================================
A timelapse run (``POST /timelapse/start``) leaves behind a folder of stills, one
subfolder per camera:

    captures/timelapse_20260812_124335/
        zelux-29559/cs165mu_20260812_124335_245076.tiff
        zelux-32943/cs165mu_20260812_124335_238758.tiff
        zelux-35969/cs165mu_20260812_124335_249126.tiff

This turns that into one watchable video and burns in the thing a timelapse is
unreadable without: **both frame rates**. A timelapse has an acquisition rate
(how often the world was sampled) and a playback rate (how fast the video runs);
neither alone tells you what a second of video is worth — only their ratio does.
Both, and that ratio, sit in a banner on every frame, alongside the wall clock.

Cadence is *measured from the frame filenames*, never assumed. The capturer
drops a slot rather than firing a catch-up burst (see ``webapp.Timelapse``), so
the requested interval and the delivered one can differ; what gets annotated is
what actually landed on disk. The median inter-frame gap is used, so a handful
of missed slots shifts nothing.

Cameras are aligned by *timestamp*, not by index, so a camera that missed a slot
stays in sync with the others instead of sliding a frame ahead for the rest of
the run.

Run it:
    python analysis/compile_timelapse.py                    # newest captures/timelapse_*
    python analysis/compile_timelapse.py <dir>              # a specific run
    python analysis/compile_timelapse.py <dir> --fps 15     # faster playback
    python analysis/compile_timelapse.py <dir> --step 2     # use every 2nd frame
    python analysis/compile_timelapse.py <dir> --per-camera # one video per camera
"""

from __future__ import annotations

import argparse
import bisect
import glob
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timedelta

import cv2
import numpy as np

try:
    from zoneinfo import ZoneInfo
    _EASTERN = ZoneInfo("America/New_York")   # New Haven, CT (DST-aware)
except Exception:
    _EASTERN = None    # tz database unavailable (see requirements: tzdata)

# captures/ resolved from THIS file (repo root is the parent of analysis/), so the
# default works no matter what directory the script is launched from.
REPO_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
DEFAULT_CAPTURES_DIR = os.path.join(REPO_ROOT, "captures")

FRAME_EXTS = (".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp", ".npy")

# webapp.Timelapse names every still "<slug>_%Y%m%d_%H%M%S_%f".
_STAMP_RE = re.compile(r"_(\d{8})_(\d{6})(?:_(\d{1,6}))?$")

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
BANNER_BG = (18, 18, 18)
# Hershey fonts are ASCII-only — a "->" or "x" here, never an arrow or multiply
# sign, which cv2.putText would silently render as "?".
ARROW, TIMES = "->", "x"


# --- discovery ------------------------------------------------------------

def parse_stamp(path: str):
    """Wall-clock time encoded in a frame's filename, or None if absent."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _STAMP_RE.search(stem)
    if m is None:
        return None
    day, tod, us = m.group(1), m.group(2), (m.group(3) or "0")
    try:
        dt = datetime.strptime(day + tod, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    dt = dt.replace(microsecond=int(us.ljust(6, "0")[:6]))
    # The capturer stamps naive local time; this rig runs on New Haven time, so
    # attaching Eastern is what makes the burnt-in label say EDT/EST.
    return dt.replace(tzinfo=_EASTERN) if _EASTERN else dt


def frames_in(directory: str):
    """Sorted [(datetime, path), ...] for every image file in one directory."""
    out = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(name)[1].lower() not in FRAME_EXTS:
            continue
        dt = parse_stamp(path)
        if dt is None:      # unstamped file: fall back to mtime so it still sorts
            dt = datetime.fromtimestamp(os.path.getmtime(path), tz=_EASTERN)
        out.append((dt, path))
    out.sort(key=lambda item: item[0])
    return out


def discover(root: str):
    """Return [(camera_name, frames), ...] — per-camera subfolders, else flat."""
    groups = []
    for name in sorted(os.listdir(root)):
        sub = os.path.join(root, name)
        if os.path.isdir(sub):
            frames = frames_in(sub)
            if frames:
                groups.append((name, frames))
    if not groups:
        frames = frames_in(root)
        if frames:
            groups.append((os.path.basename(os.path.normpath(root)), frames))
    return groups


def run_time(path: str) -> datetime:
    """When a capture folder was made, for ordering runs.

    Most folders carry the timestamp in the name (timelapse_20260812_124335),
    but not all do (timelapse_overnight) — and sorting by name alone puts
    "overnight" above every dated run. Fall back to the folder's mtime.
    """
    m = re.search(r"(\d{8})_(\d{6})", os.path.basename(os.path.normpath(path)))
    if m:
        try:
            return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    return datetime.fromtimestamp(os.path.getmtime(path))


def newest_capture(captures_dir: str):
    """Most recent captures/timelapse_* folder."""
    runs = [d for d in glob.glob(os.path.join(captures_dir, "timelapse_*"))
            if os.path.isdir(d)]
    return max(runs, key=run_time) if runs else None


# --- cadence --------------------------------------------------------------

def cadence(times):
    """Measured acquisition cadence: median interval, span, and missed slots.

    Median, not mean: a run that lost a slot has one double-length gap, which
    would drag a mean off the true cadence but leaves the median untouched.
    """
    n = len(times)
    if n < 2:
        return {"n": n, "interval_s": None, "fps": None, "span_s": 0.0,
                "missed": 0, "jitter_s": None}
    deltas = [(b - a).total_seconds() for a, b in zip(times, times[1:])]
    interval = statistics.median(deltas)
    missed = 0
    if interval > 0:
        # A gap of k intervals means k-1 slots produced no frame.
        missed = sum(max(0, int(round(d / interval)) - 1)
                     for d in deltas if d > 1.5 * interval)
    return {
        "n": n,
        "interval_s": interval,
        "fps": (1.0 / interval) if interval > 0 else None,
        "span_s": (times[-1] - times[0]).total_seconds(),
        "missed": missed,
        "jitter_s": statistics.median(abs(d - interval) for d in deltas),
    }


def align(groups, tol_s: float):
    """Align every camera onto the densest camera's timeline.

    Returns (ref_times, {camera: [frame index or None per ref time]}). A camera
    that missed a slot carries its previous frame forward, so its pane holds the
    last real image instead of flashing black or drifting out of step.
    """
    _, ref = max(groups, key=lambda g: len(g[1]))
    ref_times = [t for t, _ in ref]
    picks = {}
    for name, frames in groups:
        times = [t for t, _ in frames]
        chosen, last = [], None
        for rt in ref_times:
            i = bisect.bisect_left(times, rt)
            best, best_d = None, None
            for j in (i - 1, i):
                if 0 <= j < len(times):
                    d = abs((times[j] - rt).total_seconds())
                    if best_d is None or d < best_d:
                        best, best_d = j, d
            if best is not None and best_d <= tol_s:
                last = best
            chosen.append(last)
        picks[name] = chosen
    return ref_times, picks


# --- pixels ---------------------------------------------------------------

def load_frame(path: str):
    if path.lower().endswith(".npy"):
        return np.load(path)
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def infer_bit_depth(paths, sample: int = 8) -> int:
    """Guess a uint16 frame's real bit depth from the brightest sampled pixel.

    Timelapse stills saved at scale=raw keep the sensor's native depth in a
    16-bit container, so shifting by a fixed 8 would black out a 10- or 12-bit
    frame. Inferred once for the whole run, never per frame — a per-frame
    normalisation would make the video flicker as the scene brightness changes.
    """
    peak = 0
    for path in paths[:sample]:
        img = load_frame(path)
        if img is not None:
            peak = max(peak, int(img.max()))
    for depth in (8, 10, 12, 14, 16):
        if peak <= (1 << depth) - 1:
            return depth
    return 16


def to_bgr(img, bit_depth: int, size):
    """Native frame -> 8-bit BGR of exactly ``size`` (w, h). None -> black."""
    w, h = size
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    if img.dtype != np.uint8:
        shift = max(0, int(bit_depth) - 8)
        img = (np.right_shift(img.astype(np.uint32), shift)
               if shift else img).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def _measure(text: str, fs: float, th: int):
    (tw, tht), base = cv2.getTextSize(text, FONT, fs, th)
    return tw, tht, base


class Banner:
    """Fixed-geometry caption strip: the two rates on the left, the clock right.

    Geometry is computed once, from the widest text the run can produce, and
    reused for every frame — a banner that changed height mid-run would change
    the frame size and break the encode.
    """

    def __init__(self, width: int, rate_text: str, clock_sample: str):
        self.width = width
        self.rate_text = rate_text
        self.fs = max(0.42, width / 1900.0)
        self.th = max(1, int(round(self.fs * 2)))
        self.pad = max(6, int(round(self.fs * 12)))

        lw, tht, base = _measure(rate_text, self.fs, self.th)
        rw, _, _ = _measure(clock_sample, self.fs, self.th)
        self.line_h = tht + base
        # Side by side if they fit; stacked if the frame is too narrow (a single
        # 720-wide camera pane cannot hold both on one line).
        self.two_rows = (lw + rw + 3 * self.pad) > width
        rows = 2 if self.two_rows else 1
        height = rows * (self.line_h + self.pad) + self.pad
        self.height = height + (height % 2)   # even: yuv420p needs even dimensions

    def render(self, clock_text: str):
        banner = np.full((self.height, self.width, 3), BANNER_BG, np.uint8)
        y = self.pad + self.line_h
        cv2.putText(banner, self.rate_text, (self.pad, y), FONT, self.fs,
                    GREEN, self.th, cv2.LINE_AA)
        if self.two_rows:
            cv2.putText(banner, clock_text, (self.pad, y + self.line_h + self.pad),
                        FONT, self.fs, GREEN, self.th, cv2.LINE_AA)
        else:
            rw, _, _ = _measure(clock_text, self.fs, self.th)
            cv2.putText(banner, clock_text, (self.width - rw - self.pad, y),
                        FONT, self.fs, GREEN, self.th, cv2.LINE_AA)
        return banner


def label_pane(bgr, text: str):
    """Camera name, top-left, over a translucent plate so it survives a white scene."""
    fs = max(0.4, bgr.shape[1] / 1400.0)
    th = max(1, int(round(fs * 2)))
    tw, tht, base = _measure(text, fs, th)
    pad = 6
    plate = bgr.copy()
    cv2.rectangle(plate, (0, 0), (tw + 2 * pad, tht + base + 2 * pad), (0, 0, 0), -1)
    cv2.addWeighted(plate, 0.5, bgr, 0.5, 0, bgr)
    cv2.putText(bgr, text, (pad, tht + pad), FONT, fs, GREEN, th, cv2.LINE_AA)
    return bgr


# --- encoding -------------------------------------------------------------

def find_ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    # winget installs Gyan.FFmpeg as a portable package and edits PATH; shells
    # started before the install never see it.
    hits = sorted(glob.glob(os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages",
        "Gyan.FFmpeg*", "*", "bin", "ffmpeg.exe")))
    return hits[-1] if hits else None


class FFmpegEncoder:
    """H.264/mp4 via a raw-frame pipe — no temp files, no frame held in RAM.

    Note the absence of an output ``-r``: the rawvideo demuxer already carries
    the rate from ``-framerate``, and adding ``-r`` makes ffmpeg re-rate the
    stream and duplicate frames.
    """

    suffix = ".mp4"

    def __init__(self, exe: str, path: str, size, fps: float, crf: int = 18):
        w, h = size
        self.proc = subprocess.Popen(
            [exe, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-framerate", f"{fps:g}", "-i", "pipe:0",
             "-an", "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)

    def write(self, bgr):
        self.proc.stdin.write(np.ascontiguousarray(bgr).tobytes())

    def close(self) -> bool:
        self.proc.stdin.close()
        return self.proc.wait() == 0


class Cv2Encoder:
    """Fallback when ffmpeg is missing: MJPG/avi, the codec the recorder falls back to."""

    suffix = ".avi"

    def __init__(self, path: str, size, fps: float, crf: int = 18):
        self.writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"MJPG"), max(fps, 1.0), size, True)
        if not self.writer.isOpened():
            raise IOError(f"could not open a video writer for {path}")

    def write(self, bgr):
        self.writer.write(bgr)

    def close(self) -> bool:
        self.writer.release()
        return True


def open_encoder(path_stem: str, size, fps: float, crf: int):
    exe = find_ffmpeg()
    if exe:
        path = path_stem + FFmpegEncoder.suffix
        return FFmpegEncoder(exe, path, size, fps, crf), path
    path = path_stem + Cv2Encoder.suffix
    print("NOTE: ffmpeg not found — falling back to MJPG/avi (larger file).")
    return Cv2Encoder(path, size, fps, crf), path


# --- assembly -------------------------------------------------------------

def format_clock(dt: datetime, elapsed: timedelta) -> str:
    tz = dt.tzname() or ""
    total = int(elapsed.total_seconds())
    hms = f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
    return dt.strftime("%Y-%m-%d %H:%M:%S") + (f" {tz}" if tz else "") + f"  +{hms}"


def rate_caption(interval_s: float, play_fps: float) -> str:
    """The line the whole script exists for: both rates and what they multiply to."""
    if not interval_s or interval_s <= 0:
        return f"PLAY {play_fps:g} fps"
    acq_fps = 1.0 / interval_s
    speed = play_fps * interval_s
    speed_txt = f"{speed:,.0f}" if speed >= 10 else f"{speed:.3g}"
    per = f"1 frame / {interval_s:.6g} s"
    return (f"ACQ {per} ({acq_fps:.4g} fps)   {ARROW}   PLAY {play_fps:g} fps"
            f"   =   {speed_txt}{TIMES} real time")


def compile_run(groups, out_stem: str, play_fps: float, step: int, crf: int,
                tol_frac: float = 0.5) -> dict:
    """Render one video from one or more cameras (side by side if several)."""
    ref_times, picks = align(groups, tol_s=_tolerance(groups, tol_frac))
    ref_times = ref_times[::step]
    picks = {name: idx[::step] for name, idx in picks.items()}

    stats = cadence(ref_times)
    interval = stats["interval_s"] or 0.0

    # Pane geometry from the first readable frame of each camera.
    names = [name for name, _ in groups]
    sizes = {}
    for name, frames in groups:
        img = load_frame(frames[0][1])
        sizes[name] = (img.shape[1], img.shape[0]) if img is not None else (0, 0)
    pane_h = max(h for _, h in sizes.values())
    depths = {name: (infer_bit_depth([p for _, p in frames]))
              for name, frames in groups}

    # Panes keep their aspect ratio against a common height.
    pane_w = {name: max(2, int(round(w * pane_h / h))) if h else 2
              for name, (w, h) in sizes.items()}
    width = sum(pane_w[name] for name in names)
    width += width % 2

    banner = Banner(width, rate_caption(interval, play_fps),
                    format_clock(ref_times[0], timedelta(seconds=99999)))
    size = (width, pane_h + banner.height)

    enc, out_path = open_encoder(out_stem, size, play_fps, crf)
    t0 = ref_times[0]
    written = 0
    try:
        for k, rt in enumerate(ref_times):
            panes = []
            for name, frames in groups:
                j = picks[name][k]
                img = load_frame(frames[j][1]) if j is not None else None
                pane = to_bgr(img, depths[name], (pane_w[name], pane_h))
                panes.append(label_pane(pane, name) if len(names) > 1 else pane)
            montage = np.hstack(panes) if len(panes) > 1 else panes[0]
            if montage.shape[1] != width:      # odd-width padding
                montage = cv2.copyMakeBorder(
                    montage, 0, 0, 0, width - montage.shape[1],
                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
            frame = np.vstack([banner.render(format_clock(rt, rt - t0)), montage])
            enc.write(frame)
            written += 1
            if written % 100 == 0:
                print(f"  {written}/{len(ref_times)} frames", flush=True)
    finally:
        ok = enc.close()

    return {
        "ok": bool(ok),
        "video": out_path,
        "cameras": names,
        "frames": written,
        "width": size[0], "height": size[1],
        "acquisition": {
            "interval_s": round(interval, 6) if interval else None,
            "fps": round(stats["fps"], 6) if stats["fps"] else None,
            "missed_slots": stats["missed"],
            "jitter_s": round(stats["jitter_s"], 6) if stats["jitter_s"] else None,
            "step": step,
            "span_s": round(stats["span_s"], 3),
            "first": ref_times[0].isoformat(),
            "last": ref_times[-1].isoformat(),
        },
        "playback": {
            "fps": play_fps,
            "duration_s": round(written / play_fps, 3) if play_fps else None,
            "speedup": round(play_fps * interval, 4) if interval else None,
        },
        "caption": banner.rate_text,
    }


def _tolerance(groups, frac: float) -> float:
    """Half a cadence: how far a camera's frame may sit from a timeline slot."""
    _, ref = max(groups, key=lambda g: len(g[1]))
    stats = cadence([t for t, _ in ref])
    return (stats["interval_s"] or 1.0) * frac


def describe(result: dict) -> str:
    acq, play = result["acquisition"], result["playback"]
    span = timedelta(seconds=int(acq["span_s"]))
    lines = [
        f"  cameras   {', '.join(result['cameras'])}",
        f"  frames    {result['frames']}  ({result['width']}x{result['height']})",
        f"  acquired  {acq['fps']:.4g} fps  (1 frame / {acq['interval_s']:.6g} s)"
        f"   over {span}",
        f"  playback  {play['fps']:g} fps  ->  {play['duration_s']:.1f} s"
        f"   ({play['speedup']:,.0f}x real time)",
    ]
    if acq["missed_slots"]:
        lines.append(f"  missed    {acq['missed_slots']} slot(s) with no frame")
    return "\n".join(lines)


# --- cli ------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("Run it:")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", default=None,
                   help="timelapse capture folder (default: newest captures/timelapse_*)")
    p.add_argument("--fps", type=float, default=10.0, metavar="F",
                   help="playback frame rate (default: 10)")
    p.add_argument("--step", type=int, default=1, metavar="N",
                   help="use every Nth captured frame (default: 1)")
    p.add_argument("--per-camera", action="store_true",
                   help="one video per camera instead of a side-by-side montage")
    p.add_argument("--out", metavar="PATH",
                   help="output path stem (default: alongside the capture folder)")
    p.add_argument("--crf", type=int, default=18, metavar="N",
                   help="H.264 quality, lower is better (default: 18)")
    args = p.parse_args(argv)

    root = args.path or newest_capture(DEFAULT_CAPTURES_DIR)
    if not root:
        p.error(f"no timelapse_* folders in {DEFAULT_CAPTURES_DIR}")
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        p.error(f"not a directory: {root}")
    if args.step < 1:
        p.error("--step must be >= 1")
    if args.fps <= 0:
        p.error("--fps must be > 0")

    groups = discover(root)
    if not groups:
        p.error(f"no image frames found under {root}")

    name = os.path.basename(os.path.normpath(root))
    stem = args.out or os.path.join(os.path.dirname(root), name)
    print(f"compiling {name}  ({len(groups)} camera(s), "
          f"{sum(len(f) for _, f in groups)} stills)")

    jobs = ([([g], f"{stem}_{g[0]}") for g in groups] if args.per_camera
            else [(groups, stem)])

    results = []
    for job_groups, job_stem in jobs:
        result = compile_run(job_groups, job_stem, args.fps, args.step, args.crf)
        results.append(result)
        # The container's fps is only a playback setting — the measured cadence
        # lives in a sidecar, same convention as camera_dock/metadata.py.
        with open(os.path.splitext(result["video"])[0] + ".json", "w") as fh:
            json.dump({"source": root, **result}, fh, indent=2)
        status = "wrote" if result["ok"] else "FAILED"
        print(f"{status} {result['video']}")
        print(describe(result))

    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
