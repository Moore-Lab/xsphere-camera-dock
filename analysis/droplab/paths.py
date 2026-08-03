"""Directory anchors for droplab.

Everything is derived from THIS file's location so the server works no matter
the process CWD (bat launch, IDE run button, uvicorn reload):

    REPO_ROOT      = <repo>                the camera-dock repo root
    RECORDINGS_DIR = <repo>/recordings     where the dock recorder writes
    ANALYSIS_ROOT  = <repo>/analysis       droplab's own output tree

Output tree mirrors the count-droplets project layout exactly, so its
downstream scripts (track_droplets, analyze_distributions, longitudinal) can be
pointed at ANALYSIS_ROOT as their project root and just work:

    analysis/data/<video_basename>/<basename>_droplet_data.npy
    analysis/data/<video_basename>/<basename>_tracking_data.npy
    analysis/data/<video_basename>/<basename>_test_results.json
    analysis/logs/<basename>_<YYYY-MM-DD_HH-MM-SS>.json
    analysis/plots/<video_basename>/*.png
"""

from __future__ import annotations

import os

ANALYSIS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
REPO_ROOT = os.path.normpath(os.path.join(ANALYSIS_ROOT, os.pardir))
RECORDINGS_DIR = os.path.join(REPO_ROOT, "recordings")

DATA_DIR = os.path.join(ANALYSIS_ROOT, "data")
LOGS_DIR = os.path.join(ANALYSIS_ROOT, "logs")
PLOTS_DIR = os.path.join(ANALYSIS_ROOT, "plots")

VIDEO_EXTS = (".mp4", ".avi")


def video_basename(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def data_dir_for(video_path: str, *, create: bool = False) -> str:
    d = os.path.join(DATA_DIR, video_basename(video_path))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def plots_dir_for(video_path: str, *, create: bool = False) -> str:
    d = os.path.join(PLOTS_DIR, video_basename(video_path))
    if create:
        os.makedirs(d, exist_ok=True)
    return d


def logs_dir(*, create: bool = False) -> str:
    if create:
        os.makedirs(LOGS_DIR, exist_ok=True)
    return LOGS_DIR
