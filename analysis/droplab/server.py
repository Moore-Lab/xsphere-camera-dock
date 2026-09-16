"""droplab — browser-based droplet analysis for camera-dock recordings.

Runs beside the camera dock so recordings can be analyzed while the next one
is being captured in another browser window:

    python -m analysis.droplab.server                # http://127.0.0.1:8020
    python -m analysis.droplab.server --port 8020 --dir extra/videos/dir

Pages
    GET  /                      recording browser (auto-refreshing)
    GET  /lab/{video}           analysis page for one recording

API (all JSON unless noted)
    GET  /videos                                    recordings, newest first
    GET  /video/{video}/info                        probe + prefill defaults
    GET  /video/{video}/frame/{idx}                 JPEG; ?view=raw|droplets|processed
                                                    &threshold&min_area&max_area
                                                    &x0&x1&y0&y1  (ROI overlay)
                                                    droplet counts in X-Droplet-* headers
    GET  /video/{video}/sum                         JPEG persistence view (cached);
                                                    ?start&end&threshold&mode&x0&x1&y0&y1
    POST /video/{video}/analyze                     start detection job (JSON body)
    POST /video/{video}/track                       start tracking job
    POST /video/{video}/distributions               start distributions job
    POST /video/{video}/track_analysis              start track-analysis job
    POST /video/{video}/longitudinal                start longitudinal job
    GET  /video/{video}/results                     saved outputs + stats + plot list
    GET  /video/{video}/streamlines                 track polylines for the viewer
    GET  /video/{video}/logs   ·  GET /log?path=    parameter logs (Load Log)
    GET  /jobs?video=  ·  GET /jobs/{id}  ·  POST /jobs/{id}/cancel
    GET  /plots/{basename}/{fname}                  generated PNGs

Error convention (as camera_dock.webapp): 404 unknown video · 409 job already
running · 422 bad parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from . import detect, store
from .jobs import JobManager
from .paths import (DATA_DIR, LOGS_DIR, PLOTS_DIR, RECORDINGS_DIR,
                    data_dir_for, video_basename)
from .video import (ReaderCache, VideoReader, list_videos, probe,
                    resolve_video, sidecar_meta)

JPEG_QUALITY = 85
SUM_MAX_FRAMES = 1500          # decimate persistence view beyond this many frames
_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _vendor():
    """Import the vendored analysis stack on first use.

    matplotlib + scipy + the count-droplets scripts cost ~1.5 s and ~150 MB;
    paying that at import time would tax every DAQ panel launch even when
    nobody opens the analysis tab. Detection, previews and the track overlay
    never touch it.
    """
    from . import vendor_api
    return vendor_api


def _ppum_provenance(info: dict) -> dict:
    """Where the prefilled pixels_per_um came from, and whether to trust it.

    Every physical result downstream — radius, velocity, volume, mass — scales
    with this number, so a constant borrowed from another rig must never look
    like a measurement. ``from_sidecar`` drives the caption's colour, and it is
    deliberately False for anisotropic binning: the scale is the horizontal one
    and is applied isotropically downstream, so it is simply wrong in y.
    """
    ppum = info.get("pixels_per_um")
    if not ppum:
        return {"pixels_per_um_source":
                    "count-droplets default — this recording has no calibration",
                "pixels_per_um_from_sidecar": False}
    named = f'recording calibration “{info["calibration"]}”' if info.get("calibration") \
        else "recording calibration"
    if info.get("anisotropic_binning"):
        # Name the IMAGE's axes: after a quarter turn the recording's horizontal
        # axis is sensor y, and the scale follows it (video.sidecar_image_binning).
        b = info.get("image_binning") or info.get("binning") or []
        bxby = f"{b[0]}×{b[1]}" if len(b) > 1 else "anisotropic"
        return {"pixels_per_um_source":
                    f"{named} — image binning is {bxby} (horizontal×vertical), so "
                    f"this scale is valid for the horizontal axis only",
                "pixels_per_um_from_sidecar": False}
    return {"pixels_per_um_source": named, "pixels_per_um_from_sidecar": True}


def create_app(video_dirs: Optional[list[str]] = None) -> FastAPI:
    dirs = video_dirs or [RECORDINGS_DIR]
    readers = ReaderCache(max_open=4)
    jobs = JobManager(max_workers=2)
    sum_cache: dict[tuple, bytes] = {}
    sum_lock = threading.Lock()
    sum_inflight: dict[tuple, threading.Lock] = {}

    def _shutdown():
        jobs.shutdown()                     # stop/drain analysis work first…
        readers.close_all()                 # …then release the video files

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            _shutdown()

    app = FastAPI(title="droplab", lifespan=lifespan)
    # Starlette does not run a MOUNTED sub-app's lifespan, so a parent app
    # (xsphere_daq.panel) calls this itself on shutdown.
    app.state.shutdown_hooks = [_shutdown]

    # ---------------------------------------------------------------- utils
    def vid_path(video: str) -> str:
        p = resolve_video(video, dirs)
        if p is None:
            raise HTTPException(404, f"no video '{video}' in {dirs}")
        return p

    def job_key(video: str) -> str:
        # Windows paths are case-insensitive; don't let "Foo.avi" and
        # "foo.avi" run concurrent jobs on the same file.
        return video.lower()

    async def json_body(request: Request) -> dict:
        try:
            p = await request.json()
        except Exception:
            raise HTTPException(422, "request body must be JSON")
        if not isinstance(p, dict):
            raise HTTPException(422, "JSON body must be an object")
        return p

    def safe_under(root: str, *parts: str) -> str:
        """Join under *root*, 404ing any escape ('..' survives basename())."""
        full = os.path.realpath(os.path.join(root, *parts))
        rootr = os.path.realpath(root)
        try:
            inside = os.path.commonpath([full, rootr]) == rootr
        except ValueError:                  # different drives
            inside = False
        if not inside:
            raise HTTPException(404)
        return full

    def submit(kind: str, video: str, params: dict, fn) -> dict:
        job, active = jobs.submit_exclusive(kind, job_key(video), params, fn)
        if job is None:
            raise HTTPException(409, f"job '{active.kind}' already running for {video}")
        return job.as_dict()

    def parse_roi(q, w: int, h: int):
        """ROI from query params (x0,x1,y0,y1), clamped; defaults to full frame."""
        try:
            x0 = int(q.get("x0", 0)); x1 = int(q.get("x1", w))
            y0 = int(q.get("y0", 0)); y1 = int(q.get("y1", h))
        except (TypeError, ValueError):
            raise HTTPException(422, "ROI params must be integers")
        x0 = max(0, min(x0, w - 1)); y0 = max(0, min(y0, h - 1))
        x1 = max(x0 + 1, min(x1, w)); y1 = max(y0 + 1, min(y1, h))
        return x0, x1, y0, y1

    def encode_jpeg(bgr) -> bytes:
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise HTTPException(500, "JPEG encode failed")
        return bytes(buf)

    # ---------------------------------------------------------- pages/static
    @app.get("/")
    def index():
        return FileResponse(os.path.join(_STATIC, "index.html"))

    @app.get("/static/{fname}")
    def static_file(fname: str):
        path = os.path.join(_STATIC, os.path.basename(fname))
        if not os.path.isfile(path):
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/lab/{video}")
    def lab_page(video: str):
        vid_path(video)
        return FileResponse(os.path.join(_STATIC, "lab.html"))

    # ----------------------------------------------------------- discovery
    @app.get("/videos")
    def videos():
        out = list_videos(dirs)
        for v in out:
            ddir = os.path.join(DATA_DIR, v["basename"])
            v["analyzed"] = os.path.isfile(
                os.path.join(ddir, f"{v['basename']}_droplet_data.npy"))
            v["tracked"] = os.path.isfile(
                os.path.join(ddir, f"{v['basename']}_track_data.npz"))
            active = jobs.active_for(job_key(v["name"]))
            v["active_job"] = active.kind if active else None
        return out

    @app.get("/video/{video}/info")
    def video_info(video: str):
        path = vid_path(video)
        try:
            info = probe(path)
        except RuntimeError as e:
            raise HTTPException(503, f"video not readable (still encoding?): {e}")
        # Prefill defaults for the parameter panel
        fps = info["fps"]
        info["defaults"] = {
            "threshold": 50, "min_area": 10, "max_area": 100,
            "start_time": 0.0,
            # ceil, not round — rounding down would silently drop final frames
            "end_time": (math.ceil(info["total_frames"] / fps * 1000) / 1000
                         if fps else 0.0),
            "fps_override": (round(info["sidecar_fps"], 3)
                             if info["sidecar_fps"] and info["container_fps"]
                             and abs(info["sidecar_fps"] - info["container_fps"]) > 0.5
                             else None),
            "exposure_time_s": None,
            # Prefer this recording's own calibration; fall back to the upstream
            # count-droplets constant only when the sidecar has none, and label
            # which one it is so the operator can see it at a glance.
            "pixels_per_um": (info.get("pixels_per_um")
                              or store.DEFAULT_PIXELS_PER_UM),
            **_ppum_provenance(info),
        }
        return info

    # ---------------------------------------------------------- frame views
    @app.get("/video/{video}/frame/{idx}")
    def frame(video: str, idx: int, request: Request,
              view: str = "raw", threshold: int = 50,
              min_area: int = 10, max_area: int = 100):
        path = vid_path(video)
        try:
            with readers.lease(path) as r:
                if idx >= r.total_frames:
                    r.refresh_length()     # growing file (still encoding)
                frame = r.read(idx)
        except RuntimeError as e:
            raise HTTPException(503, f"video not readable (still encoding?): {e}")
        if frame is None:
            raise HTTPException(404, f"no frame {idx}")
        h, w = frame.shape[:2]
        roi = parse_roi(request.query_params, w, h)

        headers = {"Cache-Control": "no-store"}
        if view == "processed":
            count, droplets, binary = detect.detect_droplets(
                frame, roi, threshold, min_area, max_area, return_processed=True)
            out = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            detect.draw_overlay(out, [], roi)      # ROI rectangle only
            headers["X-Droplet-Count"] = str(count)
            headers["X-Droplet-Count-Roi"] = str(sum(d["in_roi"] for d in droplets))
        elif view == "droplets":
            count, droplets = detect.detect_droplets(
                frame, roi, threshold, min_area, max_area)
            out = detect.draw_overlay(frame.copy(), droplets, roi)
            headers["X-Droplet-Count"] = str(count)
            headers["X-Droplet-Count-Roi"] = str(sum(d["in_roi"] for d in droplets))
        else:
            out = frame.copy()
            detect.draw_overlay(out, [], roi)
        return Response(encode_jpeg(out), media_type="image/jpeg", headers=headers)

    @app.get("/video/{video}/sum")
    def sum_frame(video: str, request: Request, start: int = 0, end: int = 0,
                  threshold: int = 50, mode: str = "analysis"):
        path = vid_path(video)
        try:
            info = probe(path)
        except RuntimeError as e:
            raise HTTPException(503, f"video not readable (still encoding?): {e}")
        w, h, total = info["width"], info["height"], info["total_frames"]
        roi = parse_roi(request.query_params, w, h)
        if mode == "full":
            s, e = 0, total
        else:
            s, e = max(0, start), min(end if end > 0 else total, total)
        if e <= s:
            raise HTTPException(422, "empty frame range")
        step = max(1, (e - s) // SUM_MAX_FRAMES)

        key = (path, s, e, threshold, roi, mode == "analysis", step)
        with sum_lock:
            cached = sum_cache.get(key)
            if cached is None:
                inflight = sum_inflight.setdefault(key, threading.Lock())
        if cached is None:
            # Identical concurrent requests wait on the key lock and then hit
            # the cache; the compute uses its own reader so a long accumulation
            # never blocks preview scrubbing.
            with inflight:
                with sum_lock:
                    cached = sum_cache.get(key)
                if cached is None:
                    r = VideoReader(path)
                    try:
                        img, n = detect.sum_view(
                            r.read, s, e, threshold, roi, (w, h),
                            analysis_only=(mode == "analysis"), step=step)
                    finally:
                        r.close()
                    if img is None:
                        raise HTTPException(404, "no frames to accumulate")
                    cached = encode_jpeg(img)
                    with sum_lock:
                        if len(sum_cache) > 12:
                            sum_cache.clear()
                            sum_inflight.clear()
                        sum_cache[key] = cached
        return Response(cached, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-Sum-Step": str(step)})

    # ------------------------------------------------------------- analysis
    @app.post("/video/{video}/analyze")
    async def analyze(video: str, request: Request):
        path = vid_path(video)
        p = await json_body(request)
        try:
            info = probe(path)
        except RuntimeError as e:
            raise HTTPException(503, f"video not readable (still encoding?): {e}")
        w, h = info["width"], info["height"]

        try:
            x0, x1 = int(p.get("x_start", 0)), int(p.get("x_stop", w))
            y0, y1 = int(p.get("y_start", 0)), int(p.get("y_stop", h))
            threshold = int(p.get("threshold", 50))
            min_area = int(p.get("min_area", 10))
            max_area = int(p.get("max_area", 100))
            start_time = float(p.get("start_time", 0.0))
            end_time = float(p.get("end_time", 0.0))
            fps_override = p.get("fps_override")
            fps_override = float(fps_override) if fps_override not in (None, "") else None
            exposure = p.get("exposure_time_s")
            exposure = float(exposure) if exposure not in (None, "") else None
            # An explicit client value wins; otherwise prefer THIS recording's
            # calibration (already in `info`) over the count-droplets constant,
            # which belongs to a different rig. This is the layer that writes
            # results, so the default here is the one that ends up in the log.
            ppum_in = p.get("pixels_per_um")
            pixels_per_um = (float(ppum_in) if ppum_in not in (None, "")
                             else (info.get("pixels_per_um")
                                   or store.DEFAULT_PIXELS_PER_UM))
        except (TypeError, ValueError):
            raise HTTPException(422, "bad parameter types")
        if pixels_per_um <= 0:
            raise HTTPException(422, "pixels_per_um must be positive")
        if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
            raise HTTPException(422, f"ROI out of bounds for {w}x{h}")
        if start_time < 0:
            raise HTTPException(422, "start_time must be >= 0")
        if end_time <= start_time:
            raise HTTPException(422, "end_time must be > start_time")
        if not (0 <= threshold <= 255):
            raise HTTPException(422, "threshold must be 0..255")
        if min_area < 0 or max_area < min_area:
            raise HTTPException(422, "need 0 <= min_area <= max_area")
        if fps_override is not None and fps_override <= 0:
            raise HTTPException(422, "fps_override must be positive")
        if exposure is not None and exposure <= 0:
            raise HTTPException(422, "exposure_time_s must be positive")

        fps = fps_override or info["fps"]
        roi = (x0, x1, y0, y1)
        params = {
            "start_time": start_time, "end_time": end_time,
            "x_start": x0, "x_stop": x1, "y_start": y0, "y_stop": y1,
            "threshold": threshold, "min_area": min_area, "max_area": max_area,
            "fps": fps, "fps_override": fps_override,
            "exposure_time_s": exposure, "pixels_per_um": pixels_per_um,
        }

        def work(job):
            r = VideoReader(path)          # private reader — never contends
            try:                           # with previews, never gets evicted
                total_frames = r.total_frames
                start_frame = int(start_time * fps)
                end_frame = min(int(end_time * fps), total_frames)
                if end_frame <= start_frame:
                    raise ValueError(f"empty frame window [{start_frame}, {end_frame})")
                job.message = f"detecting frames {start_frame}..{end_frame}"
                frame_data = detect.run_detection(
                    r.read, start_frame, end_frame, fps, roi,
                    (info["width"], info["height"]),
                    threshold, min_area, max_area,
                    progress=lambda d, t: job.report(d, t),
                    cancelled=job.cancelled)
            finally:
                r.close()
            if job.cancelled():
                return {"cancelled_after_frames": len(frame_data)}
            if not frame_data:
                # store.py refuses empty saves too — this is the clearer error
                raise ValueError(
                    f"no decodable frames in window [{start_frame}, {end_frame}) "
                    "(recording still encoding, or start past the real end?)")
            job.message = "saving outputs"
            npy_path, arr = store.save_droplet_data(frame_data, path)
            store.save_tracking_data(frame_data, path)
            results_path, results = store.save_results(
                frame_data, path, params, info["width"], info["height"])
            log_path = store.save_log(path, params)
            return store.json_safe({
                "frames_analyzed": len(frame_data),
                "droplets_in_roi": int(len(arr)),
                "droplet_data": npy_path, "results_json": results_path,
                "log": log_path,
                "mean_count_roi": results["mean_droplet_count_roi"],
                "std_count_roi": results["std_droplet_count_roi"],
                "mean_density_roi_per_pixel": results["mean_density_roi_per_pixel"],
            })

        return submit("detect", video, params, work)

    @app.post("/video/{video}/track")
    async def track(video: str, request: Request):
        path = vid_path(video)
        p = await json_body(request)
        auto = bool(p.get("auto_distance", False))
        try:
            max_distance = float(p.get("max_distance", 24.0))
            min_len = int(p.get("min_track_length", 1))
        except (TypeError, ValueError):
            raise HTTPException(422, "bad parameter types")
        fps, ppum = _calibration_from_results(path)

        def work(job):
            job.message = ("scanning linking distances 0..74"
                           if auto else f"linking at {max_distance:g} px")
            return _vendor().run_tracking(
                path, max_distance=max_distance, min_track_length=min_len,
                fps=fps, pixels_per_um=ppum, auto_distance=auto,
                progress=lambda d, t: job.report(d, t))

        return submit("track", video, p, work)

    @app.post("/video/{video}/distributions")
    async def distributions(video: str, request: Request):
        path = vid_path(video)
        p = await json_body(request)
        try:
            density = float(p.get("density_kg_m3", 3520.0))
            r_mm = float(p["r_mm"]) if p.get("r_mm") not in (None, "") else None
            z_mm = float(p["z_mm"]) if p.get("z_mm") not in (None, "") else None
        except (TypeError, ValueError):
            raise HTTPException(422, "bad parameter types")

        def work(job):
            job.message = "fitting distributions"
            return _vendor().run_distributions(path, density_kg_m3=density,
                                               r_mm=r_mm, z_mm=z_mm)

        return submit("distributions", video, p, work)

    @app.post("/video/{video}/track_analysis")
    async def track_analysis(video: str, request: Request):
        path = vid_path(video)
        p = await json_body(request)

        def work(job):
            job.message = "analyzing track velocities"
            return _vendor().run_track_analysis(path)

        return submit("track_analysis", video, p, work)

    @app.post("/video/{video}/longitudinal")
    async def longitudinal(video: str, request: Request):
        path = vid_path(video)
        p = await json_body(request)
        logs = store.list_logs(video_basename(path))
        if p.get("log_path"):
            # client picks by name; confine to the logs dir
            log_path = safe_under(LOGS_DIR, os.path.basename(str(p["log_path"])))
        else:
            log_path = logs[0]["path"] if logs else None
        if not log_path or not os.path.isfile(log_path):
            raise HTTPException(422, "no parameter log — run detection first")
        try:
            if store.load_log(log_path).get("video_basename") != video_basename(path):
                raise HTTPException(422, "log belongs to a different video")
        except (OSError, ValueError):
            raise HTTPException(422, "unreadable parameter log")
        try:
            chunk_s = p.get("chunk_duration_s")
            chunk_s = float(chunk_s) if chunk_s not in (None, "") else None
            chunk_f = p.get("chunk_size_frames")
            chunk_f = int(chunk_f) if chunk_f not in (None, "") else None
            density = float(p.get("density_kg_m3", 3520.0))
        except (TypeError, ValueError):
            raise HTTPException(422, "bad parameter types")
        if (chunk_s is None) == (chunk_f is None):
            raise HTTPException(422, "set exactly one of chunk_duration_s / chunk_size_frames")
        start_utc = sidecar_start_utc(path)

        def work(job):
            job.message = "running chunked longitudinal analysis"
            return _vendor().run_longitudinal(
                log_path, path, chunk_size_frames=chunk_f,
                chunk_duration_s=chunk_s, start_utc=start_utc,
                density_kg_m3=density)

        return submit("longitudinal", video, p, work)

    def _calibration_from_results(path: str) -> tuple[float, float]:
        """fps + pixels_per_um the detection stage recorded (else sane defaults)."""
        basename = video_basename(path)
        rj = os.path.join(data_dir_for(path), f"{basename}_test_results.json")
        fps, ppum = 30.0, store.DEFAULT_PIXELS_PER_UM
        if os.path.isfile(rj):
            try:
                with open(rj) as f:
                    ap = json.load(f).get("analysis_parameters", {})
                fps = float(ap.get("fps_override") or ap.get("fps") or fps)
                ppum = float(ap.get("pixels_per_um") or ppum)
            except (OSError, ValueError, TypeError):
                pass
        return fps, ppum

    def sidecar_start_utc(path: str):
        meta = sidecar_meta(path)
        return (meta or {}).get("start_unix")

    # --------------------------------------------------------------- results
    @app.get("/video/{video}/results")
    def results(video: str):
        path = vid_path(video)
        basename = video_basename(path)
        ddir = data_dir_for(path)
        pdir = os.path.join(PLOTS_DIR, basename)

        out = {"basename": basename, "detection": None, "tracking": None,
               "plots": [], "logs": store.list_logs(basename)}
        rj = os.path.join(ddir, f"{basename}_test_results.json")
        if os.path.isfile(rj):
            try:
                with open(rj) as f:
                    out["detection"] = json.load(f)
            except (OSError, ValueError):
                pass
        npz = os.path.join(ddir, f"{basename}_track_data.npz")
        if os.path.isfile(npz):
            try:
                z = np.load(npz, allow_pickle=False)
                meta = json.loads(str(z["metadata"]))
                obs = z["track_observations"]
                meta["n_observations"] = int(len(obs))
                out["tracking"] = meta
            except Exception as e:
                out["tracking"] = {"error": str(e)}
        if os.path.isdir(pdir):
            out["plots"] = sorted(
                f for f in os.listdir(pdir) if f.lower().endswith(".png"))
        longs = sorted(f for f in (os.listdir(ddir) if os.path.isdir(ddir) else [])
                       if f.startswith("longitudinal_analysis_"))
        out["longitudinal_files"] = longs
        return out

    @app.get("/video/{video}/streamlines")
    def streamlines(video: str):
        path = vid_path(video)
        try:
            return store.load_track_polylines(path)
        except FileNotFoundError:
            raise HTTPException(404, "no track data — run tracking first")

    @app.get("/video/{video}/logs")
    def video_logs(video: str):
        return store.list_logs(video_basename(vid_path(video)))

    @app.get("/log")
    def read_log(path: str):
        # only serve files that actually live in the logs dir
        full = safe_under(LOGS_DIR, os.path.basename(path))
        if not os.path.isfile(full):
            raise HTTPException(404)
        try:
            return store.load_log(full)
        except (OSError, ValueError):
            raise HTTPException(422, "unreadable log file")

    @app.get("/plots/{basename}/{fname}")
    def plot_file(basename: str, fname: str):
        # basename('..') is still '..' — safe_under rejects any escape
        path = safe_under(PLOTS_DIR, os.path.basename(basename),
                          os.path.basename(fname))
        if not os.path.isfile(path):
            raise HTTPException(404)
        return FileResponse(path, headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------ jobs
    @app.get("/jobs")
    def all_jobs(video: Optional[str] = None):
        return jobs.list(video)

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404)
        return job.as_dict()

    @app.post("/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        job = jobs.cancel(job_id)
        if job is None:
            raise HTTPException(404)
        return job.as_dict()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="droplab — browser droplet analysis for dock recordings.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--dir", action="append", default=[],
                        help="extra video directories (repeatable); "
                             "recordings/ is always included")
    args = parser.parse_args()
    dirs = [RECORDINGS_DIR] + [os.path.abspath(d) for d in args.dir]
    import uvicorn
    uvicorn.run(create_app(dirs), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
