"""Dock eval web server — stream and control cameras from the browser.

A FastAPI server that runs each :class:`~camera_dock.base.CameraDriver` through a
shared :class:`~camera_dock.engine.AcquisitionEngine`, streams live frames as
MJPEG, and exposes the standardized command set as HTTP endpoints. One server
drives several cameras at once, each under ``/cam/{name}/…``. This is the
standalone eval surface; the parent DAQ mounts the same app under ``/cameras``.

Run it::

    python -m camera_dock.webapp zelux
    python -m camera_dock.webapp zelux:12345 --host 0.0.0.0 --port 8000

Per-camera endpoints (prefix ``/cam/{name}``):

    GET  /stream?e=&hist=      MJPEG (multipart/x-mixed-replace)
    GET  /snapshot             current frame as JPEG
    POST /snapshot/save?fmt=tiff|png|npy
    GET  /info                 identity, sizes, measured fps, ROI, states
    GET  /capabilities         ControlSpec document + geometry + values
    POST /control/{ctl}?value=&normalized=     the generic funnel
    POST /auto_exposure
    POST /roi?x=&y=&w=&h=  ·  POST /roi/reset  ·  POST /binning?bx=&by=
    POST /record/start?max_seconds=&max_frames=  ·  /record/stop  ·  GET /record/status
    POST /timelapse/start?interval=&hours=&dir=&fmt=  ·  /timelapse/stop  ·  GET /timelapse/status
    GET  /presets  ·  POST /presets/save?preset=  ·  POST /presets/load?preset=
    POST /connect  ·  POST /disconnect

Error convention: 404 unknown camera · 409 state conflict (recording, timelapse,
wedged) · 422 control unsupported/invalid · 503 camera unavailable.

Locking (docs/DESIGN.md §4): the session lock serializes control writes,
connect/disconnect, record/timelapse lifecycle, and region changes. The MJPEG /
snapshot paths never touch the driver — they read the engine's latest frame and
session-cached status only.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import threading
from datetime import datetime
from time import perf_counter, sleep
from typing import Optional

import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from . import imaging, presets
from .base import Frame
from .drivers import make_camera
from .engine import AcquisitionEngine
from .paths import data_dir, resolve
from .recorder import HybridRecorder

STREAM_FPS = 30.0
JPEG_QUALITY = 80
_STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def _to_8bit(bit_depth: int):
    shift = max(bit_depth - 8, 0)
    if shift == 0:
        return lambda f: f if f.dtype == np.uint8 else f.astype(np.uint8)
    return lambda f: (f >> shift).astype(np.uint8)


def _slug(info: dict) -> str:
    return str(info.get("model", "camera")).replace(" ", "").replace("/", "-").lower()


class _Conflict(Exception):
    pass


class Timelapse:
    """Interval still-capture on a background thread (one shot every N seconds).

    Samples the engine's latest frame — never touches the camera — so it runs
    happily alongside streaming and recording. Shots are scheduled against the
    start time (t0 + k*interval) so cadence never drifts; skipped slots are
    dropped, not fired as catch-up bursts. The stale guard uses the engine's
    monotonic frame index: a tick with no NEW frame is counted, not saved.
    """

    def __init__(self, session: "CameraSession", interval_s: float,
                 duration_s: float, directory: str, fmt: str) -> None:
        self.session = session
        self.interval_s = float(interval_s)
        self.duration_s = float(duration_s)          # 0 = run until stopped
        self.directory = directory
        self.fmt = fmt
        self.taken = 0
        self.missed = 0
        self.stale = 0
        self._last_index = -1
        self.last_path = ""
        self.error = ""
        try:
            self._slug = _slug(session.camera.device_info)
        except Exception:
            self._slug = session.name
        self.t0 = perf_counter()
        self.started_wall = datetime.now()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="timelapse", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        k = 0
        while True:
            due = self.t0 + k * self.interval_s
            if self.duration_s > 0 and due - self.t0 > self.duration_s:
                break
            if self._stop.wait(max(0.0, due - perf_counter())):
                break
            frame, index = self.session.engine.latest()
            if frame is None:
                self.missed += 1
            elif index == self._last_index:
                self.stale += 1
            else:
                base = datetime.now().strftime(f"{self._slug}_%Y%m%d_%H%M%S_%f")
                try:
                    self.last_path = imaging.save_snapshot(
                        frame.data, os.path.join(self.directory, base), self.fmt)
                    self.taken += 1
                    self._last_index = index
                    self.error = ""
                except Exception as exc:
                    self.error = str(exc)
                    self.missed += 1
            k = max(k + 1, int((perf_counter() - self.t0) / self.interval_s))

    def signal_stop(self) -> None:
        self._stop.set()

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout=timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def status(self) -> dict:
        elapsed = perf_counter() - self.t0
        remaining = max(0.0, self.duration_s - elapsed) if self.duration_s > 0 else None
        next_in = self.interval_s - (elapsed % self.interval_s) if self.running else None
        return {"running": self.running, "interval_s": self.interval_s,
                "duration_s": self.duration_s, "elapsed_s": round(elapsed, 1),
                "remaining_s": round(remaining, 1) if remaining is not None else None,
                "next_in_s": round(next_in, 1) if next_in is not None else None,
                "taken": self.taken, "missed": self.missed, "stale": self.stale,
                "dir": self.directory, "fmt": self.fmt, "last_path": self.last_path,
                "error": self.error,
                "started": self.started_wall.strftime("%Y-%m-%d %H:%M:%S")}


class CameraSession:
    """One camera + engine + recording/timelapse state, behind the web API.

    Recording state machine: idle -> recording -> encoding -> idle. All
    transitions under ``self.lock``; the encode runs on a dedicated non-daemon
    thread WITHOUT the lock (it never touches the camera).
    """

    def __init__(self, name: str, camera) -> None:
        self.name = name
        self.camera = camera
        self.engine = AcquisitionEngine(camera)
        self.to8 = _to_8bit(8)
        self.bit_depth = 8
        self.has_roi = False
        self.ok = False
        self.error = ""
        self.lock = threading.Lock()

        self.rec_state = "idle"                  # idle | recording | encoding
        self.recorder: Optional[HybridRecorder] = None
        self.rec_path = ""
        self.rec_stats: Optional[dict] = None
        self._encode_thread: Optional[threading.Thread] = None

        self.timelapse: Optional[Timelapse] = None

        # Caches for the render path (never call the driver per streamed frame).
        self.expected_shape: Optional[tuple] = None   # (h, w) of current geometry
        self.status_exposure_us: Optional[float] = None
        self._sat_max = 255

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Connect + start streaming. Idempotent; never raises (ok/error)."""
        with self.lock:
            if self.ok:
                return
            try:
                self.camera.connect()
                self.bit_depth = int(getattr(self.camera, "bit_depth", 8))
                self.to8 = _to_8bit(self.bit_depth)
                self._sat_max = imaging.max_value_for(self.bit_depth)
                try:
                    self.camera.roi_range()
                    self.has_roi = True
                except Exception:
                    self.has_roi = False
                self._apply_defaults()
                # Apply a saved "default" preset before the engine starts, so an
                # ROI in the preset applies without a region-change cycle.
                try:
                    if presets.exists(self.name, "default"):
                        presets.apply(self.camera, presets.load(self.name, "default"))
                except Exception:
                    pass
                self.engine.start()
                self._refresh_caches()
                self.ok = True
                self.error = ""
            except Exception as exc:      # one bad camera never sinks the app
                self.ok, self.error = False, str(exc)

    def _apply_defaults(self) -> None:
        """Sane streaming defaults. fps first: exposure duty depends on it."""
        try:
            _, fh = self.camera.frame_rate_range()
            if fh > 0:
                self.camera.set_frame_rate(min(30.0, fh))
        except Exception:
            pass
        try:
            _, eh = self.camera.exposure_range()
            if eh > 0:
                self.camera.set_exposure(min(5000.0, eh))
        except Exception:
            pass

    def _refresh_caches(self) -> None:
        try:
            w, h = self.camera.image_size()
            self.expected_shape = (h, w)
        except Exception:
            self.expected_shape = None
        v = self.camera.get("exposure")
        self.status_exposure_us = v.value if v.ok else None

    def stop(self) -> None:
        """Release the camera (engine + disconnect). Joins a running encode."""
        with self.lock:
            self._stop_locked()
        t = self._encode_thread
        if t is not None:
            t.join(timeout=600.0)         # finalize the file (non-daemon backstop)

    def _stop_locked(self) -> None:
        if not self.ok:
            return
        try:
            tl = self.timelapse
            if tl is not None:
                tl.signal_stop()
                self.timelapse = None
            self.engine.set_sink(None)
            if self.rec_state == "recording":
                self.rec_state = "idle"    # frames stop; no encode on teardown
                self.recorder = None
            self.engine.stop()
            self.camera.disconnect()
        finally:
            self.ok = False

    # --- render path (no driver calls — session caches only) ----------------
    def render(self, frame: Frame, *, status: bool, hist: bool = False) -> np.ndarray:
        data = frame.data
        bgr = cv2.cvtColor(self.to8(data), cv2.COLOR_GRAY2BGR)
        if status:
            exp = self.status_exposure_us
            sat = imaging.saturation_fraction(data, self._sat_max)
            txt = f"acq {self.engine.acquisition_fps:5.1f} fps"
            if exp is not None:
                txt += f"   exp {exp/1000.0:.2f} ms"
            if sat > 0.001:
                txt += f"   sat {sat*100:.1f}%"
            cv2.putText(bgr, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
            state = self.rec_state
            if state != "idle":
                rec = self.recorder
                label = (f"REC {rec.captured}" if state == "recording" and rec is not None
                         else "ENCODING")
                cv2.putText(bgr, label, (bgr.shape[1] - 190, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        if hist:
            imaging.histogram_overlay(bgr, self.to8(data))
        imaging.draw_timestamp(bgr, imaging.eastern_now())
        return bgr

    def encode(self, frame: Frame, *, status: bool, hist: bool = False) -> bytes:
        ok, jpg = cv2.imencode(".jpg", self.render(frame, status=status, hist=hist),
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return jpg.tobytes() if ok else b""

    def fresh_frame(self) -> Optional[Frame]:
        """Latest frame matching the current geometry (stale shapes skipped)."""
        frame, _ = self.engine.latest()
        if frame is None:
            return None
        if self.expected_shape is not None and frame.data.shape != self.expected_shape:
            return None
        return frame

    def grab_fresh(self):
        """Wait for a genuinely NEW native frame (for auto-exposure)."""
        _, start = self.engine.latest()
        t0 = perf_counter()
        while perf_counter() - t0 < 2.0:
            f, i = self.engine.latest()
            if f is not None and i >= start + 4:
                return f.data
            sleep(0.005)
        f, _ = self.engine.latest()
        return f.data if f is not None else None

    # --- info / capabilities ------------------------------------------------
    def info(self) -> dict:
        d = {"name": self.name, "ok": self.ok, "error": self.error,
             "rec_state": self.rec_state,
             "timelapse_running": self.timelapse.running if self.timelapse else False,
             "acquisition_fps": round(self.engine.acquisition_fps, 1),
             "acquisition_errors": self.engine.error_count}
        if self.ok:
            try:
                di = self.camera.device_info
                w, h = self.camera.image_size()
                d.update({"model": di.get("model"), "serial": di.get("serial"),
                          "width": w, "height": h})
                if self.has_roi:
                    d["roi"] = list(self.camera.get_roi())
                v = self.camera.get("exposure")
                if v.ok:
                    self.status_exposure_us = v.value
            except Exception as exc:
                d["error"] = str(exc)
        return d

    def capabilities(self) -> dict:
        caps = {"controls": self.camera.capabilities(),
                "bit_depth": self.bit_depth,
                "has_roi": self.has_roi}
        try:
            caps["sensor_size"] = list(self.camera.sensor_size())
            caps["image_size"] = list(self.camera.image_size())
        except Exception:
            pass
        if self.has_roi:
            try:
                caps["roi"] = list(self.camera.get_roi())
                caps["roi_range"] = self.camera.roi_range()
            except Exception:
                caps["has_roi"] = False
        try:
            caps["binning"] = list(self.camera.get_binning())
            caps["binning_range"] = list(self.camera.binning_range())
        except Exception:
            pass
        return caps

    # --- region changes (stop -> geometry -> restart), DESIGN §2.2 ----------
    def change_region(self, action) -> dict:
        with self.lock:
            if not self.ok:
                raise _Conflict("camera not connected")
            if self.rec_state != "idle":
                raise _Conflict("stop recording before changing the region")
            if self.timelapse is not None and self.timelapse.running:
                raise _Conflict("stop the timelapse before changing the region")
            if not self.engine.stop():
                self.error = "acquisition thread wedged in the SDK"
                raise RuntimeError(self.error)   # -> 503; never touch geometry
            try:
                action()                  # driver applies + clamps + reads back
            finally:
                # Restart even if the geometry call failed — driver.start()
                # re-arms and re-applies last-commanded fps/exposure against
                # fresh ranges.
                self.engine.start()
            self._refresh_caches()
            out = {"capabilities": self.capabilities()}
            out["roi"] = out["capabilities"].get("roi")
            out["image_size"] = out["capabilities"].get("image_size")
            return out

    # --- recording -----------------------------------------------------------
    def record_start(self, max_seconds: float = 0.0, max_frames: int = 0) -> dict:
        with self.lock:
            if not self.ok:
                raise _Conflict("camera not connected")
            if self.rec_state != "idle":
                raise _Conflict(f"recorder is {self.rec_state}")
            rec_dir = data_dir("recordings")
            path = os.path.join(rec_dir, datetime.now().strftime(
                f"{_slug(self.camera.device_info)}_%Y%m%d_%H%M%S_%f.avi"))
            rec = HybridRecorder(clock=imaging.eastern_now,
                                 max_seconds=max_seconds, max_frames=max_frames)
            try:
                self.recorder = rec
                self.rec_path = path
                self.rec_stats = None
                self.rec_state = "recording"
                self.engine.set_sink(rec.submit)      # LAST — after state is set
            except Exception:
                self.engine.set_sink(None)
                self.recorder, self.rec_state = None, "idle"
                raise
            return {"rec_state": "recording", "path": path}

    def record_stop(self) -> dict:
        with self.lock:
            if self.rec_state != "recording":
                raise _Conflict(f"recorder is {self.rec_state}")
            self.engine.set_sink(None)                # barrier: no submit in flight
            rec, path = self.recorder, self.rec_path
            self.recorder = None
            self.rec_state = "encoding"
            fps = self.engine.acquisition_fps or (self.camera.get("fps").value or 30.0)
            try:
                cam_info = self.camera.device_info
            except Exception:
                cam_info = {"name": self.name}
            t = threading.Thread(target=self._encode_worker,
                                 args=(rec, path, float(fps), cam_info),
                                 name="encode", daemon=False)
            self._encode_thread = t
            t.start()
        return {"rec_state": "encoding", "path": path}

    def _encode_worker(self, rec: HybridRecorder, path: str, fps: float,
                       cam_info: dict) -> None:
        try:
            stats = rec.stop_and_encode(path, fps, self.to8,
                                        stamp=imaging.draw_timestamp, camera=cam_info)
        except Exception as exc:
            stats = {"ok": False, "error": str(exc), "path": None}
        with self.lock:
            self.rec_stats = stats
            self.rec_state = "idle"
            self._encode_thread = None

    def record_status(self) -> dict:
        with self.lock:
            d = {"rec_state": self.rec_state, "path": self.rec_path or None}
            rec = self.recorder
            if self.rec_state == "recording" and rec is not None:
                d.update({"captured": rec.captured,
                          "elapsed_s": round(rec.elapsed_s, 1),
                          "limit_reached": rec.limit_reached})
            if self.rec_stats is not None:
                d["stats"] = self.rec_stats
            return d

    # --- timelapse -----------------------------------------------------------
    def timelapse_start(self, interval_s: float, hours: float,
                        directory: str, fmt: str) -> dict:
        with self.lock:
            if not self.ok:
                raise _Conflict("camera not connected")
            if self.timelapse is not None and self.timelapse.running:
                st = self.timelapse.status()
                st["already_running"] = True
                return st
            interval_s = float(interval_s)
            if interval_s < 1.0:
                raise ValueError("interval must be at least 1 second")
            hours = float(hours)
            if hours < 0:
                raise ValueError("hours must be >= 0 (0 = run until stopped)")
            fmt = (fmt or "tiff").lower()
            if fmt not in imaging.SNAPSHOT_FORMATS:
                raise ValueError(f"format must be one of {', '.join(imaging.SNAPSHOT_FORMATS)}")
            directory = (directory or "").strip() or datetime.now().strftime(
                f"captures/timelapse_{_slug(self.camera.device_info)}_%Y%m%d_%H%M%S")
            directory = resolve(directory)
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as exc:
                raise ValueError(f"cannot create save directory: {exc}")
            self.timelapse = Timelapse(self, interval_s, hours * 3600.0, directory, fmt)
            return self.timelapse.status()

    def timelapse_stop(self) -> dict:
        with self.lock:
            tl = self.timelapse
            if tl is None:
                return {"running": False}
            tl.signal_stop()
        tl.join()                      # join OUTSIDE the lock (slow disk safe)
        return tl.status()

    def timelapse_status(self) -> dict:
        tl = self.timelapse
        return tl.status() if tl is not None else {"running": False}

    # --- presets --------------------------------------------------------------
    def save_preset(self, name: str) -> dict:
        path = presets.save(self.name, name,
                            presets.capture(self.camera, has_roi=self.has_roi))
        return {"saved": name, "path": path}

    def load_preset(self, name: str) -> dict:
        settings = presets.load(self.name, name)
        with self.lock:
            if not self.ok:
                raise _Conflict("camera not connected")
            presets.apply(self.camera, settings, set_roi=False)
        if self.has_roi and settings.get("roi"):
            x, y, w, h = (int(v) for v in settings["roi"])
            self.change_region(lambda: self.camera.set_roi(x, y, w, h))
        else:
            self._refresh_caches()
        return {"loaded": name, "settings": settings}

    def list_presets(self) -> list:
        return presets.list_presets(self.name)


# --------------------------------------------------------------------------- #
# app assembly
# --------------------------------------------------------------------------- #

def start_all(sessions: dict) -> None:
    """Connect + start every session. NEVER raises (per-session isolation)."""
    for s in sessions.values():
        try:
            s.start()
        except Exception as exc:          # belt and braces — start() shouldn't raise
            s.ok, s.error = False, str(exc)


def stop_all(sessions: dict) -> None:
    """Stop every session; one bad disconnect never skips the rest."""
    for s in sessions.values():
        try:
            s.stop()
        except Exception as exc:
            s.error = str(exc)


def create_app(sessions: dict, *, manage_lifecycle: bool = True):
    """Build the FastAPI app for ``{name: CameraSession}``.

    ``manage_lifecycle=False`` omits the startup/shutdown lifespan so a parent
    app (xsphere_daq.panel) can own start_all/stop_all itself. All pages derive
    their API base from window.location, so mounting under any prefix works.
    """
    @asynccontextmanager
    async def lifespan(app):
        start_all(sessions)
        try:
            yield
        finally:
            stop_all(sessions)

    app = FastAPI(lifespan=lifespan) if manage_lifecycle else FastAPI()

    def sess(name: str, *, need_ok: bool = True) -> CameraSession:
        s = sessions.get(name)
        if s is None:
            raise HTTPException(404, f"no camera '{name}'")
        if need_ok and not s.ok:
            raise HTTPException(503, f"camera '{name}' unavailable: {s.error}")
        return s

    # --- pages / static ---
    @app.get("/")
    def index():
        return FileResponse(os.path.join(_STATIC, "index.html"))

    @app.get("/static/{fname}")
    def static_file(fname: str):
        path = os.path.join(_STATIC, os.path.basename(fname))
        if not os.path.isfile(path):
            raise HTTPException(404)
        return FileResponse(path)

    @app.get("/favicon.ico")
    def favicon():
        return FileResponse(os.path.join(_STATIC, "xsphere.ico"))

    @app.get("/cam/{name}")
    def cam_page(name: str):
        if name not in sessions:
            raise HTTPException(404)
        return FileResponse(os.path.join(_STATIC, "cam.html"))

    @app.get("/cameras")
    def cameras():
        return [s.info() for s in sessions.values()]

    # --- stream / snapshot ---
    @app.get("/cam/{name}/stream")
    def stream(name: str, e: int = 0, hist: int = 0):
        s = sess(name)

        async def mjpeg():
            period = 1.0 / STREAM_FPS
            while s.ok:
                frame = s.fresh_frame()
                if frame is not None:
                    jpg = await asyncio.to_thread(s.encode, frame, status=True,
                                                  hist=bool(hist))
                    if jpg:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                               + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(period)

        return StreamingResponse(mjpeg(),
                                 media_type="multipart/x-mixed-replace; boundary=frame",
                                 headers={"Cache-Control": "no-store"})

    @app.get("/cam/{name}/snapshot")
    def snapshot(name: str):
        s = sess(name)
        frame = s.fresh_frame()
        if frame is None:
            raise HTTPException(503, "no frame yet")
        return Response(content=s.encode(frame, status=False), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.post("/cam/{name}/snapshot/save")
    def snapshot_save(name: str, fmt: str = "tiff"):
        s = sess(name)
        frame = s.fresh_frame()
        if frame is None:
            raise HTTPException(503, "no frame yet")
        base = datetime.now().strftime(f"{_slug(s.camera.device_info)}_%Y%m%d_%H%M%S_%f")
        return {"path": imaging.save_snapshot(
            frame.data, os.path.join(data_dir("captures"), base), fmt)}

    # --- info / capabilities / controls ---
    @app.get("/cam/{name}/info")
    def info(name: str):
        return sess(name, need_ok=False).info()

    @app.get("/cam/{name}/capabilities")
    def capabilities(name: str):
        return sess(name).capabilities()

    @app.post("/cam/{name}/control/{ctl}")
    def set_control(name: str, ctl: str, value: float, normalized: bool = False):
        s = sess(name)
        with s.lock:
            if not s.ok:
                raise HTTPException(503, "camera disconnected")
            result = s.camera.set(ctl, value, normalized=normalized)
        if not result.supported:
            raise HTTPException(422, result.error)
        if not result.ok:
            raise HTTPException(422, result.error)
        if ctl == "exposure" and result.value is not None:
            s.status_exposure_us = result.value
        se = result.side_effects.get("exposure")
        if se and se.get("value") is not None:
            s.status_exposure_us = se["value"]
        return {"control": ctl, **result.as_dict()}

    @app.get("/cam/{name}/control/{ctl}")
    def get_control(name: str, ctl: str):
        result = sess(name).camera.get(ctl)
        if not result.supported:
            raise HTTPException(422, result.error)
        return {"control": ctl, **result.as_dict()}

    @app.post("/cam/{name}/auto_exposure")
    def auto_exposure(name: str):
        s = sess(name)
        if s.rec_state != "idle":
            raise HTTPException(409, "stop recording before auto-exposure")
        exp = imaging.auto_expose(s.camera, s.grab_fresh)
        s.status_exposure_us = exp
        return {"exposure": exp}

    # --- geometry ---
    def _region(s: CameraSession, action):
        try:
            return s.change_region(action)
        except _Conflict as exc:
            raise HTTPException(409, str(exc))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))
        except Exception as exc:
            raise HTTPException(422, str(exc))

    @app.post("/cam/{name}/roi")
    def set_roi(name: str, x: int, y: int, w: int, h: int):
        s = sess(name)
        if not s.has_roi:
            raise HTTPException(422, "ROI not supported")
        return _region(s, lambda: s.camera.set_roi(x, y, w, h))

    @app.post("/cam/{name}/roi/reset")
    def reset_roi(name: str):
        s = sess(name)
        if not s.has_roi:
            raise HTTPException(422, "ROI not supported")
        return _region(s, s.camera.reset_roi)

    @app.post("/cam/{name}/binning")
    def set_binning(name: str, bx: int, by: int):
        s = sess(name)
        return _region(s, lambda: s.camera.set_binning(bx, by))

    # --- recording ---
    @app.post("/cam/{name}/record/start")
    def record_start(name: str, max_seconds: float = 0.0, max_frames: int = 0):
        try:
            return sess(name).record_start(max_seconds, max_frames)
        except _Conflict as exc:
            raise HTTPException(409, str(exc))

    @app.post("/cam/{name}/record/stop")
    def record_stop(name: str):
        try:
            return sess(name, need_ok=False).record_stop()
        except _Conflict as exc:
            raise HTTPException(409, str(exc))

    @app.get("/cam/{name}/record/status")
    def record_status(name: str):
        return sess(name, need_ok=False).record_status()

    # --- timelapse ---
    @app.post("/cam/{name}/timelapse/start")
    def timelapse_start(name: str, interval: float = 30.0, hours: float = 0.0,
                        dir: str = "", fmt: str = "tiff"):
        s = sess(name)
        try:
            return s.timelapse_start(interval, hours, dir, fmt)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        except _Conflict as exc:
            raise HTTPException(409, str(exc))

    @app.post("/cam/{name}/timelapse/stop")
    def timelapse_stop(name: str):
        return sess(name, need_ok=False).timelapse_stop()

    @app.get("/cam/{name}/timelapse/status")
    def timelapse_status(name: str):
        return sess(name, need_ok=False).timelapse_status()

    # --- presets ---
    @app.get("/cam/{name}/presets")
    def list_presets(name: str):
        return sess(name).list_presets()

    @app.post("/cam/{name}/presets/save")
    def save_preset(name: str, preset: str = "default"):
        return sess(name).save_preset(preset)

    @app.post("/cam/{name}/presets/load")
    def load_preset(name: str, preset: str = "default"):
        s = sess(name)
        try:
            return s.load_preset(preset)
        except FileNotFoundError:
            raise HTTPException(404, f"no preset '{preset}'")
        except _Conflict as exc:
            raise HTTPException(409, str(exc))

    # --- connect / disconnect ---
    @app.post("/cam/{name}/connect")
    def cam_connect(name: str):
        s = sess(name, need_ok=False)
        s.start()
        if not s.ok:
            raise HTTPException(503, s.error or "failed to connect")
        return {"connected": True}

    @app.post("/cam/{name}/disconnect")
    def cam_disconnect(name: str):
        s = sess(name, need_ok=False)
        with s.lock:
            if s.rec_state == "recording":
                raise HTTPException(409, "stop recording before disconnecting")
            if s.timelapse is not None and s.timelapse.running:
                raise HTTPException(409, "stop the timelapse before disconnecting")
            s._stop_locked()
        return {"connected": False}

    return app


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def _make_camera(token: str):
    """Back-compat alias for the parent panel; see drivers.make_camera."""
    return make_camera(token)


def serve(sessions: dict, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(create_app(sessions), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and control cameras over the web.")
    parser.add_argument("cameras", nargs="+",
                        help="cameras to serve: a driver token, or token:SERIAL to "
                             "pick a specific unit (e.g. zelux, zelux:12345). "
                             "Registered drivers: zelux. NOTE: the Thorlabs TSI SDK "
                             "is a per-process singleton — serve each zelux in its "
                             "own process on its own --port.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    sessions = {}
    for tok in args.cameras:
        key = tok.replace(":", "-")          # URL-safe /cam/{key} namespace
        if key in sessions:
            raise SystemExit(f"duplicate camera '{tok}' — each must be distinct")
        sessions[key] = CameraSession(key, make_camera(tok))
    serve(sessions, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
