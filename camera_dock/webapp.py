"""Dock web app — stream and control one or more cameras from the browser.

A FastAPI server that runs each :class:`~camera_dock.base.CameraBase` camera through
its own shared :class:`~camera_dock.engine.AcquisitionEngine`, streams the live
frames as MJPEG, and exposes the camera's controls (exposure, gain, frame rate, ROI,
auto-exposure, recording) as endpoints. One server can drive several cameras at once
— each gets its own namespace ``/cam/{name}/...`` — all over the same `CameraBase`
surface the test GUIs use, so nothing is duplicated.

Run it::

    python -m camera_dock.webapp basler                 # one camera
    python -m camera_dock.webapp basler zelux --host 0.0.0.0 --port 8000

Then open http://<host>:<port>/ — an overview of every camera's live stream, each
linking to its full control page at /cam/<name>.

Per-camera endpoints (prefix ``/cam/{name}``):

    GET  /stream      MJPEG (multipart/x-mixed-replace)
    GET  /snapshot    current frame as JPEG
    POST /snapshot/save?fmt=tiff|png|npy
    GET  /info        JSON: model, serial, size, acquisition fps, ok
    GET  /controls    JSON: values, ranges, capabilities, rec state
    POST /controls/exposure|gain|fps?value=
    POST /controls/auto_exposure
    POST /controls/roi?x=&y=&w=&h=   ·   /controls/roi/reset   (blocked while recording)
    POST /record/start   ·   /record/stop
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import threading
from datetime import datetime
from time import perf_counter, sleep

import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

from . import imaging, presets
from .engine import AcquisitionEngine
from .recorder import HybridRecorder

STREAM_FPS = 30.0
JPEG_QUALITY = 80


def _to_8bit(bit_depth: int):
    shift = max(bit_depth - 8, 0)
    if shift == 0:
        return lambda f: f if f.dtype == np.uint8 else f.astype(np.uint8)
    return lambda f: (f >> shift).astype(np.uint8)


def _slug(info: dict) -> str:
    return str(info.get("model", "camera")).replace(" ", "").replace("/", "-").lower()


class CameraSession:
    """One camera + its acquisition engine + recording state, behind the web API."""

    def __init__(self, name: str, camera) -> None:
        self.name = name
        self.camera = camera
        self.engine = AcquisitionEngine(camera)
        self.to8 = _to_8bit(8)
        self.bit_depth = 8
        self.has_roi = False
        self.ok = False
        self.error = ""
        self.recorder: HybridRecorder | None = None
        self.rec_path = ""
        self.lock = threading.Lock()

    # --- lifecycle (also driven at runtime by /connect and /disconnect) ---
    def start(self) -> None:
        if self.ok:
            return
        try:
            self.camera.connect()
            self.bit_depth = int(getattr(self.camera, "bit_depth", 8))
            self.to8 = _to_8bit(self.bit_depth)
            try:
                self.camera.roi_range()
                self.has_roi = True
            except Exception:
                self.has_roi = False
            self._apply_defaults()
            # Bring the rig up configured: apply a saved "default" preset if present
            # (camera not grabbing yet, so ROI can be set directly).
            try:
                if presets.exists(self.name, "default"):
                    presets.apply(self.camera, presets.load(self.name, "default"))
            except Exception:
                pass
            self.engine.start()
            self.ok = True
        except Exception as exc:               # one bad camera shouldn't sink the app
            self.ok, self.error = False, str(exc)

    def _apply_defaults(self) -> None:
        """Sane out-of-the-box streaming settings (mirrors the preview GUI) so a
        camera streams at a usable rate even though nothing has been set yet.
        Frame rate first: the exposure range can depend on the frame period."""
        try:
            _, fh = self.camera.frame_rate_range()
            if fh > 0:
                self.camera.set_frame_rate(min(30.0, fh))
        except Exception:
            pass
        try:
            _, eh = self.camera.exposure_range()
            self.camera.set_exposure(min(5000.0, eh))
        except Exception:
            pass

    def stop(self) -> None:
        """Release the camera (stop engine + disconnect) — frees it for other clients."""
        if not self.ok:
            return
        try:
            self.engine.set_sink(None)
            self.recorder = None
            self.engine.stop()
            self.camera.disconnect()
        finally:
            self.ok = False

    # --- frames ---
    def render(self, frame: np.ndarray, *, status: bool) -> np.ndarray:
        bgr = cv2.cvtColor(self.to8(frame), cv2.COLOR_GRAY2BGR)
        if status:
            cv2.putText(bgr, f"acq {self.engine.acquisition_fps:5.1f} fps   "
                             f"exp {self.camera.get_exposure():.0f}us", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
            if self.recorder is not None:
                cv2.putText(bgr, f"REC {self.recorder._captured}", (bgr.shape[1] - 150, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        imaging.draw_timestamp(bgr, imaging.eastern_now())
        return bgr

    def encode(self, frame: np.ndarray, *, status: bool) -> bytes:
        ok, jpg = cv2.imencode(".jpg", self.render(frame, status=status),
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return jpg.tobytes() if ok else b""

    def grab_fresh(self):
        _, start = self.engine.latest()
        t0 = perf_counter()
        while perf_counter() - t0 < 2.0:
            f, i = self.engine.latest()
            if f is not None and i >= start + 4:
                return f
            sleep(0.005)
        return self.engine.latest()[0]

    # --- info / controls ---
    def info(self) -> dict:
        di = self.camera.device_info if self.ok else {}
        w, h = self.camera.sensor_size() if self.ok else (0, 0)
        return {"name": self.name, "ok": self.ok, "error": self.error,
                "model": di.get("model"), "serial": di.get("serial"),
                "width": w, "height": h,
                "acquisition_fps": round(self.engine.acquisition_fps, 1)}

    def controls(self) -> dict:
        c = self.camera
        gl, gh = c.gain_range()
        el, eh = c.exposure_range()
        fl, fh = c.frame_rate_range()
        return {"exposure": c.get_exposure(), "exposure_range": [el, eh],
                "has_exposure": eh > el,
                "gain": c.get_gain(), "gain_range": [gl, gh], "has_gain": gh > gl,
                "fps": c.get_frame_rate(), "fps_range": [fl, fh], "has_fps": fh > 0,
                "roi": list(c.get_roi()) if self.has_roi else None,
                "has_roi": self.has_roi, "sensor": list(c.sensor_size()),
                "bit_depth": self.bit_depth, "recording": self.recorder is not None,
                "acquisition_fps": round(self.engine.acquisition_fps, 1)}

    # --- region change (stop -> set -> restart), blocked while recording ---
    def _change_region(self, action) -> dict:
        with self.lock:
            if self.recorder is not None:
                raise _Conflict("stop recording before changing ROI")
            self.engine.stop()
            try:
                action()
            finally:
                self.engine.start()
            return {"roi": list(self.camera.get_roi()), "sensor": list(self.camera.sensor_size())}

    # --- recording ---
    def record_start(self) -> dict:
        with self.lock:
            if self.recorder is not None:
                return {"recording": True}
            rec = HybridRecorder(clock=imaging.eastern_now)
            self.engine.set_sink(rec.submit)
            os.makedirs("recordings", exist_ok=True)
            self.rec_path = datetime.now().strftime(
                f"recordings/{_slug(self.camera.device_info)}_%Y%m%d_%H%M%S.avi")
            self.recorder = rec
            return {"recording": True, "path": self.rec_path}

    def record_stop(self) -> dict:
        with self.lock:
            rec = self.recorder
            if rec is None:
                return {"recording": False}
            self.engine.set_sink(None)
            self.recorder = None
            path = self.rec_path
            fps = self.engine.acquisition_fps or self.camera.get_frame_rate() or 30.0
        stats = rec.stop_and_encode(path, fps, self.to8, stamp=imaging.draw_timestamp)
        return {"recording": False, "stats": stats}

    # --- presets ---
    def save_preset(self, name: str) -> dict:
        path = presets.save(self.name, name, presets.capture(self.camera, has_roi=self.has_roi))
        return {"saved": name, "path": path}

    def load_preset(self, name: str) -> dict:
        settings = presets.load(self.name, name)
        presets.apply(self.camera, settings, set_roi=False)   # exposure/gain/fps/binning live
        if self.has_roi and settings.get("roi"):              # ROI needs a region change
            x, y, w, h = (int(v) for v in settings["roi"])
            self._change_region(lambda: self.camera.set_roi(x, y, w, h))
        return {"loaded": name, "settings": settings}

    def list_presets(self) -> list:
        return presets.list_presets(self.name)


class _Conflict(Exception):
    pass


def start_all(sessions: dict) -> None:
    """Connect + start every session (for a parent app that manages the lifecycle)."""
    for s in sessions.values():
        s.start()


def stop_all(sessions: dict) -> None:
    for s in sessions.values():
        s.stop()


def create_app(sessions: dict, *, manage_lifecycle: bool = True):
    """Build the FastAPI app managing all camera sessions (``{name: CameraSession}``).

    ``manage_lifecycle=False`` skips the startup/shutdown lifespan so the app can be
    mounted as a sub-application of a parent that starts/stops the sessions itself
    (see :mod:`xsphere_daq.panel`). HTML links honour the mount prefix (root_path),
    so it works both standalone and mounted under e.g. ``/cameras``.
    """
    @asynccontextmanager
    async def lifespan(app):
        start_all(sessions)
        try:
            yield
        finally:
            stop_all(sessions)

    app = FastAPI(lifespan=lifespan) if manage_lifecycle else FastAPI()

    def sess(name: str) -> CameraSession:
        s = sessions.get(name)
        if s is None:
            raise HTTPException(404, f"no camera '{name}'")
        if not s.ok:
            raise HTTPException(503, f"camera '{name}' unavailable: {s.error}")
        return s

    # --- overview ---
    @app.get("/")
    def index(request: Request):
        p = request.scope.get("root_path", "")          # mount prefix, if any
        cards = "".join(
            f'<div class="card"><a href="{p}/cam/{n}"><img src="{p}/cam/{n}/stream"></a>'
            f'<div class="cap">{n}{"" if s.ok else " (unavailable)"}</div></div>'
            for n, s in sessions.items())
        return HTMLResponse(_OVERVIEW.format(cards=cards))

    @app.get("/cameras")
    def cameras():
        return [s.info() for s in sessions.values()]

    @app.get("/cam/{name}")
    def cam_page(request: Request, name: str):
        s = sessions.get(name)
        if s is None:
            raise HTTPException(404)
        title = (s.camera.device_info.get("model", name) if s.ok else name)
        root = request.scope.get("root_path", "")
        return HTMLResponse(_CONTROL.replace("__BASE__", root + f"/cam/{name}")
                            .replace("__ROOT__", root).replace("__TITLE__", title))

    # --- per-camera stream / snapshot ---
    @app.get("/cam/{name}/stream")
    def stream(name: str):
        s = sess(name)

        async def mjpeg():
            period = 1.0 / STREAM_FPS
            while True:
                frame, _ = s.engine.latest()
                if frame is not None:
                    # JPEG encode is CPU-bound; run it off the event loop so
                    # concurrent streams (multi-camera) interleave fairly.
                    jpg = await asyncio.to_thread(s.encode, frame, status=True)
                    if jpg:
                        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                               + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                await asyncio.sleep(period)

        return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/cam/{name}/snapshot")
    def snapshot(name: str):
        s = sess(name)
        frame, _ = s.engine.latest()
        if frame is None:
            raise HTTPException(503, "no frame yet")
        return Response(content=s.encode(frame, status=False), media_type="image/jpeg")

    @app.post("/cam/{name}/snapshot/save")
    def snapshot_save(name: str, fmt: str = "tiff"):
        s = sess(name)
        frame, _ = s.engine.latest()
        if frame is None:
            raise HTTPException(503, "no frame yet")
        os.makedirs("captures", exist_ok=True)
        base = datetime.now().strftime(f"captures/{_slug(s.camera.device_info)}_%Y%m%d_%H%M%S_%f")
        return {"path": imaging.save_snapshot(frame, base, fmt)}

    @app.get("/cam/{name}/info")
    def info(name: str):
        s = sessions.get(name)
        if s is None:
            raise HTTPException(404)
        return s.info()

    @app.post("/cam/{name}/connect")
    def cam_connect(name: str):
        s = sessions.get(name)
        if s is None:
            raise HTTPException(404)
        s.start()                       # idempotent: no-op if already connected
        if not s.ok:
            raise HTTPException(503, s.error or "failed to connect")
        return {"connected": True}

    @app.post("/cam/{name}/disconnect")
    def cam_disconnect(name: str):
        s = sessions.get(name)
        if s is None:
            raise HTTPException(404)
        if s.recorder is not None:
            raise HTTPException(409, "stop recording before disconnecting")
        s.stop()                        # releases the camera for other clients
        return {"connected": False}

    # --- per-camera controls ---
    @app.get("/cam/{name}/controls")
    def controls(name: str):
        return sess(name).controls()

    @app.post("/cam/{name}/controls/exposure")
    def set_exposure(name: str, value: float):
        c = sess(name).camera
        c.set_exposure(value)
        return {"exposure": c.get_exposure()}

    @app.post("/cam/{name}/controls/gain")
    def set_gain(name: str, value: float):
        c = sess(name).camera
        c.set_gain(value)
        return {"gain": c.get_gain()}

    @app.post("/cam/{name}/controls/fps")
    def set_fps(name: str, value: float):
        c = sess(name).camera
        c.set_frame_rate(value)
        return {"fps": c.get_frame_rate()}

    @app.post("/cam/{name}/controls/auto_exposure")
    def auto_exposure(name: str):
        s = sess(name)
        return {"exposure": imaging.auto_expose(s.camera, s.grab_fresh)}

    @app.post("/cam/{name}/controls/roi")
    def set_roi(name: str, x: int, y: int, w: int, h: int):
        s = sess(name)
        if not s.has_roi:
            raise HTTPException(400, "ROI not supported")
        try:
            return s._change_region(lambda: s.camera.set_roi(x, y, w, h))
        except _Conflict as exc:
            raise HTTPException(409, str(exc))

    @app.post("/cam/{name}/controls/roi/reset")
    def reset_roi(name: str):
        s = sess(name)
        if not s.has_roi:
            raise HTTPException(400, "ROI not supported")
        try:
            return s._change_region(s.camera.reset_roi)
        except _Conflict as exc:
            raise HTTPException(409, str(exc))

    # --- per-camera recording ---
    @app.post("/cam/{name}/record/start")
    def record_start(name: str):
        return sess(name).record_start()

    @app.post("/cam/{name}/record/stop")
    def record_stop(name: str):
        return sess(name).record_stop()

    # --- per-camera presets ---
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

    return app


_OVERVIEW = """<!doctype html>
<html><head><meta charset="utf-8"><title>xsphere camera dock</title>
<style>
  body {{ margin:0; background:#111; color:#ddd; font-family:system-ui,sans-serif; }}
  header {{ padding:10px 16px; background:#000; font-size:16px; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:14px; padding:14px; }}
  .card {{ background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:8px; }}
  .card img {{ display:block; max-width:46vw; height:auto; border:1px solid #333; }}
  .cap {{ text-align:center; padding-top:6px; color:#7cf; }}
  a {{ color:inherit; text-decoration:none; }}
</style></head>
<body><header><b>xsphere camera dock</b> &mdash; live streams</header>
<div class="grid">{cards}</div></body></html>"""


_CONTROL = """<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
  body { margin:0; background:#111; color:#ddd; font-family:system-ui,sans-serif; font-size:14px; }
  header { padding:8px 14px; background:#000; } #info { color:#7f7; } a { color:#8cf; }
  .main { display:flex; flex-wrap:wrap; gap:14px; padding:12px; }
  .stream img { max-width:70vw; height:auto; border:1px solid #333; }
  .panel { min-width:280px; background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:12px; }
  .row { margin:10px 0; } label { display:block; margin-bottom:3px; color:#aaa; }
  input[type=range] { width:100%; } input[type=text] { width:140px; background:#222; color:#ddd; border:1px solid #444; }
  .val { color:#7cf; float:right; }
  button { background:#264; color:#dfd; border:1px solid #486; border-radius:4px; padding:6px 10px; cursor:pointer; }
  button:hover { background:#386; } button.rec { background:#622; color:#fdd; border-color:#a55; }
  #stat { color:#fc7; margin-top:8px; min-height:1.2em; }
</style></head>
<body>
  <header><a href="__ROOT__/">&larr; all cameras</a> &nbsp; <b>__TITLE__</b> &mdash; <span id="info">connecting…</span>
    &nbsp; <button id="conn" onclick="toggleConn()">disconnect</button></header>
  <div class="main">
    <div class="stream"><img src="__BASE__/stream" alt="camera stream"></div>
    <div class="panel">
      <div class="row" id="exp-row"><label>exposure (us) <span class="val" id="exp-v"></span></label>
        <input type="range" id="exp" min="0" max="1000"></div>
      <div class="row" id="gain-row"><label>gain <span class="val" id="gain-v"></span></label>
        <input type="range" id="gain" min="0" max="1000"></div>
      <div class="row" id="fps-row"><label>fps <span class="val" id="fps-v"></span></label>
        <input type="range" id="fps" min="0" max="1000"></div>
      <div class="row"><button id="autoexp" onclick="autoExp()">auto-exposure</button>
        <button onclick="save()">snapshot</button></div>
      <div class="row" id="roi-row"><label>ROI x,y,w,h</label>
        <input type="text" id="roi" placeholder="x,y,w,h">
        <button onclick="setRoi()">set</button> <button onclick="resetRoi()">full</button></div>
      <div class="row"><button id="rec" class="rec" onclick="toggleRec()">● record</button></div>
      <div class="row"><label>preset</label>
        <input type="text" id="pname" placeholder="name" list="plist"><datalist id="plist"></datalist>
        <button onclick="savePreset()">save</button> <button onclick="loadPreset()">load</button></div>
      <div id="stat"></div>
    </div>
  </div>
<script>
const BASE="__BASE__"; let R={}, rec=false; const $=id=>document.getElementById(id);
const geom=(lo,hi,f)=>lo*Math.pow(hi/Math.max(lo,1e-6),f);
const geomFrac=(lo,hi,v)=>Math.log(Math.max(v,lo)/Math.max(lo,1e-6))/Math.log(hi/Math.max(lo,1e-6));
async function post(u){return (await fetch(BASE+u,{method:'POST'})).json();}
function bindGeom(id,lo,hi,val,name,fmt){const s=$(id); s.value=Math.round(geomFrac(lo,hi,val)*1000);
  $(id+'-v').textContent=fmt(val);
  s.oninput=async()=>{const v=geom(lo,hi,s.value/1000); $(id+'-v').textContent=fmt(v);
    const r=await post('/controls/'+name+'?value='+v); $(id+'-v').textContent=fmt(r[name]);};}
function bindLin(id,lo,hi,val,name,fmt){const s=$(id); s.value=Math.round((val-lo)/(hi-lo)*1000);
  $(id+'-v').textContent=fmt(val);
  s.oninput=async()=>{const v=lo+(hi-lo)*s.value/1000; $(id+'-v').textContent=fmt(v);
    const r=await post('/controls/'+name+'?value='+v); $(id+'-v').textContent=fmt(r[name]);};}
async function load(){let resp;
  try{ resp=await fetch(BASE+'/controls'); if(!resp.ok) throw 0; R=await resp.json(); }
  catch(e){ return; }   // camera disconnected — skip control binding
  if(R.has_exposure) bindGeom('exp',R.exposure_range[0],R.exposure_range[1],R.exposure,'exposure',v=>v.toFixed(0));
  else { $('exp-row').style.display='none'; $('autoexp').style.display='none'; }
  if(R.has_gain) bindLin('gain',R.gain_range[0],R.gain_range[1],R.gain,'gain',v=>v.toFixed(1)); else $('gain-row').style.display='none';
  if(R.has_fps) bindLin('fps',R.fps_range[0],R.fps_range[1],R.fps,'fps',v=>v.toFixed(1)); else $('fps-row').style.display='none';
  if(R.has_roi) $('roi').value=R.roi.join(','); else $('roi-row').style.display='none';
  setRec(R.recording); refreshPresets();}
async function refreshPresets(){try{const l=await (await fetch(BASE+'/presets')).json();
  $('plist').innerHTML=l.map(n=>`<option value="${n}">`).join('');}catch(e){}}
async function savePreset(){const n=$('pname').value||'default';
  const r=await (await fetch(BASE+'/presets/save?preset='+encodeURIComponent(n),{method:'POST'})).json();
  $('stat').textContent='preset saved → '+r.path; refreshPresets();}
async function loadPreset(){const n=$('pname').value||'default';
  const r=await fetch(BASE+'/presets/load?preset='+encodeURIComponent(n),{method:'POST'}); const j=await r.json();
  $('stat').textContent=r.ok?('preset loaded: '+n):('preset: '+(j.detail||'error')); load();}
async function autoExp(){$('stat').textContent='auto-exposing…'; const r=await post('/controls/auto_exposure');
  $('stat').textContent='exposure → '+r.exposure.toFixed(0)+' us'; load();}
async function save(){const r=await post('/snapshot/save'); $('stat').textContent='snapshot → '+r.path;}
async function setRoi(){const p=$('roi').value.split(',').map(Number); if(p.length!==4) return;
  const r=await fetch(`${BASE}/controls/roi?x=${p[0]}&y=${p[1]}&w=${p[2]}&h=${p[3]}`,{method:'POST'}); const j=await r.json();
  $('stat').textContent = r.ok?('ROI → '+j.roi.join(',')):('ROI: '+(j.detail||'error'));}
async function resetRoi(){const j=await post('/controls/roi/reset'); $('roi').value=j.roi.join(','); $('stat').textContent='ROI → full';}
function setRec(on){rec=on; $('rec').textContent=on?'■ stop':'● record'; $('rec').style.background=on?'#a33':'#622';}
async function toggleRec(){if(!rec){const r=await post('/record/start'); setRec(true); $('stat').textContent='recording → '+r.path;}
  else{$('stat').textContent='encoding…'; const r=await post('/record/stop'); setRec(false); const s=r.stats;
    $('stat').textContent=`saved ${s.path} (${s.encoded} frames @ ${s.capture_fps} fps, dropped ${s.dropped})`;}}
async function toggleConn(){
  let j={}; try{ j=await (await fetch(BASE+'/info')).json(); }catch(e){}
  const ep = j.ok ? '/disconnect' : '/connect';
  $('stat').textContent = j.ok ? 'disconnecting (freeing camera)…' : 'connecting…';
  const r = await fetch(BASE+ep, {method:'POST'});
  if(!r.ok){ const e=await r.json().catch(()=>({})); $('stat').textContent='error: '+(e.detail||r.status); return; }
  setTimeout(()=>location.reload(), 500);   // re-establish stream + controls
}
async function poll(){try{const j=await (await fetch(BASE+'/info')).json();
  if(j.ok){ $('info').textContent=`${j.model} s/n ${j.serial} · ${j.width}x${j.height} · acq ${j.acquisition_fps.toFixed(1)} fps`; $('conn').textContent='disconnect'; }
  else { $('info').textContent='disconnected'+(j.error?(' · '+j.error):''); $('conn').textContent='connect'; }
}catch(e){} setTimeout(poll,1000);}
load(); poll();
</script>
</body></html>"""


# --- camera selection / entry point ---------------------------------------
def _make_camera(name: str):
    dock_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    table = {
        "basler": ("basler-acA1440", "basler_acA1440", "BaslerACA1440"),
        "zelux": ("zelux-cs165mu", "zelux_cs165mu", "ZeluxCS165MU"),
        "hayear": ("hayear", "hayear", "HayearCamera"),
        "ids": ("ids-ueye", "ids_ueye", "IDSUEye"),
    }
    if name not in table:
        raise SystemExit(f"Unknown camera '{name}'. Choose from: {', '.join(table)}")
    subdir, pkg, cls = table[name]
    sys.path.insert(0, os.path.join(dock_root, subdir))
    return getattr(__import__(pkg), cls)()


def serve(sessions: dict, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(create_app(sessions), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and control cameras over the web.")
    parser.add_argument("cameras", nargs="+", choices=["basler", "zelux", "hayear", "ids"],
                        help="one or more cameras to serve")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    sessions = {name: CameraSession(name, _make_camera(name)) for name in args.cameras}
    serve(sessions, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
