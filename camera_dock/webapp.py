"""Dock web app — stream a camera feed and drive it from the browser.

A FastAPI server that runs any :class:`~camera_dock.base.CameraBase` camera through
the shared :class:`~camera_dock.engine.AcquisitionEngine`, streams the live frames
as MJPEG, and exposes the camera's controls (exposure, gain, frame rate, ROI,
auto-exposure, recording) as endpoints — all over the same surface the test GUIs
use, so nothing is duplicated.

Run it::

    python -m camera_dock.webapp basler             # or zelux / hayear
    python -m camera_dock.webapp zelux --host 0.0.0.0 --port 8000

Endpoints:

    GET  /                     control page + live stream
    GET  /stream               MJPEG (multipart/x-mixed-replace)
    GET  /snapshot             current frame as JPEG
    POST /snapshot/save        save current frame to captures/ (full bit depth)
    GET  /info                 JSON: model, serial, size, acquisition fps
    GET  /controls             JSON: all values, ranges, capabilities, rec state
    POST /controls/exposure?value=US
    POST /controls/gain?value=G
    POST /controls/fps?value=FPS
    POST /controls/roi?x=&y=&w=&h=     (blocked while recording)
    POST /controls/roi/reset
    POST /controls/auto_exposure
    POST /record/start
    POST /record/stop          -> encode stats
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

from . import imaging
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


def create_app(camera):
    """Build the FastAPI app that streams and controls ``camera`` (any CameraBase)."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, Response, StreamingResponse

    engine = AcquisitionEngine(camera)
    state = {"to8": _to_8bit(8), "bit_depth": 8, "has_roi": False,
             "recorder": None, "rec_path": "", "lock": threading.Lock()}

    @asynccontextmanager
    async def lifespan(app):
        camera.connect()
        state["bit_depth"] = int(getattr(camera, "bit_depth", 8))
        state["to8"] = _to_8bit(state["bit_depth"])
        try:
            camera.roi_range()
            state["has_roi"] = True
        except Exception:
            state["has_roi"] = False
        engine.start()
        try:
            yield
        finally:
            engine.set_sink(None)
            engine.stop()
            camera.disconnect()

    app = FastAPI(lifespan=lifespan)

    # --- frame rendering / encoding ---------------------------------------
    def render(frame: np.ndarray, *, status: bool) -> np.ndarray:
        bgr = cv2.cvtColor(state["to8"](frame), cv2.COLOR_GRAY2BGR)
        if status:
            txt = f"acq {engine.acquisition_fps:5.1f} fps   exp {camera.get_exposure():.0f}us"
            cv2.putText(bgr, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
            if state["recorder"] is not None:
                cv2.putText(bgr, f"REC {state['recorder']._captured}", (bgr.shape[1] - 150, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
        imaging.draw_timestamp(bgr, imaging.eastern_now())
        return bgr

    def encode(frame: np.ndarray, *, status: bool) -> bytes:
        ok, jpg = cv2.imencode(".jpg", render(frame, status=status),
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return jpg.tobytes() if ok else b""

    def grab_fresh():
        _, start = engine.latest()
        t0 = perf_counter()
        while perf_counter() - t0 < 2.0:
            f, i = engine.latest()
            if f is not None and i >= start + 4:
                return f
            sleep(0.005)
        return engine.latest()[0]

    # --- pages / stream ----------------------------------------------------
    @app.get("/")
    def index():
        return HTMLResponse(_PAGE.format(title=camera.device_info.get("model", "camera")))

    async def mjpeg():
        period = 1.0 / STREAM_FPS
        while True:
            frame, _ = engine.latest()
            if frame is not None:
                jpg = encode(frame, status=True)
                if jpg:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            await asyncio.sleep(period)

    @app.get("/stream")
    def stream():
        return StreamingResponse(mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot")
    def snapshot():
        frame, _ = engine.latest()
        if frame is None:
            raise HTTPException(503, "no frame yet")
        return Response(content=encode(frame, status=False), media_type="image/jpeg")

    @app.post("/snapshot/save")
    def snapshot_save(fmt: str = "tiff"):
        frame, _ = engine.latest()
        if frame is None:
            raise HTTPException(503, "no frame yet")
        os.makedirs("captures", exist_ok=True)
        base = datetime.now().strftime(f"captures/{_slug(camera.device_info)}_%Y%m%d_%H%M%S_%f")
        return {"path": imaging.save_snapshot(frame, base, fmt)}

    # --- info / controls ---------------------------------------------------
    @app.get("/info")
    def info():
        di = camera.device_info
        w, h = camera.sensor_size()
        return {"model": di.get("model"), "serial": di.get("serial"),
                "width": w, "height": h,
                "acquisition_fps": round(engine.acquisition_fps, 1)}

    @app.get("/controls")
    def controls():
        gl, gh = camera.gain_range()
        el, eh = camera.exposure_range()
        fl, fh = camera.frame_rate_range()
        return {
            "exposure": camera.get_exposure(), "exposure_range": [el, eh],
            "gain": camera.get_gain(), "gain_range": [gl, gh], "has_gain": gh > gl,
            "fps": camera.get_frame_rate(), "fps_range": [fl, fh], "has_fps": fh > 0,
            "roi": list(camera.get_roi()) if state["has_roi"] else None,
            "has_roi": state["has_roi"], "sensor": list(camera.sensor_size()),
            "bit_depth": state["bit_depth"],
            "recording": state["recorder"] is not None,
            "acquisition_fps": round(engine.acquisition_fps, 1),
        }

    @app.post("/controls/exposure")
    def set_exposure(value: float):
        camera.set_exposure(value)
        return {"exposure": camera.get_exposure()}

    @app.post("/controls/gain")
    def set_gain(value: float):
        camera.set_gain(value)
        return {"gain": camera.get_gain()}

    @app.post("/controls/fps")
    def set_fps(value: float):
        camera.set_frame_rate(value)
        return {"fps": camera.get_frame_rate()}

    @app.post("/controls/auto_exposure")
    def auto_exposure():
        return {"exposure": imaging.auto_expose(camera, grab_fresh)}

    @app.post("/controls/roi")
    def set_roi(x: int, y: int, w: int, h: int):
        if not state["has_roi"]:
            raise HTTPException(400, "ROI not supported")
        with state["lock"]:
            if state["recorder"] is not None:
                raise HTTPException(409, "stop recording before changing ROI")
            engine.stop()
            try:
                camera.set_roi(x, y, w, h)
            finally:
                engine.start()
            return {"roi": list(camera.get_roi()), "sensor": list(camera.sensor_size())}

    @app.post("/controls/roi/reset")
    def reset_roi():
        if not state["has_roi"]:
            raise HTTPException(400, "ROI not supported")
        with state["lock"]:
            if state["recorder"] is not None:
                raise HTTPException(409, "stop recording before changing ROI")
            engine.stop()
            try:
                camera.reset_roi()
            finally:
                engine.start()
            return {"roi": list(camera.get_roi()), "sensor": list(camera.sensor_size())}

    # --- recording ---------------------------------------------------------
    @app.post("/record/start")
    def record_start():
        with state["lock"]:
            if state["recorder"] is not None:
                return {"recording": True}
            rec = HybridRecorder(clock=imaging.eastern_now)
            engine.set_sink(rec.submit)
            os.makedirs("recordings", exist_ok=True)
            state["rec_path"] = datetime.now().strftime(
                f"recordings/{_slug(camera.device_info)}_%Y%m%d_%H%M%S.avi")
            state["recorder"] = rec
            return {"recording": True, "path": state["rec_path"]}

    @app.post("/record/stop")
    def record_stop():
        with state["lock"]:
            rec = state["recorder"]
            if rec is None:
                return {"recording": False}
            engine.set_sink(None)
            state["recorder"] = None
            path, fps = state["rec_path"], engine.acquisition_fps or camera.get_frame_rate() or 30.0
        stats = rec.stop_and_encode(path, fps, state["to8"], stamp=imaging.draw_timestamp)
        return {"recording": False, "stats": stats}

    return app


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ margin:0; background:#111; color:#ddd; font-family:system-ui,sans-serif; font-size:14px; }}
  header {{ padding:8px 14px; background:#000; }} #info {{ color:#7f7; }}
  .main {{ display:flex; flex-wrap:wrap; gap:14px; padding:12px; }}
  .stream img {{ max-width:70vw; height:auto; border:1px solid #333; }}
  .panel {{ min-width:280px; background:#1a1a1a; border:1px solid #333; border-radius:6px; padding:12px; }}
  .row {{ margin:10px 0; }} label {{ display:block; margin-bottom:3px; color:#aaa; }}
  input[type=range] {{ width:100%; }} input[type=text] {{ width:140px; background:#222; color:#ddd; border:1px solid #444; }}
  .val {{ color:#7cf; float:right; }}
  button {{ background:#264; color:#dfd; border:1px solid #486; border-radius:4px; padding:6px 10px; cursor:pointer; }}
  button:hover {{ background:#386; }} button.rec {{ background:#622; color:#fdd; border-color:#a55; }}
  #stat {{ color:#fc7; margin-top:8px; min-height:1.2em; }}
</style></head>
<body>
  <header><b>{title}</b> &mdash; <span id="info">connecting…</span></header>
  <div class="main">
    <div class="stream"><img src="/stream" alt="camera stream"></div>
    <div class="panel">
      <div class="row" id="exp-row"><label>exposure (us) <span class="val" id="exp-v"></span></label>
        <input type="range" id="exp" min="0" max="1000"></div>
      <div class="row" id="gain-row"><label>gain <span class="val" id="gain-v"></span></label>
        <input type="range" id="gain" min="0" max="1000"></div>
      <div class="row" id="fps-row"><label>fps <span class="val" id="fps-v"></span></label>
        <input type="range" id="fps" min="0" max="1000"></div>
      <div class="row"><button onclick="autoExp()">auto-exposure</button>
        <button onclick="save()">snapshot</button></div>
      <div class="row" id="roi-row"><label>ROI x,y,w,h</label>
        <input type="text" id="roi" placeholder="x,y,w,h">
        <button onclick="setRoi()">set</button> <button onclick="resetRoi()">full</button></div>
      <div class="row"><button id="rec" class="rec" onclick="toggleRec()">● record</button></div>
      <div id="stat"></div>
    </div>
  </div>
<script>
let R={{}}, rec=false;
const $=id=>document.getElementById(id);
const geom=(lo,hi,f)=>lo*Math.pow(hi/Math.max(lo,1e-6),f);
const geomFrac=(lo,hi,v)=>Math.log(Math.max(v,lo)/Math.max(lo,1e-6))/Math.log(hi/Math.max(lo,1e-6));
async function post(u){{return (await fetch(u,{{method:'POST'}})).json();}}
function bindGeom(id,lo,hi,val,name,fmt){{
  const s=$(id); s.value=Math.round(geomFrac(lo,hi,val)*1000);
  $(id+'-v').textContent=fmt(val);
  s.oninput=async()=>{{const v=geom(lo,hi,s.value/1000); $(id+'-v').textContent=fmt(v);
    const r=await post('/controls/'+name+'?value='+v); $(id+'-v').textContent=fmt(r[name]);}};
}}
function bindLin(id,lo,hi,val,name,fmt){{
  const s=$(id); s.value=Math.round((val-lo)/(hi-lo)*1000);
  $(id+'-v').textContent=fmt(val);
  s.oninput=async()=>{{const v=lo+(hi-lo)*s.value/1000; $(id+'-v').textContent=fmt(v);
    const r=await post('/controls/'+name+'?value='+v); $(id+'-v').textContent=fmt(r[name]);}};
}}
async function load(){{
  R=await (await fetch('/controls')).json();
  bindGeom('exp',R.exposure_range[0],R.exposure_range[1],R.exposure,'exposure',v=>v.toFixed(0));
  if(R.has_gain) bindLin('gain',R.gain_range[0],R.gain_range[1],R.gain,'gain',v=>v.toFixed(1));
  else $('gain-row').style.display='none';
  if(R.has_fps) bindLin('fps',R.fps_range[0],R.fps_range[1],R.fps,'fps',v=>v.toFixed(1));
  else $('fps-row').style.display='none';
  if(R.has_roi) $('roi').value=R.roi.join(','); else $('roi-row').style.display='none';
  setRec(R.recording);
}}
async function autoExp(){{$('stat').textContent='auto-exposing…'; const r=await post('/controls/auto_exposure');
  $('stat').textContent='exposure → '+r.exposure.toFixed(0)+' us'; load();}}
async function save(){{const r=await post('/snapshot/save'); $('stat').textContent='snapshot → '+r.path;}}
async function setRoi(){{const p=$('roi').value.split(',').map(Number); if(p.length!==4) return;
  const r=await fetch(`/controls/roi?x=${{p[0]}}&y=${{p[1]}}&w=${{p[2]}}&h=${{p[3]}}`,{{method:'POST'}});
  const j=await r.json(); $('stat').textContent = r.ok?('ROI → '+j.roi.join(',')):('ROI: '+(j.detail||'error'));}}
async function resetRoi(){{const j=await post('/controls/roi/reset'); $('roi').value=j.roi.join(','); $('stat').textContent='ROI → full';}}
function setRec(on){{rec=on; $('rec').textContent=on?'■ stop':'● record'; $('rec').style.background=on?'#a33':'#622';}}
async function toggleRec(){{
  if(!rec){{const r=await post('/record/start'); setRec(true); $('stat').textContent='recording → '+r.path;}}
  else{{$('stat').textContent='encoding…'; const r=await post('/record/stop'); setRec(false);
    const s=r.stats; $('stat').textContent=`saved ${{s.path}} (${{s.encoded}} frames @ ${{s.capture_fps}} fps, dropped ${{s.dropped}})`;}}
}}
async function poll(){{try{{const j=await (await fetch('/info')).json();
  $('info').textContent=`${{j.model}} s/n ${{j.serial}} · ${{j.width}}x${{j.height}} · acq ${{j.acquisition_fps.toFixed(1)}} fps`;}}catch(e){{}}
  setTimeout(poll,1000);}}
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
    }
    if name not in table:
        raise SystemExit(f"Unknown camera '{name}'. Choose from: {', '.join(table)}")
    subdir, pkg, cls = table[name]
    sys.path.insert(0, os.path.join(dock_root, subdir))
    return getattr(__import__(pkg), cls)()


def serve(camera, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(create_app(camera), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and control a camera over the web.")
    parser.add_argument("camera", choices=["basler", "zelux", "hayear"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(_make_camera(args.camera), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
