"""Dock web app — stream a camera feed to a web page.

The first piece of the dock's web UI: a small FastAPI server that runs a camera
through the shared :class:`~camera_dock.engine.AcquisitionEngine` and streams the
live frames to the browser as MJPEG (a plain ``<img>`` tag, works everywhere). It
is camera-agnostic (``create_app(camera)`` takes any ``CameraBase``), so it streams
the Basler, the Zelux, or any future camera unchanged — the same engine that backs
the test GUIs now feeds the web page.

Run it::

    python -m camera_dock.webapp basler            # or zelux / hayear
    python -m camera_dock.webapp zelux --host 0.0.0.0 --port 8000

Then open http://<host>:<port>/ in a browser. Endpoints:

    /          HTML page with the live stream
    /stream    MJPEG (multipart/x-mixed-replace) frame stream
    /snapshot  single current frame as JPEG
    /info      JSON: model, serial, size, acquisition fps
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import cv2
import numpy as np

from . import imaging
from .engine import AcquisitionEngine

STREAM_FPS = 30.0
JPEG_QUALITY = 80


def _to_8bit(bit_depth: int):
    shift = max(bit_depth - 8, 0)
    if shift == 0:
        return lambda f: f if f.dtype == np.uint8 else f.astype(np.uint8)
    return lambda f: (f >> shift).astype(np.uint8)


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ margin:0; background:#111; color:#ddd; font-family:system-ui,sans-serif; }}
  header {{ padding:10px 16px; background:#000; font-size:14px; }}
  #info {{ color:#7f7; }}
  .wrap {{ display:flex; justify-content:center; padding:12px; }}
  img {{ max-width:100%; height:auto; border:1px solid #333; }}
</style></head>
<body>
  <header>{title} &mdash; <span id="info">connecting…</span></header>
  <div class="wrap"><img src="/stream" alt="camera stream"></div>
  <script>
    async function poll() {{
      try {{
        const r = await fetch('/info'); const j = await r.json();
        document.getElementById('info').textContent =
          `${{j.model}} s/n ${{j.serial}} · ${{j.width}}x${{j.height}} · acq ${{j.acquisition_fps.toFixed(1)}} fps`;
      }} catch (e) {{}}
      setTimeout(poll, 1000);
    }}
    poll();
  </script>
</body></html>"""


def create_app(camera):
    """Build the FastAPI app that streams ``camera`` (any CameraBase driver)."""
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, Response, StreamingResponse

    engine = AcquisitionEngine(camera)
    state = {"to8": None, "bit_depth": 8}

    @asynccontextmanager
    async def lifespan(app):
        camera.connect()
        state["bit_depth"] = int(getattr(camera, "bit_depth", 8))
        state["to8"] = _to_8bit(state["bit_depth"])
        engine.start()
        try:
            yield
        finally:
            engine.stop()
            camera.disconnect()

    app = FastAPI(lifespan=lifespan)

    def render(frame: np.ndarray, *, status: bool) -> np.ndarray:
        bgr = cv2.cvtColor(state["to8"](frame), cv2.COLOR_GRAY2BGR)
        if status:
            txt = f"acq {engine.acquisition_fps:5.1f} fps   exp {camera.get_exposure():.0f}us"
            cv2.putText(bgr, txt, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0), 1, cv2.LINE_AA)
        imaging.draw_timestamp(bgr, imaging.eastern_now())
        return bgr

    def encode(frame: np.ndarray, *, status: bool) -> bytes:
        ok, jpg = cv2.imencode(".jpg", render(frame, status=status),
                               [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return jpg.tobytes() if ok else b""

    @app.get("/")
    def index():
        info = camera.device_info
        return HTMLResponse(_PAGE.format(title=info.get("model", "camera")))

    @app.get("/info")
    def info():
        di = camera.device_info
        w, h = camera.sensor_size()
        return {"model": di.get("model"), "serial": di.get("serial"),
                "width": w, "height": h,
                "acquisition_fps": round(engine.acquisition_fps, 1)}

    @app.get("/snapshot")
    def snapshot():
        frame, _ = engine.latest()
        if frame is None:
            return Response(status_code=503)
        return Response(content=encode(frame, status=False), media_type="image/jpeg")

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

    return app


# --- camera selection / entry point ---------------------------------------
def _make_camera(name: str):
    """Construct a driver by short name, wiring the submodule onto sys.path."""
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
    module = __import__(pkg)
    return getattr(module, cls)()


def serve(camera, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(create_app(camera), host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream a camera feed to a web page.")
    parser.add_argument("camera", choices=["basler", "zelux", "hayear"], help="which camera")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(_make_camera(args.camera), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
