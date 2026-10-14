# hayear

Driver/API + test GUI for the **Hayear HY-1136** camera.

The connected HY-1136 is a **UVC webcam-class colour camera** (it streams as a standard
DirectShow/UVC device; the ToupTek/hycam SDK does not enumerate it). So this driver uses
**OpenCV** (`cv2.VideoCapture`), not the hycam SDK. It is consumed as a git submodule by
[`xsphere-camera-dock`](https://github.com/Moore-Lab/xsphere-camera-dock); its driver
implements the shared `CameraBase` interface, so it inherits the dock's acquisition
engine, recorder, streaming, snapshot, and timestamp.

## Goals / characteristics

- **Driver (`hayear/camera.py`)** — `HayearCamera` over `cv2.VideoCapture`: connect,
  select resolution (640×480 / 1280×720 / 1920×1080), grab/stream frames. Frames are
  grayscaled to mono `uint8` to match the (currently mono) dock pipeline.
- **No manual exposure / gain / frame-rate / ROI** — the HY-1136 is auto-only over UVC
  (DSHOW and MSMF both refuse control), so those are reported as unsupported and the dock
  gates the corresponding controls off. It is a **stream / snapshot / record** camera.
- **Test GUI (`hayear/gui.py`)** — a thin shell over `camera_dock.preview`.

## Backend

OpenCV / UVC (DirectShow). No vendor SDK needed.

```bash
pip install -r requirements.txt
```

## Status

Driver implemented (UVC/OpenCV) and validated streaming on the connected HY-1136.
Use in the dock via `python -m xsphere_daq.panel hayear` (alongside other cameras).
See the development plan in the parent `xsphere-daq` repo.
