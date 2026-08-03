# recordings/

Default drop-zone for videos captured by `reference/dualcam_fast.py` (and other
dock recorders). Point the recorder's *Save Video* dialog here.

Each recording is a small **bundle** sharing one `<stem>`:

| File                     | Written by                     | Contents                                            |
| ------------------------ | ------------------------------ | --------------------------------------------------- |
| `<stem>.mp4`             | `cv2.VideoWriter` (MJPG)       | frames — grayscale replicated to 3-channel BGR      |
| `<stem>.json`            | `video_metadata.write_sidecar` | true `measured_fps`, `hw_measured_fps`, drops, w/h  |
| `<stem>_hwclock.npz`     | `write_recording_sidecars`     | `hw_frame_count`, `hw_timestamp_ns`, `ring_in_use`  |
| `<stem>_timestamps.npy`  | `write_recording_sidecars`     | software monotonic grab times (back-compat)         |

The `.mp4` container stores a *nominal* fps, so the true rate can only be recovered
from the sidecars — that's what `analysis/analyze_frames.py` reads.

The media files are git-ignored (only this README and `.gitkeep` are tracked), so
recordings stay local.

## Analyze a recording

```bash
# Printed report (timing / intensity / motion) for one file …
python analysis/analyze_frames.py recordings/camera_1_20260715_120000.mp4
# … or the whole folder:
python analysis/analyze_frames.py recordings/
```

## Click through frames interactively

```bash
# Open a chooser starting in recordings/ …
python analysis/analyze_frames.py recordings/ --view
# … or jump straight into one file:
python analysis/analyze_frames.py recordings/camera_1_20260715_120000.mp4 --view
```

In the viewer: **Open recording…** picks the file; the transport buttons
**−100 / −1 / Play / +1 / +100** step or play at the true recorded rate; the
**Zoom −/+/Fit/1:1** buttons scale the image (pan with the scrollbars when zoomed
in). Keyboard: `Space` play/pause, `←`/`→` = ±1, `PgUp`/`PgDn` = ±100,
`Home`/`End` = first/last, `Ctrl +`/`Ctrl -` = zoom, `F` = fit.
