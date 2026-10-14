"""Presets: driver-specific funnel controls survive a save / load / apply.

A preset used to keep only exposure / gain / fps / ROI / binning, so a control a
driver registers on top of those (the uEye's ``pixel_clock``, the Zelux's
``black_level``) was lost on every restart, the auto-applied ``default`` preset
included. Those now ride along under ``controls`` and are applied FIRST: the
pixel clock re-ranges fps and exposure, so setting it last would clamp them.

Hardware-free: the sim camera, given a pixel clock that scales its fps ceiling
like the uEye's. The preset directory is redirected into a temp dir, so running
this never touches the rig's saved presets.
"""
import json

import pytest

from camera_dock import presets
from camera_dock.base import ControlSpec
from camera_dock.drivers.sim import SimCamera

PCLK_DEFAULT = 24.0


class PixelClockSim(SimCamera):
    """Sim plus a ``pixel_clock`` that scales the fps ceiling (like the uEye),
    and a read-only ``temperature`` (which no preset can apply)."""

    def __init__(self) -> None:
        super().__init__()
        self._pclk = PCLK_DEFAULT

    def connect(self) -> None:
        super().connect()
        self._register(ControlSpec(
            name="pixel_clock", units="MHz", kind="int", step=1,
            get_range=lambda: (5.0, 43.0),
            getter=lambda: self._pclk,
            setter=self._set_pclk))
        self._register(ControlSpec(
            name="temperature", units="C",
            get_range=lambda: (-40.0, 85.0),
            getter=lambda: 31.5))

    def _set_pclk(self, mhz: float) -> None:
        self._pclk = float(mhz)
        self._fps = min(self._fps, self._fps_range()[1])   # hardware re-clamps

    def _fps_range(self):
        lo, hi = super()._fps_range()
        return lo, hi * self._pclk / PCLK_DEFAULT


@pytest.fixture
def preset_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "_dir", lambda: str(tmp_path))
    return tmp_path


def _connected(cls=PixelClockSim):
    cam = cls()
    cam.connect()
    return cam


def test_capture_keeps_extra_controls_under_their_own_key():
    cam = _connected()
    cam.set("pixel_clock", 40)
    s = presets.capture(cam, has_roi=True)
    # The old keys are unchanged: exposure/gain/fps stay top-level, not duplicated.
    assert {"exposure", "gain", "fps", "roi", "binning"} <= set(s)
    assert s["controls"] == {"pixel_clock": 40.0}       # read-only skipped


def test_a_camera_without_extras_keeps_the_old_json_shape():
    s = presets.capture(_connected(SimCamera), has_roi=True)
    assert set(s) == {"exposure", "gain", "fps", "roi", "binning"}


def test_a_non_default_pixel_clock_round_trips(preset_dir):
    cam = _connected()
    full_frame_ceiling = cam.frame_rate_range()[1]
    cam.set("pixel_clock", 40)
    # Only reachable at the faster clock: applying fps before the clock clamps it.
    fps = full_frame_ceiling * 1.5
    assert cam.set("fps", fps).value == pytest.approx(fps)
    cam.set("exposure", 12345)
    cam.set("gain", 120)
    presets.save("cam", "default", presets.capture(cam, has_roi=True))

    on_disk = json.loads((preset_dir / "cam__default.json").read_text())
    assert on_disk["controls"] == {"pixel_clock": 40.0}

    fresh = _connected()                       # a restart: clock back at default
    assert fresh.get("pixel_clock").value == PCLK_DEFAULT
    presets.apply(fresh, presets.load("cam", "default"))
    assert fresh.get("pixel_clock").value == 40.0
    assert fresh.get_frame_rate() == pytest.approx(fps)
    assert fresh.get_exposure() == 12345.0
    assert fresh.get_gain() == 120.0


def test_a_preset_from_before_controls_applies_unchanged(preset_dir):
    # The exact shape already on disk (presets/ids__overnight.json).
    old = {"exposure": 71556.0, "gain": 0.0, "fps": 13.962044182888818,
           "roi": [0, 0, 1280, 1024], "binning": [1, 1]}
    (preset_dir / "cam__overnight.json").write_text(json.dumps(old, indent=2))

    cam = _connected()
    presets.apply(cam, presets.load("cam", "overnight"))
    assert cam.get_exposure() == 71556.0
    assert cam.get_gain() == 0.0
    assert cam.get_frame_rate() == pytest.approx(13.962044182888818)
    assert cam.get_roi() == (0, 0, 1280, 1024)
    assert cam.get("pixel_clock").value == PCLK_DEFAULT   # untouched


def test_a_control_this_camera_lacks_is_skipped_not_fatal():
    cam = _connected(SimCamera)                # no pixel_clock registered
    presets.apply(cam, {"controls": {"pixel_clock": 40.0, "temperature": 20.0},
                        "fps": 20.0, "exposure": 8000.0})
    assert cam.get("pixel_clock").supported is False
    assert cam.get_frame_rate() == 20.0
    assert cam.get_exposure() == 8000.0
