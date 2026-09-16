"""Camera orientation: frames, button composition, and the page's coordinate maps.

Three things must agree, or a rotated camera quietly lies:

1. what the engine does to a frame (``Orientation.apply``) — this is what gets
   recorded, so it is the ground truth;
2. how a button composes onto the saved orientation (``compose``);
3. how cam.js maps a point on screen back to a sensor pixel (``orientFwd`` /
   ``orientInv``), which decides where a dragged ROI lands and where every
   measurement sits.

(3) is JavaScript and this rig has no Node, so the two formulas are mirrored
here VERBATIM (``_js_fwd`` / ``_js_inv``) and pinned against numpy. Change them
in cam.js, change them here.

Hardware-free: pure numpy, plus the sim camera for the end-to-end checks. The
per-camera orientation file is redirected into a temp dir, so running this
never re-orients a real camera.
"""
import threading
import time

import numpy as np
import pytest

from camera_dock import orientation as orient_mod
from camera_dock.base import Frame, FrameMeta
from camera_dock.orientation import Orientation

ALL = [Orientation(r, m) for r in (0, 90, 180, 270) for m in (False, True)]
IDS = [f"rot{o.rotate}{'-mirror' if o.mirror else ''}" for o in ALL]

NUMPY_OPS = {
    "cw": lambda a: np.rot90(a, -1),
    "ccw": lambda a: np.rot90(a, 1),
    "flip_h": np.fliplr,
    "flip_v": np.flipud,
}


def _scene(h=3, w=5):
    """Every pixel distinct and the image non-square, so no two orientations
    can be confused for one another."""
    return np.arange(h * w, dtype=np.uint16).reshape(h, w)


# --- 1. frames -----------------------------------------------------------------
def test_the_eight_orientations_are_distinct():
    a = _scene()
    seen = {(o.apply(a).shape, o.apply(a).tobytes()) for o in ALL}
    assert len(seen) == 8


@pytest.mark.parametrize("o", ALL, ids=IDS)
def test_apply_is_mirror_first_then_clockwise(o):
    a = _scene()
    ref = np.rot90(np.fliplr(a) if o.mirror else a, -(o.rotate // 90))
    assert np.array_equal(o.apply(a), ref)


@pytest.mark.parametrize("o", ALL, ids=IDS)
def test_apply_is_contiguous_and_reports_its_size(o):
    a = _scene()
    out = o.apply(a)
    assert out.flags["C_CONTIGUOUS"]         # cv2 and the raw spill writer need it
    assert (out.shape[1], out.shape[0]) == o.size(a.shape[1], a.shape[0])


def test_upright_costs_nothing():
    a = _scene()
    assert Orientation().apply(a) is a


def test_normalisation_and_validation():
    assert Orientation(450, 1) == Orientation(90, True)
    assert Orientation(-90).rotate == 270
    with pytest.raises(ValueError):
        Orientation(45)
    with pytest.raises(ValueError):
        Orientation().compose("sideways")
    assert Orientation.from_dict({"rotate": "bogus"}).is_identity
    assert Orientation.from_dict(None).is_identity


# --- 2. composition ------------------------------------------------------------
@pytest.mark.parametrize("op", sorted(NUMPY_OPS))
@pytest.mark.parametrize("o", ALL, ids=IDS)
def test_a_button_acts_on_the_picture_as_displayed(o, op):
    a = _scene()
    assert np.array_equal(o.compose(op).apply(a), NUMPY_OPS[op](o.apply(a)))


def test_reset_returns_upright():
    assert all(o.compose("reset").is_identity for o in ALL)


# --- 3. the page's maps (mirrors of cam.js) -------------------------------------
def _js_fwd(x, y, o, W, H):
    """cam.js orientFwd: native image -> oriented image; W,H native."""
    if o.mirror:
        x = W - x
    for _ in range(o.rotate // 90):
        x, y = H - y, x
        W, H = H, W
    return x, y


def _js_inv(x, y, o, W, H):
    """cam.js orientInv: oriented image -> native image; W,H oriented."""
    for _ in range(o.rotate // 90):
        x, y = y, W - x
        W, H = H, W
    if o.mirror:
        x = W - x
    return x, y


@pytest.mark.parametrize("o", ALL, ids=IDS)
def test_page_maps_put_every_pixel_where_the_frame_has_it(o):
    """A marker at native pixel (c, r) must be found in the oriented FRAME exactly
    where the page's forward map sends that pixel's centre, and the inverse must
    bring it home. This is what keeps a measurement on the feature it was drawn on."""
    h, w = 4, 7
    ow, oh = o.size(w, h)
    for r in range(h):
        for c in range(w):
            a = np.zeros((h, w), np.uint16)
            a[r, c] = 1
            rr, cc = np.argwhere(o.apply(a))[0]
            fx, fy = _js_fwd(c + 0.5, r + 0.5, o, w, h)
            assert (int(fx), int(fy)) == (cc, rr)
            assert _js_inv(fx, fy, o, ow, oh) == pytest.approx((c + 0.5, r + 0.5))


@pytest.mark.parametrize("o", ALL, ids=IDS)
def test_a_dragged_box_crops_exactly_the_pixels_it_covers(o):
    """cam.js pointerup: an inclusive box of ORIENTED pixels, sent through the
    inverse map by its outer edges, must select exactly the native pixels shown
    inside it — otherwise drag-to-crop zooms somewhere else."""
    h, w = 4, 7
    ow, oh = o.size(w, h)
    native = np.arange(h * w).reshape(h, w)
    shown = o.apply(native)
    boxes = [(0, 0, ow - 1, oh - 1), (1, 0, 2, 1), (ow - 2, oh - 1, ow - 1, oh - 1),
             (0, 1, 0, 3)]
    for ox0, oy0, ox1, oy1 in boxes:
        ex0, ey0 = _js_inv(ox0, oy0, o, ow, oh)
        ex1, ey1 = _js_inv(ox1 + 1, oy1 + 1, o, ow, oh)
        ix0, ix1 = min(ex0, ex1), max(ex0, ex1) - 1
        iy0, iy1 = min(ey0, ey1), max(ey0, ey1) - 1
        assert set(native[iy0:iy1 + 1, ix0:ix1 + 1].ravel()) == \
               set(shown[oy0:oy1 + 1, ox0:ox1 + 1].ravel())
        assert (ix1 - ix0 + 1) * (iy1 - iy0 + 1) == (ox1 - ox0 + 1) * (oy1 - oy0 + 1)


# --- end to end, with the sim camera ------------------------------------------------
@pytest.fixture
def orient_dir(tmp_path, monkeypatch):
    """Keep the per-camera orientation file out of the real rig config."""
    monkeypatch.setattr(orient_mod, "DATA_ROOT", str(tmp_path))

    def data_dir(name):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        return str(d)

    monkeypatch.setattr(orient_mod, "data_dir", data_dir)
    return tmp_path


def _next_frame(session, after_index, timeout=5.0):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        frame, index = session.engine.latest()
        if frame is not None and index > after_index:
            return frame, index
        time.sleep(0.01)
    raise AssertionError("no new frame from the sim camera")


def test_the_session_orients_everything_and_remembers_it(orient_dir):
    from camera_dock import webapp

    s = webapp.CameraSession("orient-test", webapp.make_camera("sim"))
    s.start()
    try:
        assert s.ok, s.error
        w, h = s.camera.image_size()
        frame, i = _next_frame(s, 0)
        assert frame.data.shape == (h, w)

        out = s.change_orientation(op="cw")
        assert out["orientation"]["rotate"] == 90
        frame, i = _next_frame(s, i)
        # The frame, the stream's stale-geometry guard, and what the page is told
        # must all agree on the new shape — or the stream goes black.
        assert frame.data.shape == (w, h)
        assert s.fresh_frame() is not None
        caps = out["capabilities"]
        assert caps["image_size"] == [w, h]            # native: the ROI grid
        assert caps["display_size"] == [h, w]          # oriented: what is shown
        info = s.info()
        assert (info["width"], info["height"]) == (h, w)
        assert s._recording_extra()["orientation"]["rotate"] == 90

        # Refused mid-recording and mid-timelapse, like a region change.
        s.rec_state = "recording"
        with pytest.raises(webapp._Conflict):
            s.change_orientation(op="flip_h")
        s.rec_state = "idle"
        assert s.engine.orientation.rotate == 90       # nothing changed
    finally:
        s.stop()

    # A restart comes back the way the camera is mounted.
    again = webapp.CameraSession("orient-test", webapp.make_camera("sim"))
    assert again.engine.orientation == Orientation(90, False)


def test_the_http_route(orient_dir):
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from camera_dock import webapp

    s = webapp.CameraSession("orient-http", webapp.make_camera("sim"))
    s.start()
    try:
        client = TestClient(webapp.create_app({"orient-http": s}, manage_lifecycle=False))
        r = client.post("/cam/orient-http/orientation?op=flip_h")
        assert r.status_code == 200 and r.json()["orientation"]["mirror"] is True
        assert client.get("/cam/orient-http/orientation").json()["mirror"] is True
        assert client.post("/cam/orient-http/orientation?op=sideways").status_code == 422
        assert client.post("/cam/orient-http/orientation").status_code == 422
        assert client.post("/cam/orient-http/orientation?rotate=45").status_code == 422
        r = client.post("/cam/orient-http/orientation?rotate=270&mirror=false")
        assert r.json()["orientation"] == {"rotate": 270, "mirror": False,
                                           "label": "rotated 90° counter-clockwise"}
    finally:
        s.stop()


# --- review follow-ups ------------------------------------------------------------
class _FakeCam:
    """Just enough camera for the engine: a non-square frame every few ms."""

    def start(self, max_throughput=True):
        pass

    def stop(self):
        pass

    def grab(self, timeout_ms=2000):
        time.sleep(0.005)
        return Frame(np.zeros((3, 5), np.uint16), FrameMeta(t_sw=time.perf_counter()))


class _ReorientAtTheSink:
    """Stands in for the engine's sink lock and re-orients the instant the loop
    reaches the sink — after a frame is published, before it is handed over.
    That is the window a record_start right after a re-orientation falls into."""

    def __init__(self, engine, new):
        self._lock, self._engine, self._new = threading.Lock(), engine, new
        self.fired = False

    def __enter__(self):
        if not self.fired and self._engine._sink is not None:
            self.fired = True
            self._engine.set_orientation(self._new)
        return self._lock.__enter__()

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)


def test_a_frame_from_the_old_orientation_never_reaches_the_recorder():
    """The recorder latches the first frame shape it sees and silently drops any
    frame that differs. One stale-orientation frame at the start would therefore
    cost the whole take."""
    from camera_dock.engine import AcquisitionEngine

    eng = AcquisitionEngine(_FakeCam())
    shapes = []
    eng.set_sink(lambda f, i, t: shapes.append(f.data.shape))
    race = _ReorientAtTheSink(eng, Orientation(90))
    eng._sink_lock = race
    eng.start()
    try:
        t0 = time.perf_counter()
        while len(shapes) < 5 and time.perf_counter() - t0 < 5:
            time.sleep(0.01)
    finally:
        eng.stop()
    assert race.fired
    assert shapes and all(s == (5, 3) for s in shapes), shapes


@pytest.mark.parametrize("rotate, horizontal_binning",
                         [(None, 1), (0, 1), (90, 2), (180, 1), (270, 2)])
def test_a_recordings_scale_follows_its_own_horizontal_axis(rotate, horizontal_binning):
    """Sidecar binning is in SENSOR axes; after a quarter turn the recording's
    horizontal axis is sensor y. droplab (and measure.html, same rule) must scale
    by the binning of the axis the image actually has — a sidecar written before
    orientation existed reads as upright."""
    from analysis.droplab.video import sidecar_image_binning, sidecar_pixels_per_um

    meta = {"um_per_px": 3.45, "binning": [1, 2]}
    if rotate is not None:
        meta["orientation"] = {"rotate": rotate, "mirror": False}
    assert sidecar_pixels_per_um(meta) == pytest.approx(1 / (3.45 * horizontal_binning))
    h, v = sidecar_image_binning(meta)
    assert (h, v) == (horizontal_binning, 3 - horizontal_binning)
