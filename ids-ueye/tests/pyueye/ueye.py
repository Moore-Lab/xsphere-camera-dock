"""Mock of the pyueye.ueye API subset used by ids_ueye.camera.

Emulates one DCC1545M-GL-like camera (1280x1024 Mono8, AOI steps 4/2,
pixel clocks [5,10,20,30,43], 1..87 fps, 0.08..200 ms exposure) with a
frame counter that advances on a wall-clock schedule after capture starts.
State asserts mirror real SDK preconditions (e.g. AOI only while stopped).
"""
import ctypes
import time

IS_SUCCESS = 0
IS_TIMED_OUT = 122
IS_SET_EVENT_FRAME = 2
IS_SET_TRIGGER_OFF = 0x0000
IS_DONT_WAIT = 0x0000
IS_FORCE_VIDEO_STOP = 0x4000
IS_CM_MONO8 = 6
IS_GET_FRAMERATE = 0x8000
IS_AOI_IMAGE_SET_AOI = 0x0001
IS_AOI_IMAGE_GET_AOI = 0x0002
IS_AOI_IMAGE_GET_POS_INC = 0x0004  # values unimportant; identity-checked
IS_AOI_IMAGE_GET_SIZE_INC = 0x0005
IS_AOI_IMAGE_GET_SIZE_MIN = 0x0006
IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE = 6
IS_EXPOSURE_CMD_GET_EXPOSURE = 7
IS_EXPOSURE_CMD_SET_EXPOSURE = 12
IS_PIXELCLOCK_CMD_GET_NUMBER = 1
IS_PIXELCLOCK_CMD_GET_LIST = 2
IS_PIXELCLOCK_CMD_SET = 6
IS_PIXELCLOCK_CMD_GET = 4


class IS_RECT(ctypes.Structure):
    _fields_ = [("s32X", ctypes.c_int), ("s32Y", ctypes.c_int),
                ("s32Width", ctypes.c_int), ("s32Height", ctypes.c_int)]


class IS_SIZE_2D(ctypes.Structure):
    _fields_ = [("s32Width", ctypes.c_int), ("s32Height", ctypes.c_int)]


class IS_POINT_2D(ctypes.Structure):
    _fields_ = [("s32X", ctypes.c_int), ("s32Y", ctypes.c_int)]


class c_mem_p(ctypes.c_void_p):
    pass


class SENSORINFO(ctypes.Structure):
    _fields_ = [("nMaxWidth", ctypes.c_uint), ("nMaxHeight", ctypes.c_uint),
                ("strSensorName", ctypes.c_char * 32)]


class BOARDINFO(ctypes.Structure):
    _fields_ = [("SerNo", ctypes.c_char * 12)]


class UEYEIMAGEINFO(ctypes.Structure):
    _fields_ = [("u64FrameNumber", ctypes.c_uint64),
                ("u64TimestampDevice", ctypes.c_uint64),
                ("dwImageBuffersInUse", ctypes.c_uint)]


class UEYE_CAMERA_INFO(ctypes.Structure):
    _fields_ = [("dwDeviceID", ctypes.c_uint), ("dwInUse", ctypes.c_uint),
                ("Model", ctypes.c_char * 16), ("FullModelName", ctypes.c_char * 32),
                ("SerNo", ctypes.c_char * 16)]


def UEYE_CAMERA_LIST(array_type):
    class _List:
        def __init__(self):
            self.dwCount = 0
            self.uci = array_type()
    return _List()


SENSOR_W, SENSOR_H = 1280, 1024


class _State:
    def __init__(self):
        self.open = False
        self.capturing = False
        self.event_enabled = False
        self.color_mode = None
        self.aoi = (0, 0, SENSOR_W, SENSOR_H)
        self.buffers = {}         # mem_id -> (w, h)
        self.sequence = []        # mem_ids in queue order
        self.next_mem_id = 1
        self.fps = 25.0           # uEye freerun default
        self.commanded_fps_survives_rearm = False
        self.exposure_ms = 20.0
        self.pixel_clock = 20
        self.t_start = None
        self.frames_produced = 0
        self.locked = set()
        self.stall = False        # freeze frame delivery (test hook)
        self.fail_aoi_set = False # make the next is_AOI SET fail (test hook)

    def frames_now(self):
        if not self.capturing:
            return self.frames_produced
        n = int((time.perf_counter() - self.t_start) * self.fps)
        return n


S = _State()
CALLS = []


def _log(name):
    CALLS.append(name)


def is_GetNumberOfCameras(n):
    n.value = 1   # driver passes the ctypes object directly
    return IS_SUCCESS


def is_GetCameraList(cam_list):
    cam_list.dwCount = 1
    info = cam_list.uci[0]
    info.dwDeviceID = 1
    info.dwInUse = 0
    info.Model = b"MOCK1545"
    info.FullModelName = b"DCC1545M-GL-MOCK"
    info.SerNo = b"4103000000"
    return IS_SUCCESS


def is_InitCamera(hCam, ptr):
    _log("InitCamera")
    assert hCam.value & 0x8000, "expected IS_USE_DEVICE_ID flag"
    S.open = True
    return IS_SUCCESS


def is_ExitCamera(hCam):
    _log("ExitCamera")
    assert S.open, "ExitCamera on closed camera"
    assert not S.sequence, "ExitCamera with buffers still queued (leak)"
    S.open = False
    return IS_SUCCESS


def is_GetSensorInfo(hCam, sinfo):
    sinfo.nMaxWidth, sinfo.nMaxHeight = SENSOR_W, SENSOR_H
    sinfo.strSensorName = b"DCC1545M-GL-MOCK"
    return IS_SUCCESS


def is_GetCameraInfo(hCam, binfo):
    binfo.SerNo = b"4103000000"
    return IS_SUCCESS


def is_SetColorMode(hCam, mode):
    S.color_mode = mode
    return IS_SUCCESS


def is_AllocImageMem(hCam, w, h, bpp, mem_ptr, mem_id):
    assert bpp == 8
    mem_id.value = S.next_mem_id
    mem_ptr.value = 0x10000 + S.next_mem_id
    S.buffers[S.next_mem_id] = (int(w), int(h))
    S.next_mem_id += 1
    return IS_SUCCESS


def is_AddToSequence(hCam, mem_ptr, mem_id):
    S.sequence.append(mem_id.value)
    return IS_SUCCESS


def is_ClearSequence(hCam):
    S.sequence.clear()
    return IS_SUCCESS


def is_FreeImageMem(hCam, mem_ptr, mem_id):
    assert mem_id.value not in S.sequence, "freeing a buffer still in the sequence"
    S.buffers.pop(mem_id.value, None)
    return IS_SUCCESS


def is_GetImageMemPitch(hCam, pitch):
    w = S.aoi[2]
    pitch.value = (w + 3) & ~3   # 4-byte aligned rows -> exercises pitch strip
    return IS_SUCCESS


def is_EnableEvent(hCam, ev):
    S.event_enabled = True
    return IS_SUCCESS


def is_DisableEvent(hCam, ev):
    S.event_enabled = False
    return IS_SUCCESS


def is_SetExternalTrigger(hCam, mode):
    return IS_SUCCESS


def is_CaptureVideo(hCam, mode):
    _log("CaptureVideo")
    assert S.buffers, "CaptureVideo with no image memory"
    assert not S.capturing, "CaptureVideo while already live"
    S.capturing = True
    S.t_start = time.perf_counter()
    S.frames_produced = 0
    if not S.commanded_fps_survives_rearm:
        S.fps = 25.0   # freerun rate resets on re-arm (real uEye behavior)
    return IS_SUCCESS


def is_StopLiveVideo(hCam, mode):
    _log("StopLiveVideo")
    S.frames_produced = S.frames_now()
    S.capturing = False
    return IS_SUCCESS


def is_WaitEvent(hCam, ev, timeout_ms):
    if not S.event_enabled:
        return -1   # real SDK errors out on a destroyed event
    if S.stall:
        time.sleep(timeout_ms / 1000.0)
        return IS_TIMED_OUT
    if not S.capturing:
        time.sleep(timeout_ms / 1000.0)
        return IS_TIMED_OUT
    # Wait until the next frame boundary or timeout
    deadline = time.perf_counter() + timeout_ms / 1000.0
    start_count = S.frames_now()
    while time.perf_counter() < deadline:
        if S.frames_now() > start_count:
            return IS_SUCCESS
        time.sleep(0.001)
    return IS_TIMED_OUT


def is_GetActSeqBuf(hCam, nNum, pcMem, pcMemLast):
    if not S.capturing or S.frames_now() == 0:
        return -1
    n = S.frames_now()
    idx = (n - 1) % len(S.sequence)
    nNum.value = S.sequence[idx]
    pcMemLast.value = 0x10000 + S.sequence[idx]
    return IS_SUCCESS


def is_UnlockSeqBuf(hCam, nNum, pcMem):
    return IS_SUCCESS


def get_data(mem_ptr, w, h, bpp, pitch, copy):
    assert copy, "driver must copy out of the ring buffer"
    import numpy as np
    w, h, pitch = int(w), int(h), int(pitch)
    frame_no = S.frames_now()
    buf = np.full(h * pitch, frame_no % 256, dtype=np.uint8)
    # Mark padding columns so a pitch-strip failure is detectable
    rows = buf.reshape(h, pitch)
    rows[:, w:] = 255
    return buf.tobytes()


def is_SetFrameRate(hCam, fps, newFPS):
    if isinstance(fps, int) and fps == IS_GET_FRAMERATE:
        newFPS.value = S.fps
        return IS_SUCCESS
    val = fps.value if hasattr(fps, "value") else float(fps)
    if val == IS_GET_FRAMERATE:   # driver passed the constant unwrapped
        newFPS.value = S.fps
        return IS_SUCCESS
    lo, hi = _fps_range()
    S.fps = min(max(val, lo), hi)
    newFPS.value = S.fps
    return IS_SUCCESS


def _fps_range():
    # Higher pixel clock and smaller AOI -> higher max fps
    area_factor = (SENSOR_W * SENSOR_H) / (S.aoi[2] * S.aoi[3])
    return 1.0, min(500.0, 87.0 * (S.pixel_clock / 43.0) * area_factor)


def is_GetFrameTimeRange(hCam, tmin, tmax, tint):
    lo, hi = _fps_range()
    tmin.value = 1.0 / hi
    tmax.value = 1.0 / lo
    tint.value = 1e-4
    return IS_SUCCESS


def is_GetFramesPerSecond(hCam, fps):
    fps.value = S.fps if S.capturing else 0.0
    return IS_SUCCESS


def is_Exposure(hCam, cmd, param, size):
    if cmd == IS_EXPOSURE_CMD_SET_EXPOSURE:
        S.exposure_ms = min(max(param.value, 0.08), min(200.0, 1000.0 / S.fps))
        param.value = S.exposure_ms
        return IS_SUCCESS
    if cmd == IS_EXPOSURE_CMD_GET_EXPOSURE:
        param.value = S.exposure_ms
        return IS_SUCCESS
    if cmd == IS_EXPOSURE_CMD_GET_EXPOSURE_RANGE:
        param[0], param[1], param[2] = 0.08, min(200.0, 1000.0 / S.fps), 0.01
        return IS_SUCCESS
    return -1


def is_PixelClock(hCam, cmd, param, size):
    clocks = [5, 10, 20, 30, 43]
    if cmd == IS_PIXELCLOCK_CMD_GET_NUMBER:
        param.value = len(clocks)
        return IS_SUCCESS
    if cmd == IS_PIXELCLOCK_CMD_GET_LIST:
        for i, c in enumerate(clocks):
            param[i] = c
        return IS_SUCCESS
    if cmd == IS_PIXELCLOCK_CMD_GET:
        param.value = S.pixel_clock
        return IS_SUCCESS
    if cmd == IS_PIXELCLOCK_CMD_SET:
        assert param.value in clocks
        S.pixel_clock = param.value
        return IS_SUCCESS
    return -1


def is_AOI(hCam, cmd, param, size):
    if cmd == IS_AOI_IMAGE_SET_AOI:
        assert not S.capturing, "AOI change while live video is running"
        if S.fail_aoi_set:
            S.fail_aoi_set = False
            return -1
        x, y, w, h = param.s32X, param.s32Y, param.s32Width, param.s32Height
        assert x % 4 == 0 and w % 4 == 0, f"AOI x/w not step-4 aligned: {x},{w}"
        assert y % 2 == 0 and h % 2 == 0, f"AOI y/h not step-2 aligned: {y},{h}"
        assert w >= 32 and h >= 4, f"AOI below minimum: {w}x{h}"
        assert x + w <= SENSOR_W and y + h <= SENSOR_H, "AOI outside sensor"
        S.aoi = (x, y, w, h)
        return IS_SUCCESS
    if cmd == IS_AOI_IMAGE_GET_AOI:
        param.s32X, param.s32Y, param.s32Width, param.s32Height = S.aoi
        return IS_SUCCESS
    if cmd == IS_AOI_IMAGE_GET_SIZE_MIN:
        param.s32Width, param.s32Height = 32, 4
        return IS_SUCCESS
    if cmd == IS_AOI_IMAGE_GET_SIZE_INC:
        param.s32Width, param.s32Height = 4, 2
        return IS_SUCCESS
    if cmd == IS_AOI_IMAGE_GET_POS_INC:
        param.s32X, param.s32Y = 4, 2
        return IS_SUCCESS
    return -1


def is_GetImageInfo(hCam, mem_id, info, size):
    info.u64FrameNumber = S.frames_now()
    info.u64TimestampDevice = int(time.perf_counter() * 1e7)  # 100ns ticks
    info.dwImageBuffersInUse = 1
    return IS_SUCCESS
