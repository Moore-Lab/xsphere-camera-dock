"""Validate the Thorlabs uc480 fallback against the REAL DCx driver stack.

Needs: ``pip install pyueye`` and ThorCam with DCx Camera Support installed
(``uc480_64.dll``); no IDS Software Suite. No camera needs to be attached —
this only checks DLL binding, function resolution, and a live enumeration
call. Skips cleanly when the prerequisites are missing.

Run with::

    python tests/test_uc480_real.py
"""
import ctypes.util
import importlib.util
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))        # ids-ueye/tests
PKG_DIR = os.path.dirname(HERE)                          # ids-ueye
# Drop the script dir (auto-added as sys.path[0]) — it holds the MOCK pyueye
# used by test_driver_mock.py; this test must see only a real install.
sys.path = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != HERE]
sys.path.insert(0, PKG_DIR)

if importlib.util.find_spec("pyueye") is None:
    print("SKIP: pyueye is not installed (pip install pyueye).")
    sys.exit(0)
if ctypes.util.find_library("ueye_api_64") or ctypes.util.find_library("ueye_api"):
    print("SKIP: the IDS Software Suite is installed — the fallback under test "
          "only runs when ueye_api.dll is absent (the driver already works here).")
    sys.exit(0)

failures = []


def check(name, cond, detail=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


print("1. fallback binds the Thorlabs uc480_64.dll")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    from ids_ueye.camera import _import_ueye
    try:
        ueye = _import_ueye()
    except RuntimeError as exc:
        print(f"SKIP: no uc480 DLL found either — install ThorCam/DCx. ({exc})")
        sys.exit(0)
libfile = str(getattr(ueye, "get_dll_file", ""))
check("bound uc480_64.dll", "uc480" in libfile.lower(), libfile)

print("2. every function the driver uses resolved in the DLL")
NEEDED = [
    "is_InitCamera", "is_ExitCamera", "is_GetNumberOfCameras", "is_GetCameraList",
    "is_GetSensorInfo", "is_GetCameraInfo", "is_SetColorMode", "is_AllocImageMem",
    "is_AddToSequence", "is_ClearSequence", "is_FreeImageMem", "is_GetImageMemPitch",
    "is_EnableEvent", "is_DisableEvent", "is_GetActSeqBuf",
    "is_UnlockSeqBuf", "is_SetExternalTrigger", "is_CaptureVideo", "is_StopLiveVideo",
    "is_SetFrameRate", "is_GetFrameTimeRange", "is_GetFramesPerSecond",
    "is_Exposure", "is_PixelClock", "is_AOI", "is_GetImageInfo",
    "is_InitEvent", "is_ExitEvent",   # uc480 event path (is_WaitEvent is absent)
]
missing = {fn for w in caught if "not found" in str(w.message)
           for fn in NEEDED if f"'{fn}'" in str(w.message)}
check("no needed function missing", not missing, str(sorted(missing)))

print("3. live SDK call through the DLL")
import ctypes
n = ctypes.c_int(-1)
check("is_GetNumberOfCameras succeeds",
      ueye.is_GetNumberOfCameras(n) == ueye.IS_SUCCESS and n.value >= 0, str(n.value))
print(f"     (cameras currently attached: {n.value})")

print("4. driver selects the uc480 event path")
from ids_ueye.camera import _FrameEvent
check("winevent mode selected", _FrameEvent(ueye, None)._use_wait_event is False)

print()
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
