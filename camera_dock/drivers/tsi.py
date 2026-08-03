"""Process-wide owner of the Thorlabs TSI SDK.

The TSI SDK is a **per-process singleton**: one ``TLCameraSDK`` per process, no
matter how many cameras. It used to be created inside ``ZeluxCamera.connect()``,
one per camera object, which is why the dock could only ever drive one Zelux --
a second camera either failed outright or, worse, tripped a recovery path that
force-cleared the singleton flag out from under the first camera. This module
owns the SDK instead, so N cameras share one.

Three non-obvious facts drive the design; each is load-bearing.

1. **An open camera does not keep the SDK alive.** ``TLCameraSDK.open_camera``
   returns ``TLCamera._create(self._sdk, handle)`` -- the raw ``cdll`` object,
   not the wrapper -- so a ``TLCamera`` holds no reference back. Since
   ``TLCameraSDK.__del__`` calls ``dispose()`` -> ``tl_camera_close_sdk()``,
   letting the SDK become garbage closes it underneath every live camera. The
   module global ``_sdk`` is therefore the *only* strong reference, and it is
   never dropped while handles are open.

2. **``dispose()`` can wedge the class flag permanently.** It calls
   ``tl_camera_close_sdk()`` first and only clears ``_is_sdk_open`` *after* the
   error check, so one non-zero close leaves the flag set with no working SDK
   and nothing can ever open it again. So we simply never dispose on refcount
   zero -- only once, at process teardown, and even then we *neuter* rather than
   close if any handle is still open (a real close under the engine's daemon
   poll threads is a segfault). The force-reset workaround survives in exactly
   one place, gated on provably-no-handle-in-this-process.

3. **``discover_available_cameras()`` must run before ``open_camera()``.** It is
   an instance method that populates the SDK's internal table; without it
   ``open_camera`` fails with error 1004 "Serial number not found". Measured on
   the rig: it is safe to call while other cameras are streaming.

Locking is two-tier, and the rule is about what the lock is held *across*:
``TSI_LOCK`` (here) serializes individual **native calls** process-wide, because
the TSI DLL is not concurrency-safe. Each driver keeps its own per-camera lock
for the *sequences* those calls belong to (a whole ``set_roi``, a frame copy).
Holding this lock across a sequence would stall every other camera; holding it
across one call costs tens of microseconds. Lock order is always
``camera._lock -> TSI_LOCK``, never the reverse, and TSI_LOCK is never held
while calling back into driver, session or roster code.

Every acquisition is bounded, so a wedged DLL costs one camera and a stream of
503s rather than freezing every request handler in the process.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Optional, Tuple

# Default location of the Thorlabs native DLLs on a standard ThorCam install.
_DEFAULT_DLL_DIR = r"C:\Program Files\Thorlabs\Scientific Imaging\ThorCam"

TSI_LOCK = threading.RLock()

_sdk = None                       # the ONLY strong reference to the TLCameraSDK
_open: dict = {}                  # serial -> TLCamera currently open here
_roster: list = []                # serials from the last discovery
_table_suspect = False            # a dispose() failed; handle table is unknown
_dll_done = False


class SdkUnavailable(RuntimeError):
    """The Thorlabs SDK could not be loaded or opened."""


class TsiBusy(RuntimeError):
    """Could not get the TSI lock in time -- the DLL is wedged or contended."""


class CameraInUse(RuntimeError):
    """That serial is already open in this process."""


class TableSuspect(RuntimeError):
    """A previous dispose failed; opening more cameras risks a segfault."""


def _ensure_dll_dir() -> None:
    """Put the Thorlabs native-DLL dir on the search path. Runs once per process.

    BOTH steps are required: ``os.add_dll_directory`` lets ctypes find the main
    DLL; the PATH prepend lets that DLL's own dependencies resolve. Guarded
    because it used to run on every connect(), leaking a directory cookie and
    re-prepending the same path each time -- unbounded once cameras can be
    loaded and unloaded at runtime, and inherited by any child process.
    """
    global _dll_done
    if _dll_done:
        return
    dll_dir = os.environ.get("THORLABS_TSI_DLL_DIR", _DEFAULT_DLL_DIR)
    if os.path.isdir(dll_dir):
        try:
            os.add_dll_directory(dll_dir)      # Windows-only; absent on Linux
        except (OSError, AttributeError):
            pass
        os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    _dll_done = True


@contextmanager
def guard(timeout: float = 2.0):
    """Hold TSI_LOCK across ONE native call. Raises TsiBusy rather than hanging.

    Never wrap a sequence, a memcpy or a sleep in this -- that is what the
    per-camera lock is for, and it is the difference between microseconds of
    contention and stalling every other camera.
    """
    if not TSI_LOCK.acquire(timeout=timeout):
        raise TsiBusy(
            "the Thorlabs SDK is busy or wedged (waited "
            f"{timeout:g}s). If this persists, restart the DAQ.")
    try:
        yield
    finally:
        TSI_LOCK.release()


def _import_sdk():
    _ensure_dll_dir()
    try:
        from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
        return TLCameraSDK
    except (ImportError, OSError) as exc:
        raise SdkUnavailable(
            "Could not load the Thorlabs SDK (thorlabs_tsi_sdk). Install the "
            "wheel from the Scientific Camera SDK and ensure the native DLLs "
            "are discoverable (THORLABS_TSI_DLL_DIR or a ThorCam install). "
            "Original error: " + str(exc)) from exc


def _ensure_sdk():
    """Create the process-wide SDK if needed. Caller holds TSI_LOCK."""
    global _sdk
    if _sdk is not None:
        return _sdk
    TLCameraSDK = _import_sdk()
    try:
        _sdk = TLCameraSDK()
    except Exception:
        # A prior failed dispose leaves the singleton flag set without a working
        # SDK. Only safe to force-clear when we hold no handle at all -- with a
        # camera open, clearing it would let a second SDK be built over a live
        # handle table. That is exactly the bug this module exists to remove.
        if getattr(TLCameraSDK, "_is_sdk_open", False) and not _open:
            TLCameraSDK._is_sdk_open = False
            _sdk = TLCameraSDK()
        else:
            raise
    return _sdk


def discover(force: bool = False) -> list:
    """Serials of attached Thorlabs cameras. Populates the SDK's internal table.

    Must run at least once before ``open_camera`` (error 1004 otherwise).
    Measured safe while other cameras stream. Never raises: an unavailable SDK
    yields an empty list, matching the drivers' never-raise contract.
    """
    global _roster
    if not force and _roster:
        return list(_roster)
    try:
        with guard():
            sdk = _ensure_sdk()
            _roster = [str(s) for s in sdk.discover_available_cameras()]
    except Exception:
        return list(_roster)
    return list(_roster)


def open_serials() -> set:
    """Serials currently open in this process."""
    with guard():
        return set(_open)


def open_camera(serial: Optional[str] = None) -> Tuple[object, str]:
    """Open one camera and return (TLCamera, resolved_serial).

    Raises rather than returning a stub -- the caller (a driver's connect()) is
    the layer that owns the never-raise policy.
    """
    with guard():
        if _table_suspect:
            raise TableSuspect(
                "a Thorlabs camera failed to close cleanly, so the SDK's handle "
                "table is in an unknown state. Restart the DAQ before loading "
                "more cameras.")
        _ensure_sdk()

        target = str(serial).strip() if serial else ""
        if not target:
            # With three identical CS165MU, "just take the first one" silently
            # binds an arbitrary camera -- and would re-bind the SAME one for
            # every session. Only allow it when nothing is open yet.
            if _open:
                raise RuntimeError(
                    "a serial number is required when another Thorlabs camera "
                    "is already loaded (several identical cameras are attached)")
            roster = discover(force=True)
            if not roster:
                raise RuntimeError("No Thorlabs cameras found.")
            target = roster[0]

        if target in _open:
            raise CameraInUse(f"camera {target} is already loaded")
        if target not in _roster:
            discover(force=True)
        if target not in _roster:
            raise RuntimeError(
                f"Camera {target} not found. Available: {', '.join(_roster)}")

        # Re-raised untouched, with no global state mutated: a failure to open
        # one camera must leave the SDK and every other camera exactly as it was.
        cam = _sdk.open_camera(target)
        _open[target] = cam
        return cam, target


def close_camera(tlcam, serial: Optional[str] = None) -> bool:
    """Close one camera. NEVER raises. True if it closed cleanly.

    On failure the SDK's internal handle table may be corrupted, and the
    reference implementation's remedy (rebuild the whole SDK) is unavailable
    with other cameras open. So we quarantine the whole process: keep the SDK,
    keep the other cameras streaming, and refuse further opens.
    """
    global _table_suspect
    ok = True
    try:
        with guard():
            try:
                if getattr(tlcam, "is_armed", False):
                    tlcam.disarm()
            except Exception:
                pass
            try:
                tlcam.dispose()
            except Exception:
                ok = False
                # TLCamera.dispose() raises WITHOUT setting _disposed, so __del__
                # would retry the same failing close later, off-thread. Defuse it.
                try:
                    tlcam._disposed = True
                except Exception:
                    pass
                _table_suspect = True
            for s, c in list(_open.items()):
                if c is tlcam or (serial and s == str(serial)):
                    _open.pop(s, None)
    except TsiBusy:
        return False
    return ok


def status() -> dict:
    """Diagnostics for the panel: what the SDK owner thinks is going on."""
    try:
        with guard(timeout=0.25):
            return {"sdk_open": _sdk is not None, "roster": list(_roster),
                    "open": sorted(_open), "table_suspect": _table_suspect}
    except TsiBusy:
        return {"sdk_open": _sdk is not None, "roster": list(_roster),
                "open": None, "table_suspect": _table_suspect, "busy": True}


def shutdown() -> None:
    """Release the SDK at process teardown. NEVER raises. Call after stop_all.

    If any handle is still open the acquisition threads are daemons that were
    never joined, so a real close would be a segfault. Neuter the wrapper
    instead (dispose() short-circuits on _disposed) and let the process exit
    reclaim it.
    """
    global _sdk
    try:
        with guard(timeout=5.0):
            if _sdk is None:
                return
            if _open:
                print(f"[tsi] {len(_open)} camera(s) still open at shutdown "
                      f"({', '.join(sorted(_open))}) — leaving the SDK to the "
                      f"process exit rather than closing under them", flush=True)
                try:
                    _sdk._disposed = True
                except Exception:
                    pass
                _sdk = None
                return
            try:
                _sdk.dispose()
            except Exception as exc:
                print(f"[tsi] SDK dispose failed: {exc}", flush=True)
            _sdk = None
    except Exception:
        pass
