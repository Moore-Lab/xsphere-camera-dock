"""Hardware smoke test for the IDS uEye driver.

Run with the camera plugged in and the IDS Software Suite (uEye driver) plus
``pyueye`` installed::

    python smoke_test.py

It enumerates devices, opens the camera, reports info/ranges/pixel clocks, sets
a modest exposure + frame rate, grabs a short burst, and prints the measured
frame rate. No GUI, no files written.
"""

from __future__ import annotations

import time

from ids_ueye.camera import IDSUEye, list_devices


def main() -> None:
    devices = list_devices()
    if not devices:
        print("No IDS uEye cameras found (or pyueye / the uEye driver is not installed).")
    for d in devices:
        print(f"Device {d['device_id']}: {d['model']} SN:{d['serial']}"
              f"{' [IN USE]' if d['in_use'] else ''}")

    with IDSUEye() as cam:
        print("\nConnected:", cam.device_info)
        print("Sensor size (w x h):", cam.sensor_size())
        print("Bit depth:", cam.bit_depth)
        print("Pixel clocks (MHz):", cam.get_pixel_clock_list(),
              "current:", cam.get_pixel_clock())
        print("Exposure range (us):", cam.exposure_range())
        print("Frame-rate range (fps):", cam.frame_rate_range())
        print("ROI range:", cam.roi_range())

        cam.set_frame_rate(30.0)
        cam.set_exposure(5000.0)   # 5 ms (after fps: uEye clamps exposure to the period)
        print(f"\nSet target fps -> {cam.get_frame_rate():.1f}, "
              f"exposure -> {cam.get_exposure():.1f} us")

        n = 60
        print(f"\nGrabbing {n} frames...")
        cam.start()
        t0 = time.perf_counter()
        shape = None
        for _ in range(n):
            frame = cam.grab()
            shape = frame.shape
        dt = time.perf_counter() - t0
        measured = cam.resulting_frame_rate()
        cam.stop()

        print(f"Grabbed {n} frames of shape {shape} in {dt:.3f} s "
              f"=> {n / dt:.1f} fps wall-clock, {measured:.1f} fps SDK-measured")
        print("\nSmoke test OK.")


if __name__ == "__main__":
    main()
