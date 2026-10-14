"""Hardware smoke test for the Hayear HY-1136 (UVC) driver.

Run with the camera plugged in::

    python smoke_test.py

Opens the camera, reports resolution, grabs a short burst, and prints the measured
frame rate. No GUI, no files written.
"""

from __future__ import annotations

import time

from hayear.camera import HayearCamera


def main() -> None:
    with HayearCamera() as cam:
        print("Connected:", cam.device_info)
        print("Resolution (w x h):", cam.sensor_size())

        n = 60
        print(f"\nGrabbing {n} frames...")
        cam.start()
        t0 = time.perf_counter()
        shape = None
        for _ in range(n):
            shape = cam.grab().shape
        dt = time.perf_counter() - t0
        cam.stop()

        print(f"Grabbed {n} frames of shape {shape} in {dt:.3f} s => {n / dt:.1f} fps")
        print("\nSmoke test OK.")


if __name__ == "__main__":
    main()
