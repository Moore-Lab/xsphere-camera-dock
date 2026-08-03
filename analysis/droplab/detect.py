"""Droplet detection core — headless port of count-droplets analyze_droplets.py.

The algorithm is kept EXACTLY as in Moore-Lab/count-droplets (same cv2 calls,
same parameters, same droplet fields) so results are interchangeable with the
original pipeline:

    1. grayscale -> CLAHE(clipLimit=2.0, tile 8x8)
    2. GaussianBlur 3x3
    3. binary threshold at *threshold*
    4. morphological CLOSE, 3x3 kernel, 1 iteration
    5. external contours, area-filtered to [min_area, max_area]
    6. minAreaRect (>=5 points) for length/width/angle, boundingRect fallback

Detection always runs on the ENTIRE frame; the spatial ROI only tags each
droplet with ``in_roi`` (used for density normalization and for filtering what
gets saved), exactly like the original.
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np

Progress = Optional[Callable[[int, int], None]]     # (done, total)


def detect_droplets(frame, roi, threshold=50, min_area=10, max_area=100,
                    return_processed=False):
    """Detect droplets in one BGR (or gray) frame.

    *roi* is (x_start, x_stop, y_start, y_stop) in pixels — only used for the
    per-droplet ``in_roi`` tag.

    Returns (count, droplets[, binary]) where each droplet dict has:
    x, y, radius, length, width, angle, in_roi,
    length_start, length_end, width_start, width_end   (each an (x, y) tuple).
    """
    x_start, x_stop, y_start, y_stop = roi

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    droplets = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue
        # Upstream counts every area-passing contour, then only skips the
        # droplet RECORD for degenerate (zero-moment) contours — matters when
        # min_area <= 0 lets area-0 contours through.
        count += 1
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        radius = int(np.sqrt(area / np.pi))

        if len(contour) >= 5:
            (rect_cx, rect_cy), (rect_w, rect_h), angle = cv2.minAreaRect(contour)
            if rect_w >= rect_h:
                length, width = rect_w, rect_h
            else:
                length, width = rect_h, rect_w
                angle = angle + 90          # angle always follows the length axis

            angle_rad = np.deg2rad(angle)
            len_dx = (length / 2.0) * np.cos(angle_rad)
            len_dy = (length / 2.0) * np.sin(angle_rad)
            length_start = (rect_cx - len_dx, rect_cy - len_dy)
            length_end = (rect_cx + len_dx, rect_cy + len_dy)

            wid_rad = angle_rad + np.pi / 2
            wid_dx = (width / 2.0) * np.cos(wid_rad)
            wid_dy = (width / 2.0) * np.sin(wid_rad)
            width_start = (rect_cx - wid_dx, rect_cy - wid_dy)
            width_end = (rect_cx + wid_dx, rect_cy + wid_dy)
        else:
            _, _, w_bound, h_bound = cv2.boundingRect(contour)
            length = max(w_bound, h_bound)
            width = min(w_bound, h_bound)
            angle = 0.0
            if w_bound >= h_bound:
                length_start = (cx - length / 2, cy)
                length_end = (cx + length / 2, cy)
                width_start = (cx, cy - width / 2)
                width_end = (cx, cy + width / 2)
            else:
                length_start = (cx, cy - length / 2)
                length_end = (cx, cy + length / 2)
                width_start = (cx - width / 2, cy)
                width_end = (cx + width / 2, cy)

        droplets.append({
            'x': cx, 'y': cy, 'radius': radius,
            'length': length, 'width': width, 'angle': angle,
            'in_roi': (x_start <= cx < x_stop) and (y_start <= cy < y_stop),
            'length_start': length_start, 'length_end': length_end,
            'width_start': width_start, 'width_end': width_end,
        })

    if return_processed:
        return count, droplets, binary
    return count, droplets


# ── overlay rendering (browser preview) ─────────────────────────────────────

def draw_overlay(frame, droplets, roi, *, draw_roi=True):
    """Draw the count-droplets annotation scheme onto a BGR frame (in place).

    Lime circles at radius*10 (as in the original GUI), red length line,
    blue width line, red ROI rectangle.
    """
    x_start, x_stop, y_start, y_stop = roi
    for d in droplets:
        cv2.circle(frame, (int(d['x']), int(d['y'])),
                   max(int(d['radius'] * 10), 2), (0, 255, 0), 1)
        ls, le = d['length_start'], d['length_end']
        cv2.line(frame, (int(ls[0]), int(ls[1])), (int(le[0]), int(le[1])),
                 (0, 0, 255), 1)
        ws, we = d['width_start'], d['width_end']
        cv2.line(frame, (int(ws[0]), int(ws[1])), (int(we[0]), int(we[1])),
                 (255, 0, 0), 1)
    if draw_roi:
        cv2.rectangle(frame, (int(x_start), int(y_start)),
                      (int(x_stop) - 1, int(y_stop) - 1), (0, 0, 255), 2)
    return frame


# ── persistence / sum view ──────────────────────────────────────────────────

def sum_view(read_frame, start_frame, end_frame, threshold, roi, frame_size,
             *, analysis_only=True, step=1, progress: Progress = None):
    """Accumulate above-threshold pixels over a frame range (persistence view).

    *read_frame(idx)* returns a BGR frame or None. *frame_size* is (w, h).
    ``analysis_only=True`` restricts to the spatial ROI (re-embedded into a
    black full frame, like the original); ``step`` decimates for speed.
    Returns (bgr_image, frames_accumulated).
    """
    x_start, x_stop, y_start, y_stop = roi
    w, h = frame_size
    acc = None
    n = 0
    idxs = range(start_frame, end_frame, max(step, 1))
    for i, idx in enumerate(idxs):
        frame = read_frame(idx)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if analysis_only:
            gray = gray[y_start:y_stop, x_start:x_stop]
        keep = np.where(gray >= threshold, gray, 0).astype(np.float32) / 255.0
        acc = keep if acc is None else acc + keep
        n += 1
        if progress is not None and n % 25 == 0:
            progress(i + 1, len(idxs))

    if acc is None or n == 0:
        return None, 0
    peak = float(acc.max())
    disp = (acc / peak * 255).astype(np.uint8) if peak > 0 else acc.astype(np.uint8)
    bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
    if analysis_only:
        full = np.zeros((h, w, 3), dtype=np.uint8)
        full[y_start:y_stop, x_start:x_stop] = bgr
        return full, n
    return bgr, n


# ── full detection run ──────────────────────────────────────────────────────

def run_detection(read_frame, start_frame, end_frame, fps, roi, frame_size,
                  threshold=50, min_area=10, max_area=100,
                  progress: Progress = None, cancelled=None):
    """Detect droplets over [start_frame, end_frame) — port of analyze().

    Returns the frame_data list in the original's exact per-frame shape.
    *cancelled* is an optional zero-arg callable checked between frames.
    """
    x_start, x_stop, y_start, y_stop = roi
    w, h = frame_size
    frame_data = []
    total = max(end_frame - start_frame, 1)

    for current in range(start_frame, end_frame):
        if cancelled is not None and cancelled():
            break
        frame = read_frame(current)
        if frame is None:
            break
        count, droplets = detect_droplets(frame, roi, threshold, min_area, max_area)
        frame_data.append({
            'frame_number': current,
            'time_seconds': current / fps,
            'droplet_count': count,
            'droplet_count_in_roi': sum(1 for d in droplets if d['in_roi']),
            'droplets': droplets,
            'roi_coordinates': {'x_start': x_start, 'x_stop': x_stop,
                                'y_start': y_start, 'y_stop': y_stop},
            'frame_dimensions': {'width': w, 'height': h},
        })
        if progress is not None and len(frame_data) % 20 == 0:
            progress(current - start_frame + 1, total)

    if progress is not None:
        progress(len(frame_data), total)
    return frame_data
