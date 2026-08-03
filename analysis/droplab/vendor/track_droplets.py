
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import defaultdict


# -----------------------------
# Utilities: endpoints & loading
# -----------------------------

def compute_endpoints(x, y, length, width, angle_deg):
    """
    Compute streak endpoint coordinates (start, end) from centroid (x,y),
    length, width, and angle (degrees). "Start" and "End" are along the length axis.
    """
    angle = np.deg2rad(angle_deg)
    half_len = 0.5 * float(length)
    dx = half_len * np.cos(angle)
    dy = half_len * np.sin(angle)
    start = (float(x) - dx, float(y) - dy)
    end   = (float(x) + dx, float(y) + dy)
    return start, end


def ensure_endpoints_in_array(data):
    """
    Ensure the structured numpy array has:
      length_start_x/y and length_end_x/y.
    If missing, compute from x, y, length, angle.

    Returns a structured array with endpoints present.
    """
    names = set(data.dtype.names)
    need = {'length_start_x', 'length_start_y', 'length_end_x', 'length_end_y'}
    if need.issubset(names):
        return data

    # Build new dtype appending endpoint fields
    base_descr = list(data.dtype.descr)
    endpoint_descr = [
        ('length_start_x', 'f4'),
        ('length_start_y', 'f4'),
        ('length_end_x',   'f4'),
        ('length_end_y',   'f4'),
    ]
    new_dtype = np.dtype(base_descr + endpoint_descr)
    out = np.empty(len(data), dtype=new_dtype)

    # copy original fields
    for name in data.dtype.names:
        out[name] = data[name]

    # compute endpoints
    for i, d in enumerate(data):
        (sx, sy), (ex, ey) = compute_endpoints(d['x'], d['y'], d['length'], d['width'], d['angle'])
        out['length_start_x'][i] = sx
        out['length_start_y'][i] = sy
        out['length_end_x'][i]   = ex
        out['length_end_y'][i]   = ey

    return out


def load_tracking_data(filepath):
    """
    Load .npy detections and ensure endpoints exist.
    Expected base fields: frame, droplet_id, x, y, length, width, angle, (in_roi optional)
    """
    data = np.load(filepath, allow_pickle=False)
    data = ensure_endpoints_in_array(data)
    return data


# -------------------------------------------------------
# Endpoint-based linker (one-to-one, end(prev) -> start(curr))
# -------------------------------------------------------

def _endpoint_distance(p, q):
    dx, dy = p[0] - q[0], p[1] - q[1]
    return float(np.hypot(dx, dy))


def link_droplets_by_endpoints(data, max_distance=50.0):
    """
    Link detections across consecutive frames using endpoint proximity.

    Rules:
      - For each pair of consecutive frames (t, t+1):
          * Build candidates for linking each prev detection to each current detection.
          * Distance is computed **from prev END** to both endpoints of current:
                d1 = || prev_end - curr_start ||
                d2 = || prev_end - curr_end   ||
            For the **first link of a track** (when prev has no prior link), we allow
            using **prev START** as well if it yields a closer candidate and then
            swap the prev orientation.
          * Accept candidates with distance <= max_distance.
          * Sort all candidates by distance and do greedy one-to-one matching
            (no merges/splits).
          * If the chosen endpoint on the current detection is "end", swap its start/end
            so that the link becomes prev_end -> curr_start.
      - Detections in the first frame start new tracks.
      - Unmatched detections in later frames start new tracks.
      - Tracks can end (droplets can disappear).

    Returns:
      endpoints_dict: {
         track_id: [
            {"frame": f, "start": (sx, sy), "end": (ex, ey)},
            ...
         ],  # sorted by frame
         ...
      }
    """
    frames = np.unique(data['frame'])
    frames.sort()
    if len(frames) == 0:
        return {}

    # Track store: track_id -> list[ dict(frame, start, end) ]
    tracks = {}
    # For each detection index, store its assigned track_id
    track_assignments = np.full(len(data), -1, dtype=np.int32)

    # Build index list per frame
    frame_to_indices = {f: np.where(data['frame'] == f)[0] for f in frames}

    next_tid = 0

    # Initialize tracks from first frame; orientation as-is for now
    first = frames[0]
    for idx in frame_to_indices[first]:
        tid = next_tid; next_tid += 1
        tracks[tid] = []
        r = data[idx]
        tracks[tid].append({
            "frame": int(r['frame']),
            "start": (float(r['length_start_x']), float(r['length_start_y'])),
            "end":   (float(r['length_end_x']),   float(r['length_end_y'])),
        })
        track_assignments[idx] = tid

    # Helper: get oriented prev_end for a track's last record
    def _get_prev_end(tid):
        last = tracks[tid][-1]
        return last["end"]

    # Helper: swap orientation of a record in-place (start <-> end)
    def _swap_record_orientation(rec):
        rec["start"], rec["end"] = rec["end"], rec["start"]

    # Iterate over frame pairs
    for i in range(1, len(frames)):
        prev_f, curr_f = frames[i-1], frames[i]
        prev_idxs = frame_to_indices[prev_f]
        curr_idxs = frame_to_indices[curr_f]

        if len(curr_idxs) == 0:
            continue

        # Build candidates: (dist, prev_local, curr_local, prev_tid, use_prev_end, curr_endpoint_is_start)
        # We also keep the information needed to swap orientations if needed.
        candidates = []

        # Map prev local index -> tid and whether it's the first link (track length == 1)
        prev_local_info = {}
        for p_loc, p_idx in enumerate(prev_idxs):
            tid = int(track_assignments[p_idx])
            is_first_link = (len(tracks[tid]) == 1)
            prev_local_info[p_loc] = (tid, is_first_link)

        # Build all candidate matches within threshold
        for p_loc, p_idx in enumerate(prev_idxs):
            tid, is_first_link = prev_local_info[p_loc]
            prev_rec = tracks[tid][-1]

            # candidates normally from prev_end only
            prev_end = prev_rec["end"]
            for c_loc, c_idx in enumerate(curr_idxs):
                curr = data[c_idx]
                curr_start = (float(curr['length_start_x']), float(curr['length_start_y']))
                curr_end   = (float(curr['length_end_x']),   float(curr['length_end_y']))

                # Distances from prev_end to curr endpoints
                d_es = _endpoint_distance(prev_end, curr_start)
                d_ee = _endpoint_distance(prev_end, curr_end)

                # choose best (start vs end) for current under prev_end
                if d_es <= d_ee:
                    best_d = d_es
                    curr_is_start = True
                else:
                    best_d = d_ee
                    curr_is_start = False

                if best_d <= max_distance:
                    candidates.append((best_d, p_loc, c_loc, True, curr_is_start))  # True => using prev_end

                # If first link in this track, allow prev_start as well (to fix initial orientation)
                if is_first_link:
                    prev_start = prev_rec["start"]
                    d_ss = _endpoint_distance(prev_start, curr_start)
                    d_se = _endpoint_distance(prev_start, curr_end)
                    if d_ss <= d_se:
                        best_d2 = d_ss
                        curr_is_start2 = True
                    else:
                        best_d2 = d_se
                        curr_is_start2 = False
                    if best_d2 <= max_distance:
                        # False => using prev_start (we'll swap prev record if chosen)
                        candidates.append((best_d2, p_loc, c_loc, False, curr_is_start2))

        # Greedy one-to-one by ascending distance
        candidates.sort(key=lambda x: x[0])
        assigned_prev = set()
        assigned_curr = set()

        for best_d, p_loc, c_loc, use_prev_end, curr_is_start in candidates:
            if p_loc in assigned_prev or c_loc in assigned_curr:
                continue

            p_idx = prev_idxs[p_loc]
            c_idx = curr_idxs[c_loc]
            tid = int(track_assignments[p_idx])

            # If using prev_start (first link), swap the last record of that track so its end becomes the endpoint used
            if not use_prev_end:
                _swap_record_orientation(tracks[tid][-1])  # now prev_rec.end is the endpoint we just used

            # For the current detection, if the best endpoint was "end", we swap it so that it becomes the "start"
            curr = data[c_idx]
            curr_start = (float(curr['length_start_x']), float(curr['length_start_y']))
            curr_end   = (float(curr['length_end_x']),   float(curr['length_end_y']))
            if curr_is_start:
                new_start, new_end = curr_start, curr_end
            else:
                new_start, new_end = curr_end, curr_start  # swap so that link is prev_end -> curr_start

            # Append oriented current record to the same track
            tracks[tid].append({
                "frame": int(curr['frame']),
                "start": new_start,
                "end":   new_end,
            })
            track_assignments[c_idx] = tid

            assigned_prev.add(p_loc)
            assigned_curr.add(c_loc)

        # Start new tracks for any unassigned current detections
        for c_loc, c_idx in enumerate(curr_idxs):
            if c_loc not in assigned_curr and track_assignments[c_idx] == -1:
                new_tid = next_tid; next_tid += 1
                tracks[new_tid] = []
                r = data[c_idx]
                tracks[new_tid].append({
                    "frame": int(r['frame']),
                    "start": (float(r['length_start_x']), float(r['length_start_y'])),
                    "end":   (float(r['length_end_x']),   float(r['length_end_y'])),
                })
                track_assignments[c_idx] = new_tid

    # Sort records within each track by frame
    for tid in tracks.keys():
        tracks[tid].sort(key=lambda rec: rec["frame"])
    return tracks


# -----------------------------
# Save track data
# -----------------------------

def save_track_data(endpoints_by_track, filepath="track_data.npz",
                    fps=30.0, pixels_per_um=0.1878,
                    max_distance=None, min_track_length=None):
    """
    Flatten endpoints_by_track into a structured array with pre-computed
    velocities and save as .npz for downstream analysis.

    Velocity is computed from the linking displacement (prev_end -> curr_start)
    divided by the inter-frame time.  First observation per track gets NaN.
    """
    rows = []
    dt_frame = 1.0 / fps

    for tid, recs in endpoints_by_track.items():
        for i, rec in enumerate(recs):
            if i == 0:
                vx = vy = speed = direction = np.nan
            else:
                prev = recs[i - 1]
                dx = rec["start"][0] - prev["end"][0]
                dy = rec["start"][1] - prev["end"][1]
                n_frames = rec["frame"] - prev["frame"]
                dt = n_frames * dt_frame
                if dt > 0:
                    vx = dx / dt
                    vy = dy / dt
                else:
                    vx = vy = np.nan
                speed = np.hypot(vx, vy)
                direction = np.degrees(np.arctan2(vy, vx)) % 360.0
            rows.append((tid, rec["frame"],
                         rec["start"][0], rec["start"][1],
                         rec["end"][0], rec["end"][1],
                         vx, vy, speed, direction))

    dtype = [('track_id', 'i4'), ('frame', 'i4'),
             ('start_x', 'f4'), ('start_y', 'f4'),
             ('end_x', 'f4'), ('end_y', 'f4'),
             ('vx', 'f4'), ('vy', 'f4'),
             ('speed', 'f4'), ('direction', 'f4')]

    track_array = np.array(rows, dtype=dtype) if rows else np.array([], dtype=dtype)

    metadata = json.dumps({
        "fps": fps,
        "pixels_per_um": pixels_per_um,
        "max_distance": max_distance,
        "min_track_length": min_track_length,
        "n_tracks": len(endpoints_by_track),
        "n_observations": len(rows),
    })

    np.savez(filepath,
             track_observations=track_array,
             metadata=np.array(metadata))
    print(f"[save] Track data saved to: {filepath} "
          f"({len(endpoints_by_track)} tracks, {len(rows)} observations)")


# -----------------------------
# Plotting helpers
# -----------------------------

def _dynamic_heads(mag, base_hw=6.0, base_hl=9.0):
    if mag <= 0:
        return base_hw, base_hl
    scale = np.clip(mag / 50.0, 0.5, 2.0)
    return base_hw * scale, base_hl * scale




def plot_endpoints_vectors(
    endpoints_dict,
    canvas=(1920, 1080),
    invert_y=True,
    arrow_alpha=0.85,
    linewidth=1.4,
    title="Endpoint Algorithm — All Streak Vectors",
    show_ids=True,
    id_at="last_end",          # 'first_end' | 'last_end' | 'first_start' | 'last_start'
    id_offset_px=(8, -8),
    id_prefix="ID ",
    run=True,
    plot_first_n=10,           # <— NEW: only draw the first N tracks (default 10). None => all.
    sort_ids=True              # <— NEW: ordering for "first N": True = sorted; False = insertion order
):
    """
    Plot streak vectors (start -> end) per frame for each track_id.
    Draw only the first N tracks (plot_first_n), and show IDs for all plotted tracks.

    Parameters
    ----------
    endpoints_dict : dict
        {track_id: [{"start": (sx, sy), "end": (ex, ey)}, ...], ...}
    canvas : (W, H)
    invert_y : bool
    arrow_alpha : float
    linewidth : float
    title : str
    show_ids : bool
    id_at : str
        Where to anchor the ID label for each track: 'first_end'|'last_end'|'first_start'|'last_start'
    id_offset_px : (dx, dy)
        Pixel offset for the label text.
    id_prefix : str
        Text prefix for labels, e.g., "ID ".
    run : bool
        If False, do nothing.
    plot_first_n : int or None
        If an int N (>=0), draw only the first N track_ids by chosen order. If None, draw all.
    sort_ids : bool
        If True, order track_ids by sorted ascending value; if False, preserve dict insertion order.
    """
    if not run:
        return

    W, H = canvas
    fig, ax = plt.subplots(figsize=(10, 5.625))
    ax.set_xlim(0, W)
    if invert_y:
        ax.set_ylim(H, 0)
    else:
        ax.set_ylim(0, H)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)" + (" (top-down)" if invert_y else ""))
    ax.set_title(title)

    # Choose ordering and subset to plot
    track_ids_all = list(endpoints_dict.keys())
    if sort_ids:
        track_ids_all = sorted(track_ids_all)
    if plot_first_n is None:
        track_ids_to_plot = track_ids_all
    else:
        track_ids_to_plot = track_ids_all[:max(0, int(plot_first_n))]

    cmap = plt.cm.get_cmap('tab20', max(1, len(track_ids_to_plot)))

    for idx, tid in enumerate(track_ids_to_plot):
        recs = endpoints_dict.get(tid, [])
        if not recs:
            continue

        color = cmap(idx)
        # Draw vectors
        for r in recs:
            (sx, sy), (ex, ey) = r["start"], r["end"]
            dx, dy = ex - sx, ey - sy
            mag = float(np.hypot(dx, dy))
            hw, hl = _dynamic_heads(mag)  # assumes your helper exists
            ax.arrow(
                sx, sy, dx, dy,
                head_width=hw, head_length=hl, length_includes_head=True,
                fc=color, ec=color, alpha=arrow_alpha, linewidth=linewidth
            )

        # Label all plotted tracks (if enabled)
        if show_ids:
            ldx, ldy = id_offset_px
            anchor = {
                "first_end":   recs[0]["end"],
                "last_end":    recs[-1]["end"],
                "first_start": recs[0]["start"],
                "last_start":  recs[-1]["start"],
            }.get(id_at, recs[-1]["end"])
            ax.text(
                anchor[0] + ldx, anchor[1] + ldy,
                f"{id_prefix}{tid}",
                color=color, fontsize=9, fontweight='bold',
                ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', boxstyle='round,pad=0.2')
            )

    plt.tight_layout()
    plt.show()




# --------------------------------
# Streamlines: build + visualize
# --------------------------------

def build_streamlines(endpoints_dict):
    """
    Build streamline arrays per track by concatenating (start, end) per frame:
      X: [start_x_f0, end_x_f0, start_x_f1, end_x_f1, ...]
      Y: [start_y_f0, end_y_f0, start_y_f1, end_y_f1, ...]
    """
    streamlines = {}
    for tid, recs in endpoints_dict.items():
        if not recs:
            continue
        xs, ys = [], []
        for r in recs:
            (sx, sy), (ex, ey) = r["start"], r["end"]
            xs.extend([sx, ex])
            ys.extend([sy, ey])
        streamlines[tid] = {
            "frames": [r["frame"] for r in recs],
            "X": np.array(xs, dtype=float),
            "Y": np.array(ys, dtype=float),
        }
    return streamlines




def plot_streamlines(
    streamlines,
    canvas=(1920, 1080),
    invert_y=True,
    line_width=2.0,
    line_alpha=0.95,
    arrow_alpha=0.95,
    title="Streamlines — Start/End Concatenation",
    show_ids=True,
    id_offset_px=(8, -8),
    id_prefix="ID ",
    run=True,
    plot_first_n=10,           # <— NEW: only draw the first N tracks (default 10). None => all.
    sort_ids=True              # <— NEW: ordering for "first N": True = sorted; False = insertion order
):
    """
    Plot concatenated streamlines for each track; draw a single arrowhead on the last streak.
    Draw only the first N tracks (plot_first_n), and show IDs for all plotted tracks.

    Parameters
    ----------
    streamlines : dict
        {track_id: {"X": np.ndarray, "Y": np.ndarray}, ...}, where X and Y are
        continuous polylines for the full streamline.
    canvas : (W, H)
    invert_y : bool
        If True, y-axis increases downward (image-style coordinates).
    line_width : float
        Width of the streamline polyline.
    line_alpha : float
        Alpha for the line polyline.
    arrow_alpha : float
        Alpha for the arrowhead on the last segment.
    title : str
        Plot title.
    show_ids : bool
        Whether to draw ID labels.
    id_offset_px : (dx, dy)
        Pixel offset for the ID text anchor.
    id_prefix : str
        Prefix for the ID text, e.g., "ID ".
    run : bool
        If False, do nothing and return.
    plot_first_n : int or None
        If an int N (>=0), draw only the first N track_ids by chosen order. If None, draw all.
    sort_ids : bool
        If True, order track_ids by sorted ascending value; if False, preserve dict insertion order.
    """
    if not run:
        return

    W, H = canvas
    fig, ax = plt.subplots(figsize=(10, 5.625))
    ax.set_xlim(0, W)
    if invert_y:
        ax.set_ylim(H, 0)
    else:
        ax.set_ylim(0, H)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)" + (" (top-down)" if invert_y else ""))
    ax.set_title(title)

    # Choose ordering and subset to plot
    track_ids_all = list(streamlines.keys())
    if sort_ids:
        track_ids_all = sorted(track_ids_all)
    if plot_first_n is None:
        track_ids_to_plot = track_ids_all
    else:
        track_ids_to_plot = track_ids_all[:max(0, int(plot_first_n))]

    cmap = plt.cm.get_cmap('tab20', max(1, len(track_ids_to_plot)))

    for idx, tid in enumerate(track_ids_to_plot):
        s = streamlines.get(tid, None)
        if s is None:
            continue

        X, Y = s["X"], s["Y"]
        if X.size < 2:
            continue

        color = cmap(idx)

        # Continuous polyline
        ax.plot(X, Y, '-', color=color, linewidth=line_width, alpha=line_alpha)

        # Final arrow on last segment (use last two points)
        sxN, syN = X[-2], Y[-2]
        exN, eyN = X[-1], Y[-1]
        dx, dy = exN - sxN, eyN - syN
        mag = float(np.hypot(dx, dy))
        hw, hl = _dynamic_heads(mag)  # assumes your helper exists
        ax.arrow(
            sxN, syN, dx, dy,
            head_width=hw, head_length=hl, length_includes_head=True,
            fc=color, ec=color, alpha=arrow_alpha, linewidth=line_width
        )

        # Label all plotted tracks (if enabled)
        if show_ids:
            ldx, ldy = id_offset_px
            ax.text(
                exN + ldx, eyN + ldy,
                f"{id_prefix}{tid}",
                color=color, fontsize=9, fontweight='bold',
                ha='left', va='center',
                bbox=dict(facecolor='white', alpha=0.65, edgecolor='none', boxstyle='round,pad=0.2')
            )

    plt.tight_layout()
    plt.show()




# ------------------------------------------------
# Summed track visualization with vectors overlaid
# ------------------------------------------------

def sum_track_frames(video_path, track_records, padding=50):
    """
    Sum video frames for a specific track, cropped around the droplet region.
    Only sums the frames where the droplet exists. No processing - raw color frames.

    Args:
        video_path: Path to the video file
        track_records: List of track records [{"frame": f, "start": (sx,sy), "end": (ex,ey)}, ...]
        padding: Pixels to add around the bounding box of all track positions

    Returns:
        summed_image: Summed color image (float32, BGR)
        bbox: (x_min, y_min, x_max, y_max) bounding box in original frame coords
        frame_count: Number of frames summed
    """
    if not track_records:
        return None, None, 0

    # Get all coordinates to determine bounding box
    all_x = []
    all_y = []
    frames = []
    for rec in track_records:
        sx, sy = rec["start"]
        ex, ey = rec["end"]
        all_x.extend([sx, ex])
        all_y.extend([sy, ey])
        frames.append(rec["frame"])

    # Compute bounding box with padding
    x_min = int(max(0, min(all_x) - padding))
    x_max = int(max(all_x) + padding)
    y_min = int(max(0, min(all_y) - padding))
    y_max = int(max(all_y) + padding)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Clamp to frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    x_max = min(x_max, frame_width)
    y_max = min(y_max, frame_height)

    bbox = (x_min, y_min, x_max, y_max)

    # Sum frames (only the frames where the droplet exists)
    summed = None
    frame_count = 0

    for frame_num in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        # Crop and keep color (BGR)
        cropped = frame[y_min:y_max, x_min:x_max].astype(np.float32)

        if summed is None:
            summed = cropped
        else:
            summed += cropped

        frame_count += 1

    cap.release()
    return summed, bbox, frame_count


def plot_track_with_summed_image(
    video_path,
    track_id,
    endpoints_dict,
    padding=50,
    invert_y=True,
    arrow_alpha=0.9,
    linewidth=2.0,
    show_streamline=True,
    title=None,
    save_path=None
):
    """
    Plot a summed image of a droplet across frames with streamline/vectors overlaid.
    Shows the raw color frames summed (green droplets on black background).

    Args:
        video_path: Path to the video file
        track_id: Which track to visualize
        endpoints_dict: Dictionary from link_droplets_by_endpoints
        padding: Pixels around the track bounding box
        invert_y: Whether to invert y-axis (image coordinates)
        arrow_alpha: Alpha for arrows
        linewidth: Line width for vectors
        show_streamline: If True, draw continuous streamline; if False, draw individual vectors
        title: Plot title (auto-generated if None)
        save_path: If provided, save figure to this path
    """
    if track_id not in endpoints_dict:
        print(f"Track {track_id} not found in endpoints_dict")
        return

    track_records = endpoints_dict[track_id]
    if not track_records:
        print(f"Track {track_id} has no records")
        return

    # Sum frames for this track (raw color, no processing)
    summed, bbox, frame_count = sum_track_frames(
        video_path, track_records, padding=padding
    )

    if summed is None or frame_count == 0:
        print(f"Could not sum frames for track {track_id}")
        return

    x_min, y_min, x_max, y_max = bbox

    # Normalize summed image for display and convert BGR to RGB
    max_val = np.max(summed)
    if max_val > 0:
        display_img = (summed / max_val * 255).astype(np.uint8)
    else:
        display_img = summed.astype(np.uint8)

    # Convert BGR to RGB for matplotlib
    display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Show summed image (no colormap - actual colors)
    extent = [x_min, x_max, y_max, y_min] if invert_y else [x_min, x_max, y_min, y_max]
    ax.imshow(display_img, extent=extent, aspect='equal')

    # Draw vectors/streamline
    color = 'cyan'

    if show_streamline:
        # Draw continuous streamline
        xs, ys = [], []
        for rec in track_records:
            sx, sy = rec["start"]
            ex, ey = rec["end"]
            xs.extend([sx, ex])
            ys.extend([sy, ey])

        ax.plot(xs, ys, '-', color=color, linewidth=linewidth, alpha=arrow_alpha)

        # Arrow on last segment
        if len(xs) >= 2:
            dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
            mag = np.hypot(dx, dy)
            hw, hl = _dynamic_heads(mag)
            ax.arrow(
                xs[-2], ys[-2], dx, dy,
                head_width=hw, head_length=hl, length_includes_head=True,
                fc=color, ec=color, alpha=arrow_alpha, linewidth=linewidth
            )
    else:
        # Draw individual vectors for each frame
        for rec in track_records:
            sx, sy = rec["start"]
            ex, ey = rec["end"]
            dx, dy = ex - sx, ey - sy
            mag = np.hypot(dx, dy)
            hw, hl = _dynamic_heads(mag)
            ax.arrow(
                sx, sy, dx, dy,
                head_width=hw, head_length=hl, length_includes_head=True,
                fc=color, ec=color, alpha=arrow_alpha, linewidth=linewidth
            )

    # Labels and title
    frames_str = f"{track_records[0]['frame']}-{track_records[-1]['frame']}"
    if title is None:
        title = f"Track {track_id} — Summed over {frame_count} frames ({frames_str})"
    ax.set_title(title)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    if invert_y:
        ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


def plot_all_tracks_summed(
    video_path,
    endpoints_dict,
    min_track_length=2,
    max_tracks=10,
    padding=50,
    threshold=None,
    save_dir=None,
    **kwargs
):
    """
    Plot summed images for multiple tracks.

    Args:
        video_path: Path to the video file
        endpoints_dict: Dictionary from link_droplets_by_endpoints
        min_track_length: Only plot tracks with at least this many frames
        max_tracks: Maximum number of tracks to plot
        padding: Pixels around each track bounding box
        threshold: Threshold for summing
        save_dir: If provided, save figures to this directory
        **kwargs: Additional arguments passed to plot_track_with_summed_image
    """
    import os

    # Filter and sort tracks by length (longest first)
    filtered = {
        tid: recs for tid, recs in endpoints_dict.items()
        if len(recs) >= min_track_length
    }
    sorted_tids = sorted(filtered.keys(), key=lambda t: len(filtered[t]), reverse=True)

    if max_tracks is not None:
        sorted_tids = sorted_tids[:max_tracks]

    print(f"Plotting {len(sorted_tids)} tracks (min_length={min_track_length})...")

    for tid in sorted_tids:
        save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"track_{tid:04d}.png")

        plot_track_with_summed_image(
            video_path, tid, endpoints_dict,
            padding=padding, threshold=threshold,
            save_path=save_path, **kwargs
        )


# -----------------
# Example main flow
# -----------------

def main(
    input_file="droplet_data.npy",
    video_path=None,
    canvas=(1920, 1080),
    invert_y=True,
    max_distance=10.0,
    min_track_length=1,
    plot_tracks=True,
    plot_summed_track_id=None,
    summed_padding=50,
    summed_save_path=None,
    fps=30.0,
    pixels_per_um=0.1878,
):
    """
    Main tracking and visualization pipeline.

    Args:
        input_file: Path to droplet_data.npy
        video_path: Path to video file (required for summed track plot)
        canvas: (width, height) of the frame
        invert_y: Use image coordinates (origin top-left)
        max_distance: Linking radius in pixels
        min_track_length: Minimum frames for a track to be included
        plot_tracks: Plot endpoint vectors and streamlines
        plot_summed_track_id: Track ID to plot summed image for (None to skip)
        summed_padding: Pixels around track bounding box
        summed_save_path: Path to save summed track plot (None to just display)

    Returns:
        endpoints_by_track: Dictionary of linked tracks
    """
    # Load & ensure endpoints exist
    data = load_tracking_data(input_file)

    if max_distance is None:
        import os as _os
        n_droplets = []
        for i in range(75):
            n = main(input_file=input_file, max_distance=i, plot_tracks=False)
            n_droplets.append(n)

        # Save the scan results alongside the data file
        _dv_dir = _os.path.dirname(_os.path.abspath(input_file))
        _dv_base = _os.path.splitext(_os.path.basename(input_file))[0]
        if _dv_base.endswith("_droplet_data"):
            _dv_base = _dv_base[: -len("_droplet_data")]
        _dv_path = _os.path.join(_dv_dir, f"{_dv_base}_droplets_vs_distance.npy")
        np.save(_dv_path, np.array(n_droplets, dtype=np.int32))

        import set_distance_threshold as sdt
        best_threshold, n_tracked = sdt.plot_droplets_vs_distance(
            data=np.array(n_droplets, dtype=np.int32),
            fit_mode="joint",
            k_sigma=2.0,
            two_sided=False,
            show_plot=True
        )
        print(f"Suggested best threshold: {best_threshold} (tracks {n_tracked} droplets)")
        max_distance = best_threshold

    # Link across frames by endpoint proximity
    endpoints_by_track = link_droplets_by_endpoints(data, max_distance=max_distance)

    # Filter short tracks
    if min_track_length > 1:
        endpoints_by_track = {
            tid: recs for tid, recs in endpoints_by_track.items()
            if len(recs) >= min_track_length
        }

    print(f"Total linked droplets (tracks) found: {len(endpoints_by_track)}")

    # Plot endpoint vectors
    plot_endpoints_vectors(
        endpoints_by_track,
        canvas=canvas,
        invert_y=invert_y,
        arrow_alpha=0.85,
        linewidth=1.6,
        title="Endpoint Algorithm — All Streak Vectors (Linked)",
        show_ids=True,
        id_at="last_end",
        id_offset_px=(8, -8),
        id_prefix="ID ",
        run=plot_tracks
    )

    # Build and plot streamlines
    streamlines = build_streamlines(endpoints_by_track)
    plot_streamlines(
        streamlines,
        canvas=canvas,
        invert_y=invert_y,
        line_width=2.2,
        line_alpha=0.95,
        arrow_alpha=0.95,
        title="Streamlines — Start/End Concatenation (Linked)",
        show_ids=True,
        id_offset_px=(8, -8),
        id_prefix="ID ",
        run=plot_tracks
    )

    # Plot summed image for a specific track ID
    if plot_summed_track_id is not None:
        if video_path is None:
            print("Warning: video_path required for summed track plot. Skipping.")
        elif plot_summed_track_id not in endpoints_by_track:
            print(f"Track ID {plot_summed_track_id} not found. Available IDs: {list(endpoints_by_track.keys())[:20]}...")
        else:
            plot_track_with_summed_image(
                video_path,
                plot_summed_track_id,
                endpoints_by_track,
                padding=summed_padding,
                invert_y=invert_y,
                save_path=summed_save_path
            )

    # Save track data alongside the droplet data file
    import os as _os
    _data_dir = _os.path.dirname(_os.path.abspath(input_file))
    _basename = _os.path.splitext(_os.path.basename(input_file))[0]
    # Strip _droplet_data suffix if present, then append _track_data
    if _basename.endswith("_droplet_data"):
        _basename = _basename[: -len("_droplet_data")]
    track_npz_path = _os.path.join(_data_dir, f"{_basename}_track_data.npz")
    save_track_data(endpoints_by_track, track_npz_path,
                    fps=fps, pixels_per_um=pixels_per_um,
                    max_distance=max_distance,
                    min_track_length=min_track_length)

    # Run track-based analysis
    import analyze_tracks
    track_data = analyze_tracks.load_track_data(track_npz_path)
    analyze_tracks.analyze_tracks(track_data)

    return endpoints_by_track


if __name__ == "__main__":
    # === USER SETTINGS ===
    # Update VIDEO_BASENAME to match your video filename (without extension)
    VIDEO_BASENAME = "water_constexp2"
    import os
    _root = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(_root, "data", VIDEO_BASENAME, f"{VIDEO_BASENAME}_droplet_data.npy")
    video_path = None  # Path to video file (optional, for summed track visualization)
    canvas = (1920, 1080)
    invert_y = True           # image-like coordinates (origin top-left)
    max_distance = 24         # linking radius in pixels
    min_track_length = 1      # set >1 to keep only multi-frame tracks

    # Summed track visualization for a specific droplet
    plot_summed_track_id = 5  # Set to a track ID to plot its summed image (None to skip)
    summed_padding = 50       # Pixels around track bounding box
    summed_save_path = None   # Path to save (or None to just display)

    main(
        input_file=input_file,
        video_path=video_path,
        canvas=canvas,
        invert_y=invert_y,
        max_distance=max_distance,
        min_track_length=min_track_length,
        plot_summed_track_id=plot_summed_track_id,
        summed_padding=summed_padding,
        summed_save_path=summed_save_path,
        fps=30.0,
        pixels_per_um=0.1878,
    )
