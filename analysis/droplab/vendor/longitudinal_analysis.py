"""
Longitudinal Analysis of Droplet Data

Loads a log file produced by analyze_droplets.py, partitions the droplet data
into time or frame chunks, runs analyze_distributions.analyze_independent() on
each chunk, extracts summary statistics (raw moments — no fits), and saves:

  - Per-chunk distribution plots in  plots/<video_basename>/
  - A longitudinal numpy data file   data/<video_basename>/longitudinal_analysis_<basename>_<timestamp>.npy
  - Longitudinal time-series plots   plots/<video_basename>/longitudinal_<param>.png

Usage: edit the USER SETTINGS block in the __main__ section and run:
    python longitudinal_analysis.py
"""

import os
import json
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for saving
import matplotlib.pyplot as plt

import analyze_distributions


# =============================================================================
# Helpers
# =============================================================================

def _get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def load_log(log_path: str) -> dict:
    """Load an analysis log JSON produced by analyze_droplets.save_log()."""
    with open(log_path, 'r') as f:
        return json.load(f)


def build_chunks_from_frames(total_frames: int, chunk_size_frames: int):
    """
    Partition [0, total_frames) into chunks of chunk_size_frames.
    Returns list of (start_frame, end_frame) tuples (end_frame exclusive).
    """
    chunks = []
    start = 0
    while start < total_frames:
        end = min(start + chunk_size_frames, total_frames)
        chunks.append((start, end))
        start = end
    return chunks


def build_chunks_from_time(total_frames: int, fps: float, chunk_duration_s: float):
    """
    Partition frames into chunks each spanning approximately chunk_duration_s seconds.

    Smart rounding: ideal chunk size = chunk_duration_s * fps.  If this is not an
    integer, distribute the remainder so that no frames are left unanalyzed.
    Each chunk gets either floor(ideal) or ceil(ideal) frames.

    Returns list of (start_frame, end_frame) tuples (end_frame exclusive).
    """
    ideal = chunk_duration_s * fps
    base = int(ideal)           # floor
    extra = ideal - base        # fractional part
    if base == 0:
        base = 1

    n_chunks = int(np.ceil(total_frames / ideal)) if ideal > 0 else total_frames
    chunks = []
    start = 0
    accumulated_extra = 0.0

    while start < total_frames:
        accumulated_extra += extra
        if accumulated_extra >= 1.0:
            size = base + 1
            accumulated_extra -= 1.0
        else:
            size = base
        end = min(start + size, total_frames)
        chunks.append((start, end))
        start = end

    return chunks


def _extract_moments(per_droplet: dict, metadata: dict) -> dict:
    """
    Extract raw statistical moments from per-droplet data.
    Returns dict of scalar values (no fits used).
    """
    counts  = np.asarray(per_droplet['counts'])
    lengths = np.asarray(per_droplet['lengths'])
    widths  = np.asarray(per_droplet['widths'])
    angles  = np.asarray(per_droplet['angles'])

    pixels_per_um = float(metadata.get('pixels_per_um', 0.1878))
    fps = float(metadata.get('fps', 30.0))

    # exposure for velocity calculation
    if 'exposure_time_s' in metadata and metadata['exposure_time_s'] is not None:
        exposure_s = float(metadata['exposure_time_s'])
    else:
        exposure_s = 1.0 / fps

    pixels_per_m = pixels_per_um * 1e6
    px_to_m = 1.0 / pixels_per_m

    # Volume for density
    roi_h_px = int(metadata.get('roi_h_px', metadata.get('frame_height_px',
                    metadata.get('roi_height', 0))))
    roi_w_px = int(metadata.get('roi_w_px', metadata.get('frame_width_px',
                    metadata.get('roi_width', 0))))
    if (roi_h_px <= 0) and (per_droplet.get('y') is not None) and (len(per_droplet['y']) > 0):
        roi_h_px = int(np.max(per_droplet['y']) - np.min(per_droplet['y']) + 1)
    if (roi_w_px <= 0) and (per_droplet.get('x') is not None) and (len(per_droplet['x']) > 0):
        roi_w_px = int(np.max(per_droplet['x']) - np.min(per_droplet['x']) + 1)
    mm_per_px = 1.0 / (pixels_per_um * 1000.0)
    r_mm = 0.5 * roi_w_px * mm_per_px  # R = horizontal = x extent
    z_mm = 1.0 * roi_h_px * mm_per_px  # Z = vertical   = y extent
    volume_mm3 = np.pi * (r_mm ** 2) * z_mm if (r_mm > 0 and z_mm > 0) else np.nan

    # --- Counts / density ---
    mean_count = float(np.mean(counts)) if counts.size > 0 else np.nan
    std_count  = float(np.std(counts))  if counts.size > 0 else np.nan
    mean_density_mm3 = mean_count / volume_mm3 if (np.isfinite(mean_count) and np.isfinite(volume_mm3) and volume_mm3 > 0) else np.nan
    std_density_mm3  = std_count  / volume_mm3 if (np.isfinite(std_count)  and np.isfinite(volume_mm3) and volume_mm3 > 0) else np.nan

    # --- Width ---
    mean_w_px = float(np.mean(widths)) if widths.size > 0 else np.nan
    std_w_px  = float(np.std(widths))  if widths.size > 0 else np.nan
    mean_w_um = mean_w_px / pixels_per_um if np.isfinite(mean_w_px) else np.nan
    std_w_um  = std_w_px  / pixels_per_um if np.isfinite(std_w_px)  else np.nan

    # --- Speed ---
    valid_mask = np.isfinite(lengths) & (lengths >= 0)
    v_mps = (lengths[valid_mask] * px_to_m) / exposure_s
    mean_v_mps = float(np.mean(v_mps)) if v_mps.size > 0 else np.nan
    std_v_mps  = float(np.std(v_mps))  if v_mps.size > 0 else np.nan

    # --- Projected velocities ---
    angles_rad = np.deg2rad(angles[valid_mask])
    v_z = np.abs(v_mps * np.sin(angles_rad))  # Z = vertical   = sin component
    v_r = np.abs(v_mps * np.cos(angles_rad))  # R = horizontal = cos component
    mean_vy_mps = float(np.mean(v_r)) if v_r.size > 0 else np.nan  # stored as vy (R component)
    std_vy_mps  = float(np.std(v_r))  if v_r.size > 0 else np.nan
    mean_vz_mps = float(np.mean(v_z)) if v_z.size > 0 else np.nan
    std_vz_mps  = float(np.std(v_z))  if v_z.size > 0 else np.nan

    # --- Wind direction and anisotropy (from angle distribution) ---
    if angles_rad.size > 0:
        # Circular mean of orientations (fold into [0,pi))
        angles_folded = angles_rad % np.pi
        sin_sum = np.sum(np.sin(2 * angles_folded))
        cos_sum = np.sum(np.cos(2 * angles_folded))
        circular_mean_rad = 0.5 * np.arctan2(sin_sum, cos_sum)
        circular_mean_deg = float(np.degrees(circular_mean_rad)) % 180.0

        # Anisotropy: max/min of binned count histogram
        n_bins = 12
        angle_edges = np.linspace(0, np.pi, n_bins + 1)
        count_hist, _ = np.histogram(angles_folded, bins=angle_edges)
        aniso = float(np.max(count_hist)) / float(np.min(count_hist)) if np.min(count_hist) > 0 else np.inf
    else:
        circular_mean_deg = np.nan
        aniso = np.nan

    return {
        # Counts / density
        'mean_count':        mean_count,
        'std_count':         std_count,
        'mean_density_mm3':  mean_density_mm3,
        'std_density_mm3':   std_density_mm3,
        # Width
        'mean_width_px':     mean_w_px,
        'std_width_px':      std_w_px,
        'mean_width_um':     mean_w_um,
        'std_width_um':      std_w_um,
        # Speed
        'mean_speed_mps':    mean_v_mps,
        'std_speed_mps':     std_v_mps,
        # Projected velocities
        'mean_vy_mps':       mean_vy_mps,
        'std_vy_mps':        std_vy_mps,
        'mean_vz_mps':       mean_vz_mps,
        'std_vz_mps':        std_vz_mps,
        # Wind / orientation
        'wind_direction_deg': circular_mean_deg,
        'anisotropy_ratio':   aniso,
    }


# =============================================================================
# Longitudinal plots
# =============================================================================

def _save_longitudinal_plot(chunk_times, values, stds, ylabel_main, ylabel_std,
                             title, save_path,
                             twin_values=None, twin_ylabel=None, twin_color='steelblue'):
    """
    Save a 2-row longitudinal plot: parameter (top) and std dev (bottom).
    If twin_values is provided, add a twin y-axis to the top panel.
    """
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Top: parameter
    ax_top.plot(chunk_times, values, 'o-', color='black', lw=1.5, markersize=5)
    ax_top.set_ylabel(ylabel_main, fontsize=11)
    ax_top.set_title(title, fontsize=13, fontweight='bold')
    ax_top.grid(True, alpha=0.3)

    if twin_values is not None:
        ax_twin = ax_top.twinx()
        ax_twin.plot(chunk_times, twin_values, 's--', color=twin_color, lw=1.2, markersize=4, alpha=0.7)
        ax_twin.set_ylabel(twin_ylabel, fontsize=11, color=twin_color)
        ax_twin.tick_params(axis='y', labelcolor=twin_color)

    # Bottom: std dev
    ax_bot.plot(chunk_times, stds, 'o-', color='gray', lw=1.5, markersize=5)
    ax_bot.set_ylabel(ylabel_std, fontsize=11)
    ax_bot.set_xlabel('Chunk mid-time (s)', fontsize=11)
    ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [longitudinal plot] saved: {save_path}")


def save_all_longitudinal_plots(chunk_mid_times, rows, plots_dir, basename):
    """Save one longitudinal plot per parameter group."""
    os.makedirs(plots_dir, exist_ok=True)

    def _col(key):
        return np.array([r[key] for r in rows], dtype=float)

    # 1. Count / density
    _save_longitudinal_plot(
        chunk_mid_times,
        _col('mean_count'), _col('std_count'),
        ylabel_main='Mean droplets per frame',
        ylabel_std='Std dev (droplets)',
        title='Droplet Count per Frame',
        save_path=os.path.join(plots_dir, f"{basename}_longitudinal_count.png"),
        twin_values=_col('mean_density_mm3'),
        twin_ylabel='Density (droplets/mm³)',
        twin_color='steelblue'
    )

    # 2. Width
    _save_longitudinal_plot(
        chunk_mid_times,
        _col('mean_width_px'), _col('std_width_px'),
        ylabel_main='Mean streak width (px)',
        ylabel_std='Std dev (px)',
        title='Streak Width',
        save_path=os.path.join(plots_dir, f"{basename}_longitudinal_width.png"),
        twin_values=_col('mean_width_um'),
        twin_ylabel='Mean width (µm)',
        twin_color='darkorange'
    )

    # 3. Speed
    def _mmps(key):
        return _col(key) * 1e3

    _save_longitudinal_plot(
        chunk_mid_times,
        _mmps('mean_speed_mps'), _mmps('std_speed_mps'),
        ylabel_main='Mean speed (mm/s)',
        ylabel_std='Std dev speed (mm/s)',
        title='Droplet Speed',
        save_path=os.path.join(plots_dir, f"{basename}_longitudinal_speed.png"),
    )

    # 4. Radial velocity (vy)
    _save_longitudinal_plot(
        chunk_mid_times,
        _mmps('mean_vy_mps'), _mmps('std_vy_mps'),
        ylabel_main='Mean |v_r| (mm/s)',
        ylabel_std='Std dev |v_r| (mm/s)',
        title='Radial Velocity Component',
        save_path=os.path.join(plots_dir, f"{basename}_longitudinal_vy.png"),
    )

    # 5. Axial velocity (vz)
    _save_longitudinal_plot(
        chunk_mid_times,
        _mmps('mean_vz_mps'), _mmps('std_vz_mps'),
        ylabel_main='Mean |v_z| (mm/s)',
        ylabel_std='Std dev |v_z| (mm/s)',
        title='Axial Velocity Component',
        save_path=os.path.join(plots_dir, f"{basename}_longitudinal_vz.png"),
    )

    # 6. Wind direction and anisotropy
    _save_longitudinal_plot(
        chunk_mid_times,
        _col('wind_direction_deg'), np.full(len(rows), np.nan),
        ylabel_main='Circular mean angle (°)',
        ylabel_std='(N/A)',
        title='Wind Direction',
        save_path=os.path.join(plots_dir, f"{basename}_longitudinal_wind_direction.png"),
        twin_values=_col('anisotropy_ratio'),
        twin_ylabel='Anisotropy ratio',
        twin_color='crimson'
    )


# =============================================================================
# Main analysis routine
# =============================================================================

def run_longitudinal_analysis(
    log_path: str,
    chunk_size_frames: int = None,
    chunk_duration_s: float = None,
    start_utc: float = None,
    density_kg_m3: float = 3520.0,
):
    """
    Run the longitudinal analysis.

    Parameters
    ----------
    log_path : str
        Path to the log JSON produced by analyze_droplets.save_log().
    chunk_size_frames : int, optional
        Chunk size in frames. Mutually exclusive with chunk_duration_s.
    chunk_duration_s : float, optional
        Chunk duration in seconds. Mutually exclusive with chunk_size_frames.
    start_utc : float, optional
        UTC Unix timestamp for the start of the analysis (for time-stamping data).
    density_kg_m3 : float
        Material density in kg/m³ (default 3520 for xenon microspheres).
    """
    if (chunk_size_frames is None) == (chunk_duration_s is None):
        raise ValueError("Specify exactly one of chunk_size_frames or chunk_duration_s.")

    # ---- Load log ----
    log = load_log(log_path)
    params   = log['analysis_parameters']
    basename = log['video_basename']
    fps      = float(params['fps'])
    start_frame_abs = int(params['start_time'] * fps)
    end_frame_abs   = int(params['end_time']   * fps)
    total_frames = end_frame_abs - start_frame_abs

    droplet_data_path = log['droplet_data_path']
    results_json_path = log['results_json_path']

    print(f"[longitudinal] Video: {basename}")
    print(f"[longitudinal] Log: {log_path}")
    print(f"[longitudinal] Data: {droplet_data_path}")
    print(f"[longitudinal] Frames: {start_frame_abs} – {end_frame_abs} ({total_frames} frames @ {fps} fps)")

    # ---- Build chunks ----
    if chunk_size_frames is not None:
        chunk_defs = build_chunks_from_frames(total_frames, chunk_size_frames)
        print(f"[longitudinal] Chunk size: {chunk_size_frames} frames → {len(chunk_defs)} chunks")
    else:
        chunk_defs = build_chunks_from_time(total_frames, fps, chunk_duration_s)
        print(f"[longitudinal] Chunk duration: {chunk_duration_s} s → {len(chunk_defs)} chunks")

    # ---- Load full per-droplet data ----
    full_data = analyze_distributions.load_droplet_data(
        droplet_data_path, results_json=results_json_path
    )
    metadata = full_data['metadata']

    # Raw arrays (indexed by droplet, with absolute frame numbers)
    frames_all  = np.load(droplet_data_path, allow_pickle=False)['frame']
    loaded_raw  = np.load(droplet_data_path, allow_pickle=False)
    frames_all  = loaded_raw['frame'].astype(int)
    lengths_all = loaded_raw['length'].astype(float)
    widths_all  = loaded_raw['width'].astype(float)
    angles_all  = loaded_raw['angle'].astype(float)
    x_all       = loaded_raw['x'].astype(float)
    y_all       = loaded_raw['y'].astype(float)

    # ---- Directories ----
    root_dir   = _get_project_root()
    plots_root = os.path.join(root_dir, "plots", basename)
    os.makedirs(plots_root, exist_ok=True)
    data_dir   = os.path.join(root_dir, "data", basename)
    os.makedirs(data_dir, exist_ok=True)

    # ---- Analyse each chunk ----
    rows = []

    for chunk_idx, (c_start, c_end) in enumerate(chunk_defs):
        # Absolute frame range for this chunk
        f_start = start_frame_abs + c_start
        f_end   = start_frame_abs + c_end - 1  # inclusive end

        # Mask droplets in this frame range
        mask = (frames_all >= f_start) & (frames_all <= f_end)

        # Build per-frame counts for this chunk
        chunk_frames_range = np.arange(f_start, f_end + 1)
        all_frame_counts = []
        for fr in chunk_frames_range:
            all_frame_counts.append(np.sum(frames_all == fr))
        counts_chunk = np.array(all_frame_counts, dtype=int)

        chunk_data = {
            'counts':   counts_chunk,
            'lengths':  lengths_all[mask],
            'widths':   widths_all[mask],
            'angles':   angles_all[mask],
            'x':        x_all[mask],
            'y':        y_all[mask],
            'metadata': metadata,
        }

        n_droplets = int(np.sum(mask))
        mid_frame  = (f_start + f_end) / 2.0
        mid_time_s = mid_frame / fps
        chunk_start_time_s = f_start / fps
        chunk_end_time_s   = (f_end + 1) / fps

        print(f"  Chunk {chunk_idx:3d}: frames {f_start}–{f_end}  "
              f"({chunk_start_time_s:.2f}–{chunk_end_time_s:.2f} s)  "
              f"droplets={n_droplets}")

        if n_droplets == 0:
            print(f"    [skip] no droplets in chunk {chunk_idx}")
            # Store NaN row
            row = {k: np.nan for k in [
                'mean_count','std_count','mean_density_mm3','std_density_mm3',
                'mean_width_px','std_width_px','mean_width_um','std_width_um',
                'mean_speed_mps','std_speed_mps',
                'mean_vy_mps','std_vy_mps','mean_vz_mps','std_vz_mps',
                'wind_direction_deg','anisotropy_ratio',
            ]}
        else:
            # Save distribution plot for this chunk
            plot_path = os.path.join(plots_root, f"chunk_{chunk_idx:04d}.png")
            try:
                analyze_distributions.analyze_independent(
                    chunk_data,
                    density_kg_m3=density_kg_m3,
                    save_path=plot_path,
                )
            except Exception as e:
                print(f"    [warn] plot failed for chunk {chunk_idx}: {e}")

            # Extract raw moments
            row = _extract_moments(chunk_data, metadata)

        row['chunk_index']        = chunk_idx
        row['chunk_start_frame']  = f_start
        row['chunk_end_frame']    = f_end
        row['chunk_start_time_s'] = chunk_start_time_s
        row['chunk_end_time_s']   = chunk_end_time_s
        row['chunk_mid_time_s']   = mid_time_s
        row['n_droplets']         = n_droplets
        rows.append(row)

    # ---- Build output numpy structure ----
    n = len(rows)
    dtype = np.dtype([
        ('chunk_index',        'i4'),
        ('chunk_start_frame',  'i4'),
        ('chunk_end_frame',    'i4'),
        ('chunk_start_time_s', 'f8'),
        ('chunk_end_time_s',   'f8'),
        ('chunk_mid_time_s',   'f8'),
        ('n_droplets',         'i4'),
        # Count / density
        ('mean_count',         'f8'),
        ('std_count',          'f8'),
        ('mean_density_mm3',   'f8'),
        ('std_density_mm3',    'f8'),
        # Width
        ('mean_width_px',      'f8'),
        ('std_width_px',       'f8'),
        ('mean_width_um',      'f8'),
        ('std_width_um',       'f8'),
        # Speed
        ('mean_speed_mps',     'f8'),
        ('std_speed_mps',      'f8'),
        # Projected velocities
        ('mean_vy_mps',        'f8'),
        ('std_vy_mps',         'f8'),
        ('mean_vz_mps',        'f8'),
        ('std_vz_mps',         'f8'),
        # Wind
        ('wind_direction_deg', 'f8'),
        ('anisotropy_ratio',   'f8'),
        # UTC metadata
        ('start_utc',          'f8'),
    ])

    arr = np.zeros(n, dtype=dtype)
    scalar_fields = [f[0] for f in dtype.descr if f[0] != 'start_utc']
    for i, row in enumerate(rows):
        for field in scalar_fields:
            val = row.get(field, np.nan)
            arr[field][i] = val if val is not None else np.nan
        arr['start_utc'][i] = float(start_utc) if start_utc is not None else np.nan

    # ---- Save numpy file ----
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    out_filename = f"longitudinal_analysis_{basename}_{timestamp}.npy"
    out_path = os.path.join(data_dir, out_filename)
    np.save(out_path, arr)
    print(f"\n[longitudinal] Data saved to: {out_path}")

    # ---- Save longitudinal plots ----
    chunk_mid_times = arr['chunk_mid_time_s']
    rows_for_plots = [dict(zip(dtype.names, row)) for row in arr]
    save_all_longitudinal_plots(chunk_mid_times, rows_for_plots, plots_root, basename)

    print(f"[longitudinal] Done. {n} chunks processed.")
    return arr, out_path


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # USER SETTINGS — edit these
    # =========================================================================

    # Path to the log file produced by analyze_droplets.py
    LOG_PATH = os.path.join(_get_project_root(), "logs", "xenon_fill_10_3-7-2026_2026-03-12_10-04-18.json")

    # Chunk size: specify EITHER frames OR seconds (set the other to None)
    CHUNK_SIZE_FRAMES   = None    # e.g. 300  (frames per chunk)
    CHUNK_DURATION_S    = 5.0     # e.g. 5.0  (seconds per chunk)

    # UTC Unix timestamp for the start of the acquisition
    # Obtain from: import datetime; datetime.datetime(2026, 3, 11, 9, 16, 43, tzinfo=datetime.timezone.utc).timestamp()
    START_UTC = None              # e.g. 1741687003.0  (replace with actual timestamp)

    # Material density (kg/m^3)
    DENSITY_KG_M3 = 3520.0

    # =========================================================================
    result_arr, result_path = run_longitudinal_analysis(
        log_path=LOG_PATH,
        chunk_size_frames=CHUNK_SIZE_FRAMES,
        chunk_duration_s=CHUNK_DURATION_S,
        start_utc=START_UTC,
        density_kg_m3=DENSITY_KG_M3,
    )

    # Quick summary
    print("\nColumn names in output array:")
    print(result_arr.dtype.names)
    print("\nTo plot chunk mid-time vs density:")
    print("  import numpy as np")
    print("  import matplotlib.pyplot as plt")
    print(f"  data = np.load(r'{result_path}', allow_pickle=False)")
    print("  plt.plot(data['chunk_mid_time_s'], data['mean_density_mm3'])")
    print("  plt.xlabel('Time (s)'); plt.ylabel('Density (droplets/mm³)')")
    print("  plt.show()")
