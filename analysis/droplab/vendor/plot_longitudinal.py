"""
plot_longitudinal.py — Interactive overview plot for a longitudinal analysis file.

Loads a longitudinal_analysis_*.npy file produced by longitudinal_analysis.py
and creates a comprehensive multi-panel figure showing all extracted parameters
vs chunk mid-time, with error bars (± 1 std dev).

Usage: edit the USER SETTINGS block and run:
    python plot_longitudinal.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# Helpers
# =============================================================================

def _get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def load(npy_path: str) -> np.ndarray:
    data = np.load(npy_path, allow_pickle=False)
    print(f"Loaded: {npy_path}")
    print(f"  {len(data)} chunks, fields: {data.dtype.names}")
    return data


def _col(data, key):
    return data[key].astype(float)


def _errorbar(ax, t, mean, std, color, label, unit_scale=1.0, fmt='o-', ms=4, lw=1.5):
    """Plot mean ± std errorbar with connecting line."""
    m = mean * unit_scale
    s = std  * unit_scale
    ax.plot(t, m, fmt, color=color, markersize=ms, linewidth=lw, label=label)
    ax.fill_between(t, m - s, m + s, color=color, alpha=0.18)


# =============================================================================
# Main plot function
# =============================================================================

def plot_longitudinal(data: np.ndarray, title_prefix: str = "", save_path: str = None):
    """
    Create a multi-panel longitudinal overview figure.

    Parameters
    ----------
    data       : structured numpy array from longitudinal_analysis.npy
    title_prefix : string prepended to the figure suptitle (e.g. video basename)
    save_path  : if given, save the figure here instead of showing interactively
    """
    t = _col(data, 'chunk_mid_time_s')

    fig = plt.figure(figsize=(14, 13))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.38)

    # ------------------------------------------------------------------
    # 1   Count per frame (left)  +  twin axis for density
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _errorbar(ax1, t, _col(data,'mean_count'), _col(data,'std_count'),
              color='steelblue', label='Count / frame')
    ax1.set_ylabel('Droplets per frame', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_title('Droplet count & density')
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    _errorbar(ax1b, t, _col(data,'mean_density_mm3'), _col(data,'std_density_mm3'),
              color='navy', label='Density', fmt='s--', ms=3, lw=1.2)
    ax1b.set_ylabel('Density (droplets/mm³)', color='navy')
    ax1b.tick_params(axis='y', labelcolor='navy')

    # ------------------------------------------------------------------
    # 2   Streak width (pixels + µm twin)
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _errorbar(ax2, t, _col(data,'mean_width_px'), _col(data,'std_width_px'),
              color='darkorange', label='Width (px)')
    ax2.set_ylabel('Width (pixels)', color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.set_title('Streak width')
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    _errorbar(ax2b, t, _col(data,'mean_width_um'), _col(data,'std_width_um'),
              color='saddlebrown', label='Width (µm)', fmt='s--', ms=3, lw=1.2)
    ax2b.set_ylabel('Width (µm)', color='saddlebrown')
    ax2b.tick_params(axis='y', labelcolor='saddlebrown')

    # ------------------------------------------------------------------
    # 3   Wind direction  (points colored by |v| total)
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])

    wind_deg  = _col(data, 'wind_direction_deg')
    speed_mms = _col(data, 'mean_speed_mps') * 1e3   # mm/s for colormap

    valid_w = np.isfinite(wind_deg) & np.isfinite(speed_mms)

    # scatter colored by speed (no connecting line)
    sc3 = ax3.scatter(t[valid_w], wind_deg[valid_w],
                      c=speed_mms[valid_w], cmap='plasma',
                      s=35, zorder=3, edgecolors='none',
                      vmin=np.nanmin(speed_mms), vmax=np.nanmax(speed_mms))
    cb3 = fig.colorbar(sc3, ax=ax3, pad=0.02, fraction=0.046)
    cb3.set_label('|v| total (mm/s)', fontsize=8)
    cb3.ax.tick_params(labelsize=7)

    ax3.set_ylabel('Circular mean angle (°)')
    ax3.set_ylim(-5, 185)
    ax3.set_yticks([0, 45, 90, 135, 180])
    ax3.set_title('Wind direction\n'
                  r'0° $\rightarrow$ R (horiz)  ·  90° $\uparrow$ Z (vert)',
                  fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 4   Anisotropy ratio
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    aniso = _col(data, 'anisotropy_ratio')
    finite = np.isfinite(aniso)
    ax4.plot(t[finite], aniso[finite], 'o-', color='darkorchid', markersize=4, linewidth=1.5)
    ax4.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='isotropic')
    ax4.set_ylabel('Anisotropy ratio (max/min)')
    ax4.set_title('Orientation anisotropy')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # 5   Speed component summary (full width)
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, :])
    _errorbar(ax5, t, _col(data,'mean_speed_mps'), _col(data,'std_speed_mps'),
              color='seagreen', label='|v| total', unit_scale=1e3)
    _errorbar(ax5, t, _col(data,'mean_vy_mps'), _col(data,'std_vy_mps'),
              color='royalblue', label='|v_r|', unit_scale=1e3, fmt='s--', ms=3, lw=1.2)
    _errorbar(ax5, t, _col(data,'mean_vz_mps'), _col(data,'std_vz_mps'),
              color='crimson', label='|v_z|', unit_scale=1e3, fmt='^--', ms=3, lw=1.2)
    ax5.set_ylabel('Speed (mm/s)')
    ax5.set_xlabel('Chunk mid-time (s)')
    ax5.set_title('Speed component summary')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Shared x-axis label and suptitle
    # ------------------------------------------------------------------
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Chunk mid-time (s)', fontsize=9)

    n_chunks = len(data)
    t_span   = f"{t.min():.1f} – {t.max():.1f} s" if n_chunks > 0 else ""
    fig.suptitle(
        f"{title_prefix}  |  {n_chunks} chunks  {t_span}",
        fontsize=13, fontweight='bold', y=0.995
    )

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"Figure saved: {save_path}")
    else:
        plt.tight_layout()
        plt.show()

    return fig


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # =========================================================================
    # USER SETTINGS — edit this path
    # =========================================================================

    LONGITUDINAL_FILE = os.path.join(
        _get_project_root(), "data", "xenon_fill_10_3-7-2026",
        "longitudinal_analysis_xenon_fill_10_3-7-2026_2026-03-12_10-10-09.npy"
    )

    # Set to a path string to save instead of displaying (e.g. "plots/overview.png")
    SAVE_PATH = None

    # =========================================================================

    if not os.path.exists(LONGITUDINAL_FILE):
        # Try to find the most recent longitudinal file for the given video
        parts = os.path.basename(LONGITUDINAL_FILE).split("_")
        data_dir = os.path.dirname(LONGITUDINAL_FILE)
        if os.path.isdir(data_dir):
            candidates = [f for f in os.listdir(data_dir)
                          if f.startswith("longitudinal_analysis_") and f.endswith(".npy")]
            if candidates:
                candidates.sort(reverse=True)  # most recent first
                LONGITUDINAL_FILE = os.path.join(data_dir, candidates[0])
                print(f"Auto-selected most recent: {LONGITUDINAL_FILE}")
            else:
                print(f"No longitudinal_analysis_*.npy files found in {data_dir}")
                sys.exit(1)
        else:
            print(f"File not found: {LONGITUDINAL_FILE}")
            sys.exit(1)

    data = load(LONGITUDINAL_FILE)

    # Derive a title from the filename
    basename = os.path.basename(LONGITUDINAL_FILE).replace("longitudinal_analysis_", "").replace(".npy", "")
    # basename is now  <video>_<date>_<time>  — split off the date/time
    parts = basename.rsplit("_", 3)
    title = parts[0] if len(parts) >= 4 else basename

    plot_longitudinal(data, title_prefix=title, save_path=SAVE_PATH)
