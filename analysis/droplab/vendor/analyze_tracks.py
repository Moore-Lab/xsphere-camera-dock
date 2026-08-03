#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Track-Based Anisotropy and Flow Analysis

Analyzes tracked droplet trajectories to characterize flow anisotropy,
bulk flow velocity, and directional speed distributions.  Reads
track_data.npz produced by track_droplets.save_track_data().
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Any, Dict, Optional


# =============================================================================
# Loading
# =============================================================================

def load_track_data(filepath: str = "track_data.npz") -> Dict[str, Any]:
    """
    Load track data saved by track_droplets.save_track_data().

    Returns
    -------
    dict with keys
        observations : structured ndarray  (all rows)
        metadata     : dict
        velocities   : structured ndarray  (rows where speed is not NaN)
    """
    npz = np.load(filepath, allow_pickle=False)
    obs = npz["track_observations"]
    metadata = json.loads(str(npz["metadata"]))

    valid = np.isfinite(obs["speed"])
    velocities = obs[valid]

    print(f"[load] {filepath}: {metadata['n_tracks']} tracks, "
          f"{len(obs)} observations, {len(velocities)} velocity measurements")

    return {
        "observations": obs,
        "metadata": metadata,
        "velocities": velocities,
    }


# =============================================================================
# Ellipse fitting helper
# =============================================================================

def _fit_anisotropy_ellipse(vx, vy):
    """
    Fit a 2D Gaussian to (vx, vy) velocity components.

    Returns
    -------
    dict with centroid, covariance eigenvalues/vectors, eccentricity, tilt.
    """
    cx, cy = np.mean(vx), np.mean(vy)
    cov = np.cov(vx, vy)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh returns ascending order; we want major axis first
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    sigma_major = np.sqrt(max(eigvals[0], 0))
    sigma_minor = np.sqrt(max(eigvals[1], 0))

    if sigma_major > 0:
        eccentricity = np.sqrt(1.0 - (sigma_minor / sigma_major) ** 2)
    else:
        eccentricity = 0.0

    tilt_rad = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
    tilt_deg = np.degrees(tilt_rad)

    bulk_speed = np.hypot(cx, cy)
    bulk_dir = np.degrees(np.arctan2(cy, cx)) % 360.0

    return {
        "centroid_vx": cx,
        "centroid_vy": cy,
        "bulk_flow_speed": bulk_speed,
        "bulk_flow_direction_deg": bulk_dir,
        "sigma_major": sigma_major,
        "sigma_minor": sigma_minor,
        "eccentricity": eccentricity,
        "tilt_angle_deg": tilt_deg,
        "sigma_ratio": sigma_minor / sigma_major if sigma_major > 0 else np.nan,
        "cov_matrix": cov,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
    }


# =============================================================================
# Main analysis
# =============================================================================

def analyze_tracks(
    track_data: Dict[str, Any],
    *,
    save_path: Optional[str] = None,
    n_angle_bins: int = 24,
    n_time_windows: int = 5,
    spatial_bins: int = 8,
    min_obs_per_bin: int = 3,
) -> Dict[str, Any]:
    """
    Produce a 3x2 analysis figure and return a structured results dict.

    Parameters
    ----------
    track_data : dict from load_track_data()
    save_path  : path to save figure (None = display only)
    n_angle_bins    : angular bins for heatmap and sigma profile
    n_time_windows  : temporal windows for evolution panel
    spatial_bins    : grid divisions for quiver plot
    min_obs_per_bin : minimum observations to draw a quiver arrow
    """
    obs = track_data["observations"]
    vel = track_data["velocities"]
    meta = track_data["metadata"]

    fps = meta.get("fps", 30.0)
    pixels_per_um = meta.get("pixels_per_um", 0.1878)
    pixels_per_m = pixels_per_um * 1e6

    if len(vel) == 0:
        print("[analyze_tracks] No velocity measurements — all tracks are single-frame. Skipping.")
        return {"error": "no velocity data"}

    # Extract velocity arrays
    vx = vel["vx"].astype(float)
    vy = vel["vy"].astype(float)
    speed = vel["speed"].astype(float)
    direction = vel["direction"].astype(float)
    frames = vel["frame"].astype(float)

    # Convert to physical units (m/s)
    vx_mps = vx / pixels_per_m
    vy_mps = vy / pixels_per_m
    speed_mps = speed / pixels_per_m

    # -------------------------------------------------------------------------
    # Figure setup: 3x2 grid
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Track-Based Flow Analysis", fontsize=16, fontweight="bold", y=0.98)

    # =====================================================================
    # Panel 1: Speed–angle heatmap
    # =====================================================================
    ax = axes[0, 0]
    speed_mmps = speed_mps * 1e3
    h = ax.hist2d(direction, speed_mmps, bins=[n_angle_bins, 30],
                  cmap="hot", cmin=1)
    fig.colorbar(h[3], ax=ax, label="Count")
    ax.set_xlabel("Direction (deg)")
    ax.set_ylabel("Speed (mm/s)")
    ax.set_title("Speed–Direction Heatmap")
    ax.set_xlim(0, 360)

    # =====================================================================
    # Panel 2: Velocity scatter + anisotropy ellipse
    # =====================================================================
    ax = axes[0, 1]
    vx_mmps = vx_mps * 1e3
    vy_mmps = vy_mps * 1e3

    ax.scatter(vx_mmps, vy_mmps, s=4, alpha=0.3, c="steelblue", edgecolors="none")

    ellipse_info = _fit_anisotropy_ellipse(vx_mmps, vy_mmps)
    cx = ellipse_info["centroid_vx"]
    cy = ellipse_info["centroid_vy"]

    # Draw 1-sigma and 2-sigma ellipses
    for n_sigma, ls in [(1, "-"), (2, "--")]:
        w = 2 * n_sigma * ellipse_info["sigma_major"]
        h_el = 2 * n_sigma * ellipse_info["sigma_minor"]
        ell = Ellipse((cx, cy), w, h_el,
                      angle=ellipse_info["tilt_angle_deg"],
                      fill=False, edgecolor="red", linewidth=1.5, linestyle=ls)
        ax.add_patch(ell)

    ax.plot(cx, cy, "r+", markersize=12, markeredgewidth=2)
    ax.axhline(0, color="gray", linewidth=0.5, zorder=0)
    ax.axvline(0, color="gray", linewidth=0.5, zorder=0)
    ax.set_xlabel("$v_x$ (mm/s)")
    ax.set_ylabel("$v_y$ (mm/s)")
    ax.set_title("Velocity Scatter + Anisotropy Ellipse")
    ax.set_aspect("equal", adjustable="datalim")

    ecc = ellipse_info["eccentricity"]
    bulk = ellipse_info["bulk_flow_speed"]
    bulk_dir = ellipse_info["bulk_flow_direction_deg"]
    ax.annotate(
        f"ecc = {ecc:.2f}\n"
        f"bulk = {bulk:.2f} mm/s @ {bulk_dir:.0f}°\n"
        f"$\\sigma_{{maj}}$ = {ellipse_info['sigma_major']:.2f}\n"
        f"$\\sigma_{{min}}$ = {ellipse_info['sigma_minor']:.2f}",
        xy=(0.02, 0.98), xycoords="axes fraction",
        va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # =====================================================================
    # Panel 3: Angular sigma profile (polar)
    # =====================================================================
    # Replace cartesian axes[1,0] with polar
    pos = axes[1, 0].get_position()
    axes[1, 0].remove()
    ax_polar = fig.add_axes(pos, polar=True)

    bin_edges = np.linspace(0, 360, n_angle_bins + 1)
    bin_centers_deg = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_centers_rad = np.deg2rad(bin_centers_deg)
    bin_idx = np.digitize(direction, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, n_angle_bins - 1)

    mean_speed_bin = np.full(n_angle_bins, np.nan)
    std_speed_bin = np.full(n_angle_bins, np.nan)
    for b in range(n_angle_bins):
        mask = bin_idx == b
        if np.sum(mask) >= min_obs_per_bin:
            mean_speed_bin[b] = np.mean(speed_mmps[mask])
            std_speed_bin[b] = np.std(speed_mmps[mask])

    valid_bins = np.isfinite(mean_speed_bin)
    if np.any(valid_bins):
        # Close the polar curve
        theta_plot = np.append(bin_centers_rad[valid_bins], bin_centers_rad[valid_bins][0])
        r_plot = np.append(mean_speed_bin[valid_bins], mean_speed_bin[valid_bins][0])

        ax_polar.plot(theta_plot, r_plot, "o-", color="darkorange", linewidth=2, markersize=4)
        # Uniform reference
        mean_all = np.nanmean(speed_mmps)
        ax_polar.plot(np.linspace(0, 2 * np.pi, 100),
                      np.full(100, mean_all),
                      "--", color="gray", linewidth=1, alpha=0.7, label=f"mean = {mean_all:.1f}")
        ax_polar.legend(loc="upper right", fontsize=8)

    ax_polar.set_title("Angular Speed Profile", pad=15)
    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(-1)

    # =====================================================================
    # Panel 4: Spatial flow field (quiver)
    # =====================================================================
    ax = axes[1, 1]

    # Use midpoint of streak as position
    mid_x = vel["start_x"].astype(float)
    mid_y = vel["start_y"].astype(float)

    x_edges = np.linspace(np.min(mid_x), np.max(mid_x), spatial_bins + 1)
    y_edges = np.linspace(np.min(mid_y), np.max(mid_y), spatial_bins + 1)

    mean_vx_grid = np.full((spatial_bins, spatial_bins), np.nan)
    mean_vy_grid = np.full((spatial_bins, spatial_bins), np.nan)
    count_grid = np.zeros((spatial_bins, spatial_bins), dtype=int)

    ix = np.clip(np.digitize(mid_x, x_edges) - 1, 0, spatial_bins - 1)
    iy = np.clip(np.digitize(mid_y, y_edges) - 1, 0, spatial_bins - 1)

    for bx in range(spatial_bins):
        for by in range(spatial_bins):
            mask = (ix == bx) & (iy == by)
            n = np.sum(mask)
            count_grid[by, bx] = n
            if n >= min_obs_per_bin:
                mean_vx_grid[by, bx] = np.mean(vx_mmps[mask])
                mean_vy_grid[by, bx] = np.mean(vy_mmps[mask])

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers)

    valid_q = np.isfinite(mean_vx_grid)
    if np.any(valid_q):
        speed_grid = np.hypot(mean_vx_grid, mean_vy_grid)
        ax.imshow(count_grid, extent=[x_edges[0], x_edges[-1], y_edges[-1], y_edges[0]],
                  cmap="Greys", alpha=0.3, aspect="auto")
        q = ax.quiver(X[valid_q], Y[valid_q],
                      mean_vx_grid[valid_q], mean_vy_grid[valid_q],
                      speed_grid[valid_q], cmap="coolwarm", scale_units="xy",
                      angles="xy")
        fig.colorbar(q, ax=ax, label="Mean speed (mm/s)")

    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    ax.set_title("Spatial Flow Field")
    ax.invert_yaxis()

    # =====================================================================
    # Panel 5: Temporal evolution of anisotropy
    # =====================================================================
    ax = axes[2, 0]

    frame_min, frame_max = frames.min(), frames.max()
    if n_time_windows > 0 and frame_max > frame_min:
        window_edges = np.linspace(frame_min, frame_max + 1, n_time_windows + 1)
        window_centers = 0.5 * (window_edges[:-1] + window_edges[1:])

        ecc_t = np.full(n_time_windows, np.nan)
        bulk_t = np.full(n_time_windows, np.nan)
        sigma_ratio_t = np.full(n_time_windows, np.nan)

        for w in range(n_time_windows):
            mask = (frames >= window_edges[w]) & (frames < window_edges[w + 1])
            if np.sum(mask) < 6:
                continue
            info = _fit_anisotropy_ellipse(vx_mmps[mask], vy_mmps[mask])
            ecc_t[w] = info["eccentricity"]
            bulk_t[w] = info["bulk_flow_speed"]
            sigma_ratio_t[w] = info["sigma_ratio"]

        ax.plot(window_centers, ecc_t, "s-", color="crimson", label="Eccentricity")
        ax2 = ax.twinx()
        ax2.plot(window_centers, bulk_t, "o-", color="steelblue", label="Bulk flow (mm/s)")
        ax2.set_ylabel("Bulk flow speed (mm/s)", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")

        ax.set_xlabel("Frame")
        ax.set_ylabel("Eccentricity", color="crimson")
        ax.tick_params(axis="y", labelcolor="crimson")
        ax.set_title("Temporal Evolution of Anisotropy")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Insufficient temporal range",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Temporal Evolution of Anisotropy")

    # =====================================================================
    # Panel 6: Summary statistics text
    # =====================================================================
    ax = axes[2, 1]
    ax.axis("off")

    n_tracks = meta.get("n_tracks", 0)
    n_vel = len(vel)
    mean_spd = np.mean(speed_mmps)
    rms_spd = np.sqrt(np.mean(speed_mmps ** 2))
    std_spd = np.std(speed_mmps)

    lines = [
        f"Tracks:  {n_tracks}",
        f"Velocity measurements:  {n_vel}",
        "",
        f"Mean speed:  {mean_spd:.2f} mm/s",
        f"RMS speed:   {rms_spd:.2f} mm/s",
        f"Std speed:   {std_spd:.2f} mm/s",
        "",
        f"Bulk flow:   {ellipse_info['bulk_flow_speed']:.2f} mm/s "
        f"@ {ellipse_info['bulk_flow_direction_deg']:.0f}°",
        f"Eccentricity:  {ellipse_info['eccentricity']:.3f}",
        f"Tilt angle:    {ellipse_info['tilt_angle_deg']:.1f}°",
        f"$\\sigma_{{major}}$:  {ellipse_info['sigma_major']:.2f} mm/s",
        f"$\\sigma_{{minor}}$:  {ellipse_info['sigma_minor']:.2f} mm/s",
        f"$\\sigma$ ratio (min/maj):  {ellipse_info['sigma_ratio']:.3f}",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", fontsize=11, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="#f9f9f9", ec="gray"))
    ax.set_title("Summary")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[save] Figure saved to: {save_path}")

    plt.show()

    # -------------------------------------------------------------------------
    # Build results dict
    # -------------------------------------------------------------------------
    results = {
        "metadata_used": meta,
        "velocity_stats": {
            "n_tracks": n_tracks,
            "n_velocity_measurements": n_vel,
            "mean_speed_mps": float(np.mean(speed_mps)),
            "rms_speed_mps": float(np.sqrt(np.mean(speed_mps ** 2))),
            "std_speed_mps": float(np.std(speed_mps)),
            "mean_speed_mmps": float(mean_spd),
        },
        "anisotropy_ellipse": {
            k: (float(v) if isinstance(v, (float, np.floating)) else v)
            for k, v in ellipse_info.items()
            if k not in ("cov_matrix", "eigvals", "eigvecs")
        },
        "angular_profile": {
            "bin_centers_deg": bin_centers_deg.tolist(),
            "mean_speed_mmps_per_bin": [
                float(v) if np.isfinite(v) else None for v in mean_speed_bin
            ],
        },
    }

    # Add temporal if computed
    if n_time_windows > 0 and frame_max > frame_min:
        results["temporal_evolution"] = {
            "window_centers_frame": window_centers.tolist(),
            "eccentricity_vs_time": [
                float(v) if np.isfinite(v) else None for v in ecc_t
            ],
            "bulk_flow_vs_time": [
                float(v) if np.isfinite(v) else None for v in bulk_t
            ],
            "sigma_ratio_vs_time": [
                float(v) if np.isfinite(v) else None for v in sigma_ratio_t
            ],
        }

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("TRACK ANALYSIS SUMMARY")
    print("=" * 62)
    print(f"Tracks analyzed:         {n_tracks}")
    print(f"Velocity measurements:   {n_vel}")
    print(f"fps = {fps}, pixels_per_um = {pixels_per_um}")

    print(f"\n--- Speed ---")
    print(f"Mean speed:              {mean_spd:.2f} mm/s")
    print(f"RMS speed:               {rms_spd:.2f} mm/s")
    print(f"Std speed:               {std_spd:.2f} mm/s")

    print(f"\n--- Anisotropy Ellipse ---")
    print(f"Bulk flow velocity:      {ellipse_info['bulk_flow_speed']:.2f} mm/s "
          f"@ {ellipse_info['bulk_flow_direction_deg']:.0f} deg")
    print(f"Eccentricity:            {ellipse_info['eccentricity']:.3f}")
    print(f"Tilt angle:              {ellipse_info['tilt_angle_deg']:.1f} deg")
    print(f"sigma_major:             {ellipse_info['sigma_major']:.2f} mm/s")
    print(f"sigma_minor:             {ellipse_info['sigma_minor']:.2f} mm/s")
    print(f"sigma ratio (min/maj):   {ellipse_info['sigma_ratio']:.3f}")

    if "temporal_evolution" in results:
        valid_ecc = [v for v in ecc_t if np.isfinite(v)]
        if valid_ecc:
            print(f"\n--- Temporal Stability ---")
            print(f"Eccentricity range:      [{min(valid_ecc):.3f}, {max(valid_ecc):.3f}] "
                  f"over {n_time_windows} windows")
        valid_bulk = [v for v in bulk_t if np.isfinite(v)]
        if valid_bulk:
            print(f"Bulk flow range:         [{min(valid_bulk):.2f}, {max(valid_bulk):.2f}] mm/s")

    print("=" * 62 + "\n")

    return results


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # === USER SETTINGS ===
    # Update VIDEO_BASENAME to match your video filename (without extension)
    VIDEO_BASENAME = "water_constexp2"
    import os as _os
    _root = _os.path.dirname(_os.path.abspath(__file__))
    track_file = _os.path.join(_root, "data", VIDEO_BASENAME, f"{VIDEO_BASENAME}_track_data.npz")

    track_data = load_track_data(track_file)
    results = analyze_tracks(track_data)
