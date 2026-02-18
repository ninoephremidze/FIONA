"""
Utility functions for comparing FIONA results with other packages.

Author: Nino Ephremidze
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_overlays_ws(
    w_grid,
    F_fiona,
    F_glow,
    n,
    Umax,
    title="Comparison",
    align_phase=False,
):
    """
    Plot overlays comparing FIONA and GLoW results in frequency domain.

    Parameters
    ----------
    w_grid : array_like
        Frequency grid points
    F_fiona : array_like
        FIONA results (complex amplitudes)
    F_glow : array_like
        GLoW results (complex amplitudes)
    n : int
        Number of grid points used in computation
    Umax : float
        Maximum U value used in computation
    title : str, optional
        Plot title
    align_phase : bool, optional
        If True, align phases between FIONA and GLoW for better comparison
    """
    # Convert to numpy arrays
    w_grid = np.asarray(w_grid)
    F_fiona = np.asarray(F_fiona)
    F_glow = np.asarray(F_glow)

    # Optionally align phases
    if align_phase:
        # Find a common phase offset at the first point
        phase_offset = np.angle(F_glow[0]) - np.angle(F_fiona[0])
        F_fiona = F_fiona * np.exp(1j * phase_offset)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"{title} (n={n}, Umax={Umax})", fontsize=14, fontweight='bold')

    # Amplitude comparison
    ax = axes[0, 0]
    ax.loglog(w_grid, np.abs(F_fiona), 'b-', label='FIONA', linewidth=2)
    ax.loglog(w_grid, np.abs(F_glow), 'r--', label='GLoW', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Frequency w')
    ax.set_ylabel('|F(w)|')
    ax.set_title('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Phase comparison
    ax = axes[0, 1]
    ax.semilogx(w_grid, np.angle(F_fiona), 'b-', label='FIONA', linewidth=2)
    ax.semilogx(w_grid, np.angle(F_glow), 'r--', label='GLoW', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Frequency w')
    ax.set_ylabel('Phase (radians)')
    ax.set_title('Phase')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Relative amplitude error
    ax = axes[1, 0]
    rel_amp_error = np.abs((np.abs(F_fiona) - np.abs(F_glow)) / np.abs(F_glow))
    ax.loglog(w_grid, rel_amp_error, 'g-', linewidth=2)
    ax.set_xlabel('Frequency w')
    ax.set_ylabel('Relative Amplitude Error')
    ax.set_title('|F_FIONA| - |F_GLoW| / |F_GLoW|')
    ax.grid(True, alpha=0.3)

    # Phase difference
    ax = axes[1, 1]
    phase_diff = np.angle(F_fiona) - np.angle(F_glow)
    # Wrap to [-pi, pi]
    phase_diff = np.angle(np.exp(1j * phase_diff))
    ax.semilogx(w_grid, phase_diff, 'g-', linewidth=2)
    ax.set_xlabel('Frequency w')
    ax.set_ylabel('Phase Difference (radians)')
    ax.set_title('Phase(F_FIONA) - Phase(F_GLoW)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print("\n" + "="*60)
    print("Comparison Summary")
    print("="*60)
    print(f"Max relative amplitude error: {np.max(rel_amp_error):.6e}")
    print(f"Mean relative amplitude error: {np.mean(rel_amp_error):.6e}")
    print(f"Max phase difference: {np.max(np.abs(phase_diff)):.6e} rad")
    print(f"Mean phase difference: {np.mean(np.abs(phase_diff)):.6e} rad")
    print("="*60 + "\n")
