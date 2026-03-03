####################################################################
# fiona/axisym.py
####################################################################

import numpy as np
from pathlib import Path
import time

from .lenses import AxisymmetricLens
from .utils import CPUTracker, gauss_legendre_1d
import os

_HAS_SCIPY_FHT = False
_SCIPY_FHT_ERR = None

try:
    from scipy.fft import fht as _scipy_fht, fhtoffset as _scipy_fhtoffset
    _HAS_SCIPY_FHT = True
except Exception as e:
    _HAS_SCIPY_FHT = False
    _SCIPY_FHT_ERR = e

_HAS_FHT = False
_FHT_ERR = None

try:
    from _pynufht import nufht as _c_nufht
    _HAS_FHT = True

except Exception as e:
    _HAS_FHT = False
    _FHT_ERR = e

_HAS_NUFHT_BATCH = False
_NUFHT_BATCH_ERR = None

try:
    import _pynufht as _pn_module
    _pn_module.nufht_batch  # verify the symbol exists
    _HAS_NUFHT_BATCH = True
except Exception as e:
    _HAS_NUFHT_BATCH = False
    _NUFHT_BATCH_ERR = e

import concurrent.futures


def _window_taper_1d(r, R, frac):
    """Smooth taper for the lens potential near the boundary (1D radial)."""
    return 0.5 * (1.0 - np.tanh(r - frac * R))


def _window_u_taper_1d(r, R, width_frac):
    """Smooth radial weight taper near r = R (1D radial)."""
    if width_frac <= 0.0:
        return 1.0
    return 0.5 * (1.0 - np.tanh((r - R) / (width_frac * R)))


def _adaptive_n_gl(w_abs):
    """
    Choose the number of 1-D Gauss–Legendre nodes for a given |w|.

    Frequencies are binned in steps of 10 (clamped to [1, 10]); each bin
    uses bin_index * 1000 nodes so that higher-frequency integrands, which
    oscillate more rapidly, receive more quadrature points.

    This mirrors the adaptive schedule used in general.py, where n_gl^2
    total nodes are used for the 2-D case.  Here n_gl is the 1-D node count.
    """
    if w_abs <= 0.0:
        raise ValueError("w must be nonzero for adaptive quadrature.")
    bin_idx = int(np.floor(w_abs / 10.0)) + 1
    bin_idx = max(1, min(10, bin_idx))
    return bin_idx * 1000


def _nufht_batch_worker(args):
    """
    Top-level worker function (must be module-level for picklability).

    Receives a chunk of frequencies and unit GL nodes, then for each frequency
    computes the per-frequency x-domain integration radius
    Xmax(w) = max(min_physical_radius, sqrt(n_gl / (2*|w|))), rescales the
    unit nodes to x-space, and calls ``pn.nufht`` individually (since output
    points ``|w|*y_vec`` differ per frequency).

    When ``n_gl == 0`` (fixed mode), uses ``Xmax = min_physical_radius`` for
    all frequencies without per-frequency adaptation; in that case the x-nodes
    are the same for every w and ``integrand_phase = x²/2 − ψ(x)`` is
    precomputed once.

    Parameters
    ----------
    args : tuple
        (lens, rs_unit, w_unit, y_vec, w_chunk, nu, tol, min_physical_radius, n_gl,
        window_potential, window_radius_fraction, window_u, window_u_width,
        use_tail_correction)
        where ``rs_unit`` and ``w_unit`` are GL nodes/weights on the unit interval.

    Returns
    -------
    G_re, G_im : ndarray of shape (n_y, batch)
    """
    import _pynufht as pn
    import numpy as np

    (lens, rs_unit, w_unit, y_vec, w_chunk, nu, tol, min_physical_radius, n_gl,
     window_potential, window_radius_fraction, window_u, window_u_width,
     use_tail_correction) = args

    n_y = y_vec.size
    batch = len(w_chunk)

    G_re = np.empty((n_y, batch), dtype=float)
    G_im = np.empty((n_y, batch), dtype=float)

    # Fixed mode: precompute x-nodes and integrand_phase once for all frequencies.
    if n_gl == 0:
        Xmax = min_physical_radius
        xs_fixed = rs_unit * Xmax
        x_weights_fixed = xs_fixed * (w_unit * Xmax)
        if window_u:
            x_weights_fixed = x_weights_fixed * _window_u_taper_1d(
                xs_fixed, Xmax, window_u_width)
        psi_fixed = lens.psi_r(xs_fixed)
        if window_potential:
            psi_fixed = psi_fixed * _window_taper_1d(
                xs_fixed, Xmax, window_radius_fraction)
        integrand_phase_fixed = 0.5 * xs_fixed**2 - psi_fixed

    for j, w in enumerate(w_chunk):
        # Per-frequency x-domain integration radius
        if n_gl > 0:
            Xmax_w = max(min_physical_radius, np.sqrt(n_gl / (2.0 * abs(w))))
            xs = rs_unit * Xmax_w
            x_weights = xs * (w_unit * Xmax_w)
            if window_u:
                x_weights = x_weights * _window_u_taper_1d(xs, Xmax_w, window_u_width)
            psi_vals = lens.psi_r(xs)
            if window_potential:
                psi_vals = psi_vals * _window_taper_1d(xs, Xmax_w, window_radius_fraction)
            integrand_phase = 0.5 * xs**2 - psi_vals
        else:
            xs = xs_fixed
            x_weights = x_weights_fixed
            integrand_phase = integrand_phase_fixed

        # fw = exp(iw * (x²/2 − ψ(x))); subtract free-space term for tail correction
        if use_tail_correction:
            fw = (np.exp(1j * w * integrand_phase)
                  - np.exp(1j * w * 0.5 * xs**2))
        else:
            fw = np.exp(1j * w * integrand_phase)
        ck = x_weights * fw

        # NUFHT output points are |w|·y (x-domain: J₀(w·x·y) = J₀(|w|·x·y))
        out_pts = abs(w) * y_vec
        G_re[:, j] = pn.nufht(nu, xs, ck.real, out_pts, tol=tol)
        G_im[:, j] = pn.nufht(nu, xs, ck.imag, out_pts, tol=tol)

    return G_re, G_im


def _pct(part, total):
    """Return percentage (part/total * 100), safe for total=0."""
    if total <= 0.0:
        return 0.0
    return 100.0 * part / total
    
# ----------------------------------------------------------------------
# Fresnel integral with *precomputed* 1-D Gauss–Legendre nodes
# ----------------------------------------------------------------------
class FresnelNUFHT:
    """
    Fresnel integral for axisymmetric lenses using Gauss-Legendre nodes
    and **FastHankelTransform NUFHT** in the **x-domain** formulation.

    The axisymmetric Fresnel integral in x-domain is::

        F(w, y) = (w/i) · e^{iwy²/2} ∫₀^∞ x dx exp{iw[x²/2 − ψ(x)]} J₀(wxy)

    The integrand phase ``x²/2 − ψ(x)`` is independent of ``w``, so in
    fixed-grid mode (``auto_R_from_gl_nodes=False``) the lens potential
    ``ψ(x)`` is evaluated only once for all frequencies.

    GL nodes are loaded from precomputed files if available, or computed
    on-the-fly and saved to disk (via gauss_legendre_1d from utils.py).

    We discard x<0 (symmetry) and use:
        xs = x[x>0]
        x_weights = xs * w[x>0]

    Parameters
    ----------
    lens : AxisymmetricLens
        Axisymmetric lens object with psi_r method.
    gl_nodes_per_dim : int
        Number of Gauss-Legendre nodes per dimension.
    min_physical_radius : float
        Minimum physical radius (Xmax) to use.
    auto_R_from_gl_nodes : bool
        If True, adapt Xmax based on frequency range.
        If False, use fixed min_physical_radius.
    gl_dir : str or None
        Directory for GL node files. If None, uses FIONA_GL2D_DIR env var.
    tol : float
        Tolerance for NUFHT.
    """

    def __init__(self,
                 lens: AxisymmetricLens,
                 gl_nodes_per_dim: int = None,
                 min_physical_radius: float = None,
                 auto_R_from_gl_nodes: bool = True,
                 gl_dir: str = None,
                 tol: float = 1e-12,
                 window_potential: bool = True,
                 window_radius_fraction: float = 0.75,
                 window_u: bool = True,
                 window_u_width: float = 0.02,
                 use_tail_correction: bool = True,
                 # Deprecated parameters for backward compatibility
                 n_gl: int = None,
                 Umax: float = None):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError("FresnelHankelAxisymmetric requires AxisymmetricLens")

        if not _HAS_FHT:
            raise ImportError("NUFHT C extension (_pynufht) cannot be loaded: "
                              f"{_FHT_ERR!r}")

        # Handle backward compatibility for old parameter names
        if n_gl is not None:
            if gl_nodes_per_dim is not None:
                raise ValueError("Cannot specify both n_gl and gl_nodes_per_dim")
            gl_nodes_per_dim = n_gl
        if Umax is not None:
            if min_physical_radius is not None:
                raise ValueError("Cannot specify both Umax and min_physical_radius")
            min_physical_radius = Umax
            # If old Umax is specified, default to no auto-adaptation for backward compat
            if n_gl is not None:  # Both old params specified
                auto_R_from_gl_nodes = False

        # Set defaults if still None
        if gl_nodes_per_dim is None:
            gl_nodes_per_dim = 128
        if min_physical_radius is None:
            min_physical_radius = 1.0

        self.lens = lens
        self.gl_nodes_per_dim = int(gl_nodes_per_dim)
        self.min_physical_radius = float(min_physical_radius)
        self.auto_R_from_gl_nodes = bool(auto_R_from_gl_nodes)
        self.tol = float(tol)
        self.window_potential = bool(window_potential)
        self.window_radius_fraction = float(window_radius_fraction)
        self.window_u = bool(window_u)
        self.window_u_width = float(window_u_width)
        self.use_tail_correction = bool(use_tail_correction)

        # Store gl_dir for later use
        if gl_dir is None:
            gl_dir = os.environ.get("FIONA_GL2D_DIR", "")
            if not gl_dir:
                raise RuntimeError(
                    "gl_dir not provided and FIONA_GL2D_DIR environment variable not set."
                )
        self._gl_dir = gl_dir

        # Don't load nodes yet - will be loaded in __call__ based on frequency range

    def _load_gl_nodes(self, Xmax):
        """
        Load or compute GL nodes for given Xmax (x-domain integration radius).
        Uses gauss_legendre_1d from utils, which computes on-the-fly if needed.
        """
        # Set FIONA_GL2D_DIR temporarily if needed
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            
            x, w = gauss_legendre_1d(self.gl_nodes_per_dim, Xmax)
            
            # Keep only positive x (symmetry of Gauss–Legendre)
            mask = x > 0
            xs = x[mask]            # x_k
            dx = w[mask]            # Δx_k
            x_weights = xs * dx     # x_k Δx_k for ∫ x f(x) dx

            return xs.astype(float), x_weights.astype(float)
        finally:
            # Restore old environment variable
            if old_env is None:
                os.environ.pop("FIONA_GL2D_DIR", None)
            else:
                os.environ["FIONA_GL2D_DIR"] = old_env

    def _load_gl_nodes_unit(self):
        """Load or compute GL nodes on the unit interval [-1, 1]."""
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            x, w = gauss_legendre_1d(self.gl_nodes_per_dim, 1.0)
            mask = x > 0
            rs_unit = x[mask].astype(float)
            w_unit = w[mask].astype(float)
            return rs_unit, w_unit
        finally:
            if old_env is None:
                os.environ.pop("FIONA_GL2D_DIR", None)
            else:
                os.environ["FIONA_GL2D_DIR"] = old_env

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def __call__(self, w_vec, y_vec):

        t0 = time.perf_counter()

        w_vec = np.asarray(w_vec, dtype=float).ravel()
        y_vec = np.asarray(y_vec, dtype=float).ravel()
        if np.any(w_vec == 0):
            raise ValueError("All w must be nonzero.")

        # Load GL nodes on the unit interval once; rescale per frequency below.
        rs_unit, w_unit = self._load_gl_nodes_unit()
        n_gl = self.gl_nodes_per_dim

        nu = 0
        tol = self.tol

        n_w = len(w_vec)
        n_y = len(y_vec)

        # ---- Setup / allocations (Python) ----
        setup_start = time.perf_counter()
        F = np.empty((n_w, n_y), dtype=np.complex128)
        quad_phase = 0.5 * y_vec**2

        # Fixed mode: precompute x-nodes and integrand_phase = x²/2 − ψ(x) once.
        if not self.auto_R_from_gl_nodes:
            Xmax_fixed = self.min_physical_radius
            xs_fixed = rs_unit * Xmax_fixed
            x_weights_fixed = xs_fixed * (w_unit * Xmax_fixed)
            if self.window_u:
                x_weights_fixed = x_weights_fixed * _window_u_taper_1d(
                    xs_fixed, Xmax_fixed, self.window_u_width)
            psi_fixed = self.lens.psi_r(xs_fixed)
            if self.window_potential:
                psi_fixed = psi_fixed * _window_taper_1d(
                    xs_fixed, Xmax_fixed, self.window_radius_fraction)
            integrand_phase_fixed = 0.5 * xs_fixed**2 - psi_fixed
        setup_end = time.perf_counter()

        with CPUTracker() as tracker:

            # ---------- Step 1: Build c_re and c_im batches (Python) ----------
            s1_alloc_start = time.perf_counter()
            c_re = np.empty((n_w, rs_unit.size), float)
            c_im = np.empty((n_w, rs_unit.size), float)
            xs_per_freq = []   # store per-frequency x-nodes for Step 2
            s1_alloc_end = time.perf_counter()

            coeff_loop_start = time.perf_counter()
            psi_time = 0.0
            phase_time = 0.0
            mul_time = 0.0
            assign_time = 0.0

            for i, w in enumerate(w_vec):
                if self.auto_R_from_gl_nodes:
                    # Adaptive mode: compute x-nodes and integrand_phase per frequency
                    Xmax_w = max(self.min_physical_radius,
                                 np.sqrt(n_gl / (2.0 * abs(w))))
                    xs = rs_unit * Xmax_w
                    x_weights = xs * (w_unit * Xmax_w)
                    xs_per_freq.append(xs)

                    if self.window_u:
                        x_weights = x_weights * _window_u_taper_1d(
                            xs, Xmax_w, self.window_u_width)

                    # lens potential ψ(x) — independent of w
                    t_a = time.perf_counter()
                    psi_vals = self.lens.psi_r(xs)
                    if self.window_potential:
                        psi_vals = psi_vals * _window_taper_1d(
                            xs, Xmax_w, self.window_radius_fraction)
                    t_b = time.perf_counter()

                    # integrand phase: x²/2 − ψ(x)
                    integrand_phase = 0.5 * xs**2 - psi_vals
                    t_c = time.perf_counter()
                else:
                    # Fixed mode: reuse precomputed quantities
                    xs = xs_fixed
                    x_weights = x_weights_fixed
                    xs_per_freq.append(xs)
                    integrand_phase = integrand_phase_fixed
                    t_a = t_b = t_c = time.perf_counter()

                # fw = exp(iw · (x²/2 − ψ(x))); subtract free-space for tail correction
                if self.use_tail_correction:
                    fw = (np.exp(1j * w * integrand_phase)
                          - np.exp(1j * w * 0.5 * xs**2))
                else:
                    fw = np.exp(1j * w * integrand_phase)
                t_d = time.perf_counter()

                # coefficients c_k(w) = x_k Δx_k f_w(x_k)
                ck = x_weights * fw
                t_e = time.perf_counter()

                # write into batches
                c_re[i,:] = ck.real
                c_im[i,:] = ck.imag
                t_f = time.perf_counter()

                # accumulate sub-timings
                psi_time    += (t_b - t_a)
                phase_time  += (t_d - t_c)
                mul_time    += (t_e - t_d)
                assign_time += (t_f - t_e)

            coeff_loop_end = time.perf_counter()
            step1_end = coeff_loop_end

            # ---------- Step 2: C NUFHT calls ----------
            # 2a. Allocate output arrays
            nufht_alloc_start = time.perf_counter()
            g_re = np.empty((n_w, n_y), dtype=float)
            g_im = np.empty((n_w, n_y), dtype=float)
            nufht_alloc_end = time.perf_counter()

            # 2b. NUFHT calls at output points |w|·y (x-domain scaling)
            nufht_call_start = time.perf_counter()
            for i in range(n_w):
                xs_i = xs_per_freq[i]
                out_pts = abs(w_vec[i]) * y_vec
                # Call C NUFHT for real part
                g_re[i, :] = _c_nufht(nu, xs_i, c_re[i, :], out_pts)
                # Call C NUFHT for imaginary part
                g_im[i, :] = _c_nufht(nu, xs_i, c_im[i, :], out_pts)
            nufht_call_end = time.perf_counter()
            step2_end = nufht_call_end

            # ---------- Step 3: Assemble full Fresnel integral (Python) ----------
            # F = e^{iwy²/2} · (w/i) · g   [x-domain prefactor]
            step3_loop_start = time.perf_counter()
            for i, w in enumerate(w_vec):
                g = g_re[i] + 1j*g_im[i]
                F[i,:] = np.exp(1j*w*quad_phase) * (w / 1j) * g
                if self.use_tail_correction:
                    F[i,:] = 1.0 + F[i,:]
            step3_loop_end = time.perf_counter()
            step3_end = step3_loop_end

        t_end = time.perf_counter()

        # ==================================================================
        # Timing breakdown
        # ==================================================================
        total_time = t_end - t0

        # Step 0: setup (outside CPUTracker but part of total)
        step0_time = setup_end - setup_start

        # Step 1: coefficients
        step1_total = step1_end - s1_alloc_start
        alloc_time = s1_alloc_end - s1_alloc_start
        coeff_loop_time = coeff_loop_end - coeff_loop_start

        coeff_sub_total = (psi_time + phase_time + mul_time + assign_time)
        coeff_unaccounted = coeff_loop_time - coeff_sub_total
        step1_unaccounted = step1_total - (alloc_time + coeff_loop_time)

        # Step 2: C NUFHT
        step2_total = step2_end - step1_end
        nufht_alloc_time = nufht_alloc_end - nufht_alloc_start
        nufht_time = nufht_call_end - nufht_call_start
        step2_unaccounted = step2_total - (nufht_alloc_time + nufht_time)

        # Step 3: final loop
        step3_total = step3_end - step2_end
        final_loop_time = step3_loop_end - step3_loop_start
        step3_unaccounted = step3_total - final_loop_time

        # Overall unaccounted
        overall_unaccounted = total_time - (step0_time + step1_total +
                                            step2_total + step3_total)

        # ==================================================================
        # Pretty printing
        # ==================================================================
        print()
        print("────────────────────────────────────────────────────────────")
        print(" FresnelHankelAxisymmetric (GL_precomputed + C NUFHT, x-domain)")
        print("────────────────────────────────────────────────────────────")
        print(tracker.report("  CPU usage summary"))
        print()

        # ---- Step 0 ----
        print("  Step 0: Setup / allocations")
        print("  ───────────────────────────")
        print(f"    0a. F, quad_phase (+ fixed precomp): "
              f"{_pct(step0_time, total_time):6.2f}%  ({step0_time:10.6f} s)")
        print()

        # ---- Step 1 ----
        print("  Step 1: Coefficient Computation (Python)")
        print("  ───────────────────────────────────────")
        print(f"    1a. allocate c_re/c_im           : "
              f"{_pct(alloc_time, step1_total):6.2f}%  ({alloc_time:10.6f} s)")
        print(f"    1b. coefficient loop (total)     : "
              f"{_pct(coeff_loop_time, step1_total):6.2f}%  ({coeff_loop_time:10.6f} s)")
        print(f"        ├─ lens potential ψ(x)       : "
              f"{_pct(psi_time, step1_total):6.2f}%  ({psi_time:10.6f} s)")
        print(f"        ├─ integrand phase + exp(iw·φ): "
              f"{_pct(phase_time, step1_total):6.2f}%  ({phase_time:10.6f} s)")
        print(f"        ├─ multiply by x_k Δx_k      : "
              f"{_pct(mul_time, step1_total):6.2f}%  ({mul_time:10.6f} s)")
        print(f"        ├─ assign into c_re/c_im     : "
              f"{_pct(assign_time, step1_total):6.2f}%  ({assign_time:10.6f} s)")
        print(f"        └─ unaccounted (loop)        : "
              f"{_pct(coeff_unaccounted, step1_total):6.2f}%  ({coeff_unaccounted:10.6f} s)")
        print(f"    1c. other (Step 1)               : "
              f"{_pct(step1_unaccounted, step1_total):6.2f}%  ({step1_unaccounted:10.6f} s)")
        print()
        print(f"    Step 1 total                     : "
              f"{_pct(step1_total, total_time):6.2f}%  ({step1_total:10.6f} s)")
        print()

        # ---- Step 2 ----
        print("  Step 2: C NUFHT")
        print("  ───────────────")
        print(f"    2a. allocate g_re/g_im           : "
              f"{_pct(nufht_alloc_time, step2_total):6.2f}%  ({nufht_alloc_time:10.6f} s)")
        print(f"    2b. NUFHT calls (loop over w)    : "
              f"{_pct(nufht_time, step2_total):6.2f}%  ({nufht_time:10.6f} s)")
        print(f"    2c. other (Step 2)               : "
              f"{_pct(step2_unaccounted, step2_total):6.2f}%  ({step2_unaccounted:10.6f} s)")
        print()
        print(f"    Step 2 total                     : "
              f"{_pct(step2_total, total_time):6.2f}%  ({step2_total:10.6f} s)")
        print()

        # ---- Step 3 ----
        print("  Step 3: Final per-w loop (Python)")
        print("  ─────────────────────────────────")
        print(f"    3a. apply quad_phase & (w/i)     : "
              f"{_pct(final_loop_time, step3_total):6.2f}%  ({final_loop_time:10.6f} s)")
        print(f"    3b. other (Step 3)               : "
              f"{_pct(step3_unaccounted, step3_total):6.2f}%  ({step3_unaccounted:10.6f} s)")
        print()
        print(f"    Step 3 total                     : "
              f"{_pct(step3_total, total_time):6.2f}%  ({step3_total:10.6f} s)")
        print()

        # ---- Overall ----
        print("Overall Timing Summary (percent of TOTAL)")
        print("────────────────────────────────────────────────────────────")
        print(f"  0. Setup                           : "
              f"{_pct(step0_time, total_time):6.2f}%  ({step0_time:10.6f} s)")
        print(f"  1. Coefficients (Python)           : "
              f"{_pct(step1_total, total_time):6.2f}%  ({step1_total:10.6f} s)")
        print(f"  2. C NUFHT                         : "
              f"{_pct(step2_total, total_time):6.2f}%  ({step2_total:10.6f} s)")
        print(f"  3. Final per-w loop (Python)       : "
              f"{_pct(step3_total, total_time):6.2f}%  ({step3_total:10.6f} s)")
        print(f"  4. Unaccounted / overhead          : "
              f"{_pct(overall_unaccounted, total_time):6.2f}%  ({overall_unaccounted:10.6f} s)")
        print("────────────────────────────────────────────────────────────")
        print(f"  TOTAL                              : "
              f"{_pct(total_time, total_time):6.2f}%  ({total_time:10.6f} s)")
        print("────────────────────────────────────────────────────────────")
        print()

        return F


# ----------------------------------------------------------------------
# Batched NUFHT with process-level parallelism
# ----------------------------------------------------------------------
class FresnelNUFHTBatched:
    """
    Fresnel integral for axisymmetric lenses using **batched** NUFHT
    (``_pynufht.nufht_batch``) with process-based parallelism.

    This class matches the design of the ``marckamion/NUFHT`` project, which
    exposes a batched C API::

        nufht_batch(nu, rs, cs_matrix, ws, tol=...) -> array(n_y, batch)

    where ``cs_matrix`` is a Fortran-contiguous ``(m, batch)`` array.

    Instead of iterating over frequencies one at a time (as in
    :class:`FresnelNUFHT`), this implementation:

    1. Splits ``w_vec`` into up to ``n_workers`` chunks.
    2. Dispatches each chunk to a subprocess via
       ``concurrent.futures.ProcessPoolExecutor``.
    3. Within each subprocess a *single* ``nufht_batch`` call is made for
       the real coefficients and a *single* call for the imaginary
       coefficients (two calls total per worker-chunk).

    **Prerequisites**

    - ``_pynufht`` must expose ``nufht_batch``.  If only ``nufht`` is
      available, use :class:`FresnelNUFHT` instead.
    - The ``lens`` object must be **pickleable** (required for
      ``ProcessPoolExecutor``).  If your lens is not pickleable you can
      set ``n_workers=1``, which runs the computation in the current
      process and avoids pickling entirely.

    Parameters
    ----------
    lens : AxisymmetricLens
        Axisymmetric lens object with ``psi_r`` method.  Must be
        pickleable when ``n_workers > 1``.
    gl_nodes_per_dim : int, optional
        Number of Gauss–Legendre nodes per dimension (default 128).
        Ignored when ``adaptive_n_gl=True``.
    min_physical_radius : float, optional
        Minimum physical radius ``Xmax`` to use (default 1.0).
    auto_R_from_gl_nodes : bool, optional
        If ``True`` (default), adapt ``Xmax`` from the frequency range.
    adaptive_n_gl : bool, optional
        If ``True`` (default), automatically choose the number of GL nodes
        per frequency bin using the same schedule as ``general.py``:
        n_gl = 1000 * clip(floor(|w|/10) + 1, 1, 10).  Frequencies in the
        same bin share a single GL grid.  ``gl_nodes_per_dim`` is ignored
        when this is enabled.
    gl_dir : str or None, optional
        Directory for GL node files.  Falls back to ``FIONA_GL2D_DIR``
        environment variable.
    tol : float, optional
        Tolerance passed to ``nufht_batch`` (default ``1e-12``).
    n_workers : int, optional
        Number of parallel worker processes (default 112).  Set to 1
        to run in-process without a ``ProcessPoolExecutor``.
    """

    _DEFAULT_WORKERS = 112

    def __init__(self,
                 lens: AxisymmetricLens,
                 gl_nodes_per_dim: int = None,
                 min_physical_radius: float = None,
                 auto_R_from_gl_nodes: bool = True,
                 adaptive_n_gl: bool = True,
                 gl_dir: str = None,
                 tol: float = 1e-12,
                 n_workers: int = None,
                 window_potential: bool = True,
                 window_radius_fraction: float = 0.75,
                 window_u: bool = True,
                 window_u_width: float = 0.02,
                 use_tail_correction: bool = True):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError("FresnelNUFHTBatched requires AxisymmetricLens")

        if not _HAS_NUFHT_BATCH:
            raise ImportError(
                "_pynufht.nufht_batch is not available. "
                "Install the marckamion/NUFHT extension and ensure it exposes "
                f"nufht_batch.  Original error: {_NUFHT_BATCH_ERR!r}"
            )

        if gl_nodes_per_dim is None:
            gl_nodes_per_dim = 128
        if min_physical_radius is None:
            min_physical_radius = 1.0
        if n_workers is None:
            n_workers = self._DEFAULT_WORKERS

        self.lens = lens
        self.gl_nodes_per_dim = int(gl_nodes_per_dim)
        self.min_physical_radius = float(min_physical_radius)
        self.auto_R_from_gl_nodes = bool(auto_R_from_gl_nodes)
        self.adaptive_n_gl = bool(adaptive_n_gl)
        self.tol = float(tol)
        self.n_workers = int(n_workers)
        self.window_potential = bool(window_potential)
        self.window_radius_fraction = float(window_radius_fraction)
        self.window_u = bool(window_u)
        self.window_u_width = float(window_u_width)
        self.use_tail_correction = bool(use_tail_correction)

        if gl_dir is None:
            gl_dir = os.environ.get("FIONA_GL2D_DIR", "")
            if not gl_dir:
                raise RuntimeError(
                    "gl_dir not provided and FIONA_GL2D_DIR environment variable "
                    "not set."
                )
        self._gl_dir = gl_dir

    def _load_gl_nodes(self, Xmax, n_gl=None):
        """Load or compute GL nodes for given *Xmax* (x-domain radius) and *n_gl*."""
        if n_gl is None:
            n_gl = self.gl_nodes_per_dim
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            x, w = gauss_legendre_1d(n_gl, Xmax)
            mask = x > 0
            xs = x[mask].astype(float)
            dx = w[mask].astype(float)
            x_weights = xs * dx
            return xs, x_weights
        finally:
            if old_env is None:
                os.environ.pop("FIONA_GL2D_DIR", None)
            else:
                os.environ["FIONA_GL2D_DIR"] = old_env

    def _load_gl_nodes_unit(self, n_gl=None):
        """Load or compute GL nodes on the unit interval [-1, 1] for given *n_gl*."""
        if n_gl is None:
            n_gl = self.gl_nodes_per_dim
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            x, w = gauss_legendre_1d(n_gl, 1.0)
            mask = x > 0
            rs_unit = x[mask].astype(float)
            w_unit = w[mask].astype(float)
            return rs_unit, w_unit
        finally:
            if old_env is None:
                os.environ.pop("FIONA_GL2D_DIR", None)
            else:
                os.environ["FIONA_GL2D_DIR"] = old_env

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def __call__(self, w_vec, y_vec):
        """
        Evaluate ``F(w, y)`` for arrays of frequencies *w* and radii *y*.

        Parameters
        ----------
        w_vec : array_like, shape (n_w,)
            Frequencies (must all be nonzero).
        y_vec : array_like, shape (n_y,)
            Output radii.

        Returns
        -------
        F : ndarray of complex128, shape (n_w, n_y)
        """
        w_vec = np.asarray(w_vec, dtype=float).ravel()
        y_vec = np.asarray(y_vec, dtype=float).ravel()
        if np.any(w_vec == 0):
            raise ValueError("All w must be nonzero.")

        n_w = len(w_vec)
        n_y = len(y_vec)
        F = np.empty((n_w, n_y), dtype=np.complex128)
        quad_phase = 0.5 * y_vec ** 2
        nu = 0

        # Group frequencies by n_gl bin (adaptive) or use a single group (fixed)
        if self.adaptive_n_gl:
            w_abs = np.abs(w_vec)
            n_gl_vec = np.array([_adaptive_n_gl(v) for v in w_abs], dtype=int)
            groups = {}
            for idx, n_gl in enumerate(n_gl_vec):
                groups.setdefault(n_gl, []).append(idx)
            group_items = [
                (n_gl, np.asarray(idxs, dtype=int))
                for n_gl, idxs in sorted(groups.items())
            ]
        else:
            group_items = [(self.gl_nodes_per_dim, np.arange(n_w, dtype=int))]

        for n_gl, idxs in group_items:
            w_sub = w_vec[idxs]

            # Load GL nodes on the unit interval [-1, 1] once per group.
            # Each worker will rescale them per frequency using
            # Xmax(w) = max(min_physical_radius, sqrt(n_gl / (2*|w|))).
            rs_unit, w_unit = self._load_gl_nodes_unit(n_gl)

            # n_gl_for_worker > 0 enables per-frequency Xmax adaptation;
            # 0 signals fixed mode (Xmax = min_physical_radius for all w).
            n_gl_for_worker = n_gl if self.auto_R_from_gl_nodes else 0

            # Split this group's frequencies into chunks for parallel workers
            n_sub = len(idxs)
            n_chunks = min(self.n_workers, n_sub)
            if n_chunks < 1:
                n_chunks = 1
            chunk_splits = np.array_split(np.arange(n_sub), n_chunks)

            worker_args = [
                (self.lens, rs_unit, w_unit, y_vec, w_sub[ch], nu, self.tol,
                 self.min_physical_radius, n_gl_for_worker,
                 self.window_potential, self.window_radius_fraction,
                 self.window_u, self.window_u_width, self.use_tail_correction)
                for ch in chunk_splits
                if len(ch) > 0
            ]
            # Track which global indices each chunk covers
            valid_chunks = [ch for ch in chunk_splits if len(ch) > 0]

            if self.n_workers == 1:
                results = [_nufht_batch_worker(a) for a in worker_args]
            else:
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=self.n_workers
                ) as executor:
                    results = list(executor.map(_nufht_batch_worker, worker_args))

            # Assemble results for this group
            for ch, (G_re, G_im) in zip(valid_chunks, results):
                for j, sub_j in enumerate(ch):
                    i = idxs[sub_j]
                    w = w_vec[i]
                    g = G_re[:, j] + 1j * G_im[:, j]
                    F[i, :] = np.exp(1j * w * quad_phase) * (w / 1j) * g
                    if self.use_tail_correction:
                        F[i, :] = 1.0 + F[i, :]

        return F


class FresnelHankelAxisymmetricTrapezoidal:
    r"""
    Fresnel integral for axisymmetric lenses using a fast Hankel transform
    in the **x-domain** formulation.

        F(w, y) = (w/i) · e^{iwy²/2} ∫_0^∞ x dx
                  exp{iw[x²/2 − ψ(x)]} J_0(wxy),

    We discretize the radial x-integral on [0, Xmax] and evaluate the Bessel sum
    with C-based NUFHT (nonuniform fast Hankel transform).

        ∫_0^Xmax x f_w(x) J_0(wxy) dx ≈ ∑_k c_k(w) J_0(|w|y r_k),

    where r_k are radial x-nodes, and

        c_k(w) = x_k Δx_k * exp{iw[x_k²/2 − ψ(x_k)]}.

    nufht(ν, r_k, c_k, |w|·y_j) then returns the vector of Hankel sums
    g_j ≈ ∑_k c_k J_ν(|w|·y_j · r_k).  We use a zeroth-order FHT (ν=0).

    Parameters
    ----------
    lens : AxisymmetricLens
        Axisymmetric lens object with psi_r method.
    n_r : int
        Number of radial grid points.
    min_physical_radius : float
        Minimum physical radius (Xmax) to use.
    auto_R_from_gl_nodes : bool
        If True, adapt Xmax based on frequency range.
        If False, use fixed min_physical_radius.
        Note: For this class we use n_r instead of gl_nodes_per_dim in the formula.
    tol : float
        Tolerance for NUFHT.
    """

    def __init__(self, lens: AxisymmetricLens,
                 n_r: int = 1024,
                 min_physical_radius: float = None,
                 auto_R_from_gl_nodes: bool = False,
                 tol: float = 1e-12,
                 window_potential: bool = True,
                 window_radius_fraction: float = 0.75,
                 window_u: bool = True,
                 window_u_width: float = 0.02,
                 use_tail_correction: bool = True,
                 # Deprecated parameter for backward compatibility
                 Umax: float = None):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError(
                "FresnelHankelAxisymmetric requires an AxisymmetricLens instance."
            )

        if not _HAS_FHT:
            raise ImportError(
                "NUFHT C extension (_pynufht) is not available.\n"
                f"Original import error: {_FHT_ERR!r}"
            )

        # Handle backward compatibility for old parameter name
        if Umax is not None:
            if min_physical_radius is not None:
                raise ValueError("Cannot specify both Umax and min_physical_radius")
            min_physical_radius = Umax
            # If old Umax is specified, default to no auto-adaptation for backward compat
            auto_R_from_gl_nodes = False

        # Set default if still None
        if min_physical_radius is None:
            min_physical_radius = 50.0

        self.lens = lens
        self.n_r = int(n_r)
        self.min_physical_radius = float(min_physical_radius)
        self.auto_R_from_gl_nodes = bool(auto_R_from_gl_nodes)
        self.tol = float(tol)
        self.window_potential = bool(window_potential)
        self.window_radius_fraction = float(window_radius_fraction)
        self.window_u = bool(window_u)
        self.window_u_width = float(window_u_width)
        self.use_tail_correction = bool(use_tail_correction)

        # Don't build grid yet - will be built in __call__ based on frequency range

    # ------------------------------------------------------------------
    # Radial grid and quadrature weights
    # ------------------------------------------------------------------
    def _build_radial_grid(self, Xmax):
        """
        Simple uniform radial grid on (0, Xmax] with trapezoidal weights.

        We avoid r=0 to keep things well-behaved numerically; the missing
        interval [0, r_min] is negligible for sufficiently large n_r.
        """
        n = self.n_r

        # n+1 points from 0 to Xmax, then drop the first (0).
        xs_full = np.linspace(0.0, Xmax, n + 1, dtype=float)
        xs = xs_full[1:]               # shape (n,)
        dx = xs_full[1] - xs_full[0]

        # Trapezoidal weights for ∫_0^{Xmax} … dx.
        w = np.ones_like(xs) * dx
        w[0] *= 0.5
        w[-1] *= 0.5

        # For ∫ x f(x) dx, combine with the extra factor x:
        x_weights = xs * w            # x_k Δx_k

        return xs, x_weights

    def __call__(self, w_vec, y_vec):
        """
        Evaluate F(w, y) for arrays of frequencies w and radii y.
        """
        w_vec = np.asarray(w_vec, dtype=float).ravel()
        y_vec = np.asarray(y_vec, dtype=float).ravel()

        if np.any(w_vec == 0.0):
            raise ValueError("All w must be nonzero.")

        # Determine Xmax per frequency (computed inside the loop below).
        # For fixed mode (auto_R_from_gl_nodes=False), Xmax = min_physical_radius.
        nu = 0                       # J_0 Hankel transform
        tol = self.tol

        n_w = len(w_vec)
        n_y = len(y_vec)

        F = np.empty((n_w, n_y), dtype=np.complex128)
        quad_phase = 0.5 * (y_vec ** 2)   # y^2 / 2

        # ───────────────────────────────
        # Wall-clock timing
        # ───────────────────────────────
        t0 = time.perf_counter()

        with CPUTracker() as tracker:
            # ---------------- Step 1: Precompute coefficients in Python ----------------
            # 1a. basic allocations
            s1_alloc_start = time.perf_counter()
            all_c_re = []
            all_c_im = []
            xs_per_freq = []
            s1_alloc_end = time.perf_counter()

            # 1b. main coefficient loop, with internal breakdown
            coeff_loop_start = time.perf_counter()
            psi_time = 0.0
            phase_time = 0.0
            mul_time = 0.0

            for w in w_vec:
                # Per-frequency x-domain integration radius
                if self.auto_R_from_gl_nodes:
                    Xmax_w = max(self.min_physical_radius,
                                 np.sqrt(self.n_r / (2.0 * abs(w))))
                else:
                    Xmax_w = self.min_physical_radius
                xs, x_weights = self._build_radial_grid(Xmax_w)
                xs_per_freq.append(xs)

                if self.window_u:
                    x_weights = x_weights * _window_u_taper_1d(xs, Xmax_w, self.window_u_width)

                # lens potential ψ(x) — independent of w
                t_a = time.perf_counter()
                psi_vals = self.lens.psi_r(xs)
                if self.window_potential:
                    psi_vals = psi_vals * _window_taper_1d(xs, Xmax_w, self.window_radius_fraction)
                t_b = time.perf_counter()

                # fw = exp(iw·(x²/2 − ψ(x))); subtract free-space for tail correction
                if self.use_tail_correction:
                    integrand_phase = 0.5 * xs**2 - psi_vals
                    f_w = (np.exp(1j * w * integrand_phase)
                           - np.exp(1j * w * 0.5 * xs**2))
                else:
                    integrand_phase = 0.5 * xs**2 - psi_vals
                    f_w = np.exp(1j * w * integrand_phase)
                t_c = time.perf_counter()

                # coefficients c_k(w) = x_k Δx_k f_w(x_k)
                c_k = x_weights * f_w
                t_d = time.perf_counter()

                all_c_re.append(c_k.real.astype(float))
                all_c_im.append(c_k.imag.astype(float))

                # accumulate sub-timings
                psi_time   += (t_b - t_a)
                phase_time += (t_c - t_b)
                mul_time   += (t_d - t_c)

            coeff_loop_end = time.perf_counter()

            t1 = coeff_loop_end

            # ---------------- Step 2: C NUFHT ----------------
            # 2a. Allocate output arrays
            nufht_alloc_start = time.perf_counter()
            g_re = np.empty((n_w, n_y), dtype=float)
            g_im = np.empty((n_w, n_y), dtype=float)
            nufht_alloc_end = time.perf_counter()

            # 2b. NUFHT calls at output points |w|·y (x-domain scaling)
            nufht_call_start = time.perf_counter()
            for i in range(n_w):
                xs_i = xs_per_freq[i]
                out_pts = abs(w_vec[i]) * y_vec
                # Call C NUFHT for real part
                g_re[i, :] = _c_nufht(nu, xs_i, all_c_re[i], out_pts)
                # Call C NUFHT for imaginary part
                g_im[i, :] = _c_nufht(nu, xs_i, all_c_im[i], out_pts)
            nufht_call_end = time.perf_counter()

            t2 = nufht_call_end

            # ---------------- Step 3: Final assembly in Python ----------------
            # F = e^{iwy²/2} · (w/i) · g   [x-domain prefactor]
            final_loop_start = time.perf_counter()
            for i, w in enumerate(w_vec):
                g = g_re[i, :] + 1j * g_im[i, :]
                F[i, :] = np.exp(1j * w * quad_phase) * (w / 1j) * g
                if self.use_tail_correction:
                    F[i, :] = 1.0 + F[i, :]
            final_loop_end = time.perf_counter()
            t3 = final_loop_end

        t4 = time.perf_counter()

        # ───────────────────────────────
        # Aggregate timings
        # ───────────────────────────────
        total_time = t4 - t0

        # Step 1 breakdown
        step1_total = t1 - t0
        alloc_lists_time = s1_alloc_end - s1_alloc_start
        coeff_loop_time = coeff_loop_end - coeff_loop_start

        coeff_sub_total = psi_time + phase_time + mul_time
        coeff_unaccounted = coeff_loop_time - coeff_sub_total
        step1_unaccounted = step1_total - (alloc_lists_time + coeff_loop_time)

        # Step 2 breakdown
        step2_total = t2 - t1
        nufht_alloc_time = nufht_alloc_end - nufht_alloc_start
        nufht_time = nufht_call_end - nufht_call_start
        step2_unaccounted = step2_total - (nufht_alloc_time + nufht_time)

        # Step 3 breakdown
        step3_total = t3 - t2
        final_loop_time = final_loop_end - final_loop_start  # should be ≈ step3_total
        step3_unaccounted = step3_total - final_loop_time

        # Overall unaccounted (e.g. tiny overhead, CPUTracker context, etc.)
        overall_unaccounted = total_time - (step1_total + step2_total + step3_total)

        # ───────────────────────────────
        # Pretty printing
        # ───────────────────────────────
        print()
        print("────────────────────────────────────────────────────────────")
        print(" FresnelHankelAxisymmetric (Trapezoidal + C NUFHT, x-domain)")
        print("────────────────────────────────────────────────────────────")
        print(tracker.report("  CPU usage summary"))
        print()

        # ---- Step 1 ----
        print("  Step 1: Coefficient Computation (Python + NumPy)")
        print("  ────────────────────────────────────────────────")
        print(f"    1a. list setup/alloc (all_c_*)   : "
              f"{_pct(alloc_lists_time, step1_total):6.2f}%  ({alloc_lists_time:10.6f} s)")
        print(f"    1b. coefficient loop (total)     : "
              f"{_pct(coeff_loop_time, step1_total):6.2f}%  ({coeff_loop_time:10.6f} s)")
        print(f"        ├─ lens potential ψ(x)       : "
              f"{_pct(psi_time, step1_total):6.2f}%  ({psi_time:10.6f} s)")
        print(f"        ├─ integrand phase + exp(iw·φ): "
              f"{_pct(phase_time, step1_total):6.2f}%  ({phase_time:10.6f} s)")
        print(f"        ├─ multiply by x_k Δx_k      : "
              f"{_pct(mul_time, step1_total):6.2f}%  ({mul_time:10.6f} s)")
        print(f"        └─ unaccounted (loop)        : "
              f"{_pct(coeff_unaccounted, step1_total):6.2f}%  ({coeff_unaccounted:10.6f} s)")
        print(f"    1c. other (Step 1)               : "
              f"{_pct(step1_unaccounted, step1_total):6.2f}%  ({step1_unaccounted:10.6f} s)")
        print()
        print(f"    Step 1 total                     : "
              f"{_pct(step1_total, total_time):6.2f}%  ({step1_total:10.6f} s)")
        print()

        # ---- Step 2 ----
        print("  Step 2: C NUFHT")
        print("  ───────────────")
        print(f"    2a. allocate g_re/g_im           : "
              f"{_pct(nufht_alloc_time, step2_total):6.2f}%  ({nufht_alloc_time:10.6f} s)")
        print(f"    2b. NUFHT calls (loop over w)    : "
              f"{_pct(nufht_time, step2_total):6.2f}%  ({nufht_time:10.6f} s)")
        print(f"    2c. other (Step 2)               : "
              f"{_pct(step2_unaccounted, step2_total):6.2f}%  ({step2_unaccounted:10.6f} s)")
        print()
        print(f"    Step 2 total                     : "
              f"{_pct(step2_total, total_time):6.2f}%  ({step2_total:10.6f} s)")
        print()

        # ---- Step 3 ----
        print("  Step 3: Final per-w loop (Python)")
        print("  ──────────────────────────────────")
        print(f"    3a. apply quad_phase & (w/i)     : "
              f"{_pct(final_loop_time, step3_total):6.2f}%  ({final_loop_time:10.6f} s)")
        print(f"    3b. other (Step 3)               : "
              f"{_pct(step3_unaccounted, step3_total):6.2f}%  ({step3_unaccounted:10.6f} s)")
        print()
        print(f"    Step 3 total                     : "
              f"{_pct(step3_total, total_time):6.2f}%  ({step3_total:10.6f} s)")
        print()

        # ---- Overall ----
        print("Overall Timing Summary (percent of TOTAL)")
        print("────────────────────────────────────────────────────────────")
        print(f"  1. Coefficients (Step 1)           : "
              f"{_pct(step1_total, total_time):6.2f}%  ({step1_total:10.6f} s)")
        print(f"  2. C NUFHT (Step 2)                : "
              f"{_pct(step2_total, total_time):6.2f}%  ({step2_total:10.6f} s)")
        print(f"  3. Final per-w loop (Step 3)       : "
              f"{_pct(step3_total, total_time):6.2f}%  ({step3_total:10.6f} s)")
        print(f"  4. Unaccounted / overhead          : "
              f"{_pct(overall_unaccounted, total_time):6.2f}%  ({overall_unaccounted:10.6f} s)")
        print("────────────────────────────────────────────────────────────")
        print(f"  TOTAL                              : "
              f"{_pct(total_time, total_time):6.2f}%  ({total_time:10.6f} s)")
        print("────────────────────────────────────────────────────────────")
        print()

        return F

class FresnelHankelAxisymmetricSciPy:
    """
    Fresnel integral for axisymmetric lenses using Gauss-Legendre nodes
    for configuration (gl_nodes_per_dim, Xmax) in the **x-domain**, and
    performing the Hankel transform with SciPy's FFTLog-based fast Hankel
    transform (scipy.fft.fht).

    GL nodes are loaded from precomputed files if available, or computed
    on-the-fly and saved to disk (via gauss_legendre_1d from utils.py).

    Parameters
    ----------
    lens : AxisymmetricLens
        Axisymmetric lens object with psi_r method.
    gl_nodes_per_dim : int
        Number of Gauss-Legendre nodes per dimension.
    min_physical_radius : float
        Minimum physical radius (Xmax) to use.
    auto_R_from_gl_nodes : bool
        If True, adapt Xmax based on frequency range.
        If False, use fixed min_physical_radius.
    gl_dir : str or None
        Directory for GL node files. If None, uses FIONA_GL2D_DIR env var.
    tol : float
        Tolerance for computation.
    """

    def __init__(self,
                 lens: AxisymmetricLens,
                 gl_nodes_per_dim: int = None,
                 min_physical_radius: float = None,
                 auto_R_from_gl_nodes: bool = True,
                 gl_dir: str = None,
                 tol: float = 1e-12,
                 window_potential: bool = True,
                 window_radius_fraction: float = 0.75,
                 window_u: bool = True,
                 window_u_width: float = 0.02,
                 use_tail_correction: bool = True,
                 # Deprecated parameters for backward compatibility
                 n_gl: int = None,
                 Umax: float = None):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError("FresnelHankelAxisymmetricSciPy requires AxisymmetricLens")

        if not _HAS_SCIPY_FHT:
            raise ImportError(
                "scipy.fft.fht (fast Hankel transform) is not available.\n"
                f"Original import error: {_SCIPY_FHT_ERR!r}"
            )

        # Handle backward compatibility for old parameter names
        if n_gl is not None:
            if gl_nodes_per_dim is not None:
                raise ValueError("Cannot specify both n_gl and gl_nodes_per_dim")
            gl_nodes_per_dim = n_gl
        if Umax is not None:
            if min_physical_radius is not None:
                raise ValueError("Cannot specify both Umax and min_physical_radius")
            min_physical_radius = Umax
            # If old Umax is specified, default to no auto-adaptation for backward compat
            if n_gl is not None:  # Both old params specified
                auto_R_from_gl_nodes = False

        # Set defaults if still None
        if gl_nodes_per_dim is None:
            gl_nodes_per_dim = 128
        if min_physical_radius is None:
            min_physical_radius = 1.0

        self.lens = lens
        self.gl_nodes_per_dim = int(gl_nodes_per_dim)
        self.min_physical_radius = float(min_physical_radius)
        self.auto_R_from_gl_nodes = bool(auto_R_from_gl_nodes)
        self.tol = float(tol)
        self.window_potential = bool(window_potential)
        self.window_radius_fraction = float(window_radius_fraction)
        self.window_u = bool(window_u)
        self.window_u_width = float(window_u_width)
        self.use_tail_correction = bool(use_tail_correction)

        # Store gl_dir for later use
        if gl_dir is None:
            gl_dir = os.environ.get("FIONA_GL2D_DIR", "")
            if not gl_dir:
                raise RuntimeError(
                    "gl_dir not provided and FIONA_GL2D_DIR environment variable not set."
                )
        self._gl_dir = gl_dir

        # Don't load nodes yet - will be loaded in __call__ based on frequency range

    def _load_and_setup_grid(self, Xmax):
        """
        Load or compute GL nodes for given Xmax (x-domain radius) and setup FFTLog grid.
        Uses gauss_legendre_1d from utils, which computes on-the-fly if needed.
        """
        # Set FIONA_GL2D_DIR temporarily if needed
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            
            x, w = gauss_legendre_1d(self.gl_nodes_per_dim, Xmax)
            
            mask = x > 0
            xs_gl = x[mask]      # original GL radial nodes x_k

            # We use the GL nodes to define the radial range and sample count
            r_min = float(xs_gl.min())
            r_max = float(xs_gl.max())
            n_r = xs_gl.size

            # --- Build logarithmic radial grid, r_j = r_c * exp[(j-j_c)*dln] ---
            r = np.geomspace(r_min, r_max, n_r)
            dln = float(np.log(r[1] / r[0]))
            mu = 0.0
            bias = 0.0
            offset = _scipy_fhtoffset(dln, mu=mu, bias=bias)
            k = np.exp(offset) / r[::-1]

            return r, dln, mu, bias, offset, k
        finally:
            # Restore old environment variable
            if old_env is None:
                os.environ.pop("FIONA_GL2D_DIR", None)
            else:
                os.environ["FIONA_GL2D_DIR"] = old_env

    def __call__(self, w_vec, y_vec):
        """
        Evaluate F(w, y) on arrays of frequencies `w_vec` and radii `y_vec`.

        Uses SciPy's fast Hankel transform (FFTLog) in the x-domain.
        For each w, we compute the x-domain integral

            g_w(y) = ∫_0^∞ x f_w(x) J_0(|w|xy) dx,

        by mapping to SciPy's definition

            A(k) = ∫_0^∞ a(r) J_0(k r) k dr,

        with a(r) = x f_w(x), so that A(k) ≈ k · g_w(k/|w|), hence

            g_w(y) ≈ A(|w|y) / (|w|y),

        and then interpolate in log-space onto the requested y values.
        """
        w_vec = np.asarray(w_vec, dtype=float).ravel()
        y_vec = np.asarray(y_vec, dtype=float).ravel()

        if np.any(w_vec == 0.0):
            raise ValueError("All w must be nonzero.")
        if np.any(y_vec <= 0.0):
            raise ValueError("FresnelHankelAxisymmetricSciPy currently requires y > 0.")

        # Set up the unit grid once; per-frequency xs and k are derived by scaling.
        # dln and offset are scale-invariant and shared across all frequencies.
        r_unit, dln, mu, bias, offset, k_unit = self._load_and_setup_grid(1.0)

        # The sort order of k = k_unit / Xmax is independent of Xmax (positive scaling),
        # so pre-compute it and the sorted unit k once.
        idx = np.argsort(k_unit)
        k_unit_sorted = k_unit[idx]
        logk_unit_sorted = np.log(k_unit_sorted)

        n_w = len(w_vec)
        n_y = len(y_vec)

        F = np.empty((n_w, n_y), dtype=np.complex128)
        quad_phase = 0.5 * y_vec**2

        # ───────────────────────────────
        # Wall-clock timing
        # ───────────────────────────────
        t0 = time.perf_counter()

        with CPUTracker() as tracker:
            # ---------------- Per-frequency loop (coeff + FHT + interp) ----------------
            coeff_loop_start = time.perf_counter()
            logy = np.log(y_vec)
            for i, w in enumerate(w_vec):
                # Per-frequency x-domain integration radius, then scale the unit grid
                if self.auto_R_from_gl_nodes:
                    Xmax_w = max(self.min_physical_radius,
                                 np.sqrt(self.gl_nodes_per_dim / (2.0 * abs(w))))
                else:
                    Xmax_w = self.min_physical_radius
                xs = r_unit * Xmax_w

                # lens potential ψ(x) — independent of w
                psi_vals = self.lens.psi_r(xs)
                if self.window_potential:
                    psi_vals = psi_vals * _window_taper_1d(xs, Xmax_w, self.window_radius_fraction)

                # fw = exp(iw·(x²/2 − ψ(x))); subtract free-space for tail correction
                if self.use_tail_correction:
                    integrand_phase = 0.5 * xs**2 - psi_vals
                    f_w = (np.exp(1j * w * integrand_phase)
                           - np.exp(1j * w * 0.5 * xs**2))
                else:
                    integrand_phase = 0.5 * xs**2 - psi_vals
                    f_w = np.exp(1j * w * integrand_phase)

                if self.window_u:
                    f_w = f_w * _window_u_taper_1d(xs, Xmax_w, self.window_u_width)

                a_r = xs * f_w  # a(x) = x f_w(x) for SciPy's integral definition

                # FHT for this frequency: A(k) = k · ∫ x f_w(x) J_0(kx) dx
                A_re_i = _scipy_fht(a_r.real, dln, mu=mu, offset=offset)
                A_im_i = _scipy_fht(a_r.imag, dln, mu=mu, offset=offset)
                A_i = A_re_i + 1j * A_im_i

                # k = k_unit / Xmax_w; use pre-sorted unit arrays and adjust for scale.
                # Interpolation target: k = |w|·y  →  log k = log|w| + log y
                logk_sorted = logk_unit_sorted - np.log(Xmax_w)
                Ak = A_i[idx]
                Ak_over_k = Ak * Xmax_w / k_unit_sorted  # = Ak / (k_unit_sorted / Xmax_w)

                # g_w(y) = A(|w|y) / (|w|y); interpolate at log(|w|·y)
                logy_w = logy + np.log(abs(w))
                real_part = np.interp(logy_w, logk_sorted, Ak_over_k.real)
                imag_part = np.interp(logy_w, logk_sorted, Ak_over_k.imag)
                g_y = real_part + 1j * imag_part

                # F = e^{iwy²/2} · (w/i) · g_w(y)   [x-domain prefactor]
                F[i, :] = np.exp(1j * w * quad_phase) * (w / 1j) * g_y
                if self.use_tail_correction:
                    F[i, :] = 1.0 + F[i, :]

            coeff_loop_end = time.perf_counter()

        t4 = time.perf_counter()

        # ───────────────────────────────
        # Timing breakdown
        # ───────────────────────────────
        total_time = t4 - t0
        loop_time = coeff_loop_end - coeff_loop_start
        unaccounted = total_time - loop_time

        # ───────────────────────────────
        # Pretty printing
        # ───────────────────────────────
        print()
        print("────────────────────────────────────────────────────────────")
        print(" FresnelHankelAxisymmetricSciPy Timing (x-domain)")
        print("────────────────────────────────────────────────────────────")
        print(tracker.report("  CPU usage summary"))
        print()

        print("  Per-frequency loop (coeff + FHT + interp)")
        print("  ───────────────────────────────────────────")
        print(f"    Per-frequency loop (all w)       : "
              f"{_pct(loop_time, total_time):6.2f}%  ({loop_time:10.6f} s)")
        print()

        print("Overall Timing Summary (percent of TOTAL)")
        print("────────────────────────────────────────────────────────────")
        print(f"  1. Per-frequency loop              : "
              f"{_pct(loop_time, total_time):6.2f}%  ({loop_time:10.6f} s)")
        print(f"  2. Unaccounted / overhead          : "
              f"{_pct(unaccounted, total_time):6.2f}%  ({unaccounted:10.6f} s)")
        print("────────────────────────────────────────────────────────────")
        print(f"  TOTAL                              : "
              f"{_pct(total_time, total_time):6.2f}%  ({total_time:10.6f} s)")
        print("────────────────────────────────────────────────────────────")
        print()

        return F
