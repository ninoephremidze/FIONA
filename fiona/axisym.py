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
import multiprocessing as mp

# Optional numexpr for vectorised phase-exponential computation (Optimization 8).
_HAS_NUMEXPR = False
try:
    import numexpr as _ne
    _ne.evaluate  # verify the symbol exists
    _HAS_NUMEXPR = True
except Exception:
    pass

# Thread-pinning support for worker processes (Optimization 6).
_AXISYM_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _axisym_set_thread_env(nthreads):
    """Pin OpenMP / OpenBLAS / MKL thread counts inside a worker process."""
    n = str(int(nthreads))
    for var in _AXISYM_THREAD_ENV_VARS:
        os.environ[var] = n


# Module-level worker context (Optimization 2 & 7).
# Set once in the parent process and inherited by forked children;
# passed explicitly via initargs when using the "spawn" start method.
_AXISYM_WORKER_CTX = None


def _init_axisym_worker(ctx=None):
    """
    Initialise per-process state for axisym worker processes.

    On fork-based multiprocessing the context is inherited from the parent;
    this function is called with no argument just to pin thread counts.
    On spawn-based systems ``ctx`` is passed explicitly as an initializer
    argument.
    """
    global _AXISYM_WORKER_CTX
    if ctx is not None:
        _AXISYM_WORKER_CTX = ctx
    _axisym_set_thread_env(1)
    if _HAS_NUMEXPR:
        try:
            _ne.set_num_threads(1)
        except Exception:
            pass


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


def _nufht_batch_compute(lens, rs_unit, w_unit, y_vec, w_chunk, nu, tol,
                          min_physical_radius, n_gl, window_potential,
                          window_radius_fraction, window_u, window_u_width,
                          use_tail_correction, xs_fixed=None,
                          x_weights_fixed=None, integrand_phase_fixed=None):
    """
    Core NUFHT computation shared by ``_nufht_batch_worker`` and
    ``_axisym_batch_task``.

    In fixed mode (``n_gl == 0``) the caller must supply ``xs_fixed``,
    ``x_weights_fixed``, and ``integrand_phase_fixed``.  In adaptive mode
    (``n_gl > 0``) those keyword arguments are ignored.

    Implements:
    - Optimization 1: group fixed-mode frequencies by quantised |w| and call
      ``nufht_batch`` once per bin when available.
    - Optimization 3: vectorised coefficient computation via broadcasting.
    - Optimization 8: numexpr for the phase-exponential evaluation.

    Returns
    -------
    G_re, G_im : ndarray of shape (n_y, batch)
    """
    import _pynufht as pn

    n_y = y_vec.size
    batch = len(w_chunk)
    G_re = np.empty((n_y, batch), dtype=float)
    G_im = np.empty((n_y, batch), dtype=float)

    if n_gl == 0:  # ── Fixed mode: xs is the same for every frequency ────────
        xs = xs_fixed
        x_weights = x_weights_fixed
        integrand_phase = integrand_phase_fixed

        # Optimization 3: vectorised coefficient computation
        phase_matrix = np.outer(w_chunk, integrand_phase)  # (batch, m)
        if _HAS_NUMEXPR:
            cos_p = _ne.evaluate("cos(phase_matrix)")
            sin_p = _ne.evaluate("sin(phase_matrix)")
            fw_matrix = cos_p + 1j * sin_p
        else:
            fw_matrix = np.exp(1j * phase_matrix)

        if use_tail_correction:
            free_phase = np.outer(w_chunk, 0.5 * xs ** 2)
            if _HAS_NUMEXPR:
                cos_f = _ne.evaluate("cos(free_phase)")
                sin_f = _ne.evaluate("sin(free_phase)")
                fw_matrix -= cos_f + 1j * sin_f
            else:
                fw_matrix -= np.exp(1j * free_phase)

        ck_matrix = x_weights[np.newaxis, :] * fw_matrix  # (batch, m)

        # Optimization 1: group by quantised |w| → call nufht_batch per bin
        w_abs = np.abs(w_chunk)
        w_abs_q = np.round(w_abs, 6)
        unique_abs_w = np.unique(w_abs_q)

        if _HAS_NUFHT_BATCH and len(unique_abs_w) < batch:
            for abs_w in unique_abs_w:
                bin_idx = np.where(w_abs_q == abs_w)[0]
                out_pts = abs_w * y_vec
                if len(bin_idx) > 1:
                    ck_bin_re = np.asfortranarray(ck_matrix[bin_idx, :].real.T)
                    ck_bin_im = np.asfortranarray(ck_matrix[bin_idx, :].imag.T)
                    G_re[:, bin_idx] = pn.nufht_batch(
                        nu, xs, ck_bin_re, out_pts, tol=tol)
                    G_im[:, bin_idx] = pn.nufht_batch(
                        nu, xs, ck_bin_im, out_pts, tol=tol)
                else:
                    j = bin_idx[0]
                    G_re[:, j] = pn.nufht(
                        nu, xs, np.ascontiguousarray(ck_matrix[j].real),
                        out_pts, tol=tol)
                    G_im[:, j] = pn.nufht(
                        nu, xs, np.ascontiguousarray(ck_matrix[j].imag),
                        out_pts, tol=tol)
        else:
            for j in range(batch):
                out_pts = w_abs[j] * y_vec
                G_re[:, j] = pn.nufht(
                    nu, xs, np.ascontiguousarray(ck_matrix[j].real),
                    out_pts, tol=tol)
                G_im[:, j] = pn.nufht(
                    nu, xs, np.ascontiguousarray(ck_matrix[j].imag),
                    out_pts, tol=tol)

    else:  # ── Adaptive mode: xs differs per frequency ────────────────────────
        for j, w in enumerate(w_chunk):
            Xmax_w = max(min_physical_radius, np.sqrt(n_gl / (2.0 * abs(w))))
            xs = rs_unit * Xmax_w
            x_weights = xs * (w_unit * Xmax_w)
            if window_u:
                x_weights = x_weights * _window_u_taper_1d(
                    xs, Xmax_w, window_u_width)
            psi_vals = lens.psi_r(xs)
            if window_potential:
                psi_vals = psi_vals * _window_taper_1d(
                    xs, Xmax_w, window_radius_fraction)
            integrand_phase = 0.5 * xs ** 2 - psi_vals

            # Optimization 8: numexpr for per-frequency phase exponential
            phase_w = w * integrand_phase
            if _HAS_NUMEXPR:
                cos_p = _ne.evaluate("cos(phase_w)")
                sin_p = _ne.evaluate("sin(phase_w)")
                fw = cos_p + 1j * sin_p
            else:
                fw = np.exp(1j * phase_w)

            if use_tail_correction:
                free_phase_w = w * 0.5 * xs ** 2
                if _HAS_NUMEXPR:
                    cos_f = _ne.evaluate("cos(free_phase_w)")
                    sin_f = _ne.evaluate("sin(free_phase_w)")
                    fw -= cos_f + 1j * sin_f
                else:
                    fw -= np.exp(1j * free_phase_w)

            ck = x_weights * fw
            out_pts = abs(w) * y_vec
            G_re[:, j] = pn.nufht(nu, xs, ck.real.copy(), out_pts, tol=tol)
            G_im[:, j] = pn.nufht(nu, xs, ck.imag.copy(), out_pts, tol=tol)

    return G_re, G_im


def _nufht_batch_worker(args):
    """
    Top-level worker function (must be module-level for picklability).

    Kept for backward compatibility; delegates to ``_nufht_batch_compute``.
    In fixed mode (``n_gl == 0``) the x-nodes and integrand phase are
    precomputed here before handing off to the shared core.

    Parameters
    ----------
    args : tuple
        (lens, rs_unit, w_unit, y_vec, w_chunk, nu, tol, min_physical_radius, n_gl,
        window_potential, window_radius_fraction, window_u, window_u_width,
        use_tail_correction)

    Returns
    -------
    G_re, G_im : ndarray of shape (n_y, batch)
    """
    (lens, rs_unit, w_unit, y_vec, w_chunk, nu, tol, min_physical_radius, n_gl,
     window_potential, window_radius_fraction, window_u, window_u_width,
     use_tail_correction) = args

    xs_fixed = x_weights_fixed = integrand_phase_fixed = None
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
        integrand_phase_fixed = 0.5 * xs_fixed ** 2 - psi_fixed

    return _nufht_batch_compute(
        lens, rs_unit, w_unit, y_vec, w_chunk, nu, tol,
        min_physical_radius, n_gl, window_potential, window_radius_fraction,
        window_u, window_u_width, use_tail_correction,
        xs_fixed=xs_fixed, x_weights_fixed=x_weights_fixed,
        integrand_phase_fixed=integrand_phase_fixed)


def _axisym_batch_task(w_indices):
    """
    Context-aware worker entry-point for fork / spawn Pool dispatch.

    Reads all shared data from the module-level ``_AXISYM_WORKER_CTX``
    (set by the parent before forking, or passed via the pool initializer
    for spawn).  Only ``w_indices`` — indices into the context's ``w_vec``
    — are transferred per task, minimising pickle payload (Optimization 7).

    Returns
    -------
    w_indices, G_re, G_im
        ``w_indices`` echoed back so the caller can assemble results in any
        order (compatible with ``imap_unordered``).
    """
    ctx = _AXISYM_WORKER_CTX
    w_chunk = ctx["w_vec"][w_indices]
    G_re, G_im = _nufht_batch_compute(
        ctx["lens"], ctx["rs_unit"], ctx["w_unit"], ctx["y_vec"],
        w_chunk, ctx["nu"], ctx["tol"], ctx["min_physical_radius"],
        ctx["n_gl"], ctx["window_potential"], ctx["window_radius_fraction"],
        ctx["window_u"], ctx["window_u_width"], ctx["use_tail_correction"],
        xs_fixed=ctx.get("xs_fixed"),
        x_weights_fixed=ctx.get("x_weights_fixed"),
        integrand_phase_fixed=ctx.get("integrand_phase_fixed"))
    return w_indices, G_re, G_im


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
    gl_nodes_per_dim : int, optional
        Number of Gauss-Legendre nodes per dimension (default 128).
    min_physical_radius : float, optional
        Minimum physical radius (Xmax) to use (default 1.0).
    auto_R_from_gl_nodes : bool, optional
        If True (default), adapt Xmax based on frequency range.
        If False, use fixed min_physical_radius.
    gl_dir : str or None, optional
        Directory for GL node files. If None, uses FIONA_GL2D_DIR env var.
    tol : float, optional
        Tolerance for NUFHT (default 1e-12).
    window_potential : bool, optional
        If True (default), apply a smooth taper to the lens potential near Xmax.
    window_radius_fraction : float, optional
        Fractional radius at which the potential window starts (default 0.75).
    window_u : bool, optional
        If True (default), apply a smooth taper to the quadrature weights near Xmax.
    window_u_width : float, optional
        Fractional width of the weight taper (default 0.02).
    use_tail_correction : bool, optional
        If True (default), subtract the free-space tail from the integrand.
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
                 use_tail_correction: bool = True):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError("FresnelHankelAxisymmetric requires AxisymmetricLens")

        if not _HAS_FHT:
            raise ImportError("NUFHT C extension (_pynufht) cannot be loaded: "
                              f"{_FHT_ERR!r}")

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

        # Optimization 4: in-memory GL-node cache.
        # _gl_cache: n_gl -> (rs_unit, w_unit) for unit-interval nodes.
        # _gl_nodes_cache: (n_gl, Xmax) -> (xs, x_weights) for scaled nodes.
        self._gl_cache = {}
        self._gl_nodes_cache = {}

    def _load_gl_nodes(self, Xmax):
        """
        Load or compute GL nodes for given Xmax (x-domain integration radius).
        Uses gauss_legendre_1d from utils, which computes on-the-fly if needed.
        Results are cached in ``_gl_nodes_cache`` (Optimization 4).
        """
        key = (self.gl_nodes_per_dim, Xmax)
        if key in self._gl_nodes_cache:
            return self._gl_nodes_cache[key]
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            x, w = gauss_legendre_1d(self.gl_nodes_per_dim, Xmax)
            mask = x > 0
            xs = x[mask].astype(float)
            x_weights = xs * w[mask].astype(float)
            result = (xs, x_weights)
            self._gl_nodes_cache[key] = result
            return result
        finally:
            if old_env is None:
                os.environ.pop("FIONA_GL2D_DIR", None)
            else:
                os.environ["FIONA_GL2D_DIR"] = old_env

    def _load_gl_nodes_unit(self):
        """Load or compute GL nodes on the unit interval [-1, 1].
        Results are cached in ``_gl_cache`` (Optimization 4).
        """
        n_gl = self.gl_nodes_per_dim
        if n_gl in self._gl_cache:
            return self._gl_cache[n_gl]
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            x, w = gauss_legendre_1d(n_gl, 1.0)
            mask = x > 0
            rs_unit = x[mask].astype(float)
            w_unit = w[mask].astype(float)
            self._gl_cache[n_gl] = (rs_unit, w_unit)
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

            if not self.auto_R_from_gl_nodes:
                # Optimization 3: vectorised coefficient computation for fixed mode.
                xs_per_freq = [xs_fixed] * n_w
                phase_matrix = np.outer(w_vec, integrand_phase_fixed)  # (n_w, m)
                if _HAS_NUMEXPR:
                    cos_p = _ne.evaluate("cos(phase_matrix)")
                    sin_p = _ne.evaluate("sin(phase_matrix)")
                    fw_matrix = cos_p + 1j * sin_p
                else:
                    fw_matrix = np.exp(1j * phase_matrix)
                if self.use_tail_correction:
                    free_phase = np.outer(w_vec, 0.5 * xs_fixed ** 2)
                    if _HAS_NUMEXPR:
                        cos_f = _ne.evaluate("cos(free_phase)")
                        sin_f = _ne.evaluate("sin(free_phase)")
                        fw_matrix -= cos_f + 1j * sin_f
                    else:
                        fw_matrix -= np.exp(1j * free_phase)
                ck_matrix = x_weights_fixed[np.newaxis, :] * fw_matrix  # (n_w, m)
                c_re[:] = ck_matrix.real
                c_im[:] = ck_matrix.imag
            else:
                for i, w in enumerate(w_vec):
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

                    # Optimization 8: numexpr for phase exponential
                    phase_w = w * integrand_phase
                    if _HAS_NUMEXPR:
                        cos_p = _ne.evaluate("cos(phase_w)")
                        sin_p = _ne.evaluate("sin(phase_w)")
                        fw = cos_p + 1j * sin_p
                    else:
                        fw = np.exp(1j * phase_w)
                    if self.use_tail_correction:
                        free_phase_w = w * 0.5 * xs ** 2
                        if _HAS_NUMEXPR:
                            cos_f = _ne.evaluate("cos(free_phase_w)")
                            sin_f = _ne.evaluate("sin(free_phase_w)")
                            fw -= cos_f + 1j * sin_f
                        else:
                            fw -= np.exp(1j * free_phase_w)
                    t_d = time.perf_counter()

                    # coefficients c_k(w) = x_k Δx_k f_w(x_k)
                    ck = x_weights * fw
                    t_e = time.perf_counter()

                    # write into batches
                    c_re[i, :] = ck.real
                    c_im[i, :] = ck.imag
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

            # ---------- Step 3: Assemble full Fresnel integral ----------
            # Optimization 5: vectorised final assembly.
            # F = e^{iwy²/2} · (w/i) · g   [x-domain prefactor]
            step3_loop_start = time.perf_counter()
            g_all = g_re + 1j * g_im                            # (n_w, n_y)
            w_col = w_vec[:, np.newaxis]                         # (n_w, 1)
            if _HAS_NUMEXPR:
                phase_arg = w_col * quad_phase[np.newaxis, :]    # (n_w, n_y)
                cos_p = _ne.evaluate("cos(phase_arg)")
                sin_p = _ne.evaluate("sin(phase_arg)")
                phase = cos_p + 1j * sin_p
            else:
                phase = np.exp(1j * w_col * quad_phase[np.newaxis, :])
            F[:] = phase * (w_col / 1j) * g_all
            if self.use_tail_correction:
                F += 1.0
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
    print_timing : bool, optional
        If ``True`` (default), print a timing/performance summary after each
        ``__call__`` invocation, including total runtime and a breakdown of
        major phases (setup, GL node loading, worker dispatch, result
        assembly).  Set to ``False`` to silence all timing output.
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
                 use_tail_correction: bool = True,
                 print_timing: bool = True):

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
        self.print_timing = bool(print_timing)

        if gl_dir is None:
            gl_dir = os.environ.get("FIONA_GL2D_DIR", "")
            if not gl_dir:
                raise RuntimeError(
                    "gl_dir not provided and FIONA_GL2D_DIR environment variable "
                    "not set."
                )
        self._gl_dir = gl_dir

        # Optimization 4: in-memory GL-node cache.
        self._gl_cache = {}        # n_gl -> (rs_unit, w_unit)
        self._gl_nodes_cache = {}  # (n_gl, Xmax) -> (xs, x_weights)

    def _load_gl_nodes(self, Xmax, n_gl=None):
        """Load or compute GL nodes for given *Xmax* and *n_gl*.
        Results are cached in ``_gl_nodes_cache`` (Optimization 4).
        """
        if n_gl is None:
            n_gl = self.gl_nodes_per_dim
        key = (n_gl, Xmax)
        if key in self._gl_nodes_cache:
            return self._gl_nodes_cache[key]
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            x, w = gauss_legendre_1d(n_gl, Xmax)
            mask = x > 0
            xs = x[mask].astype(float)
            x_weights = xs * w[mask].astype(float)
            result = (xs, x_weights)
            self._gl_nodes_cache[key] = result
            return result
        finally:
            if old_env is None:
                os.environ.pop("FIONA_GL2D_DIR", None)
            else:
                os.environ["FIONA_GL2D_DIR"] = old_env

    def _load_gl_nodes_unit(self, n_gl=None):
        """Load or compute GL nodes on the unit interval [-1, 1] for *n_gl*.
        Results are cached in ``_gl_cache`` (Optimization 4).
        """
        if n_gl is None:
            n_gl = self.gl_nodes_per_dim
        if n_gl in self._gl_cache:
            return self._gl_cache[n_gl]
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            x, w = gauss_legendre_1d(n_gl, 1.0)
            mask = x > 0
            rs_unit = x[mask].astype(float)
            w_unit = w[mask].astype(float)
            self._gl_cache[n_gl] = (rs_unit, w_unit)
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

        # Declare module-level context variable for worker sharing (Optimization 2).
        global _AXISYM_WORKER_CTX

        n_w = len(w_vec)
        n_y = len(y_vec)
        nu = 0

        # ───────────────────────────────
        # Wall-clock timing
        # ───────────────────────────────
        t0 = time.perf_counter()

        with CPUTracker() as tracker:

            # ---------- Step 0: Setup ----------
            setup_start = time.perf_counter()
            F = np.empty((n_w, n_y), dtype=np.complex128)
            quad_phase = 0.5 * y_vec ** 2

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
            setup_end = time.perf_counter()

            # Accumulators for per-group phases
            gl_load_time = 0.0
            dispatch_time = 0.0
            assembly_time = 0.0

            # Determine multiprocessing start method once (Optimization 2).
            try:
                _mp_fork_ctx = mp.get_context("fork")
                _axisym_mp_method = "fork"
            except Exception:
                _mp_fork_ctx = mp.get_context()
                _axisym_mp_method = "spawn"

            for n_gl, idxs in group_items:
                w_sub = w_vec[idxs]
                n_sub = len(idxs)

                # Load GL nodes on the unit interval [-1, 1] once per group.
                gl_start = time.perf_counter()
                rs_unit, w_unit = self._load_gl_nodes_unit(n_gl)
                gl_end = time.perf_counter()
                gl_load_time += gl_end - gl_start

                # n_gl_for_worker > 0 enables per-frequency Xmax adaptation;
                # 0 signals fixed mode (Xmax = min_physical_radius for all w).
                n_gl_for_worker = n_gl if self.auto_R_from_gl_nodes else 0

                # Build the shared context for this group (Optimizations 2 & 7).
                ctx = {
                    "lens": self.lens,
                    "rs_unit": rs_unit,
                    "w_unit": w_unit,
                    "y_vec": y_vec,
                    "w_vec": w_sub,
                    "nu": nu,
                    "tol": self.tol,
                    "min_physical_radius": self.min_physical_radius,
                    "n_gl": n_gl_for_worker,
                    "window_potential": self.window_potential,
                    "window_radius_fraction": self.window_radius_fraction,
                    "window_u": self.window_u,
                    "window_u_width": self.window_u_width,
                    "use_tail_correction": self.use_tail_correction,
                }
                # Optimization 7: precompute fixed-mode quantities in the parent
                # so workers need not call lens.psi_r at all.
                if n_gl_for_worker == 0:
                    Xmax = self.min_physical_radius
                    xs_f = rs_unit * Xmax
                    xw_f = xs_f * (w_unit * Xmax)
                    if self.window_u:
                        xw_f = xw_f * _window_u_taper_1d(
                            xs_f, Xmax, self.window_u_width)
                    psi_f = self.lens.psi_r(xs_f)
                    if self.window_potential:
                        psi_f = psi_f * _window_taper_1d(
                            xs_f, Xmax, self.window_radius_fraction)
                    ctx["xs_fixed"] = xs_f
                    ctx["x_weights_fixed"] = xw_f
                    ctx["integrand_phase_fixed"] = 0.5 * xs_f ** 2 - psi_f

                # Split this group's frequencies into chunks for parallel workers
                n_chunks = max(1, min(self.n_workers, n_sub))
                chunk_splits = np.array_split(np.arange(n_sub), n_chunks)
                tasks = [ch for ch in chunk_splits if len(ch) > 0]

                # Optimization 2: fork-based Pool replaces ProcessPoolExecutor.
                dispatch_start = time.perf_counter()
                if self.n_workers == 1:
                    _AXISYM_WORKER_CTX = ctx
                    _init_axisym_worker()
                    results_raw = [_axisym_batch_task(ch) for ch in tasks]
                else:
                    if _axisym_mp_method == "fork":
                        _AXISYM_WORKER_CTX = ctx
                        _init_axisym_worker()
                        pool = _mp_fork_ctx.Pool(
                            processes=self.n_workers,
                            initializer=_init_axisym_worker)
                    else:
                        pool = _mp_fork_ctx.Pool(
                            processes=self.n_workers,
                            initializer=_init_axisym_worker,
                            initargs=(ctx,))
                    try:
                        results_raw = list(pool.imap_unordered(
                            _axisym_batch_task, tasks, chunksize=1))
                    finally:
                        pool.close()
                        pool.join()
                dispatch_end = time.perf_counter()
                dispatch_time += dispatch_end - dispatch_start

                # Optimization 5: vectorised result assembly.
                assembly_start = time.perf_counter()
                for w_indices, G_re, G_im in results_raw:
                    global_indices = idxs[w_indices]
                    w_chunk = w_sub[w_indices]
                    w_col = w_chunk[:, np.newaxis]          # (batch, 1)
                    g_T = (G_re + 1j * G_im).T             # (batch, n_y)
                    if _HAS_NUMEXPR:
                        phase_arg = w_col * quad_phase[np.newaxis, :]
                        cos_p = _ne.evaluate("cos(phase_arg)")
                        sin_p = _ne.evaluate("sin(phase_arg)")
                        phase = cos_p + 1j * sin_p
                    else:
                        phase = np.exp(1j * w_col * quad_phase[np.newaxis, :])
                    chunk_F = phase * (w_col / 1j) * g_T   # (batch, n_y)
                    if self.use_tail_correction:
                        chunk_F += 1.0
                    F[global_indices, :] = chunk_F
                assembly_end = time.perf_counter()
                assembly_time += assembly_end - assembly_start

        t_end = time.perf_counter()

        # ───────────────────────────────
        # Timing breakdown
        # ───────────────────────────────
        if self.print_timing:
            total_time = t_end - t0
            setup_time = setup_end - setup_start
            n_groups = len(group_items)
            accounted = setup_time + gl_load_time + dispatch_time + assembly_time
            unaccounted = total_time - accounted

            print()
            print("────────────────────────────────────────────────────────────")
            print(" FresnelNUFHTBatched Timing")
            print("────────────────────────────────────────────────────────────")
            print(tracker.report("  CPU usage summary"))
            print()

            print(f"  Configuration: {n_w} frequencies, {n_y} output radii, "
                  f"{n_groups} GL group(s), {self.n_workers} worker(s)")
            print()

            print("  Phase breakdown")
            print("  ───────────────")
            print(f"    0. Setup (grouping, allocation)  : "
                  f"{_pct(setup_time, total_time):6.2f}%  ({setup_time:10.6f} s)")
            print(f"    1. GL node loading               : "
                  f"{_pct(gl_load_time, total_time):6.2f}%  ({gl_load_time:10.6f} s)")
            print(f"    2. Worker dispatch + execution   : "
                  f"{_pct(dispatch_time, total_time):6.2f}%  ({dispatch_time:10.6f} s)")
            print(f"    3. Result assembly               : "
                  f"{_pct(assembly_time, total_time):6.2f}%  ({assembly_time:10.6f} s)")
            print(f"    4. Unaccounted / overhead        : "
                  f"{_pct(unaccounted, total_time):6.2f}%  ({unaccounted:10.6f} s)")
            print("────────────────────────────────────────────────────────────")
            print(f"  TOTAL                              : "
                  f"{_pct(total_time, total_time):6.2f}%  ({total_time:10.6f} s)")
            print("────────────────────────────────────────────────────────────")
            print()

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
    n_r : int, optional
        Number of radial grid points (default 1024).
    min_physical_radius : float, optional
        Minimum physical radius (Xmax) to use (default 50.0).
    auto_R_from_gl_nodes : bool, optional
        If True, adapt Xmax based on frequency range.
        If False (default), use fixed min_physical_radius.
        Note: For this class we use n_r instead of gl_nodes_per_dim in the formula.
    tol : float, optional
        Tolerance for NUFHT (default 1e-12).
    window_potential : bool, optional
        If True (default), apply a smooth taper to the lens potential near Xmax.
    window_radius_fraction : float, optional
        Fractional radius at which the potential window starts (default 0.75).
    window_u : bool, optional
        If True (default), apply a smooth taper to the quadrature weights near Xmax.
    window_u_width : float, optional
        Fractional width of the weight taper (default 0.02).
    use_tail_correction : bool, optional
        If True (default), subtract the free-space tail from the integrand.
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
                 use_tail_correction: bool = True):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError(
                "FresnelHankelAxisymmetric requires an AxisymmetricLens instance."
            )

        if not _HAS_FHT:
            raise ImportError(
                "NUFHT C extension (_pynufht) is not available.\n"
                f"Original import error: {_FHT_ERR!r}"
            )

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

            # ---------------- Step 3: Final assembly ----------
            # Optimization 5: vectorised final assembly.
            # F = e^{iwy²/2} · (w/i) · g   [x-domain prefactor]
            final_loop_start = time.perf_counter()
            g_all = g_re + 1j * g_im                            # (n_w, n_y)
            w_col = w_vec[:, np.newaxis]                         # (n_w, 1)
            if _HAS_NUMEXPR:
                phase_arg = w_col * quad_phase[np.newaxis, :]    # (n_w, n_y)
                cos_p = _ne.evaluate("cos(phase_arg)")
                sin_p = _ne.evaluate("sin(phase_arg)")
                phase = cos_p + 1j * sin_p
            else:
                phase = np.exp(1j * w_col * quad_phase[np.newaxis, :])
            F = phase * (w_col / 1j) * g_all
            if self.use_tail_correction:
                F += 1.0
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
    gl_nodes_per_dim : int, optional
        Number of Gauss-Legendre nodes per dimension (default 128).
    min_physical_radius : float, optional
        Minimum physical radius (Xmax) to use (default 1.0).
    auto_R_from_gl_nodes : bool, optional
        If True (default), adapt Xmax based on frequency range.
        If False, use fixed min_physical_radius.
    gl_dir : str or None, optional
        Directory for GL node files. If None, uses FIONA_GL2D_DIR env var.
    tol : float, optional
        Tolerance for computation (default 1e-12).
    window_potential : bool, optional
        If True (default), apply a smooth taper to the lens potential near Xmax.
    window_radius_fraction : float, optional
        Fractional radius at which the potential window starts (default 0.75).
    window_u : bool, optional
        If True (default), apply a smooth taper to the quadrature weights near Xmax.
    window_u_width : float, optional
        Fractional width of the weight taper (default 0.02).
    use_tail_correction : bool, optional
        If True (default), subtract the free-space tail from the integrand.
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
                 use_tail_correction: bool = True):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError("FresnelHankelAxisymmetricSciPy requires AxisymmetricLens")

        if not _HAS_SCIPY_FHT:
            raise ImportError(
                "scipy.fft.fht (fast Hankel transform) is not available.\n"
                f"Original import error: {_SCIPY_FHT_ERR!r}"
            )

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

        # Optimization 4: in-memory cache for _load_and_setup_grid results.
        self._gl_cache = {}  # (n_gl, Xmax) -> (r, dln, mu, bias, offset, k)

    def _load_and_setup_grid(self, Xmax):
        """
        Load or compute GL nodes for given Xmax and setup FFTLog grid.
        Results are cached in ``_gl_cache`` (Optimization 4).
        """
        key = (self.gl_nodes_per_dim, Xmax)
        if key in self._gl_cache:
            return self._gl_cache[key]
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            x, w = gauss_legendre_1d(self.gl_nodes_per_dim, Xmax)
            mask = x > 0
            xs_gl = x[mask]
            r_min = float(xs_gl.min())
            r_max = float(xs_gl.max())
            n_r = xs_gl.size
            r = np.geomspace(r_min, r_max, n_r)
            dln = float(np.log(r[1] / r[0]))
            mu = 0.0
            bias = 0.0
            offset = _scipy_fhtoffset(dln, mu=mu, bias=bias)
            k = np.exp(offset) / r[::-1]
            result = r, dln, mu, bias, offset, k
            self._gl_cache[key] = result
            return result
        finally:
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
