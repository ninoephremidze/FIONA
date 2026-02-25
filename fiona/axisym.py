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

    Receives a chunk of frequencies, builds Fortran-contiguous coefficient
    matrices of shape ``(m, batch)``, then calls ``pn.nufht_batch`` twice
    (once for the real coefficients, once for the imaginary coefficients).

    Parameters
    ----------
    args : tuple
        (lens, rs, u_weights, y_vec, w_chunk, nu, tol)

    Returns
    -------
    G_re, G_im : ndarray of shape (n_y, batch)
    """
    import _pynufht as pn
    import numpy as np

    lens, rs, u_weights, y_vec, w_chunk, nu, tol = args

    m = rs.size
    batch = len(w_chunk)

    # Build Fortran-contiguous coefficient matrices (m, batch)
    cs_re = np.empty((m, batch), dtype=float, order='F')
    cs_im = np.empty((m, batch), dtype=float, order='F')

    for j, w in enumerate(w_chunk):
        u_over_w = rs / w
        psi_vals = lens.psi_r(u_over_w)
        phase = (rs * rs) / (2.0 * w) - w * psi_vals
        fw = np.exp(1j * phase)
        ck = u_weights * fw
        cs_re[:, j] = ck.real
        cs_im[:, j] = ck.imag

    # Two batched NUFHT calls; each returns shape (n_y, batch)
    G_re = pn.nufht_batch(nu, rs, cs_re, y_vec, tol=tol)
    G_im = pn.nufht_batch(nu, rs, cs_im, y_vec, tol=tol)

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
    and **FastHankelTransform NUFHT**.

    GL nodes are loaded from precomputed files if available, or computed
    on-the-fly and saved to disk (via gauss_legendre_1d from utils.py).

    We discard x<0 (symmetry) and use:
        rs = x[x>0]
        u_weights = rs * w[x>0]

    Parameters
    ----------
    lens : AxisymmetricLens
        Axisymmetric lens object with psi_r method.
    gl_nodes_per_dim : int
        Number of Gauss-Legendre nodes per dimension.
    min_physical_radius : float
        Minimum physical radius (Umax) to use.
    auto_R_from_gl_nodes : bool
        If True, adapt Umax based on frequency range.
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

        # Store gl_dir for later use
        if gl_dir is None:
            gl_dir = os.environ.get("FIONA_GL2D_DIR", "")
            if not gl_dir:
                raise RuntimeError(
                    "gl_dir not provided and FIONA_GL2D_DIR environment variable not set."
                )
        self._gl_dir = gl_dir

        # Don't load nodes yet - will be loaded in __call__ based on frequency range

    def _load_gl_nodes(self, Umax):
        """
        Load or compute GL nodes for given Umax.
        Uses gauss_legendre_1d from utils, which computes on-the-fly if needed.
        """
        # Set FIONA_GL2D_DIR temporarily if needed
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            
            x, w = gauss_legendre_1d(self.gl_nodes_per_dim, Umax)
            
            # Keep only positive u (symmetry of Gauss–Legendre)
            mask = x > 0
            rs = x[mask]          # u_k
            du = w[mask]          # Δu_k
            u_weights = rs * du   # u_k Δu_k for ∫ u f(u) du

            return rs.astype(float), u_weights.astype(float)
        finally:
            # Restore old environment variable
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

        # Determine Umax based on frequency range
        if self.auto_R_from_gl_nodes:
            w_use = float(np.max(np.abs(w_vec)))
            if w_use <= 0:
                raise ValueError("All w must be nonzero for auto_R_from_gl_nodes.")
            Umax_adapt = np.sqrt(self.gl_nodes_per_dim / (2.0 * w_use))
            Umax = max(self.min_physical_radius, float(Umax_adapt))
        else:
            Umax = self.min_physical_radius

        # Load GL nodes for this Umax
        rs, u_weights = self._load_gl_nodes(Umax)

        nu = 0
        tol = self.tol

        n_w = len(w_vec)
        n_y = len(y_vec)

        # ---- Setup / allocations (Python) ----
        setup_start = time.perf_counter()
        F = np.empty((n_w, n_y), dtype=np.complex128)
        quad_phase = 0.5 * y_vec**2
        setup_end = time.perf_counter()

        with CPUTracker() as tracker:

            # ---------- Step 1: Build c_re and c_im batches (Python) ----------
            s1_alloc_start = time.perf_counter()
            c_re = np.empty((n_w, rs.size), float)
            c_im = np.empty((n_w, rs.size), float)
            s1_alloc_end = time.perf_counter()

            coeff_loop_start = time.perf_counter()
            scale_uw_time = 0.0
            psi_time = 0.0
            phase_time = 0.0
            exp_time = 0.0
            mul_time = 0.0
            assign_time = 0.0

            for i, w in enumerate(w_vec):
                # scale u/w
                t_a = time.perf_counter()
                u_over_w = rs / w
                t_b = time.perf_counter()

                # lens potential ψ(u/w)
                psi_vals = self.lens.psi_r(u_over_w)
                t_c = time.perf_counter()

                # phase = u^2/(2w) - w ψ(u/w)
                phase = (rs**2)/(2.0*w) - w*psi_vals
                t_d = time.perf_counter()

                # exp(i * phase)
                fw = np.exp(1j*phase)
                t_e = time.perf_counter()

                # coefficients c_k(w) = u_k Δu_k f_w(u_k)
                ck = u_weights * fw
                t_f = time.perf_counter()

                # write into batches
                c_re[i,:] = ck.real
                c_im[i,:] = ck.imag
                t_g = time.perf_counter()

                # accumulate sub-timings
                scale_uw_time += (t_b - t_a)
                psi_time      += (t_c - t_b)
                phase_time    += (t_d - t_c)
                exp_time      += (t_e - t_d)
                mul_time      += (t_f - t_e)
                assign_time   += (t_g - t_f)

            coeff_loop_end = time.perf_counter()
            step1_end = coeff_loop_end

            # ---------- Step 2: C NUFHT calls ----------
            # 2a. Allocate output arrays
            nufht_alloc_start = time.perf_counter()
            g_re = np.empty((n_w, n_y), dtype=float)
            g_im = np.empty((n_w, n_y), dtype=float)
            nufht_alloc_end = time.perf_counter()

            # 2b. NUFHT calls (loop over frequencies)
            nufht_call_start = time.perf_counter()
            for i in range(n_w):
                # Call C NUFHT for real part
                g_re[i, :] = _c_nufht(nu, rs, c_re[i, :], y_vec)
                # Call C NUFHT for imaginary part
                g_im[i, :] = _c_nufht(nu, rs, c_im[i, :], y_vec)
            nufht_call_end = time.perf_counter()
            step2_end = nufht_call_end

            # ---------- Step 3: Assemble full Fresnel integral (Python) ----------
            step3_loop_start = time.perf_counter()
            for i, w in enumerate(w_vec):
                g = g_re[i] + 1j*g_im[i]
                F[i,:] = np.exp(1j*w*quad_phase) * (g/(1j*w))
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

        coeff_sub_total = (scale_uw_time + psi_time + phase_time +
                           exp_time + mul_time + assign_time)
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
        print(" FresnelHankelAxisymmetric (GL_precomputed + C NUFHT)")
        print("────────────────────────────────────────────────────────────")
        print(tracker.report("  CPU usage summary"))
        print()

        # ---- Step 0 ----
        print("  Step 0: Setup / allocations")
        print("  ───────────────────────────")
        print(f"    0a. F, quad_phase                : "
              f"{_pct(step0_time, total_time):6.2f}%  ({step0_time:10.6f} s)")
        print()

        # ---- Step 1 ----
        print("  Step 1: Coefficient Computation (Python)")
        print("  ───────────────────────────────────────")
        print(f"    1a. allocate c_re/c_im           : "
              f"{_pct(alloc_time, step1_total):6.2f}%  ({alloc_time:10.6f} s)")
        print(f"    1b. coefficient loop (total)     : "
              f"{_pct(coeff_loop_time, step1_total):6.2f}%  ({coeff_loop_time:10.6f} s)")
        print(f"        ├─ scale u/w (all w)         : "
              f"{_pct(scale_uw_time, step1_total):6.2f}%  ({scale_uw_time:10.6f} s)")
        print(f"        ├─ lens potential ψ(u/w)     : "
              f"{_pct(psi_time, step1_total):6.2f}%  ({psi_time:10.6f} s)")
        print(f"        ├─ phase calculation         : "
              f"{_pct(phase_time, step1_total):6.2f}%  ({phase_time:10.6f} s)")
        print(f"        ├─ exp(i·phase)              : "
              f"{_pct(exp_time, step1_total):6.2f}%  ({exp_time:10.6f} s)")
        print(f"        ├─ multiply by u_k Δu_k      : "
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
        print(f"    3a. apply quad_phase & 1/(i w)   : "
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
        Minimum physical radius ``Umax`` to use (default 1.0).
    auto_R_from_gl_nodes : bool, optional
        If ``True`` (default), adapt ``Umax`` from the frequency range.
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
                 n_workers: int = None):

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

        if gl_dir is None:
            gl_dir = os.environ.get("FIONA_GL2D_DIR", "")
            if not gl_dir:
                raise RuntimeError(
                    "gl_dir not provided and FIONA_GL2D_DIR environment variable "
                    "not set."
                )
        self._gl_dir = gl_dir

    def _load_gl_nodes(self, Umax, n_gl=None):
        """Load or compute GL nodes for given *Umax* and *n_gl*."""
        if n_gl is None:
            n_gl = self.gl_nodes_per_dim
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            x, w = gauss_legendre_1d(n_gl, Umax)
            mask = x > 0
            rs = x[mask].astype(float)
            du = w[mask].astype(float)
            u_weights = rs * du
            return rs, u_weights
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

            # Determine Umax for this group
            if self.auto_R_from_gl_nodes:
                w_use = float(np.max(np.abs(w_sub)))
                Umax_adapt = np.sqrt(n_gl / (2.0 * w_use))
                Umax = max(self.min_physical_radius, float(Umax_adapt))
            else:
                Umax = self.min_physical_radius

            rs, u_weights = self._load_gl_nodes(Umax, n_gl=n_gl)

            # Split this group's frequencies into chunks for parallel workers
            n_sub = len(idxs)
            n_chunks = min(self.n_workers, n_sub)
            if n_chunks < 1:
                n_chunks = 1
            chunk_splits = np.array_split(np.arange(n_sub), n_chunks)

            worker_args = [
                (self.lens, rs, u_weights, y_vec, w_sub[ch], nu, self.tol)
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
                    F[i, :] = np.exp(1j * w * quad_phase) * g / (1j * w)

        return F


class FresnelHankelAxisymmetricTrapezoidal:
    r"""
    Fresnel integral for axisymmetric lenses using a fast Hankel transform.

        F(w, y) = e^{i w y^2 / 2} / (i w) ∫_0^∞ u du
                  exp{i w [ u^2 / (2 w^2) - ψ(u / w) ]} J_0(u y),

    We discretize the radial u-integral on [0, Umax] and evaluate the Bessel sum 
    with C-based NUFHT (nonuniform fast Hankel transform). 

        ∫_0^Umax u f_w(u) J_0(u y) du ≈ ∑_k c_k(w) J_0(y r_k),

    where r_k are radial nodes, and

        c_k(w) = u_k Δu_k * exp{i w [ u_k^2 / (2 w^2) - ψ(u_k / w) ]}.

    nufht(ν, r_k, c_k, y_j) then returns the vector of Hankel sums
    g_j ≈ ∑_k c_k J_ν(y_j r_k).  We have to perform a zeroth-order FHT (ν=0).

    Parameters
    ----------
    lens : AxisymmetricLens
        Axisymmetric lens object with psi_r method.
    n_r : int
        Number of radial grid points.
    min_physical_radius : float
        Minimum physical radius (Umax) to use.
    auto_R_from_gl_nodes : bool
        If True, adapt Umax based on frequency range.
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

        # Don't build grid yet - will be built in __call__ based on frequency range

    # ------------------------------------------------------------------
    # Radial grid and quadrature weights
    # ------------------------------------------------------------------
    def _build_radial_grid(self, Umax):
        """
        Simple uniform radial grid on (0, Umax] with trapezoidal weights.

        We avoid r=0 to keep things well-behaved numerically; the missing
        interval [0, r_min] is negligible for sufficiently large n_r.
        """
        n = self.n_r

        # n+1 points from 0 to Umax, then drop the first (0).
        rs_full = np.linspace(0.0, Umax, n + 1, dtype=float)
        rs = rs_full[1:]               # shape (n,)
        dr = rs_full[1] - rs_full[0]

        # Trapezoidal weights for ∫_0^{Umax} … du.
        w = np.ones_like(rs) * dr
        w[0] *= 0.5
        w[-1] *= 0.5

        # For ∫ u f(u) du, combine with the extra factor u:
        u_weights = rs * w            # u_k Δu_k

        return rs, u_weights

    def __call__(self, w_vec, y_vec):
        """
        Evaluate F(w, y) for arrays of frequencies w and radii y.
        """
        w_vec = np.asarray(w_vec, dtype=float).ravel()
        y_vec = np.asarray(y_vec, dtype=float).ravel()

        if np.any(w_vec == 0.0):
            raise ValueError("All w must be nonzero.")

        # Determine Umax based on frequency range
        if self.auto_R_from_gl_nodes:
            w_use = float(np.max(np.abs(w_vec)))
            if w_use <= 0:
                raise ValueError("All w must be nonzero for auto_R_from_gl_nodes.")
            # Use n_r as the node count for the adaptation formula
            Umax_adapt = np.sqrt(self.n_r / (2.0 * w_use))
            Umax = max(self.min_physical_radius, float(Umax_adapt))
        else:
            Umax = self.min_physical_radius

        # Build radial grid for this Umax
        rs, u_weights = self._build_radial_grid(Umax)

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
            s1_alloc_end = time.perf_counter()

            # 1b. main coefficient loop, with internal breakdown
            coeff_loop_start = time.perf_counter()
            scale_uw_time = 0.0
            psi_time = 0.0
            phase_time = 0.0
            exp_time = 0.0
            mul_time = 0.0

            for w in w_vec:
                # scale u/w
                t_a = time.perf_counter()
                u_over_w = rs / w
                t_b = time.perf_counter()

                # lens potential ψ(u/w)
                psi_vals = self.lens.psi_r(u_over_w)
                t_c = time.perf_counter()

                # phase(u) = u^2/(2w) - w ψ(u/w)
                phase = (rs * rs) / (2.0 * w) - w * psi_vals
                t_d = time.perf_counter()

                # exp(i * phase)
                f_w = np.exp(1j * phase)
                t_e = time.perf_counter()

                # coefficients c_k(w) = u_k Δu_k f_w(u_k)
                c_k = u_weights * f_w
                t_f = time.perf_counter()

                all_c_re.append(c_k.real.astype(float))
                all_c_im.append(c_k.imag.astype(float))

                # accumulate sub-timings
                scale_uw_time += (t_b - t_a)
                psi_time += (t_c - t_b)
                phase_time += (t_d - t_c)
                exp_time += (t_e - t_d)
                mul_time += (t_f - t_e)

            coeff_loop_end = time.perf_counter()

            # 1c. stack into batched arrays
            stack_start = time.perf_counter()
            c_re_batch = np.stack(all_c_re, axis=0)
            c_im_batch = np.stack(all_c_im, axis=0)
            stack_end = time.perf_counter()

            t1 = stack_end

            # ---------------- Step 2: C NUFHT ----------------
            # 2a. Allocate output arrays
            nufht_alloc_start = time.perf_counter()
            g_re = np.empty((n_w, n_y), dtype=float)
            g_im = np.empty((n_w, n_y), dtype=float)
            nufht_alloc_end = time.perf_counter()

            # 2b. NUFHT calls (loop over frequencies)
            nufht_call_start = time.perf_counter()
            for i in range(n_w):
                # Call C NUFHT for real part
                g_re[i, :] = _c_nufht(nu, rs, c_re_batch[i, :], y_vec)
                # Call C NUFHT for imaginary part
                g_im[i, :] = _c_nufht(nu, rs, c_im_batch[i, :], y_vec)
            nufht_call_end = time.perf_counter()

            t2 = nufht_call_end

            # ---------------- Step 3: Final assembly in Python ----------------
            final_loop_start = time.perf_counter()
            for i, w in enumerate(w_vec):
                g = g_re[i, :] + 1j * g_im[i, :]
                F[i, :] = np.exp(1j * w * quad_phase) * g / (1j * w)
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
        stack_time = stack_end - stack_start

        coeff_sub_total = scale_uw_time + psi_time + phase_time + exp_time + mul_time
        coeff_unaccounted = coeff_loop_time - coeff_sub_total
        step1_unaccounted = step1_total - (alloc_lists_time + coeff_loop_time + stack_time)

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
        print(" FresnelHankelAxisymmetric (Trapezoidal + C NUFHT)")
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
        print(f"        ├─ scale u/w (all w)         : "
              f"{_pct(scale_uw_time, step1_total):6.2f}%  ({scale_uw_time:10.6f} s)")
        print(f"        ├─ lens potential ψ(u/w)     : "
              f"{_pct(psi_time, step1_total):6.2f}%  ({psi_time:10.6f} s)")
        print(f"        ├─ phase calculation         : "
              f"{_pct(phase_time, step1_total):6.2f}%  ({phase_time:10.6f} s)")
        print(f"        ├─ exp(i·phase)              : "
              f"{_pct(exp_time, step1_total):6.2f}%  ({exp_time:10.6f} s)")
        print(f"        ├─ multiply by u_k Δu_k      : "
              f"{_pct(mul_time, step1_total):6.2f}%  ({mul_time:10.6f} s)")
        print(f"        └─ unaccounted (loop)        : "
              f"{_pct(coeff_unaccounted, step1_total):6.2f}%  ({coeff_unaccounted:10.6f} s)")
        print(f"    1c. stack to c_re/c_im batch     : "
              f"{_pct(stack_time, step1_total):6.2f}%  ({stack_time:10.6f} s)")
        print(f"    1d. other (Step 1)               : "
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
        print(f"    3a. apply phase & 1/(i w)        : "
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
    for configuration (gl_nodes_per_dim, Umax), and performing the Hankel
    transform with SciPy's FFTLog-based fast Hankel transform (scipy.fft.fht).

    GL nodes are loaded from precomputed files if available, or computed
    on-the-fly and saved to disk (via gauss_legendre_1d from utils.py).

    Parameters
    ----------
    lens : AxisymmetricLens
        Axisymmetric lens object with psi_r method.
    gl_nodes_per_dim : int
        Number of Gauss-Legendre nodes per dimension.
    min_physical_radius : float
        Minimum physical radius (Umax) to use.
    auto_R_from_gl_nodes : bool
        If True, adapt Umax based on frequency range.
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

        # Store gl_dir for later use
        if gl_dir is None:
            gl_dir = os.environ.get("FIONA_GL2D_DIR", "")
            if not gl_dir:
                raise RuntimeError(
                    "gl_dir not provided and FIONA_GL2D_DIR environment variable not set."
                )
        self._gl_dir = gl_dir

        # Don't load nodes yet - will be loaded in __call__ based on frequency range

    def _load_and_setup_grid(self, Umax):
        """
        Load or compute GL nodes for given Umax and setup FFTLog grid.
        Uses gauss_legendre_1d from utils, which computes on-the-fly if needed.
        """
        # Set FIONA_GL2D_DIR temporarily if needed
        old_env = os.environ.get("FIONA_GL2D_DIR")
        try:
            os.environ["FIONA_GL2D_DIR"] = self._gl_dir
            
            x, w = gauss_legendre_1d(self.gl_nodes_per_dim, Umax)
            
            mask = x > 0
            rs_gl = x[mask]      # original GL radial nodes u_k

            # We use the GL nodes to define the radial range and sample count
            r_min = float(rs_gl.min())
            r_max = float(rs_gl.max())
            n_r = rs_gl.size

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

        Uses SciPy's fast Hankel transform (FFTLog). For each w, we compute,

            g_w(y) = ∫_0^∞ u f_w(u) J_0(u y) du,

        by mapping to SciPy's definition

            A(k) = ∫_0^∞ a(r) J_0(k r) k dr,

        with a(r) = r f_w(r), so that A(k) ≈ k * g_w(k), hence

            g_w(k) ≈ A(k)/k,

        and then interpolate in log-space onto the requested y values.
        """
        w_vec = np.asarray(w_vec, dtype=float).ravel()
        y_vec = np.asarray(y_vec, dtype=float).ravel()

        if np.any(w_vec == 0.0):
            raise ValueError("All w must be nonzero.")
        if np.any(y_vec <= 0.0):
            raise ValueError("FresnelHankelAxisymmetricSciPy currently requires y > 0.")

        # Determine Umax based on frequency range
        if self.auto_R_from_gl_nodes:
            w_use = float(np.max(np.abs(w_vec)))
            if w_use <= 0:
                raise ValueError("All w must be nonzero for auto_R_from_gl_nodes.")
            Umax_adapt = np.sqrt(self.gl_nodes_per_dim / (2.0 * w_use))
            Umax = max(self.min_physical_radius, float(Umax_adapt))
        else:
            Umax = self.min_physical_radius

        # Load GL nodes and setup grid for this Umax
        r, dln, mu, bias, offset, k = self._load_and_setup_grid(Umax)

        n_w = len(w_vec)
        n_y = len(y_vec)

        F = np.empty((n_w, n_y), dtype=np.complex128)
        quad_phase = 0.5 * y_vec**2

        # ───────────────────────────────
        # Wall-clock timing
        # ───────────────────────────────
        t0 = time.perf_counter()

        with CPUTracker() as tracker:
            # ---------------- Step 1: Build a_re, a_im ----------------
            t1a = time.perf_counter()
            a_re = np.empty((n_w, r.size), dtype=float)
            a_im = np.empty_like(a_re)
            t1b = time.perf_counter()

            coeff_loop_start = time.perf_counter()
            for i, w in enumerate(w_vec):
                # phase = r^2/(2w) - w ψ(r/w)
                phase = (r * r) / (2.0 * w) - w * self.lens.psi_r(r / w)
                f_w = np.exp(1j * phase)
                a_r = r * f_w  # a(r) = r f_w(r) for SciPy's integral definition

                a_re[i, :] = a_r.real
                a_im[i, :] = a_r.imag
            coeff_loop_end = time.perf_counter()
            t1 = coeff_loop_end

            # ---------------- Step 2: SciPy fht calls ----------------
            fht_start = time.perf_counter()
            A_re = _scipy_fht(a_re, dln, mu=mu, offset=offset)
            A_im = _scipy_fht(a_im, dln, mu=mu, offset=offset)
            fht_end = time.perf_counter()
            t2 = fht_end

            A = A_re + 1j * A_im  # shape (n_w, n_k)

            # ---------------- Step 3: Sorting + interp + final assembly ----------------
            t3a = time.perf_counter()
            # Sorting k for monotonic interpolation in log k
            idx = np.argsort(k)
            k_sorted = k[idx]
            logk_sorted = np.log(k_sorted)
            t3b = time.perf_counter()

            interp_start = time.perf_counter()
            logy = np.log(y_vec)

            for i, w in enumerate(w_vec):
                Ak = A[i, idx]

                # Convert from SciPy's "k dr" normalization to the usual "r dr"
                # Hankel integral by dividing by k.
                Ak_over_k = Ak / k_sorted

                # Interpolate Ak_over_k(k) onto the requested y (using log-space).
                real_part = np.interp(logy, logk_sorted, Ak_over_k.real)
                imag_part = np.interp(logy, logk_sorted, Ak_over_k.imag)
                g_y = real_part + 1j * imag_part

                F[i, :] = np.exp(1j * w * quad_phase) * g_y / (1j * w)

            interp_end = time.perf_counter()
            t3 = interp_end

        t4 = time.perf_counter()

        # ───────────────────────────────
        # Timing breakdown
        # ───────────────────────────────
        total_time = t4 - t0

        step1_total = t1 - t0           # allocations + coefficient loop
        alloc_time = t1b - t1a          # a_re / a_im allocations
        coeff_loop_time = coeff_loop_end - coeff_loop_start

        step2_total = t2 - t1           # both fht calls
        fht_time = fht_end - fht_start  # SciPy fht calls (should equal step2_total)

        step3_total = t3 - t2           # sort + logk + interpolation + final assembly
        sort_time = t3b - t3a
        interp_time = interp_end - interp_start
        other_step3 = step3_total - sort_time - interp_time

        unaccounted = total_time - (step1_total + step2_total + step3_total)

        # ───────────────────────────────
        # Pretty printing
        # ───────────────────────────────
        print()
        print("────────────────────────────────────────────────────────────")
        print(" FresnelHankelAxisymmetricSciPy Timing")
        print("────────────────────────────────────────────────────────────")
        print(tracker.report("  CPU usage summary"))
        print()

        print("  Step 1: Coefficient build (Python + NumPy)")
        print("  ───────────────────────────────────────────")
        print(f"    1a. Alloc a_re/a_im              : "
              f"{_pct(alloc_time, step1_total):6.2f}%  ({alloc_time:10.6f} s)")
        print(f"    1b. Coefficient loop (all w)     : "
              f"{_pct(coeff_loop_time, step1_total):6.2f}%  ({coeff_loop_time:10.6f} s)")
        print()
        print(f"    Step 1 total                     : "
              f"{_pct(step1_total, total_time):6.2f}%  ({step1_total:10.6f} s)")
        print()

        print("  Step 2: SciPy fast Hankel transform (FFTLog)")
        print("  ────────────────────────────────────────────")
        print(f"    2a. fht(a_re) + fht(a_im)        : "
              f"{_pct(fht_time, total_time):6.2f}%  ({fht_time:10.6f} s)")
        print()
        print(f"    Step 2 total                     : "
              f"{_pct(step2_total, total_time):6.2f}%  ({step2_total:10.6f} s)")
        print()

        print("  Step 3: Post-processing and interpolation")
        print("  ──────────────────────────────────────────")
        print(f"    3a. sort k, logk                 : "
              f"{_pct(sort_time, step3_total):6.2f}%  ({sort_time:10.6f} s)")
        print(f"    3b. log-space interp + assembly  : "
              f"{_pct(interp_time, step3_total):6.2f}%  ({interp_time:10.6f} s)")
        print(f"    3c. other (step 3)               : "
              f"{_pct(other_step3, step3_total):6.2f}%  ({other_step3:10.6f} s)")
        print()
        print(f"    Step 3 total                     : "
              f"{_pct(step3_total, total_time):6.2f}%  ({step3_total:10.6f} s)")
        print()

        print("Overall Timing Summary (percent of TOTAL)")
        print("────────────────────────────────────────────────────────────")
        print(f"  1. Coefficients (Step 1)           : "
              f"{_pct(step1_total, total_time):6.2f}%  ({step1_total:10.6f} s)")
        print(f"  2. SciPy FHT (Step 2)              : "
              f"{_pct(step2_total, total_time):6.2f}%  ({step2_total:10.6f} s)")
        print(f"  3. Interp + assembly (Step 3)      : "
              f"{_pct(step3_total, total_time):6.2f}%  ({step3_total:10.6f} s)")
        print(f"  4. Unaccounted / overhead          : "
              f"{_pct(unaccounted, total_time):6.2f}%  ({unaccounted:10.6f} s)")
        print("────────────────────────────────────────────────────────────")
        print(f"  TOTAL                              : "
              f"{_pct(total_time, total_time):6.2f}%  ({total_time:10.6f} s)")
        print("────────────────────────────────────────────────────────────")
        print()

        return F
