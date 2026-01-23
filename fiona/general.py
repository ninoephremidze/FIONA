####################################################################
# fiona/general.py
####################################################################

### This code computes the Fresnel integral in the x-domain.
### In the x-domain formulation, the lens potential ψ(x) is evaluated once on
### a fixed spatial quadrature grid and reused across frequencies. The NUFFT
### kernel depends on w through exp(-i w y·x), so we must execute one NUFFT per
### frequency. To keep throughput high, each NUFFT is forced to use a single
### core, and frequencies are distributed across cores in parallel.

import os
import time
import inspect
import multiprocessing as mp
import numpy as np
import numexpr as ne

from .utils import (
    gauss_legendre_2d,
    gauss_legendre_polar_2d,
    gauss_legendre_polar_uniform_theta_2d,  # keep utils name; see polar branch
    CPUTracker,
)

try:
    from finufft import nufft2d3
    _FINUFFT = True
    try:
        _NUFFT_HAS_NTHREADS = "nthreads" in inspect.signature(nufft2d3).parameters
    except Exception:
        _NUFFT_HAS_NTHREADS = False
except Exception:
    _FINUFFT = False
    _NUFFT_HAS_NTHREADS = False

_TWO_PI = 2.0 * np.pi

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "FINUFFT_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _set_thread_env(nthreads):
    if nthreads is None:
        return
    n = str(int(nthreads))
    for var in _THREAD_ENV_VARS:
        os.environ[var] = n
    try:
        import finufft
        finufft.set_num_threads(int(nthreads))
    except Exception:
        pass


def _nufft2d3_call(xj, yj, cj, sk, tk, eps, isign, nthreads):
    if _NUFFT_HAS_NTHREADS and nthreads is not None:
        return nufft2d3(xj, yj, cj, sk, tk, isign=isign, eps=eps, nthreads=int(nthreads))
    return nufft2d3(xj, yj, cj, sk, tk, isign=isign, eps=eps)


_WORKER_CTX = None


def _init_worker_ctx(ctx=None):
    global _WORKER_CTX
    if ctx is not None:
        _WORKER_CTX = ctx
    if _WORKER_CTX is not None:
        _set_thread_env(_WORKER_CTX.get("nufft_nthreads"))


def _eval_frequency(w, ctx):
    base = ctx["base"]
    quad = ctx["quad"]
    W = ctx["W"]
    tile_size = ctx.get("nufft_tile_size")
    n_sources = base.size

    h = ctx["h"]
    sk = (w * ctx["y1"]) / h
    tk = (w * ctx["y2"]) / h

    if tile_size is None or n_sources <= tile_size:
        if ctx["use_tail_correction"]:
            phase_base = w * base
            phase_quad = w * quad
            exp_base = np.exp(1j * phase_base)
            exp_quad = np.exp(1j * phase_quad)
            cj = (exp_base - exp_quad) * W
        else:
            phase = w * base
            cj = np.exp(1j * phase) * W

        I = _nufft2d3_call(
            ctx["xj"],
            ctx["yj"],
            cj,
            sk,
            tk,
            eps=ctx["nufft_tol"],
            isign=-1,
            nthreads=ctx["nufft_nthreads"],
        )
    else:
        I = np.zeros_like(sk, dtype=np.complex128)
        for start in range(0, n_sources, tile_size):
            stop = min(n_sources, start + tile_size)
            sl = slice(start, stop)
            if ctx["use_tail_correction"]:
                phase_base = w * base[sl]
                phase_quad = w * quad[sl]
                exp_base = np.exp(1j * phase_base)
                exp_quad = np.exp(1j * phase_quad)
                cj = (exp_base - exp_quad) * W[sl]
            else:
                phase = w * base[sl]
                cj = np.exp(1j * phase) * W[sl]

            I += _nufft2d3_call(
                ctx["xj"][sl],
                ctx["yj"][sl],
                cj,
                sk,
                tk,
                eps=ctx["nufft_tol"],
                isign=-1,
                nthreads=ctx["nufft_nthreads"],
            )

    Fw = np.exp(1j * w * ctx["quad_phase"]) * I * (w / (1j * _TWO_PI))
    if ctx["use_tail_correction"]:
        Fw = 1.0 + Fw
    return Fw


def _worker_task(task):
    idx, w = task
    return idx, _eval_frequency(w, _WORKER_CTX)


def _integrand_coeffs(u1, u2, w, lens, W):
    """
    I(y) = ∫ d^2x e^{-i w x·y} exp{i w [x^2/2 - ψ(x)]}.
    """
    phase = w * ((u1 * u1 + u2 * u2) / 2.0 - lens.psi_xy(u1, u2))
    return np.exp(1j * phase) * W


def _fmt_s(t):
    return f"{t:8.4f}s"


class FresnelNUFFT3:
    """
    Fresnel integral evaluator using FINUFFT (type-3 2D NUFFT) on a quadrature grid.

    Key ideas
    ---------
    - x-domain integration: evaluate ψ(x) once on a fixed spatial grid.
    - The oscillatory kernel depends on w, so each frequency needs its own NUFFT.
    - Frequencies are parallelized across processes; each NUFFT uses one thread.

    Parameters (public API)
    -----------------------
    lens : object
        Must implement lens.psi_xy(u1, u2) with numpy broadcasting.

    gl_nodes_per_dim : int
        Gauss-Legendre nodes per dimension (Cartesian). Total nodes ~ gl_nodes_per_dim^2.

    min_physical_radius : float
        Physical radius R (or floor on R if auto_R_from_gl_nodes=True).

    nufft_tol : float
        FINUFFT accuracy tolerance (passed as eps=...).

    batch_frequencies, chunk_frequencies, frequency_binning, frequency_bin_width : legacy
        Retained for API compatibility with the older u-domain implementation.
        These options are ignored in x-domain mode.

    auto_R_from_gl_nodes : bool
        If True, choose R from the maximum |w| in the call using:
            R_adapt = sqrt(gl_nodes_per_dim / (2 * w_max))
            R = max(min_physical_radius, R_adapt)
        If False, use R = min_physical_radius.

    use_tail_correction : bool
        If True, use the "(exp(-iwψ) - 1)" formulation and add +1 afterward.

    coordinate_system : {"cartesian", "polar"}
        Quadrature coordinate system.

    polar_radial_nodes, polar_angular_nodes : int
        Required if coordinate_system="polar".

    uniform_angular_sampling : bool
        If True, use gauss_legendre_polar_uniform_theta_2d; else gauss_legendre_polar_2d.

    numexpr_nthreads : int or None
        Thread count for numexpr only (not FINUFFT). Will be capped by NUMEXPR_MAX_THREADS and cores.

    parallel_frequencies : bool
        If True, distribute frequencies across processes (one NUFFT per process).

    nufft_workers : int or None
        Maximum number of worker processes to use. Defaults to os.cpu_count().

    nufft_nthreads : int
        Thread count per NUFFT (passed to FINUFFT when supported, and enforced via env vars).

    nufft_tile_max_points : int or None
        If not None and the total number of quadrature nodes (n_gl^2) exceeds
        this value, split the NUFFT into tiles with at most this many sources.
        Default 4000**2 corresponds to tiling when n_gl > 4000.

    verbose : bool
        Print configuration and timing diagnostics.
    """

    def __init__(
        self,
        lens,
        gl_nodes_per_dim=128,
        min_physical_radius=1.0,
        nufft_tol=1e-12,
        batch_frequencies=True,
        chunk_frequencies=True,
        frequency_binning="log",     # "log" or "linear"
        frequency_bin_width=0.5,     # decades if log, |w| units if linear
        auto_R_from_gl_nodes=True,
        use_tail_correction=True,
        coordinate_system="cartesian",  # "cartesian" or "polar"
        polar_radial_nodes=None,
        polar_angular_nodes=None,
        uniform_angular_sampling=True,
        numexpr_nthreads=None,
        parallel_frequencies=True,
        nufft_workers=None,
        nufft_nthreads=1,
        nufft_tile_max_points=4000**2,
        verbose=True,
    ):
        if not _FINUFFT:
            raise ImportError(
                "finufft is required for FresnelNUFFT3; install finufft or use FresnelDirect3."
            )

        self.lens = lens

        # Core numeric controls
        self.gl_nodes_per_dim = int(gl_nodes_per_dim)
        self.min_physical_radius = float(min_physical_radius)
        self.nufft_tol = float(nufft_tol)

        # Execution strategy
        self.batch_frequencies = bool(batch_frequencies)
        self.chunk_frequencies = bool(chunk_frequencies)

        if frequency_binning not in ("log", "linear"):
            raise ValueError("frequency_binning must be 'log' or 'linear'.")
        self.frequency_binning = frequency_binning

        self.frequency_bin_width = None if frequency_bin_width is None else float(frequency_bin_width)

        self.auto_R_from_gl_nodes = bool(auto_R_from_gl_nodes)

        # Model/derivation options
        self.use_tail_correction = bool(use_tail_correction)

        # Coordinates
        if coordinate_system not in ("cartesian", "polar"):
            raise ValueError("coordinate_system must be 'cartesian' or 'polar'.")
        self.coordinate_system = coordinate_system
        self.uniform_angular_sampling = bool(uniform_angular_sampling)

        if self.coordinate_system == "polar":
            if polar_radial_nodes is None or polar_angular_nodes is None:
                raise ValueError(
                    "When coordinate_system='polar', both polar_radial_nodes and "
                    "polar_angular_nodes must be specified."
                )
            self.polar_radial_nodes = int(polar_radial_nodes)
            self.polar_angular_nodes = int(polar_angular_nodes)
        else:
            # cartesian
            if self.gl_nodes_per_dim < 2:
                raise ValueError("gl_nodes_per_dim must be >= 2.")

        # Verbosity / reporting
        self.verbose = bool(verbose)

        # NumExpr threading (only affects numexpr evaluations, not FINUFFT)
        self._numexpr_nthreads = None
        if numexpr_nthreads is not None:
            requested = int(numexpr_nthreads)

            max_env = os.environ.get("NUMEXPR_MAX_THREADS")
            if max_env is not None:
                try:
                    max_env_int = int(max_env)
                except ValueError:
                    max_env_int = requested
            else:
                max_env_int = requested

            max_cores = ne.detect_number_of_cores()
            effective = min(requested, max_env_int, max_cores)
            effective = max(effective, 1)

            ne.set_num_threads(effective)
            self._numexpr_nthreads = effective

            if self.verbose:
                print(
                    f"[numexpr] using {effective} threads "
                    f"(requested={requested}, MAX={max_env_int}, cores={max_cores})"
                )
        else:
            self._numexpr_nthreads = ne.get_num_threads()
            if self.verbose:
                print(f"[numexpr] using default thread count: {self._numexpr_nthreads}")

        # Parallel NUFFT controls
        self.parallel_frequencies = bool(parallel_frequencies)

        if nufft_workers is None:
            self.nufft_workers = None
        else:
            nufft_workers = int(nufft_workers)
            if nufft_workers < 1:
                raise ValueError("nufft_workers must be >= 1.")
            self.nufft_workers = nufft_workers

        if nufft_nthreads is None:
            nufft_nthreads = 1
        nufft_nthreads = int(nufft_nthreads)
        if nufft_nthreads < 1:
            raise ValueError("nufft_nthreads must be >= 1.")
        self.nufft_nthreads = nufft_nthreads
        _set_thread_env(self.nufft_nthreads)

        if nufft_tile_max_points is None:
            self.nufft_tile_max_points = None
        else:
            nufft_tile_max_points = int(nufft_tile_max_points)
            if nufft_tile_max_points < 1:
                raise ValueError("nufft_tile_max_points must be >= 1.")
            self.nufft_tile_max_points = nufft_tile_max_points

    def __call__(self, w, y1, y2, verbose=None):
        """
        Evaluate F(w, y) for frequencies w and target coordinates (y1, y2).

        Parameters
        ----------
        w : array_like
            Frequencies (nonzero). Can be scalar or vector.
        y1, y2 : array_like
            Target coordinates; must have the same shape.
        verbose : bool or None
            If None, uses self.verbose.
        """
        if verbose is None:
            verbose = self.verbose

        t_total0 = time.perf_counter()
        t_input0 = time.perf_counter()

        w_vec = np.asarray(w, dtype=float).ravel()
        if np.any(w_vec == 0.0):
            raise ValueError("All w must be nonzero.")

        y1 = np.asarray(y1, dtype=float)
        y2 = np.asarray(y2, dtype=float)
        if y1.shape != y2.shape:
            raise ValueError("y1 and y2 must have the same shape.")

        quad_phase = (y1**2 + y2**2) / 2.0
        t_input = time.perf_counter() - t_input0

        t_choose0 = time.perf_counter()
        if self.auto_R_from_gl_nodes:
            w_use = float(np.max(np.abs(w_vec)))
            R_adapt = np.sqrt(self.gl_nodes_per_dim / (2.0 * w_use))
            R = max(self.min_physical_radius, float(R_adapt))
        else:
            R_adapt = None
            R = self.min_physical_radius

        if R <= 0.0:
            raise ValueError("Integration radius R must be positive.")
        h = np.pi / R
        t_choose = time.perf_counter() - t_choose0

        t_quad0 = time.perf_counter()
        if self.coordinate_system == "cartesian":
            x1, x2, W = gauss_legendre_2d(self.gl_nodes_per_dim, R)
        else:
            if self.uniform_angular_sampling:
                r, theta, W = gauss_legendre_polar_uniform_theta_2d(
                    self.polar_radial_nodes,
                    self.polar_angular_nodes,
                    R,
                )
            else:
                r, theta, W = gauss_legendre_polar_2d(
                    self.polar_radial_nodes,
                    self.polar_angular_nodes,
                    R,
                )
            x1 = r * np.cos(theta)
            x2 = r * np.sin(theta)
        t_quad = time.perf_counter() - t_quad0

        t_pot0 = time.perf_counter()
        psi = self.lens.psi_xy(x1, x2)
        quad = 0.5 * (x1 * x1 + x2 * x2)
        base = quad - psi
        t_pot = time.perf_counter() - t_pot0

        t_scale0 = time.perf_counter()
        xj = h * x1
        yj = h * x2
        t_scale = time.perf_counter() - t_scale0

        n_sources = x1.size
        tile_size = None
        tile_count = 1
        if self.nufft_tile_max_points is not None and n_sources > self.nufft_tile_max_points:
            tile_size = self.nufft_tile_max_points
            tile_count = (n_sources + tile_size - 1) // tile_size

        t_alloc0 = time.perf_counter()
        F_out = np.empty((len(w_vec),) + y1.shape, dtype=np.complex128)
        t_alloc = time.perf_counter() - t_alloc0

        ctx = {
            "xj": xj,
            "yj": yj,
            "W": W,
            "base": base,
            "quad": quad,
            "y1": y1,
            "y2": y2,
            "quad_phase": quad_phase,
            "h": h,
            "nufft_tol": self.nufft_tol,
            "use_tail_correction": self.use_tail_correction,
            "nufft_nthreads": self.nufft_nthreads,
            "nufft_tile_size": tile_size,
        }

        n_w = len(w_vec)
        max_workers = os.cpu_count() or 1
        if self.nufft_workers is None:
            n_workers = min(n_w, max_workers)
        else:
            n_workers = min(n_w, self.nufft_workers, max_workers)

        if not self.parallel_frequencies or n_w <= 1:
            n_workers = 1

        t_nufft0 = time.perf_counter()
        if n_workers > 1:
            mp_ctx = None
            try:
                mp_ctx = mp.get_context("fork")
            except ValueError:
                mp_ctx = mp.get_context()

            start_method = mp_ctx.get_start_method()
            if start_method == "fork":
                _init_worker_ctx(ctx)
                pool = mp_ctx.Pool(processes=n_workers, initializer=_init_worker_ctx)
            else:
                pool = mp_ctx.Pool(processes=n_workers, initializer=_init_worker_ctx, initargs=(ctx,))

            try:
                for idx, Fw in pool.imap_unordered(_worker_task, enumerate(w_vec), chunksize=1):
                    F_out[idx] = Fw
            finally:
                pool.close()
                pool.join()
        else:
            _init_worker_ctx(ctx)
            for i, w_i in enumerate(w_vec):
                F_out[i] = _eval_frequency(w_i, ctx)

        t_nufft = time.perf_counter() - t_nufft0
        t_total = time.perf_counter() - t_total0

        if verbose:
            if self.auto_R_from_gl_nodes:
                floored = (R > float(R_adapt) + 1e-15)
                cap_msg = " (floored by min_physical_radius)" if floored else ""
                r_msg = f"R={R:.6g}{cap_msg} (R_adapt={R_adapt:.6g})"
            else:
                r_msg = f"R={R:.6g} (fixed)"

            if tile_size is None:
                tile_msg = "  tiling:        disabled"
            else:
                tile_msg = (
                    f"  tiling:        {tile_count} tiles "
                    f"(max_points={tile_size}, sources={n_sources})"
                )

            print(
                "[FresnelNUFFT3] x-domain path\n"
                f"  input prep:    {_fmt_s(t_input)}\n"
                f"  choose R/h:    {_fmt_s(t_choose)} | {r_msg}, h={h:.6g}\n"
                f"  quadrature:    {_fmt_s(t_quad)}\n"
                f"  lens + base:   {_fmt_s(t_pot)}\n"
                f"  scaling:       {_fmt_s(t_scale)}\n"
                f"  alloc out:     {_fmt_s(t_alloc)}\n"
                f"  NUFFT total:   {_fmt_s(t_nufft)}\n"
                f"{tile_msg}\n"
                f"  workers:       {n_workers} (nufft_nthreads={self.nufft_nthreads})\n"
                f"  wall total:    {_fmt_s(t_total)}"
            )

        return F_out
