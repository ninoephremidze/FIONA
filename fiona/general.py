####################################################################
# fiona/general.py
####################################################################

### This code computes the Fresnel integral in the x-domain.
### In the x-domain formulation, the lens potential ψ(x) is evaluated on a
### spatial quadrature grid. When auto_R_from_gl_nodes=False, the grid is fixed
### and ψ(x) is reused across frequencies. When auto_R_from_gl_nodes=True, the
### grid adapts per w (so ψ(x) is recomputed per frequency). The NUFFT kernel
### depends on w through exp(-i w y·x), so we must execute one NUFFT per
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
    gauss_legendre_1d,
    gauss_legendre_polar_2d,
    gauss_legendre_polar_uniform_theta_2d,  # keep utils name; see polar branch
    CPUTracker,
)

def _has_nthreads(func):
    if func is None:
        return False
    try:
        return "nthreads" in inspect.signature(func).parameters
    except Exception:
        return False


try:
    import finufft as _finufft
    nufft2d3 = _finufft.nufft2d3
    nufft2d1 = getattr(_finufft, "nufft2d1", None)
    nufft2d1many = getattr(_finufft, "nufft2d1many", None)
    _FINUFFT = True
    _FINUFFT_TYPE1 = nufft2d1 is not None
    _FINUFFT_TYPE1MANY = nufft2d1many is not None
    _NUFFT3_HAS_NTHREADS = _has_nthreads(nufft2d3)
    _NUFFT1_HAS_NTHREADS = _has_nthreads(nufft2d1)
    _NUFFT1MANY_HAS_NTHREADS = _has_nthreads(nufft2d1many)
except Exception:
    _FINUFFT = False
    _FINUFFT_TYPE1 = False
    _FINUFFT_TYPE1MANY = False
    _NUFFT3_HAS_NTHREADS = False
    _NUFFT1_HAS_NTHREADS = False
    _NUFFT1MANY_HAS_NTHREADS = False

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
    if _NUFFT3_HAS_NTHREADS and nthreads is not None:
        return nufft2d3(xj, yj, cj, sk, tk, isign=isign, eps=eps, nthreads=int(nthreads))
    return nufft2d3(xj, yj, cj, sk, tk, isign=isign, eps=eps)


def _nufft2d1_call(xj, yj, cj, ms, mt, eps, iflag, nthreads):
    if nufft2d1 is None:
        raise RuntimeError("finufft.nufft2d1 is not available.")
    if _NUFFT1_HAS_NTHREADS and nthreads is not None:
        return nufft2d1(xj, yj, cj, ms, mt, isign=iflag, eps=eps, nthreads=int(nthreads))
    return nufft2d1(xj, yj, cj, ms, mt, isign=iflag, eps=eps)


def _nufft2d1many_call(xj, yj, cj, ms, mt, eps, iflag, nthreads):
    if nufft2d1many is None:
        raise RuntimeError("finufft.nufft2d1many is not available.")
    if _NUFFT1MANY_HAS_NTHREADS and nthreads is not None:
        return nufft2d1many(xj, yj, cj, ms, mt, isign=iflag, eps=eps, nthreads=int(nthreads))
    return nufft2d1many(xj, yj, cj, ms, mt, isign=iflag, eps=eps)


_WORKER_CTX = None


def _adaptive_n_gl(w_abs):
    if w_abs <= 0.0:
        raise ValueError("w must be nonzero for adaptive quadrature.")
    bin_idx = int(np.floor(w_abs / 10.0)) + 1
    bin_idx = max(1, min(10, bin_idx))
    return bin_idx * 1000


def _window_taper(r, R, frac):
    return 0.5 * (1.0 - np.tanh(r - frac * R))


def _window_u_taper(r, R, width_frac):
    if width_frac <= 0.0:
        return 1.0
    return 0.5 * (1.0 - np.tanh((r - R) / (width_frac * R)))


def _detect_uniform_grid(y1, y2, tol=1e-12):
    if y1.ndim != 2 or y2.ndim != 2:
        return None
    if y1.shape != y2.shape:
        return None
    y1_axis = y1[0, :]
    y2_axis = y2[:, 0]
    if not np.allclose(y1, y1_axis[None, :], atol=tol, rtol=0.0):
        return None
    if not np.allclose(y2, y2_axis[:, None], atol=tol, rtol=0.0):
        return None
    if y1_axis.size < 2 or y2_axis.size < 2:
        return None
    dy1 = np.diff(y1_axis)
    dy2 = np.diff(y2_axis)
    if not (np.allclose(dy1, dy1[0], atol=tol, rtol=0.0) and np.allclose(dy2, dy2[0], atol=tol, rtol=0.0)):
        return None
    return {
        "y1_axis": y1_axis,
        "y2_axis": y2_axis,
        "dy1": float(dy1[0]),
        "dy2": float(dy2[0]),
    }


def _interp_bilinear_grid(F, k1, k2, kmax):
    ny = F.shape[-2]
    nx = F.shape[-1]
    if ny != nx or kmax * 2 + 1 != ny:
        raise ValueError("Unexpected grid shape for interpolation.")
    if kmax == 0:
        out = np.zeros_like(k1, dtype=np.complex128)
        mask = (np.abs(k1) < 0.5) & (np.abs(k2) < 0.5)
        if np.any(mask):
            out[mask] = F[0, 0]
        return out
    k1f = k1 + kmax
    k2f = k2 + kmax
    i0 = np.floor(k1f).astype(int)
    j0 = np.floor(k2f).astype(int)
    i1 = i0 + 1
    j1 = j0 + 1

    w1 = k1f - i0
    w2 = k2f - j0

    valid = (i0 >= 0) & (j0 >= 0) & (i1 < ny) & (j1 < nx)
    out = np.zeros_like(k1f, dtype=np.complex128)
    if not np.any(valid):
        return out

    i0v = i0[valid]
    j0v = j0[valid]
    i1v = i1[valid]
    j1v = j1[valid]
    w1v = w1[valid]
    w2v = w2[valid]

    f00 = F[i0v, j0v]
    f01 = F[i0v, j1v]
    f10 = F[i1v, j0v]
    f11 = F[i1v, j1v]

    out[valid] = (
        (1.0 - w1v) * (1.0 - w2v) * f00
        + (1.0 - w1v) * w2v * f01
        + w1v * (1.0 - w2v) * f10
        + w1v * w2v * f11
    )
    return out


def _init_worker_ctx(ctx=None):
    global _WORKER_CTX
    if ctx is not None:
        _WORKER_CTX = ctx
    if _WORKER_CTX is not None:
        _set_thread_env(_WORKER_CTX.get("nufft_nthreads"))


def _eval_frequency(w, ctx):
    w_abs = abs(w)
    tile_from_1d = ctx.get("tile_from_1d", False)

    if ctx.get("adaptive"):
        n_gl = ctx["n_gl"]
        R = np.sqrt(n_gl / (2.0 * w_abs))
        h = np.pi / R
        tile_size = ctx.get("nufft_tile_size")
        if tile_from_1d:
            x_1d = ctx["x_1d"]
            w_1d = ctx["w_1d"]
            x_1d_unit = ctx.get("x_1d_unit", True)
            n_sources = n_gl * n_gl
        else:
            u1_base = ctx["u1_base"]
            u2_base = ctx["u2_base"]
            W_base = ctx["W_base"]
            x1 = u1_base * R
            x2 = u2_base * R
            W = W_base * (R * R)
            r = np.hypot(x1, x2)
            if ctx["window_u"]:
                W = W * _window_u_taper(r, R, ctx["window_u_width"])
            psi = ctx["lens"].psi_xy(x1, x2)
            if ctx["window_potential"]:
                psi = psi * _window_taper(r, R, ctx["window_radius_fraction"])
            quad = 0.5 * (x1 * x1 + x2 * x2)
            base = quad - psi
            xj = h * x1
            yj = h * x2
            n_sources = base.size
    else:
        R = ctx.get("R")
        h = ctx["h"]
        tile_size = ctx.get("nufft_tile_size")
        if tile_from_1d:
            x_1d = ctx["x_1d"]
            w_1d = ctx["w_1d"]
            x_1d_unit = ctx.get("x_1d_unit", False)
            n_gl = ctx["n_gl"]
            n_sources = n_gl * n_gl
        else:
            base = ctx["base"]
            quad = ctx["quad"]
            W = ctx["W"]
            xj = ctx["xj"]
            yj = ctx["yj"]
            n_sources = base.size

    sk = (w * ctx["y1"]) / h
    tk = (w * ctx["y2"]) / h

    if tile_size is None or n_sources <= tile_size:
        if tile_from_1d:
            idx = np.arange(n_sources)
            j = idx // n_gl
            k = idx - j * n_gl
            scale = R if x_1d_unit else 1.0
            x1 = x_1d[j] * scale
            x2 = x_1d[k] * scale
            W = w_1d[j] * w_1d[k]
            if x_1d_unit and R is not None:
                W = W * (R * R)
            r = np.hypot(x1, x2)
            if ctx["window_u"]:
                W = W * _window_u_taper(r, R, ctx["window_u_width"])
            psi = ctx["lens"].psi_xy(x1, x2)
            if ctx["window_potential"]:
                psi = psi * _window_taper(r, R, ctx["window_radius_fraction"])
            quad = 0.5 * (x1 * x1 + x2 * x2)
            base = quad - psi
            xj = h * x1
            yj = h * x2

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
            xj,
            yj,
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
            if tile_from_1d:
                idx = np.arange(start, stop)
                j = idx // n_gl
                k = idx - j * n_gl
                scale = R if x_1d_unit else 1.0
                x1 = x_1d[j] * scale
                x2 = x_1d[k] * scale
                W = w_1d[j] * w_1d[k]
                if x_1d_unit and R is not None:
                    W = W * (R * R)
                r = np.hypot(x1, x2)
                if ctx["window_u"]:
                    W = W * _window_u_taper(r, R, ctx["window_u_width"])
                psi = ctx["lens"].psi_xy(x1, x2)
                if ctx["window_potential"]:
                    psi = psi * _window_taper(r, R, ctx["window_radius_fraction"])
                quad = 0.5 * (x1 * x1 + x2 * x2)
                base = quad - psi
                xj = h * x1
                yj = h * x2
            else:
                sl = slice(start, stop)
                xj = xj[sl]
                yj = yj[sl]
                base = base[sl]
                quad = quad[sl]
                W = W[sl]

            if ctx["use_tail_correction"]:
                phase_base = w * base
                phase_quad = w * quad
                exp_base = np.exp(1j * phase_base)
                exp_quad = np.exp(1j * phase_quad)
                cj = (exp_base - exp_quad) * W
            else:
                phase = w * base
                cj = np.exp(1j * phase) * W

            I += _nufft2d3_call(
                xj,
                yj,
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
    - x-domain integration: evaluate ψ(x) on a spatial grid.
      (Fixed when auto_R_from_gl_nodes=False, adaptive per w otherwise.)
    - The oscillatory kernel depends on w, so each frequency needs its own NUFFT.
    - Frequencies are parallelized across processes; each NUFFT uses one thread.

    Parameters (public API)
    -----------------------
    lens : object
        Must implement lens.psi_xy(u1, u2) with numpy broadcasting.

    gl_nodes_per_dim : int
        Gauss-Legendre nodes per dimension (Cartesian) when auto_R_from_gl_nodes=False.
        Total nodes ~ gl_nodes_per_dim^2.

    min_physical_radius : float
        Physical radius R when auto_R_from_gl_nodes=False.

    nufft_tol : float
        FINUFFT accuracy tolerance (passed as eps=...).

    batch_frequencies, chunk_frequencies, frequency_binning, frequency_bin_width : legacy
        Retained for API compatibility with the older u-domain implementation.
        These options are ignored in x-domain mode.

    auto_R_from_gl_nodes : bool
        If True, use adaptive quadrature per frequency bin:
          - n_gl(w) = 1000 * clip(floor(|w|/10) + 1, 1, 10)
          - R(w) = sqrt(n_gl / (2 * |w|))
        Frequencies that share n_gl reuse the same *unit* Gauss–Legendre grid,
        which is scaled per w; ψ(x) is therefore recomputed per frequency.
        Currently implemented only for coordinate_system='cartesian'.
        If False, use fixed n_gl=gl_nodes_per_dim and R=min_physical_radius.

    use_tail_correction : bool
        If True, use the "(exp(-iwψ) - 1)" formulation and add +1 afterward.

    window_potential : bool
        If True, apply psi(x) -> psi(x) * W(x) with
        W(x) = 0.5 * (1 - tanh(|x| - 3R/4)).

    window_radius_fraction : float
        Fraction of R used in the window center (default 0.75 for 3R/4).

    window_u : bool
        If True, apply a radial taper near |x|=R consistent with the
        u-space window used in the reference C implementation.

    window_u_width : float
        Relative width for the u-window (du = width * u_max, default 0.02).

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

    nufft_tile_autotune : bool
        If True, auto-tune tile sizes per n_gl using a short NUFFT timing.

    nufft_tile_candidates : sequence of int
        Candidate tile sizes for auto-tuning.

    use_type1_grid : bool
        If True (and y targets form a uniform grid), use type-1 NUFFT batching
        across w values for faster evaluation; otherwise fall back to type-3.

    type1_interpolate : bool
        If True, interpolate type-1 grid outputs onto the requested y1,y2 grid
        when y does not align exactly with integer k indices.

    type1_max_batch : int
        Maximum number of w values per type-1many call.

    type1_iflag : int
        FINUFFT sign for type-1 (default +1 to match reference C code).

    type1_fftshift : bool
        If True, apply fftshift to center k=0 in type-1 outputs.

    verbose : bool
        Print configuration and timing diagnostics.
    """

    def __init__(
        self,
        lens,
        gl_nodes_per_dim=128,
        min_physical_radius=1.0,
        nufft_tol=1e-9,
        batch_frequencies=True,
        chunk_frequencies=True,
        frequency_binning="log",     # "log" or "linear"
        frequency_bin_width=0.5,     # decades if log, |w| units if linear
        auto_R_from_gl_nodes=True,
        use_tail_correction=True,
        window_potential=True,
        window_radius_fraction=0.75,
        window_u=True,
        window_u_width=0.02,
        coordinate_system="cartesian",  # "cartesian" or "polar"
        polar_radial_nodes=None,
        polar_angular_nodes=None,
        uniform_angular_sampling=True,
        numexpr_nthreads=None,
        parallel_frequencies=True,
        nufft_workers=None,
        nufft_nthreads=1,
        nufft_tile_max_points=4000**2,
        nufft_tile_autotune=True,
        nufft_tile_candidates=(500000, 1000000, 2000000, 4000000),
        use_type1_grid=True,
        type1_interpolate=True,
        type1_max_batch=3,
        type1_iflag=1,
        type1_fftshift=True,
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
        self.window_potential = bool(window_potential)
        self.window_radius_fraction = float(window_radius_fraction)
        if not (0.0 < self.window_radius_fraction < 1.0):
            raise ValueError("window_radius_fraction must be in (0, 1).")
        self.window_u = bool(window_u)
        self.window_u_width = float(window_u_width)
        if self.window_u_width <= 0.0:
            raise ValueError("window_u_width must be > 0.")

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
        if numexpr_nthreads is None and parallel_frequencies:
            numexpr_nthreads = 1
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

        self.nufft_tile_autotune = bool(nufft_tile_autotune)
        self.nufft_tile_candidates = tuple(int(v) for v in nufft_tile_candidates)
        if any(v <= 0 for v in self.nufft_tile_candidates):
            raise ValueError("nufft_tile_candidates must be positive integers.")
        self._tile_size_cache = {}

        self.use_type1_grid = bool(use_type1_grid)
        self.type1_interpolate = bool(type1_interpolate)
        self.type1_max_batch = int(type1_max_batch)
        if self.type1_max_batch < 1:
            raise ValueError("type1_max_batch must be >= 1.")
        self.type1_iflag = int(type1_iflag)
        self.type1_fftshift = bool(type1_fftshift)

    def _choose_tile_size(self, n_gl, n_sources, mode):
        if self.nufft_tile_max_points is None:
            return None
        if n_sources <= self.nufft_tile_max_points:
            return None
        key = (mode, int(n_gl))
        if key in self._tile_size_cache:
            return self._tile_size_cache[key]
        if not self.nufft_tile_autotune:
            tile_size = min(self.nufft_tile_max_points, n_sources)
            self._tile_size_cache[key] = tile_size
            return tile_size

        candidates = []
        for cand in self.nufft_tile_candidates:
            Mtile = min(int(cand), n_sources, self.nufft_tile_max_points)
            if Mtile > 0 and Mtile not in candidates:
                candidates.append(Mtile)
        if not candidates:
            return None

        rng = np.random.default_rng(12345)
        k_targets = 16
        best = candidates[0]
        best_time = float("inf")
        for Mtile in candidates:
            xj = rng.uniform(-np.pi, np.pi, Mtile)
            yj = rng.uniform(-np.pi, np.pi, Mtile)
            cj = rng.normal(size=Mtile) + 1j * rng.normal(size=Mtile)
            t0 = time.perf_counter()
            try:
                if mode == "type1":
                    ms = mt = int(max(8, np.sqrt(k_targets)))
                    _nufft2d1_call(
                        xj, yj, cj, ms, mt,
                        eps=self.nufft_tol,
                        iflag=self.type1_iflag,
                        nthreads=self.nufft_nthreads,
                    )
                else:
                    sk = rng.uniform(-np.pi, np.pi, k_targets)
                    tk = rng.uniform(-np.pi, np.pi, k_targets)
                    _nufft2d3_call(
                        xj, yj, cj, sk, tk,
                        eps=self.nufft_tol,
                        isign=-1,
                        nthreads=self.nufft_nthreads,
                    )
            except Exception:
                continue
            t1 = time.perf_counter()
            dt = t1 - t0
            if dt < best_time:
                best_time = dt
                best = Mtile

        self._tile_size_cache[key] = best
        return best

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

        adaptive = self.auto_R_from_gl_nodes
        if adaptive and self.coordinate_system != "cartesian":
            raise NotImplementedError(
                "Adaptive quadrature is currently implemented only for coordinate_system='cartesian'."
            )

        t_choose0 = time.perf_counter()
        if adaptive:
            w_abs = np.abs(w_vec)
            n_gl_vec = np.array([_adaptive_n_gl(val) for val in w_abs], dtype=int)
            groups = {}
            for idx, n_gl in enumerate(n_gl_vec):
                groups.setdefault(n_gl, []).append(idx)
            group_items = [
                (n_gl, np.asarray(idxs, dtype=int)) for n_gl, idxs in sorted(groups.items())
            ]
        else:
            group_items = [(self.gl_nodes_per_dim, np.arange(len(w_vec)))]
        t_choose = time.perf_counter() - t_choose0

        grid_info = _detect_uniform_grid(y1, y2) if self.use_type1_grid else None
        use_type1 = (
            self.use_type1_grid
            and grid_info is not None
            and _FINUFFT_TYPE1MANY
            and self.coordinate_system == "cartesian"
        )
        if use_type1 and len(w_vec) > 1 and not self.type1_interpolate:
            use_type1 = False
        if self.use_type1_grid and not use_type1 and verbose:
            print(
                "[FresnelNUFFT3] type-1 grid path unavailable; "
                "falling back to type-3."
            )

        if use_type1:
            t_alloc0 = time.perf_counter()
            F_out = np.empty((len(w_vec),) + y1.shape, dtype=np.complex128)
            t_alloc = time.perf_counter() - t_alloc0

            y_abs_max = float(max(np.max(np.abs(y1)), np.max(np.abs(y2))))
            t_quad_total = 0.0
            t_coeff_total = 0.0
            t_nufft_total = 0.0
            group_meta = []

            for n_gl, idxs in group_items:
                w_sub = w_vec[idxs]
                n_sources = n_gl * n_gl
                tile_size = self._choose_tile_size(n_gl, n_sources, mode="type1")
                tile_count = 1
                if tile_size is not None:
                    tile_count = (n_sources + tile_size - 1) // tile_size

                t_quad0 = time.perf_counter()
                x_1d, w_1d = gauss_legendre_1d(n_gl, 1.0, label="R_unit", verbose=verbose)
                x_1d_pi = x_1d * np.pi
                t_quad = time.perf_counter() - t_quad0
                t_quad_total += t_quad

                if tile_size is None:
                    xj = np.repeat(x_1d_pi, n_gl)
                    yj = np.tile(x_1d_pi, n_gl)
                    W_base = np.repeat(w_1d, n_gl) * np.tile(w_1d, n_gl)
                    r_pi = np.hypot(xj, yj)
                    if self.window_u:
                        W_base = W_base * _window_u_taper(r_pi, np.pi, self.window_u_width)

                for t0 in range(0, len(w_sub), self.type1_max_batch):
                    w_chunk = w_sub[t0:t0 + self.type1_max_batch]
                    w_abs_chunk = np.abs(w_chunk)
                    w_max = float(w_abs_chunk.max())
                    k_max = int(np.ceil(y_abs_max * np.sqrt(n_gl * w_max / 2.0) / np.pi))
                    k_max = max(0, k_max)
                    Ny = 2 * k_max + 1

                    Fy_total = np.zeros((len(w_chunk), Ny, Ny), dtype=np.complex128)

                    if tile_size is None:
                        t_coeff0 = time.perf_counter()
                        c = np.empty((len(w_chunk), n_sources), dtype=np.complex128)
                        for t, w_i in enumerate(w_chunk):
                            w_abs_i = abs(w_i)
                            R = np.sqrt(n_gl / (2.0 * w_abs_i))
                            scale = R / np.pi
                            x_phys = xj * scale
                            y_phys = yj * scale
                            r_phys = r_pi * scale
                            psi = self.lens.psi_xy(x_phys, y_phys)
                            if self.window_potential:
                                psi = psi * _window_taper(r_phys, R, self.window_radius_fraction)
                            quad = 0.5 * (x_phys * x_phys + y_phys * y_phys)
                            W = W_base * (R * R)
                            if self.use_tail_correction:
                                c[t] = W * np.exp(1j * w_i * quad) * (np.exp(-1j * w_i * psi) - 1.0)
                            else:
                                c[t] = W * np.exp(1j * w_i * (quad - psi))
                        t_coeff_total += time.perf_counter() - t_coeff0

                        t_n0 = time.perf_counter()
                        Fy = _nufft2d1many_call(
                            xj, yj, c, Ny, Ny,
                            eps=self.nufft_tol,
                            iflag=self.type1_iflag,
                            nthreads=self.nufft_nthreads,
                        )
                        t_nufft_total += time.perf_counter() - t_n0
                        if self.type1_fftshift:
                            Fy = np.fft.fftshift(Fy, axes=(-2, -1))
                        Fy_total = Fy
                    else:
                        for start in range(0, n_sources, tile_size):
                            stop = min(n_sources, start + tile_size)
                            idx = np.arange(start, stop)
                            j = idx // n_gl
                            k = idx - j * n_gl
                            xj_tile = x_1d_pi[j]
                            yj_tile = x_1d_pi[k]
                            W_base_tile = w_1d[j] * w_1d[k]
                            r_pi = np.hypot(xj_tile, yj_tile)
                            if self.window_u:
                                W_base_tile = W_base_tile * _window_u_taper(r_pi, np.pi, self.window_u_width)

                            t_coeff0 = time.perf_counter()
                            c = np.empty((len(w_chunk), stop - start), dtype=np.complex128)
                            for t, w_i in enumerate(w_chunk):
                                w_abs_i = abs(w_i)
                                R = np.sqrt(n_gl / (2.0 * w_abs_i))
                                scale = R / np.pi
                                x_phys = xj_tile * scale
                                y_phys = yj_tile * scale
                                r_phys = r_pi * scale
                                psi = self.lens.psi_xy(x_phys, y_phys)
                                if self.window_potential:
                                    psi = psi * _window_taper(r_phys, R, self.window_radius_fraction)
                                quad = 0.5 * (x_phys * x_phys + y_phys * y_phys)
                                W = W_base_tile * (R * R)
                                if self.use_tail_correction:
                                    c[t] = W * np.exp(1j * w_i * quad) * (np.exp(-1j * w_i * psi) - 1.0)
                                else:
                                    c[t] = W * np.exp(1j * w_i * (quad - psi))
                            t_coeff_total += time.perf_counter() - t_coeff0

                            t_n0 = time.perf_counter()
                            Fy_tile = _nufft2d1many_call(
                                xj_tile, yj_tile, c, Ny, Ny,
                                eps=self.nufft_tol,
                                iflag=self.type1_iflag,
                                nthreads=self.nufft_nthreads,
                            )
                            t_nufft_total += time.perf_counter() - t_n0
                            if self.type1_fftshift:
                                Fy_tile = np.fft.fftshift(Fy_tile, axes=(-2, -1))
                            Fy_total += Fy_tile

                    for t, w_i in enumerate(w_chunk):
                        w_abs_i = abs(w_i)
                        R = np.sqrt(n_gl / (2.0 * w_abs_i))
                        k1 = y1 * w_i * R / np.pi
                        k2 = y2 * w_i * R / np.pi
                        if self.type1_interpolate:
                            Iw = _interp_bilinear_grid(Fy_total[t], k1, k2, k_max)
                        else:
                            k1i = np.rint(k1).astype(int)
                            k2i = np.rint(k2).astype(int)
                            if not (np.allclose(k1, k1i) and np.allclose(k2, k2i)):
                                raise ValueError("Non-integer k indices; enable type1_interpolate.")
                            valid = (
                                (k1i >= -k_max) & (k1i <= k_max) &
                                (k2i >= -k_max) & (k2i <= k_max)
                            )
                            Iw = np.zeros_like(k1, dtype=np.complex128)
                            Iw[valid] = Fy_total[t][k1i[valid] + k_max, k2i[valid] + k_max]
                        val = np.exp(1j * w_i * quad_phase) * Iw * (w_i / (1j * _TWO_PI))
                        if self.use_tail_correction:
                            val = 1.0 + val
                        F_out[idxs[t0 + t]] = val

                group_meta.append(
                    {
                        "n_gl": n_gl,
                        "w_min": float(np.min(np.abs(w_sub))),
                        "w_max": float(np.max(np.abs(w_sub))),
                        "tile_size": tile_size,
                        "tile_count": tile_count,
                    }
                )

            t_total = time.perf_counter() - t_total0
            if verbose:
                n_gl_used = np.array([meta["n_gl"] for meta in group_meta], dtype=int)
                group_msg = f"  grouping:      {_fmt_s(t_choose)} | groups={len(group_meta)}"
                if self.nufft_tile_max_points is None:
                    tile_msg = "  tiling:        disabled"
                else:
                    tile_counts = np.array([meta["tile_count"] for meta in group_meta], dtype=int)
                    tile_msg = (
                        "  tiling:        adaptive (tiles in "
                        f"[{tile_counts.min()}, {tile_counts.max()}], "
                        f"max_points={self.nufft_tile_max_points})"
                    )
                print(
                    "[FresnelNUFFT3] type-1 grid path\n"
                    f"  input prep:    {_fmt_s(t_input)}\n"
                    f"{group_msg}\n"
                    f"  n_gl range:    [{n_gl_used.min()}, {n_gl_used.max()}]\n"
                    f"  quadrature:    {_fmt_s(t_quad_total)}\n"
                    f"  coeffs:        {_fmt_s(t_coeff_total)}\n"
                    f"  NUFFT total:   {_fmt_s(t_nufft_total)}\n"
                    f"{tile_msg}\n"
                    f"  wall total:    {_fmt_s(t_total)}"
                )
            return F_out

        t_alloc0 = time.perf_counter()
        F_out = np.empty((len(w_vec),) + y1.shape, dtype=np.complex128)
        t_alloc = time.perf_counter() - t_alloc0

        t_quad_total = 0.0
        t_pot_total = 0.0
        t_scale_total = 0.0
        t_nufft_total = 0.0
        group_meta = []

        max_workers = os.cpu_count() or 1
        mp_ctx = None
        start_method = None
        if self.parallel_frequencies:
            try:
                mp_ctx = mp.get_context("fork")
            except ValueError:
                mp_ctx = mp.get_context()
            start_method = mp_ctx.get_start_method()

        for n_gl, idxs in group_items:
            w_sub = w_vec[idxs]
            w_abs_sub = np.abs(w_sub)
            if adaptive:
                R = None
            else:
                R = self.min_physical_radius
                if R <= 0.0:
                    raise ValueError("Integration radius R must be positive.")
            n_sources = n_gl * n_gl
            tile_size = self._choose_tile_size(n_gl, n_sources, mode="type3")
            tile_count = 1
            if tile_size is not None:
                tile_count = (n_sources + tile_size - 1) // tile_size
            tile_from_1d = tile_size is not None and self.coordinate_system == "cartesian"

            if adaptive:
                t_quad0 = time.perf_counter()
                if tile_from_1d:
                    x_1d, w_1d = gauss_legendre_1d(n_gl, 1.0, label="R_unit", verbose=verbose)
                    u1_base = None
                    u2_base = None
                    W_base = None
                else:
                    u1_base, u2_base, W_base = gauss_legendre_2d(
                        n_gl, 1.0, label="R_unit", verbose=verbose
                    )
                    x_1d = None
                    w_1d = None
                t_quad = time.perf_counter() - t_quad0
                t_quad_total += t_quad

                ctx = {
                    "adaptive": True,
                    "n_gl": n_gl,
                    "u1_base": u1_base,
                    "u2_base": u2_base,
                    "W_base": W_base,
                    "x_1d": x_1d,
                    "w_1d": w_1d,
                    "x_1d_unit": True if tile_from_1d else False,
                    "tile_from_1d": tile_from_1d,
                    "lens": self.lens,
                    "window_potential": self.window_potential,
                    "window_radius_fraction": self.window_radius_fraction,
                    "window_u": self.window_u,
                    "window_u_width": self.window_u_width,
                    "y1": y1,
                    "y2": y2,
                    "quad_phase": quad_phase,
                    "nufft_tol": self.nufft_tol,
                    "use_tail_correction": self.use_tail_correction,
                    "nufft_nthreads": self.nufft_nthreads,
                    "nufft_tile_size": tile_size,
                }
            else:
                h = np.pi / R
                t_quad0 = time.perf_counter()
                if self.coordinate_system == "cartesian":
                    if tile_from_1d:
                        x_1d, w_1d = gauss_legendre_1d(n_gl, R, label="R", verbose=verbose)
                        x1 = None
                        x2 = None
                        W = None
                    else:
                        x1, x2, W = gauss_legendre_2d(n_gl, R, label="R", verbose=verbose)
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
                t_quad_total += t_quad

                if tile_from_1d:
                    t_pot = 0.0
                    t_scale = 0.0
                    ctx = {
                        "adaptive": False,
                        "tile_from_1d": True,
                        "n_gl": n_gl,
                        "x_1d": x_1d,
                        "w_1d": w_1d,
                        "x_1d_unit": False,
                        "R": R,
                        "y1": y1,
                        "y2": y2,
                        "quad_phase": quad_phase,
                        "h": h,
                        "nufft_tol": self.nufft_tol,
                        "use_tail_correction": self.use_tail_correction,
                        "nufft_nthreads": self.nufft_nthreads,
                        "nufft_tile_size": tile_size,
                        "lens": self.lens,
                        "window_potential": self.window_potential,
                        "window_radius_fraction": self.window_radius_fraction,
                        "window_u": self.window_u,
                        "window_u_width": self.window_u_width,
                    }
                else:
                    t_pot0 = time.perf_counter()
                    psi = self.lens.psi_xy(x1, x2)
                    r = np.hypot(x1, x2)
                    if self.window_potential:
                        psi = psi * _window_taper(r, R, self.window_radius_fraction)
                    quad = 0.5 * (x1 * x1 + x2 * x2)
                    base = quad - psi
                    W_eff = W
                    if self.window_u:
                        W_eff = W_eff * _window_u_taper(r, R, self.window_u_width)
                    t_pot = time.perf_counter() - t_pot0
                    t_pot_total += t_pot

                    t_scale0 = time.perf_counter()
                    xj = h * x1
                    yj = h * x2
                    t_scale = time.perf_counter() - t_scale0
                    t_scale_total += t_scale

                    ctx = {
                        "adaptive": False,
                        "tile_from_1d": False,
                        "xj": xj,
                        "yj": yj,
                        "W": W_eff,
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

            n_w_sub = len(w_sub)
            if self.nufft_workers is None:
                n_workers = min(n_w_sub, max_workers)
            else:
                n_workers = min(n_w_sub, self.nufft_workers, max_workers)
            if not self.parallel_frequencies or n_w_sub <= 1:
                n_workers = 1

            t_nufft0 = time.perf_counter()
            if n_workers > 1:
                if start_method == "fork":
                    _init_worker_ctx(ctx)
                    pool = mp_ctx.Pool(processes=n_workers, initializer=_init_worker_ctx)
                else:
                    pool = mp_ctx.Pool(processes=n_workers, initializer=_init_worker_ctx, initargs=(ctx,))
                try:
                    tasks = zip(idxs, w_sub)
                    for idx, Fw in pool.imap_unordered(_worker_task, tasks, chunksize=1):
                        F_out[idx] = Fw
                finally:
                    pool.close()
                    pool.join()
            else:
                _init_worker_ctx(ctx)
                for idx, w_i in zip(idxs, w_sub):
                    F_out[idx] = _eval_frequency(w_i, ctx)
            t_nufft_total += time.perf_counter() - t_nufft0

            group_meta.append(
                {
                    "n_gl": n_gl,
                    "w_min": float(w_abs_sub.min()),
                    "w_max": float(w_abs_sub.max()),
                    "R_min": float(np.sqrt(n_gl / (2.0 * w_abs_sub.max()))) if adaptive else float(R),
                    "R_max": float(np.sqrt(n_gl / (2.0 * w_abs_sub.min()))) if adaptive else float(R),
                    "tile_size": tile_size,
                    "tile_count": tile_count,
                    "n_sources": n_sources,
                }
            )

        t_nufft = t_nufft_total
        t_total = time.perf_counter() - t_total0

        if verbose:
            if adaptive:
                n_gl_used = np.array([meta["n_gl"] for meta in group_meta], dtype=int)
                R_min_used = np.array([meta["R_min"] for meta in group_meta], dtype=float)
                R_max_used = np.array([meta["R_max"] for meta in group_meta], dtype=float)
                R_low = float(R_min_used.min())
                R_high = float(R_max_used.max())
                h_low = np.pi / R_high
                h_high = np.pi / R_low
                r_msg = (
                    f"R in [{R_low:.6g}, {R_high:.6g}] | "
                    f"n_gl in [{n_gl_used.min()}, {n_gl_used.max()}]"
                )
                h_msg = f"h in [{h_low:.6g}, {h_high:.6g}]"
                group_msg = f"  grouping:      {_fmt_s(t_choose)} | groups={len(group_meta)}"
                range_msg = f"  adaptive R/h:  {r_msg}, {h_msg}"

                if self.nufft_tile_max_points is None:
                    tile_msg = "  tiling:        disabled"
                else:
                    tile_counts = np.array([meta['tile_count'] for meta in group_meta], dtype=int)
                    tile_msg = (
                        "  tiling:        adaptive (tiles in "
                        f"[{tile_counts.min()}, {tile_counts.max()}], "
                        f"max_points={self.nufft_tile_max_points})"
                    )
            else:
                meta = group_meta[0]
                r_msg = f"R={meta['R']:.6g} (fixed)"
                h_msg = f"h={np.pi / meta['R']:.6g}"
                group_msg = f"  setup:         {_fmt_s(t_choose)}"
                range_msg = f"  R/h:           {r_msg}, {h_msg}"
                if meta["tile_size"] is None:
                    tile_msg = "  tiling:        disabled"
                else:
                    tile_msg = (
                        f"  tiling:        {meta['tile_count']} tiles "
                        f"(max_points={meta['tile_size']}, sources={meta['n_sources']})"
                    )

            print(
                "[FresnelNUFFT3] x-domain path\n"
                f"  input prep:    {_fmt_s(t_input)}\n"
                f"{group_msg}\n"
                f"{range_msg}\n"
                f"  quadrature:    {_fmt_s(t_quad_total)}\n"
                f"  lens + base:   {_fmt_s(t_pot_total)}\n"
                f"  scaling:       {_fmt_s(t_scale_total)}\n"
                f"  alloc out:     {_fmt_s(t_alloc)}\n"
                f"  NUFFT total:   {_fmt_s(t_nufft)}\n"
                f"{tile_msg}\n"
                f"  workers:       up to {max_workers} (nufft_nthreads={self.nufft_nthreads})\n"
                f"  wall total:    {_fmt_s(t_total)}"
            )

        return F_out
