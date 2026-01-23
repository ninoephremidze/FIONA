####################################################################
# fiona/utils.py
####################################################################

from __future__ import annotations

import os
import json
import math
import time
import hashlib
import pathlib
from math import ceil
from multiprocessing import Pool

import numpy as np
from numpy.polynomial.legendre import leggauss

try:
    import resource
    _HAS_RESOURCE = True
except Exception:
    _HAS_RESOURCE = False

import psutil

# =============================================================================
# Configuration via environment variables
# =============================================================================

_FIONA_GL2D_DIR = os.environ.get("FIONA_GL2D_DIR", "")
_FIONA_GL2D_STRICT = os.environ.get("FIONA_GL2D_STRICT", "0") == "1"

NUM_WORKERS = min(psutil.cpu_count(logical=True) or 1, 112)
TARGET_CHUNKS_PER_PROC = 8

# =============================================================================
# Filesystem helpers
# =============================================================================

def _ensure_store_dir():
    if not _FIONA_GL2D_DIR:
        return None
    p = pathlib.Path(_FIONA_GL2D_DIR)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return p


def _gl2d_base(n, u_max):
    store = _ensure_store_dir()
    if store is None:
        return None
    return store / f"gl2d_n{int(n)}_U{int(u_max)}"


def _gl2d_paths(n, u_max):
    base = _gl2d_base(n, u_max)
    if base is None:
        return None
    return {
        "base": base,
        "x":    base.with_suffix(".x.npy"),
        "w":    base.with_suffix(".w.npy"),
        "u1":   base.with_suffix(".u1.npy"),
        "u2":   base.with_suffix(".u2.npy"),
        "W":    base.with_suffix(".W.npy"),
        "meta": base.with_suffix(".meta.json"),
    }


def _gl2d_exists(n, u_max):
    p = _gl2d_paths(n, u_max)
    if p is None:
        return False
    return all(p[k].exists() for k in ("x", "w", "u1", "u2", "W", "meta"))


# =============================================================================
# Hash helper
# =============================================================================

def _sha256_file(path: pathlib.Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# =============================================================================
# Parallel memmap worker
# =============================================================================

def _fill_rows_worker(args):
    """
    Fill contiguous row blocks of flattened (n*n) GL arrays.
    """
    (i0, i1, n, x_path, w_path, u1_path, u2_path, W_path) = args

    x = np.load(x_path, mmap_mode="r")
    w = np.load(w_path, mmap_mode="r")

    u1mm = np.memmap(u1_path, dtype=np.float64, mode="r+", shape=(n * n,))
    u2mm = np.memmap(u2_path, dtype=np.float64, mode="r+", shape=(n * n,))
    Wmm  = np.memmap(W_path,  dtype=np.float64, mode="r+", shape=(n * n,))

    for i in range(i0, i1):
        row = slice(i * n, (i + 1) * n)
        u1mm[row] = x[i]
        u2mm[row] = x[:]
        Wmm[row]  = w[i] * w[:]

    u1mm.flush()
    u2mm.flush()
    Wmm.flush()


# =============================================================================
# Core GL2D builder (memmap + multiprocessing)
# =============================================================================

def _compute_and_store_gl2d(
    n: int,
    u_max: float,
    nprocs: int = NUM_WORKERS,
    target_chunks_per_proc: int = TARGET_CHUNKS_PER_PROC,
):
    n = int(n)
    u_max = float(u_max)

    paths = _gl2d_paths(n, u_max)
    store = _ensure_store_dir()
    if store is None:
        raise RuntimeError("FIONA_GL2D_DIR is not set or not writable.")

    # --------------------------------------------------
    # 1) 1D Gauss–Legendre (cheap)
    # --------------------------------------------------
    xi, wi = leggauss(n)
    x = u_max * xi
    w = u_max * wi

    np.save(paths["x"], x)
    np.save(paths["w"], w)

    # --------------------------------------------------
    # 2) Allocate output memmaps
    # --------------------------------------------------
    for key in ("u1", "u2", "W"):
        mm = np.memmap(paths[key], dtype=np.float64,
                       mode="w+", shape=(n * n,))
        mm[:] = 0.0
        mm.flush()
        del mm

    # --------------------------------------------------
    # 3) Chunk rows
    # --------------------------------------------------
    nprocs = max(1, min(int(nprocs), n))
    target_tasks = max(nprocs, target_chunks_per_proc * nprocs)
    chunk_rows = max(1, int(ceil(n / target_tasks)))

    tasks = []
    for i0 in range(0, n, chunk_rows):
        i1 = min(n, i0 + chunk_rows)
        tasks.append((
            i0, i1, n,
            str(paths["x"]),
            str(paths["w"]),
            str(paths["u1"]),
            str(paths["u2"]),
            str(paths["W"]),
        ))

    # --------------------------------------------------
    # 4) Parallel fill
    # --------------------------------------------------
    t0 = time.perf_counter()
    with Pool(processes=nprocs) as pool:
        for _ in pool.imap_unordered(_fill_rows_worker, tasks):
            pass
    t1 = time.perf_counter()

    # --------------------------------------------------
    # 5) Metadata
    # --------------------------------------------------
    meta = {
        "version": 2,
        "n": n,
        "u_max": u_max,
        "dtype": "float64",
        "dim": 2,
        "build_time_sec": t1 - t0,
        "cores_used": nprocs,
        "files": {k: paths[k].name for k in ("x", "w", "u1", "u2", "W")},
        "sha256": {k: _sha256_file(paths[k]) for k in ("x", "w", "u1", "u2", "W")},
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "created": time.time(),
    }

    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2)

        
def _compute_and_store_gl2d_polar(n_r, n_theta, u_max):
    n_r = int(n_r)
    n_theta = int(n_theta)
    u_max = float(u_max)

    store = _ensure_store_dir()
    if store is None:
        raise RuntimeError("FIONA_GL2D_DIR is not set or not writable.")

    base = store / f"gl2dpolar_nr{n_r}_nt{n_theta}_U{int(u_max)}"

    # --------------------------------------------------
    # 1D Gauss–Legendre rules
    # --------------------------------------------------
    xi_r, wi_r = leggauss(n_r)
    xi_t, wi_t = leggauss(n_theta)

    # Map domains
    r = 0.5 * u_max * (xi_r + 1.0)      # [0, Umax]
    wr = 0.5 * u_max * wi_r

    theta = np.pi * (xi_t + 1.0)        # [0, 2π]
    wt = np.pi * wi_t

    # --------------------------------------------------
    # Allocate flattened arrays
    # --------------------------------------------------
    size = n_r * n_theta
    r_mm     = np.memmap(base.with_suffix(".r.npy"),
                          dtype=np.float64, mode="w+", shape=(size,))
    theta_mm = np.memmap(base.with_suffix(".theta.npy"),
                          dtype=np.float64, mode="w+", shape=(size,))
    W_mm     = np.memmap(base.with_suffix(".W.npy"),
                          dtype=np.float64, mode="w+", shape=(size,))

    # --------------------------------------------------
    # Fill arrays
    # --------------------------------------------------
    k = 0
    for i in range(n_r):
        ri = r[i]
        wi = wr[i]
        for j in range(n_theta):
            r_mm[k]     = ri
            theta_mm[k] = theta[j]
            W_mm[k]     = wi * wt[j] * ri   # Jacobian r
            k += 1

    r_mm.flush()
    theta_mm.flush()
    W_mm.flush()

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------
    meta = {
        "version": 1,
        "coord": "polar",
        "n_r": n_r,
        "n_theta": n_theta,
        "u_max": u_max,
        "dim": 2,
        "created": time.time(),
    }
    with open(base.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def _compute_and_store_gl2d_polar_uniform_theta(n_r, n_theta, u_max):
    n_r = int(n_r)
    n_theta = int(n_theta)
    u_max = float(u_max)

    store = _ensure_store_dir()
    if store is None:
        raise RuntimeError("FIONA_GL2D_DIR is not set or not writable.")

    base = store / f"gl2dpolarU_nr{n_r}_nt{n_theta}_U{int(u_max)}"

    # --- GL in r ---
    xi_r, wi_r = leggauss(n_r)
    r = 0.5 * u_max * (xi_r + 1.0)
    wr = 0.5 * u_max * wi_r

    # --- Uniform theta ---
    theta = 2.0 * np.pi * np.arange(n_theta) / n_theta
    wt = 2.0 * np.pi / n_theta

    size = n_r * n_theta
    r_mm     = np.memmap(base.with_suffix(".r.npy"), dtype=np.float64, mode="w+", shape=(size,))
    theta_mm = np.memmap(base.with_suffix(".theta.npy"), dtype=np.float64, mode="w+", shape=(size,))
    W_mm     = np.memmap(base.with_suffix(".W.npy"), dtype=np.float64, mode="w+", shape=(size,))

    k = 0
    for i in range(n_r):
        ri = r[i]
        wi = wr[i]
        for j in range(n_theta):
            r_mm[k]     = ri
            theta_mm[k] = theta[j]
            W_mm[k]     = wi * wt * ri   # Jacobian r
            k += 1

    r_mm.flush()
    theta_mm.flush()
    W_mm.flush()

    meta = {
        "version": 1,
        "coord": "polar",
        "theta_rule": "uniform",
        "n_r": n_r,
        "n_theta": n_theta,
        "u_max": u_max,
        "created": time.time(),
    }

    with open(base.with_suffix(".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# Public API
# =============================================================================

def gauss_legendre_2d(n, u_max):
    """
    Load or build 2D Gauss–Legendre nodes and weights on [-u_max, u_max]^2.

    Returns:
        u1, u2, W : memory-mapped 1D arrays of length n*n
    """
    n = int(n)
    u_max = float(u_max)

    if not _FIONA_GL2D_DIR:
        raise RuntimeError("FIONA_GL2D_DIR is not set.")

    if not _gl2d_exists(n, u_max):
        if _FIONA_GL2D_STRICT:
            raise FileNotFoundError(
                f"No precomputed GL2D files for (n={n}, Umax={u_max})."
            )
        print(f"[FIONA] computing GL2D (n={n}, Umax={u_max})...")
        _compute_and_store_gl2d(n, u_max)

    p = _gl2d_paths(n, u_max)

    u1 = np.memmap(p["u1"], dtype=np.float64, mode="r", shape=(n * n,))
    u2 = np.memmap(p["u2"], dtype=np.float64, mode="r", shape=(n * n,))
    W  = np.memmap(p["W"],  dtype=np.float64, mode="r", shape=(n * n,))

    return u1, u2, W


def gauss_legendre_polar_2d(n_r, n_theta, u_max):
    """
    Load or build 2D Gauss–Legendre nodes and weights in polar coordinates.

    Returns:
        r, theta, W : memory-mapped 1D arrays of length n_r * n_theta
    """
    n_r = int(n_r)
    n_theta = int(n_theta)
    u_max = float(u_max)

    if not _FIONA_GL2D_DIR:
        raise RuntimeError("FIONA_GL2D_DIR is not set.")

    store = pathlib.Path(_FIONA_GL2D_DIR)
    base = store / f"gl2dpolar_nr{n_r}_nt{n_theta}_U{int(u_max)}"

    r_path     = base.with_suffix(".r.npy")
    theta_path = base.with_suffix(".theta.npy")
    W_path     = base.with_suffix(".W.npy")
    meta_path  = base.with_suffix(".meta.json")

    if not (r_path.exists() and theta_path.exists() and W_path.exists()):
        if _FIONA_GL2D_STRICT:
            raise FileNotFoundError(
                f"No polar GL2D files for (n_r={n_r}, n_theta={n_theta}, U={u_max})"
            )
        print(
            f"[FIONA] computing polar GL2D "
            f"(n_r={n_r}, n_theta={n_theta}, Umax={u_max})..."
        )
        _compute_and_store_gl2d_polar(n_r, n_theta, u_max)

    r     = np.memmap(r_path,     dtype=np.float64, mode="r",
                      shape=(n_r * n_theta,))
    theta = np.memmap(theta_path, dtype=np.float64, mode="r",
                      shape=(n_r * n_theta,))
    W     = np.memmap(W_path,     dtype=np.float64, mode="r",
                      shape=(n_r * n_theta,))

    return r, theta, W


def gauss_legendre_polar_uniform_theta_2d(n_r, n_theta, u_max):
    n_r = int(n_r)
    n_theta = int(n_theta)
    u_max = float(u_max)

    if not _FIONA_GL2D_DIR:
        raise RuntimeError("FIONA_GL2D_DIR is not set.")

    base = pathlib.Path(_FIONA_GL2D_DIR) / f"gl2dpolarU_nr{n_r}_nt{n_theta}_U{int(u_max)}"

    r_path     = base.with_suffix(".r.npy")
    theta_path = base.with_suffix(".theta.npy")
    W_path     = base.with_suffix(".W.npy")

    if not (r_path.exists() and theta_path.exists() and W_path.exists()):
        if _FIONA_GL2D_STRICT:
            raise FileNotFoundError(
                f"No polar-uniform GL2D files for (n_r={n_r}, n_theta={n_theta}, U={u_max})"
            )
        print(
            f"[FIONA] computing polar GL (uniform theta) "
            f"(n_r={n_r}, n_theta={n_theta}, Umax={u_max})..."
        )
        _compute_and_store_gl2d_polar_uniform_theta(n_r, n_theta, u_max)

    r     = np.memmap(r_path,     dtype=np.float64, mode="r", shape=(n_r * n_theta,))
    theta = np.memmap(theta_path, dtype=np.float64, mode="r", shape=(n_r * n_theta,))
    W     = np.memmap(W_path,     dtype=np.float64, mode="r", shape=(n_r * n_theta,))

    return r, theta, W

# =============================================================================
# Complex interpolation helper (unchanged)
# =============================================================================

def interp_complex_logx(x, xp, fp):
    xr = np.interp(np.log(x), np.log(xp), np.real(fp))
    xi = np.interp(np.log(x), np.log(xp), np.imag(fp))
    return xr + 1j * xi


# =============================================================================
# CPU usage tracker
# =============================================================================

class CPUTracker:
    """
    Measure average parallelism over a code block:
      effective_cores = CPU_seconds / wall_seconds
    """

    def __init__(self, include_children=True):
        self.include_children = include_children

    def __enter__(self):
        self._t0 = time.perf_counter()
        self._p0 = time.process_time()
        if _HAS_RESOURCE:
            r_self0 = resource.getrusage(resource.RUSAGE_SELF)
            r_child0 = resource.getrusage(resource.RUSAGE_CHILDREN)
            self._r0s = (
                r_self0.ru_utime + r_self0.ru_stime,
                r_child0.ru_utime + r_child0.ru_stime,
            )
        else:
            self._r0s = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self._t1 = time.perf_counter()
        self._p1 = time.process_time()
        wall = max(1e-12, self._t1 - self._t0)

        if self._r0s is not None:
            r_self1 = resource.getrusage(resource.RUSAGE_SELF)
            r_child1 = resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu = (
                (r_self1.ru_utime + r_self1.ru_stime) - self._r0s[0]
                + (r_child1.ru_utime + r_child1.ru_stime) - self._r0s[1]
            )
        else:
            cpu = max(0.0, self._p1 - self._p0)

        self.wall = wall
        self.cpu = max(0.0, cpu)
        n = psutil.cpu_count(logical=True) or os.cpu_count() or 1
        self.effective_cores = self.cpu / self.wall
        self.avg_cpu_percent = 100.0 * self.effective_cores / n
        self.n_logical = n

    def report(self, label="[CPU]"):
        return (
            f"{label} avg {self.effective_cores:.2f} cores over {self.wall:.3f}s "
            f"({self.avg_cpu_percent:.1f}% of {self.n_logical} logical cores; "
            f"CPU sec={self.cpu:.3f})"
        )


# =============================================================================
# Timing helpers
# =============================================================================

def time_func(fn, *args, repeat=3, warmup=1, **kwargs):
    for _ in range(max(0, warmup)):
        fn(*args, **kwargs)
    best = float("inf")
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best


def align_global_phase(a, b):
    m = np.mean(a / np.maximum(1e-300, b))
    phi = np.angle(m)
    return a * np.exp(-1j * phi), phi

# =============================================================================
# Plotting settings
# =============================================================================

try:
    from scipy.interpolate import CubicSpline, UnivariateSpline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    
    
def _make_sorted_unique(x, y):
    """Ensure x is strictly increasing for spline routines."""
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    mask = np.concatenate(([True], np.diff(x) > 0))
    return x[mask], y[mask]


def spline_fit_eval(x, y, x_fine, method="cubic", smooth_s=None):
    """
    Fit/evaluate a spline for y(x) on x_fine.

    method:
      - "cubic": interpolating cubic spline (passes through points)
      - "smooth": smoothing spline (needs SciPy; uses UnivariateSpline)
    smooth_s:
      - only used for method="smooth". Larger => smoother.
    """
    x, y = _make_sorted_unique(x, y)
    x_fine = np.asarray(x_fine)

    if _HAS_SCIPY:
        if method == "smooth":
            if smooth_s is None:
                smooth_s = 0.001 * len(x) * np.var(y)
            spl = UnivariateSpline(x, y, k=3, s=smooth_s)
            return spl(x_fine)
        else:
            spl = CubicSpline(x, y, bc_type="natural")
            return spl(x_fine)

    return np.interp(x_fine, x, y)