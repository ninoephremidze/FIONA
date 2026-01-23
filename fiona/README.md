# FIONA

FIONA (Fresnel Integral Optimization via Non-uniform trAnsforms) provides fast
numerical evaluation of the gravitational-wave lensing amplification factor
F(w, y) using NUFFT-based quadrature. It includes:

- 2D Fresnel integrals for general (non-axisymmetric) lenses.
- Axisymmetric solvers based on fast Hankel transforms (optional Julia/SciPy).
- A collection of lens models compatible with numpy broadcasting.
- Utilities for caching Gauss-Legendre quadrature grids and controlling threads.

This README documents the main public API and recommended usage patterns.

## Installation

FIONA is a pure-Python package but relies on optional numerical backends:

- finufft (required for FresnelNUFFT3)
- julia + FastHankelTransform (optional, for FresnelNUFHT)
- scipy (optional, for spline helpers and SciPy Hankel transform)

Install the dependencies you need in your environment. Example:

```
pip install finufft numpy numexpr
```

Optional:

```
pip install scipy
```

## Environment configuration

FIONA caches Gauss-Legendre nodes and weights on disk. Configure a cache
directory with:

```
export FIONA_GL2D_DIR="/path/to/gl_cache"
export FIONA_GL2D_STRICT=0
```

If FIONA_GL2D_STRICT=1 and a cache file is missing, an error is raised instead
of rebuilding the grid.

Threading controls (recommended for performance stability):

```
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

FIONA also provides:

```
from fiona import set_num_threads
set_num_threads(1)
```

## Quick start (2D Fresnel integral)

```python
import numpy as np
from fiona import SIS, FresnelNUFFT3

lens = SIS(psi0=1.0)
solver = FresnelNUFFT3(
    lens,
    gl_nodes_per_dim=2000,
    min_physical_radius=10.0,
    auto_R_from_gl_nodes=True,
    use_tail_correction=True,
    window_potential=True,
    window_u=True,
    nufft_tol=1e-9,
    nufft_workers=32,
    nufft_nthreads=1,
    verbose=True,
)

w = np.linspace(0.1, 20.0, 200)
y1 = np.array([0.3])
y2 = np.array([0.0])

F = solver(w, y1, y2)  # shape (len(w), 1)
F = F[:, 0]
```

## Lenses

All lens classes expose:

```
psi_xy(x1, x2)
```

for broadcasting numpy arrays. Selected lenses include:

- SIS (Singular Isothermal Sphere)
- PointLens (Plummer softened point lens)
- CIS (Cored Isothermal Sphere)
- NFW, OffcenterNFW
- PIED, EllipticalSIS
- ClumpySIELens, ClumpyNFWLens
- Shear, SIE, EPL, NFW_ELLIPSE_POTENTIAL

Axisymmetric lenses also implement:

```
psi_r(r)
```

## FresnelNUFFT3 (general 2D solver)

Main signature:

```
FresnelNUFFT3(lens, ...)
F = solver(w, y1, y2, verbose=None)
```

Key options:

- gl_nodes_per_dim: number of Gauss-Legendre nodes per side (n_gl).
- min_physical_radius: R for fixed-grid mode.
- auto_R_from_gl_nodes: adaptive mode; chooses n_gl by w-bin and sets
  R(w) = sqrt(n_gl / (2*|w|)).
- use_tail_correction: use exp(i w x^2/2) * (exp(-i w psi) - 1) and add +1.
- window_potential: apply psi(x) -> psi(x) * W(x) with
  W(x) = 0.5 * (1 - tanh(|x| - 0.75*R)).
- window_u: apply an additional radial taper near |x|=R to stabilize the
  integrand, similar to the reference C implementation.
- nufft_tol: FINUFFT accuracy (default 1e-9).
- parallel_frequencies: parallelize over w (one NUFFT per worker).
- nufft_workers: number of worker processes (default os.cpu_count()).
- nufft_nthreads: FINUFFT threads per call (usually 1).
- nufft_tile_max_points: maximum number of sources per NUFFT call; enables
  tiling for large n_gl.
- nufft_tile_autotune: choose tile size from candidates by short timing.

### Adaptive quadrature bins

When auto_R_from_gl_nodes=True, n_gl is chosen from |w|:

- w < 10   -> n_gl = 1000
- 10-20    -> n_gl = 2000
- ...
- 90-100   -> n_gl = 10000

R is computed per w as:

```
R(w) = sqrt(n_gl / (2*|w|))
```

### Type-1 grid path (optional)

If y1,y2 form a uniform Cartesian grid (e.g., meshgrid), FresnelNUFFT3 can
use a type-1 NUFFT to batch across w values (faster). Enable with:

```
use_type1_grid=True
```

If the finufft build does not expose nufft2d1many, FIONA falls back to
per-w nufft2d1 calls (still type-1).

If targets are not on a uniform grid, the solver falls back to the type-3
path automatically.

## Axisymmetric solvers

- FresnelNUFHT: Julia FastHankelTransform backend (optional).
- FresnelHankelAxisymmetricTrapezoidal: uses nufht with trapezoidal grid.
- FresnelHankelAxisymmetricSciPy: uses scipy.fft.fht (optional).

These are useful for spherically symmetric lenses and 1D (radial) targets.

## Utilities

From fiona.utils:

- gauss_legendre_2d / gauss_legendre_1d
- spline_fit_eval (requires SciPy)
- align_global_phase
- CPUTracker

## Performance tips

1. Set OMP_NUM_THREADS=1 and use parallel_frequencies with nufft_workers.
2. Use nufft_tol=1e-9 unless you need higher accuracy.
3. Use adaptive quadrature if you sweep a wide range of w.
4. For large n_gl, enable tiling and auto-tune tile sizes.
5. If your y targets form a uniform grid, enable use_type1_grid.

## Troubleshooting

GL cache issues:

- If you see non-finite or absurd weights, delete the cached GL files in
  FIONA_GL2D_DIR and rerun.

Type-1 grid path unavailable:

- Ensure y1,y2 are uniform meshgrid arrays.
- Ensure your finufft build exposes nufft2d1many (optional).

## Version

See fiona/__init__.py for __version__.
