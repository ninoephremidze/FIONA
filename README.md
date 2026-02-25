# FIONA

**Fresnel Integral Optimization via Non-uniform trAnsforms**

FIONA is a Python package for fast, high-accuracy computation of Fresnel diffraction integrals arising in gravitational-wave lensing. It combines Gauss–Legendre quadrature with Non-Uniform Fast Fourier/Hankel Transforms (NUFFT/NUFHT) to evaluate the Fresnel–Kirchhoff diffraction integral over a wide range of gravitational lens models — both axisymmetric and fully two-dimensional.

---

## Table of Contents

- [Features](#features)
- [Mathematical Background](#mathematical-background)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Example Notebooks](#example-notebooks)
- [Lens Models](#lens-models)
- [Solvers](#solvers)
- [Configuration](#configuration)
- [Utilities](#utilities)
- [Dependencies](#dependencies)
- [Author](#author)

---

## Features

### Mathematical method

FIONA evaluates the **Fresnel–Kirchhoff diffraction integral** (amplification factor) arising in the wave-optics treatment of gravitational lensing:

$$\Large F(w, y) = \frac{w}{2\pi i} \iint d^2x \; \exp\left\lbrace iw \left[ \frac{|\mathbf{x} - \mathbf{y}|^2}{2} - \psi(\mathbf{x}) \right] \right\rbrace$$

where $w$ is the dimensionless frequency, $y$ is the (dimensionless) source position, $\mathbf{x}$ is the image-plane coordinate, and $\psi(\mathbf{x})$ is the projected lensing potential.

- **Gauss–Legendre (GL) quadrature** discretises the image-plane integral into a set of carefully chosen nodes and weights, replacing the continuous integral with a weighted sum:

  $$\Large F(w, y) \approx \frac{w}{2\pi i} \sum_k w_k \exp\left\lbrace iw \left[ \frac{|\mathbf{x}_k|^2}{2} - \psi(\mathbf{x}_k) \right] \right\rbrace \exp\left\lbrace -iw\, \mathbf{x}_k \cdot \mathbf{y} \right\rbrace$$

  The GL grid is computed on demand, cached to disk, and memory-mapped for reuse. When `auto_R_from_gl_nodes=True` the outer radius of integration is automatically tuned to the frequency range so that the oscillatory integrand is well-sampled without unnecessary computation.

- **Axisymmetric solver — Non-Uniform Fast Hankel Transform (NUFHT)**: For lenses with circular symmetry the 2-D integral reduces to a 1-D Hankel transform. FIONA uses a compiled Non-Uniform Fast Hankel Transform extension (`_pynufht`) to evaluate:

  $$\Large F(w, y) = \frac{1}{i} \exp\left(\frac{iwy^2}{2}\right) \int_0^\infty u\, du\; f_w(u)\, J_0(wuy)$$

  where $f_w(u) = \exp\left(i\left[\frac{u^2}{2w} - w\psi\left(\frac{u}{w}\right)\right]\right)$ and $J_0$ is the Bessel function of order zero. The NUFHT evaluates this transform at arbitrary (non-uniform) output points $y$ in $O(N \log N)$ time.

- **General 2-D solver — Non-Uniform Fast Fourier Transform (NUFFT)**: For fully two-dimensional lens potentials the Fourier-like kernel $\exp(-iw\mathbf{x}\cdot\mathbf{y})$ is evaluated using FINUFFT's 2-D type-3 (scattered-to-scattered) or type-1 (scattered-to-uniform-grid) NUFFT. This reduces the cost of each frequency evaluation from $O(N^2)$ to $O(N \log N)$ while preserving the full flexibility of arbitrary lens geometries.

### Implementation details

- **Axisymmetric solver** (`FresnelNUFHT`): exploits circular symmetry via a compiled Non-Uniform Fast Hankel Transform for maximum speed on axis-symmetric lenses.
- **General 2-D solver** (`FresnelNUFFT3`): handles arbitrary (non-axisymmetric) lens potentials using 2-D NUFFT type-3 (scattered targets) or type-1 (uniform output grid). Frequencies are distributed across CPU worker processes for high throughput.
- **Rich lens library**: includes SIS, NFW, CIS, SIE, EPL, external shear, elliptical and pseudo-elliptical models, off-center and clumpy lenses — see [Lens Models](#lens-models).
- **Adaptive quadrature**: Gauss–Legendre nodes are built on demand, cached to disk as memory-mapped arrays, and reused across solver calls.
- **Thread control**: a single function call (`set_num_threads`) propagates thread limits to NumPy, FINUFFT, OpenBLAS, and MKL simultaneously, preventing thread over-subscription when worker processes run NUFFTs in parallel.
- **CPU profiling**: built-in `CPUTracker` context manager measures and reports effective CPU core utilisation over any code block.

---

## Mathematical Background

### The Fresnel–Kirchhoff diffraction integral

In the wave-optics (Fresnel) limit, the complex amplification factor for a gravitationally lensed gravitational wave is:

$$\Large F(w, y) = \frac{w}{2\pi i} \iint d^2x \; \exp\left\lbrace iw\,\varphi(\mathbf{x}, \mathbf{y}) \right\rbrace$$

where the **Fermat potential** is

$$\Large \varphi(\mathbf{x}, \mathbf{y}) = \frac{|\mathbf{x} - \mathbf{y}|^2}{2} - \psi(\mathbf{x})$$

$w = (1 + z_L)(M_{Lz}/M_\odot)(f/f_0)$ is a dimensionless frequency proportional to the lens mass and the gravitational-wave frequency, $y$ is the dimensionless source position (in units of the Einstein radius), and $\psi(\mathbf{x})$ is the (dimensionless) projected lensing potential.

The geometric-optics limit is recovered as $w \to \infty$; the full wave-optics computation via FIONA is essential in the intermediate and low-frequency regimes.

### Gauss–Legendre quadrature

FIONA replaces the continuous integral with a discrete sum over $N$ Gauss–Legendre (GL) quadrature nodes $\lbrace x_k, w_k \rbrace$ on a square or polar grid of half-extent $U_{\max}$:

$$\Large F(w, y) \approx \frac{w}{2\pi i} \sum_k w_k \exp\left\lbrace iw\,\varphi(\mathbf{x}_k, \mathbf{y}) \right\rbrace$$

The nodes and weights are computed once, cached to disk, and memory-mapped for fast reuse. When `auto_R_from_gl_nodes=True` the integration radius is set to

$$\Large U_{\max} = \max\left( U_{\min},\; \sqrt{\frac{N}{2w_{\max}}} \right)$$

so that the rapidly oscillating integrand has at least one GL node per half-period at the highest frequency $w_{\max}$.

### Axisymmetric case — Non-Uniform Fast Hankel Transform

When $\psi(\mathbf{x}) = \psi(r)$ depends only on $r = |\mathbf{x}|$, the 2-D integral reduces via the Jacobi–Anger expansion to a 1-D **Hankel transform of order zero**:

$$\Large F(w, y) = \frac{\exp\left(iwy^2/2\right)}{i} \int_0^\infty r\,dr\; f_w(r)\,J_0(wry)$$

where the integrand kernel is

$$\Large f_w(r) = \exp\left\lbrace i\left[ \frac{r^2}{2w} - w\,\psi\left(\frac{r}{w}\right) \right] \right\rbrace$$

FIONA discretises the radial integral with 1-D GL nodes and evaluates the resulting Non-Uniform Discrete Hankel Transform using a compiled C extension (`_pynufht`). The transform runs in $O(N \log N)$ time and accepts arbitrary (non-uniform) target points $y$.

### General 2-D case — Non-Uniform Fast Fourier Transform

For a fully two-dimensional lensing potential $\psi(x_1, x_2)$ the oscillatory factor $\exp(-iw\mathbf{x}\cdot\mathbf{y})$ is a Fourier kernel. Separating the slowly varying part of the integrand,

$$\Large F(w, y) = \frac{w}{2\pi i} \sum_k \left[ w_k \exp\left(\frac{iw|\mathbf{x}_k|^2}{2}\right) \exp\left(-iw\,\psi(\mathbf{x}_k)\right) \right] \exp\left(-iw\,\mathbf{x}_k \cdot \mathbf{y}\right)$$

the bracketed coefficient $c_k(w)$ depends only on the quadrature node and the frequency, not on the target position. The sum over $k$ is then a **Non-Uniform DFT** (NUDFT) with non-uniform sources $\lbrace w\mathbf{x}_k \rbrace$ and targets $\lbrace \mathbf{y} \rbrace$, which FIONA delegates to FINUFFT:

- **Type-3** (`nufft2d3`): sources and targets both non-uniform — used for arbitrary source-plane positions.
- **Type-1** (`nufft2d1`): non-uniform sources mapped to a uniform output grid — used for dense source-plane grids.

Both variants run in $O(N \log N + M)$ time ($N$ quadrature nodes, $M$ target positions), enabling evaluation over large source-plane grids at negligible marginal cost.

---

## Installation

FIONA depends on [FINUFFT](https://finufft.readthedocs.io/) and optionally on a compiled Non-Uniform Fast Hankel Transform extension (`_pynufht`). Install the Python dependencies first:

```bash
pip install numpy scipy numexpr psutil finufft
```

Then install FIONA from source:

```bash
git clone https://github.com/ninoephremidze/FIONA.git
cd FIONA
pip install -e .
```

**Optional dependencies**

| Package | Purpose |
|---------|---------|
| `jax` | GPU-accelerated clumpy NFW lens (`JAXClumpyNFWLens`) |
| `numexpr` | Vectorised expression evaluation (significant speed-up) |
| `scipy` | FHT-based axisymmetric solver (`FresnelHankelAxisymmetricSciPy`) |

---

## Quick Start

### Axisymmetric lens (SIS)

```python
import numpy as np
import fiona

# 1. Create a Singular Isothermal Sphere lens
lens = fiona.SIS(psi0=1.0)

# 2. Set up the axisymmetric Fresnel solver
#    Precomputed GL nodes are stored in / read from FIONA_GL2D_DIR
import os
os.environ["FIONA_GL2D_DIR"] = "/tmp/fiona_gl_cache"

solver = fiona.FresnelNUFHT(lens, gl_nodes_per_dim=500)

# 3. Evaluate the diffraction integral at a range of frequencies w
w = np.geomspace(1.0, 100.0, 64)
F_w = solver(w)          # complex amplification factor F(w)
```

### General 2-D lens (SIS + external shear)

```python
import numpy as np
import fiona, os

os.environ["FIONA_GL2D_DIR"] = "/tmp/fiona_gl_cache"

lens = fiona.SISPlusExternalShear(psi0=1.0, gamma1=0.05, gamma2=0.0)

solver = fiona.FresnelNUFFT3(lens, n_gl=300, Umax=8.0)

w = np.array([10.0, 20.0, 50.0])
y = np.array([[0.3, 0.0]])   # source position (y1, y2)
F_w = solver(w, y)
```

### Thread control

```python
import fiona

# Use at most 4 threads across NumPy / FINUFFT / OpenBLAS / MKL
fiona.set_num_threads(4)
```

---

## Example Notebooks

The repository includes four Jupyter notebooks that demonstrate FIONA on progressively more complex lens systems.

### `validation_tests.ipynb` — FIONA vs GLoW

Validates FIONA against the independent **GLoW** reference implementation for two scenarios:

1. **Single SIS (axisymmetric)** — uses `FresnelNUFHT` (NUFHT solver) for three source positions ($y = 0.25, 0.5, 1.0$) and compares $|F(w)|$ curves.
2. **Composite 4×SIS (general 2-D)** — uses `FresnelNUFFT3` (NUFFT solver) for the same source positions and verifies agreement up to a global phase convention.

### `elliptical_sheared_lenses.ipynb` — Elliptical + Shear

Produces 2-D amplification maps `|F(w, y)|` on a 500×500 source-plane grid for:

1. **Elliptical Power-Law (EPL) + external shear** — sweeps the power-law slope $\gamma \in \lbrace 1.2, 1.7, 2.2 \rbrace$ at fixed ellipticity and shear, using both the NUFFT type-1 (uniform grid) and type-3 (scattered) back-ends.
2. **Elliptical NFW + external shear** — sweeps the NFW concentration $c_{200} \in \lbrace 15, 30, 60 \rbrace$ at fixed halo mass $10^{13}\,M_\odot$, redshift $z_L = 0.5$, $z_S = 1.5$. Physical parameters are converted to lenstronomy angular units via `LensCosmo`.

Each figure is a grid where rows correspond to the varied parameter and columns to the dimensionless frequency $w \in \lbrace 1, 10, 100 \rbrace$.

### `multi_component_lenses.ipynb` — Multi-Component Lenses

Demonstrates `FresnelNUFFT3` on three multi-sub-lens configurations:

1. **4 CIS sub-lenses in a symmetric cross** (`±offset` along each axis) — sweeps core radius `xc`, showing both 2-D magnification maps and `|F(w)|` vs `w` curves at fixed source positions.
2. **4 random SIS sub-lenses** — positions drawn uniformly, centred on their centre of mass; produces magnification maps and frequency curves.
3. **10 random SIS sub-lenses** — same setup with more sub-lenses, yielding a denser caustic network and richer interference pattern.

### `probing_dark_matter_subhalos.ipynb` — Dark Matter Subhalo Detection

Investigates the wave-optics signature of an NFW subhalo superimposed on an elliptical host lens:

- **Host lens**: elliptical NFW halo + external shear, physical parameters converted to angular units via `LensCosmo`.
- **Subhalo**: additional NFW component placed at $(1.0, 0.0)$ arcsec with masses $M_{\rm sub} \in \lbrace 10^9, 10^{10}, 10^{11} \rbrace\,M_\odot$ and concentrations at $c_{\rm CDM}$, $2c_{\rm CDM}$, $3c_{\rm CDM}$ (CDM concentration–mass relation).
- **Output**: $|F(w)|$ curves for the host-only and host+subhalo cases overlaid on the same axes, quantifying the detectability of the subhalo imprint in the GW frequency spectrum.

---

## Lens Models

All lens classes live in `fiona.lenses` and are re-exported from the top-level `fiona` namespace.

### Abstract base classes

| Class | Description |
|-------|-------------|
| `Lens` | Abstract base; requires `psi_xy(x1, x2)` |
| `AxisymmetricLens` | Subclass of `Lens`; requires `psi_r(r)` |

### Axisymmetric lenses

These implement `psi_r(r)` and are compatible with the fast `FresnelNUFHT` solver as well as the general `FresnelNUFFT3` solver.

| Class | Potential $\psi(r)$ | Key parameters |
|-------|----------------|----------------|
| `SIS` | $\psi_0 r$ | `psi0` |
| `PointLens` | $\frac{1}{2}\psi_0 \log(r^2 + x_c^2)$ (Plummer-softened point mass) | `psi0`, `xc` |
| `CIS` | $\psi_0\sqrt{x_c^2 + r^2} + x_c\psi_0\log\left[\frac{2x_c}{\sqrt{x_c^2+r^2}+x_c}\right]$ (reduces to SIS as $x_c \to 0$) | `psi0`, `xc` |
| `NFW` | $\frac{1}{2}\psi_0\left[\log^2\left(\frac{r}{2x_s}\right) + \left(\frac{r^2}{x_s^2}-1\right)F\left(\frac{r}{x_s}\right)^2\right]$ with analytic auxiliary $F$ | `psi0`, `xs` |

### Non-axisymmetric single-component lenses

These implement `psi_xy(x1, x2)` and require the general `FresnelNUFFT3` solver.

| Class | Potential $\psi(x_1, x_2)$ | Key parameters |
|-------|----------------------|----------------|
| `OffcenterNFW` | NFW evaluated at $r = \sqrt{(x_1-x_{c1})^2 + (x_2-x_{c2})^2}$ | `psi0`, `xs`, `xc1`, `xc2` |
| `SISPlusExternalShear` | $\psi_0 r + \frac{1}{2}\gamma_1(x_1^2-x_2^2) + \gamma_2 x_1 x_2$ | `psi0`, `gamma1`, `gamma2` |
| `PIED` | $\psi_0\sqrt{r_c^2 + x_1^2 + x_2^2/q^2}$ (pseudo-isothermal elliptical) | `psi0`, `q`, `r_core` |
| `EllipticalSIS` | $\psi_0\sqrt{x_1'^2 + (x_2'/q)^2}$ in rotated frame | `psi0`, `q`, `alpha`, `xc1`, `xc2` |
| `Shear` | $\frac{1}{2}\gamma_1(x_1^2-x_2^2) + \gamma_2 x_1 x_2$ (pure external shear, lenstronomy `(e1,e2)` parameterisation) | `gamma1`, `gamma2`, `ra_0`, `dec_0` |
| `SIE` | Singular Isothermal Ellipsoid (lenstronomy parameterisation) | `theta_E`, `e1`, `e2`, `center_x`, `center_y` |
| `EPL` | Elliptical Power Law with slope $\gamma$ (lenstronomy parameterisation) | `theta_E`, `gamma`, `e1`, `e2`, `center_x`, `center_y` |
| `NFW_ELLIPSE_POTENTIAL` | Elliptical NFW potential (lenstronomy parameterisation) | `Rs`, `alpha_Rs`, `e1`, `e2`, `center_x`, `center_y` |

### Multi-component / clumpy lenses

Vectorised lenses combining a host halo with an array of subhalos.

| Class | Description | Key parameters |
|-------|-------------|----------------|
| `ClumpySIELens` | NFW host + K elliptical SIS subhalos | `psi0_host`, `xs_host`, arrays of subhalo `psi0`, `q`, `alpha`, `xc1`, `xc2` |
| `ClumpyNFWLens` | NFW host + K off-center NFW subhalos | `psi0_host`, `xs_host`, arrays of subhalo `psi0`, `xs`, `xc1`, `xc2` |
| `JAXClumpyNFWLens` | GPU-accelerated clumpy NFW (requires JAX) | same as `ClumpyNFWLens` |

---

## Solvers

### `FresnelNUFHT` (axisymmetric)

Uses Gauss–Legendre quadrature in the radial direction combined with a Non-Uniform Fast Hankel Transform. Best choice for axisymmetric lenses.

```python
solver = fiona.FresnelNUFHT(
    lens,                          # AxisymmetricLens instance
    gl_nodes_per_dim=500,          # GL nodes (radial)
    min_physical_radius=None,      # fixed Umax; None → auto-adapt per w
    auto_R_from_gl_nodes=True,     # adapt integration radius to frequency
    tol=1e-12,                     # NUFHT tolerance
)
F_w = solver(w)                    # w: 1-D array of frequencies
```

**Aliases**: `FresnelHankelAxisymmetric` is a backward-compatible alias for `FresnelNUFHT`.

### `FresnelHankelAxisymmetricTrapezoidal` (axisymmetric)

Trapezoidal-rule variant — useful for benchmarking or when GL nodes are not available.

### `FresnelHankelAxisymmetricSciPy` (axisymmetric)

Wraps SciPy's built-in Fast Hankel Transform (`scipy.fft.fht`). Requires `scipy >= 1.4`.

### `FresnelNUFFT3` (general 2-D)

Handles arbitrary lens potentials using FINUFFT's 2-D type-3 (or type-1) non-uniform FFT. Frequencies are distributed across CPU cores for throughput.

```python
solver = fiona.FresnelNUFFT3(
    lens,               # any Lens instance
    n_gl=300,           # GL nodes per dimension
    Umax=8.0,           # integration half-extent
    nufft_tol=1e-9,     # NUFFT tolerance
)
F_w = solver(w, y)      # y: source position array, shape (..., 2)
```

---

## Configuration

FIONA reads the following environment variables at import time:

| Variable | Default | Description |
|----------|---------|-------------|
| `FIONA_GL2D_DIR` | `""` | Directory for cached Gauss–Legendre node files. Must be set before calling any solver that uses GL quadrature. |
| `FIONA_GL2D_STRICT` | `"0"` | If `"1"`, raise `FileNotFoundError` instead of computing GL nodes on the fly. |

Set these before importing FIONA, or use Python:

```python
import os
os.environ["FIONA_GL2D_DIR"] = "/path/to/cache"
import fiona
```

---

## Utilities

### `CPUTracker`

Context manager that measures effective CPU parallelism over a code block.

```python
with fiona.CPUTracker() as cpu:
    F_w = solver(w)

print(cpu.report("[solver]"))
# [solver] avg 7.84 cores over 2.341s (49.0% of 16 logical cores; CPU sec=18.354)
```

### `set_num_threads(n)`

Propagates the thread limit to OMP, OpenBLAS, MKL, and FINUFFT simultaneously.

```python
fiona.set_num_threads(8)
```

---

## Dependencies

| Package | Required | Minimum version |
|---------|----------|----------------|
| `numpy` | ✔ | 1.20 |
| `scipy` | ✔ | 1.4 |
| `numexpr` | ✔ | 2.7 |
| `psutil` | ✔ | 5.0 |
| `finufft` | ✔ | 2.1 |
| `jax` | optional | 0.4 |

---

## Author

**Nino Ephremidze**

FIONA version `0.1.1`