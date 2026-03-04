"""Tests for parallel Gauss–Legendre quadrature grid builders in fiona/utils.py.

Tests cover:
- Output shapes for 1D, 2D Cartesian, and both polar 2D builders.
- Basic weight invariants (weights positive for Cartesian and radial).
- Approximate integral correctness for simple functions.
- Cache files are created and are readable.
- Metadata fields ``cores_used`` and ``build_time_sec`` are present.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure FIONA_GL2D_DIR points to a per-test temp directory so tests are
# isolated and do not pollute each other's caches.
# ---------------------------------------------------------------------------

@pytest.fixture()
def gl_dir(tmp_path, monkeypatch):
    """Set FIONA_GL2D_DIR to a fresh temp directory for the duration of the test.

    ``fiona.utils`` reads ``FIONA_GL2D_DIR`` at module-load time into a
    module-level variable; we patch that variable directly so that all builder
    functions (including multiprocessing worker subprocesses that re-import the
    real ``fiona.utils`` module) see the correct temp directory.
    """
    monkeypatch.setenv("FIONA_GL2D_DIR", str(tmp_path))
    import fiona.utils as utils
    monkeypatch.setattr(utils, "_FIONA_GL2D_DIR", str(tmp_path))
    return tmp_path, utils


# ---------------------------------------------------------------------------
# 1D tests
# ---------------------------------------------------------------------------

class TestGL1D:
    def test_shape(self, gl_dir):
        tmp, utils = gl_dir
        n, u_max = 8, 3.0
        x, w = utils.gauss_legendre_1d(n, u_max, verbose=False)
        assert x.shape == (n,)
        assert w.shape == (n,)

    def test_files_created(self, gl_dir):
        tmp, utils = gl_dir
        n, u_max = 8, 3.0
        utils.gauss_legendre_1d(n, u_max, verbose=False)
        # locate the files by searching the tmp dir
        npy_files = list(tmp.glob("gl1d_*.x.npy"))
        assert len(npy_files) == 1
        meta_files = list(tmp.glob("gl1d_*.meta.json"))
        assert len(meta_files) == 1

    def test_metadata_fields(self, gl_dir):
        tmp, utils = gl_dir
        n, u_max = 8, 3.0
        utils.gauss_legendre_1d(n, u_max, verbose=False)
        meta_file = next(tmp.glob("gl1d_*.meta.json"))
        meta = json.loads(meta_file.read_text())
        assert "build_time_sec" in meta
        assert "cores_used" in meta
        assert meta["n"] == n
        assert meta["u_max"] == u_max

    def test_integral_constant(self, gl_dir):
        """Integral of 1 over [-u_max, u_max] should equal 2*u_max."""
        tmp, utils = gl_dir
        n, u_max = 10, 5.0
        x, w = utils.gauss_legendre_1d(n, u_max, verbose=False)
        assert abs(w.sum() - 2.0 * u_max) < 1e-10

    def test_nodes_in_range(self, gl_dir):
        tmp, utils = gl_dir
        n, u_max = 12, 2.0
        x, w = utils.gauss_legendre_1d(n, u_max, verbose=False)
        assert np.all(x > -u_max) and np.all(x < u_max)


# ---------------------------------------------------------------------------
# 2D Cartesian tests
# ---------------------------------------------------------------------------

class TestGL2D:
    def test_shape(self, gl_dir):
        tmp, utils = gl_dir
        n, u_max = 6, 2.0
        u1, u2, W = utils.gauss_legendre_2d(n, u_max, verbose=False)
        assert u1.shape == (n * n,)
        assert u2.shape == (n * n,)
        assert W.shape == (n * n,)

    def test_weights_positive(self, gl_dir):
        tmp, utils = gl_dir
        n, u_max = 6, 2.0
        _, _, W = utils.gauss_legendre_2d(n, u_max, verbose=False)
        assert np.all(W > 0)

    def test_files_created(self, gl_dir):
        tmp, utils = gl_dir
        n, u_max = 6, 2.0
        utils.gauss_legendre_2d(n, u_max, verbose=False)
        assert (tmp / f"gl2d_n{n}_U{int(u_max)}.u1.npy").exists()
        assert (tmp / f"gl2d_n{n}_U{int(u_max)}.u2.npy").exists()
        assert (tmp / f"gl2d_n{n}_U{int(u_max)}.W.npy").exists()
        assert (tmp / f"gl2d_n{n}_U{int(u_max)}.meta.json").exists()

    def test_metadata_fields(self, gl_dir):
        tmp, utils = gl_dir
        n, u_max = 6, 2.0
        utils.gauss_legendre_2d(n, u_max, verbose=False)
        meta = json.loads((tmp / f"gl2d_n{n}_U{int(u_max)}.meta.json").read_text())
        assert "build_time_sec" in meta
        assert "cores_used" in meta

    def test_integral_constant(self, gl_dir):
        """Integral of 1 over [-u_max, u_max]^2 should equal (2*u_max)^2."""
        tmp, utils = gl_dir
        n, u_max = 8, 2.0
        _, _, W = utils.gauss_legendre_2d(n, u_max, verbose=False)
        assert abs(W.sum() - (2.0 * u_max) ** 2) < 1e-10


# ---------------------------------------------------------------------------
# 2D Polar (GL theta) tests
# ---------------------------------------------------------------------------

class TestGL2DPolar:
    def test_shape(self, gl_dir):
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 6, 8, 3.0
        r, theta, W = utils.gauss_legendre_polar_2d(n_r, n_theta, u_max)
        assert r.shape == (n_r * n_theta,)
        assert theta.shape == (n_r * n_theta,)
        assert W.shape == (n_r * n_theta,)

    def test_radii_non_negative(self, gl_dir):
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 6, 8, 3.0
        r, _, _ = utils.gauss_legendre_polar_2d(n_r, n_theta, u_max)
        assert np.all(r >= 0)

    def test_theta_in_range(self, gl_dir):
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 6, 8, 3.0
        _, theta, _ = utils.gauss_legendre_polar_2d(n_r, n_theta, u_max)
        assert np.all(theta >= 0) and np.all(theta <= 2.0 * np.pi)

    def test_files_created(self, gl_dir):
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 6, 8, 3.0
        utils.gauss_legendre_polar_2d(n_r, n_theta, u_max)
        base = tmp / f"gl2dpolar_nr{n_r}_nt{n_theta}_U{int(u_max)}"
        assert base.with_suffix(".r.npy").exists()
        assert base.with_suffix(".theta.npy").exists()
        assert base.with_suffix(".W.npy").exists()
        assert base.with_suffix(".meta.json").exists()

    def test_metadata_fields(self, gl_dir):
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 6, 8, 3.0
        utils.gauss_legendre_polar_2d(n_r, n_theta, u_max)
        base = tmp / f"gl2dpolar_nr{n_r}_nt{n_theta}_U{int(u_max)}"
        meta = json.loads(base.with_suffix(".meta.json").read_text())
        assert "build_time_sec" in meta
        assert "cores_used" in meta
        assert meta["n_r"] == n_r
        assert meta["n_theta"] == n_theta

    def test_integral_constant(self, gl_dir):
        """Integral of 1 over a disk of radius u_max should equal π*u_max^2."""
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 10, 12, 3.0
        _, _, W = utils.gauss_legendre_polar_2d(n_r, n_theta, u_max)
        expected = np.pi * u_max ** 2
        assert abs(W.sum() - expected) < 1e-8


# ---------------------------------------------------------------------------
# 2D Polar (uniform theta) tests
# ---------------------------------------------------------------------------

class TestGL2DPolarUniformTheta:
    def test_shape(self, gl_dir):
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 6, 16, 3.0
        r, theta, W = utils.gauss_legendre_polar_uniform_theta_2d(n_r, n_theta, u_max)
        assert r.shape == (n_r * n_theta,)
        assert theta.shape == (n_r * n_theta,)
        assert W.shape == (n_r * n_theta,)

    def test_radii_non_negative(self, gl_dir):
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 6, 16, 3.0
        r, _, _ = utils.gauss_legendre_polar_uniform_theta_2d(n_r, n_theta, u_max)
        assert np.all(r >= 0)

    def test_theta_in_range(self, gl_dir):
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 6, 16, 3.0
        _, theta, _ = utils.gauss_legendre_polar_uniform_theta_2d(n_r, n_theta, u_max)
        # Uniform theta: values are 2π*k/n_theta for k in [0, n_theta-1],
        # so strictly less than 2π.
        assert np.all(theta >= 0) and np.all(theta < 2.0 * np.pi)

    def test_files_created(self, gl_dir):
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 6, 16, 3.0
        utils.gauss_legendre_polar_uniform_theta_2d(n_r, n_theta, u_max)
        base = tmp / f"gl2dpolarU_nr{n_r}_nt{n_theta}_U{int(u_max)}"
        assert base.with_suffix(".r.npy").exists()
        assert base.with_suffix(".theta.npy").exists()
        assert base.with_suffix(".W.npy").exists()
        assert base.with_suffix(".meta.json").exists()

    def test_metadata_fields(self, gl_dir):
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 6, 16, 3.0
        utils.gauss_legendre_polar_uniform_theta_2d(n_r, n_theta, u_max)
        base = tmp / f"gl2dpolarU_nr{n_r}_nt{n_theta}_U{int(u_max)}"
        meta = json.loads(base.with_suffix(".meta.json").read_text())
        assert "build_time_sec" in meta
        assert "cores_used" in meta
        assert meta.get("theta_rule") == "uniform"

    def test_integral_constant(self, gl_dir):
        """Integral of 1 over a disk of radius u_max should equal π*u_max^2."""
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 10, 32, 3.0
        _, _, W = utils.gauss_legendre_polar_uniform_theta_2d(n_r, n_theta, u_max)
        expected = np.pi * u_max ** 2
        assert abs(W.sum() - expected) < 1e-8

    def test_uniform_theta_spacing(self, gl_dir):
        """Theta values for a given radial index should be evenly spaced."""
        tmp, utils = gl_dir
        n_r, n_theta, u_max = 4, 8, 2.0
        _, theta, _ = utils.gauss_legendre_polar_uniform_theta_2d(n_r, n_theta, u_max)
        # First n_theta values correspond to the first radial point
        diffs = np.diff(theta[:n_theta])
        expected_step = 2.0 * np.pi / n_theta
        assert np.allclose(diffs, expected_step, atol=1e-12)
