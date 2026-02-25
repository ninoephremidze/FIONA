####################################################################
# fiona/lenses.py
####################################################################

import numpy as np
from abc import ABC, abstractmethod

try:
    import numexpr as ne
    _HAS_NUMEXPR = True
except Exception:
    _HAS_NUMEXPR = False

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)  # we want float64 for lensing
    _HAS_JAX = True
except Exception:
    _HAS_JAX = False


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------

def _rotate(x, y, phi):
    """
    lenstronomy.Util.util.rotate convention:
      x_rot =  cos(phi)*x + sin(phi)*y
      y_rot = -sin(phi)*x + cos(phi)*y
    """
    c = np.cos(phi)
    s = np.sin(phi)
    return c * x + s * y, -s * x + c * y


def ellipticity2phi_q(e1, e2):
    """
    lenstronomy.Util.param_util.ellipticity2phi_q:
      phi = 0.5 * arctan2(e2, e1)
      c   = sqrt(e1^2 + e2^2), clipped to < 1
      q   = (1 - c) / (1 + c)
    """
    phi = np.arctan2(e2, e1) / 2.0
    c = np.sqrt(e1 * e1 + e2 * e2)
    c = np.minimum(c, 0.9999)
    q = (1.0 - c) / (1.0 + c)
    return phi, q


def q2e(q):
    """
    lenstronomy.Util.param_util.q2e:
      e = |1 - q^2| / (1 + q^2)
    """
    return np.abs(1.0 - q * q) / (1.0 + q * q)


def transform_e1e2_square_average(x, y, e1, e2, center_x, center_y):
    """
    lenstronomy.Util.param_util.transform_e1e2_square_average:
      Maps (x,y) -> (x_,y_) such that the *spherical* profile evaluated at
      R_ = sqrt(x_^2 + y_^2) yields an elliptical potential.

      Steps:
        phi_g, q = ellipticity2phi_q(e1,e2)
        e = q2e(q)
        rotate into principal axes, then scale:
          x_ = (cos*dx + sin*dy)*sqrt(1-e)
          y_ = (-sin*dx + cos*dy)*sqrt(1+e)
    """
    phi_g, q = ellipticity2phi_q(e1, e2)
    dx = x - center_x
    dy = y - center_y
    c = np.cos(phi_g)
    s = np.sin(phi_g)
    e = q2e(q)
    x_ = (c * dx + s * dy) * np.sqrt(1.0 - e)
    y_ = (-s * dx + c * dy) * np.sqrt(1.0 + e)
    return x_, y_


# ----------------------------------------------------------------------
# Base classes
# ----------------------------------------------------------------------

class Lens(ABC):
    """Abstract base for a projected lensing potential ψ(x1,x2)."""

    @abstractmethod
    def psi_xy(self, x1, x2):
        """Return ψ(x) on R^2 (broadcast over arrays)."""
        ...


class AxisymmetricLens(Lens):
    """Axisymmetric lenses: ψ(x) = ψ(r), with r = sqrt(x1^2 + x2^2)."""

    @abstractmethod
    def psi_r(self, r):
        """Return ψ(r) for radial distance r (broadcast over arrays)."""
        ...

    def psi_xy(self, x1, x2):
        """
        Compute ψ(x1,x2) by first forming r = sqrt(x1^2 + x2^2)
        and then calling psi_r(r). Uses numexpr if available.
        """
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        if _HAS_NUMEXPR:
            x1b, x2b = np.broadcast_arrays(x1, x2)
            r = ne.evaluate("sqrt(x1b*x1b + x2b*x2b)")
        else:
            r = np.hypot(x1, x2)

        return self.psi_r(r)


# ----------------------------------------------------------------------
# Simple axisymmetric lenses
# ----------------------------------------------------------------------

class SIS(AxisymmetricLens):
    r"""Singular Isothermal Sphere: ψ(r) = ψ0 * r."""
    def __init__(self, psi0=1.0):
        self.psi0 = float(psi0)

    def psi_r(self, r):
        r = np.asarray(r, dtype=float)
        psi0 = self.psi0
        if _HAS_NUMEXPR:
            return ne.evaluate("psi0 * r")
        else:
            return psi0 * r


class PointLens(AxisymmetricLens):
    r"""Point lens with Plummer softening: ψ(r) = 0.5 ψ0 log(r^2 + x_c^2)."""
    def __init__(self, psi0=1.0, xc=0.0):
        self.psi0 = float(psi0)
        self.xc   = float(xc)

    def psi_r(self, r):
        r = np.asarray(r, dtype=float)
        psi0 = self.psi0
        xc2  = self.xc * self.xc
        if _HAS_NUMEXPR:
            return ne.evaluate("0.5 * psi0 * log(r*r + xc2)")
        else:
            return 0.5 * psi0 * np.log(r*r + xc2)


class CIS(AxisymmetricLens):
    r"""Cored Isothermal Sphere (CIS) / softened isothermal sphere.

    Lensing potential (dimensionless) implemented:

        ψ(r) = ψ0 * sqrt(xc^2 + r^2)
               + xc * ψ0 * log( 2*xc / ( sqrt(xc^2 + r^2) + xc ) )

    where ``xc`` is the dimensionless core radius (r_c / ξ0) and ``psi0`` is the
    potential normalization. In the limit ``xc -> 0`` this reduces to the
    Singular Isothermal Sphere (SIS): ψ(r) = ψ0 * r.

    Notes:
    - The class accepts scalar or array-like `r`.
    - For xc == 0 the implementation falls back to the SIS formula for exact
      behaviour and to avoid division-by-zero / log-of-one issues.
    """
    def __init__(self, psi0: float = 1.0, xc: float = 0.0):
        self.psi0 = float(psi0)
        self.xc = float(xc)

    def psi_r(self, r):
        # Accept scalar or array-like r.
        r = np.asarray(r, dtype=float)
        psi0 = self.psi0
        xc = float(self.xc)

        # If core radius is zero (or extremely small), fall back to SIS behaviour.
        if xc == 0.0:
            if _HAS_NUMEXPR:
                return ne.evaluate("psi0 * r")
            else:
                return psi0 * r

        # Compute s = sqrt(r^2 + xc^2)
        s = np.sqrt(r * r + xc * xc)

        # log term: log( (2*xc) / (s + xc) )
        # Evaluate with numpy for stability (numexpr can't mix Python scalars easily).
        logterm = np.log((2.0 * xc) / (s + xc))

        return psi0 * s + xc * psi0 * logterm


# ----------------------------------------------------------------------
# NFW + off-center NFW
# ----------------------------------------------------------------------

class NFW(AxisymmetricLens):
    r"""
    Axisymmetric Navarro–Frenk–White (NFW) lens.

    Follows the analytic form used in GLoW (up to small-u series approximations):

        u = r / x_s

        F(u) = 1/sqrt(u^2 - 1) * arctan(sqrt(u^2 - 1))          (u > 1)
             = 1/sqrt(1 - u^2) * atanh(sqrt(1 - u^2))          (u < 1)
             = 1                                              (u = 1)

        ψ(r) = 0.5 * ψ0 * [ log^2(u/2) + (u^2 - 1) * F(u)^2 ].
    """
    def __init__(self, psi0=1.0, xs=0.1):
        self.psi0 = float(psi0)
        self.xs   = float(xs)

    @staticmethod
    def _F_nfw(u):
        """
        Compute the NFW auxiliary function F(u):
          F(u) = arctan(sqrt(u^2 - 1)) / sqrt(u^2 - 1)   (u > 1)
               = arctanh(sqrt(1 - u^2)) / sqrt(1 - u^2)  (u < 1)
               = 1                                        (u = 1)
        """
        u = np.asarray(u, dtype=float)
        out = np.empty_like(u)

        gt1 = u > 1.0
        lt1 = u < 1.0
        eq1 = ~(gt1 | lt1)

        if np.any(gt1):
            ug = u[gt1]
            s = np.sqrt(ug*ug - 1.0)
            out[gt1] = np.arctan(s) / s

        if np.any(lt1):
            ul = u[lt1]
            s = np.sqrt(1.0 - ul*ul)
            out[lt1] = np.arctanh(s) / s

        if np.any(eq1):
            out[eq1] = 1.0

        return out

    def psi_r(self, r):
        r = np.asarray(r, dtype=float)
        psi0 = self.psi0
        xs   = self.xs

        u = r / xs
        F = self._F_nfw(u)

        if _HAS_NUMEXPR:
            return ne.evaluate(
                "0.5 * psi0 * (log(u/2.0)**2 + (u*u - 1.0) * F*F)"
            )
        else:
            log_term = np.log(u / 2.0)
            return 0.5 * psi0 * (log_term*log_term + (u*u - 1.0)*F*F)


class OffcenterNFW(Lens):
    r"""
    Off-center NFW lens:

        ψ(x1, x2) = ψ_NFW( sqrt((x1 - xc1)^2 + (x2 - xc2)^2) ).

    Parameters match GLoW's Psi_offcenterNFW (psi0, xs, xc1, xc2).
    """
    def __init__(self, psi0=1.0, xs=0.1, xc1=0.0, xc2=0.0):
        self.psi0 = float(psi0)
        self.xs   = float(xs)
        self.xc1  = float(xc1)
        self.xc2  = float(xc2)
        self._nfw = NFW(psi0=self.psi0, xs=self.xs)

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)
        dx1 = x1 - self.xc1
        dx2 = x2 - self.xc2

        if _HAS_NUMEXPR:
            r = ne.evaluate("sqrt(dx1*dx1 + dx2*dx2)")
        else:
            r = np.hypot(dx1, dx2)

        return self._nfw.psi_r(r)


# ----------------------------------------------------------------------
# Non-axisymmetric single-lens models
# ----------------------------------------------------------------------

class SISPlusExternalShear(Lens):
    r"""
    SIS + external shear:

        ψ(x1, x2) = ψ0 * sqrt(x1^2 + x2^2)
                    + 0.5 * γ1 * (x1^2 - x2^2)
                    + γ2 * x1 * x2

    γ1, γ2 are the usual Cartesian shear components.  When (γ1, γ2) = (0, 0)
    this reduces to a pure SIS.
    """
    def __init__(self, psi0=1.0, gamma1=0.1, gamma2=0.0):
        self.psi0   = float(psi0)
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        psi0 = self.psi0
        g1   = self.gamma1
        g2   = self.gamma2

        if _HAS_NUMEXPR:
            return ne.evaluate(
                "psi0 * sqrt(x1*x1 + x2*x2)"
                " + 0.5 * g1 * (x1*x1 - x2*x2)"
                " + g2 * x1 * x2"
            )
        else:
            r = np.hypot(x1, x2)
            return psi0 * r + 0.5 * g1 * (x1*x1 - x2*x2) + g2 * x1 * x2


class PIED(Lens):
    r"""
    Elliptical pseudo-isothermal potential (toy model):

        ψ(x1, x2) = ψ0 * sqrt(r_c^2 + x1^2 + x2^2 / q^2)

    where q is the axis ratio (q < 1: flattened along x2; q > 1: flattened along x1)
    and r_c is a core radius to keep the center finite.
    """
    def __init__(self, psi0=1.0, q=0.7, r_core=0.05):
        if q <= 0:
            raise ValueError("Axis ratio q must be > 0.")
        self.psi0   = float(psi0)
        self.q      = float(q)
        self.r_core = float(r_core)

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        psi0 = self.psi0
        q    = self.q
        rc2  = self.r_core * self.r_core

        if _HAS_NUMEXPR:
            return ne.evaluate("psi0 * sqrt(rc2 + x1*x1 + (x2*x2) / (q*q))")
        else:
            return psi0 * np.sqrt(rc2 + x1*x1 + (x2*x2) / (q*q))


class EllipticalSIS(Lens):
    r"""
    Elliptical SIS (pseudo-elliptical potential):

        Define rotated coordinates (x1', x2') around center (xc1, xc2):

            dx1 = x1 - xc1
            dx2 = x2 - xc2
            x1' =  cosα * dx1 + sinα * dx2
            x2' = -sinα * dx1 + cosα * dx2

        Elliptical radius:
            R = sqrt( x1'^2 + (x2'/q)^2 )

        Potential:
            ψ(x1, x2) = ψ0 * R

        For q = 1, this reduces to a circular SIS with ψ = ψ0 * r.
    """
    def __init__(self, psi0=1.0, q=0.7, alpha=0.0, xc1=0.0, xc2=0.0):
        if q <= 0:
            raise ValueError("Axis ratio q must be > 0.")
        self.psi0  = float(psi0)
        self.q     = float(q)
        self.alpha = float(alpha)
        self.xc1   = float(xc1)
        self.xc2   = float(xc2)

        self._cos = np.cos(self.alpha)
        self._sin = np.sin(self.alpha)

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        dx1 = x1 - self.xc1
        dx2 = x2 - self.xc2

        c = self._cos
        s = self._sin
        q = self.q
        psi0 = self.psi0

        if _HAS_NUMEXPR:
            x1p = ne.evaluate("dx1*c + dx2*s")
            x2p = ne.evaluate("-dx1*s + dx2*c")
            R   = ne.evaluate("sqrt(x1p*x1p + (x2p/q)*(x2p/q))")
            return ne.evaluate("psi0 * R")
        else:
            x1p = c*dx1 + s*dx2
            x2p = -s*dx1 + c*dx2
            R = np.sqrt(x1p*x1p + (x2p/q)**2)
            return psi0 * R


# ----------------------------------------------------------------------
# Vectorized clumpy lens: host NFW + many elliptical SIS subhalos
# ----------------------------------------------------------------------

class ClumpySIELens(Lens):
    r"""
    Vectorized "clumpy" lens consisting of:
      - One axisymmetric NFW host centered at the origin.
      - K elliptical SIS (eSIS) subhalos.
    """
    def __init__(
        self,
        psi0_host,
        xs_host,
        psi0_sub,
        q_sub,
        alpha_sub,
        xc1_sub,
        xc2_sub,
    ):
        self.host = NFW(psi0=psi0_host, xs=xs_host)

        psi0_sub  = np.asarray(psi0_sub,  dtype=float)
        q_sub     = np.asarray(q_sub,     dtype=float)
        alpha_sub = np.asarray(alpha_sub, dtype=float)
        xc1_sub   = np.asarray(xc1_sub,   dtype=float)
        xc2_sub   = np.asarray(xc2_sub,   dtype=float)

        if not (psi0_sub.shape == q_sub.shape == alpha_sub.shape == xc1_sub.shape == xc2_sub.shape):
            raise ValueError("All subhalo parameter arrays must have the same shape.")
        if np.any(q_sub <= 0):
            raise ValueError("All subhalo axis ratios q must be > 0.")

        self.psi0_sub  = psi0_sub
        self.q_sub     = q_sub
        self.alpha_sub = alpha_sub
        self.xc1_sub   = xc1_sub
        self.xc2_sub   = xc2_sub

        self.cos_alpha = np.cos(alpha_sub)
        self.sin_alpha = np.sin(alpha_sub)

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        if _HAS_NUMEXPR:
            x1b, x2b = np.broadcast_arrays(x1, x2)
            r_host = ne.evaluate("sqrt(x1b*x1b + x2b*x2b)")
        else:
            r_host = np.hypot(x1, x2)
        psi_host = self.host.psi_r(r_host)

        dx1 = x1[..., None] - self.xc1_sub
        dx2 = x2[..., None] - self.xc2_sub

        c = self.cos_alpha
        s = self.sin_alpha
        q = self.q_sub
        psi0 = self.psi0_sub

        if _HAS_NUMEXPR:
            x1p = ne.evaluate("dx1*c + dx2*s")
            x2p = ne.evaluate("-dx1*s + dx2*c")
            R   = ne.evaluate("sqrt(x1p*x1p + (x2p/q)*(x2p/q))")
            psi_sub = ne.evaluate("psi0 * R")
        else:
            x1p = dx1*c + dx2*s
            x2p = -dx1*s + dx2*c
            R   = np.sqrt(x1p*x1p + (x2p/q)**2)
            psi_sub = psi0 * R

        psi_sub_total = np.sum(psi_sub, axis=-1)
        return psi_host + psi_sub_total


class ClumpyNFWLens(Lens):
    """
    Host NFW at the origin + many off-center NFW subhalos, evaluated
    in a single vectorized pass.
    """
    def __init__(self, psi0_host, xs_host, psi0_sub, xs_sub, xc1_sub, xc2_sub):
        self.host = NFW(psi0=psi0_host, xs=xs_host)

        psi0_sub = np.asarray(psi0_sub, dtype=float)
        xs_sub   = np.asarray(xs_sub,   dtype=float)
        xc1_sub  = np.asarray(xc1_sub,  dtype=float)
        xc2_sub  = np.asarray(xc2_sub,  dtype=float)

        if not (psi0_sub.shape == xs_sub.shape == xc1_sub.shape == xc2_sub.shape):
            raise ValueError("Subhalo parameter arrays must all have the same shape.")

        self.psi0_sub = psi0_sub
        self.xs_sub   = xs_sub
        self.xc1_sub  = xc1_sub
        self.xc2_sub  = xc2_sub

    def psi_xy(self, x1, x2):
        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        r_host = np.hypot(x1, x2)
        psi_host = self.host.psi_r(r_host)

        dx1 = x1[..., None] - self.xc1_sub
        dx2 = x2[..., None] - self.xc2_sub
        r   = np.hypot(dx1, dx2)

        u = r / self.xs_sub
        u_flat = u.ravel()
        F_flat = NFW._F_nfw(u_flat)
        F = F_flat.reshape(u.shape)

        log_term = np.log(u / 2.0)
        psi_sub = 0.5 * self.psi0_sub * (log_term*log_term + (u*u - 1.0)*F*F)

        psi_sub_total = np.sum(psi_sub, axis=-1)
        return psi_host + psi_sub_total


# ----------------------------------------------------------------------
# JAX helpers + JAX clumpy NFW lens
# ----------------------------------------------------------------------

if _HAS_JAX:
    def _jax_F_nfw(u):
        u = jnp.asarray(u, dtype=jnp.float64)
        gt1 = u > 1.0
        lt1 = u < 1.0

        def branch_gt1(u_):
            s = jnp.sqrt(u_*u_ - 1.0)
            return jnp.arctan(s) / s

        def branch_lt1(u_):
            s = jnp.sqrt(1.0 - u_*u_)
            return jnp.arctanh(s) / s

        F_gt = branch_gt1(u)
        F_lt = branch_lt1(u)

        out = jnp.where(gt1, F_gt, jnp.where(lt1, F_lt, 1.0))
        return out

    def _jax_nfw_potential(r, psi0, xs):
        r = jnp.asarray(r, dtype=jnp.float64)
        psi0 = jnp.asarray(psi0, dtype=jnp.float64)
        xs   = jnp.asarray(xs,   dtype=jnp.float64)

        u = r / xs
        F = _jax_F_nfw(u)
        log_term = jnp.log(u / 2.0)
        return 0.5 * psi0 * (log_term*log_term + (u*u - 1.0) * F * F)


class JAXClumpyNFWLens(Lens):
    """
    Host NFW at the origin + many off-center NFW subhalos,
    evaluated by a single JAX-jitted kernel.
    """
    def __init__(self, psi0_host, xs_host, psi0_sub, xs_sub, xc1_sub, xc2_sub):
        if not _HAS_JAX:
            raise ImportError("JAX is not available; cannot use JAXClumpyNFWLens.")

        self.psi0_host = float(psi0_host)
        self.xs_host   = float(xs_host)

        psi0_sub = np.asarray(psi0_sub, dtype=float)
        xs_sub   = np.asarray(xs_sub,   dtype=float)
        xc1_sub  = np.asarray(xc1_sub,  dtype=float)
        xc2_sub  = np.asarray(xc2_sub,  dtype=float)

        if not (psi0_sub.shape == xs_sub.shape == xc1_sub.shape == xc2_sub.shape):
            raise ValueError("Subhalo parameter arrays must all have the same shape.")

        self.psi0_sub = psi0_sub
        self.xs_sub   = xs_sub
        self.xc1_sub  = xc1_sub
        self.xc2_sub  = xc2_sub

        self._psi0_host_j = jnp.asarray(self.psi0_host, dtype=jnp.float64)
        self._xs_host_j   = jnp.asarray(self.xs_host,   dtype=jnp.float64)
        self._psi0_sub_j  = jnp.asarray(self.psi0_sub, dtype=jnp.float64)
        self._xs_sub_j    = jnp.asarray(self.xs_sub,   dtype=jnp.float64)
        self._xc1_sub_j   = jnp.asarray(self.xc1_sub,  dtype=jnp.float64)
        self._xc2_sub_j   = jnp.asarray(self.xc2_sub,  dtype=jnp.float64)

        self._psi_kernel = jax.jit(self._psi_kernel_fn)

    def _psi_kernel_fn(self, x1, x2):
        r_host = jnp.sqrt(x1*x1 + x2*x2)
        psi_host = _jax_nfw_potential(r_host, self._psi0_host_j, self._xs_host_j)

        dx1 = x1[..., None] - self._xc1_sub_j
        dx2 = x2[..., None] - self._xc2_sub_j
        r   = jnp.sqrt(dx1*dx1 + dx2*dx2)

        psi_sub = _jax_nfw_potential(r, self._psi0_sub_j, self._xs_sub_j)
        psi_sub_total = jnp.sum(psi_sub, axis=-1)
        return psi_host + psi_sub_total

    def psi_xy(self, x1, x2):
        if not _HAS_JAX:
            raise ImportError("JAX is not available; cannot use JAXClumpyNFWLens.")

        x1 = np.asarray(x1, dtype=float)
        x2 = np.asarray(x2, dtype=float)

        x1_j = jnp.asarray(x1, dtype=jnp.float64)
        x2_j = jnp.asarray(x2, dtype=jnp.float64)

        psi_j = self._psi_kernel(x1_j, x2_j)
        return np.asarray(psi_j)


# ----------------------------------------------------------------------
# Lenstronomy-style non-axisymmetric potentials
# ----------------------------------------------------------------------

class Shear(Lens):
    r"""
    lenstronomy 'SHEAR' potential:

        ψ(x, y) = 1/2 * γ1 * (x - ra_0)^2 - 1/2 * γ1 * (y - dec_0)^2
                  + γ2 * (x - ra_0) * (y - dec_0)

    Parameters:
      gamma1, gamma2 : Cartesian shear components
      ra_0, dec_0    : reference point where shear deflection is 0
    """
    def __init__(self, gamma1=0.0, gamma2=0.0, ra_0=0.0, dec_0=0.0):
        self.gamma1 = float(gamma1)
        self.gamma2 = float(gamma2)
        self.ra_0 = float(ra_0)
        self.dec_0 = float(dec_0)

    def psi_xy(self, x1, x2):
        x = np.asarray(x1, dtype=float) - self.ra_0
        y = np.asarray(x2, dtype=float) - self.dec_0
        g1 = self.gamma1
        g2 = self.gamma2
        return 0.5 * g1 * (x * x - y * y) + g2 * x * y

class SIE(Lens):
    r"""
    lenstronomy 'SIE' implemented via the same *elliptical potential* mapping used
    in lenstronomy's NIE_POTENTIAL limiting case (tiny core).

    Input parameters match lenstronomy:
      theta_E, e1, e2, center_x, center_y
    plus:
      s_scale : small core/softening (lenstronomy uses ~1e-10 internally)
    """
    def __init__(self, theta_E=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0, s_scale=1e-10):
        self.theta_E = float(theta_E)
        self.e1 = float(e1)
        self.e2 = float(e2)
        self.center_x = float(center_x)
        self.center_y = float(center_y)
        self.s_scale = float(s_scale)

    @staticmethod
    def _theta_q_convert(theta, q):
        # from lenstronomy NIE_POTENTIAL._theta_q_convert
        return theta / (np.sqrt((1.0 + q**2) / (2.0 * q)))

    def psi_xy(self, x1, x2):
        x = np.asarray(x1, dtype=float)
        y = np.asarray(x2, dtype=float)

        # lenstronomy: eps = sqrt(e1^2 + e2^2), phi_G, q from ellipticity2phi_q
        eps = np.sqrt(self.e1 * self.e1 + self.e2 * self.e2)
        phi_G, q = ellipticity2phi_q(self.e1, self.e2)

        theta_E_conv = self._theta_q_convert(self.theta_E, q)
        theta_c_conv = self._theta_q_convert(self.s_scale, q)

        # shift and rotate
        x_shift = x - self.center_x
        y_shift = y - self.center_y
        x_rot, y_rot = _rotate(x_shift, y_shift, phi_G)

        # NIEPotentialMajorAxis.function (lenstronomy):
        #   ψ = theta_E * sqrt(theta_c^2 + (1-eps)x^2 + (1+eps)y^2)
        return theta_E_conv * np.sqrt(theta_c_conv**2 + (1.0 - eps) * x_rot**2 + (1.0 + eps) * y_rot**2)


class EPL(Lens):
    r"""
    Elliptical Power-law *Potential* in the lenstronomy SPEP convention.

    This matches the lenstronomy 'SPEP' potential (softening s=0), which is the
    potential model commonly paired with 'SHEAR' in lenstronomy examples.

    Parameters (lenstronomy SPEP):
      theta_E, gamma, e1, e2, center_x, center_y

    Potential:
      ψ(x,y) = 2 E^2 / η^2 * ((p^2)/E^2)^(η/2)
      η = 3 - gamma
      p^2 = x_t^2 + y_t^2 / q^2    (coordinates aligned with principal axes)
      E = (theta_E*q) / [ ((3-gamma)/2)^(1/(1-gamma)) * sqrt(q) ]

    Notes:
      - lenstronomy bounds gamma into [1.4, 2.9] and q>=0.01.
      - Uses q derived from (e1,e2) via ellipticity2phi_q (complex ellipticity).
    """
    def __init__(self, theta_E=1.0, gamma=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0):
        self.theta_E = float(theta_E)
        self.gamma = float(gamma)
        self.e1 = float(e1)
        self.e2 = float(e2)
        self.center_x = float(center_x)
        self.center_y = float(center_y)

    @staticmethod
    def _param_bounds(gamma, q):
        # lenstronomy SPEP._param_bounds
        if gamma < 1.4:
            gamma = 1.4
        if gamma > 2.9:
            gamma = 2.9
        if q < 0.01:
            q = 0.01
        return float(gamma), q

    def psi_xy(self, x1, x2):
        x = np.asarray(x1, dtype=float)
        y = np.asarray(x2, dtype=float)

        phi_G, q = ellipticity2phi_q(self.e1, self.e2)
        gamma, q = self._param_bounds(self.gamma, q)

        # lenstronomy SPEP: theta_E *= q
        theta_E_eff = self.theta_E * q

        # shift and rotate into principal axes
        x_shift = x - self.center_x
        y_shift = y - self.center_y
        xt1 = np.cos(phi_G) * x_shift + np.sin(phi_G) * y_shift
        xt2 = -np.sin(phi_G) * x_shift + np.cos(phi_G) * y_shift

        eta = 3.0 - gamma
        # E normalization (lenstronomy SPEP)
        denom = (((3.0 - gamma) / 2.0) ** (1.0 / (1.0 - gamma))) * np.sqrt(q)
        E = theta_E_eff / denom

        # p^2
        p2 = xt1 * xt1 + (xt2 * xt2) / (q * q)

        # avoid p2=0 singular issues for fractional powers
        p2 = np.maximum(p2, 1e-12)

        return (2.0 * E * E / (eta * eta)) * ((p2 / (E * E)) ** (eta / 2.0))


class NFW_ELLIPSE_POTENTIAL(Lens):
    r"""
    lenstronomy 'NFW_ELLIPSE_POTENTIAL' model.

    This is an *elliptical potential* construction:
      1) apply transform_e1e2_square_average -> (x_, y_)
      2) evaluate spherical NFW potential at R_ = sqrt(x_^2 + y_^2)

    Parameters (lenstronomy):
      Rs       : scale radius
      alpha_Rs : deflection angle at Rs (normalization)
      e1, e2   : complex ellipticity components
      center_x, center_y

    Spherical NFW potential in lenstronomy is:
      ρ0 = alpha_Rs / [4 Rs^2 (1 + ln(1/2))]
      X  = R/Rs
      ψ(R) = 2 ρ0 Rs^3 h(X)
    where
      h(X) = ln(X/2)^2 - arccosh(1/X)^2   (X < 1)
           = ln(X/2)^2 + arccos(1/X)^2    (X >= 1)
    """
    def __init__(self, Rs=1.0, alpha_Rs=1.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0):
        self.Rs = float(Rs)
        self.alpha_Rs = float(alpha_Rs)
        self.e1 = float(e1)
        self.e2 = float(e2)
        self.center_x = float(center_x)
        self.center_y = float(center_y)

    @staticmethod
    def alpha2rho0(alpha_Rs, Rs):
        # lenstronomy NFW.alpha2rho0
        return alpha_Rs / (4.0 * Rs**2 * (1.0 + np.log(1.0 / 2.0)))

    @staticmethod
    def _h(X):
        # lenstronomy NFW._h (vectorized)
        X = np.asarray(X, dtype=float)
        c = 1e-6
        if np.isscalar(X):
            x = max(c, X)
            if x < 1.0:
                return np.log(x / 2.0) ** 2 - np.arccosh(1.0 / x) ** 2
            else:
                return np.log(x / 2.0) ** 2 + np.arccos(1.0 / x) ** 2

        out = np.empty_like(X)
        Xc = X.copy()
        Xc[Xc <= c] = c

        m = Xc < 1.0
        out[m] = np.log(Xc[m] / 2.0) ** 2 - np.arccosh(1.0 / Xc[m]) ** 2

        mp = ~m
        out[mp] = np.log(Xc[mp] / 2.0) ** 2 + np.arccos(1.0 / Xc[mp]) ** 2
        return out

    def psi_xy(self, x1, x2):
        x = np.asarray(x1, dtype=float)
        y = np.asarray(x2, dtype=float)

        # elliptical potential mapping
        x_, y_ = transform_e1e2_square_average(
            x, y, self.e1, self.e2, self.center_x, self.center_y
        )
        R = np.hypot(x_, y_)
        X = R / self.Rs

        rho0 = self.alpha2rho0(self.alpha_Rs, self.Rs)
        return 2.0 * rho0 * (self.Rs ** 3) * self._h(X)
