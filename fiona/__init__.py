"""
FIONA: Fresnel Integral Optimization via Non-uniform trAnsforms.

This package provides:
- Gravitational lens models (axisymmetric and non-axisymmetric).
- Efficient Fresnel integral solvers backed by NUFFTs and Hankel transforms.
- Utility helpers for quadrature, timing, and CPU tracking.

Author: Nino Ephremidze
"""
import os

# Lenses (base + originals + new non-axisymmetric)
from .lenses import (
    Lens,
    AxisymmetricLens,
    SIS,
    PointLens,
    CIS,
    NFW,
    OffcenterNFW,
    SISPlusExternalShear,
    PIED,
    EllipticalSIS,
    ClumpySIELens,
    ClumpyNFWLens,
    JAXClumpyNFWLens,
    # New (lenstronomy-style)
    Shear,
    SIE,
    EPL,
    NFW_ELLIPSE_POTENTIAL,
)

from .axisym import (
    FresnelNUFHT,
    FresnelNUFHTBatched,
    FresnelHankelAxisymmetricTrapezoidal,
    FresnelHankelAxisymmetricSciPy,
)
from .general import FresnelNUFFT3
from .utils import CPUTracker

# Backward compatibility alias
FresnelHankelAxisymmetric = FresnelNUFHT

__all__ = [
    # Base lenses
    "Lens",
    "AxisymmetricLens",

    # Original simple lenses
    "SIS",
    "PointLens",
    "CIS",

    # Original NFW family
    "NFW",
    "OffcenterNFW",

    # Original non-axisymmetric toys
    "SISPlusExternalShear",
    "PIED",
    "EllipticalSIS",

    # Original clumpy lenses
    "ClumpySIELens",
    "ClumpyNFWLens",
    "JAXClumpyNFWLens",

    # New lenstronomy-style lenses
    "Shear",
    "SIE",
    "EPL",
    "NFW_ELLIPSE_POTENTIAL",

    # Axisymmetric solvers
    "FresnelNUFHT",
    "FresnelNUFHTBatched",
    "FresnelHankelAxisymmetric",  # Alias for FresnelNUFHT
    "FresnelHankelAxisymmetricTrapezoidal",
    "FresnelHankelAxisymmetricSciPy",

    # 2-D solvers
    "FresnelNUFFT3",

    # controls
    "set_num_threads",

    # CPU Tracker
    "CPUTracker",
]

def set_num_threads(n: int):
    """
    Hint FINUFFT/NumPy/OMP to use up to `n` threads.
    Call this once early in your script.

    Sets the OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS
    environment variables, and also calls finufft.set_num_threads if FINUFFT
    is available.
    """
    n = int(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    try:
        import finufft
        finufft.set_num_threads(n)
    except Exception:
        pass

__version__ = "0.1.1"  # Package version following semantic versioning (MAJOR.MINOR.PATCH)
