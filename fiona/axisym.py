####################################################################
# fiona/axisym.py
####################################################################

import numpy as np
from pathlib import Path
import time

from .lenses import AxisymmetricLens
from .utils import CPUTracker

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
    from juliacall import Main as jl

    jl.seval("""
        using FastHankelTransform
        Base.eval(FastHankelTransform, :(using ForwardDiff))
        import FastHankelTransform: nufht

        function fht_batch(nu, rs, ys, c_re_batch, c_im_batch; tol=1e-12)
            n_w = size(c_re_batch, 1)
            n_y = length(ys)
            g_re_all = Matrix{Float64}(undef, n_w, n_y)
            g_im_all = Matrix{Float64}(undef, n_w, n_y)
            for i in 1:n_w
                cre = @view c_re_batch[i, :]
                cim = @view c_im_batch[i, :]
                gre = nufht(nu, rs, cre, ys; tol=tol)
                gim = nufht(nu, rs, cim, ys; tol=tol)
                @inbounds g_re_all[i, :] .= gre
                @inbounds g_im_all[i, :] .= gim
            end
            return g_re_all, g_im_all
        end
    """)

    j_fht_batch = jl.fht_batch
    _HAS_FHT = True

except Exception as e:
    _HAS_FHT = False
    _FHT_ERR = e
    
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
    Fresnel integral for axisymmetric lenses using **precomputed GL nodes**
    and **FastHankelTransform NUFHT**.

    Precomputed files must exist:

        {GL_DIR}/gl2d_n{n_gl}_U{Umax}.x.npy
        {GL_DIR}/gl2d_n{n_gl}_U{Umax}.w.npy

    We discard x<0 (symmetry) and use:
        rs = x[x>0]
        u_weights = rs * w[x>0]
    """

    def __init__(self,
                 lens: AxisymmetricLens,
                 n_gl: int,
                 Umax: float,
                 gl_dir: str,
                 tol: float = 1e-12):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError("FresnelHankelAxisymmetric requires AxisymmetricLens")

        if not _HAS_FHT:
            raise ImportError("FastHankelTransform.jl cannot be loaded: "
                              f"{_FHT_ERR!r}")

        self.lens = lens
        self.n_gl = int(n_gl)
        self.Umax = float(Umax)
        self.tol = float(tol)

        # Load precomputed GL nodes
        gl_dir = Path(gl_dir)
        x_path = gl_dir / f"gl2d_n{n_gl}_U{int(Umax)}.x.npy"
        w_path = gl_dir / f"gl2d_n{n_gl}_U{int(Umax)}.w.npy"

        if not x_path.exists() or not w_path.exists():
            raise FileNotFoundError(f"Missing GL files:\n  {x_path}\n  {w_path}")

        x = np.load(x_path)
        w = np.load(w_path)

        # Keep only positive u (symmetry of Gauss–Legendre)
        mask = x > 0
        rs = x[mask]          # u_k
        du = w[mask]          # Δu_k
        u_weights = rs * du   # u_k Δu_k for ∫ u f(u) du

        self._rs = rs.astype(float)
        self._u_weights = u_weights.astype(float)

        # Julia arrays
        self._jr = jl.Array(self._rs)

    # ------------------------------------------------------------------
    # Main evaluation
    # ------------------------------------------------------------------
    def __call__(self, w_vec, y_vec):

        t0 = time.perf_counter()

        w_vec = np.asarray(w_vec, dtype=float).ravel()
        y_vec = np.asarray(y_vec, dtype=float).ravel()
        if np.any(w_vec == 0):
            raise ValueError("All w must be nonzero.")

        rs = self._rs
        u_weights = self._u_weights

        jr = self._jr

        nu = 0
        tol = self.tol

        n_w = len(w_vec)
        n_y = len(y_vec)

        # ---- Setup / allocations (Python + Julia) ----
        setup_start = time.perf_counter()
        jy = jl.Array(y_vec)  # Python → Julia (y grid)
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

            # ---------- Step 2: Python/Julia interface + Julia NUFHT ----------
            # 2a. Python → Julia
            py_to_jl_start = time.perf_counter()
            j_c_re = jl.Array(c_re)
            j_c_im = jl.Array(c_im)
            py_to_jl_end = time.perf_counter()

            # 2b. Julia NUFHT call (Julia compute + call overhead)
            julia_call_start = time.perf_counter()
            g_re_all, g_im_all = j_fht_batch(nu, jr, jy, j_c_re, j_c_im, tol=tol)
            julia_call_end = time.perf_counter()

            # 2c. Julia → Python
            jl_to_py_start = time.perf_counter()
            g_re = np.array(g_re_all)
            g_im = np.array(g_im_all)
            jl_to_py_end = time.perf_counter()
            step2_end = jl_to_py_end

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

        # Step 2: Python/Julia + Julia
        step2_total = step2_end - step1_end
        py_to_jl_time = py_to_jl_end - py_to_jl_start
        julia_time = julia_call_end - julia_call_start
        jl_to_py_time = jl_to_py_end - jl_to_py_start
        step2_unaccounted = step2_total - (py_to_jl_time + julia_time + jl_to_py_time)

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
        print(" FresnelHankelAxisymmetric (GL_precomputed + Julia NUFHT)")
        print("────────────────────────────────────────────────────────────")
        print(tracker.report("  CPU usage summary"))
        print()

        # ---- Step 0 ----
        print("  Step 0: Setup / allocations")
        print("  ───────────────────────────")
        print(f"    0a. jy (Julia) + F, quad_phase   : "
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
        print("  Step 2: Julia NUFHT + Python/Julia interface")
        print("  ───────────────────────────────────────────")
        print(f"    2a. Python → Julia (c_re/c_im)   : "
              f"{_pct(py_to_jl_time, step2_total):6.2f}%  ({py_to_jl_time:10.6f} s)")
        print(f"    2b. Julia NUFHT (j_fht_batch)    : "
              f"{_pct(julia_time, step2_total):6.2f}%  ({julia_time:10.6f} s)")
        print(f"    2c. Julia → Python (g_re/g_im)   : "
              f"{_pct(jl_to_py_time, step2_total):6.2f}%  ({jl_to_py_time:10.6f} s)")
        print(f"    2d. other (Step 2)               : "
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
        print(f"  2. Julia NUFHT + interface         : "
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

class FresnelHankelAxisymmetricTrapezoidal:
    r"""
    Fresnel integral for axisymmetric lenses using a fast Hankel transform.

        F(w, y) = e^{i w y^2 / 2} / (i w) ∫_0^∞ u du
                  exp{i w [ u^2 / (2 w^2) - ψ(u / w) ]} J_0(u y),

    We discretize the radial u-integral on [0, Umax] and evaluate the Bessel sum 
    with FastHankelTransform.jl's `nufht` (nonuniform fast Hankel transform). 

        ∫_0^Umax u f_w(u) J_0(u y) du ≈ ∑_k c_k(w) J_0(y r_k),

    where r_k are radial nodes, and

        c_k(w) = u_k Δu_k * exp{i w [ u_k^2 / (2 w^2) - ψ(u_k / w) ]}.

    nufht(ν, r_k, c_k, y_j; tol) then returns the vector of Hankel sums
    g_j ≈ ∑_k c_k J_ν(y_j r_k).  We have to perform a zeroth-order FHT (ν=0).
    """

    def __init__(self, lens: AxisymmetricLens,
                 n_r: int = 1024,
                 Umax: float = 50.0,
                 tol: float = 1e-12):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError(
                "FresnelHankelAxisymmetric requires an AxisymmetricLens instance."
            )

        if not _HAS_FHT:
            raise ImportError(
                "FastHankelTransform.jl is not available via juliacall.\n"
                f"Original import error: {_FHT_ERR!r}"
            )

        self.lens = lens
        self.n_r = int(n_r)
        self.Umax = float(Umax)
        self.tol = float(tol)

        # Build radial grid r_k and weights w_k for ∫_0^{Umax} u f(u) du.
        self._rs, self._u_weights = self._build_radial_grid()

        # Julia arrays we can reuse on each call
        self._jr = jl.Array(self._rs)

    # ------------------------------------------------------------------
    # Radial grid and quadrature weights
    # ------------------------------------------------------------------
    def _build_radial_grid(self):
        """
        Simple uniform radial grid on (0, Umax] with trapezoidal weights.

        We avoid r=0 to keep things well-behaved numerically; the missing
        interval [0, r_min] is negligible for sufficiently large n_r.
        """
        n = self.n_r
        Umax = self.Umax

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

        rs = self._rs                # u_k
        u_weights = self._u_weights  # u_k Δu_k

        jr = self._jr                # Julia array for rs
        jy = jl.Array(y_vec)         # Julia array for y

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

            # ---------------- Step 2: Python/Julia interface + Julia NUFHT ----------------
            # 2a. Python → Julia
            py_to_jl_start = time.perf_counter()
            j_c_re = jl.Array(c_re_batch)
            j_c_im = jl.Array(c_im_batch)
            py_to_jl_end = time.perf_counter()

            # 2b. Julia NUFHT call (includes Julia compute + call overhead)
            julia_call_start = time.perf_counter()
            g_re_all, g_im_all = j_fht_batch(
                nu, jr, jy, j_c_re, j_c_im, tol=tol
            )
            julia_call_end = time.perf_counter()

            # 2c. Julia → Python
            jl_to_py_start = time.perf_counter()
            g_re = np.array(g_re_all, dtype=float)  # shape (n_w, n_y)
            g_im = np.array(g_im_all, dtype=float)
            jl_to_py_end = time.perf_counter()

            t2 = jl_to_py_end

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
        py_to_jl_time = py_to_jl_end - py_to_jl_start
        julia_time = julia_call_end - julia_call_start
        jl_to_py_time = jl_to_py_end - jl_to_py_start
        step2_unaccounted = step2_total - (py_to_jl_time + julia_time + jl_to_py_time)

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
        print(" FresnelHankelAxisymmetric (Trapezoidal + Julia NUFHT)")
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
        print("  Step 2: Julia NUFHT + Python/Julia interface")
        print("  ─────────────────────────────────────────────")
        print(f"    2a. Python → Julia (Array)       : "
              f"{_pct(py_to_jl_time, step2_total):6.2f}%  ({py_to_jl_time:10.6f} s)")
        print(f"    2b. Julia NUFHT (j_fht_batch)    : "
              f"{_pct(julia_time, step2_total):6.2f}%  ({julia_time:10.6f} s)")
        print(f"    2c. Julia → Python (np.array)    : "
              f"{_pct(jl_to_py_time, step2_total):6.2f}%  ({jl_to_py_time:10.6f} s)")
        print(f"    2d. other (Step 2)               : "
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
        print(f"  2. Julia NUFHT + interface (Step 2): "
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
    Fresnel integral for axisymmetric lenses using **precomputed GL nodes**
    for configuration (n_gl, Umax), but performing the Hankel transform with
    SciPy's FFTLog-based fast Hankel transform (scipy.fft.fht).
    """

    def __init__(self,
                 lens: AxisymmetricLens,
                 n_gl: int,
                 Umax: float,
                 gl_dir: str,
                 tol: float = 1e-12):

        if not isinstance(lens, AxisymmetricLens):
            raise TypeError("FresnelHankelAxisymmetricSciPy requires AxisymmetricLens")

        if not _HAS_SCIPY_FHT:
            raise ImportError(
                "scipy.fft.fht (fast Hankel transform) is not available.\n"
                f"Original import error: {_SCIPY_FHT_ERR!r}"
            )

        self.lens = lens
        self.n_gl = int(n_gl)
        self.Umax = float(Umax)
        self.tol = float(tol)

        # --- Load precomputed 1-D GL nodes ---
        gl_dir = Path(gl_dir)
        x_path = gl_dir / f"gl2d_n{n_gl}_U{int(Umax)}.x.npy"
        w_path = gl_dir / f"gl2d_n{n_gl}_U{int(Umax)}.w.npy"

        if not x_path.exists() or not w_path.exists():
            raise FileNotFoundError(f"Missing GL files:\n  {x_path}\n  {w_path}")

        x = np.load(x_path)
        w = np.load(w_path)

        mask = x > 0
        rs_gl = x[mask]      # original GL radial nodes u_k

        # We use the GL nodes to define the radial range and sample count
        self._r_min = float(rs_gl.min())
        self._r_max = float(rs_gl.max())
        self._n_r = rs_gl.size

        # --- Build logarithmic radial grid, r_j = r_c * exp[(j-j_c)*dln] ---
        r = np.geomspace(self._r_min, self._r_max, self._n_r)
        dln = float(np.log(r[1] / r[0]))
        mu = 0.0
        bias = 0.0
        offset = _scipy_fhtoffset(dln, mu=mu, bias=bias)
        k = np.exp(offset) / r[::-1]

        self._r = r
        self._dln = dln
        self._mu = mu
        self._bias = bias
        self._offset = offset
        self._k = k

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

        r = self._r
        dln = self._dln
        mu = self._mu
        offset = self._offset
        k = self._k

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