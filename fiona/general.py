####################################################################
# fiona/general.py
####################################################################

### This code computes integral in the u-domain.
### In the u-domain formulation, all frequency dependence is moved into the integrand coefficients, 
### while the oscillatory kernel is frequency-independent. This allows reuse of a single NUFFT geometry 
### and efficient batched evaluation over many frequencies, but requires recomputing the lens potential
### at scaled arguments ψ(u/w) for each frequency. In contrast, the x-domain formulation evaluates the
### lens potential ψ(x) once on a fixed spatial quadrature grid and is therefore advantageous when ψ is
### expensive to compute, but the NUFFT targets scale with frequency, so a separate NUFFT must be executed
### for each frequency, making it less efficient when many frequencies are needed.

import os
import time
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
except Exception:
    _FINUFFT = False

_TWO_PI = 2.0 * np.pi


def _integrand_coeffs(u1, u2, w, lens, W):
    """
    I(y) = ∫ d^2u e^{-i u·y} exp{i [u^2/(2w) - w ψ(u/w)]}.
    """
    phase = (u1 * u1 + u2 * u2) / (2.0 * w) - w * lens.psi_xy(u1 / w, u2 / w)
    return np.exp(1j * phase) * W


def _fmt_s(t):
    return f"{t:8.4f}s"


class FresnelNUFFT3:
    """
    Fresnel integral evaluator using FINUFFT (type-3 2D NUFFT) on a quadrature grid.

    Key ideas
    ---------
    - Per-frequency mode: build quadrature grid and call NUFFT for each w independently.
    - Batched mode: share one quadrature + NUFFT setup across a *group* of frequencies.
      Optionally chunk frequencies into groups, using either log or linear binning in |w|.

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

    batch_frequencies : bool
        If True, evaluate frequencies in batches (shared quadrature + NUFFT per batch).

    chunk_frequencies : bool
        If True (and batch_frequencies=True), split w into multiple batches by binning.

    frequency_binning : {"log", "linear"}
        Controls how frequencies are binned when chunk_frequencies=True:
          - "log": bins in log10(|w|)
          - "linear": bins in |w|

    frequency_bin_width : float or None
        Bin width for chunking:
          - if frequency_binning="log": width in decades of log10(|w|)
          - if frequency_binning="linear": width in units of |w|
        If None, chunking is disabled (single batch).

    auto_R_from_gl_nodes : bool
        If True (Cartesian + batched), choose R per batch using:
            R_adapt = sqrt(gl_nodes_per_dim / (2 * w_use))
            R = max(min_physical_radius, R_adapt)
        where w_use = max(|w|) in the batch.
        If False, use R = min_physical_radius.

    use_tail_correction : bool
        If True, use the "(exp(-iwψ) - 1)" formulation and add +1 afterward.

    coordinate_system : {"cartesian", "polar"}
        Quadrature coordinate system. Note: chunking/auto_R currently only for cartesian.

    polar_radial_nodes, polar_angular_nodes : int
        Required if coordinate_system="polar".

    uniform_angular_sampling : bool
        If True, use gauss_legendre_polar_uniform_theta_2d; else gauss_legendre_polar_2d.

    numexpr_nthreads : int or None
        Thread count for numexpr only (not FINUFFT). Will be capped by NUMEXPR_MAX_THREADS and cores.

    verbose : bool
        Print configuration and chunk diagnostics.
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

        # Current limitation guard (matches your existing behavior)
        if self.coordinate_system == "polar" and (self.auto_R_from_gl_nodes or self.chunk_frequencies):
            raise NotImplementedError(
                "chunk_frequencies/auto_R_from_gl_nodes are currently implemented only for "
                "coordinate_system='cartesian' (Cartesian Gauss-Legendre in u1,u2)."
            )

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

        # Precompute target quadratic phase once (used in both modes)
        quad_phase = (y1**2 + y2**2) / 2.0

        t_input = time.perf_counter() - t_input0

        # =========================
        # Batched (shared-geometry) path
        # =========================
        if self.batch_frequencies:
            t_groups0 = time.perf_counter()
            # ---- build groups (chunking independent of auto_R) ----
            if (
                self.chunk_frequencies
                and (self.frequency_bin_width is not None)
                and (len(w_vec) > 1)
            ):
                absw = np.abs(w_vec)

                if self.frequency_bin_width <= 0.0:
                    groups = [np.arange(len(w_vec))]
                else:
                    if self.frequency_binning == "log":
                        # bin in log10(|w|)
                        logw = np.log10(absw)
                        vmin = float(logw.min())
                        vmax = float(logw.max())
                        span = vmax - vmin

                        if span <= 0.0:
                            groups = [np.arange(len(w_vec))]
                        else:
                            n_bins = int(np.ceil(span / self.frequency_bin_width))
                            n_bins = max(1, n_bins)
                            edges = vmin + self.frequency_bin_width * np.arange(n_bins + 1, dtype=float)
                            edges[-1] = vmax + 1e-12  # include right edge
                            bin_idx = np.digitize(logw, edges) - 1
                            groups = [np.where(bin_idx == k)[0] for k in range(n_bins)]
                            groups = [g for g in groups if g.size > 0]
                    else:
                        # bin in linear |w|
                        vmin = float(absw.min())
                        vmax = float(absw.max())
                        span = vmax - vmin

                        if span <= 0.0:
                            groups = [np.arange(len(w_vec))]
                        else:
                            n_bins = int(np.ceil(span / self.frequency_bin_width))
                            n_bins = max(1, n_bins)
                            edges = vmin + self.frequency_bin_width * np.arange(n_bins + 1, dtype=float)
                            edges[-1] = vmax + 1e-12  # include right edge
                            bin_idx = np.digitize(absw, edges) - 1
                            groups = [np.where(bin_idx == k)[0] for k in range(n_bins)]
                            groups = [g for g in groups if g.size > 0]
            else:
                groups = [np.arange(len(w_vec))]
            t_groups = time.perf_counter() - t_groups0

            t_alloc0 = time.perf_counter()
            F_out = np.empty((len(w_vec),) + y1.shape, dtype=np.complex128)
            t_alloc = time.perf_counter() - t_alloc0

            t_choose_total = 0.0
            t_quad_total = 0.0
            t_scale_total = 0.0
            t_coeff_total = 0.0
            t_nufft_total = 0.0
            t_post_total = 0.0
            t_chunks_total = 0.0

            t_cj_shapes_total = 0.0
            t_cj_u_over_w_total = 0.0
            t_cj_psi_total = 0.0
            t_cj_base_quad_total = 0.0
            t_cj_tail_quad_phase_total = 0.0
            t_cj_tail_lens_phase_total = 0.0
            t_cj_tail_assemble_total = 0.0
            t_cj_notail_phase_total = 0.0
            t_cj_notail_trig_total = 0.0
            t_cj_notail_assemble_total = 0.0

            if verbose:
                print(
                    "[FresnelNUFFT3] batched path\n"
                    f"  input prep:  {_fmt_s(t_input)}\n"
                    f"  group build: {_fmt_s(t_groups)}\n"
                    f"  alloc out:   {_fmt_s(t_alloc)}\n"
                    f"  chunks:      {len(groups)}"
                )

            for chunk_idx, g in enumerate(groups):
                t_chunk0 = time.perf_counter()

                w_sub = w_vec[g]
                w_use = float(np.max(np.abs(w_sub)))  # representative for sizing

                t_choose0 = time.perf_counter()
                # ---- choose R and Umax for this chunk ----
                if self.auto_R_from_gl_nodes:
                    # Keep gl_nodes_per_dim fixed and choose R from the chunk's max |w|
                    R_adapt = np.sqrt(self.gl_nodes_per_dim / (2.0 * w_use))
                    R = max(self.min_physical_radius, float(R_adapt))
                else:
                    # Fixed physical window radius
                    R_adapt = None
                    R = self.min_physical_radius

                Umax = w_use * R
                h = np.pi / Umax
                t_choose = time.perf_counter() - t_choose0
                t_choose_total += t_choose

                # ---- reporting ----
                if verbose:
                    w_abs = np.abs(w_sub)
                    w_min = float(w_abs.min())
                    w_max = float(w_abs.max())

                    if self.chunk_frequencies and (self.frequency_bin_width is not None):
                        if self.frequency_binning == "log":
                            bin_msg = f" | binning=log10(|w|), Δ={self.frequency_bin_width:g} decades"
                        else:
                            bin_msg = f" | binning=linear(|w|), Δ={self.frequency_bin_width:g}"
                    else:
                        bin_msg = ""

                    if self.auto_R_from_gl_nodes:
                        floored = (R > float(R_adapt) + 1e-15)
                        cap_msg = " (floored by min_physical_radius)" if floored else ""

                        msg = (
                            f"[chunk {chunk_idx+1}/{len(groups)}] "
                            f"count={len(w_sub)} | |w| in [{w_min:.6g}, {w_max:.6g}] | "
                            f"N={self.gl_nodes_per_dim} fixed -> "
                            f"R={R:.6g}{cap_msg}, Umax={Umax:.6g}, h={h:.6g}"
                        )
                    else:
                        msg = (
                            f"[chunk {chunk_idx+1}/{len(groups)}] "
                            f"count={len(w_sub)} | |w| in [{w_min:.6g}, {w_max:.6g}] | "
                            f"R={R:.6g} fixed, N={self.gl_nodes_per_dim} (per dim), "
                            f"Umax={Umax:.6g}, h={h:.6g}"
                        )
                    print(msg + bin_msg)

                t_quad0 = time.perf_counter()
                # ---- 1) Quadrature nodes/weights in u-space (u1,u2 in [-Umax, Umax]) ----
                u1, u2, W = gauss_legendre_2d(self.gl_nodes_per_dim, Umax)
                t_quad = time.perf_counter() - t_quad0
                t_quad_total += t_quad

                t_scale0 = time.perf_counter()
                # ---- 2) NUFFT scalings (invariant product: (h*u) * (y/h) = u*y) ----
                xj = h * u1
                yj = h * u2
                sk = y1 / h
                tk = y2 / h
                t_scale = time.perf_counter() - t_scale0
                t_scale_total += t_scale

                t_coeff0 = time.perf_counter()
                # ---- 3) Build coefficients cj for all w in this chunk ----

                t0 = time.perf_counter()
                w2d = w_sub[:, None]    # (M, 1)
                u1_2d = u1[None, :]     # (1, N)
                u2_2d = u2[None, :]     # (1, N)
                W_2d = W[None, :]       # (1, N)
                t_cj_shapes = time.perf_counter() - t0
                t_cj_shapes_total += t_cj_shapes

                t0 = time.perf_counter()
                # u/w for lens evaluation
                u1_over_w = ne.evaluate("u1_2d / w2d")
                u2_over_w = ne.evaluate("u2_2d / w2d")
                t_cj_u_over_w = time.perf_counter() - t0
                t_cj_u_over_w_total += t_cj_u_over_w

                t0 = time.perf_counter()
                psi = self.lens.psi_xy(u1_over_w, u2_over_w)  # (M, N)
                t_cj_psi = time.perf_counter() - t0
                t_cj_psi_total += t_cj_psi

                t0 = time.perf_counter()
                base_quad = (u1 * u1 + u2 * u2) / 2.0     # (N,)
                base_quad_2d = base_quad[None, :]         # (1, N)
                t_cj_base_quad = time.perf_counter() - t0
                t_cj_base_quad_total += t_cj_base_quad

                if self.use_tail_correction:
                    t0 = time.perf_counter()
                    # cj = exp(i * (u^2/(2w))) * (exp(-i*w*psi) - 1) * W
                    phase_quad = ne.evaluate("base_quad_2d / w2d")
                    cosq = ne.evaluate("cos(phase_quad)")
                    sinq = ne.evaluate("sin(phase_quad)")
                    exp_quad = cosq + 1j * sinq
                    t_cj_tail_quad_phase = time.perf_counter() - t0
                    t_cj_tail_quad_phase_total += t_cj_tail_quad_phase

                    t0 = time.perf_counter()
                    phase_lens = ne.evaluate("-w2d * psi")
                    cosl = ne.evaluate("cos(phase_lens)")
                    sinl = ne.evaluate("sin(phase_lens)")
                    exp_lens = cosl + 1j * sinl
                    t_cj_tail_lens_phase = time.perf_counter() - t0
                    t_cj_tail_lens_phase_total += t_cj_tail_lens_phase

                    t0 = time.perf_counter()
                    cj = ne.evaluate("W_2d") * exp_quad * (exp_lens - 1.0)
                    t_cj_tail_assemble = time.perf_counter() - t0
                    t_cj_tail_assemble_total += t_cj_tail_assemble
                else:
                    t0 = time.perf_counter()
                    # cj = exp(i * (u^2/(2w) - w*psi)) * W
                    phase = ne.evaluate("base_quad_2d / w2d - w2d * psi")
                    t_cj_notail_phase = time.perf_counter() - t0
                    t_cj_notail_phase_total += t_cj_notail_phase

                    t0 = time.perf_counter()
                    cosp = ne.evaluate("cos(phase)")
                    sinp = ne.evaluate("sin(phase)")
                    t_cj_notail_trig = time.perf_counter() - t0
                    t_cj_notail_trig_total += t_cj_notail_trig

                    t0 = time.perf_counter()
                    cj = (cosp + 1j * sinp) * W_2d
                    t_cj_notail_assemble = time.perf_counter() - t0
                    t_cj_notail_assemble_total += t_cj_notail_assemble

                t_coeff = time.perf_counter() - t_coeff0
                t_coeff_total += t_coeff

                t_nufft0 = time.perf_counter()
                # ---- 4) NUFFT ----
                I = nufft2d3(xj, yj, cj, sk, tk, isign=-1, eps=self.nufft_tol)
                t_nufft = time.perf_counter() - t_nufft0
                t_nufft_total += t_nufft

                t_post0 = time.perf_counter()
                # ---- 5) Fresnel prefactor (+ optional "+1" tail correction) ----
                for i, w_i in enumerate(w_sub):
                    val = np.exp(1j * w_i * quad_phase) * I[i] / (1j * w_i * _TWO_PI)
                    if self.use_tail_correction:
                        val = 1.0 + val
                    F_out[g[i]] = val
                t_post = time.perf_counter() - t_post0
                t_post_total += t_post

                t_chunk = time.perf_counter() - t_chunk0
                t_chunks_total += t_chunk

                if verbose:
                    if self.use_tail_correction:
                        print(
                            "  timing:\n"
                            f"    choose R/Umax/h: {_fmt_s(t_choose)}\n"
                            f"    quadrature:      {_fmt_s(t_quad)}\n"
                            f"    scaling:         {_fmt_s(t_scale)}\n"
                            f"    coeffs (cj):     {_fmt_s(t_coeff)}\n"
                            f"      shapes:        {_fmt_s(t_cj_shapes)}\n"
                            f"      u/w:           {_fmt_s(t_cj_u_over_w)}\n"
                            f"      psi:           {_fmt_s(t_cj_psi)}\n"
                            f"      base_quad:     {_fmt_s(t_cj_base_quad)}\n"
                            f"      quad exp:      {_fmt_s(t_cj_tail_quad_phase)}\n"
                            f"      lens exp:      {_fmt_s(t_cj_tail_lens_phase)}\n"
                            f"      assemble:      {_fmt_s(t_cj_tail_assemble)}\n"
                            f"    NUFFT:           {_fmt_s(t_nufft)}\n"
                            f"    post:            {_fmt_s(t_post)}\n"
                            f"    chunk total:     {_fmt_s(t_chunk)}"
                        )
                    else:
                        print(
                            "  timing:\n"
                            f"    choose R/Umax/h: {_fmt_s(t_choose)}\n"
                            f"    quadrature:      {_fmt_s(t_quad)}\n"
                            f"    scaling:         {_fmt_s(t_scale)}\n"
                            f"    coeffs (cj):     {_fmt_s(t_coeff)}\n"
                            f"      shapes:        {_fmt_s(t_cj_shapes)}\n"
                            f"      u/w:           {_fmt_s(t_cj_u_over_w)}\n"
                            f"      psi:           {_fmt_s(t_cj_psi)}\n"
                            f"      base_quad:     {_fmt_s(t_cj_base_quad)}\n"
                            f"      phase:         {_fmt_s(t_cj_notail_phase)}\n"
                            f"      trig:          {_fmt_s(t_cj_notail_trig)}\n"
                            f"      assemble:      {_fmt_s(t_cj_notail_assemble)}\n"
                            f"    NUFFT:           {_fmt_s(t_nufft)}\n"
                            f"    post:            {_fmt_s(t_post)}\n"
                            f"    chunk total:     {_fmt_s(t_chunk)}"
                        )

            t_total = time.perf_counter() - t_total0
            if verbose:
                if self.use_tail_correction:
                    print(
                        "[FresnelNUFFT3] totals (batched)\n"
                        f"  input prep:        {_fmt_s(t_input)}\n"
                        f"  group build:       {_fmt_s(t_groups)}\n"
                        f"  alloc out:         {_fmt_s(t_alloc)}\n"
                        f"  choose R/Umax/h:   {_fmt_s(t_choose_total)}\n"
                        f"  quadrature:        {_fmt_s(t_quad_total)}\n"
                        f"  scaling:           {_fmt_s(t_scale_total)}\n"
                        f"  coeffs (cj):       {_fmt_s(t_coeff_total)}\n"
                        f"    shapes:          {_fmt_s(t_cj_shapes_total)}\n"
                        f"    u/w:             {_fmt_s(t_cj_u_over_w_total)}\n"
                        f"    psi:             {_fmt_s(t_cj_psi_total)}\n"
                        f"    base_quad:       {_fmt_s(t_cj_base_quad_total)}\n"
                        f"    quad exp:        {_fmt_s(t_cj_tail_quad_phase_total)}\n"
                        f"    lens exp:        {_fmt_s(t_cj_tail_lens_phase_total)}\n"
                        f"    assemble:        {_fmt_s(t_cj_tail_assemble_total)}\n"
                        f"  NUFFT:             {_fmt_s(t_nufft_total)}\n"
                        f"  post:              {_fmt_s(t_post_total)}\n"
                        f"  chunks total:      {_fmt_s(t_chunks_total)}\n"
                        f"  wall total:        {_fmt_s(t_total)}"
                    )
                else:
                    print(
                        "[FresnelNUFFT3] totals (batched)\n"
                        f"  input prep:        {_fmt_s(t_input)}\n"
                        f"  group build:       {_fmt_s(t_groups)}\n"
                        f"  alloc out:         {_fmt_s(t_alloc)}\n"
                        f"  choose R/Umax/h:   {_fmt_s(t_choose_total)}\n"
                        f"  quadrature:        {_fmt_s(t_quad_total)}\n"
                        f"  scaling:           {_fmt_s(t_scale_total)}\n"
                        f"  coeffs (cj):       {_fmt_s(t_coeff_total)}\n"
                        f"    shapes:          {_fmt_s(t_cj_shapes_total)}\n"
                        f"    u/w:             {_fmt_s(t_cj_u_over_w_total)}\n"
                        f"    psi:             {_fmt_s(t_cj_psi_total)}\n"
                        f"    base_quad:       {_fmt_s(t_cj_base_quad_total)}\n"
                        f"    phase:           {_fmt_s(t_cj_notail_phase_total)}\n"
                        f"    trig:            {_fmt_s(t_cj_notail_trig_total)}\n"
                        f"    assemble:        {_fmt_s(t_cj_notail_assemble_total)}\n"
                        f"  NUFFT:             {_fmt_s(t_nufft_total)}\n"
                        f"  post:              {_fmt_s(t_post_total)}\n"
                        f"  chunks total:      {_fmt_s(t_chunks_total)}\n"
                        f"  wall total:        {_fmt_s(t_total)}"
                    )

            return F_out

        # =========================
        # Per-frequency path (independent geometry per w)
        # =========================
        t_alloc0 = time.perf_counter()
        F_list = []
        t_alloc = time.perf_counter() - t_alloc0

        t_quad_total = 0.0
        t_scale_total = 0.0
        t_coeff_total = 0.0
        t_nufft_total = 0.0
        t_post_total = 0.0
        t_loop_total = 0.0

        if verbose:
            print(
                "[FresnelNUFFT3] per-frequency path\n"
                f"  input prep:  {_fmt_s(t_input)}\n"
                f"  alloc list:  {_fmt_s(t_alloc)}\n"
                f"  count(w):    {len(w_vec)}"
            )

        for w_i in w_vec:
            t_loop0 = time.perf_counter()
            Umax = abs(w_i) * self.min_physical_radius

            t_quad0 = time.perf_counter()
            if self.coordinate_system == "cartesian":
                u1, u2, W = gauss_legendre_2d(self.gl_nodes_per_dim, Umax)
            else:
                # coordinate_system == "polar"
                if self.uniform_angular_sampling:
                    r, theta, W = gauss_legendre_polar_uniform_theta_2d(
                        self.polar_radial_nodes,
                        self.polar_angular_nodes,
                        Umax,
                    )
                else:
                    r, theta, W = gauss_legendre_polar_2d(
                        self.polar_radial_nodes,
                        self.polar_angular_nodes,
                        Umax,
                    )
                u1 = r * np.cos(theta)
                u2 = r * np.sin(theta)
            t_quad = time.perf_counter() - t_quad0
            t_quad_total += t_quad

            t_scale0 = time.perf_counter()
            h = np.pi / Umax
            xj = h * u1
            yj = h * u2
            sk = y1 / h
            tk = y2 / h
            t_scale = time.perf_counter() - t_scale0
            t_scale_total += t_scale

            t_coeff0 = time.perf_counter()
            phase = (u1 * u1 + u2 * u2) / (2.0 * w_i) - w_i * self.lens.psi_xy(u1 / w_i, u2 / w_i)
            cj = np.exp(1j * phase) * W
            t_coeff = time.perf_counter() - t_coeff0
            t_coeff_total += t_coeff

            t_nufft0 = time.perf_counter()
            I = nufft2d3(xj, yj, cj, sk, tk, isign=-1, eps=self.nufft_tol)
            t_nufft = time.perf_counter() - t_nufft0
            t_nufft_total += t_nufft

            t_post0 = time.perf_counter()
            Fw = np.exp(1j * w_i * quad_phase) * I / (1j * w_i * _TWO_PI)
            if self.use_tail_correction:
                Fw = 1.0 + Fw
            t_post = time.perf_counter() - t_post0
            t_post_total += t_post

            F_list.append(Fw)

            t_loop = time.perf_counter() - t_loop0
            t_loop_total += t_loop

            if verbose:
                print(
                    f"[w={w_i:.6g}] "
                    f"Umax={Umax:.6g} | "
                    f"t_quad={_fmt_s(t_quad)} "
                    f"t_scale={_fmt_s(t_scale)} "
                    f"t_coeff={_fmt_s(t_coeff)} "
                    f"t_nufft={_fmt_s(t_nufft)} "
                    f"t_post={_fmt_s(t_post)} "
                    f"t_total={_fmt_s(t_loop)}"
                )

        t_total = time.perf_counter() - t_total0
        if verbose:
            print(
                "[FresnelNUFFT3] totals (per-frequency)\n"
                f"  input prep:        {_fmt_s(t_input)}\n"
                f"  quadrature:        {_fmt_s(t_quad_total)}\n"
                f"  scaling:           {_fmt_s(t_scale_total)}\n"
                f"  coeffs (cj):       {_fmt_s(t_coeff_total)}\n"
                f"  NUFFT:             {_fmt_s(t_nufft_total)}\n"
                f"  post:              {_fmt_s(t_post_total)}\n"
                f"  loop total:        {_fmt_s(t_loop_total)}\n"
                f"  wall total:        {_fmt_s(t_total)}"
            )

        return np.stack(F_list, axis=0)
