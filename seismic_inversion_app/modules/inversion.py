"""The four post-stack inversion engines, plus the volume runner.

Every engine exposes the same trace-level signature::

    invert(trace, wavelet, low_freq_trace=None, **params) -> dict

and returns a dict with (at least) these keys:

===================  =====================================================
``reflectivity``     estimated reflectivity series (seismic sampling)
``relative_ai``      band-limited log-impedance, no absolute level
``absolute_ai``      absolute acoustic impedance, or ``None`` without an LFM
``synthetic``        wavelet convolved with the estimated reflectivity
``residual``         ``trace - synthetic``
``misfit``           normalised RMS residual (0 = perfect)
``correlation``      zero-lag correlation of synthetic against the input
===================  =====================================================

Keeping the interface identical is what lets :func:`run_volume` drive any of
them, and what lets the QC panels be written once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy import sparse
from scipy.linalg import solveh_banded
from scipy.optimize import minimize
from scipy.sparse.linalg import cg, spsolve

from . import utils

METHODS = ("coloured", "sparse-spike", "model-based", "bayesian")


# ==========================================================================
# Shared operator machinery
# ==========================================================================

def convolution_matrix(wavelet: np.ndarray, n: int) -> sparse.csr_matrix:
    """Banded (Toeplitz) convolution matrix ``C`` such that ``C @ r`` is the
    'same'-mode convolution of ``r`` with the wavelet.

    Built sparse because ``n`` is the trace length (hundreds to thousands) and
    the band is only as wide as the wavelet.
    """
    w = np.asarray(wavelet, dtype=float)
    n_w = w.size
    centre = n_w // 2
    offsets, diagonals = [], []
    for k in range(n_w):
        off = centre - k
        if abs(off) >= n:
            continue
        offsets.append(off)
        diagonals.append(np.full(n - abs(off), w[k]))
    return sparse.diags(diagonals, offsets, shape=(n, n), format="csr")


def difference_matrix(n: int) -> sparse.csr_matrix:
    """``D`` mapping log-impedance to reflectivity: ``r_i = 0.5 (m_{i+1} - m_i)``.

    The last row is zero so the output stays length ``n`` and aligned with the
    seismic sampling (matching :func:`utils.reflectivity_from_ai`).
    """
    main = np.full(n, -0.5)
    main[-1] = 0.0
    upper = np.full(n - 1, 0.5)
    return sparse.diags([main, upper], [0, 1], shape=(n, n), format="csr")


def second_difference_matrix(n: int) -> sparse.csr_matrix:
    """Roughness operator: ``(L x)_i = x_{i-1} - 2 x_i + x_{i+1}``.

    The first and last rows are zeroed.  Left truncated they read ``[-2, 1]``
    and ``[1, -2]``, which do not annihilate a constant -- so the "roughness"
    of a flat trace comes out as ``2 * (4 - 4 + 1) x^2`` rather than zero, and
    any penalty built on it quietly drags the *level* of log-impedance toward
    zero instead of merely smoothing it.  Zeroing the boundary rows leaves an
    operator that is blind to constants and linear ramps, which is what a
    curvature penalty is supposed to be.
    """
    if n < 3:
        return sparse.csr_matrix((n, n))
    main = np.full(n, -2.0)
    lower = np.ones(n - 1)
    upper = np.ones(n - 1)
    main[0] = main[-1] = 0.0
    upper[0] = 0.0            # row 0 -> all zeros
    lower[-1] = 0.0           # row n-1 -> all zeros
    return sparse.diags([lower, main, upper], [-1, 0, 1], shape=(n, n), format="csr").tocsr()


def _symmetric_to_banded(mat: sparse.spmatrix, n: int) -> np.ndarray:
    """Upper-banded (LAPACK ``ab``) form of a symmetric banded sparse matrix."""
    csr = mat.tocsr()
    # Widest non-zero superdiagonal fixes the bandwidth.
    coo = csr.tocoo()
    u = int(np.max(coo.col - coo.row)) if coo.nnz else 0
    u = max(min(u, n - 1), 0)
    ab = np.zeros((u + 1, n), dtype=float)
    for k in range(u + 1):
        d = csr.diagonal(k)
        ab[u - k, k:] = d
    return ab


class _OperatorCache:
    """Memoise the sparse operators; they only depend on ``(n, wavelet)``.

    Rebuilding ``C``, ``D`` and especially ``C.T @ C`` for every trace dominated
    the runtime of a full-volume sparse-spike run -- they are identical for
    every trace in the cube, so they are built once and reused.
    """

    def __init__(self, max_entries: int = 4) -> None:
        self._store: dict = {}
        self._max = max_entries

    def _entry(self, wavelet: np.ndarray, n: int) -> dict:
        key = (n, wavelet.tobytes())
        hit = self._store.get(key)
        if hit is None:
            if len(self._store) >= self._max:
                self._store.pop(next(iter(self._store)))
            hit = {"C": convolution_matrix(wavelet, n), "D": difference_matrix(n)}
            self._store[key] = hit
        return hit

    def get(self, wavelet: np.ndarray, n: int):
        """``(C, D, C @ D)`` -- the forward operator for model-based inversion."""
        e = self._entry(wavelet, n)
        if "CD" not in e:
            e["CD"] = (e["C"] @ e["D"]).tocsr()
        return e["C"], e["D"], e["CD"]

    def convolution(self, wavelet: np.ndarray, n: int):
        """``(C, C.T @ C)`` -- the normal equations for sparse-spike."""
        e = self._entry(wavelet, n)
        if "CtC" not in e:
            e["CtC"] = (e["C"].T @ e["C"]).tocsr()
        return e["C"], e["CtC"]

    def banded_normal(self, wavelet: np.ndarray, n: int):
        """``(C, C.T@C, ab)`` where ``ab`` is ``C.T@C`` in LAPACK upper-banded form.

        ``C.T @ C`` is symmetric and banded (half-bandwidth twice the wavelet's),
        so a banded Cholesky solves the IRLS system in ``O(n * b^2)`` -- far
        cheaper than the hundreds of CG matvecs the ill-conditioned reweighted
        system otherwise needs.
        """
        C, CtC = self.convolution(wavelet, n)
        e = self._entry(wavelet, n)
        if "ab" not in e:
            e["ab"] = _symmetric_to_banded(CtC, n)
        return C, CtC, e["ab"]

    def banded_model(self, wavelet: np.ndarray, n: int):
        """``(G, G.T@G, ab)`` for the log-impedance operator ``G = C @ D``.

        Bayesian inversion needs the same normal equations as the model-based
        engine, but assembled once rather than re-formed per trace.
        """
        _C, _D, G = self.get(wavelet, n)
        e = self._entry(wavelet, n)
        if "GtG" not in e:
            e["GtG"] = (G.T @ G).tocsr()
            e["ab_G"] = _symmetric_to_banded(e["GtG"], n)
        return G, e["GtG"], e["ab_G"]

    def clear(self) -> None:
        self._store.clear()


_OPERATORS = _OperatorCache()


def _qc(trace: np.ndarray, synthetic: np.ndarray) -> dict:
    """Residual, normalised RMS misfit and correlation for a QC panel."""
    residual = trace - synthetic
    denom = float(np.sqrt(np.mean(trace ** 2)))
    misfit = float(np.sqrt(np.mean(residual ** 2)) / denom) if denom > 0 else float("nan")
    return {
        "synthetic": synthetic,
        "residual": residual,
        "misfit": misfit,
        "correlation": utils.normalised_correlation(synthetic, trace),
    }


def _finalise(
    trace: np.ndarray,
    reflectivity: np.ndarray,
    wavelet: np.ndarray,
    low_freq_trace: np.ndarray | None,
    dt: float,
    merge_freq: float,
    method: str,
    extra: dict | None = None,
) -> dict:
    """Assemble the common result dict from an estimated reflectivity series."""
    synthetic = utils.convolve_same(reflectivity, wavelet)
    rel = utils.relative_impedance(reflectivity)

    absolute = None
    if low_freq_trace is not None:
        lf = np.asarray(low_freq_trace, dtype=float)
        if lf.size == rel.size and np.isfinite(lf).any():
            try:
                absolute = utils.merge_with_low_frequency(rel, lf, dt, merge_freq)
            except ValueError:
                absolute = None

    out = {
        "method": method,
        "reflectivity": reflectivity,
        "relative_ai": rel,
        "absolute_ai": absolute,
        **_qc(trace, synthetic),
    }
    if extra:
        out.update(extra)
    return out


# ==========================================================================
# 1. Coloured inversion  (Lancaster & Whitcombe, 2000)
# ==========================================================================

@dataclass
class ColourOperator:
    """The spectral shaping filter plus the fit that produced it."""

    samples: np.ndarray
    dt: float
    exponent: float                  # power-law slope beta of |R(f)| ~ f^beta
    intercept: float
    f_low: float
    f_high: float
    seismic_freq: np.ndarray = field(default_factory=lambda: np.array([]))
    seismic_amp: np.ndarray = field(default_factory=lambda: np.array([]))
    target_amp: np.ndarray = field(default_factory=lambda: np.array([]))
    operator_amp: np.ndarray = field(default_factory=lambda: np.array([]))
    n_wells: int = 0
    scalar: float = 1.0
    n_calibration_wells: int = 0

    def summary(self) -> dict:
        return {
            "power-law exponent (beta)": f"{self.exponent:+.3f}",
            "design band (Hz)": f"{self.f_low:.1f} - {self.f_high:.1f}",
            "operator length (ms)": f"{self.samples.size * self.dt * 1000:.0f}",
            "wells in spectral fit": self.n_wells,
            "amplitude scalar": f"{self.scalar:.4g}",
            "calibration wells": self.n_calibration_wells,
        }


def fit_reflectivity_power_law(
    freq: np.ndarray,
    amp: np.ndarray,
    f_low: float,
    f_high: float,
) -> tuple[float, float]:
    """Least-squares fit of ``log|R| = a + beta*log(f)`` over the design band.

    Returns ``(beta, a)``.  This is the heart of coloured inversion: the earth's
    reflectivity spectrum is not white but follows a power law, and beta is what
    the seismic spectrum has to be reshaped toward.
    """
    freq = np.asarray(freq, dtype=float)
    amp = np.asarray(amp, dtype=float)
    band = (freq >= max(f_low, 1e-3)) & (freq <= f_high) & (amp > 0) & np.isfinite(amp)
    if band.sum() < 4:
        raise ValueError(f"design band {f_low}-{f_high} Hz has too few valid spectral points")
    x = np.log(freq[band])
    y = np.log(amp[band])
    beta, a = np.polyfit(x, y, 1)
    return float(beta), float(a)


def design_colour_operator(
    volume,
    ties: Sequence,
    f_low: float = 8.0,
    f_high: float = 60.0,
    operator_length_ms: float = 200.0,
    max_traces: int = 400,
    taper: float = 0.20,
    white_noise_pct: float = 2.0,
    smooth_hz: float = 4.0,
    seed: int = 0,
) -> ColourOperator:
    """Design the operator that reshapes the seismic spectrum to the earth's.

    Steps, following Lancaster & Whitcombe:

    1. Average amplitude spectrum of a random sample of live seismic traces,
       smoothed over ``smooth_hz``.  Smoothing matters: the operator divides by
       this spectrum, so an unsmoothed one turns every interference notch into
       a spike of gain.
    2. Average amplitude spectrum of the well reflectivity series.
    3. Power-law fit ``|R(f)| ~ f^beta`` over the design band.
    4. Operator amplitude = target / seismic, stabilised by white noise and
       tapered to zero outside the design band so we never boost pure noise.
    5. Zero-phase operator by inverse FFT; the -90 degree rotation and the
       ``1/f`` of the classic single-step operator are supplied instead by the
       explicit integration step in :func:`coloured_inversion`, which is
       mathematically equivalent and easier to QC.
    """
    dt = volume.dt
    flat = volume.flat_data()
    live = np.flatnonzero(volume.live_mask())
    if live.size == 0:
        raise ValueError("volume contains no live traces")
    rng = np.random.default_rng(seed)
    pick = live if live.size <= max_traces else rng.choice(live, size=max_traces, replace=False)

    n_fft = int(2 ** np.ceil(np.log2(max(flat.shape[1], 2)))) * 2
    freq, seis_amp = utils.average_amplitude_spectrum(flat[pick, :], dt, pad=n_fft)
    seis_amp = _smooth_spectrum(seis_amp, freq, smooth_hz)

    refl_spectra = []
    for tie in ties:
        r = np.nan_to_num(np.asarray(tie.reflectivity, dtype=float))
        if np.allclose(r, 0):
            continue
        _, a = utils.amplitude_spectrum(r, dt, pad=n_fft)
        refl_spectra.append(a)
    if not refl_spectra:
        raise ValueError(
            "coloured inversion needs well reflectivity to fit the target spectrum -- "
            "no located, tied well produced a non-zero reflectivity series"
        )
    refl_amp = np.mean(np.stack(refl_spectra), axis=0)

    beta, intercept = fit_reflectivity_power_law(freq, refl_amp, f_low, f_high)
    with np.errstate(divide="ignore"):
        target = np.exp(intercept) * np.power(np.maximum(freq, 1e-6), beta)

    # Stabilised spectral division.
    eps = (white_noise_pct / 100.0) * float(np.max(seis_amp))
    op_amp = target / (seis_amp + eps)

    # Match overall level to the seismic band so amplitudes stay interpretable,
    # then band-limit with a cosine taper into and out of the design band.
    band = (freq >= f_low) & (freq <= f_high)
    if band.any() and np.max(op_amp[band]) > 0:
        op_amp = op_amp / np.max(op_amp[band])
    op_amp = op_amp * _cosine_band_taper(freq, f_low, f_high)
    op_amp[~np.isfinite(op_amp)] = 0.0

    op = np.fft.fftshift(np.fft.irfft(op_amp, n=n_fft))
    n_op = int(round(operator_length_ms / 1000.0 / dt))
    n_op = max(n_op + 1 if n_op % 2 == 0 else n_op, 9)
    n_op = min(n_op, n_fft - 1)
    centre, half = n_fft // 2, n_op // 2
    op = op[centre - half: centre + half + 1] * utils.taper_window(n_op, taper)

    return ColourOperator(
        samples=op, dt=dt, exponent=beta, intercept=intercept, f_low=f_low, f_high=f_high,
        seismic_freq=freq, seismic_amp=seis_amp, target_amp=target, operator_amp=op_amp,
        n_wells=len(refl_spectra),
    )


def _smooth_spectrum(amp: np.ndarray, freq: np.ndarray, smooth_hz: float) -> np.ndarray:
    """Running mean over a frequency window, to take the notches out."""
    if smooth_hz <= 0 or freq.size < 4:
        return amp
    df = float(freq[1] - freq[0])
    win = max(int(round(smooth_hz / df)), 1)
    if win < 2:
        return amp
    # Reflect-pad so the smoothing does not pull the band edges toward zero.
    pad = win // 2
    padded = np.concatenate([amp[pad:0:-1], amp, amp[-2:-pad - 2:-1]]) if pad > 0 else amp
    smoothed = utils.convolve_same(padded, np.ones(win) / win)
    return smoothed[pad: pad + amp.size] if pad > 0 else smoothed


def _cosine_band_taper(freq: np.ndarray, f_low: float, f_high: float, roll: float = 0.35) -> np.ndarray:
    """Cosine roll-on/roll-off so the operator dies smoothly outside the band."""
    freq = np.asarray(freq, dtype=float)
    w = np.zeros_like(freq)
    lo_roll = max(f_low * roll, 1.0)
    hi_roll = max((freq.max() - f_high) * roll, 1.0)

    inside = (freq >= f_low) & (freq <= f_high)
    w[inside] = 1.0

    ramp_lo = (freq >= f_low - lo_roll) & (freq < f_low)
    if ramp_lo.any():
        w[ramp_lo] = 0.5 * (1 - np.cos(np.pi * (freq[ramp_lo] - (f_low - lo_roll)) / lo_roll))

    ramp_hi = (freq > f_high) & (freq <= f_high + hi_roll)
    if ramp_hi.any():
        w[ramp_hi] = 0.5 * (1 + np.cos(np.pi * (freq[ramp_hi] - f_high) / hi_roll))
    return w


def calibrate_colour_operator(
    op: ColourOperator,
    volume,
    ties: Sequence,
    t_min: float | None = None,
    t_max: float | None = None,
) -> ColourOperator:
    """Scale the operator so its relative impedance matches the wells'.

    Coloured inversion recovers the *shape* of the relative impedance but its
    absolute level depends on how the seismic was scaled, which is arbitrary.
    Fitting one scalar against the band-limited well log-impedance is what
    makes the output splice sensibly onto the low-frequency model.
    """
    twt = np.asarray(volume.twt, dtype=float)
    i0 = int(np.searchsorted(twt, t_min)) if t_min is not None else 0
    i1 = int(np.searchsorted(twt, t_max)) if t_max is not None else twt.size
    dt = op.dt

    num = 0.0
    den = 0.0
    n_used = 0
    for tie in ties:
        trace = np.nan_to_num(np.asarray(tie.seismic, dtype=float))
        ai = np.asarray(tie.ai, dtype=float)
        good = np.isfinite(ai) & (ai > 0)
        if good.sum() < 32:
            continue
        # Band-limit the well log-impedance to the operator's design band so we
        # compare like with like: the operator cannot produce anything outside it.
        target = utils.bandpass(np.log(utils.fill_nan_1d(np.where(good, ai, np.nan))),
                                dt, op.f_low, op.f_high)
        shaped = utils.convolve_same(trace, op.samples)
        rel = utils.relative_impedance(shaped - shaped.mean())
        rel = utils.bandpass(rel, dt, op.f_low, op.f_high)

        a_seg, b_seg = rel[i0:i1], target[i0:i1]
        num += float(np.dot(a_seg, b_seg))
        den += float(np.dot(a_seg, a_seg))
        n_used += 1

    scaled = ColourOperator(**{**op.__dict__})
    if n_used == 0 or den <= 0 or not np.isfinite(num / den) or num / den == 0:
        scaled.scalar = 1.0
        return scaled

    a = num / den
    scaled.samples = op.samples * a
    scaled.scalar = float(a)
    scaled.n_calibration_wells = n_used
    return scaled


def coloured_inversion(
    trace: np.ndarray,
    wavelet: np.ndarray | None = None,
    low_freq_trace: np.ndarray | None = None,
    operator: ColourOperator | np.ndarray | None = None,
    dt: float = 0.002,
    merge_freq: float = 10.0,
    **_ignored,
) -> dict:
    """Apply the shaping operator, then integrate to relative impedance.

    ``wavelet`` is accepted for interface compatibility but unused: coloured
    inversion is deliberately wavelet-free, which is exactly why it is robust
    enough to be the first-pass, full-volume method.
    """
    trace = np.nan_to_num(np.asarray(trace, dtype=float))
    if operator is None:
        raise ValueError("coloured inversion needs a designed operator (see design_colour_operator)")
    op = operator.samples if isinstance(operator, ColourOperator) else np.asarray(operator, dtype=float)
    if isinstance(operator, ColourOperator):
        dt = operator.dt

    shaped = utils.convolve_same(trace, op)

    # Remove any DC the shaping introduced before integrating; a constant here
    # becomes a linear ramp in the integrated trace.
    shaped = shaped - shaped.mean()
    rel = utils.relative_impedance(shaped)

    absolute = None
    if low_freq_trace is not None:
        lf = np.asarray(low_freq_trace, dtype=float)
        if lf.size == rel.size and np.isfinite(lf).any():
            try:
                absolute = utils.merge_with_low_frequency(rel, lf, dt, merge_freq)
            except ValueError:
                absolute = None

    # Coloured inversion never forms a wavelet synthetic, so there is no data
    # residual to report.  Returning NaN is honest; returning zero would look
    # like a perfect fit in the QC panels.
    return {
        "method": "coloured",
        "reflectivity": shaped,
        "relative_ai": rel,
        "absolute_ai": absolute,
        "synthetic": shaped,
        "residual": np.zeros_like(trace),
        "misfit": float("nan"),
        "correlation": utils.normalised_correlation(shaped, trace),
    }


# ==========================================================================
# 2. Sparse-spike inversion  (L1-regularised deconvolution via IRLS)
# ==========================================================================

def sparse_spike_inversion(
    trace: np.ndarray,
    wavelet: np.ndarray,
    low_freq_trace: np.ndarray | None = None,
    dt: float = 0.002,
    sparsity: float = 0.05,
    n_iter: int = 12,
    tol: float = 1e-4,
    eps: float = 1e-4,
    merge_freq: float = 10.0,
    solver: str = "banded",
    **_ignored,
) -> dict:
    """Solve ``min ||W r - s||^2 + lambda ||r||_1`` by iteratively reweighted
    least squares.

    IRLS replaces the L1 term with a weighted L2 term ``sum(r_i^2 / |r_i|)``,
    re-deriving the weights each iteration.  A handful of iterations is enough:
    the weights concentrate energy onto a sparse set of spikes and the rest of
    the reflectivity is driven toward zero.

    ``solver`` selects the linear solver for each IRLS pass: ``banded`` (a
    banded Cholesky -- the fast default), ``cg`` or ``direct``.

    ``sparsity`` is relative -- it is scaled by ``trace(W^T W)/n`` internally so
    the same slider position means the same thing whatever the data amplitude.
    """
    trace = np.nan_to_num(np.asarray(trace, dtype=float))
    w = np.nan_to_num(np.asarray(wavelet, dtype=float))
    n = trace.size
    if n < 8:
        raise ValueError("trace is too short to invert")

    W, WtW, ab_base = _OPERATORS.banded_normal(w, n)
    Wts = W.T @ trace
    n_band = ab_base.shape[0] - 1

    # Scale-free regularisation weight.
    scale = float(WtW.diagonal().mean()) or 1.0
    lam = float(sparsity) * scale

    # Iteration 1 is a plain damped least-squares (all weights equal); from
    # then on the weights are 1/|r|, which is what turns the L2 penalty into
    # an L1 one.  The weight vector is normalised to unit mean each pass so
    # ``sparsity`` keeps the same meaning from the first iteration to the last
    # -- only the *relative* weighting drives the sparsity.
    r = np.zeros(n)
    iterations = 0
    for iterations in range(1, int(n_iter) + 1):
        if iterations == 1:
            weights = np.ones(n)
        else:
            floor = max(eps * float(np.max(np.abs(r))), 1e-12)
            weights = 1.0 / np.maximum(np.abs(r), floor)
            weights = weights / weights.mean()

        r_new = None
        if solver == "banded":
            ab = ab_base.copy()
            ab[n_band, :] += lam * weights
            try:
                r_new = solveh_banded(ab, Wts, lower=False, check_finite=False)
            except Exception:  # noqa: BLE001 - not positive definite; fall back
                r_new = None
        if r_new is None:
            A = (WtW + lam * sparse.diags(weights)).tocsr()
            if solver == "direct":
                r_new = spsolve(A, Wts)
            else:
                r_new, _ = cg(A, Wts, x0=r, rtol=1e-6, maxiter=400)
        r_new = np.nan_to_num(r_new)

        change = np.linalg.norm(r_new - r) / max(np.linalg.norm(r_new), 1e-12)
        r = r_new
        if change < tol:
            break

    r = np.clip(r, -0.95, 0.95)
    return _finalise(
        trace=trace, reflectivity=r, wavelet=w, low_freq_trace=low_freq_trace,
        dt=dt, merge_freq=merge_freq, method="sparse-spike",
        extra={"iterations": iterations, "lambda": lam,
               "n_spikes": int(np.sum(np.abs(r) > 0.10 * max(np.max(np.abs(r)), 1e-12))),
               "sparsity_ratio": _sparsity_ratio(r)},
    )


def select_sparsity(
    traces: np.ndarray,
    wavelet: np.ndarray,
    dt: float = 0.002,
    noise_pct: float = 10.0,
    lo: float = 1e-5,
    hi: float = 1e5,
    n_bisect: int = 16,
    max_traces: int = 12,
    seed: int = 0,
    **params,
) -> dict:
    """Choose the sparsity weight from the noise level, not from a slider.

    Sparse-spike has no natural scale for ``sparsity``: too small and it fits
    the noise, too large and it returns nothing.  On real data that is not a
    cosmetic choice.  Tested against Penobscot L-30, an under-regularised run
    drove the seismic residual to 0.5% -- an almost exact fit -- and scored
    *below* the background model at the well, because everything it added above
    the background was noise.

    Morozov's discrepancy principle gives the missing scale: stop fitting when
    the residual is as large as the noise is believed to be, and no further.
    ``noise_pct`` is the noise standard deviation as a percentage of trace RMS,
    the same convention the Bayesian engine uses.  The residual grows
    monotonically with ``sparsity``, so a bisection in log-space finds the
    crossing in a dozen solves.

    The search runs on a sample of traces rather than all of them: the answer
    is a property of the wavelet and the noise level, not of one trace.

    The bracket spans ten decades on purpose.  IRLS renormalises its weights to
    unit mean each pass, which is what keeps ``sparsity`` meaning the same thing
    from the first iteration to the last, but it also flattens the response: on
    the synthetic case the residual moves only from 0.7% to 6.6% between
    ``1e-5`` and ``5``, and needs ``1e5`` before it approaches the trace energy.
    A bracket sized by intuition rather than measurement silently returns its
    own endpoint.
    """
    traces = np.atleast_2d(np.asarray(traces, dtype=float))
    if traces.shape[0] > max_traces:
        rng = np.random.default_rng(seed)
        keep = rng.choice(traces.shape[0], size=max_traces, replace=False)
        traces = traces[keep]
    live = traces[np.any(np.abs(traces) > 0, axis=1)]
    if live.size == 0:
        return {"sparsity": float(lo), "note": "no live traces; sparsity left at the floor"}

    target = float(noise_pct) / 100.0
    params = {k: v for k, v in params.items() if k not in ("sparsity", "low_freq_trace")}

    def misfit(value: float) -> float:
        got = []
        for tr in live:
            out = sparse_spike_inversion(tr, wavelet, None, dt=dt, sparsity=value, **params)
            got.append(out["misfit"])
        return float(np.nanmean(got))

    lo_f, hi_f = float(lo), float(hi)
    m_lo, m_hi = misfit(lo_f), misfit(hi_f)
    evaluations = 2
    if m_lo > target:
        return {"sparsity": lo_f, "achieved_pct": m_lo * 100.0, "target_pct": target * 100.0,
                "evaluations": evaluations,
                "note": "even the weakest regularisation leaves more residual than the stated "
                        "noise; the wavelet or the tie is the limit, not the sparsity"}
    if m_hi < target:
        return {"sparsity": hi_f, "achieved_pct": m_hi * 100.0, "target_pct": target * 100.0,
                "evaluations": evaluations,
                "note": "even the strongest regularisation fits closer than the stated noise"}

    for _ in range(int(n_bisect)):
        mid = float(np.sqrt(lo_f * hi_f))         # geometric bisection
        m_mid = misfit(mid)
        evaluations += 1
        if m_mid < target:
            lo_f = mid
        else:
            hi_f = mid
        if abs(m_mid - target) < 0.002:
            break
    chosen = float(np.sqrt(lo_f * hi_f))
    return {"sparsity": chosen, "achieved_pct": misfit(chosen) * 100.0,
            "target_pct": target * 100.0, "evaluations": evaluations + 1,
            "traces_used": int(live.shape[0]),
            "note": "chosen by the discrepancy principle: residual matched to the noise level"}


def _sparsity_ratio(r: np.ndarray) -> float:
    """Effective fraction of samples carrying energy, via the L1/L2 ratio.

    ``(sum|r|)^2 / (n * sum r^2)`` is 1 for a perfectly flat series and 1/n for
    a single spike, so it reads directly as "how much of the trace is active".
    """
    r = np.asarray(r, dtype=float)
    l2 = float(np.sum(r ** 2))
    if l2 <= 0:
        return 0.0
    return float(np.sum(np.abs(r)) ** 2 / (r.size * l2))


# ==========================================================================
# 3. Model-based inversion  (Russell & Hampson generalized linear inversion)
# ==========================================================================

def model_based_inversion(
    trace: np.ndarray,
    wavelet: np.ndarray,
    low_freq_trace: np.ndarray | None = None,
    dt: float = 0.002,
    model_weight: float = 0.10,
    roughness_weight: float = 0.02,
    max_iter: int = 60,
    max_change: float = 0.35,
    merge_freq: float = 10.0,
    **_ignored,
) -> dict:
    """Perturb log-impedance until the synthetic matches the trace.

    Objective (all terms in log-impedance ``m = ln AI``)::

        J(m) = 0.5 ||C D m - s||^2
             + 0.5 * mu  ||m - m0||^2          (stay near the background model)
             + 0.5 * eta ||L m||^2             (stay smooth)

    ``C`` is the wavelet convolution matrix, ``D`` the reflectivity operator and
    ``L`` a second-difference roughness operator.  The problem is quadratic, so
    the analytic gradient below is exact and L-BFGS-B converges quickly.

    ``max_change`` bounds each sample's departure from the background model (in
    natural-log units, so 0.35 is roughly +/-42% in impedance), which is the
    "hard constraint" of a Hampson-Russell style run.
    """
    trace = np.nan_to_num(np.asarray(trace, dtype=float))
    w = np.nan_to_num(np.asarray(wavelet, dtype=float))
    n = trace.size
    if low_freq_trace is None:
        raise ValueError("model-based inversion requires a low-frequency model trace")

    lf = np.asarray(low_freq_trace, dtype=float)
    if lf.size != n:
        raise ValueError(f"low-frequency trace length {lf.size} does not match the seismic trace ({n})")
    lf = utils.fill_nan_1d(np.where(np.isfinite(lf) & (lf > 0), lf, np.nan))
    if not np.isfinite(lf).all():
        raise ValueError("low-frequency model trace has no valid samples")

    m0 = np.log(lf)
    _, D, G = _OPERATORS.get(w, n)

    # Scale the penalties against the data term so the weights are dimensionless.
    data_scale = float(np.mean(trace ** 2)) or 1.0
    model_scale = max(float(np.var(m0)), 1e-8)
    mu = float(model_weight) * data_scale / model_scale
    L = second_difference_matrix(n)
    eta = float(roughness_weight) * data_scale / model_scale
    LtL = (L.T @ L).tocsr()

    def objective(m):
        resid = G @ m - trace
        dm = m - m0
        Lm = L @ m
        j = 0.5 * float(resid @ resid) + 0.5 * mu * float(dm @ dm) + 0.5 * eta * float(Lm @ Lm)
        grad = (G.T @ resid) + mu * dm + eta * (LtL @ m)
        return j, np.asarray(grad, dtype=float)

    bounds = [(m0[i] - max_change, m0[i] + max_change) for i in range(n)] if max_change else None

    res = minimize(
        objective, x0=m0.copy(), jac=True, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": int(max_iter), "ftol": 1e-12, "gtol": 1e-10},
    )
    m = np.asarray(res.x, dtype=float)

    reflectivity = np.asarray(D @ m, dtype=float)
    absolute = np.exp(m)
    synthetic = np.asarray(G @ m, dtype=float)

    qc = _qc(trace, synthetic)
    return {
        "method": "model-based",
        "reflectivity": reflectivity,
        "relative_ai": m - np.mean(m),
        "absolute_ai": absolute,
        "log_impedance": m,
        "background_log_impedance": m0,
        "iterations": int(res.nit),
        "converged": bool(res.success),
        "objective": float(res.fun),
        "message": str(res.message),
        **qc,
    }


# ==========================================================================
# 4. Bayesian linear inversion  (closed-form Gaussian posterior)
# ==========================================================================

def _add_banded(ab_wide: np.ndarray, ab_narrow: np.ndarray) -> np.ndarray:
    """Add a narrow banded matrix into a wider one, aligning the diagonals.

    ``ab_wide`` must be at least as wide as ``ab_narrow``; both are in LAPACK
    upper-banded form, where the last row holds the main diagonal.
    """
    if ab_narrow.shape[0] > ab_wide.shape[0]:
        raise ValueError("ab_wide must have at least as many bands as ab_narrow")
    out = np.array(ab_wide, dtype=float, copy=True)
    u_w = out.shape[0] - 1
    u_n = ab_narrow.shape[0] - 1
    for k in range(u_n + 1):
        out[u_w - k, :] += ab_narrow[u_n - k, :]
    return out


def _prior_precision_banded(n: int, prior_std: float, smoothness: float) -> np.ndarray:
    """Prior precision on log-impedance, in LAPACK upper-banded form.

    ``Q = (1 / prior_std^2) * (I + smoothness * L'L)`` -- two statements at
    once: log-impedance stays within ``prior_std`` of the background model, and
    its curvature is penalised in proportion to ``smoothness``.  Both terms are
    banded, which is what keeps the whole posterior solvable in ``O(n b^2)``.
    """
    a = 1.0 / max(float(prior_std), 1e-9) ** 2
    ident = np.zeros((1, n))
    ident[0, :] = a
    if not smoothness or smoothness <= 0 or n < 3:
        return ident
    L = second_difference_matrix(n)
    return _add_banded(_symmetric_to_banded((L.T @ L).tocsr(), n) * (a * float(smoothness)), ident)


def bayesian_inversion(
    trace: np.ndarray,
    wavelet: np.ndarray,
    low_freq_trace: np.ndarray | None = None,
    dt: float = 0.002,
    prior_std: float = 0.08,
    smoothness: float = 0.05,
    noise_pct: float = 10.0,
    uncertainty: bool = True,
    **_ignored,
) -> dict:
    """Closed-form Gaussian posterior for log-impedance.

    With a linear operator, a Gaussian prior and Gaussian noise the posterior is
    Gaussian and available in closed form -- no iteration, and, unlike every
    other engine here, what comes back is a *distribution* rather than a single
    answer::

        A     = G' G / sigma_d^2 + Q      posterior precision
        m     = A^-1 (G' d / sigma_d^2 + Q_amp m0)
        Cpost = A^-1

    ``Q = (I + smoothness * L'L) / prior_std^2`` says two things: log-impedance
    stays within ``prior_std`` of the background model, and its curvature is
    penalised.  Only the amplitude part carries the background mean, so the
    effective prior mean is a mildly smoothed ``m0`` rather than ``m0`` itself.
    That is a proper Gaussian prior, but it is only *near* the background while
    ``smoothness`` stays small -- at large values the prior mean drifts away
    from the background and the posterior drifts with it, which is why the
    default is 0.05 and ``prior_drift`` is reported back.

    ``noise_pct`` is the noise standard deviation as a percentage of the trace
    RMS; it sets how far the data is allowed to pull the answer off the prior.

    Measured on the synthetic case against the four wells, the defaults give
    band-limited correlation on a par with the model-based engine while
    recovering absolute impedance appreciably better (log-impedance RMSE 0.09
    against 0.14), and they carry a posterior standard deviation with them.
    """
    trace = np.nan_to_num(np.asarray(trace, dtype=float))
    w = np.nan_to_num(np.asarray(wavelet, dtype=float))
    n = trace.size
    if low_freq_trace is None:
        raise ValueError("Bayesian inversion requires a low-frequency model trace (the prior mean)")

    lf = np.asarray(low_freq_trace, dtype=float)
    if lf.size != n:
        raise ValueError(f"low-frequency trace length {lf.size} does not match the trace ({n})")
    lf = utils.fill_nan_1d(np.where(np.isfinite(lf) & (lf > 0), lf, np.nan))
    if not np.isfinite(lf).all():
        raise ValueError("low-frequency model trace has no valid samples")
    m0 = np.log(lf)

    G, _GtG, ab_G = _OPERATORS.banded_model(w, n)

    rms = float(np.sqrt(np.mean(trace ** 2)))
    sigma_d = max(rms * float(noise_pct) / 100.0, 1e-12)
    inv_var_d = 1.0 / sigma_d ** 2

    Q = _prior_precision_banded(n, prior_std, smoothness)
    ab = _add_banded(ab_G * inv_var_d, Q)
    rhs = (G.T @ trace) * inv_var_d + m0 / max(float(prior_std), 1e-9) ** 2

    from scipy.linalg import cho_solve_banded, cholesky_banded

    factor = cholesky_banded(ab, lower=False, check_finite=False)
    m = cho_solve_banded((factor, False), rhs, check_finite=False)

    posterior_std = None
    if uncertainty:
        # diag(A^-1) via the Takahashi recursion: O(n b^2) rather than the
        # O(n^2 b) flops and O(n^2) memory of back-substituting against a full
        # identity, which is what makes long traces affordable.
        posterior_std = np.sqrt(np.clip(selected_inverse_diagonal(factor), 0.0, None))

    reflectivity = np.asarray(_OPERATORS.get(w, n)[1] @ m, dtype=float)
    synthetic = np.asarray(G @ m, dtype=float)

    out = {
        "method": "bayesian",
        "reflectivity": reflectivity,
        "relative_ai": m - float(np.mean(m)),
        "absolute_ai": np.exp(m),
        "log_impedance": m,
        "background_log_impedance": m0,
        "prior_std": float(prior_std),
        "noise_std": sigma_d,
        # How far the answer wandered from the background, in natural-log
        # units: 0.1 is about 10%, and anything approaching 1 means the
        # smoothness weight has pulled the prior mean off the background.
        "prior_drift": float(np.max(np.abs(m - m0))),
        **_qc(trace, synthetic),
    }
    if posterior_std is not None:
        # Log-impedance is Gaussian, so impedance is log-normal: the quantiles
        # are exponentials of the Gaussian ones, not mean +/- k * sd.
        z90 = 1.2815515655446004
        out["posterior_std"] = posterior_std
        out["ai_p10"] = np.exp(m - z90 * posterior_std)
        out["ai_p90"] = np.exp(m + z90 * posterior_std)
        out["uncertainty_reduction"] = float(
            1.0 - np.mean(posterior_std) / max(float(prior_std), 1e-12))
    return out


def selected_inverse_diagonal(factor: np.ndarray) -> np.ndarray:
    """``diag(A^-1)`` from the banded Cholesky factor of ``A``, without forming it.

    The obvious way to get the posterior variance is to back-substitute against
    the identity, but that computes all ``n^2`` entries of the inverse to keep
    ``n`` of them: it costs ``O(n^2 b)`` flops and ``O(n^2)`` memory, so it goes
    quadratic exactly when traces get long.  Measured on a 31-band system it
    took 16 ms at n=701 and 180 ms at n=1501.

    The Takahashi recursion computes only the entries of the inverse that lie
    inside the factor's band, which is ``O(n b^2)``.  Writing ``A = U' U`` with
    ``U`` upper triangular, ``U * Sigma`` equals ``U'^-1``, which is zero above
    the diagonal and ``1/U_ii`` on it, so for ``j >= i``::

        Sigma_ij = ( [i == j] / U_ii  -  sum_{k>i} U_ik Sigma_kj ) / U_ii

    Rows are filled from the bottom up, and every ``Sigma_kj`` the sum needs is
    already known (using symmetry where ``k > j``).  The band of ``Sigma`` is
    carried as a small dense sliding window so each row costs one matrix-vector
    product rather than ``b`` dot products -- without that the Python loop
    overhead cancels the flop saving.

    ``factor`` is scipy's upper banded form, as returned by ``cholesky_banded``.
    """
    factor = np.asarray(factor, dtype=float)
    b = factor.shape[0] - 1
    n = factor.shape[1]
    diag_u = factor[b, :]
    if b == 0:
        return 1.0 / np.square(diag_u)

    # window[a, c] = Sigma[i + 1 + a, i + 1 + c] as row i is being computed.
    # Entries that would index past the last sample stay zero; the matching
    # U values are zero too, so they never reach a live result.
    window = np.zeros((b, b))
    out = np.empty(n)
    u = np.zeros(b)

    for i in range(n - 1, -1, -1):
        u[:] = 0.0
        k_max = min(i + b, n - 1)
        m = k_max - i
        if m > 0:
            ks = np.arange(i + 1, k_max + 1)
            u[:m] = factor[b + i - ks, ks]

        row = np.empty(b + 1)                 # row[c] = Sigma[i, i + c]
        row[1:] = -(u @ window) / diag_u[i] if m > 0 else 0.0
        row[0] = (1.0 / diag_u[i] - float(u @ row[1:])) / diag_u[i]
        out[i] = row[0]

        nxt = np.empty((b, b))
        nxt[0, :] = row[:b]
        nxt[1:, 0] = row[1:b]
        nxt[1:, 1:] = window[:b - 1, :b - 1]
        window = nxt

    return out


def bayesian_realisations(
    trace: np.ndarray,
    wavelet: np.ndarray,
    low_freq_trace: np.ndarray,
    n_realisations: int = 20,
    seed: int = 0,
    dt: float = 0.002,
    prior_std: float = 0.08,
    smoothness: float = 0.05,
    noise_pct: float = 10.0,
    **_ignored,
) -> dict:
    """Draw equiprobable impedance realisations from the Bayesian posterior.

    The point estimate is the posterior *mean*: smooth, band-limited, and by
    construction the one answer no realisation actually looks like.  Sampling
    the same posterior gives models that are all consistent with the seismic
    and the prior, and whose spread *is* the uncertainty -- which is what you
    want before quoting a thickness, a contact, or a volume.

    Sampling is almost free once the factorisation exists.  With ``A = U' U``
    and ``z`` standard normal, ``U^-1 z`` has covariance ``U^-1 U'^-1 = A^-1``,
    so each realisation is one banded triangular solve::

        m_sample = m_map + solve(U, z)

    No Gibbs sampler, no burn-in, no convergence to argue about: these are
    independent exact draws, because the posterior really is Gaussian.
    """
    from scipy.linalg import cho_solve_banded, cholesky_banded, solve_banded

    trace = np.nan_to_num(np.asarray(trace, dtype=float))
    w = np.nan_to_num(np.asarray(wavelet, dtype=float))
    n = trace.size
    lf = np.asarray(low_freq_trace, dtype=float)
    if lf.size != n:
        raise ValueError(f"low-frequency trace length {lf.size} does not match the trace ({n})")
    lf = utils.fill_nan_1d(np.where(np.isfinite(lf) & (lf > 0), lf, np.nan))
    if not np.isfinite(lf).all():
        raise ValueError("low-frequency model trace has no valid samples")
    m0 = np.log(lf)

    G, _GtG, ab_G = _OPERATORS.banded_model(w, n)
    rms = float(np.sqrt(np.mean(trace ** 2)))
    sigma_d = max(rms * float(noise_pct) / 100.0, 1e-12)
    inv_var_d = 1.0 / sigma_d ** 2

    Q = _prior_precision_banded(n, prior_std, smoothness)
    ab = _add_banded(ab_G * inv_var_d, Q)
    rhs = (G.T @ trace) * inv_var_d + m0 / max(float(prior_std), 1e-9) ** 2

    factor = cholesky_banded(ab, lower=False, check_finite=False)
    m_map = cho_solve_banded((factor, False), rhs, check_finite=False)

    b = factor.shape[0] - 1
    rng = np.random.default_rng(seed)
    draws = np.empty((int(n_realisations), n), dtype=np.float32)
    for k in range(int(n_realisations)):
        z = rng.standard_normal(n)
        draws[k] = m_map + solve_banded((0, b), factor, z, check_finite=False)

    ai = np.exp(draws.astype(float))
    return {
        "method": "bayesian-realisations",
        "log_impedance": draws,
        "absolute_ai": ai,
        "mean_ai": np.exp(m_map),
        "p10": np.percentile(ai, 10, axis=0),
        "p50": np.percentile(ai, 50, axis=0),
        "p90": np.percentile(ai, 90, axis=0),
        "spread": float(np.mean(np.std(draws, axis=0))),
        "n_realisations": int(n_realisations),
        "noise_std": sigma_d,
    }


def coupled_bayesian_volume(
    volume,
    wavelet: np.ndarray,
    low_freq_model,
    lateral_weight: float = 0.5,
    n_sweeps: int = 3,
    il_range: tuple[int, int] | None = None,
    xl_range: tuple[int, int] | None = None,
    progress: Callable[[float, str], None] | None = None,
    skip_dead: bool = True,
    uncertainty: bool = False,
    **params,
) -> InversionResult:
    """Bayesian inversion with the traces solved *together*, not one by one.

    Every other engine here treats each trace as an independent 1D problem, so
    nothing stops two neighbouring traces disagreeing by more than the seismic
    can justify.  That is what vertical striping in a noisy inverted section
    actually is: independent estimation errors, side by side.

    The fix is a prior that knows the traces are neighbours.  Adding a lateral
    roughness term to the joint precision gives::

        A_total = blockdiag(A_i) + lambda_lat * (Laplacian_lateral kron I)

    which is far too large to factor directly -- it is (traces x samples)
    square.  But it is block-sparse, so a block Gauss-Seidel sweep solves it
    one trace at a time using the *current* estimate of that trace's
    neighbours::

        (A_i + lambda_lat * deg_i I) m_i = rhs_i + lambda_lat * sum_j m_j

    Each block solve is the same banded Cholesky as the 1D engine, so a run
    costs ``n_sweeps`` times the 1D run and converges monotonically: the sweep
    is coordinate descent on a convex quadratic.

    ``lateral_weight`` is relative to the prior precision, so 0 reproduces the
    independent 1D result exactly and larger values buy lateral continuity at
    the price of resolution.
    """
    from scipy.linalg import cho_solve_banded, cholesky_banded

    n_il, n_xl, n_t = volume.data.shape
    il_sl = slice(*il_range) if il_range else slice(0, n_il)
    xl_sl = slice(*xl_range) if xl_range else slice(0, n_xl)
    sub = np.asarray(volume.data[il_sl, xl_sl, :], dtype=float)
    s_il, s_xl = sub.shape[0], sub.shape[1]
    if low_freq_model is None:
        raise ValueError("spatially-coupled inversion requires a low-frequency model")
    lf_cube = np.asarray(low_freq_model.ai[il_sl, xl_sl, :], dtype=float)
    if lf_cube.shape != sub.shape:
        raise ValueError("low-frequency model does not match the seismic geometry")

    prior_std = float(params.get("prior_std", 0.08))
    smoothness = float(params.get("smoothness", 0.05))
    noise_pct = float(params.get("noise_pct", 10.0))
    dt = float(params.get("dt", volume.dt))

    w = np.nan_to_num(np.asarray(wavelet, dtype=float))
    G, _GtG, ab_G = _OPERATORS.banded_model(w, n_t)
    Q = _prior_precision_banded(n_t, prior_std, smoothness)

    live = np.max(np.abs(sub), axis=2) > 0 if skip_dead else np.ones((s_il, s_xl), bool)
    lf_safe = np.where(np.isfinite(lf_cube) & (lf_cube > 0), lf_cube, np.nan)
    m0 = np.log(np.where(np.isfinite(lf_safe), lf_safe, np.nanmean(lf_safe)))
    m = m0.copy()

    # Per-trace data terms are fixed across sweeps; only the neighbour term moves.
    inv_var = np.zeros((s_il, s_xl))
    rhs0 = np.zeros((s_il, s_xl, n_t))
    for i in range(s_il):
        for j in range(s_xl):
            if not live[i, j]:
                continue
            rms = float(np.sqrt(np.mean(sub[i, j] ** 2)))
            sigma_d = max(rms * noise_pct / 100.0, 1e-12)
            inv_var[i, j] = 1.0 / sigma_d ** 2
            rhs0[i, j] = (G.T @ sub[i, j]) * inv_var[i, j] + m0[i, j] / max(prior_std, 1e-9) ** 2

    # Scale the lateral weight to the prior precision so it is dimensionless.
    lam = float(lateral_weight) / max(prior_std, 1e-9) ** 2
    start = time.time()
    total = max(int(n_sweeps), 1) * s_il * s_xl
    done = 0
    for sweep in range(max(int(n_sweeps), 1)):
        for i in range(s_il):
            for j in range(s_xl):
                done += 1
                if not live[i, j]:
                    continue
                neighbours = []
                for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    a, b_ = i + di, j + dj
                    if 0 <= a < s_il and 0 <= b_ < s_xl and live[a, b_]:
                        neighbours.append(m[a, b_])
                deg = len(neighbours)
                ab = _add_banded(ab_G * inv_var[i, j], Q)
                rhs = rhs0[i, j].copy()
                if deg and lam > 0:
                    ab[-1, :] += lam * deg
                    rhs += lam * np.sum(neighbours, axis=0)
                try:
                    factor = cholesky_banded(ab, lower=False, check_finite=False)
                    m[i, j] = cho_solve_banded((factor, False), rhs, check_finite=False)
                except Exception:  # noqa: BLE001 - keep the previous estimate
                    pass
            if progress is not None:
                progress(done / total, f"sweep {sweep + 1}/{n_sweeps}")

    # Assemble the same outputs the 1D path produces.
    _C, D, _CD = _OPERATORS.get(w, n_t)
    rel = np.zeros((s_il, s_xl, n_t), dtype=np.float32)
    refl = np.zeros_like(rel)
    resid = np.zeros_like(rel)
    absolute = np.zeros_like(rel)
    misfit = np.full((s_il, s_xl), np.nan, dtype=np.float32)
    corr = np.full((s_il, s_xl), np.nan, dtype=np.float32)
    posterior = np.zeros_like(rel) if uncertainty else None
    for i in range(s_il):
        for j in range(s_xl):
            if not live[i, j]:
                continue
            mij = m[i, j]
            synthetic = np.asarray(G @ mij, dtype=float)
            rel[i, j] = mij - float(np.mean(mij))
            refl[i, j] = np.asarray(D @ mij, dtype=float)
            resid[i, j] = sub[i, j] - synthetic
            absolute[i, j] = np.exp(mij)
            qc = _qc(sub[i, j], synthetic)
            misfit[i, j] = qc["misfit"]
            corr[i, j] = qc["correlation"]
            if posterior is not None:
                ab = _add_banded(ab_G * inv_var[i, j], Q)
                ab[-1, :] += lam * 4
                factor = cholesky_banded(ab, lower=False, check_finite=False)
                posterior[i, j] = np.sqrt(np.clip(selected_inverse_diagonal(factor), 0.0, None))

    if progress is not None:
        progress(1.0, "coupled run complete")
    out_params = dict(params)
    out_params.update({"lateral_weight": lateral_weight, "n_sweeps": n_sweeps,
                       "dt": dt, "uncertainty": uncertainty})
    return InversionResult(
        method="bayesian", relative_ai=rel, absolute_ai=absolute, reflectivity=refl,
        residual=resid, misfit=misfit, correlation=corr, twt=volume.twt,
        posterior_std=posterior, il_slice=il_sl, xl_slice=xl_sl, params=out_params,
        elapsed_s=time.time() - start,
        is_subset=(il_sl != slice(0, n_il) or xl_sl != slice(0, n_xl)),
    )


def _banded_matvec(ab: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Multiply a symmetric upper-banded matrix by a vector."""
    u = ab.shape[0] - 1
    out = ab[u, :] * x
    for k in range(1, u + 1):
        d = ab[u - k, k:]
        out[:-k] += d * x[k:]
        out[k:] += d * x[:-k]
    return out


# ==========================================================================
# Common dispatcher + volume runner
# ==========================================================================

def invert(
    trace: np.ndarray,
    wavelet: np.ndarray | None = None,
    low_freq_trace: np.ndarray | None = None,
    method: str = "coloured",
    **params,
) -> dict:
    """Common entry point: dispatch one trace to the requested engine."""
    if method == "coloured":
        return coloured_inversion(trace, wavelet, low_freq_trace, **params)
    if method == "sparse-spike":
        if wavelet is None:
            raise ValueError("sparse-spike inversion requires a wavelet")
        return sparse_spike_inversion(trace, wavelet, low_freq_trace, **params)
    if method == "model-based":
        if wavelet is None:
            raise ValueError("model-based inversion requires a wavelet")
        return model_based_inversion(trace, wavelet, low_freq_trace, **params)
    if method == "bayesian":
        if wavelet is None:
            raise ValueError("Bayesian inversion requires a wavelet")
        return bayesian_inversion(trace, wavelet, low_freq_trace, **params)
    raise ValueError(f"unknown inversion method '{method}' (expected one of {METHODS})")


@dataclass
class InversionResult:
    """Volume-level output: the cubes plus the run's QC statistics."""

    method: str
    relative_ai: np.ndarray
    absolute_ai: np.ndarray | None
    reflectivity: np.ndarray
    residual: np.ndarray
    misfit: np.ndarray                    # (n_il, n_xl) per-trace normalised RMS
    correlation: np.ndarray               # (n_il, n_xl)
    twt: np.ndarray
    # Posterior standard deviation of log-impedance, when the engine produced
    # one (Bayesian only).  None for the point-estimate engines.
    posterior_std: np.ndarray | None = None
    il_slice: slice = field(default_factory=lambda: slice(None))
    xl_slice: slice = field(default_factory=lambda: slice(None))
    params: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    is_subset: bool = False

    def summary(self) -> dict:
        corr = self.correlation[np.isfinite(self.correlation)]
        mis = self.misfit[np.isfinite(self.misfit)]
        return {
            "method": self.method,
            "traces inverted": int(self.relative_ai.shape[0] * self.relative_ai.shape[1]),
            "subset run": self.is_subset,
            "mean correlation": f"{corr.mean():.3f}" if corr.size else "n/a",
            "mean misfit": f"{mis.mean():.3f}" if mis.size else "n/a",
            "absolute AI": "yes" if self.absolute_ai is not None else "no (relative only)",
            "uncertainty": "yes" if self.posterior_std is not None else "no",
            "elapsed (s)": f"{self.elapsed_s:.1f}",
        }


def run_volume(
    volume,
    method: str,
    wavelet: np.ndarray | None = None,
    low_freq_model=None,
    il_range: tuple[int, int] | None = None,
    xl_range: tuple[int, int] | None = None,
    progress: Callable[[float, str], None] | None = None,
    chunk_size: int = 64,
    skip_dead: bool = True,
    **params,
) -> InversionResult:
    """Run an engine over the cube (or a subset), trace by trace, in chunks.

    ``il_range`` / ``xl_range`` are *index* ranges into the cube, which is what
    the "preview on a subset" button uses to get an answer in seconds before
    committing to the full volume.  ``progress`` is called with
    ``(fraction, message)`` roughly once per chunk.
    """
    if method not in METHODS:
        raise ValueError(f"unknown inversion method '{method}' (expected one of {METHODS})")

    n_il, n_xl, n_t = volume.data.shape
    il_sl = slice(*il_range) if il_range else slice(0, n_il)
    xl_sl = slice(*xl_range) if xl_range else slice(0, n_xl)
    sub = volume.data[il_sl, xl_sl, :]
    s_il, s_xl = sub.shape[0], sub.shape[1]
    if s_il == 0 or s_xl == 0:
        raise ValueError("the selected inline/crossline subset is empty")

    lf_cube = None
    if low_freq_model is not None:
        lf_cube = low_freq_model.ai[il_sl, xl_sl, :]
        if lf_cube.shape != sub.shape:
            raise ValueError("low-frequency model does not match the seismic geometry")
    if method == "model-based" and lf_cube is None:
        raise ValueError("model-based inversion requires a low-frequency model -- build one first")

    params.setdefault("dt", volume.dt)

    rel = np.zeros((s_il, s_xl, n_t), dtype=np.float32)
    refl = np.zeros_like(rel)
    resid = np.zeros_like(rel)
    absolute = np.zeros_like(rel) if (lf_cube is not None) else None
    posterior = np.zeros_like(rel) if method == "bayesian" and params.get("uncertainty", True) else None
    misfit = np.full((s_il, s_xl), np.nan, dtype=np.float32)
    corr = np.full((s_il, s_xl), np.nan, dtype=np.float32)

    flat = sub.reshape(s_il * s_xl, n_t)
    flat_lf = lf_cube.reshape(s_il * s_xl, n_t) if lf_cube is not None else None
    n_traces = flat.shape[0]

    live = np.ones(n_traces, dtype=bool)
    if skip_dead:
        live = np.nanmax(np.abs(flat), axis=1) > 0

    start = time.time()
    done = 0
    for lo, hi in utils.chunk_indices(n_traces, chunk_size):
        for t in range(lo, hi):
            if not live[t]:
                continue
            lf_trace = flat_lf[t] if flat_lf is not None else None
            try:
                res = invert(flat[t].astype(float), wavelet, lf_trace, method=method, **params)
            except Exception:  # noqa: BLE001 - one bad trace must not kill the run
                continue
            i, j = divmod(t, s_xl)
            rel[i, j, :] = res["relative_ai"]
            refl[i, j, :] = res["reflectivity"]
            resid[i, j, :] = res["residual"]
            if absolute is not None and res.get("absolute_ai") is not None:
                absolute[i, j, :] = res["absolute_ai"]
            if posterior is not None and res.get("posterior_std") is not None:
                posterior[i, j, :] = res["posterior_std"]
            misfit[i, j] = res["misfit"]
            corr[i, j] = res["correlation"]
        done = hi
        if progress is not None:
            progress(done / n_traces, f"{done:,} / {n_traces:,} traces")

    if progress is not None:
        progress(1.0, f"{n_traces:,} traces complete")

    if absolute is not None and not np.any(absolute):
        absolute = None
    if posterior is not None and not np.any(posterior):
        posterior = None

    return InversionResult(
        method=method, relative_ai=rel, absolute_ai=absolute, reflectivity=refl,
        residual=resid, misfit=misfit, correlation=corr, twt=volume.twt,
        posterior_std=posterior,
        il_slice=il_sl, xl_slice=xl_sl, params=dict(params), elapsed_s=time.time() - start,
        is_subset=(il_sl != slice(0, n_il) or xl_sl != slice(0, n_xl)),
    )


def estimate_runtime(
    volume,
    method: str,
    wavelet: np.ndarray | None,
    low_freq_model=None,
    n_probe: int = 8,
    **params,
) -> float:
    """Time a few traces to predict the full-volume runtime (seconds).

    Cheap enough to run before the user commits to a big job, and honest about
    the fact that sparse-spike and model-based are orders slower than coloured.
    """
    flat = volume.flat_data()
    live = np.flatnonzero(volume.live_mask())
    if live.size == 0:
        return 0.0
    probe = live[np.linspace(0, live.size - 1, min(n_probe, live.size)).astype(int)]
    lf_flat = low_freq_model.ai.reshape(-1, volume.data.shape[2]) if low_freq_model is not None else None
    params.setdefault("dt", volume.dt)

    start = time.time()
    n_ok = 0
    for t in probe:
        try:
            invert(flat[t].astype(float), wavelet, lf_flat[t] if lf_flat is not None else None,
                   method=method, **params)
            n_ok += 1
        except Exception:  # noqa: BLE001
            continue
    if n_ok == 0:
        return float("nan")
    per_trace = (time.time() - start) / n_ok
    return per_trace * float(live.size)


def crossplot_at_wells(
    result: InversionResult,
    volume,
    ties: Sequence,
    use_absolute: bool = True,
    t_min: float | None = None,
    t_max: float | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Pair inverted impedance against well AI at each well, for QC.

    Only wells whose trace falls inside the inverted subset are returned; the
    caller reports the rest so a silently-empty crossplot is impossible.
    """
    cube = result.absolute_ai if (use_absolute and result.absolute_ai is not None) else result.relative_ai
    il0 = result.il_slice.start or 0
    xl0 = result.xl_slice.start or 0
    n_il, n_xl = cube.shape[0], cube.shape[1]

    twt = volume.twt
    i0 = int(np.searchsorted(twt, t_min)) if t_min is not None else 0
    i1 = int(np.searchsorted(twt, t_max)) if t_max is not None else twt.size

    out: dict[str, dict[str, np.ndarray]] = {}
    for tie in ties:
        i, j = tie.il_index - il0, tie.xl_index - xl0
        if not (0 <= i < n_il and 0 <= j < n_xl):
            continue
        inverted = np.asarray(cube[i, j, i0:i1], dtype=float)
        well_ai = np.asarray(tie.ai[i0:i1], dtype=float)
        good = np.isfinite(inverted) & np.isfinite(well_ai) & (well_ai > 0)
        if good.sum() < 5:
            continue
        out[tie.well] = {
            "inverted": inverted[good],
            "well": well_ai[good],
            "twt": twt[i0:i1][good],
            "correlation": utils.normalised_correlation(inverted[good], well_ai[good]),
        }
    return out
