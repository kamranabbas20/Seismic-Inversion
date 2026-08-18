"""Shared helpers: unit conversion, filtering, geometry and impedance algebra.

Everything in here is deliberately free of Streamlit imports so the numeric
core can be unit-tested (and re-used from a notebook) without a running app.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import signal
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------
# Unit conversion
# --------------------------------------------------------------------------

FT_PER_M = 3.280839895013123
US_PER_S = 1.0e6


def sonic_to_velocity(dt: np.ndarray, unit: str = "us/ft") -> np.ndarray:
    """Convert a sonic (slowness) log to velocity in m/s.

    ``unit`` is either ``us/ft`` (the usual DT/DTC curve unit) or ``us/m``.
    Zero / negative / non-finite slowness is returned as NaN rather than inf.
    """
    dt = np.asarray(dt, dtype=float)
    out = np.full(dt.shape, np.nan, dtype=float)
    good = np.isfinite(dt) & (dt > 0)
    if unit.lower().replace(" ", "") in ("us/ft", "usec/ft", "us_ft"):
        # 1e6 / dt gives ft/s, then convert to m/s
        out[good] = (US_PER_S / dt[good]) / FT_PER_M
    else:
        out[good] = US_PER_S / dt[good]
    return out


def density_to_si(rhob: np.ndarray, unit: str = "g/cm3") -> np.ndarray:
    """Convert a density log to kg/m^3."""
    rhob = np.asarray(rhob, dtype=float)
    if unit.lower().replace(" ", "") in ("g/cm3", "g/c3", "gcm3", "g/cc"):
        return rhob * 1000.0
    return rhob


def acoustic_impedance(vp: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """AI = Vp * Rho.  Units are whatever you feed it (m/s * kg/m^3)."""
    return np.asarray(vp, dtype=float) * np.asarray(rho, dtype=float)


# --------------------------------------------------------------------------
# Impedance <-> reflectivity
# --------------------------------------------------------------------------

def reflectivity_from_ai(ai: np.ndarray) -> np.ndarray:
    """Normal-incidence reflectivity from an acoustic impedance series.

    Returns an array the same length as ``ai``; the last sample is zero so the
    series stays aligned with the input sampling.
    """
    ai = np.asarray(ai, dtype=float)
    r = np.zeros_like(ai)
    denom = ai[1:] + ai[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r[:-1] = np.where(np.abs(denom) > 0, (ai[1:] - ai[:-1]) / denom, 0.0)
    return np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def ai_from_reflectivity(r: np.ndarray, ai0: float = 1.0) -> np.ndarray:
    """Exact recursive inversion of :func:`reflectivity_from_ai`.

    ``r`` is clipped just inside (-1, 1) so the recursion cannot blow up on a
    noisy spike.
    """
    r = np.clip(np.nan_to_num(np.asarray(r, dtype=float)), -0.95, 0.95)
    ratio = (1.0 + r) / (1.0 - r)
    ai = ai0 * np.concatenate([[1.0], np.cumprod(ratio[:-1])])
    return ai


def relative_impedance(r: np.ndarray) -> np.ndarray:
    """Band-limited relative impedance, ``2 * cumsum(r)`` (i.e. log-AI).

    This is the small-reflectivity approximation ln(AI_i/AI_0) ~= 2 * sum(r),
    which is what "relative impedance" means for coloured / sparse-spike
    outputs: it carries no absolute level, only the seismic band.
    """
    r = np.nan_to_num(np.asarray(r, dtype=float))
    return 2.0 * np.cumsum(r)


def merge_with_low_frequency(
    rel_log_ai: np.ndarray,
    low_freq_ai: np.ndarray,
    dt: float,
    merge_freq: float = 10.0,
) -> np.ndarray:
    """Splice a band-limited log-impedance trace onto a low-frequency model.

    The relative trace is high-pass filtered at ``merge_freq`` and added to the
    log of the background model, which supplies everything below the seismic
    band.  Returns absolute AI in the same units as ``low_freq_ai``.
    """
    rel = np.nan_to_num(np.asarray(rel_log_ai, dtype=float))
    lfm = np.asarray(low_freq_ai, dtype=float)
    lfm = np.where(np.isfinite(lfm) & (lfm > 0), lfm, np.nan)
    if np.all(np.isnan(lfm)):
        raise ValueError("low-frequency model trace is entirely invalid")
    lfm = fill_nan_1d(lfm)

    rel_hp = highpass(rel, dt, merge_freq)
    return np.exp(np.log(lfm) + rel_hp)


# --------------------------------------------------------------------------
# Filtering / spectra
# --------------------------------------------------------------------------

def _sos(dt: float, freq, btype: str, order: int = 4):
    nyq = 0.5 / dt
    wn = np.atleast_1d(np.asarray(freq, dtype=float)) / nyq
    wn = np.clip(wn, 1e-6, 0.999999)
    wn = wn[0] if wn.size == 1 else wn
    return signal.butter(order, wn, btype=btype, output="sos")


def _filtfilt(x: np.ndarray, sos) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    padlen = min(3 * (sos.shape[0] * 2), max(len(x) - 1, 0))
    if len(x) < 10:
        return x
    return signal.sosfiltfilt(sos, x, padlen=padlen)


def lowpass(x: np.ndarray, dt: float, cutoff: float, order: int = 4) -> np.ndarray:
    return _filtfilt(x, _sos(dt, cutoff, "lowpass", order))


def highpass(x: np.ndarray, dt: float, cutoff: float, order: int = 4) -> np.ndarray:
    return _filtfilt(x, _sos(dt, cutoff, "highpass", order))


def bandpass(x: np.ndarray, dt: float, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    return _filtfilt(x, _sos(dt, [f_lo, f_hi], "bandpass", order))


def amplitude_spectrum(x: np.ndarray, dt: float, pad: int | None = None):
    """Single-sided amplitude spectrum.  Returns ``(freq, amp)``."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    n = pad or int(2 ** np.ceil(np.log2(max(len(x), 2))) * 4)
    spec = np.fft.rfft(x, n=n)
    freq = np.fft.rfftfreq(n, d=dt)
    return freq, np.abs(spec)


def average_amplitude_spectrum(traces: np.ndarray, dt: float, pad: int | None = None):
    """Mean amplitude spectrum over a 2D stack of traces (n_traces, n_samples)."""
    traces = np.atleast_2d(np.nan_to_num(np.asarray(traces, dtype=float)))
    n = pad or int(2 ** np.ceil(np.log2(max(traces.shape[1], 2))) * 4)
    spec = np.abs(np.fft.rfft(traces, n=n, axis=-1))
    freq = np.fft.rfftfreq(n, d=dt)
    return freq, spec.mean(axis=0)


def phase_spectrum(x: np.ndarray, dt: float, pad: int | None = None):
    x = np.nan_to_num(np.asarray(x, dtype=float))
    n = pad or int(2 ** np.ceil(np.log2(max(len(x), 2))) * 4)
    spec = np.fft.rfft(x, n=n)
    freq = np.fft.rfftfreq(n, d=dt)
    return freq, np.degrees(np.angle(spec))


def rotate_phase(x: np.ndarray, degrees: float) -> np.ndarray:
    """Constant-phase rotation via the analytic signal."""
    x = np.nan_to_num(np.asarray(x, dtype=float))
    phi = np.radians(degrees)
    a = signal.hilbert(x)
    return np.real(a) * np.cos(phi) - np.imag(a) * np.sin(phi)


def convolve_same(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Convolve ``x`` with ``w``, always returning ``len(x)`` samples.

    ``np.convolve(..., mode="same")`` returns the length of the *longer* input,
    so it silently changes the trace length whenever the wavelet or operator is
    longer than the trace -- which happens with a short analysis window or a
    long coloured-inversion operator.  This centres the full convolution on the
    input instead, which matches "same" whenever "same" is correct.
    """
    x = np.nan_to_num(np.asarray(x, dtype=float))
    w = np.nan_to_num(np.asarray(w, dtype=float))
    if x.size == 0 or w.size == 0:
        return np.zeros(x.size)
    full = np.convolve(x, w, mode="full")
    start = (w.size - 1) // 2
    return full[start: start + x.size]


def taper_window(n: int, fraction: float = 0.15, kind: str = "tukey") -> np.ndarray:
    """Symmetric taper for wavelets / operators.  ``fraction`` is the total
    fraction of the window that is tapered (split between both ends)."""
    n = int(n)
    if n <= 2:
        return np.ones(max(n, 0))
    fraction = float(np.clip(fraction, 0.0, 1.0))
    if fraction <= 0:
        return np.ones(n)
    if kind == "hann":
        return signal.windows.hann(n, sym=True)
    if kind == "blackman":
        return signal.windows.blackman(n, sym=True)
    return signal.windows.tukey(n, alpha=fraction, sym=True)


# --------------------------------------------------------------------------
# Resampling / gap filling
# --------------------------------------------------------------------------

def fill_nan_1d(x: np.ndarray) -> np.ndarray:
    """Linear interpolation across interior NaNs, edge-hold at the ends."""
    x = np.asarray(x, dtype=float).copy()
    good = np.isfinite(x)
    if not good.any():
        return x
    idx = np.arange(x.size)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x


def resample_to_time(
    values: np.ndarray,
    src_time: np.ndarray,
    dst_time: np.ndarray,
    smooth: bool = True,
) -> np.ndarray:
    """Resample an irregularly-sampled log onto a regular time axis.

    When the source is finer than the destination (the usual case for a log
    sampled every 0.15 m against 2 ms seismic) a running mean is applied first
    so we decimate rather than alias.
    """
    values = np.asarray(values, dtype=float)
    src_time = np.asarray(src_time, dtype=float)
    dst_time = np.asarray(dst_time, dtype=float)

    good = np.isfinite(values) & np.isfinite(src_time)
    if good.sum() < 2:
        return np.full(dst_time.shape, np.nan)

    v, t = values[good], src_time[good]
    order = np.argsort(t)
    v, t = v[order], t[order]

    if smooth and len(t) > 4 and len(dst_time) > 1:
        src_dt = np.median(np.diff(t))
        dst_dt = np.median(np.diff(dst_time))
        if src_dt > 0 and dst_dt > src_dt:
            win = int(max(1, round(dst_dt / src_dt)))
            if win > 1:
                kernel = np.ones(win) / win
                # Kernel is always shorter than the log here, so "same" is safe.
                v = np.convolve(v, kernel, mode="same")

    out = np.interp(dst_time, t, v, left=np.nan, right=np.nan)
    return out


# --------------------------------------------------------------------------
# Geometry: well -> nearest live traces
# --------------------------------------------------------------------------

@dataclass
class TraceNeighbourhood:
    """Result of locating a well against the seismic grid."""

    indices: np.ndarray            # flat indices into the (il, xl) trace table
    distances: np.ndarray          # metres from the well head
    weights: np.ndarray            # inverse-distance weights, sum to 1
    inlines: np.ndarray
    crosslines: np.ndarray

    @property
    def nearest_index(self) -> int:
        return int(self.indices[0])


def idw_weights(distances: np.ndarray, power: float = 2.0, eps: float = 1e-6) -> np.ndarray:
    """Inverse-distance weights, normalised.  Exact hits win outright."""
    d = np.asarray(distances, dtype=float)
    exact = d < eps
    if exact.any():
        w = exact.astype(float)
        return w / w.sum()
    w = 1.0 / np.power(d, power)
    return w / w.sum()


def nearest_live_traces(
    trace_xy: np.ndarray,
    well_xy: Sequence[float],
    k: int = 4,
    live_mask: np.ndarray | None = None,
    power: float = 2.0,
) -> TraceNeighbourhood:
    """KD-tree lookup of the ``k`` nearest live traces to a well location.

    ``trace_xy`` is (n_traces, 2) in map coordinates; ``live_mask`` flags traces
    that actually carry data (dead/zero traces are excluded from the search so
    a well sitting on the survey edge still gets real amplitudes).
    """
    trace_xy = np.asarray(trace_xy, dtype=float)
    n_traces = trace_xy.shape[0]
    if live_mask is None:
        live_mask = np.ones(n_traces, dtype=bool)
    live_mask = np.asarray(live_mask, dtype=bool)
    if not live_mask.any():
        raise ValueError("no live traces available for well lookup")

    live_idx = np.flatnonzero(live_mask)
    tree = cKDTree(trace_xy[live_idx])
    k_eff = int(min(max(k, 1), live_idx.size))
    dist, loc = tree.query(np.asarray(well_xy, dtype=float), k=k_eff)
    dist = np.atleast_1d(dist)
    loc = np.atleast_1d(loc)
    flat = live_idx[loc]
    return TraceNeighbourhood(
        indices=flat,
        distances=dist,
        weights=idw_weights(dist, power=power),
        inlines=np.array([]),
        crosslines=np.array([]),
    )


def blend_traces(traces: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Inverse-distance weighted blend of a small set of traces."""
    traces = np.atleast_2d(np.asarray(traces, dtype=float))
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    return np.nansum(traces * w, axis=0) / np.nansum(w)


# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------

def normalised_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-lag Pearson correlation, NaN-safe.  Returns 0.0 for flat inputs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3:
        return 0.0
    a, b = a[good] - a[good].mean(), b[good] - b[good].mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def best_lag_correlation(a: np.ndarray, b: np.ndarray, max_lag: int = 50):
    """Cross-correlation peak within +/- ``max_lag`` samples.

    Returns ``(lag, correlation)`` where a positive lag means ``b`` must be
    shifted later to line up with ``a``.
    """
    a = np.nan_to_num(np.asarray(a, dtype=float))
    b = np.nan_to_num(np.asarray(b, dtype=float))
    if a.size < 3 or b.size < 3:
        return 0, 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 0:
        return 0, 0.0
    xc = signal.correlate(a, b, mode="full") / denom
    lags = signal.correlation_lags(a.size, b.size, mode="full")
    keep = np.abs(lags) <= max_lag
    xc, lags = xc[keep], lags[keep]
    i = int(np.argmax(np.abs(xc)))
    return int(lags[i]), float(xc[i])


def chunk_indices(total: int, chunk: int) -> Iterable[tuple[int, int]]:
    """Yield ``(start, stop)`` pairs covering ``range(total)``."""
    chunk = max(int(chunk), 1)
    for start in range(0, int(total), chunk):
        yield start, min(start + chunk, int(total))


def safe_percentile_clip(data: np.ndarray, pct: float = 99.0) -> float:
    """Symmetric colour limit for a seismic display."""
    d = np.asarray(data, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 1.0
    v = float(np.percentile(np.abs(d), pct))
    return v if v > 0 else 1.0
