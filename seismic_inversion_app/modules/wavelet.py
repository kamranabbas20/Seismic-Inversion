"""Wavelet generation and estimation.

Two families are supported:

* **Parametric / statistical** -- Ricker, Ormsby and Butterworth wavelets, plus
  a constant-phase wavelet whose amplitude spectrum is taken straight from the
  seismic.  These need no wells.
* **Well-based** -- least-squares (Wiener) extraction of the operator that maps
  well reflectivity onto the extracted seismic trace over a chosen time gate.
  This is the only route that recovers the true phase of the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import signal

from . import utils

WAVELET_TYPES = ("ricker", "ormsby", "butterworth")
PHASE_PRESETS = {"zero (0 deg)": 0.0, "minimum-ish (-90 deg)": -90.0, "+90 deg": 90.0, "180 deg": 180.0}


@dataclass
class Wavelet:
    """A wavelet plus the provenance needed to reproduce it."""

    samples: np.ndarray
    dt: float                       # seconds
    kind: str = "ricker"
    phase: float = 0.0
    params: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.samples = np.nan_to_num(np.asarray(self.samples, dtype=float))

    @property
    def length_ms(self) -> float:
        return float(self.samples.size * self.dt * 1000.0)

    @property
    def time_axis(self) -> np.ndarray:
        """Centred time axis in milliseconds."""
        n = self.samples.size
        return (np.arange(n) - n // 2) * self.dt * 1000.0

    def spectrum(self, pad: int = 4096):
        freq, amp = utils.amplitude_spectrum(self.samples, self.dt, pad=pad)
        return freq, amp

    def phase_spectrum(self, pad: int = 4096):
        """Phase spectrum, unwrapped and referenced to the wavelet centre.

        The raw FFT phase of a centred wavelet carries a large linear ramp from
        the time shift; removing it leaves the phase a geophysicist expects.
        """
        n = self.samples.size
        shifted = np.roll(self.samples, -(n // 2))
        spec = np.fft.rfft(shifted, n=pad)
        freq = np.fft.rfftfreq(pad, d=self.dt)
        amp = np.abs(spec)
        ph = np.degrees(np.angle(spec))
        ph[amp < amp.max() * 0.05] = np.nan   # phase is meaningless in the noise
        return freq, ph

    def dominant_frequency(self) -> float:
        freq, amp = self.spectrum()
        if amp.max() <= 0:
            return 0.0
        return float(freq[int(np.argmax(amp))])

    def bandwidth(self, level_db: float = -6.0) -> tuple[float, float]:
        freq, amp = self.spectrum()
        if amp.max() <= 0:
            return (0.0, 0.0)
        db = 20 * np.log10(np.maximum(amp / amp.max(), 1e-12))
        above = np.flatnonzero(db >= level_db)
        if above.size == 0:
            return (0.0, 0.0)
        return float(freq[above[0]]), float(freq[above[-1]])

    def normalised(self) -> "Wavelet":
        """Unit-maximum-amplitude copy (keeps convolution scaling sane)."""
        peak = np.max(np.abs(self.samples))
        if peak <= 0:
            return self
        return Wavelet(self.samples / peak, self.dt, self.kind, self.phase, dict(self.params), dict(self.quality))

    def summary(self) -> dict:
        lo, hi = self.bandwidth()
        return {
            "type": self.kind,
            "length (ms)": f"{self.length_ms:.0f}",
            "samples": int(self.samples.size),
            "phase (deg)": f"{self.phase:.0f}",
            "dominant freq (Hz)": f"{self.dominant_frequency():.1f}",
            "-6 dB band (Hz)": f"{lo:.1f} - {hi:.1f}",
            **{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in self.quality.items()},
        }


# --------------------------------------------------------------------------
# Parametric wavelets
# --------------------------------------------------------------------------

def _centred_axis(length_ms: float, dt: float) -> np.ndarray:
    """Odd-length, symmetric time axis in seconds."""
    n = int(round(length_ms / 1000.0 / dt))
    n = max(n + 1 if n % 2 == 0 else n, 5)
    return (np.arange(n) - n // 2) * dt


def ricker(freq: float, length_ms: float = 128.0, dt: float = 0.002) -> np.ndarray:
    """Classic zero-phase Ricker (second derivative of a Gaussian)."""
    t = _centred_axis(length_ms, dt)
    a = (np.pi * float(freq) * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return w / np.max(np.abs(w))


def ormsby(f1: float, f2: float, f3: float, f4: float, length_ms: float = 128.0, dt: float = 0.002) -> np.ndarray:
    """Ormsby bandpass wavelet defined by its four corner frequencies."""
    f1, f2, f3, f4 = sorted(float(f) for f in (f1, f2, f3, f4))
    t = _centred_axis(length_ms, dt)

    # Each term is (pi f)^2 sinc^2(f t); np.sinc already carries the pi.
    a = (np.pi * f4) ** 2 / (np.pi * f4 - np.pi * f3) * np.sinc(f4 * t) ** 2
    b = (np.pi * f3) ** 2 / (np.pi * f4 - np.pi * f3) * np.sinc(f3 * t) ** 2
    c = (np.pi * f2) ** 2 / (np.pi * f2 - np.pi * f1) * np.sinc(f2 * t) ** 2
    d = (np.pi * f1) ** 2 / (np.pi * f2 - np.pi * f1) * np.sinc(f1 * t) ** 2
    w = (a - b) - (c - d)
    w = w * utils.taper_window(w.size, 0.25)
    return w / np.max(np.abs(w))


def butterworth(
    f_low: float,
    f_high: float,
    length_ms: float = 128.0,
    dt: float = 0.002,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass wavelet.

    Built by band-passing a unit spike, which gives exactly the wavelet the
    equivalent filter would imprint on the data.
    """
    t = _centred_axis(length_ms, dt)
    spike = np.zeros(t.size)
    spike[t.size // 2] = 1.0
    nyq = 0.5 / dt
    lo = float(np.clip(f_low / nyq, 1e-4, 0.98))
    hi = float(np.clip(f_high / nyq, lo + 1e-3, 0.99))
    sos = signal.butter(order, [lo, hi], btype="bandpass", output="sos")
    w = signal.sosfiltfilt(sos, spike)
    w = w * utils.taper_window(w.size, 0.20)
    peak = np.max(np.abs(w))
    return w / peak if peak > 0 else w


def make_parametric_wavelet(
    kind: str,
    dt: float,
    length_ms: float = 128.0,
    phase: float = 0.0,
    taper: float = 0.15,
    **params,
) -> Wavelet:
    """Dispatch to one of the parametric families and apply phase + taper."""
    kind = kind.lower()
    if kind == "ricker":
        w = ricker(params.get("freq", 25.0), length_ms, dt)
    elif kind == "ormsby":
        w = ormsby(
            params.get("f1", 5.0), params.get("f2", 10.0),
            params.get("f3", 45.0), params.get("f4", 60.0),
            length_ms, dt,
        )
    elif kind == "butterworth":
        w = butterworth(
            params.get("f_low", 8.0), params.get("f_high", 55.0),
            length_ms, dt, order=int(params.get("order", 4)),
        )
    else:
        raise ValueError(f"unknown wavelet type '{kind}' (expected one of {WAVELET_TYPES})")

    if phase:
        w = utils.rotate_phase(w, phase)
    w = w * utils.taper_window(w.size, taper)
    peak = np.max(np.abs(w))
    if peak > 0:
        w = w / peak
    return Wavelet(w, dt, kind=kind, phase=phase, params={"length_ms": length_ms, "taper": taper, **params})


# --------------------------------------------------------------------------
# Statistical extraction
# --------------------------------------------------------------------------

def statistical_wavelet(
    traces: np.ndarray,
    dt: float,
    length_ms: float = 128.0,
    phase: float = 0.0,
    taper: float = 0.15,
    smooth_hz: float = 4.0,
) -> Wavelet:
    """Constant-phase wavelet whose amplitude spectrum is the data's own.

    The autocorrelation route (average amplitude spectrum -> inverse FFT ->
    symmetric operator) assumes white reflectivity, which is the standard
    statistical-extraction assumption.  Phase is *asserted*, not measured --
    that is what "constant phase" means here.
    """
    traces = np.atleast_2d(np.nan_to_num(np.asarray(traces, dtype=float)))
    n = max(int(2 ** np.ceil(np.log2(max(traces.shape[1], 2)))) * 2, 512)

    amp = np.abs(np.fft.rfft(traces, n=n, axis=-1)).mean(axis=0)
    freq = np.fft.rfftfreq(n, d=dt)

    # Smooth the spectrum so the operator is a wavelet, not a matched filter.
    if smooth_hz > 0 and freq.size > 4:
        df = float(freq[1] - freq[0])
        win = max(int(round(smooth_hz / df)), 1)
        if win > 1:
            amp = utils.convolve_same(amp, np.ones(win) / win)

    zero_phase = np.fft.irfft(amp, n=n)
    zero_phase = np.fft.fftshift(zero_phase)

    n_w = int(round(length_ms / 1000.0 / dt))
    n_w = max(n_w + 1 if n_w % 2 == 0 else n_w, 5)
    centre = n // 2
    half = n_w // 2
    w = zero_phase[centre - half: centre + half + 1]

    if phase:
        w = utils.rotate_phase(w, phase)
    w = w * utils.taper_window(w.size, taper)
    peak = np.max(np.abs(w))
    if peak > 0:
        w = w / peak

    return Wavelet(
        w, dt, kind="statistical", phase=phase,
        params={"length_ms": length_ms, "taper": taper, "smooth_hz": smooth_hz,
                "n_traces": int(traces.shape[0])},
    )


def statistical_wavelet_from_volume(
    volume,
    dt: float,
    length_ms: float = 128.0,
    phase: float = 0.0,
    taper: float = 0.15,
    max_traces: int = 500,
    t_min: float | None = None,
    t_max: float | None = None,
    seed: int = 0,
) -> Wavelet:
    """Statistical extraction over a random sample of live traces in a cube."""
    flat = volume.flat_data()
    live = np.flatnonzero(volume.live_mask())
    if live.size == 0:
        raise ValueError("volume contains no live traces")
    rng = np.random.default_rng(seed)
    pick = live if live.size <= max_traces else rng.choice(live, size=max_traces, replace=False)

    twt = volume.twt
    sl = slice(None)
    if t_min is not None and t_max is not None and t_max > t_min:
        i0 = int(np.searchsorted(twt, t_min))
        i1 = int(np.searchsorted(twt, t_max))
        if i1 - i0 > 16:
            sl = slice(i0, i1)

    gate = flat[pick, sl]
    gate = gate * utils.taper_window(gate.shape[1], 0.10)[None, :]
    return statistical_wavelet(gate, dt, length_ms=length_ms, phase=phase, taper=taper)


# --------------------------------------------------------------------------
# Well-based (least-squares / Wiener) extraction
# --------------------------------------------------------------------------

def wiener_wavelet(
    seismic: np.ndarray,
    reflectivity: np.ndarray,
    dt: float,
    length_ms: float = 128.0,
    prewhitening: float = 1.0,
    taper: float = 0.15,
) -> Wavelet:
    """Least-squares wavelet: the operator w minimising ||r * w - s||^2.

    Solved through the normal equations with a Toeplitz autocorrelation matrix
    of the reflectivity (the standard Wiener formulation).  ``prewhitening`` is
    a percentage added to the zero-lag term, which is what keeps the inverse
    stable where the reflectivity spectrum has notches.
    """
    s = np.nan_to_num(np.asarray(seismic, dtype=float))
    r = np.nan_to_num(np.asarray(reflectivity, dtype=float))
    if s.size != r.size:
        n = min(s.size, r.size)
        s, r = s[:n], r[:n]
    if s.size < 16 or np.allclose(r, 0):
        raise ValueError("not enough valid samples in the gate for well-based extraction")

    n_w = int(round(length_ms / 1000.0 / dt))
    n_w = max(n_w + 1 if n_w % 2 == 0 else n_w, 5)
    n_w = min(n_w, s.size - 1 if s.size % 2 == 0 else s.size - 2)
    if n_w < 5:
        raise ValueError("time gate is too short for the requested wavelet length")

    # Autocorrelation of r, cross-correlation of r with s.
    auto = signal.correlate(r, r, mode="full")
    mid = auto.size // 2
    auto = auto[mid: mid + n_w]
    auto[0] *= 1.0 + prewhitening / 100.0

    cross = signal.correlate(s, r, mode="full")
    cmid = cross.size // 2
    half = n_w // 2
    rhs = cross[cmid - half: cmid + half + 1]

    from scipy.linalg import solve_toeplitz

    try:
        w = solve_toeplitz((auto, auto), rhs)
    except Exception:  # noqa: BLE001 - singular system, fall back to lstsq
        from scipy.linalg import toeplitz

        w = np.linalg.lstsq(toeplitz(auto), rhs, rcond=None)[0]

    w = w * utils.taper_window(w.size, taper)
    peak = np.max(np.abs(w))
    if peak > 0:
        w = w / peak

    wav = Wavelet(w, dt, kind="well-based", params={"length_ms": length_ms, "prewhitening": prewhitening,
                                                   "taper": taper, "gate_samples": int(s.size)})
    synth = utils.convolve_same(r, wav.samples)
    wav.quality["correlation"] = utils.normalised_correlation(synth, s)
    wav.phase = float(estimate_constant_phase(wav.samples))
    return wav


def multi_well_wavelet(
    ties: Sequence,
    dt: float,
    twt: np.ndarray,
    length_ms: float = 128.0,
    t_min: float | None = None,
    t_max: float | None = None,
    prewhitening: float = 1.0,
    taper: float = 0.15,
    weights: Sequence[float] | None = None,
) -> Wavelet:
    """Average the per-well least-squares wavelets over every usable well.

    Stacking wavelets (rather than stacking the correlations) keeps one bad
    well from quietly dominating, and lets the UI report which wells took part.
    """
    twt = np.asarray(twt, dtype=float)
    i0 = int(np.searchsorted(twt, t_min)) if t_min is not None else 0
    i1 = int(np.searchsorted(twt, t_max)) if t_max is not None else twt.size
    if i1 - i0 < 32:
        raise ValueError(f"time gate {t_min}-{t_max} ms contains too few samples ({i1 - i0})")

    per_well: list[np.ndarray] = []
    used: list[str] = []
    corrs: list[float] = []
    for tie in ties:
        s = np.asarray(tie.seismic, dtype=float)[i0:i1]
        r = np.asarray(tie.reflectivity, dtype=float)[i0:i1]
        good = np.isfinite(s) & np.isfinite(r)
        if good.sum() < 32 or np.allclose(np.nan_to_num(r), 0):
            continue
        s = np.where(good, s, 0.0)
        r = np.where(good, r, 0.0)
        try:
            w = wiener_wavelet(s, r, dt, length_ms=length_ms, prewhitening=prewhitening, taper=taper)
        except Exception:  # noqa: BLE001 - skip wells the solver can't use
            continue
        per_well.append(w.samples)
        used.append(tie.well)
        corrs.append(float(w.quality.get("correlation", 0.0)))

    if not per_well:
        raise ValueError(
            "no well produced a usable wavelet -- check that the wells are located, "
            "tied, and that the time gate overlaps the logged interval"
        )

    stack = np.stack(per_well)
    if weights is not None and len(weights) == stack.shape[0]:
        wts = np.asarray(weights, dtype=float).reshape(-1, 1)
        avg = np.sum(stack * wts, axis=0) / np.sum(wts)
    else:
        avg = stack.mean(axis=0)

    peak = np.max(np.abs(avg))
    if peak > 0:
        avg = avg / peak

    wav = Wavelet(
        avg, dt, kind="well-based",
        params={"length_ms": length_ms, "prewhitening": prewhitening, "taper": taper,
                "gate_ms": (t_min, t_max), "wells": used},
    )
    wav.phase = float(estimate_constant_phase(avg))
    wav.quality["wells used"] = len(used)
    wav.quality["mean correlation"] = float(np.mean(corrs)) if corrs else 0.0
    return wav


def estimate_constant_phase(w: np.ndarray) -> float:
    """Best-fit constant phase of a wavelet, from its analytic signal.

    Found by rotating over a 1-degree grid and taking the rotation that
    maximises the peak amplitude -- the rotation at which the wavelet is most
    "zero-phase-like" is the negative of its own constant phase.
    """
    w = np.nan_to_num(np.asarray(w, dtype=float))
    if w.size < 5 or np.allclose(w, 0):
        return 0.0
    angles = np.arange(-180, 180, 1.0)
    a = signal.hilbert(w)
    peaks = np.array([np.max(np.real(a) * np.cos(np.radians(p)) - np.imag(a) * np.sin(np.radians(p)))
                      for p in angles])
    best = angles[int(np.argmax(peaks))]
    return float(-best)


# --------------------------------------------------------------------------
# Amplitude calibration
# --------------------------------------------------------------------------

def calibrate_amplitude(
    wav: "Wavelet",
    ties: Sequence,
    twt: np.ndarray,
    t_min: float | None = None,
    t_max: float | None = None,
) -> "Wavelet":
    """Scale a wavelet so ``reflectivity * wavelet`` matches seismic amplitude.

    This step is not cosmetic.  Sparse-spike and model-based inversion both
    solve ``W r = s`` for ``r``; if ``W`` is off by a factor ``a`` then every
    recovered reflection coefficient is off by ``1/a``, and the impedance
    contrast that comes out of the integration is wrong by the same factor.
    A peak-normalised wavelet -- which is what you want for *display* -- is
    exactly such a mis-scaled operator.

    The optimal scalar is the least-squares one,
    ``a = <synthetic, seismic> / <synthetic, synthetic>``, accumulated over
    every usable well so one noisy well cannot set the scale on its own.
    """
    twt = np.asarray(twt, dtype=float)
    i0 = int(np.searchsorted(twt, t_min)) if t_min is not None else 0
    i1 = int(np.searchsorted(twt, t_max)) if t_max is not None else twt.size

    num = 0.0
    den = 0.0
    n_used = 0
    for tie in ties:
        s = np.nan_to_num(np.asarray(tie.seismic, dtype=float)[i0:i1])
        r = np.nan_to_num(np.asarray(tie.reflectivity, dtype=float)[i0:i1])
        if s.size < 16 or np.allclose(r, 0):
            continue
        synth = utils.convolve_same(r, wav.samples)
        num += float(np.dot(synth, s))
        den += float(np.dot(synth, synth))
        n_used += 1

    if n_used == 0 or den <= 0 or not np.isfinite(num / den):
        out = Wavelet(wav.samples.copy(), wav.dt, wav.kind, wav.phase, dict(wav.params), dict(wav.quality))
        out.quality["amplitude scalar"] = 1.0
        out.notes = ["amplitude NOT calibrated (no usable well) -- absolute impedance will be unreliable"]
        return out

    a = num / den
    out = Wavelet(wav.samples * a, wav.dt, wav.kind, wav.phase, dict(wav.params), dict(wav.quality))
    out.quality["amplitude scalar"] = float(a)
    out.quality["calibration wells"] = n_used

    corrs = []
    for tie in ties:
        s = np.nan_to_num(np.asarray(tie.seismic, dtype=float)[i0:i1])
        r = np.nan_to_num(np.asarray(tie.reflectivity, dtype=float)[i0:i1])
        if s.size < 16 or np.allclose(r, 0):
            continue
        corrs.append(utils.normalised_correlation(utils.convolve_same(r, out.samples), s))
    if corrs:
        out.quality["tie correlation"] = float(np.mean(corrs))
    return out


def wavelet_from_config(cfg: dict, dt: float, volume=None, ties=None) -> Wavelet:
    """Single entry point used by the app: build whatever the UI asked for."""
    method = cfg.get("method", "parametric")
    length_ms = float(cfg.get("length_ms", 128.0))
    taper = float(cfg.get("taper", 0.15))
    phase = float(cfg.get("phase", 0.0))

    if method == "parametric":
        return make_parametric_wavelet(cfg.get("kind", "ricker"), dt, length_ms, phase, taper, **cfg.get("params", {}))
    if method == "statistical":
        if volume is None:
            raise ValueError("statistical extraction needs the seismic volume")
        return statistical_wavelet_from_volume(
            volume, dt, length_ms=length_ms, phase=phase, taper=taper,
            max_traces=int(cfg.get("max_traces", 500)),
            t_min=cfg.get("t_min"), t_max=cfg.get("t_max"),
        )
    if method == "well-based":
        if not ties:
            raise ValueError("well-based extraction needs at least one located, tied well")
        return multi_well_wavelet(
            ties, dt, cfg["twt"], length_ms=length_ms,
            t_min=cfg.get("t_min"), t_max=cfg.get("t_max"),
            prewhitening=float(cfg.get("prewhitening", 1.0)), taper=taper,
        )
    raise ValueError(f"unknown wavelet method '{method}'")


# ==========================================================================
# Non-stationary wavelets and Q
# ==========================================================================

@dataclass
class NonStationaryWavelet:
    """A wavelet that is allowed to change with time.

    One wavelet for a three-second volume is an assumption, not a measurement.
    The earth is anelastic: high frequencies are absorbed faster than low ones,
    so a wavelet extracted at 2,500 ms is narrower-band and more
    phase-rotated than the same wavelet at 800 ms.  Inverting the deep section
    with the shallow wavelet over-resolves it, and the residual absorbs the
    difference.

    Wavelets are held at window centres and blended linearly between them, so
    the operator varies smoothly rather than jumping at window edges.
    """

    centres_ms: np.ndarray
    wavelets: np.ndarray                 # (n_windows, n_samples)
    dt: float
    window_ms: float
    quality: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.centres_ms = np.asarray(self.centres_ms, dtype=float)
        self.wavelets = np.atleast_2d(np.asarray(self.wavelets, dtype=float))

    def at(self, t_ms: float) -> np.ndarray:
        """The wavelet at one time, linearly blended between window centres."""
        c = self.centres_ms
        if c.size == 1:
            return self.wavelets[0].copy()
        t = float(np.clip(t_ms, c[0], c[-1]))
        j = int(np.clip(np.searchsorted(c, t) - 1, 0, c.size - 2))
        span = c[j + 1] - c[j]
        f = 0.0 if span <= 0 else (t - c[j]) / span
        return (1.0 - f) * self.wavelets[j] + f * self.wavelets[j + 1]

    def dominant_frequencies(self) -> np.ndarray:
        """Peak frequency of each window's wavelet, for a drift plot."""
        out = []
        for w in self.wavelets:
            freq, amp = utils.amplitude_spectrum(w, self.dt, pad=4096)
            out.append(float(freq[int(np.argmax(amp))]))
        return np.asarray(out)

    def summary(self) -> dict:
        f = self.dominant_frequencies()
        return {
            "windows": int(self.centres_ms.size),
            "window length (ms)": f"{self.window_ms:.0f}",
            "centres (ms)": ", ".join(f"{c:.0f}" for c in self.centres_ms),
            "dominant frequency (Hz)": ", ".join(f"{v:.0f}" for v in f),
            "frequency drift (Hz)": f"{f[0] - f[-1]:+.1f}" if f.size > 1 else "n/a",
        }


def time_varying_wavelet(
    ties: Sequence,
    dt: float,
    twt: np.ndarray,
    n_windows: int = 3,
    window_ms: float = 800.0,
    length_ms: float = 128.0,
    prewhitening: float = 1.0,
    taper: float = 0.15,
    t_min: float | None = None,
    t_max: float | None = None,
) -> NonStationaryWavelet:
    """Extract one wavelet per time window, from the wells, and stack them.

    Windows overlap so that a reflector near a boundary contributes to both
    neighbours; without the overlap the wavelet jumps where the windows meet.
    A window that yields no usable wavelet falls back to its nearest neighbour
    rather than dropping out, so the operator stays defined everywhere.
    """
    twt = np.asarray(twt, dtype=float)
    lo = float(t_min) if t_min is not None else float(twt[0])
    hi = float(t_max) if t_max is not None else float(twt[-1])
    n_windows = int(max(1, n_windows))
    if n_windows == 1:
        w = multi_well_wavelet(ties, dt, twt, length_ms=length_ms, t_min=lo, t_max=hi,
                               prewhitening=prewhitening, taper=taper)
        return NonStationaryWavelet(np.array([0.5 * (lo + hi)]), w.samples[None, :], dt,
                                    hi - lo, quality=dict(w.quality),
                                    notes=["single window: equivalent to a stationary wavelet"])

    centres = np.linspace(lo + window_ms / 2, hi - window_ms / 2, n_windows)
    centres = np.clip(centres, lo, hi)
    samples, notes, quality = [], [], {}
    for c in centres:
        a, b = c - window_ms / 2, c + window_ms / 2
        try:
            w = multi_well_wavelet(ties, dt, twt, length_ms=length_ms,
                                   t_min=max(a, lo), t_max=min(b, hi),
                                   prewhitening=prewhitening, taper=taper)
            samples.append(w.samples)
            quality[f"tie corr @ {c:.0f} ms"] = w.quality.get("tie correlation", float("nan"))
        except Exception as exc:  # noqa: BLE001 - keep the operator defined
            samples.append(None)
            notes.append(f"window at {c:.0f} ms failed ({type(exc).__name__}); filled from a neighbour")

    if all(s is None for s in samples):
        raise ValueError("no time window produced a usable wavelet")
    # Fill gaps forwards then backwards from the nearest successful window.
    for i in range(len(samples)):
        if samples[i] is None:
            nearest = min((j for j in range(len(samples)) if samples[j] is not None),
                          key=lambda j: abs(j - i))
            samples[i] = samples[nearest].copy()

    return NonStationaryWavelet(centres, np.vstack(samples), dt, window_ms,
                                quality=quality, notes=notes)


def estimate_q(
    volume,
    t_shallow: float,
    t_deep: float,
    window_ms: float = 400.0,
    f_lo: float | None = None,
    f_hi: float | None = None,
    max_traces: int = 400,
    seed: int = 0,
    band_fraction: float = 0.6,
) -> dict:
    """Estimate Q by the classical spectral-ratio method.

    Between two windows the amplitude spectra of an anelastic earth differ by
    ``exp(-pi f dt / Q)``, so ``ln(A_deep / A_shallow)`` is linear in ``f`` with
    slope ``-pi dt / Q``.  Fitting that slope over a band where both windows
    still have signal gives Q directly.

    This is a *bulk* Q between the two windows, not an interval Q profile, and
    it is only as good as the assumption that the reflectivity spectra of the
    two windows are the same.  Where they are not -- a tuned sand package
    against a shale section -- the number is measuring geology, not absorption,
    which is why the fit quality is reported alongside it.

    **The band matters more than anything else here.**  Outside the seismic's
    own bandwidth both spectra are near zero, their ratio is numerical noise
    near one, and including that flattens the fitted slope and inflates Q.
    Measured against synthetic data with a known Q of 40, a band of 8-70 Hz on
    35 Hz data returned 127; narrowing to 18-40 Hz returned 52.  So by default
    the band is taken from the data itself -- the range where the shallow
    window still holds ``band_fraction`` of its peak amplitude -- and ``f_lo``
    / ``f_hi`` only override that when you have a reason to.  ``r_squared`` is
    the honest guide either way: the badly-biased run above scored 0.56, the
    good one 0.99.

    Even with a sensible band this is a rough number, and it is biased *high*
    under strong attenuation, because the spectral ratio flattens once the deep
    window's high frequencies fall into the noise.  Measured against known
    values with the default band selection: Q 150 came back as 158, Q 80 as 86,
    and Q 40 as 62.  Treat it as an order of magnitude for setting an inverse-Q
    gain, not as a rock property.
    """
    dt = volume.dt
    twt = np.asarray(volume.twt, dtype=float)
    flat = volume.flat_data()
    live = volume.live_mask()
    idx = np.flatnonzero(live)
    if idx.size == 0:
        raise ValueError("no live traces to estimate Q from")
    if idx.size > max_traces:
        idx = np.random.default_rng(seed).choice(idx, size=max_traces, replace=False)

    def gate(centre):
        return (twt >= centre - window_ms / 2) & (twt <= centre + window_ms / 2)

    g1, g2 = gate(t_shallow), gate(t_deep)
    if g1.sum() < 16 or g2.sum() < 16:
        raise ValueError("the Q windows are too short for a spectral estimate")

    n = int(max(g1.sum(), g2.sum()))
    pad = int(2 ** np.ceil(np.log2(max(n, 32))) * 4)
    freq, a1 = utils.average_amplitude_spectrum(flat[idx][:, g1], dt, pad=pad)
    _f, a2 = utils.average_amplitude_spectrum(flat[idx][:, g2], dt, pad=pad)

    auto_band = ""
    if f_lo is None or f_hi is None:
        peak = float(np.max(a1)) if a1.size else 0.0
        strong = np.flatnonzero(a1 >= max(peak * float(band_fraction), 0.0))
        if strong.size >= 4 and peak > 0:
            lo_auto, hi_auto = float(freq[strong[0]]), float(freq[strong[-1]])
        else:
            lo_auto, hi_auto = 10.0, 60.0
        f_lo = lo_auto if f_lo is None else f_lo
        f_hi = hi_auto if f_hi is None else f_hi
        auto_band = (f"band {f_lo:.0f}-{f_hi:.0f} Hz taken from the data "
                     f"(where the shallow window holds >{band_fraction:.0%} of peak amplitude)")

    band = (freq >= f_lo) & (freq <= f_hi) & (a1 > 0) & (a2 > 0)
    if band.sum() < 4:
        raise ValueError(f"band {f_lo}-{f_hi} Hz has too little signal in both windows")

    ratio = np.log(a2[band] / a1[band])
    slope, intercept = np.polyfit(freq[band], ratio, 1)
    delta_t = (t_deep - t_shallow) / 1000.0
    q = float("inf") if slope >= 0 else float(-np.pi * delta_t / slope)

    fitted = slope * freq[band] + intercept
    ss_res = float(np.sum((ratio - fitted) ** 2))
    ss_tot = float(np.sum((ratio - ratio.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    note = ""
    if slope >= 0:
        note = ("the deep window is richer in high frequencies than the shallow one, "
                "so no absorption can be fitted -- Q is reported as infinite")
    elif r2 < 0.5:
        note = (f"the spectral ratio is a poor straight line (R^2 {r2:.2f}); this Q is "
                "probably measuring a reflectivity contrast, not absorption")
    return {"Q": q, "slope": float(slope), "r_squared": float(r2),
            "band": (float(f_lo), float(f_hi)), "windows": (t_shallow, t_deep),
            "traces": int(idx.size), "auto_band": auto_band, "note": note}


def _q_operator(freq: np.ndarray, t_s: float, q: float, f_ref: float, inverse: bool,
                gain_limit: float) -> np.ndarray:
    """Constant-Q amplitude and dispersion operator for one travel time."""
    f = np.abs(np.asarray(freq, dtype=float))
    decay = np.pi * f * t_s / max(q, 1e-6)
    # Kolsky-Futterman dispersion: the phase that must accompany the decay for
    # the medium to stay causal.
    with np.errstate(divide="ignore", invalid="ignore"):
        phase = (2.0 / np.pi) * decay * np.log(np.where(f > 0, f / f_ref, 1.0))
    phase = np.nan_to_num(phase)
    if inverse:
        gain = np.minimum(np.exp(decay), gain_limit)
        return gain * np.exp(1j * phase)
    return np.exp(-decay) * np.exp(-1j * phase)


def apply_q_filter(
    trace: np.ndarray,
    dt: float,
    twt: np.ndarray,
    q: float,
    inverse: bool = True,
    f_ref: float = 60.0,
    gain_limit_db: float = 20.0,
    window_ms: float = 200.0,
) -> np.ndarray:
    """Apply (or remove) constant-Q absorption, window by window.

    Absorption is non-stationary -- the operator depends on travel time -- so it
    cannot be applied as one filter.  The trace is split into overlapping
    Hann-tapered windows, each filtered with the constant-Q operator for its own
    centre time, and the windows summed.  The tapers form a partition of unity,
    so a run with ``Q = inf`` returns the input unchanged.

    ``gain_limit_db`` caps the inverse gain.  Without it, inverse Q filtering
    amplifies the highest frequencies without bound, and since those are mostly
    noise the result is an unusable trace that nonetheless has a beautiful
    spectrum.
    """
    trace = np.nan_to_num(np.asarray(trace, dtype=float))
    twt = np.asarray(twt, dtype=float)
    n = trace.size
    if not np.isfinite(q) or q <= 0 or n < 8:
        return trace.copy()

    gain_limit = 10.0 ** (float(gain_limit_db) / 20.0)
    half = max(int(round(window_ms / (dt * 1000.0))), 4)   # hop; window = 2 * half
    win = 2 * half

    # Reflect-pad by one hop at each end.  Without it the first and last
    # samples are covered by the rising or falling flank of a single Hann
    # window, the overlap-add weight there goes to zero, and dividing by it
    # turns the trace ends into broadband spikes that swamp the filter.
    pad = half
    padded = np.concatenate((trace[pad:0:-1], trace, trace[-2:-pad - 2:-1]))
    if padded.size < n + 2 * pad:                 # short trace: pad with edge values
        padded = np.pad(trace, pad, mode="reflect" if n > pad else "edge")
    m = padded.size
    t_pad = np.concatenate((
        twt[0] - (np.arange(pad, 0, -1)) * dt * 1000.0,
        twt,
        twt[-1] + (np.arange(1, m - n - pad + 1)) * dt * 1000.0,
    ))

    out = np.zeros(m)
    weight = np.zeros(m)
    taper = np.hanning(win + 2)[1:-1]             # zero-free, sums to 1 at 50% overlap
    freq = np.fft.rfftfreq(win, d=dt)
    for s in range(0, m - win + 1, half):
        seg = padded[s:s + win]
        centre_ms = float(t_pad[s + win // 2])
        op = _q_operator(freq, max(centre_ms, 0.0) / 1000.0, q, f_ref, inverse, gain_limit)
        out[s:s + win] += np.fft.irfft(np.fft.rfft(seg * taper) * op, n=win)
        weight[s:s + win] += taper

    filtered = np.where(weight > 1e-6, out / np.maximum(weight, 1e-6), padded)
    return filtered[pad:pad + n]
