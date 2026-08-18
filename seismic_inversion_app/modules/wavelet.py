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
