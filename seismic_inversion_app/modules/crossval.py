"""Blind-well cross-validation.

Scoring an inversion at a well that helped build its own background model
measures the background model, not the inversion.  With one well that is
unavoidable and has to be stated; with several it is simply a mistake, because
the honest experiment is available: drop a well, rebuild the low-frequency
model without it, invert, and score against the log the model never saw.

That is what this module does.  It is the difference between "the answer looks
like the well" -- which it must, since the well is in the answer -- and "the
answer predicts a well it has never seen", which is the only version worth
showing a partner.

Two numbers matter and they are reported side by side:

* the blind score for the inversion, and
* the blind score for the low-frequency model *alone*.

An inversion that cannot beat its own background at a blind well has added
nothing, however good its section looks.  The difference between the two is
reported as the uplift.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import data_io, inversion as inversion_mod, low_freq_model as lfmod, utils


@dataclass
class BlindWellScore:
    """How well a held-out well was predicted."""

    well: str
    n_samples: int
    gate: tuple[float, float]
    corr_band: float
    corr_full: float
    rmse_log: float
    corr_band_lfm: float
    rmse_log_lfm: float
    note: str = ""

    @property
    def uplift(self) -> float:
        """Band-limited correlation the inversion added over the background."""
        return float(self.corr_band - self.corr_band_lfm)

    @property
    def beat_background(self) -> bool:
        return bool(np.isfinite(self.uplift) and self.uplift > 0)

    def row(self) -> dict:
        return {
            "well": self.well,
            "samples": self.n_samples,
            "gate (ms)": f"{self.gate[0]:.0f}-{self.gate[1]:.0f}"
            if np.isfinite(self.gate[0]) else "n/a",
            "corr (band)": f"{self.corr_band:+.3f}",
            "background (band)": f"{self.corr_band_lfm:+.3f}",
            "uplift": f"{self.uplift:+.3f}",
            "corr (full)": f"{self.corr_full:+.3f}",
            "RMSE ln(AI)": f"{self.rmse_log:.4f}",
            "verdict": "beats background" if self.beat_background else "no uplift",
        }


@dataclass
class CrossValidation:
    """The whole leave-one-out experiment."""

    method: str
    scores: list[BlindWellScore] = field(default_factory=list)
    band: tuple[float, float] = (10.0, 60.0)
    cutoff_hz: float = 10.0
    notes: list[str] = field(default_factory=list)

    def table(self) -> list[dict]:
        return [s.row() for s in self.scores]

    def summary(self) -> dict:
        scored = [s for s in self.scores if np.isfinite(s.corr_band)]
        if not scored:
            return {"method": self.method, "wells scored": 0,
                    "verdict": "no well could be scored blind"}
        uplift = np.array([s.uplift for s in scored])
        beat = sum(s.beat_background for s in scored)
        return {
            "method": self.method,
            "wells scored": len(scored),
            "mean blind correlation": f"{np.mean([s.corr_band for s in scored]):+.3f}",
            "mean background correlation": f"{np.mean([s.corr_band_lfm for s in scored]):+.3f}",
            "mean uplift": f"{np.mean(uplift):+.3f}",
            "wells where it beat the background": f"{beat} of {len(scored)}",
            "verdict": ("the inversion predicts wells it has not seen"
                        if beat > len(scored) / 2 else
                        "the inversion does not reliably beat its own background"),
        }


def _score_trace(ai: np.ndarray, target: np.ndarray, good: np.ndarray,
                 dt: float, band: tuple[float, float]) -> tuple[float, float, float]:
    lo, hi = band
    b = lambda x: utils.bandpass(np.asarray(x, dtype=float), dt, lo, hi)  # noqa: E731
    corr_band = utils.normalised_correlation(b(ai)[good], b(target)[good])
    corr_full = utils.normalised_correlation(np.asarray(ai)[good], np.asarray(target)[good])
    rmse = float(np.sqrt(np.mean(
        (np.log(np.clip(np.asarray(ai)[good], 1e-9, None)) - np.log(target[good])) ** 2)))
    return float(corr_band), float(corr_full), rmse


def leave_one_out(
    volume,
    wells,
    wavelet: np.ndarray,
    method: str = "bayesian",
    cutoff_hz: float = 10.0,
    band: tuple[float, float] = (10.0, 60.0),
    lfm_kwargs: dict | None = None,
    progress=None,
    **invert_params,
) -> CrossValidation:
    """Hold each well out in turn, rebuild the background without it, and score.

    Only the held-out well's own trace is inverted -- the experiment needs one
    trace per fold, not a cube -- so the whole thing costs about as much as
    inverting a handful of traces, whatever the size of the survey.

    Needs at least two located wells: with one, dropping it leaves nothing to
    build a background model from.
    """
    located = [w for w in wells if getattr(w, "has_location", False)]
    if len(located) < 2:
        raise ValueError(
            "blind cross-validation needs at least two located wells -- with one well "
            "there is nothing left to build a background model from when it is held out")

    dt = volume.dt
    twt = np.asarray(volume.twt, dtype=float)
    lfm_kwargs = dict(lfm_kwargs or {})
    lfm_kwargs.pop("cutoff_hz", None)
    result = CrossValidation(method=method, band=band, cutoff_hz=cutoff_hz)

    for idx, held in enumerate(located):
        others = [w for w in located if w is not held]
        try:
            lf = lfmod.build_low_frequency_model(volume, others, cutoff_hz=cutoff_hz, **lfm_kwargs)
        except ValueError as exc:
            result.scores.append(_blank(held.name, f"background model failed: {exc}"))
            continue

        ties = data_io.extract_well_traces(volume, [held], k=4)
        if not ties:
            result.scores.append(_blank(held.name, "well has no location on this survey"))
            continue
        tie = ties[0]
        target = np.asarray(tie.ai, dtype=float)
        good = np.isfinite(target) & (target > 0)
        if good.sum() < 32:
            result.scores.append(_blank(held.name, "too few logged samples inside the survey"))
            continue

        lf_trace = np.asarray(lf.ai[tie.il_index, tie.xl_index, :], dtype=float)
        try:
            out = inversion_mod.invert(np.asarray(tie.seismic, dtype=float), wavelet,
                                       lf_trace, method=method, dt=dt, **invert_params)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            result.scores.append(_blank(held.name, f"inversion failed: {exc}"))
            continue
        ai = out.get("absolute_ai")
        if ai is None:
            result.scores.append(_blank(held.name, "engine produced no absolute impedance"))
            continue

        cb, cf, rmse = _score_trace(ai, target, good, dt, band)
        lb, _lf_full, l_rmse = _score_trace(lf_trace, target, good, dt, band)
        gate = (float(twt[good].min()), float(twt[good].max()))
        result.scores.append(BlindWellScore(
            well=held.name, n_samples=int(good.sum()), gate=gate,
            corr_band=cb, corr_full=cf, rmse_log=rmse,
            corr_band_lfm=lb, rmse_log_lfm=l_rmse,
            note=f"background built from {len(others)} other well(s)"))
        if progress is not None:
            progress((idx + 1) / len(located), f"held out {held.name}")

    result.notes.append(
        f"each well scored against a background model built from the other {len(located) - 1}")
    return result


def _blank(name: str, note: str) -> BlindWellScore:
    nan = float("nan")
    return BlindWellScore(well=name, n_samples=0, gate=(nan, nan), corr_band=nan,
                          corr_full=nan, rmse_log=nan, corr_band_lfm=nan,
                          rmse_log_lfm=nan, note=note)


def compare_methods(
    volume,
    wells,
    wavelet: np.ndarray,
    methods=("coloured", "sparse-spike", "model-based", "bayesian"),
    progress=None,
    **kwargs,
) -> dict[str, CrossValidation]:
    """Run the same blind experiment for several engines, so they are comparable.

    Method choice is usually argued from the look of a section.  This settles it
    on the only evidence that generalises: which engine best predicts logs it
    was not given.
    """
    out: dict[str, CrossValidation] = {}
    for i, method in enumerate(methods):
        params = dict(kwargs)
        if method == "coloured" and "operator" not in params:
            continue        # coloured needs a designed operator; skip rather than guess
        try:
            out[method] = leave_one_out(volume, wells, wavelet, method=method, **params)
        except ValueError:
            raise
        except Exception as exc:  # noqa: BLE001
            cv = CrossValidation(method=method)
            cv.notes.append(f"failed: {exc}")
            out[method] = cv
        if progress is not None:
            progress((i + 1) / max(len(methods), 1), f"cross-validated {method}")
    return out
