"""Turning impedance into a rock property, with the uncertainty carried through.

Acoustic impedance is an intermediate quantity.  Nobody drills on impedance --
they drill on porosity, on net pay, on whether a body is above a cut-off.  This
module fits that last step at the wells and applies it to the cube.

The reason it lives next to the Bayesian engine is that the engine returns a
posterior, and a posterior is exactly what a cut-off needs.  Asking "is porosity
above 12% here?" of a single deterministic number gives a yes or a no, both of
them overconfident.  Asking it of a distribution gives a probability, and
summing that probability down a trace gives an expected net thickness that
already accounts for how well the seismic actually resolved the impedance.

Two sources of uncertainty are combined, because leaving either out flatters
the answer:

* the **inversion** uncertainty -- the posterior standard deviation of
  log-impedance, mapped through the transform by the delta method, and
* the **transform** uncertainty -- the scatter of the well points about the
  fitted curve, which is irreducible no matter how good the seismic is.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import utils

# 1.2815515655446004 = the 90th percentile of a standard normal.
Z90 = 1.2815515655446004

# Depth and time index curves.  They are stored alongside the measurements but
# they are not measurements: a "predicted depth" is reproduced perfectly by the
# time attribute, or by any monotone function of impedance, and says nothing
# about the rock.  Offering one as a target invites a fit that looks excellent
# and means nothing, so they are kept off the list.
INDEX_CURVES = frozenset({"DEPT", "DEPTH", "MD", "TVD", "TVDSS", "TVDKB",
                          "TWT", "TIME", "OWT"})


def predictable_curves(wells) -> list[str]:
    """Well curves worth predicting from seismic, index curves left out.

    The curve a well was assigned as its *time* channel is excluded too,
    whatever it happens to be called: it may carry any mnemonic at all, and it
    is still an index.
    """
    assigned = {str(getattr(getattr(w, "selection", None), "time", "") or "").upper()
                for w in wells}
    assigned.discard("")
    return sorted({m for w in wells for m in getattr(w, "curves", {})
                   if m.upper() not in INDEX_CURVES and m.upper() not in assigned})


@dataclass
class PropertyFit:
    """A fitted impedance-to-property transform and how well it holds."""

    property_name: str
    coefficients: np.ndarray          # polynomial in ln(AI), highest power first
    degree: int
    r_squared: float
    rmse: float
    residual_std: float
    n_points: int
    wells: list[str] = field(default_factory=list)
    unit: str = ""
    ai_range: tuple[float, float] = (0.0, 0.0)
    notes: list[str] = field(default_factory=list)

    def predict(self, ai: np.ndarray) -> np.ndarray:
        """Property from absolute impedance."""
        ai = np.asarray(ai, dtype=float)
        return np.polyval(self.coefficients, np.log(np.clip(ai, 1e-9, None)))

    def slope(self, ai: np.ndarray) -> np.ndarray:
        """d(property)/d(ln AI) -- the delta-method sensitivity."""
        ai = np.asarray(ai, dtype=float)
        deriv = np.polyder(self.coefficients)
        return np.polyval(deriv, np.log(np.clip(ai, 1e-9, None)))

    def summary(self) -> dict:
        return {
            "property": f"{self.property_name}" + (f" [{self.unit}]" if self.unit else ""),
            "fit": f"degree {self.degree} in ln(AI)",
            "R^2": f"{self.r_squared:.3f}",
            "RMSE": f"{self.rmse:.4g}",
            "scatter about the fit": f"{self.residual_std:.4g}",
            "points": self.n_points,
            "wells": ", ".join(self.wells) or "none",
            "AI range fitted": f"{self.ai_range[0]:,.0f} - {self.ai_range[1]:,.0f}",
        }


def curve_on_time_axis(well, mnemonic: str, twt_axis: np.ndarray) -> np.ndarray:
    """One of a well's curves, anti-alias resampled onto the seismic axis."""
    values = np.asarray(well.curves.get(mnemonic), dtype=float) if mnemonic in well.curves else None
    if values is None:
        return np.full(np.shape(twt_axis), np.nan)
    good = np.isfinite(values) & np.isfinite(well.twt)
    if good.sum() < 2:
        return np.full(np.shape(twt_axis), np.nan)
    return utils.resample_to_time(values[good], well.twt[good], np.asarray(twt_axis, dtype=float))


def fit_property(
    volume,
    wells,
    mnemonic: str,
    degree: int = 1,
    gate: tuple[float, float] | None = None,
    ties=None,
    clip_percentile: float = 99.5,
) -> PropertyFit:
    """Regress a well curve against log-impedance at the wells.

    Fitting in ``ln(AI)`` rather than ``AI`` is deliberate: impedance is a
    positive, roughly log-normal quantity, and most petrophysical relations
    (Wyllie, Raymer-Hunt-Gardner, and the linear porosity-impedance trends used
    in practice) are much closer to straight in the log domain.  It also makes
    the uncertainty propagation below exact rather than approximate, because the
    inversion's posterior is Gaussian in ``ln(AI)``.
    """
    from . import data_io

    twt = np.asarray(volume.twt, dtype=float)
    located = [w for w in wells if getattr(w, "has_location", False)]
    if not located:
        raise ValueError("fitting a property transform needs at least one located well")
    if ties is None:
        ties = data_io.extract_well_traces(volume, located, k=4)
    by_name = {t.well: t for t in ties}

    xs, ys, used = [], [], []
    for well in located:
        tie = by_name.get(well.name)
        if tie is None:
            continue
        prop = curve_on_time_axis(well, mnemonic, twt)
        ai = np.asarray(tie.ai, dtype=float)
        good = np.isfinite(prop) & np.isfinite(ai) & (ai > 0)
        if gate is not None:
            good &= (twt >= gate[0]) & (twt <= gate[1])
        if good.sum() < 8:
            continue
        xs.append(np.log(ai[good]))
        ys.append(prop[good])
        used.append(well.name)

    if not xs:
        raise ValueError(
            f"no well provided usable '{mnemonic}' samples against impedance -- "
            "check the curve name and that the wells are tied")

    x = np.concatenate(xs)
    y = np.concatenate(ys)

    # Trim the far tails so one washed-out interval cannot set the slope.
    if 0 < clip_percentile < 100 and y.size > 32:
        lo, hi = np.percentile(y, [100 - clip_percentile, clip_percentile])
        keep = (y >= lo) & (y <= hi)
        if keep.sum() > 16:
            x, y = x[keep], y[keep]

    degree = int(max(1, min(degree, 3)))
    if x.size <= degree + 1:
        raise ValueError(f"only {x.size} usable points: too few for a degree-{degree} fit")
    coeffs = np.polyfit(x, y, degree)
    pred = np.polyval(coeffs, x)
    resid = y - pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = max(x.size - (degree + 1), 1)

    fit = PropertyFit(
        property_name=mnemonic, coefficients=coeffs, degree=degree,
        r_squared=float(r2), rmse=float(np.sqrt(np.mean(resid ** 2))),
        residual_std=float(np.sqrt(ss_res / dof)), n_points=int(x.size),
        wells=used, unit=str(getattr(wells[0], "curve_units", {}).get(mnemonic, "")),
        ai_range=(float(np.exp(x.min())), float(np.exp(x.max()))),
    )
    if r2 < 0.3:
        fit.notes.append(
            f"R^2 is only {r2:.2f}: impedance explains little of this curve, so the "
            "predicted cube will be closer to a constant than to a measurement")
    return fit


def apply_fit(ai_cube: np.ndarray, fit: PropertyFit, posterior_std: np.ndarray | None = None) -> dict:
    """Predict the property over a cube, with an uncertainty if one is available.

    The property standard deviation combines the two independent terms::

        sigma_P^2 = (dP/dln(AI) * sigma_lnAI)^2  +  sigma_fit^2

    The first is what the seismic did not resolve; the second is the scatter of
    the wells about the transform, which stays whatever the seismic does.  Given
    only a deterministic impedance cube the first term is absent and the answer
    is the transform scatter alone -- reported, so it cannot be mistaken for the
    full uncertainty.
    """
    ai = np.asarray(ai_cube, dtype=float)
    prop = fit.predict(ai)

    out = {"property": prop.astype(np.float32), "fit": fit}
    if posterior_std is None:
        sigma = np.full(prop.shape, fit.residual_std, dtype=np.float32)
        out["uncertainty_source"] = "transform scatter only (no posterior supplied)"
    else:
        ps = np.asarray(posterior_std, dtype=float)
        if ps.shape != ai.shape:
            raise ValueError("posterior_std does not match the impedance cube")
        sigma = np.sqrt((fit.slope(ai) * ps) ** 2 + fit.residual_std ** 2)
        out["uncertainty_source"] = "inversion posterior + transform scatter"
    out["sigma"] = np.asarray(sigma, dtype=np.float32)
    out["p10"] = (prop - Z90 * sigma).astype(np.float32)
    out["p90"] = (prop + Z90 * sigma).astype(np.float32)
    return out


def probability_above(prop: np.ndarray, sigma: np.ndarray, threshold: float,
                      below: bool = False) -> np.ndarray:
    """Per-sample probability of clearing a cut-off, given the uncertainty.

    A deterministic cube answers this with 0 or 1 everywhere, which is the one
    answer that is certainly wrong.  With a Gaussian property this is the normal
    tail, and it degrades sensibly: where the seismic constrained impedance
    tightly the probability is close to 0 or 1, and where it did not it drifts
    toward 0.5 instead of pretending to know.
    """
    from scipy.special import erf

    prop = np.asarray(prop, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-12)
    z = (prop - float(threshold)) / (sigma * np.sqrt(2.0))
    p = 0.5 * (1.0 + erf(z))
    return np.asarray(1.0 - p if below else p, dtype=np.float32)


def expected_thickness(probability: np.ndarray, dt_ms: float,
                       gate_mask: np.ndarray | None = None) -> np.ndarray:
    """Expected net thickness in ms, by summing probability down each trace.

    The expectation of a sum is the sum of expectations whatever the samples'
    correlation, so this is exact for the mean even though the samples are not
    independent.  It is *not* a distribution of thickness -- for that, sum the
    stochastic realisations instead.
    """
    p = np.asarray(probability, dtype=float)
    if gate_mask is not None:
        p = p * np.asarray(gate_mask, dtype=float)[None, None, :]
    return np.asarray(p.sum(axis=2) * float(dt_ms), dtype=np.float32)


def crossplot_points(volume, wells, mnemonic: str, gate: tuple[float, float] | None = None,
                     ties=None) -> dict:
    """The raw (impedance, property) pairs behind a fit, for plotting."""
    from . import data_io

    twt = np.asarray(volume.twt, dtype=float)
    located = [w for w in wells if getattr(w, "has_location", False)]
    if ties is None:
        ties = data_io.extract_well_traces(volume, located, k=4)
    by_name = {t.well: t for t in ties}
    out: dict[str, dict] = {}
    for well in located:
        tie = by_name.get(well.name)
        if tie is None:
            continue
        prop = curve_on_time_axis(well, mnemonic, twt)
        ai = np.asarray(tie.ai, dtype=float)
        good = np.isfinite(prop) & np.isfinite(ai) & (ai > 0)
        if gate is not None:
            good &= (twt >= gate[0]) & (twt <= gate[1])
        if good.sum() < 4:
            continue
        out[well.name] = {"ai": ai[good], "property": prop[good], "twt": twt[good]}
    return out
