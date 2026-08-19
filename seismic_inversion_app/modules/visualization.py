"""Plotly figures for the app: sections, spectra, tie QC, crossplots, base map.

Every function returns a bare ``plotly.graph_objects.Figure`` and takes plain
numpy arrays or the module dataclasses, so the figures can be reused outside
Streamlit (a notebook, a report) without change.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import utils

SEISMIC_SCALE = "RdBu"
IMPEDANCE_SCALE = "Viridis"
_TEMPLATE = "plotly_white"


# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------

def _section_slice(cube: np.ndarray, orientation: str, index: int) -> np.ndarray:
    """Pull a 2D section, returned as ``(n_samples, n_traces)`` for imshow."""
    if orientation == "inline":
        return np.asarray(cube[index, :, :]).T
    return np.asarray(cube[:, index, :]).T


def section_figure(
    cube: np.ndarray,
    twt: np.ndarray,
    axis_values: np.ndarray,
    orientation: str = "inline",
    index: int = 0,
    title: str = "",
    colorscale: str = SEISMIC_SCALE,
    symmetric: bool = True,
    clip_percentile: float = 99.0,
    colorbar_title: str = "amplitude",
    wells: Sequence[dict] | None = None,
    height: int = 520,
) -> go.Figure:
    """A single seismic / attribute section with an optional well overlay.

    ``wells`` entries are ``{"name", "position", "t_min", "t_max"}`` where
    ``position`` is the value on the horizontal axis (a crossline number on an
    inline section, and vice versa).
    """
    img = _section_slice(cube, orientation, index)

    finite = img[np.isfinite(img)]
    if symmetric:
        lim = utils.safe_percentile_clip(img, clip_percentile)
        zmin, zmax = -lim, lim
    elif finite.size:
        zmin = float(np.percentile(finite, 100 - clip_percentile))
        zmax = float(np.percentile(finite, clip_percentile))
    else:
        zmin, zmax = 0.0, 1.0

    fig = go.Figure(
        go.Heatmap(
            z=img,
            x=np.asarray(axis_values, dtype=float),
            y=np.asarray(twt, dtype=float),
            colorscale=colorscale,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title=colorbar_title, thickness=14),
            hovertemplate=(
                f"{'xline' if orientation == 'inline' else 'iline'}: %{{x}}<br>"
                "twt: %{y:.0f} ms<br>value: %{z:.4g}<extra></extra>"
            ),
        )
    )

    for w in wells or []:
        t0 = w.get("t_min", float(np.min(twt)))
        t1 = w.get("t_max", float(np.max(twt)))
        fig.add_trace(
            go.Scatter(
                x=[w["position"], w["position"]], y=[t0, t1],
                mode="lines+text", line=dict(color="black", width=2, dash="solid"),
                text=[w["name"], ""], textposition="top center",
                showlegend=False, hoverinfo="text", hovertext=w["name"],
            )
        )

    fig.update_layout(
        title=title or f"{orientation} {axis_values if np.isscalar(axis_values) else ''}",
        template=_TEMPLATE, height=height,
        xaxis_title="crossline" if orientation == "inline" else "inline",
        yaxis_title="TWT (ms)",
        margin=dict(l=60, r=20, t=50, b=45),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def dual_section_figure(
    seismic: np.ndarray,
    impedance: np.ndarray,
    twt: np.ndarray,
    axis_values: np.ndarray,
    orientation: str = "inline",
    index: int = 0,
    titles: tuple[str, str] = ("Seismic amplitude", "Inverted impedance"),
    impedance_scale: str = IMPEDANCE_SCALE,
    clip_percentile: float = 99.0,
    wells: Sequence[dict] | None = None,
    height: int = 560,
) -> go.Figure:
    """Seismic beside the inversion result, on a shared time axis.

    Sharing the y-axis is the point: it is what lets you check that an
    impedance boundary sits where the reflection is, not 20 ms off it.
    """
    left = _section_slice(seismic, orientation, index)
    right = _section_slice(impedance, orientation, index)

    lim = utils.safe_percentile_clip(left, clip_percentile)
    rf = right[np.isfinite(right)]
    rmin = float(np.percentile(rf, 100 - clip_percentile)) if rf.size else 0.0
    rmax = float(np.percentile(rf, clip_percentile)) if rf.size else 1.0

    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=list(titles),
                        horizontal_spacing=0.09)
    x = np.asarray(axis_values, dtype=float)
    y = np.asarray(twt, dtype=float)

    fig.add_trace(
        go.Heatmap(z=left, x=x, y=y, colorscale=SEISMIC_SCALE, zmin=-lim, zmax=lim,
                   colorbar=dict(title="amp", thickness=12, x=0.44), name="seismic"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(z=right, x=x, y=y, colorscale=impedance_scale, zmin=rmin, zmax=rmax,
                   colorbar=dict(title="AI", thickness=12, x=1.01), name="impedance"),
        row=1, col=2,
    )

    for w in wells or []:
        for col in (1, 2):
            fig.add_trace(
                go.Scatter(x=[w["position"], w["position"]],
                           y=[w.get("t_min", y.min()), w.get("t_max", y.max())],
                           mode="lines", line=dict(color="black", width=2),
                           showlegend=False, hoverinfo="text", hovertext=w["name"]),
                row=1, col=col,
            )

    axis_name = "crossline" if orientation == "inline" else "inline"
    fig.update_xaxes(title_text=axis_name, row=1, col=1)
    fig.update_xaxes(title_text=axis_name, row=1, col=2)
    fig.update_yaxes(title_text="TWT (ms)", autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(template=_TEMPLATE, height=height, margin=dict(l=60, r=20, t=55, b=45))
    return fig


def time_slice_figure(
    cube: np.ndarray,
    twt: np.ndarray,
    iline: np.ndarray,
    xline: np.ndarray,
    t_ms: float,
    title: str = "",
    colorscale: str = IMPEDANCE_SCALE,
    symmetric: bool = False,
    clip_percentile: float = 99.0,
    wells: Sequence[dict] | None = None,
    height: int = 520,
) -> go.Figure:
    """Map view at one TWT, with well symbols at their il/xl positions."""
    k = int(np.argmin(np.abs(np.asarray(twt, dtype=float) - float(t_ms))))
    img = np.asarray(cube[:, :, k]).T          # (xline, iline) for imshow

    finite = img[np.isfinite(img)]
    if symmetric:
        lim = utils.safe_percentile_clip(img, clip_percentile)
        zmin, zmax = -lim, lim
    elif finite.size:
        zmin = float(np.percentile(finite, 100 - clip_percentile))
        zmax = float(np.percentile(finite, clip_percentile))
    else:
        zmin, zmax = 0.0, 1.0

    fig = go.Figure(
        go.Heatmap(z=img, x=np.asarray(iline, dtype=float), y=np.asarray(xline, dtype=float),
                   colorscale=colorscale, zmin=zmin, zmax=zmax,
                   colorbar=dict(thickness=14),
                   hovertemplate="iline: %{x}<br>xline: %{y}<br>value: %{z:.4g}<extra></extra>")
    )
    if wells:
        fig.add_trace(
            go.Scatter(
                x=[w["iline"] for w in wells], y=[w["xline"] for w in wells],
                mode="markers+text", text=[w["name"] for w in wells], textposition="top center",
                marker=dict(color="black", size=9, symbol="circle-open", line=dict(width=2)),
                showlegend=False, name="wells",
            )
        )
    fig.update_layout(
        title=title or f"Time slice at {twt[k]:.0f} ms",
        template=_TEMPLATE, height=height,
        xaxis_title="inline", yaxis_title="crossline",
        margin=dict(l=60, r=20, t=50, b=45),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


# --------------------------------------------------------------------------
# Wavelet QC
# --------------------------------------------------------------------------

def wavelet_figure(wav, max_freq: float | None = None, height: int = 400) -> go.Figure:
    """Wavelet in time, plus its amplitude and phase spectra."""
    fig = make_subplots(
        rows=1, cols=3, subplot_titles=("Wavelet", "Amplitude spectrum", "Phase spectrum"),
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(x=wav.time_axis, y=wav.samples, mode="lines",
                   line=dict(color="#1f4e79", width=2), name="wavelet"),
        row=1, col=1,
    )
    fig.add_hline(y=0, line=dict(color="grey", width=1), row=1, col=1)

    freq, amp = wav.spectrum()
    nyq = 0.5 / wav.dt
    fmax = max_freq or min(nyq, 120.0)
    band = freq <= fmax
    amp_db = 20 * np.log10(np.maximum(amp / max(amp.max(), 1e-12), 1e-6))
    fig.add_trace(
        go.Scatter(x=freq[band], y=amp_db[band], mode="lines",
                   line=dict(color="#c0504d", width=2), name="amplitude"),
        row=1, col=2,
    )
    fig.add_hline(y=-6, line=dict(color="grey", width=1, dash="dot"), row=1, col=2)

    pfreq, ph = wav.phase_spectrum()
    pband = pfreq <= fmax
    fig.add_trace(
        go.Scatter(x=pfreq[pband], y=ph[pband], mode="lines",
                   line=dict(color="#4f81bd", width=2), name="phase", connectgaps=False),
        row=1, col=3,
    )

    fig.update_xaxes(title_text="time (ms)", row=1, col=1)
    fig.update_xaxes(title_text="frequency (Hz)", row=1, col=2)
    fig.update_xaxes(title_text="frequency (Hz)", row=1, col=3)
    fig.update_yaxes(title_text="amplitude", row=1, col=1)
    fig.update_yaxes(title_text="dB", range=[-40, 3], row=1, col=2)
    fig.update_yaxes(title_text="degrees", range=[-190, 190], row=1, col=3)
    fig.update_layout(template=_TEMPLATE, height=height, showlegend=False,
                      margin=dict(l=55, r=20, t=45, b=45))
    return fig


def colour_operator_figure(op, max_freq: float | None = None, height: int = 400) -> go.Figure:
    """The three spectra behind a coloured-inversion operator.

    Seeing the seismic spectrum, the power-law target and the resulting
    operator on one axis is the fastest way to spot an operator that is boosting
    noise outside the real bandwidth.
    """
    freq = op.seismic_freq
    nyq = 0.5 / op.dt
    fmax = max_freq or min(nyq, 120.0)
    band = freq <= fmax

    def db(a):
        a = np.asarray(a, dtype=float)
        return 20 * np.log10(np.maximum(a / max(np.max(a[band]) if band.any() else 1.0, 1e-12), 1e-6))

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Spectra (normalised)", "Operator in time"),
                        horizontal_spacing=0.10)
    fig.add_trace(go.Scatter(x=freq[band], y=db(op.seismic_amp)[band], mode="lines",
                             name="seismic", line=dict(color="#4f81bd", width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=freq[band], y=db(op.target_amp)[band], mode="lines",
                             name=f"target f^{op.exponent:+.2f}", line=dict(color="#c0504d", width=2, dash="dash")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=freq[band], y=db(op.operator_amp)[band], mode="lines",
                             name="operator", line=dict(color="#4bacc6", width=2)), row=1, col=1)
    fig.add_vrect(x0=op.f_low, x1=op.f_high, fillcolor="grey", opacity=0.10, line_width=0, row=1, col=1)

    t = (np.arange(op.samples.size) - op.samples.size // 2) * op.dt * 1000.0
    fig.add_trace(go.Scatter(x=t, y=op.samples, mode="lines", name="operator",
                             line=dict(color="#1f4e79", width=2), showlegend=False), row=1, col=2)

    fig.update_xaxes(title_text="frequency (Hz)", row=1, col=1)
    fig.update_xaxes(title_text="time (ms)", row=1, col=2)
    fig.update_yaxes(title_text="dB", range=[-45, 5], row=1, col=1)
    fig.update_layout(template=_TEMPLATE, height=height,
                      legend=dict(orientation="h", y=1.14, x=0),
                      margin=dict(l=55, r=20, t=60, b=45))
    return fig


def spectrum_comparison_figure(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    max_freq: float = 120.0,
    height: int = 340,
    title: str = "Amplitude spectra",
) -> go.Figure:
    """Overlay several named ``(freq, amp)`` spectra, normalised to dB."""
    fig = go.Figure()
    for name, (freq, amp) in series.items():
        freq = np.asarray(freq, dtype=float)
        amp = np.asarray(amp, dtype=float)
        band = freq <= max_freq
        if not band.any():
            continue
        db = 20 * np.log10(np.maximum(amp / max(np.max(amp[band]), 1e-12), 1e-6))
        fig.add_trace(go.Scatter(x=freq[band], y=db[band], mode="lines", name=name))
    fig.update_layout(
        title=title, template=_TEMPLATE, height=height,
        xaxis_title="frequency (Hz)", yaxis_title="dB", yaxis_range=[-45, 3],
        margin=dict(l=55, r=20, t=45, b=45),
    )
    return fig


# --------------------------------------------------------------------------
# Well tie QC
# --------------------------------------------------------------------------

def well_tie_figure(
    tie,
    wavelet: np.ndarray,
    t_min: float | None = None,
    t_max: float | None = None,
    height: int = 620,
) -> go.Figure:
    """Reflectivity, synthetic-vs-extracted overlay, and the residual.

    Wells are assumed tied upstream, so this panel exists to *verify* the tie,
    not to adjust it: if the synthetic and the extracted trace are out of phase
    here, the wavelet or the tie is wrong and the inversion below will inherit it.
    """
    twt = np.asarray(tie.twt, dtype=float)
    refl = np.nan_to_num(np.asarray(tie.reflectivity, dtype=float))
    seis = np.nan_to_num(np.asarray(tie.seismic, dtype=float))
    synth = utils.convolve_same(refl, np.asarray(wavelet, dtype=float))

    i0 = int(np.searchsorted(twt, t_min)) if t_min is not None else 0
    i1 = int(np.searchsorted(twt, t_max)) if t_max is not None else twt.size
    corr = utils.normalised_correlation(synth[i0:i1], seis[i0:i1])
    lag, lag_corr = utils.best_lag_correlation(seis[i0:i1], synth[i0:i1], max_lag=40)
    dt_ms = float(np.median(np.diff(twt))) if twt.size > 1 else 1.0

    fig = make_subplots(
        rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.035,
        subplot_titles=("Well AI", "Reflectivity", "Synthetic vs seismic", "Residual"),
    )

    fig.add_trace(go.Scatter(x=tie.ai, y=twt, mode="lines", name="well AI",
                             line=dict(color="#8064a2", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=refl, y=twt, mode="lines", name="reflectivity",
                             line=dict(color="#4f81bd", width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=seis, y=twt, mode="lines", name="seismic (extracted)",
                             line=dict(color="black", width=1.5)), row=1, col=3)
    fig.add_trace(go.Scatter(x=synth, y=twt, mode="lines", name="synthetic",
                             line=dict(color="#c0504d", width=1.5, dash="dot")), row=1, col=3)
    fig.add_trace(go.Scatter(x=seis - synth, y=twt, mode="lines", name="residual",
                             line=dict(color="#9bbb59", width=1)), row=1, col=4)

    if t_min is not None and t_max is not None:
        for col in range(1, 5):
            fig.add_hrect(y0=t_min, y1=t_max, fillcolor="orange", opacity=0.07,
                          line_width=0, row=1, col=col)

    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=1, col=1)
    for col in range(2, 5):
        fig.update_yaxes(autorange="reversed", row=1, col=col)
    fig.update_layout(
        template=_TEMPLATE, height=height,
        title=(f"{tie.well} &nbsp;|&nbsp; IL {tie.iline} / XL {tie.xline} "
               f"({tie.distance:.0f} m, {tie.n_neighbours} traces blended) &nbsp;|&nbsp; "
               f"correlation {corr:.3f} &nbsp;|&nbsp; best lag {lag * dt_ms:+.0f} ms (r={lag_corr:.3f})"),
        legend=dict(orientation="h", y=-0.08), margin=dict(l=55, r=20, t=80, b=60),
    )
    return fig


def tie_score_table(ties: Sequence, wavelet: np.ndarray, t_min=None, t_max=None) -> list[dict]:
    """Per-well tie statistics for the summary table in the app."""
    rows = []
    for tie in ties:
        twt = np.asarray(tie.twt, dtype=float)
        refl = np.nan_to_num(np.asarray(tie.reflectivity, dtype=float))
        seis = np.nan_to_num(np.asarray(tie.seismic, dtype=float))
        synth = utils.convolve_same(refl, np.asarray(wavelet, dtype=float))
        i0 = int(np.searchsorted(twt, t_min)) if t_min is not None else 0
        i1 = int(np.searchsorted(twt, t_max)) if t_max is not None else twt.size
        dt_ms = float(np.median(np.diff(twt))) if twt.size > 1 else 1.0
        lag, lag_corr = utils.best_lag_correlation(seis[i0:i1], synth[i0:i1], max_lag=40)
        rows.append({
            "well": tie.well,
            "iline": tie.iline,
            "xline": tie.xline,
            "distance (m)": round(tie.distance, 1),
            "correlation": round(utils.normalised_correlation(synth[i0:i1], seis[i0:i1]), 3),
            "best lag (ms)": round(lag * dt_ms, 1),
            "corr at best lag": round(lag_corr, 3),
        })
    return rows


# --------------------------------------------------------------------------
# Crossplot
# --------------------------------------------------------------------------

def crossplot_figure(
    crossplot: dict[str, dict],
    x_label: str = "Well AI (m/s * kg/m3)",
    y_label: str = "Inverted AI",
    height: int = 520,
    max_points: int = 4000,
) -> go.Figure:
    """Inverted impedance against well impedance, coloured by TWT, one trace per well.

    The 1:1 line is drawn deliberately: a tight cloud that sits *off* it means
    the inversion is well-correlated but mis-scaled, which is a different
    problem from a cloud that is simply diffuse.
    """
    fig = go.Figure()
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []

    for well, d in crossplot.items():
        x, y, t = d["well"], d["inverted"], d["twt"]
        if x.size > max_points:
            step = int(np.ceil(x.size / max_points))
            x, y, t = x[::step], y[::step], t[::step]
        all_x.append(x)
        all_y.append(y)
        fig.add_trace(
            go.Scatter(
                x=x, y=y, mode="markers",
                name=f"{well} (r={d['correlation']:.3f})",
                marker=dict(size=4, opacity=0.55, color=t, colorscale="Turbo", showscale=False),
                hovertemplate=(f"{well}<br>well: %{{x:.0f}}<br>inverted: %{{y:.0f}}"
                               "<br>twt: %{marker.color:.0f} ms<extra></extra>"),
            )
        )

    if all_x:
        cat_x = np.concatenate(all_x)
        cat_y = np.concatenate(all_y)
        lo = float(min(cat_x.min(), cat_y.min()))
        hi = float(max(cat_x.max(), cat_y.max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="1:1",
                                 line=dict(color="grey", width=1, dash="dash")))
        if cat_x.size > 2:
            slope, intercept = np.polyfit(cat_x, cat_y, 1)
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[slope * lo + intercept, slope * hi + intercept], mode="lines",
                name=f"fit (slope {slope:.2f}, r={utils.normalised_correlation(cat_x, cat_y):.3f})",
                line=dict(color="black", width=1.5)))

    fig.update_layout(template=_TEMPLATE, height=height, xaxis_title=x_label, yaxis_title=y_label,
                      legend=dict(orientation="h", y=-0.16), margin=dict(l=65, r=20, t=30, b=70))
    return fig


# --------------------------------------------------------------------------
# Base map and logs
# --------------------------------------------------------------------------

def basemap_figure(volume, wells: Sequence, ties: Sequence | None = None, height: int = 520) -> go.Figure:
    """Survey outline with well locations, so a mis-located well is obvious."""
    fig = go.Figure()
    x, y = volume.cdp_x, volume.cdp_y
    corners_x = [x[0, 0], x[0, -1], x[-1, -1], x[-1, 0], x[0, 0]]
    corners_y = [y[0, 0], y[0, -1], y[-1, -1], y[-1, 0], y[0, 0]]
    fig.add_trace(go.Scatter(x=corners_x, y=corners_y, mode="lines", name="survey outline",
                             line=dict(color="#4f81bd", width=2)))

    step = max(int(np.ceil(x.size / 4000) ** 0.5), 1)
    fig.add_trace(go.Scatter(
        x=x[::step, ::step].ravel(), y=y[::step, ::step].ravel(), mode="markers",
        marker=dict(size=2, color="#c9d7e8"), name="bin centres", hoverinfo="skip"))

    located = [w for w in wells if getattr(w, "has_location", False)]
    if located:
        fig.add_trace(go.Scatter(
            x=[w.x for w in located], y=[w.y for w in located],
            mode="markers+text", text=[w.name for w in located], textposition="top center",
            marker=dict(size=11, color="#c0504d", symbol="circle-open", line=dict(width=2)),
            name="wells"))

    off = [w.name for w in wells if not getattr(w, "has_location", False)]
    fig.update_layout(
        template=_TEMPLATE, height=height, xaxis_title="X", yaxis_title="Y",
        title="Survey base map" + (f" &nbsp;|&nbsp; no location: {', '.join(off)}" if off else ""),
        margin=dict(l=70, r=20, t=50, b=45), legend=dict(orientation="h", y=-0.13),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def well_log_figure(well, twt_axis: np.ndarray | None = None, height: int = 620) -> go.Figure:
    """Vp, density and AI against two-way time for one well."""
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=("Vp (m/s)", "Rho (kg/m3)", "AI"))
    good = well.valid_mask()
    twt = well.twt[good]
    fig.add_trace(go.Scatter(x=well.vp[good], y=twt, mode="lines",
                             line=dict(color="#4f81bd", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=well.rho[good], y=twt, mode="lines",
                             line=dict(color="#9bbb59", width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(x=well.ai[good], y=twt, mode="lines",
                             line=dict(color="#8064a2", width=1)), row=1, col=3)
    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=1, col=1)
    for col in (2, 3):
        fig.update_yaxes(autorange="reversed", row=1, col=col)
    fig.update_layout(template=_TEMPLATE, height=height, showlegend=False,
                      title=f"{well.name} logs", margin=dict(l=60, r=20, t=60, b=45))
    return fig


def log_qc_figure(well, vp_range=None, rho_range=None, height: int = 620) -> go.Figure:
    """Assigned Vp / density / AI against depth, with out-of-range samples marked.

    Colouring the implausible samples red is the fastest way to see a unit
    mistake: a wrongly-assigned unit turns a whole track red at once, whereas a
    bad hole section shows up as isolated patches.
    """
    from .data_io import AI_RANGE, RHO_RANGE, VP_RANGE

    vp_range = vp_range or VP_RANGE
    rho_range = rho_range or RHO_RANGE

    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=("Vp (m/s)", "Density (kg/m3)", "AI"))
    md = np.asarray(well.md, dtype=float)

    tracks = [
        (well.vp, vp_range, "#4f81bd", 1),
        (well.rho, rho_range, "#9bbb59", 2),
        (well.ai, AI_RANGE, "#8064a2", 3),
    ]
    for values, (lo, hi), colour, col in tracks:
        v = np.asarray(values, dtype=float)
        good = np.isfinite(v)
        fig.add_trace(go.Scatter(x=v[good], y=md[good], mode="lines",
                                 line=dict(color=colour, width=1), showlegend=False), row=1, col=col)
        bad = good & ((v < lo) | (v > hi))
        if bad.any():
            fig.add_trace(go.Scatter(x=v[bad], y=md[bad], mode="markers",
                                     marker=dict(color="#c0504d", size=3),
                                     name="outside plausible range",
                                     showlegend=(col == 1)), row=1, col=col)
        for edge in (lo, hi):
            fig.add_vline(x=edge, line=dict(color="grey", width=1, dash="dot"), row=1, col=col)

    fig.update_yaxes(autorange="reversed", title_text="MD (m)", row=1, col=1)
    for col in (2, 3):
        fig.update_yaxes(autorange="reversed", row=1, col=col)
    fig.update_layout(template=_TEMPLATE, height=height,
                      title=f"{well.name} - assigned curves &nbsp;|&nbsp; {well.selection.describe()}",
                      legend=dict(orientation="h", y=-0.08),
                      margin=dict(l=60, r=20, t=70, b=55))
    return fig


def curve_preview_figure(well, mnemonics, height: int = 560) -> go.Figure:
    """Raw LAS curves as they sit in the file, one track each.

    Deliberately unconverted -- this is for deciding *what a curve is*, so it
    has to show the numbers the file actually contains.
    """
    mnemonics = [m for m in mnemonics if m in well.curves][:6]
    if not mnemonics:
        return go.Figure().update_layout(template=_TEMPLATE, height=200,
                                         title="Select one or more curves to preview")

    md = np.asarray(well.md, dtype=float)
    titles = [f"{m} [{well.curve_units.get(m, '') or '-'}]" for m in mnemonics]
    fig = make_subplots(rows=1, cols=len(mnemonics), shared_yaxes=True,
                        horizontal_spacing=0.04, subplot_titles=titles)
    palette = ["#4f81bd", "#9bbb59", "#8064a2", "#c0504d", "#4bacc6", "#f79646"]
    for k, mnemonic in enumerate(mnemonics, start=1):
        v = np.asarray(well.curves[mnemonic], dtype=float)
        good = np.isfinite(v)
        fig.add_trace(go.Scatter(x=v[good], y=md[good], mode="lines",
                                 line=dict(color=palette[(k - 1) % len(palette)], width=1),
                                 showlegend=False), row=1, col=k)
        fig.update_yaxes(autorange="reversed", row=1, col=k)
    fig.update_yaxes(title_text="MD (m)", row=1, col=1)
    fig.update_layout(template=_TEMPLATE, height=height, title=f"{well.name} - raw curves",
                      margin=dict(l=60, r=20, t=70, b=45))
    return fig


def low_freq_qc_figure(model, ties: Sequence, height: int = 560) -> go.Figure:
    """Background model against the well AI it was built from, per well."""
    n = max(len(ties), 1)
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True, horizontal_spacing=0.04,
                        subplot_titles=[t.well for t in ties] or ["(no wells)"])
    for k, tie in enumerate(ties, start=1):
        lf = model.trace(tie.il_index, tie.xl_index)
        fig.add_trace(go.Scatter(x=tie.ai, y=model.twt, mode="lines", name="well AI",
                                 line=dict(color="#b0b0b0", width=1), showlegend=(k == 1)), row=1, col=k)
        fig.add_trace(go.Scatter(x=lf, y=model.twt, mode="lines", name=f"LFM (<{model.cutoff_hz:.0f} Hz)",
                                 line=dict(color="#c0504d", width=2), showlegend=(k == 1)), row=1, col=k)
        fig.update_yaxes(autorange="reversed", row=1, col=k)
    fig.update_yaxes(title_text="TWT (ms)", row=1, col=1)
    fig.update_layout(template=_TEMPLATE, height=height, title="Low-frequency model at the wells",
                      legend=dict(orientation="h", y=-0.10), margin=dict(l=60, r=20, t=60, b=60))
    return fig


def trace_comparison_figure(
    result: dict,
    twt: np.ndarray,
    trace: np.ndarray,
    low_freq: np.ndarray | None = None,
    well_ai: np.ndarray | None = None,
    height: int = 620,
) -> go.Figure:
    """Single-trace inversion QC: data fit, reflectivity, impedance."""
    twt = np.asarray(twt, dtype=float)
    fig = make_subplots(rows=1, cols=3, shared_yaxes=True, horizontal_spacing=0.05,
                        subplot_titles=("Seismic vs synthetic", "Reflectivity", "Impedance"))

    fig.add_trace(go.Scatter(x=trace, y=twt, mode="lines", name="seismic",
                             line=dict(color="black", width=1.4)), row=1, col=1)
    if result.get("method") != "coloured":
        fig.add_trace(go.Scatter(x=result["synthetic"], y=twt, mode="lines", name="synthetic",
                                 line=dict(color="#c0504d", width=1.4, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=result["reflectivity"], y=twt, mode="lines", name="reflectivity",
                             line=dict(color="#4f81bd", width=1)), row=1, col=2)

    if result.get("absolute_ai") is not None:
        fig.add_trace(go.Scatter(x=result["absolute_ai"], y=twt, mode="lines", name="inverted AI",
                                 line=dict(color="#8064a2", width=1.8)), row=1, col=3)
        if low_freq is not None:
            fig.add_trace(go.Scatter(x=low_freq, y=twt, mode="lines", name="low-frequency model",
                                     line=dict(color="#f79646", width=1.4, dash="dash")), row=1, col=3)
        if well_ai is not None:
            fig.add_trace(go.Scatter(x=well_ai, y=twt, mode="lines", name="well AI",
                                     line=dict(color="#b0b0b0", width=1)), row=1, col=3)
    else:
        fig.add_trace(go.Scatter(x=result["relative_ai"], y=twt, mode="lines", name="relative AI",
                                 line=dict(color="#8064a2", width=1.8)), row=1, col=3)

    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=1, col=1)
    for col in (2, 3):
        fig.update_yaxes(autorange="reversed", row=1, col=col)

    corr = result.get("correlation", float("nan"))
    misfit = result.get("misfit", float("nan"))
    bits = [f"method: {result.get('method', '?')}", f"correlation {corr:.3f}"]
    if np.isfinite(misfit):
        bits.append(f"misfit {misfit:.3f}")
    fig.update_layout(template=_TEMPLATE, height=height, title=" &nbsp;|&nbsp; ".join(bits),
                      legend=dict(orientation="h", y=-0.09), margin=dict(l=60, r=20, t=60, b=60))
    return fig


def qc_map_figure(values: np.ndarray, iline: np.ndarray, xline: np.ndarray,
                  title: str = "Trace QC", height: int = 440) -> go.Figure:
    """Map of a per-trace QC statistic (correlation or misfit)."""
    img = np.asarray(values, dtype=float).T
    finite = img[np.isfinite(img)]
    fig = go.Figure(go.Heatmap(
        z=img, x=np.asarray(iline, dtype=float), y=np.asarray(xline, dtype=float),
        colorscale="Cividis",
        zmin=float(np.nanpercentile(finite, 2)) if finite.size else 0.0,
        zmax=float(np.nanpercentile(finite, 98)) if finite.size else 1.0,
        colorbar=dict(thickness=14),
        hovertemplate="iline: %{x}<br>xline: %{y}<br>%{z:.3f}<extra></extra>"))
    fig.update_layout(title=title, template=_TEMPLATE, height=height,
                      xaxis_title="inline", yaxis_title="crossline",
                      margin=dict(l=60, r=20, t=50, b=45))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def well_overlay_positions(ties: Sequence, orientation: str, index: int,
                           volume, tolerance: int = 2) -> list[dict]:
    """Which wells to draw on a given section, and where on its x-axis.

    ``tolerance`` is how many bins away a well may sit and still be projected
    onto the section -- drawing a well 40 lines away would be a lie.
    """
    out = []
    for tie in ties:
        if orientation == "inline":
            if abs(tie.il_index - index) > tolerance:
                continue
            pos = float(volume.xline[tie.xl_index])
        else:
            if abs(tie.xl_index - index) > tolerance:
                continue
            pos = float(volume.iline[tie.il_index])
        good = np.isfinite(tie.ai) & (tie.ai > 0)
        t0 = float(tie.twt[good].min()) if good.any() else float(tie.twt.min())
        t1 = float(tie.twt[good].max()) if good.any() else float(tie.twt.max())
        out.append({"name": tie.well, "position": pos, "t_min": t0, "t_max": t1})
    return out
