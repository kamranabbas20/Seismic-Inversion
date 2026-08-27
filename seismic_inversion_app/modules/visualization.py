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
    gain: float = 1.0,
) -> go.Figure:
    """A single seismic / attribute section with an optional well overlay.

    ``wells`` entries are ``{"name", "position", "t_min", "t_max"}`` where
    ``position`` is the value on the horizontal axis (a crossline number on an
    inline section, and vice versa).
    """
    img = _section_slice(cube, orientation, index)

    finite = img[np.isfinite(img)]
    if symmetric:
        lim = utils.safe_percentile_clip(img, clip_percentile) / max(gain, 1e-6)
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


def arbitrary_line_figure(
    line,
    title: str = "",
    colorscale: str = SEISMIC_SCALE,
    clip_percentile: float = 99.0,
    gain: float = 1.0,
    height: int = 540,
    show_nodes: bool = True,
) -> go.Figure:
    """A section sampled along a polyline, with the well nodes marked.

    The horizontal axis is true distance along the path, not trace number, so
    an unevenly-spaced traverse is not silently stretched.
    """
    img = np.asarray(line.data, dtype=float).T
    lim = utils.safe_percentile_clip(img, clip_percentile) / max(gain, 1e-6)

    fig = go.Figure(go.Heatmap(
        z=img, x=line.distance, y=line.twt, colorscale=colorscale,
        zmin=-lim, zmax=lim, colorbar=dict(title="amp", thickness=14),
        hovertemplate="distance: %{x:.0f} m<br>twt: %{y:.0f} ms<br>amp: %{z:.4g}<extra></extra>"))

    if show_nodes and len(line.node_label):
        t0, t1 = float(np.min(line.twt)), float(np.max(line.twt))
        for distance, label in zip(line.node_distance, line.node_label):
            fig.add_trace(go.Scatter(
                x=[distance, distance], y=[t0, t1], mode="lines",
                line=dict(color="black", width=2), showlegend=False,
                hoverinfo="text", hovertext=label))
            fig.add_annotation(x=distance, y=t0, text=label, showarrow=False,
                               yshift=10, font=dict(size=11))

    fig.update_layout(
        title=title or "Arbitrary line", template=_TEMPLATE, height=height,
        xaxis_title="distance along traverse (m)", yaxis_title="TWT (ms)",
        margin=dict(l=60, r=20, t=55, b=45))
    fig.update_yaxes(autorange="reversed")
    return fig


# --------------------------------------------------------------------------
# Well correlation panel
# --------------------------------------------------------------------------

_MARKER_PALETTE = ["#c0504d", "#4f81bd", "#9bbb59", "#8064a2", "#4bacc6", "#f79646",
                   "#7f6084", "#2c7873", "#b06f3a", "#5b6b8c"]


def _wiggle_polygon(amp: np.ndarray, twt: np.ndarray, centre: float, half_width: float):
    """Positive-lobe fill polygon for a wiggle trace, plus the wiggle line.

    Returns ``(fill_x, fill_y, line_x)``.  Only the positive lobes are filled,
    which is the convention that makes a correlation panel readable: the eye
    follows the black peaks across the wells.
    """
    amp = np.nan_to_num(np.asarray(amp, dtype=float))
    line_x = centre + amp * half_width
    fill_x = centre + np.clip(amp, 0.0, None) * half_width
    poly_x = np.concatenate([fill_x, np.full(fill_x.size, centre)[::-1]])
    poly_y = np.concatenate([twt, twt[::-1]])
    return poly_x, poly_y, line_x


def correlation_figure(
    wells,
    order,
    ties=None,
    curve: str = "AI",
    show_logs: bool = True,
    show_seismic: bool = True,
    flatten_marker: str | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    gain: float = 1.0,
    marker_names=None,
    connect_markers: bool = True,
    height: int = 800,
) -> go.Figure:
    """A classic well-correlation panel: wells side by side in a chosen order.

    Drawn on a single axes with one slot per well rather than as subplots,
    because the correlation lines have to run *between* wells -- in a subplot
    grid that means paper-coordinate shapes that drift out of alignment as soon
    as the layout changes.

    Logs are scaled on a shared range across every displayed well, so a
    thickening or a sharpening really is comparable from one well to the next.
    ``flatten_marker`` hangs the panel on a chosen top; without it the datum is
    two-way time.
    """
    ties = {t.well: t for t in (ties or [])}
    by_name = {w.name: w for w in wells}
    names = [n for n in order if n in by_name]
    if not names:
        raise ValueError("no wells selected for correlation")

    # ---- vertical shifts -------------------------------------------------
    shifts: dict[str, float] = {n: 0.0 for n in names}
    missing_datum: list[str] = []
    if flatten_marker:
        picks = {}
        for n in names:
            hit = next((m for m in by_name[n].markers
                        if m.name == flatten_marker and m.twt is not None), None)
            if hit is None:
                missing_datum.append(n)
            else:
                picks[n] = float(hit.twt)
        if picks:
            reference = float(np.mean(list(picks.values())))
            shifts = {n: (reference - picks[n]) if n in picks else 0.0 for n in names}

    # ---- shared scaling --------------------------------------------------
    log_values, seis_values = [], []
    for n in names:
        well = by_name[n]
        if show_logs:
            values = _curve_values(well, curve)
            if values is not None:
                log_values.append(values[np.isfinite(values)])
        if show_seismic and n in ties:
            seis_values.append(np.nan_to_num(ties[n].seismic))
    log_lo, log_hi = _shared_range(log_values)
    seis_scale = (np.percentile(np.abs(np.concatenate(seis_values)), 99)
                  if seis_values else 1.0) or 1.0

    both = show_logs and show_seismic and log_values and seis_values
    fig = go.Figure()
    marker_colour: dict[str, str] = {}
    marker_positions: dict[str, list[tuple[float, float, float]]] = {}

    for i, n in enumerate(names):
        well = by_name[n]
        shift = shifts[n]
        if both:
            log_span, seis_span = (i + 0.06, i + 0.44), (i + 0.52, i + 0.94)
        else:
            log_span = seis_span = (i + 0.10, i + 0.90)

        if show_logs:
            values = _curve_values(well, curve)
            if values is not None:
                twt = np.asarray(well.twt, dtype=float) + shift
                good = np.isfinite(values) & np.isfinite(twt)
                if good.any():
                    lo, hi = log_span
                    scaled = lo + (np.clip(values[good], log_lo, log_hi) - log_lo) / max(log_hi - log_lo, 1e-9) * (hi - lo)
                    fig.add_trace(go.Scatter(
                        x=scaled, y=twt[good], mode="lines", name=curve,
                        line=dict(color="#8064a2", width=1),
                        legendgroup=curve, showlegend=(i == 0),
                        hovertemplate=f"{n}<br>{curve}: %{{customdata:.4g}}<br>twt: %{{y:.0f}} ms<extra></extra>",
                        customdata=values[good]))

        if show_seismic and n in ties:
            tie = ties[n]
            twt = np.asarray(tie.twt, dtype=float) + shift
            amp = np.nan_to_num(np.asarray(tie.seismic, dtype=float)) / seis_scale * gain
            amp = np.clip(amp, -1.2, 1.2)
            centre = 0.5 * (seis_span[0] + seis_span[1])
            half = 0.5 * (seis_span[1] - seis_span[0])
            poly_x, poly_y, line_x = _wiggle_polygon(amp, twt, centre, half)
            fig.add_trace(go.Scatter(
                x=poly_x, y=poly_y, fill="toself", mode="none",
                fillcolor="rgba(20,20,20,0.85)", hoverinfo="skip",
                legendgroup="seismic", showlegend=False))
            fig.add_trace(go.Scatter(
                x=line_x, y=twt, mode="lines", line=dict(color="black", width=0.8),
                name="seismic", legendgroup="seismic", showlegend=(i == 0),
                hovertemplate=f"{n}<br>amp: %{{customdata:.4g}}<br>twt: %{{y:.0f}} ms<extra></extra>",
                customdata=np.nan_to_num(tie.seismic)))

        # ---- markers -----------------------------------------------------
        for marker in well.markers:
            if marker.twt is None or not np.isfinite(marker.twt):
                continue
            if marker_names is not None and marker.name not in marker_names:
                continue
            y = float(marker.twt) + shift
            colour = marker_colour.setdefault(
                marker.name, _MARKER_PALETTE[len(marker_colour) % len(_MARKER_PALETTE)])
            fig.add_trace(go.Scatter(
                x=[i + 0.04, i + 0.96], y=[y, y], mode="lines",
                line=dict(color=colour, width=2), name=marker.name,
                legendgroup=marker.name, showlegend=marker.name not in marker_positions,
                hovertemplate=f"{n}<br>{marker.name}<br>MD {marker.md:.1f} m<br>"
                              f"TWT %{{y:.0f}} ms<extra></extra>"))
            marker_positions.setdefault(marker.name, []).append((i, y, colour))

    # ---- correlation lines between adjacent wells ------------------------
    if connect_markers:
        for name, hits in marker_positions.items():
            hits.sort()
            for (i0, y0, colour), (i1, y1, _) in zip(hits, hits[1:]):
                fig.add_trace(go.Scatter(
                    x=[i0 + 0.96, i1 + 0.04], y=[y0, y1], mode="lines",
                    line=dict(color=colour, width=1.2, dash="dot"),
                    legendgroup=name, showlegend=False, hoverinfo="skip"))

    for i in range(1, len(names)):
        fig.add_vline(x=i, line=dict(color="#d9d9d9", width=1))

    subtitle = f"flattened on {flatten_marker}" if flatten_marker else "hung on TWT"
    if missing_datum:
        subtitle += f" (no {flatten_marker} in: {', '.join(missing_datum)})"

    fig.update_layout(
        title=f"Well correlation - {len(names)} wells, {subtitle}",
        template=_TEMPLATE, height=height,
        xaxis=dict(tickmode="array", tickvals=[i + 0.5 for i in range(len(names))],
                   ticktext=names, range=[0, len(names)], showgrid=False),
        yaxis_title="TWT (ms)",
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=65, r=20, t=60, b=70))
    fig.update_yaxes(autorange="reversed")
    if t_min is not None and t_max is not None:
        fig.update_yaxes(range=[t_max, t_min])
    return fig


def _curve_values(well, curve: str):
    """Resolve a display curve by name, including the derived AI/Vp/Rho."""
    key = str(curve).upper()
    if key == "AI":
        return well.ai
    if key in ("VP", "VELOCITY"):
        return well.vp
    if key in ("RHO", "RHOB", "DENSITY"):
        return well.rho
    for mnemonic, values in well.curves.items():
        if mnemonic.upper() == key:
            return np.asarray(values, dtype=float)
    return None


def _shared_range(arrays, low: float = 2.0, high: float = 98.0):
    """Percentile range across every well, so the logs are comparable."""
    pooled = [a for a in arrays if a is not None and np.size(a)]
    if not pooled:
        return 0.0, 1.0
    flat = np.concatenate(pooled)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return 0.0, 1.0
    lo, hi = float(np.percentile(flat, low)), float(np.percentile(flat, high))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def common_curves(wells, order=None) -> list[str]:
    """Curves available in every selected well, plus the derived ones."""
    names = [w.name for w in wells] if order is None else list(order)
    chosen = [w for w in wells if w.name in names]
    if not chosen:
        return ["AI", "Vp", "Rho"]
    shared = set(chosen[0].curves)
    for well in chosen[1:]:
        shared &= set(well.curves)
    return ["AI", "Vp", "Rho"] + sorted(m for m in shared if m.upper() != "DEPT")


def common_markers(wells, order=None) -> list[str]:
    """Marker names present in at least two of the selected wells.

    A top that appears in only one well cannot be correlated, so offering it as
    a flattening datum would be misleading.
    """
    names = [w.name for w in wells] if order is None else list(order)
    counts: dict[str, int] = {}
    for well in wells:
        if well.name not in names:
            continue
        for marker in well.markers:
            if marker.twt is not None:
                counts[marker.name] = counts.get(marker.name, 0) + 1
    return sorted(name for name, count in counts.items() if count >= 2)


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
    markers=None,
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

    _add_markers(fig, markers, axis="twt", n_cols=4)

    fig.update_yaxes(autorange="reversed", title_text="TWT (ms)", row=1, col=1)
    for col in range(2, 5):
        fig.update_yaxes(autorange="reversed", row=1, col=col)
    fig.update_layout(
        template=_TEMPLATE, height=height,
        margin_r=95,
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


def time_depth_figure(td, markers=None, height: int = 480) -> go.Figure:
    """A checkshot as time against depth, beside the interval velocity it implies.

    The velocity panel is the one that matters for QC: a mis-keyed depth or a
    one-way/two-way mistake shows up there as a spike or a factor-of-two step,
    while the time-depth curve itself still looks smooth.
    """
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.08,
                        subplot_titles=("Time-depth", "Interval velocity"))

    fig.add_trace(go.Scatter(x=td.twt, y=td.md, mode="lines+markers",
                             line=dict(color="#4f81bd", width=2), marker=dict(size=5),
                             name="checkshot"), row=1, col=1)

    vi = td.interval_velocity()
    mid = 0.5 * (td.md[1:] + td.md[:-1])
    good = np.isfinite(vi)
    fig.add_trace(go.Scatter(x=vi[good], y=mid[good], mode="lines+markers",
                             line=dict(color="#c0504d", width=1.5, shape="hv"),
                             marker=dict(size=4), name="interval velocity"), row=1, col=2)

    span = float(td.md.max() - td.md.min()) or 1.0
    for depth, label in _thin_labels([(m.md, m.name) for m in (markers or [])], span):
        _hline(fig, depth, label, row=1, col=1)

    fig.update_xaxes(title_text="TWT (ms)", row=1, col=1)
    fig.update_xaxes(title_text="m/s", row=1, col=2)
    fig.update_yaxes(title_text="MD (m)", autorange="reversed", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(template=_TEMPLATE, height=height, showlegend=False,
                      margin=dict(l=60, r=90, t=45, b=45))
    return fig


def _hline(fig, y, label, row: int, col: int) -> None:
    """Horizontal marker line, annotated only when a label is supplied.

    Plotly renders ``annotation_text=None`` as the literal string "new text",
    so the annotation arguments have to be omitted entirely rather than passed
    as None.
    """
    kwargs = dict(y=y, line=dict(color="#7f7f7f", width=0.7, dash="dot"), row=row, col=col)
    if label:
        kwargs.update(annotation_text=label, annotation_position="right",
                      annotation_font_size=9)
    fig.add_hline(**kwargs)


def _thin_labels(items, span: float, min_gap_fraction: float = 0.05):
    """Keep every line but drop labels that would overlap.

    A well with twenty-five tops over three kilometres renders its names into
    an unreadable stack.  Lines still mark every top; only the text is thinned,
    so nothing is hidden -- it just stops fighting for the same pixels.
    """
    ordered = sorted((v, n) for v, n in items if v is not None and np.isfinite(v))
    if not ordered:
        return []
    min_gap = abs(span) * min_gap_fraction
    out, last = [], None
    for value, name in ordered:
        if last is None or abs(value - last) >= min_gap:
            out.append((value, name))
            last = value
        else:
            out.append((value, None))       # line without a label
    return out


def _add_markers(fig, markers, axis: str, n_cols: int) -> None:
    """Draw formation tops across every track of a figure.

    ``axis`` is ``md`` or ``twt`` -- the caller knows which the y-axis carries,
    and a marker plotted against the wrong one is worse than no marker at all.
    """
    values = [(m.md if axis == "md" else m.twt, m.name) for m in (markers or [])]
    finite = [v for v, _ in values if v is not None and np.isfinite(v)]
    if not finite:
        return
    span = (max(finite) - min(finite)) or 1.0

    for value, label in _thin_labels(values, span):
        for col in range(1, n_cols + 1):
            _hline(fig, value, label if col == n_cols else None, row=1, col=col)


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

    _add_markers(fig, getattr(well, "markers", None), axis="md", n_cols=3)

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


def property_fit_figure(points: dict, fit, height: int = 480) -> go.Figure:
    """Impedance against a well property, with the fitted transform over it.

    Colouring by well rather than pooling the points is deliberate: a transform
    that looks tight overall but splits into one cloud per well is telling you
    the relation is well-specific, and applying it across the survey will invent
    structure that follows nothing but the well locations.
    """
    fig = go.Figure()
    for i, (name, d) in enumerate(points.items()):
        fig.add_trace(go.Scattergl(
            x=np.asarray(d["ai"]) / 1e6, y=d["property"], mode="markers", name=name,
            marker=dict(size=4, opacity=0.45, color=_MARKER_PALETTE[i % len(_MARKER_PALETTE)]),
            hovertemplate="AI %{x:.2f}e6<br>%{y:.4g}<extra>" + name + "</extra>"))

    all_ai = np.concatenate([d["ai"] for d in points.values()]) if points else np.array([1.0, 2.0])
    grid = np.linspace(float(all_ai.min()), float(all_ai.max()), 200)
    fig.add_trace(go.Scatter(
        x=grid / 1e6, y=fit.predict(grid), mode="lines", name="fit",
        line=dict(color="#111111", width=2.5)))
    band = fit.residual_std
    fig.add_trace(go.Scatter(
        x=np.concatenate([grid, grid[::-1]]) / 1e6,
        y=np.concatenate([fit.predict(grid) + band, (fit.predict(grid) - band)[::-1]]),
        fill="toself", fillcolor="rgba(17,17,17,0.10)", line=dict(width=0),
        name="+/- 1 sd of the fit", hoverinfo="skip"))

    unit = f" [{fit.unit}]" if fit.unit else ""
    fig.update_layout(
        title=f"{fit.property_name}{unit} against acoustic impedance "
              f"(R² {fit.r_squared:.2f}, {fit.n_points} points)",
        template=_TEMPLATE, height=height,
        xaxis_title="acoustic impedance (10⁶ m/s·kg/m³)",
        yaxis_title=f"{fit.property_name}{unit}",
        margin=dict(l=60, r=20, t=55, b=50))
    return fig


def realisations_figure(draws: dict, twt: np.ndarray, well_ai=None,
                        gate=None, height: int = 560, max_shown: int = 40) -> go.Figure:
    """Stochastic realisations against the posterior mean and the well.

    Individual draws are plotted thin and translucent so the *envelope* is what
    the eye picks up, with the mean over the top: the point of the panel is that
    the mean is smoother than any realisation, not that any one draw matters.
    """
    twt = np.asarray(twt, dtype=float)
    ai = np.asarray(draws["absolute_ai"], dtype=float)
    fig = go.Figure()

    step = max(1, ai.shape[0] // max_shown)
    for k in range(0, ai.shape[0], step):
        fig.add_trace(go.Scatter(
            x=ai[k] / 1e6, y=twt, mode="lines",
            line=dict(color="rgba(120,140,170,0.30)", width=0.8),
            showlegend=(k == 0), name="realisations", hoverinfo="skip"))

    fig.add_trace(go.Scatter(x=np.asarray(draws["p10"]) / 1e6, y=twt, mode="lines",
                             line=dict(color="rgba(31,119,180,0.7)", width=1, dash="dot"), name="P10"))
    fig.add_trace(go.Scatter(x=np.asarray(draws["p90"]) / 1e6, y=twt, mode="lines",
                             line=dict(color="rgba(31,119,180,0.7)", width=1, dash="dot"), name="P90"))
    fig.add_trace(go.Scatter(x=np.asarray(draws["mean_ai"]) / 1e6, y=twt, mode="lines",
                             line=dict(color="#d62728", width=2), name="posterior mean"))
    if well_ai is not None:
        w = np.asarray(well_ai, dtype=float)
        good = np.isfinite(w) & (w > 0)
        if good.any():
            fig.add_trace(go.Scatter(x=w[good] / 1e6, y=twt[good], mode="lines",
                                     line=dict(color="#111111", width=1.6), name="well"))

    fig.update_layout(
        title="Equiprobable realisations from the posterior",
        template=_TEMPLATE, height=height,
        xaxis_title="acoustic impedance (10⁶ m/s·kg/m³)", yaxis_title="TWT (ms)",
        margin=dict(l=60, r=20, t=50, b=45), legend=dict(orientation="h", y=-0.12))
    fig.update_yaxes(autorange="reversed")
    if gate is not None:
        fig.update_yaxes(range=[gate[1], gate[0]])
    return fig


def nonstationary_wavelet_figure(nsw, height: int = 460) -> go.Figure:
    """Each window's wavelet and spectrum, so the drift is visible at a glance."""
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Wavelets by window", "Amplitude spectra"))
    t_axis = (np.arange(nsw.wavelets.shape[1]) - nsw.wavelets.shape[1] // 2) * nsw.dt * 1000.0
    for i, centre in enumerate(nsw.centres_ms):
        colour = _MARKER_PALETTE[i % len(_MARKER_PALETTE)]
        w = nsw.wavelets[i]
        peak = float(np.max(np.abs(w))) or 1.0
        fig.add_trace(go.Scatter(x=t_axis, y=w / peak, mode="lines", name=f"{centre:.0f} ms",
                                 line=dict(color=colour, width=1.8)), row=1, col=1)
        freq, amp = utils.amplitude_spectrum(w, nsw.dt, pad=4096)
        keep = freq <= min(0.5 / nsw.dt, 120)
        peak_a = float(np.max(amp)) or 1.0
        fig.add_trace(go.Scatter(x=freq[keep], y=amp[keep] / peak_a, mode="lines",
                                 name=f"{centre:.0f} ms", showlegend=False,
                                 line=dict(color=colour, width=1.8)), row=1, col=2)

    fig.update_xaxes(title_text="lag (ms)", row=1, col=1)
    fig.update_xaxes(title_text="frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="normalised amplitude", row=1, col=1)
    fig.update_layout(template=_TEMPLATE, height=height,
                      margin=dict(l=60, r=20, t=55, b=45),
                      legend=dict(orientation="h", y=-0.18))
    return fig


def attribute_error_figure(curve: dict, height: int = 460) -> go.Figure:
    """Training and validation error against the number of attributes.

    This is the whole argument of multi-attribute prediction in one panel.
    Training error can only fall as attributes are added -- it is not evidence
    of anything.  Validation error, measured on wells the fit never saw, turns
    back up once the extra freedom starts memorising rather than learning, and
    where it turns is where the model should stop.  The dashed line at the
    spread of the target is the error you would get by predicting the mean
    everywhere: a validation curve that never drops clearly below it means the
    seismic is not carrying the property, however good the training fit looks.
    """
    n = curve["n_attributes"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n, y=curve["training"], mode="lines+markers", name="training",
                             line=dict(color="#4f81bd", width=2.4), marker=dict(size=7)))
    fig.add_trace(go.Scatter(x=n, y=curve["validation"], mode="lines+markers",
                             name="validation (blind well)",
                             line=dict(color="#c0504d", width=2.4), marker=dict(size=7)))
    spread = float(curve.get("target_std") or 0.0)
    if spread > 0:
        fig.add_hline(y=spread, line=dict(color="#666666", width=1.4, dash="dash"),
                      annotation_text="predicting the mean", annotation_position="top right")
    chosen = int(curve.get("chosen") or 0)
    if chosen:
        fig.add_vline(x=chosen, line=dict(color="#111111", width=1.4, dash="dot"),
                      annotation_text=f"chosen: {chosen}", annotation_position="bottom right")
    top = max([spread] + list(curve["validation"]) + list(curve["training"])) * 1.15
    fig.update_layout(
        title="Where to stop adding attributes",
        template=_TEMPLATE, height=height,
        xaxis_title="number of attributes", yaxis_title="RMS prediction error",
        xaxis=dict(dtick=1), yaxis=dict(range=[0, top]),
        margin=dict(l=65, r=20, t=55, b=50),
        legend=dict(orientation="h", y=-0.2))
    return fig


def attribute_prediction_figure(predicted: np.ndarray, target: np.ndarray, twt: np.ndarray,
                                name: str, label: str, mask=None, height: int = 560) -> go.Figure:
    """A predicted curve against the measured log at one well.

    The view is cropped to the interval the model was trained over.  Outside it
    the prediction is still computed -- the operator simply pads at the ends of
    the trace -- and those padded tails run to values the log never takes, which
    would squash the part worth looking at into a narrow band.
    """
    fig = go.Figure()
    good = np.isfinite(target)
    shown = good if mask is None else (good & np.asarray(mask, dtype=bool))
    fig.add_trace(go.Scatter(x=target[good], y=twt[good], mode="lines", name="measured",
                             line=dict(color="#111111", width=2.0)))
    fig.add_trace(go.Scatter(x=predicted, y=twt, mode="lines", name="predicted",
                             line=dict(color="#c0504d", width=1.8)))
    yaxis = dict(autorange="reversed")
    xaxis = {}
    if shown.any():
        lo_t, hi_t = float(twt[shown].min()), float(twt[shown].max())
        pad_t = 0.02 * max(hi_t - lo_t, 1.0)
        yaxis = dict(range=[hi_t + pad_t, lo_t - pad_t])
        vals = np.concatenate([target[shown], np.asarray(predicted)[shown]])
        lo_v, hi_v = float(np.min(vals)), float(np.max(vals))
        pad_v = 0.05 * max(hi_v - lo_v, 1e-9)
        xaxis = dict(range=[lo_v - pad_v, hi_v + pad_v])
    fig.update_layout(title=f"{label} at {name}", template=_TEMPLATE, height=height,
                      xaxis_title=label, yaxis_title="TWT (ms)",
                      yaxis=yaxis, xaxis=xaxis,
                      margin=dict(l=65, r=20, t=55, b=50),
                      legend=dict(orientation="h", y=-0.13))
    return fig
