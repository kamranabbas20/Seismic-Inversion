"""Background (low-frequency) acoustic impedance model.

Model-based inversion is only as good as the trend it is regularised toward,
so this module is deliberately explicit about where the low frequencies come
from: well AI, low-pass filtered below the seismic band, interpolated laterally
between wells, and optionally flattened along interpreted horizons so the trend
follows structure instead of cutting across it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from . import utils

INTERP_METHODS = ("idw", "rbf", "nearest")


@dataclass
class LowFrequencyModel:
    """A 3D background AI cube on the seismic grid."""

    ai: np.ndarray                       # (n_il, n_xl, n_samples)
    twt: np.ndarray
    cutoff_hz: float
    method: str
    wells_used: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ai = np.asarray(self.ai, dtype=np.float32)
        self.twt = np.asarray(self.twt, dtype=float)

    def trace(self, i_il: int, i_xl: int) -> np.ndarray:
        return np.asarray(self.ai[i_il, i_xl, :], dtype=float)

    def summary(self) -> dict:
        finite = self.ai[np.isfinite(self.ai)]
        return {
            "method": self.method,
            "cutoff (Hz)": f"{self.cutoff_hz:.1f}",
            "wells used": ", ".join(self.wells_used) or "none",
            "AI min": f"{finite.min():.0f}" if finite.size else "n/a",
            "AI mean": f"{finite.mean():.0f}" if finite.size else "n/a",
            "AI max": f"{finite.max():.0f}" if finite.size else "n/a",
        }


# --------------------------------------------------------------------------
# Per-well low-frequency AI logs
# --------------------------------------------------------------------------

def well_low_frequency_log(
    well,
    twt_axis: np.ndarray,
    cutoff_hz: float = 10.0,
    dt: float | None = None,
    extend: bool = True,
) -> np.ndarray:
    """Low-pass filtered well AI on the seismic time axis.

    Filtering is done on ``log(AI)``: impedance is a positive, roughly
    log-normal quantity, so a low-pass in the log domain cannot produce a
    negative background and preserves relative contrasts.

    Where the log does not reach (above the top or below the base of the
    logged interval) the trend is extended by holding the end value, unless
    ``extend`` is False, in which case those samples stay NaN.
    """
    twt_axis = np.asarray(twt_axis, dtype=float)
    dt = dt if dt is not None else float(np.median(np.diff(twt_axis))) / 1000.0

    ai = well.ai_on_time_axis(twt_axis)
    good = np.isfinite(ai) & (ai > 0)
    out = np.full(twt_axis.shape, np.nan)
    if good.sum() < 8:
        return out

    log_ai = np.log(utils.fill_nan_1d(np.where(good, ai, np.nan)))
    smooth = utils.lowpass(log_ai, dt, cutoff_hz)

    idx = np.flatnonzero(good)
    i0, i1 = idx[0], idx[-1]
    out[i0:i1 + 1] = np.exp(smooth[i0:i1 + 1])

    if extend:
        out[:i0] = out[i0]
        out[i1 + 1:] = out[i1]
    return out


# --------------------------------------------------------------------------
# Lateral interpolation
# --------------------------------------------------------------------------

def _interpolate_layer(
    values: np.ndarray,
    well_xy: np.ndarray,
    grid_xy: np.ndarray,
    method: str = "idw",
    power: float = 2.0,
    smoothing: float = 0.0,
) -> np.ndarray:
    """Interpolate one time slice of well values onto the full trace grid."""
    good = np.isfinite(values)
    if good.sum() == 0:
        return np.full(grid_xy.shape[0], np.nan)
    if good.sum() == 1:
        return np.full(grid_xy.shape[0], float(values[good][0]))

    pts = well_xy[good]
    vals = values[good]

    if method == "rbf" and pts.shape[0] >= 3:
        try:
            # Linear kernel degrades gracefully to a plane with few wells,
            # which is what you want when there are only three or four.
            rbf = RBFInterpolator(pts, vals, kernel="linear", smoothing=smoothing)
            return np.asarray(rbf(grid_xy), dtype=float)
        except Exception:  # noqa: BLE001 - singular geometry (collinear wells)
            pass

    d = np.linalg.norm(grid_xy[:, None, :] - pts[None, :, :], axis=-1)
    if method == "nearest":
        return vals[np.argmin(d, axis=1)]

    eps = 1e-6
    exact = d < eps
    w = 1.0 / np.power(np.maximum(d, eps), power)
    out = (w @ vals) / w.sum(axis=1)
    hit_rows, hit_cols = np.nonzero(exact)
    out[hit_rows] = vals[hit_cols]
    return out


def build_low_frequency_model(
    volume,
    wells: Sequence,
    cutoff_hz: float = 10.0,
    method: str = "idw",
    power: float = 2.0,
    smoothing: float = 0.0,
    lateral_smooth_bins: float = 2.0,
    vertical_smooth_ms: float = 0.0,
    horizons: dict[str, np.ndarray] | None = None,
    progress=None,
) -> LowFrequencyModel:
    """Build the background AI cube.

    Wells are low-pass filtered, then interpolated laterally one time sample at
    a time.  When ``horizons`` are supplied the interpolation is done in
    flattened (horizon-relative) time so the trend follows structure; without
    them it is a straight time-slice interpolation, which is only defensible
    where structure is gentle.

    ``horizons`` maps a name to a ``(n_iline, n_xline)`` array of TWT in ms.
    A single horizon drives a constant per-trace datum shift; two or more drive
    proportional (layer-cake) flattening, where each interval is stretched onto
    the interval between the horizons' mean times, so the trend follows
    thickness variation and not just structure.  Horizons are sorted into
    stratigraphic order by mean time, so the caller's dict order does not
    matter.
    """
    twt = volume.twt
    dt = volume.dt
    n_il, n_xl, n_t = volume.data.shape

    located = [w for w in wells if getattr(w, "has_location", False)]
    if not located:
        raise ValueError(
            "the low-frequency model needs at least one well with an X/Y location -- "
            "upload a well-header CSV or use the synthetic dataset"
        )

    logs, names, xy = [], [], []
    for w in located:
        lf = well_low_frequency_log(w, twt, cutoff_hz=cutoff_hz, dt=dt)
        if np.isfinite(lf).sum() < 8:
            continue
        logs.append(lf)
        names.append(w.name)
        xy.append((w.x, w.y))
    if not logs:
        raise ValueError("no well produced a usable low-frequency AI log (check DT/RHOB and the time axis)")

    well_logs = np.vstack(logs)                       # (n_wells, n_t)
    well_xy = np.asarray(xy, dtype=float)
    grid_xy = volume.trace_xy()
    notes: list[str] = []

    flatten = None
    if horizons:
        flatten, well_map, note = _structural_knots(horizons, volume, well_xy)
        notes.append(note)
        if well_map is not None and well_logs.shape[0]:
            well_logs = _flatten_logs(well_logs, well_map[0], well_map[1], twt)

    out = np.empty((grid_xy.shape[0], n_t), dtype=np.float32)
    for k in range(n_t):
        out[:, k] = _interpolate_layer(well_logs[:, k], well_xy, grid_xy, method, power, smoothing)
        if progress is not None and (k % max(n_t // 50, 1) == 0):
            progress((k + 1) / n_t)
    if progress is not None:
        progress(1.0)

    cube = out.reshape(n_il, n_xl, n_t)

    if flatten is not None:
        cube = _unflatten_cube(cube, flatten[0], flatten[1], twt)
        notes.append("interpolated in horizon-flattened time, then restored to structure")

    if lateral_smooth_bins and lateral_smooth_bins > 0:
        cube = gaussian_filter(cube, sigma=(lateral_smooth_bins, lateral_smooth_bins, 0), mode="nearest")
        notes.append(f"lateral smoothing sigma = {lateral_smooth_bins:.1f} bins")

    if vertical_smooth_ms and vertical_smooth_ms > 0:
        sigma = vertical_smooth_ms / max(volume.sample_rate_ms, 1e-6)
        cube = gaussian_filter1d(cube, sigma=sigma, axis=2, mode="nearest")
        notes.append(f"vertical smoothing sigma = {vertical_smooth_ms:.0f} ms")

    cube = np.where(np.isfinite(cube) & (cube > 0), cube, np.nan).astype(np.float32)
    if np.isnan(cube).any():
        fill = float(np.nanmean(cube))
        cube = np.nan_to_num(cube, nan=fill)
        notes.append("residual gaps filled with the model mean")

    return LowFrequencyModel(
        ai=cube, twt=twt, cutoff_hz=cutoff_hz, method=method, wells_used=names, notes=notes,
    )


# --------------------------------------------------------------------------
# Structural guidance
# --------------------------------------------------------------------------

# Sentinel knots placed far outside the data so the piecewise-linear warp
# extrapolates with unit slope -- a constant shift -- above the shallowest
# horizon and below the deepest, rather than flattening out.
_FAR = 1.0e6


def _structural_knots(horizons: dict[str, np.ndarray], volume, well_xy: np.ndarray):
    """Per-trace horizon times and the flattened datum each maps to.

    With one horizon this is a constant shift per trace.  With two or more it
    is proportional (layer-cake) flattening: the interval between consecutive
    horizons is stretched onto the interval between their mean times, so the
    trend follows structure *and* thickness variation instead of only being
    hung off a single datum.
    """
    named = [(name, np.asarray(h, dtype=float)) for name, h in horizons.items()
             if np.asarray(h).shape == volume.data.shape[:2]]
    if not named:
        return None, None, "horizons ignored (grid shape did not match the seismic)"

    # Stratigraphic order, shallowest first -- the caller's dict order is not
    # required to be meaningful.
    named.sort(key=lambda nh: float(np.nanmean(nh[1])))
    names = [n for n, _ in named]
    grids = []
    for _, g in named:
        mean = float(np.nanmean(g))
        grids.append(np.nan_to_num(g, nan=mean if np.isfinite(mean) else 0.0))

    struct = np.stack(grids, axis=-1)                       # (n_il, n_xl, m)
    flat = np.array([float(np.mean(g)) for g in grids])     # common datums

    # Crossing picks would fold the warp.  Force a minimum separation instead
    # of trusting the interpretation to be clean.
    min_gap = max(volume.sample_rate_ms, 1e-3)
    for k in range(1, struct.shape[-1]):
        struct[..., k] = np.maximum(struct[..., k], struct[..., k - 1] + min_gap)
    flat = np.maximum.accumulate(flat)
    for k in range(1, flat.size):
        flat[k] = max(flat[k], flat[k - 1] + min_gap)

    grid_xy = volume.trace_xy()
    flat_struct = struct.reshape(-1, struct.shape[-1])
    well_knots = np.array([
        flat_struct[int(np.argmin(np.linalg.norm(grid_xy - p, axis=1)))] for p in well_xy
    ]) if len(well_xy) else np.empty((0, struct.shape[-1]))

    if len(grids) == 1:
        note = f"flattened on '{names[0]}'"
    else:
        note = f"proportionally flattened between {len(grids)} horizons: " + ", ".join(names)
    return (struct, flat), (well_knots, flat), note


def _warp_axis(t_axis: np.ndarray, src_knots: np.ndarray, dst_knots: np.ndarray) -> np.ndarray:
    """Times in the source domain that correspond to each destination sample."""
    src = np.concatenate(([src_knots[0] - _FAR], src_knots, [src_knots[-1] + _FAR]))
    dst = np.concatenate(([dst_knots[0] - _FAR], dst_knots, [dst_knots[-1] + _FAR]))
    return np.interp(t_axis, dst, src)


def _flatten_logs(logs: np.ndarray, well_knots: np.ndarray, flat_knots: np.ndarray,
                  twt: np.ndarray) -> np.ndarray:
    """Move each well log from structural time into flattened time."""
    out = np.empty_like(logs)
    for i in range(logs.shape[0]):
        src_t = _warp_axis(twt, well_knots[i], flat_knots)
        out[i] = np.interp(src_t, twt, logs[i], left=logs[i][0], right=logs[i][-1])
    return out


def _unflatten_cube(cube: np.ndarray, struct: np.ndarray, flat_knots: np.ndarray,
                    twt: np.ndarray) -> np.ndarray:
    """Restore a flattened cube to structural time, trace by trace."""
    n_il, n_xl, _ = cube.shape
    out = np.empty_like(cube)
    for i in range(n_il):
        for j in range(n_xl):
            dst_t = _warp_axis(twt, flat_knots, struct[i, j])
            trace = cube[i, j]
            out[i, j] = np.interp(dst_t, twt, trace, left=trace[0], right=trace[-1])
    return out


def load_horizon_csv(path_or_buffer, volume) -> dict[str, np.ndarray]:
    """Read horizons from a CSV with ``iline, xline, <horizon columns...>``.

    Every numeric column beyond iline/xline is treated as a horizon in TWT (ms)
    and gridded onto the seismic geometry, nearest-neighbour filling any bins
    the horizon file does not cover.
    """
    import pandas as pd

    df = pd.read_csv(path_or_buffer)
    lookup = {str(c).strip().lower(): c for c in df.columns}
    il_col = next((lookup[a] for a in ("iline", "il", "inline", "inline_3d") if a in lookup), None)
    xl_col = next((lookup[a] for a in ("xline", "xl", "crossline", "crossline_3d") if a in lookup), None)
    if il_col is None or xl_col is None:
        raise ValueError("horizon CSV needs 'iline' and 'xline' columns")

    il_index = {int(v): i for i, v in enumerate(volume.iline)}
    xl_index = {int(v): i for i, v in enumerate(volume.xline)}
    n_il, n_xl = volume.data.shape[:2]

    ii = df[il_col].astype("int64").map(il_index)
    jj = df[xl_col].astype("int64").map(xl_index)
    inside = ii.notna() & jj.notna()
    ii = ii[inside].astype(int).to_numpy()
    jj = jj[inside].astype(int).to_numpy()

    out: dict[str, np.ndarray] = {}
    for col in df.columns:
        if col in (il_col, xl_col):
            continue
        vals = pd.to_numeric(df[col], errors="coerce")[inside].to_numpy(dtype=float)
        if not np.isfinite(vals).any():
            continue
        grid = np.full((n_il, n_xl), np.nan)
        grid[ii, jj] = vals
        out[str(col)] = _fill_horizon_gaps(grid)
    if not out:
        raise ValueError("horizon CSV contained no usable horizon columns")
    return out


def _fill_horizon_gaps(grid: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill so a partially picked horizon still covers the cube."""
    g = np.array(grid, dtype=float)
    good = np.isfinite(g)
    if good.all() or not good.any():
        return np.nan_to_num(g, nan=float(np.nanmean(g)) if good.any() else 0.0)
    from scipy.ndimage import distance_transform_edt

    _, idx = distance_transform_edt(~good, return_indices=True)
    return g[tuple(idx)]
