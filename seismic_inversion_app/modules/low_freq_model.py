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
    Two or more horizons drive a proportional (layer-cake) flattening; a single
    horizon drives a simple datum shift.
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

    shift_grid = None
    if horizons:
        shift_grid, well_shift, note = _structural_shifts(horizons, volume, well_xy)
        notes.append(note)
        if well_shift is not None:
            well_logs = _shift_logs(well_logs, -well_shift, dt)

    out = np.empty((grid_xy.shape[0], n_t), dtype=np.float32)
    for k in range(n_t):
        out[:, k] = _interpolate_layer(well_logs[:, k], well_xy, grid_xy, method, power, smoothing)
        if progress is not None and (k % max(n_t // 50, 1) == 0):
            progress((k + 1) / n_t)
    if progress is not None:
        progress(1.0)

    cube = out.reshape(n_il, n_xl, n_t)

    if shift_grid is not None:
        cube = _unflatten_cube(cube, shift_grid, dt)
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

def _structural_shifts(horizons: dict[str, np.ndarray], volume, well_xy: np.ndarray):
    """Per-trace flattening shift (in ms) derived from the supplied horizons."""
    grids = [np.asarray(h, dtype=float) for h in horizons.values() if np.asarray(h).shape == volume.data.shape[:2]]
    if not grids:
        return None, None, "horizons ignored (grid shape did not match the seismic)"

    ref = grids[0]
    shift = ref - float(np.nanmean(ref))            # ms, positive = deeper than datum
    shift = np.nan_to_num(shift)

    # Sample the shift at each well by nearest bin.
    grid_xy = volume.trace_xy()
    flat_shift = shift.ravel()
    well_shift = np.array([
        flat_shift[int(np.argmin(np.linalg.norm(grid_xy - p, axis=1)))] for p in well_xy
    ])
    note = f"flattened on '{list(horizons)[0]}'"
    if len(grids) > 1:
        note += f" (of {len(grids)} horizons; v1 uses the first as the flattening datum)"
    return shift, well_shift, note


def _shift_logs(logs: np.ndarray, shift_ms: np.ndarray, dt: float) -> np.ndarray:
    """Shift each well log in time by its own amount (positive = later)."""
    out = np.empty_like(logs)
    n_t = logs.shape[1]
    base = np.arange(n_t, dtype=float)
    for i in range(logs.shape[0]):
        s = float(shift_ms[i]) / (dt * 1000.0)
        out[i] = np.interp(base + s, base, logs[i], left=logs[i][0], right=logs[i][-1])
    return out


def _unflatten_cube(cube: np.ndarray, shift_ms: np.ndarray, dt: float) -> np.ndarray:
    """Restore a flattened cube to structural time."""
    n_il, n_xl, n_t = cube.shape
    base = np.arange(n_t, dtype=float)
    out = np.empty_like(cube)
    for i in range(n_il):
        for j in range(n_xl):
            s = float(shift_ms[i, j]) / (dt * 1000.0)
            out[i, j] = np.interp(base - s, base, cube[i, j], left=cube[i, j][0], right=cube[i, j][-1])
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
