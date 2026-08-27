"""SEG-Y and LAS loading, the in-memory data model, and the synthetic fallback.

The two containers defined here (:class:`SeismicVolume` and :class:`WellData`)
are what every other module consumes, so the rest of the app never has to care
whether the data came from a real SEG-Y, a LAS file, or the synthetic generator.
"""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from scipy.spatial import cKDTree

from . import utils

# Streamlit is optional at import time so the numeric core can be used from a
# notebook or a test runner.  ``_cache`` degrades to a no-op decorator.
try:  # pragma: no cover - trivial import guard
    import streamlit as st

    _HAS_ST = True
except Exception:  # pragma: no cover
    st = None
    _HAS_ST = False


def _cache_data(func):
    if _HAS_ST:
        return st.cache_data(show_spinner=False, max_entries=4)(func)
    return func


DEFAULT_BYTES = {
    "iline": 189,
    "xline": 193,
    "cdp_x": 181,
    "cdp_y": 185,
}


# ==========================================================================
# Containers
# ==========================================================================

# Sentinel used in :class:`CurveSelection.time` to mean "use the well's
# time-depth / checkshot file" rather than a curve from the LAS.
TD_SOURCE = "(time-depth file)"

# Plausible interval velocity band, used to decide whether a time-depth column
# is one-way or two-way when the file does not say.
TD_VELOCITY_RANGE = (1400.0, 6500.0)


@dataclass
class Marker:
    """A formation top / horizon pick at a measured depth."""

    md: float
    name: str
    twt: float | None = None            # filled in once a time-depth exists


@dataclass
class TimeDepth:
    """A checkshot or time-depth table: measured depth against two-way time."""

    md: np.ndarray                       # metres
    twt: np.ndarray                      # milliseconds, two-way
    source: str = ""
    was_one_way: bool = False
    was_seconds: bool = True

    def __post_init__(self) -> None:
        self.md = np.asarray(self.md, dtype=float)
        self.twt = np.asarray(self.twt, dtype=float)
        order = np.argsort(self.md)
        self.md, self.twt = self.md[order], self.twt[order]

    @property
    def datum_md(self) -> float:
        """MD at which time is zero -- the seismic reference datum."""
        if self.twt.size and np.nanmin(np.abs(self.twt)) < 1e-9:
            return float(self.md[int(np.argmin(np.abs(self.twt)))])
        return float(self.md[0])

    def to_twt(self, md: np.ndarray) -> np.ndarray:
        """Interpolate MD to TWT (ms).

        Extrapolation beyond the table is linear in the end interval's
        velocity rather than flat, so a log that runs slightly past the deepest
        checkshot still gets a sensible -- and monotonic -- time.
        """
        md = np.asarray(md, dtype=float)
        if self.md.size < 2:
            return np.full(md.shape, np.nan)
        out = np.interp(md, self.md, self.twt)

        above = md < self.md[0]
        if above.any():
            grad = (self.twt[1] - self.twt[0]) / max(self.md[1] - self.md[0], 1e-9)
            out[above] = self.twt[0] + (md[above] - self.md[0]) * grad
        below = md > self.md[-1]
        if below.any():
            grad = (self.twt[-1] - self.twt[-2]) / max(self.md[-1] - self.md[-2], 1e-9)
            out[below] = self.twt[-1] + (md[below] - self.md[-1]) * grad
        return out

    def interval_velocity(self) -> np.ndarray:
        """Interval velocity (m/s) implied by consecutive table entries."""
        dz = np.diff(self.md)
        dt = np.diff(self.twt) / 1000.0
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(dt > 0, 2.0 * dz / dt, np.nan)

    def warnings(self) -> list[str]:
        """Flag a time-depth table that looks like a unit or convention mistake."""
        out: list[str] = []
        vi = self.interval_velocity()
        vi = vi[np.isfinite(vi)]
        if vi.size == 0:
            return ["No interval velocity could be computed - check for duplicate depths."]

        median = float(np.median(vi))

        # Judge the shallow section rather than the whole table: compaction
        # means near-surface rock is slow, so 3,800 m/s in the first second is
        # a strong hint that one-way time has been read as two-way, while the
        # same number at 3 s is unremarkable.
        #
        # The window is in *time*, not depth: a table that starts below the
        # datum has no shallow section to judge, and using depth would need a
        # datum this class cannot always know.
        mid_time = 0.5 * (self.twt[1:] + self.twt[:-1])
        vi_all = self.interval_velocity()
        shallow = np.isfinite(vi_all) & (mid_time < 1500.0)
        if shallow.sum() >= 2:
            shallow_median = float(np.median(vi_all[shallow]))
            if shallow_median > 3000:
                out.append(
                    f"Interval velocity in the first 1.5 s averages {shallow_median:,.0f} m/s, "
                    "which is fast for shallow section. If this table is one-way time being read "
                    "as two-way, every tie will be out by a factor of two - set the time unit "
                    "explicitly instead of leaving it on 'auto'.")

        if median < 1450:
            out.append(
                f"Median interval velocity is {median:,.0f} m/s, below the speed of sound in "
                "water. Check the depth units and whether the time column is two-way.")
        if vi.max() > 7000:
            out.append(f"An interval reaches {vi.max():,.0f} m/s - check for a mis-keyed depth.")
        if np.any(np.diff(self.twt) < 0):
            out.append("Time does not increase monotonically with depth.")
        return out

    def summary(self) -> dict:
        vi = self.interval_velocity()
        vi = vi[np.isfinite(vi)]
        return {
            "points": int(self.md.size),
            "MD range (m)": f"{self.md.min():.1f} - {self.md.max():.1f}",
            "TWT range (ms)": f"{self.twt.min():.0f} - {self.twt.max():.0f}",
            "datum MD (m)": f"{self.datum_md:.1f}",
            "interval velocity (m/s)": (f"{vi.min():.0f} - {vi.max():.0f}" if vi.size else "n/a"),
            "read as": ("one-way, doubled" if self.was_one_way else "two-way")
                       + (", seconds" if self.was_seconds else ", milliseconds"),
        }


@dataclass
class WellTrack:
    """A deviation survey: map position and TVDSS against measured depth."""

    md: np.ndarray
    x: np.ndarray
    y: np.ndarray
    tvdss: np.ndarray                    # positive down, relative to the datum
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("md", "x", "y", "tvdss"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float))
        order = np.argsort(self.md)
        self.md, self.x, self.y, self.tvdss = (
            self.md[order], self.x[order], self.y[order], self.tvdss[order])

    @property
    def surface_xy(self) -> tuple[float, float]:
        return float(self.x[0]), float(self.y[0])

    @property
    def kb(self) -> float:
        """Height of MD zero above the datum (metres).

        With TVDSS positive down, the shallowest station sits at a negative
        TVDSS when the reference is above the datum -- which is the usual case
        for a KB.
        """
        return float(-self.tvdss[0] + self.md[0])

    @property
    def is_vertical(self) -> bool:
        return bool(np.allclose(self.x, self.x[0], atol=1.0)
                    and np.allclose(self.y, self.y[0], atol=1.0))

    @property
    def max_deviation(self) -> float:
        """Greatest horizontal step-out from the surface location (metres)."""
        x0, y0 = self.surface_xy
        return float(np.max(np.hypot(self.x - x0, self.y - y0)))

    def xy_at(self, md: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        md = np.asarray(md, dtype=float)
        return (np.interp(md, self.md, self.x), np.interp(md, self.md, self.y))

    def tvdss_at(self, md: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(md, dtype=float), self.md, self.tvdss)

    def summary(self) -> dict:
        return {
            "stations": int(self.md.size),
            "MD range (m)": f"{self.md.min():.1f} - {self.md.max():.1f}",
            "surface X": f"{self.surface_xy[0]:.1f}",
            "surface Y": f"{self.surface_xy[1]:.1f}",
            "KB above datum (m)": f"{self.kb:.1f}",
            "geometry": "vertical" if self.is_vertical else f"deviated, {self.max_deviation:.0f} m step-out",
        }


@dataclass
class CurveSelection:
    """Which LAS curve fills each role, and the unit it is in.

    Auto-detection fills this in on load; the log-QC step lets the user
    override every field.  Keeping it as data (rather than baking the choice
    into ``vp``/``rho`` at parse time) is what makes the reassignment possible
    without re-reading the file.
    """

    sonic: str | None = None
    sonic_unit: str = "us/ft"
    density: str | None = None
    density_unit: str = "g/cm3"
    time: str | None = None
    time_unit: str = "ms (TWT)"
    constant_vp: float | None = None

    def describe(self) -> str:
        bits = []
        bits.append(f"Vp: {self.sonic} [{self.sonic_unit}]" if self.sonic
                    else (f"Vp: constant {self.constant_vp:.0f} m/s" if self.constant_vp else "Vp: none"))
        bits.append(f"Rho: {self.density} [{self.density_unit}]" if self.density else "Rho: none")
        bits.append(f"TWT: {self.time} [{self.time_unit}]" if self.time else "TWT: integrated from sonic")
        return " | ".join(bits)


def velocity_from_curve(values: np.ndarray, unit: str) -> np.ndarray:
    """Convert a sonic *or* velocity curve to Vp in m/s, per the chosen unit."""
    values = np.asarray(values, dtype=float)
    unit = unit.lower().replace(" ", "")
    if unit in ("m/s", "ms-1"):
        out = values.copy()
    elif unit in ("ft/s", "fts-1"):
        out = values / utils.FT_PER_M
    else:
        return utils.sonic_to_velocity(values, unit=unit)
    out[~np.isfinite(out) | (out <= 0)] = np.nan
    return out


DEPTH_UNITS = ("m", "ft")


def depth_unit_scale(unit: str) -> tuple[float, str]:
    """Metres per unit of a LAS depth index, and the unit that was resolved.

    ``WellData.md`` is metres everywhere downstream -- ``integrate_sonic_to_twt``
    divides it by a velocity in m/s -- so a LAS indexed in feet has to be
    converted on the way in.  Left unconverted it does not fail loudly; it just
    puts the well 3.28x too deep, which is why this is decided from the header
    rather than guessed from the numbers.  Depth magnitude cannot disambiguate
    the two (a 4,000 m well and a 4,000 ft well are both ordinary), so an
    unlabelled index is taken as metres and flagged in QC.
    """
    u = str(unit or "").strip().lower().rstrip(".")
    if u in ("ft", "f", "feet", "foot", "ftus", "usft", "ft(us)"):
        return 1.0 / utils.FT_PER_M, "ft"
    if u in ("m", "metre", "meter", "metres", "meters", "md"):
        return 1.0, "m"
    return 1.0, ""


def _depth_index_unit(las) -> str:
    """The unit of the LAS depth index: the first curve, else ``STRT``."""
    try:
        if len(las.curves):
            unit = str(getattr(las.curves[0], "unit", "") or "").strip()
            if unit:
                return unit
    except Exception:  # noqa: BLE001 - malformed curve section
        pass
    for item in getattr(las, "well", []):
        if item.mnemonic.upper() == "STRT":
            return str(getattr(item, "unit", "") or "").strip()
    return ""


def time_curve_to_twt_ms(values: np.ndarray, unit: str) -> np.ndarray:
    """Normalise a time curve to two-way time in milliseconds."""
    values = np.asarray(values, dtype=float).copy()
    unit = unit.lower()
    if "s (" in unit and "ms" not in unit:
        values = values * 1000.0
    if "owt" in unit:
        values = values * 2.0
    return values



@dataclass
class SeismicVolume:
    """A 3D post-stack volume held as ``(n_iline, n_xline, n_samples)``."""

    data: np.ndarray
    iline: np.ndarray
    xline: np.ndarray
    twt: np.ndarray                      # milliseconds
    cdp_x: np.ndarray                    # (n_iline, n_xline)
    cdp_y: np.ndarray                    # (n_iline, n_xline)
    source: str = "unknown"
    text_header: str = ""
    # SEG-Y coordinate scalar: negative divides, positive multiplies. Kept so a
    # round-trip through the writer reproduces the survey's own convention.
    coord_scalar: int = -100

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        self.iline = np.asarray(self.iline)
        self.xline = np.asarray(self.xline)
        self.twt = np.asarray(self.twt, dtype=float)
        self.cdp_x = np.asarray(self.cdp_x, dtype=float)
        self.cdp_y = np.asarray(self.cdp_y, dtype=float)

    # -- geometry ---------------------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    @property
    def n_traces(self) -> int:
        return int(self.data.shape[0] * self.data.shape[1])

    @property
    def sample_rate_ms(self) -> float:
        if self.twt.size < 2:
            return 1.0
        return float(np.median(np.diff(self.twt)))

    @property
    def dt(self) -> float:
        """Sample interval in **seconds** (what every filter here expects)."""
        return self.sample_rate_ms / 1000.0

    def trace_xy(self) -> np.ndarray:
        """Flat ``(n_traces, 2)`` map coordinates, C-ordered over (il, xl)."""
        return np.column_stack([self.cdp_x.ravel(), self.cdp_y.ravel()])

    def flat_data(self) -> np.ndarray:
        """``(n_traces, n_samples)`` view of the cube."""
        n_il, n_xl, n_t = self.data.shape
        return self.data.reshape(n_il * n_xl, n_t)

    def live_mask(self) -> np.ndarray:
        """Flat boolean mask of traces carrying non-zero, finite amplitudes."""
        flat = self.flat_data()
        return np.isfinite(flat).any(axis=1) & (np.nanmax(np.abs(flat), axis=1) > 0)

    def flat_to_ij(self, flat_index: int) -> tuple[int, int]:
        return int(flat_index // self.data.shape[1]), int(flat_index % self.data.shape[1])

    def trace_at(self, i_il: int, i_xl: int) -> np.ndarray:
        return np.asarray(self.data[i_il, i_xl, :], dtype=float)

    # -- interop ----------------------------------------------------------
    def to_xarray(self, name: str = "data"):
        """Wrap the cube as an xarray Dataset with segysak-style coordinates."""
        import xarray as xr

        ds = xr.Dataset(
            {name: (("iline", "xline", "twt"), self.data)},
            coords={
                "iline": self.iline,
                "xline": self.xline,
                "twt": self.twt,
                "cdp_x": (("iline", "xline"), self.cdp_x),
                "cdp_y": (("iline", "xline"), self.cdp_y),
            },
        )
        ds.attrs["sample_rate"] = self.sample_rate_ms
        ds.attrs["source_file"] = self.source
        ds.attrs["measurement_system"] = "m"
        ds.attrs["ns"] = int(self.data.shape[2])
        ds.attrs["srd"] = 0
        # segysak's writer requires this attribute and raises without it.
        ds.attrs["coord_scalar"] = int(self.coord_scalar)
        ds.attrs["text"] = self.text_header
        return ds

    def with_data(self, new_data: np.ndarray, source: str | None = None) -> "SeismicVolume":
        """Clone the geometry, swap the samples (used for inversion results)."""
        return SeismicVolume(
            data=new_data,
            iline=self.iline,
            xline=self.xline,
            twt=self.twt,
            cdp_x=self.cdp_x,
            cdp_y=self.cdp_y,
            source=source or f"{self.source} (derived)",
            coord_scalar=self.coord_scalar,
        )

    def summary(self) -> dict[str, Any]:
        finite = self.data[np.isfinite(self.data)]
        return {
            "source": os.path.basename(self.source),
            "inlines": f"{self.iline.min()}-{self.iline.max()} ({self.data.shape[0]})",
            "crosslines": f"{self.xline.min()}-{self.xline.max()} ({self.data.shape[1]})",
            "twt range (ms)": f"{self.twt.min():.0f}-{self.twt.max():.0f}",
            "sample rate (ms)": f"{self.sample_rate_ms:.2f}",
            "samples/trace": int(self.data.shape[2]),
            "traces": self.n_traces,
            "amplitude p99": f"{np.percentile(np.abs(finite), 99):.4g}" if finite.size else "n/a",
        }


@dataclass
class WellData:
    """A single well, already tied: ``twt`` is the time axis of every curve."""

    name: str
    md: np.ndarray                       # measured depth (m)
    twt: np.ndarray                      # two-way time (ms)
    vp: np.ndarray                       # m/s
    rho: np.ndarray                      # kg/m^3
    x: float | None = None
    y: float | None = None
    kb: float = 0.0
    uwi: str = ""
    curves: dict[str, np.ndarray] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    bulk_shift: float = 0.0
    # Piecewise-linear stretch/squeeze on top of the bulk shift: knot times on
    # the unshifted axis, and the correction (ms) to add at each.
    warp_knots: np.ndarray | None = None
    warp_shifts: np.ndarray | None = None
    # Raw LAS metadata, kept so the log-QC step can show what is actually in
    # the file rather than only what auto-detection picked.
    curve_units: dict[str, str] = field(default_factory=dict)
    curve_descr: dict[str, str] = field(default_factory=dict)
    # Depth index unit as resolved on load ("m", "ft", or "" when unlabelled).
    # ``md`` is always metres; this records what it was converted from.
    depth_unit: str = ""
    selection: CurveSelection = field(default_factory=CurveSelection)
    # Auxiliary files: checkshot, deviation survey, formation tops.
    time_depth: TimeDepth | None = None
    track: WellTrack | None = None
    markers: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.md = np.asarray(self.md, dtype=float)
        self.twt = np.asarray(self.twt, dtype=float)
        self.vp = np.asarray(self.vp, dtype=float)
        self.rho = np.asarray(self.rho, dtype=float)
        # Snapshot of the as-loaded time axis, so repeated bulk shifts are
        # applied from the original rather than accumulating on each other.
        self._twt_base = self.twt.copy()

    def set_bulk_shift(self, milliseconds: float) -> None:
        """Apply a constant time shift to the whole well.

        A bulk shift fixes a datum error.  It cannot fix a drifting time-depth
        relationship -- that needs :meth:`set_time_warp`, which sits on top of
        this one rather than replacing it.
        """
        self.bulk_shift = float(milliseconds)
        self._apply_time_model()

    def set_time_warp(self, knots_ms, shifts_ms) -> None:
        """Piecewise-linear stretch and squeeze, on top of the bulk shift.

        ``knots_ms`` are times on the *unshifted* well axis and ``shifts_ms``
        the correction to add at each one; between and beyond them the
        correction is linear and constant respectively.  A single knot is
        therefore just another bulk shift, and ``None`` clears the warp.

        The result must stay monotonic -- time cannot run backwards -- so a
        warp that would fold the log is rejected rather than quietly applied.
        """
        if knots_ms is None or np.size(knots_ms) == 0:
            self.warp_knots = None
            self.warp_shifts = None
            self._apply_time_model()
            return

        knots = np.asarray(knots_ms, dtype=float).ravel()
        shifts = np.asarray(shifts_ms, dtype=float).ravel()
        if knots.size != shifts.size:
            raise ValueError("warp knots and shifts must be the same length")
        order = np.argsort(knots)
        knots, shifts = knots[order], shifts[order]

        candidate = self._warp_times(self._twt_base, knots, shifts)
        finite = np.isfinite(candidate)
        if finite.sum() > 1 and np.any(np.diff(candidate[finite]) <= 0):
            raise ValueError(
                "this stretch would make the well's time axis non-monotonic; "
                "reduce the drift or use fewer knots")

        self.warp_knots = knots
        self.warp_shifts = shifts
        self._apply_time_model()

    def _warp_times(self, base: np.ndarray, knots=None, shifts=None) -> np.ndarray:
        """Map times on the unshifted axis through bulk shift, then warp."""
        base = np.asarray(base, dtype=float)
        knots = self.warp_knots if knots is None else knots
        shifts = self.warp_shifts if shifts is None else shifts
        out = base + float(self.bulk_shift)
        if knots is not None and np.size(knots):
            out = out + np.interp(base, knots, shifts)
        return out

    def _apply_time_model(self) -> None:
        self.twt = self._warp_times(self._twt_base)
        # Markers are picked in depth but displayed in time, so they move with
        # the well or they would point at the wrong reflector.
        self.refresh_marker_times()

    @property
    def has_warp(self) -> bool:
        return self.warp_knots is not None and np.size(self.warp_knots) > 1

    def drift_range_ms(self) -> tuple[float, float]:
        """Smallest and largest warp correction, excluding the bulk shift."""
        if not self.has_warp:
            return (0.0, 0.0)
        return (float(np.min(self.warp_shifts)), float(np.max(self.warp_shifts)))

    # -- curve assignment ------------------------------------------------
    def apply_selection(
        self,
        selection: CurveSelection,
        replacement_velocity: float = 2500.0,
        seismic_datum: float = 0.0,
    ) -> list[str]:
        """Recompute Vp, Rho and TWT from a (possibly edited) curve selection.

        Returns human-readable notes.  The bulk shift is re-applied afterwards
        so reassigning the time curve does not silently drop a shift the user
        set on the tie step.
        """
        notes: list[str] = []
        self.selection = selection

        if selection.sonic and selection.sonic in self.curves:
            self.vp = velocity_from_curve(self.curves[selection.sonic], selection.sonic_unit)
            notes.append(f"Vp from '{selection.sonic}' [{selection.sonic_unit}]")
        elif selection.constant_vp:
            self.vp = np.full(self.md.shape, float(selection.constant_vp))
            notes.append(f"Vp constant at {selection.constant_vp:.0f} m/s")
        else:
            self.vp = np.full(self.md.shape, np.nan)
            notes.append("No Vp source assigned")

        if selection.density and selection.density in self.curves:
            self.rho = utils.density_to_si(self.curves[selection.density], selection.density_unit)
            notes.append(f"Rho from '{selection.density}' [{selection.density_unit}]")
        else:
            self.rho = np.full(self.md.shape, np.nan)
            notes.append("No density curve assigned")

        if selection.time == TD_SOURCE and self.time_depth is not None:
            base = self.time_depth.to_twt(self.md)
            notes.append(f"TWT from the time-depth file ({self.time_depth.md.size} points)")
        elif selection.time and selection.time in self.curves:
            base = time_curve_to_twt_ms(self.curves[selection.time], selection.time_unit)
            notes.append(f"TWT from '{selection.time}' [{selection.time_unit}] (well assumed tied)")
        else:
            base = integrate_sonic_to_twt(
                self.md, self.vp, kb=self.kb,
                replacement_velocity=replacement_velocity, seismic_datum=seismic_datum)
            notes.append("TWT integrated from Vp (no time curve assigned)")

        self._twt_base = np.asarray(base, dtype=float)
        # Re-applied rather than dropped: reassigning the time curve must not
        # silently discard a shift or stretch the user set on the tie step.
        self._apply_time_model()
        self.notes = notes
        return notes

    # -- derived -----------------------------------------------------------
    @property
    def ai(self) -> np.ndarray:
        return utils.acoustic_impedance(self.vp, self.rho)

    @property
    def has_location(self) -> bool:
        return self.x is not None and self.y is not None and np.isfinite(self.x) and np.isfinite(self.y)

    def valid_mask(self) -> np.ndarray:
        return np.isfinite(self.twt) & np.isfinite(self.vp) & np.isfinite(self.rho) & (self.vp > 0) & (self.rho > 0)

    def time_range(self) -> tuple[float, float]:
        good = self.valid_mask()
        if not good.any():
            return (np.nan, np.nan)
        return float(self.twt[good].min()), float(self.twt[good].max())

    def ai_on_time_axis(self, twt_axis: np.ndarray) -> np.ndarray:
        """Resample AI onto a (regular) seismic time axis, anti-aliased."""
        good = self.valid_mask()
        if good.sum() < 2:
            return np.full(np.shape(twt_axis), np.nan)
        return utils.resample_to_time(self.ai[good], self.twt[good], np.asarray(twt_axis, dtype=float))

    def reflectivity_on_time_axis(self, twt_axis: np.ndarray) -> np.ndarray:
        """Normal-incidence reflectivity at seismic sampling.

        The AI series is resampled to the seismic rate *before* differencing,
        which is the correct order: differencing the log-rate series and then
        decimating would keep energy the seismic can never see.
        """
        ai = self.ai_on_time_axis(twt_axis)
        good = np.isfinite(ai)
        r = np.zeros_like(ai)
        if good.sum() < 2:
            return r
        filled = utils.fill_nan_1d(ai)
        r_full = utils.reflectivity_from_ai(filled)
        r[good] = r_full[good]
        return r

    # -- auxiliary files ---------------------------------------------------
    def attach_time_depth(self, td: TimeDepth, prefer: bool = True) -> str:
        """Attach a checkshot and, by default, adopt it as the time source.

        A measured checkshot beats both a LAS time curve and sonic integration,
        so it is adopted unless the caller says otherwise.
        """
        self.time_depth = td
        self.refresh_marker_times()
        if prefer:
            selection = CurveSelection(**{**self.selection.__dict__})
            selection.time = TD_SOURCE
            self.apply_selection(selection)
            return (f"time-depth attached ({td.md.size} points, "
                    f"{td.twt.min():.0f}-{td.twt.max():.0f} ms) and adopted as the time source")
        return f"time-depth attached ({td.md.size} points), not adopted"

    def attach_track(self, track: WellTrack) -> str:
        """Attach a deviation survey and take the surface location and KB from it."""
        self.track = track
        self.x, self.y = track.surface_xy
        self.kb = track.kb
        geometry = "vertical" if track.is_vertical else f"deviated ({track.max_deviation:.0f} m step-out)"
        return (f"located at ({self.x:.1f}, {self.y:.1f}) from the deviation survey, "
                f"KB {self.kb:.1f} m, {geometry}")

    def attach_markers(self, markers: Sequence) -> str:
        self.markers = list(markers)
        self.refresh_marker_times()
        return f"{len(self.markers)} markers attached"

    def refresh_marker_times(self) -> None:
        """Recompute each marker's TWT from the current time source.

        Always derived from the *unshifted* time axis plus the current bulk
        shift, so calling this repeatedly -- or after a shift changes -- lands
        on the same answer instead of accumulating offsets.
        """
        if not self.markers:
            return
        md = np.array([m.md for m in self.markers], dtype=float)
        if self.time_depth is not None:
            base = self.time_depth.to_twt(md)
        else:
            good = np.isfinite(self.md) & np.isfinite(self._twt_base)
            base = (np.interp(md, self.md[good], self._twt_base[good], left=np.nan, right=np.nan)
                    if good.sum() > 1 else np.full(md.shape, np.nan))
        for marker, t in zip(self.markers, self._warp_times(base)):
            marker.twt = float(t) if np.isfinite(t) else None

    def markers_in_range(self, t_min: float, t_max: float) -> list:
        return [m for m in self.markers
                if m.twt is not None and t_min <= m.twt <= t_max]

    def curve_inventory(self) -> list[dict[str, Any]]:
        """Every curve in the LAS with its unit, description and statistics.

        This is what makes the assignment step usable: you can see that
        ``DTCO`` runs 55-140 and ``RHOZ`` runs 1.9-2.7 before deciding which is
        which, instead of guessing from the mnemonic alone.
        """
        rows: list[dict[str, Any]] = []
        roles = {
            self.selection.sonic: "Vp (sonic)",
            self.selection.density: "Density",
            self.selection.time: "TWT",
        }
        for mnemonic, values in self.curves.items():
            v = np.asarray(values, dtype=float)
            good = np.isfinite(v)
            rows.append({
                "curve": mnemonic,
                "role": roles.get(mnemonic, ""),
                "LAS unit": self.curve_units.get(mnemonic, ""),
                "description": (self.curve_descr.get(mnemonic, "") or "")[:60],
                "valid %": round(100.0 * good.mean(), 1) if v.size else 0.0,
                "min": round(float(np.nanmin(v)), 3) if good.any() else None,
                "mean": round(float(np.nanmean(v)), 3) if good.any() else None,
                "max": round(float(np.nanmax(v)), 3) if good.any() else None,
            })
        return rows

    def qc_flags(self, seismic_twt: np.ndarray | None = None) -> list[tuple[str, str]]:
        """Sanity checks on the assigned curves.

        Returns ``(severity, message)`` pairs where severity is ``error``,
        ``warning`` or ``ok``.  The ranges are wide on purpose: the job here is
        to catch a unit mistake -- a factor of 3.28 or 1000 -- not to police
        unusual rocks.
        """
        flags: list[tuple[str, str]] = []
        good = self.valid_mask()

        if not good.any():
            flags.append(("error", "No depth sample has valid Vp, density and time together. "
                                   "Check the curve assignment and units."))
            return flags

        vp, rho, ai = self.vp[good], self.rho[good], self.ai[good]

        def _range_check(name, values, lo, hi, unit, hint):
            med = float(np.nanmedian(values))
            if not (lo <= med <= hi):
                flags.append(("error", f"{name} median {med:,.0f} {unit} is outside the plausible "
                                       f"range {lo:,.0f}-{hi:,.0f}. {hint}"))
                return
            out = float(np.mean((values < lo) | (values > hi)) * 100.0)
            if out > 5.0:
                flags.append(("warning", f"{out:.0f}% of {name} samples fall outside "
                                         f"{lo:,.0f}-{hi:,.0f} {unit}."))
            else:
                flags.append(("ok", f"{name} median {med:,.0f} {unit} looks plausible."))

        if self.depth_unit == "ft":
            flags.append(("ok", "Depth index was in feet; converted to metres on load."))
        elif not self.depth_unit:
            flags.append(("warning", "Depth index carries no unit; assumed metres. If the LAS is "
                                     "in feet the time-depth will be ~3.28x too deep."))

        _range_check("Vp", vp, *VP_RANGE, "m/s",
                     "A factor of ~3.28 means feet and metres are swapped; ~1000 means the unit is wrong.")
        _range_check("Density", rho, *RHO_RANGE, "kg/m3",
                     "A factor of 1000 means g/cm3 and kg/m3 are swapped.")
        _range_check("AI", ai, *AI_RANGE, "m/s*kg/m3", "Check the Vp and density units above.")

        valid_pct = 100.0 * good.mean()
        if valid_pct < 50:
            flags.append(("warning", f"Only {valid_pct:.0f}% of samples are usable "
                                     "(nulls in Vp, density or time)."))

        twt = self.twt[good]
        if np.any(np.diff(twt) < 0):
            flags.append(("error", "The time curve is not monotonically increasing with depth. "
                                   "Check the TWT assignment and its unit."))

        if seismic_twt is not None and len(seismic_twt):
            t0, t1 = float(twt.min()), float(twt.max())
            s0, s1 = float(np.min(seismic_twt)), float(np.max(seismic_twt))
            overlap = max(0.0, min(t1, s1) - max(t0, s0))
            if overlap <= 0:
                flags.append(("error", f"The logged interval ({t0:.0f}-{t1:.0f} ms) does not overlap "
                                       f"the seismic ({s0:.0f}-{s1:.0f} ms). Check the time curve unit "
                                       "and whether it is one-way or two-way."))
            elif overlap < 0.25 * (t1 - t0):
                flags.append(("warning", f"Only {overlap:.0f} ms of the logged interval overlaps the "
                                         "seismic; the wavelet gate will be short."))
            else:
                flags.append(("ok", f"{overlap:.0f} ms of log overlaps the seismic."))
        return flags

    def summary(self) -> dict[str, Any]:
        t0, t1 = self.time_range()
        return {
            "well": self.name,
            "uwi": self.uwi or "-",
            "x": f"{self.x:.1f}" if self.has_location else "-",
            "y": f"{self.y:.1f}" if self.has_location else "-",
            "kb": f"{self.kb:.1f}",
            "twt range (ms)": f"{t0:.0f}-{t1:.0f}" if np.isfinite(t0) else "-",
            "samples": int(self.valid_mask().sum()),
            "mean AI": f"{np.nanmean(self.ai):.0f}" if self.valid_mask().any() else "-",
        }


# ==========================================================================
# SEG-Y loading
# ==========================================================================

def file_digest(data: bytes | str, extra: str = "") -> str:
    """Stable cache key: content hash for bytes, path+mtime+size for a path."""
    h = hashlib.sha256()
    if isinstance(data, bytes):
        h.update(data[:1_000_000])
        h.update(str(len(data)).encode())
    else:
        stat = os.stat(data)
        h.update(str(data).encode())
        h.update(f"{stat.st_size}-{stat.st_mtime_ns}".encode())
    h.update(extra.encode())
    return h.hexdigest()


UPLOAD_CHUNK = 8 * 1024 * 1024


def persist_upload(uploaded, suffix: str = ".sgy", chunk: int = UPLOAD_CHUNK) -> str:
    """Spill an uploaded file to disk; segyio needs a real path.

    Copied in chunks rather than via ``getvalue()``.  At the 1 GB upload limit
    ``getvalue()`` would materialise a second full copy of the volume as a
    bytes object on top of the one Streamlit already holds, doubling peak
    memory for no reason.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        if hasattr(uploaded, "seek"):
            uploaded.seek(0)
        while True:
            block = uploaded.read(chunk)
            if not block:
                break
            tmp.write(block)
    finally:
        tmp.close()
        if hasattr(uploaded, "seek"):
            uploaded.seek(0)
    return tmp.name


def scan_segy_headers(path: str, max_traces: int = 1000) -> pd.DataFrame:
    """Header scan to help the user pick byte positions.

    Returns one row per candidate byte location with min/max/unique counts, so
    a non-standard survey can be diagnosed without leaving the app.
    """
    from segysak.segy import segy_header_scan

    scan = segy_header_scan(path, max_traces_scan=max_traces)
    df = pd.DataFrame(scan)
    if df.index.name is None:
        df.index.name = "field"
    return df.reset_index()


def load_segy(
    path: str,
    iline: int = DEFAULT_BYTES["iline"],
    xline: int = DEFAULT_BYTES["xline"],
    cdp_x: int = DEFAULT_BYTES["cdp_x"],
    cdp_y: int = DEFAULT_BYTES["cdp_y"],
    ix_crop: Sequence[int] | None = None,
    z_crop: Sequence[float] | None = None,
) -> SeismicVolume:
    """Load a 3D post-stack SEG-Y.

    segysak is tried first (it resolves the inline/crossline geometry and the
    coordinate scalar for us).  If that fails -- unusual headers, a file
    segysak refuses -- we fall back to raw segyio trace access and rebuild the
    grid by hand.
    """
    try:
        return _load_segy_segysak(path, iline, xline, cdp_x, cdp_y, ix_crop, z_crop)
    except Exception as exc:  # noqa: BLE001 - fall back on *any* segysak failure
        fallback = _load_segy_segyio(path, iline, xline, cdp_x, cdp_y)
        fallback.text_header = (
            f"[loaded via segyio fallback; segysak raised: {type(exc).__name__}: {exc}]\n"
            + fallback.text_header
        )
        return fallback


def _load_segy_segysak(path, iline, xline, cdp_x, cdp_y, ix_crop, z_crop) -> SeismicVolume:
    from segysak.segy import segy_loader

    kwargs: dict[str, Any] = dict(iline=int(iline), xline=int(xline))
    if cdp_x:
        kwargs["cdp_x"] = int(cdp_x)
    if cdp_y:
        kwargs["cdp_y"] = int(cdp_y)
    if ix_crop:
        kwargs["ix_crop"] = list(ix_crop)
    if z_crop:
        kwargs["z_crop"] = list(z_crop)

    ds = segy_loader(path, **kwargs)

    var = "data" if "data" in ds else list(ds.data_vars)[0]
    vert = "twt" if "twt" in ds.dims else ("depth" if "depth" in ds.dims else list(ds.dims)[-1])
    cube = ds[var].transpose("iline", "xline", vert).values

    n_il, n_xl = cube.shape[0], cube.shape[1]
    xs = _coord_grid(ds, "cdp_x", n_il, n_xl, ds["iline"].values, ds["xline"].values)
    ys = _coord_grid(ds, "cdp_y", n_il, n_xl, ds["iline"].values, ds["xline"].values, y=True)

    return SeismicVolume(
        data=cube,
        iline=ds["iline"].values,
        xline=ds["xline"].values,
        twt=ds[vert].values.astype(float),
        cdp_x=xs,
        cdp_y=ys,
        source=path,
        text_header=str(ds.attrs.get("text", ""))[:4000],
        coord_scalar=int(ds.attrs.get("coord_scalar") or -100),
    )


def _coord_grid(ds, name, n_il, n_xl, ilines, xlines, y: bool = False) -> np.ndarray:
    """Pull cdp_x/cdp_y off the dataset, synthesising a grid if they're absent."""
    if name in ds.coords or name in ds.data_vars:
        arr = np.asarray(ds[name].values, dtype=float)
        if arr.shape == (n_il, n_xl):
            if np.isfinite(arr).any() and np.nanstd(arr) > 0:
                return arr
        elif arr.size == n_il * n_xl:
            return arr.reshape(n_il, n_xl)
    # No usable coordinates: fall back to a 25 m bin grid on il/xl numbering so
    # the KD-tree lookup still has a consistent (if arbitrary) map frame.
    bin_size = 25.0
    if y:
        return np.tile((np.asarray(xlines, dtype=float) * bin_size)[None, :], (n_il, 1))
    return np.tile((np.asarray(ilines, dtype=float) * bin_size)[:, None], (1, n_xl))


def _load_segy_segyio(path, iline, xline, cdp_x, cdp_y) -> SeismicVolume:
    """Raw segyio path: read every trace header, rebuild the (il, xl) grid."""
    import segyio

    with segyio.open(path, "r", iline=int(iline), xline=int(xline), ignore_geometry=True) as f:
        n_traces = f.tracecount
        twt = np.asarray(f.samples, dtype=float)
        headers = f.header
        ils = np.empty(n_traces, dtype=np.int64)
        xls = np.empty(n_traces, dtype=np.int64)
        xs = np.empty(n_traces, dtype=float)
        ys = np.empty(n_traces, dtype=float)
        scalars = np.ones(n_traces, dtype=float)
        for i in range(n_traces):
            h = headers[i]
            ils[i] = h[int(iline)]
            xls[i] = h[int(xline)]
            xs[i] = h[int(cdp_x)] if cdp_x else 0.0
            ys[i] = h[int(cdp_y)] if cdp_y else 0.0
            scalars[i] = h[segyio.TraceField.SourceGroupScalar] or 1
        traces = np.stack([np.asarray(f.trace[i], dtype=np.float32) for i in range(n_traces)])
        text = segyio.tools.wrap(f.text[0]) if f.text else ""

    # SEG-Y coordinate scalar: negative means divide, positive means multiply.
    factor = np.where(scalars < 0, -1.0 / scalars, np.where(scalars > 0, scalars, 1.0))
    xs, ys = xs * factor, ys * factor

    u_il, il_idx = np.unique(ils, return_inverse=True)
    u_xl, xl_idx = np.unique(xls, return_inverse=True)
    cube = np.zeros((u_il.size, u_xl.size, twt.size), dtype=np.float32)
    gx = np.full((u_il.size, u_xl.size), np.nan)
    gy = np.full((u_il.size, u_xl.size), np.nan)
    cube[il_idx, xl_idx, :] = traces
    gx[il_idx, xl_idx] = xs
    gy[il_idx, xl_idx] = ys

    # Traces missing from an irregular survey leave NaN holes in the geometry.
    gx, gy = _fill_coordinate_holes(gx), _fill_coordinate_holes(gy)

    scalar_mode = int(np.median(scalars[np.isfinite(scalars) & (scalars != 0)])) if np.any(scalars) else -100
    return SeismicVolume(
        data=cube, iline=u_il, xline=u_xl, twt=twt,
        cdp_x=gx, cdp_y=gy, source=path, text_header=text,
        coord_scalar=scalar_mode or -100,
    )


def _fill_coordinate_holes(grid: np.ndarray) -> np.ndarray:
    """Fill NaN gaps in a coordinate grid by interpolating along both axes."""
    g = np.array(grid, dtype=float)
    if np.isfinite(g).all():
        return g
    if not np.isfinite(g).any():
        return np.zeros_like(g)
    for axis in (1, 0):
        g = np.apply_along_axis(
            lambda row: utils.fill_nan_1d(row) if np.isfinite(row).any() else row, axis, g
        )
    return np.nan_to_num(g, nan=float(np.nanmean(grid)))


@_cache_data
def load_segy_cached(path: str, byte_key: tuple, _digest: str) -> SeismicVolume:
    """Cache wrapper.  ``_digest`` is content-derived so a re-upload of the
    same volume with the same byte config is a cache hit."""
    iline, xline, cdp_x, cdp_y = byte_key
    return load_segy(path, iline=iline, xline=xline, cdp_x=cdp_x, cdp_y=cdp_y)


# ==========================================================================
# SEG-Y export
# ==========================================================================

def write_segy(volume: SeismicVolume, out_path: str, template_path: str | None = None) -> str:
    """Write a volume back to SEG-Y.

    When ``template_path`` points at the original file we copy it and overwrite
    the sample values, which preserves every trace header byte the survey
    depends on.  Otherwise segysak writes a fresh file from the geometry we
    hold, which keeps il/xl/x/y but not exotic header fields.
    """
    if template_path and os.path.exists(template_path):
        try:
            return _write_segy_from_template(volume, out_path, template_path)
        except Exception:  # noqa: BLE001 - fall through to the clean writer
            pass

    from segysak.segy import segy_writer

    ds = volume.to_xarray()
    segy_writer(ds, out_path, trace_header_map=dict(iline=189, xline=193, cdp_x=181, cdp_y=185), silent=True)
    return out_path


def _write_segy_from_template(volume: SeismicVolume, out_path: str, template_path: str) -> str:
    import shutil

    import segyio

    shutil.copyfile(template_path, out_path)
    with segyio.open(out_path, "r+", ignore_geometry=True) as f:
        n_samples = len(f.samples)
        if n_samples != volume.data.shape[2]:
            raise ValueError("template sample count does not match the volume")
        il_lookup = {int(v): i for i, v in enumerate(volume.iline)}
        xl_lookup = {int(v): i for i, v in enumerate(volume.xline)}
        # Header byte positions come from the template itself, so read them back.
        for t in range(f.tracecount):
            h = f.header[t]
            i = il_lookup.get(int(h[segyio.TraceField.INLINE_3D]))
            j = xl_lookup.get(int(h[segyio.TraceField.CROSSLINE_3D]))
            if i is None or j is None:
                continue
            f.trace[t] = np.nan_to_num(volume.data[i, j, :]).astype(np.float32)
    return out_path


# ==========================================================================
# LAS loading
# ==========================================================================

SONIC_ALIASES = ("DT", "DTC", "DTCO", "AC", "DT24", "DTP")
DENSITY_ALIASES = ("RHOB", "RHOZ", "DEN", "DENS", "RHO")
TIME_ALIASES = ("TWT", "TIME", "TWTT", "OWT")
VELOCITY_ALIASES = ("VP", "VEL", "PVEL", "VELP")

# Units the log-QC step offers per role.  The sonic list deliberately includes
# velocity units: plenty of LAS files carry Vp directly rather than slowness,
# and forcing the user to pretend it is slowness would corrupt the impedance.
SONIC_UNITS = ("us/ft", "us/m", "m/s", "ft/s")
DENSITY_UNITS = ("g/cm3", "kg/m3")
TIME_UNITS = ("ms (TWT)", "s (TWT)", "ms (OWT)", "s (OWT)")

# Plausible ranges used to flag suspect logs.  Deliberately wide -- these catch
# a unit mistake (a factor of 3.28 or 1000), not an unusual rock.
VP_RANGE = (1200.0, 7500.0)          # m/s
RHO_RANGE = (1400.0, 3300.0)         # kg/m3
AI_RANGE = (1.5e6, 2.3e7)            # m/s * kg/m3


def _first_curve(las, aliases: Sequence[str]) -> tuple[str | None, np.ndarray | None]:
    names = {c.mnemonic.upper(): c.mnemonic for c in las.curves}
    for alias in aliases:
        if alias in names:
            mnem = names[alias]
            return mnem, np.asarray(las[mnem], dtype=float)
    # Loose match, e.g. "DT_R" or "RHOB:1"
    for alias in aliases:
        for upper, mnem in names.items():
            if upper.startswith(alias):
                return mnem, np.asarray(las[mnem], dtype=float)
    return None, None


def autodetect_selection(
    curves: dict[str, np.ndarray],
    units: dict[str, str],
    constant_vp: float | None = None,
    sonic_unit_hint: str | None = None,
) -> CurveSelection:
    """Guess which curve fills each role, and in what unit.

    Mnemonic aliases pick the curve; the LAS unit string is trusted when it is
    recognisable, and otherwise the *magnitude* decides -- a "density" running
    around 2.4 is g/cm3, one around 2400 is kg/m3.  Every guess is overridable
    on the log-QC step, which is the point: this only has to be a good default.
    """
    upper = {k.upper(): k for k in curves}

    def match(aliases):
        for alias in aliases:
            if alias in upper:
                return upper[alias]
        for alias in aliases:
            for u, original in upper.items():
                if u.startswith(alias):
                    return original
        return None

    sel = CurveSelection(constant_vp=constant_vp)

    sonic = match(SONIC_ALIASES)
    velocity = match(VELOCITY_ALIASES)
    if sonic:
        sel.sonic = sonic
        sel.sonic_unit = _guess_sonic_unit(curves[sonic], units.get(sonic, ""), sonic_unit_hint)
    elif velocity:
        sel.sonic = velocity
        sel.sonic_unit = _guess_velocity_unit(curves[velocity], units.get(velocity, ""))

    density = match(DENSITY_ALIASES)
    if density:
        sel.density = density
        sel.density_unit = _guess_density_unit(curves[density], units.get(density, ""))

    time_curve = match(TIME_ALIASES)
    if time_curve:
        sel.time = time_curve
        sel.time_unit = _guess_time_unit(curves[time_curve], units.get(time_curve, ""), time_curve)
    return sel


def _unit_text(unit: str) -> str:
    return str(unit or "").strip().lower().replace(" ", "").replace("_", "/")


def _guess_sonic_unit(values, unit: str, hint: str | None = None) -> str:
    """Resolve a sonic curve's unit: LAS unit string first, then magnitude.

    The hint (the app-wide default) is consulted *last*, and only where the
    magnitude is genuinely ambiguous.  Letting it win earlier would make the
    other evidence useless -- the hint is always a valid unit, so it would
    short-circuit every case.

    Typical magnitudes: us/ft 40-200, us/m 130-650, m/s 1200-7500,
    ft/s 4000-25000.  Only the 130-200 band is truly ambiguous.
    """
    u = _unit_text(unit)
    if "us/m" in u or "usec/m" in u:
        return "us/m"
    if "us/f" in u or "usec/f" in u:
        return "us/ft"
    if "ft/s" in u:
        return "ft/s"
    if "m/s" in u:
        return "m/s"

    med = float(np.nanmedian(np.asarray(values, dtype=float)))
    if not np.isfinite(med) or med <= 0:
        return hint if hint in SONIC_UNITS else "us/ft"
    if med > 900:                       # velocity, not slowness
        return "ft/s" if med > 8000 else "m/s"
    if med > 250:
        return "us/m"
    if med < 130:
        return "us/ft"
    # 130-250 overlaps us/ft and us/m; this is where the hint earns its keep.
    return hint if hint in ("us/ft", "us/m") else "us/ft"


def _guess_velocity_unit(values, unit: str) -> str:
    u = _unit_text(unit)
    if "ft" in u:
        return "ft/s"
    if "m/s" in u:
        return "m/s"
    med = float(np.nanmedian(np.asarray(values, dtype=float)))
    return "ft/s" if np.isfinite(med) and med > 8000 else "m/s"


def _guess_density_unit(values, unit: str) -> str:
    u = _unit_text(unit)
    if "kg" in u:
        return "kg/m3"
    if "g/c" in u:
        return "g/cm3"
    med = float(np.nanmedian(np.asarray(values, dtype=float)))
    return "kg/m3" if np.isfinite(med) and med > 100 else "g/cm3"


def _guess_time_unit(values, unit: str, mnemonic: str) -> str:
    one_way = mnemonic.upper().startswith("OWT")
    med = float(np.nanmedian(np.abs(np.asarray(values, dtype=float))))
    seconds = "s" == _unit_text(unit) or (np.isfinite(med) and med < 20)
    if one_way:
        return "s (OWT)" if seconds else "ms (OWT)"
    return "s (TWT)" if seconds else "ms (TWT)"


def load_las(
    path_or_buffer,
    name: str | None = None,
    constant_vp: float | None = None,
    sonic_unit: str = "us/ft",
    datum_twt: float = 0.0,
    replacement_velocity: float = 2500.0,
    seismic_datum: float = 0.0,
) -> WellData:
    """Parse a LAS file into a :class:`WellData`.

    Every curve in the file is kept, along with its unit and description, and
    the roles (Vp / density / TWT) are auto-detected into a
    :class:`CurveSelection`.  Nothing is discarded on the basis of that guess:
    the log-QC step can reassign any role to any curve and change its unit
    without re-reading the file.

    Raises only when the file yields no curves at all -- a missing density is a
    QC flag to be resolved in the UI, not a hard failure that loses the well.
    """
    import lasio

    if hasattr(path_or_buffer, "read") and not isinstance(path_or_buffer, str):
        raw = path_or_buffer.getvalue() if hasattr(path_or_buffer, "getvalue") else path_or_buffer.read()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        las = lasio.read(io.StringIO(text))
    else:
        las = lasio.read(path_or_buffer)

    raw_depth_unit = _depth_index_unit(las)
    depth_scale, depth_unit = depth_unit_scale(raw_depth_unit)
    md = np.asarray(las.index, dtype=float) * depth_scale
    curves: dict[str, np.ndarray] = {}
    units: dict[str, str] = {}
    descr: dict[str, str] = {}
    for curve in las.curves:
        curves[curve.mnemonic] = np.asarray(las[curve.mnemonic], dtype=float)
        units[curve.mnemonic] = str(getattr(curve, "unit", "") or "")
        descr[curve.mnemonic] = str(getattr(curve, "descr", "") or "")

    if not curves:
        raise ValueError(f"{name or 'LAS'}: file contains no curves")

    well_name = (name or _header_value(las, ("WELL",), "") or "WELL")
    well = WellData(
        name=str(well_name).strip(),
        md=md,
        twt=np.full(md.shape, np.nan),
        vp=np.full(md.shape, np.nan),
        rho=np.full(md.shape, np.nan),
        x=_as_float(_header_value(las, ("XCOORD", "X", "SURFX", "EASTING", "LOCX"), None)),
        y=_as_float(_header_value(las, ("YCOORD", "Y", "SURFY", "NORTHING", "LOCY"), None)),
        kb=float(_as_float(_header_value(las, ("KB", "EKB", "ELEV"), 0.0)) or 0.0) * depth_scale,
        uwi=str(_header_value(las, ("UWI", "API", "WELL"), "") or ""),
        curves=curves,
        curve_units=units,
        curve_descr=descr,
        depth_unit=depth_unit,
    )

    selection = autodetect_selection(curves, units, constant_vp=constant_vp,
                                     sonic_unit_hint=sonic_unit)
    well.apply_selection(selection, replacement_velocity=replacement_velocity,
                         seismic_datum=seismic_datum)
    if depth_unit == "ft":
        well.notes.append(f"Depth index read as feet ({raw_depth_unit}); converted to metres")
    elif not depth_unit:
        well.notes.append(
            f"Depth index unit {raw_depth_unit or 'absent'} not recognised; assumed metres"
        )

    if datum_twt:
        well.set_bulk_shift(float(datum_twt))
        well.notes.append(f"Bulk shift {datum_twt:+.1f} ms applied")
    return well


def _header_value(las, keys: Sequence[str], default):
    for section in ("well", "params"):
        try:
            sec = getattr(las, section)
        except Exception:  # noqa: BLE001
            continue
        for key in keys:
            for item in sec:
                if item.mnemonic.upper() == key.upper():
                    val = item.value
                    if val not in (None, "", "-"):
                        return val
    return default


def _as_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def integrate_sonic_to_twt(
    md: np.ndarray,
    vp: np.ndarray,
    kb: float = 0.0,
    replacement_velocity: float = 2500.0,
    seismic_datum: float = 0.0,
) -> np.ndarray:
    """Sonic-derived time-depth: integrate 1/Vp, then bridge to the datum.

    This is a *drift-free-in-name-only* TD -- with no checkshot to calibrate
    against, absolute time is only as good as the sonic.  The app treats wells
    as tied upstream; this exists so a LAS without a time curve is still usable.
    """
    md = np.asarray(md, dtype=float)
    vp = np.asarray(vp, dtype=float)
    good = np.isfinite(md) & np.isfinite(vp) & (vp > 0)
    twt = np.full(md.shape, np.nan)
    if good.sum() < 2:
        return twt

    m, v = md[good], vp[good]
    dz = np.diff(m, prepend=m[0])
    owt = np.cumsum(dz / v)  # seconds, one-way, from the top of the log

    # Time from the seismic datum down to the first logged sample.
    start_depth = m[0] - kb - seismic_datum
    t0 = max(start_depth, 0.0) / max(replacement_velocity, 1.0)
    twt[good] = 2.0 * (owt + t0) * 1000.0  # ms
    return twt


# ==========================================================================
# Auxiliary well files: time-depth, deviation survey, markers
# ==========================================================================

WELL_FILE_KINDS = ("time_depth", "track", "markers")

# Filename suffixes stripped when matching an auxiliary file to a well.
_AUX_SUFFIXES = ("_td", "_t-d", "_tz", "_checkshot", "_cs", "_markers", "_marker",
                 "_tops", "_track", "_dev", "_deviation", "_survey", "_path")


def _read_text(path_or_buffer) -> str:
    """Read a text file from a path or an upload, tolerating CRLF and BOM."""
    if hasattr(path_or_buffer, "read") and not isinstance(path_or_buffer, str):
        raw = (path_or_buffer.getvalue() if hasattr(path_or_buffer, "getvalue")
               else path_or_buffer.read())
        text = raw.decode("utf-8-sig", errors="replace") if isinstance(raw, bytes) else raw
    else:
        with open(path_or_buffer, "r", encoding="utf-8-sig", errors="replace") as fh:
            text = fh.read()
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _data_lines(text: str) -> list[str]:
    """Non-empty, non-comment lines."""
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line and not line.startswith(("#", "!", "--")):
            out.append(line)
    return out


def _is_number(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def sniff_well_file(path_or_buffer) -> str | None:
    """Identify an auxiliary well file from its contents, not its name.

    Columns are the giveaway and are far more reliable than a file extension:
    two numeric columns is a time-depth table, four is a deviation survey, and
    a number followed by text is a marker list.  This is what lets the user
    drop all of a well's files in at once.
    """
    lines = _data_lines(_read_text(path_or_buffer))
    if not lines:
        return None

    votes: dict[str, int] = {}
    for line in lines[:40]:
        parts = line.split()
        if len(parts) < 2 or not _is_number(parts[0]):
            continue
        numeric = [t for t in parts if _is_number(t)]
        if len(numeric) == len(parts):
            if len(parts) == 2:
                votes["time_depth"] = votes.get("time_depth", 0) + 1
            elif len(parts) >= 4:
                votes["track"] = votes.get("track", 0) + 1
            elif len(parts) == 3:
                # Ambiguous: MD/TVD/time or X/Y/MD. Treated as a track below.
                votes["track"] = votes.get("track", 0) + 1
        else:
            votes["markers"] = votes.get("markers", 0) + 1

    return max(votes, key=votes.get) if votes else None


def load_time_depth(path_or_buffer, time_unit: str = "auto", source: str = "") -> TimeDepth:
    """Read a two-column checkshot / time-depth table (MD, time).

    ``time_unit`` may be ``auto``, or one of :data:`TIME_UNITS`.  Auto-detection
    resolves seconds against milliseconds by magnitude, then one-way against
    two-way by which reading implies a plausible interval velocity -- a
    misread here is a factor-of-two error in every tie downstream, so it is
    worth deciding from the physics rather than from a filename.
    """
    lines = _data_lines(_read_text(path_or_buffer))
    md_list, t_list = [], []
    for line in lines:
        parts = line.split()
        if len(parts) < 2 or not (_is_number(parts[0]) and _is_number(parts[1])):
            continue
        md_list.append(float(parts[0]))
        t_list.append(float(parts[1]))

    if len(md_list) < 2:
        raise ValueError("time-depth file needs at least two 'MD time' rows")

    md = np.asarray(md_list, dtype=float)
    t = np.asarray(t_list, dtype=float)

    if time_unit != "auto" and time_unit in TIME_UNITS:
        seconds = time_unit.startswith("s ")
        one_way = "OWT" in time_unit
    else:
        seconds = float(np.nanmax(np.abs(t))) < 20.0
        one_way = _looks_one_way(md, t * (1000.0 if seconds else 1.0))

    twt_ms = t * (1000.0 if seconds else 1.0)
    if one_way:
        twt_ms = twt_ms * 2.0

    return TimeDepth(md=md, twt=twt_ms, source=source,
                     was_one_way=bool(one_way), was_seconds=bool(seconds))


def _looks_one_way(md: np.ndarray, t_ms: np.ndarray) -> bool:
    """Decide whether a time column is one-way, from the implied velocity.

    Reading the column as two-way gives ``v = 2 dz/dt``.  If that lands in the
    plausible band the column is read as two-way; only when it is impossibly
    fast, and halving it is plausible, is the column taken as one-way.

    **This cannot always be decided.** A one-way table over a slow section
    reads as a plausible fast section: halving the F3 F02-1 checkshot gives
    3850 m/s, which is a perfectly ordinary rock velocity. Detection therefore
    only catches the clearly impossible case; the ambiguous middle defaults to
    two-way (much the commoner convention) and is reported in the summary as
    "read as", with a manual override and a velocity warning to back it up.
    """
    order = np.argsort(md)
    dz = np.diff(md[order])
    dt = np.diff(t_ms[order]) / 1000.0
    good = (dt > 1e-6) & (dz > 0)
    if good.sum() < 1:
        return False
    v_twt = float(np.median(2.0 * dz[good] / dt[good]))
    lo, hi = TD_VELOCITY_RANGE
    if lo <= v_twt <= hi:
        return False                      # two-way reading is plausible
    if v_twt > hi and lo <= v_twt / 2.0 <= hi:
        return True                       # only the one-way reading is plausible
    return False


def load_well_track(path_or_buffer, source: str = "") -> WellTrack:
    """Read a deviation survey.

    Handles the common OpendTect ``.track`` layout ``X Y Z MD`` (Z as TVDSS,
    positive down), and tolerates mixed tab/space delimiters -- the F3 demo
    tracks mix both on the same row.  A file whose Z decreases with MD is
    treated as an elevation and flipped to TVDSS.
    """
    lines = _data_lines(_read_text(path_or_buffer))
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 4 and all(_is_number(t) for t in parts[:4]):
            rows.append([float(v) for v in parts[:4]])
        elif len(parts) == 3 and all(_is_number(t) for t in parts):
            # X Y MD, with no separate TVD column: treat MD as vertical depth.
            x, y, md = (float(v) for v in parts)
            rows.append([x, y, md, md])

    if len(rows) < 1:
        raise ValueError("deviation file needs rows of 'X Y Z MD'")

    arr = np.asarray(rows, dtype=float)
    x, y, z, md = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

    # A single station cannot define a path; treat it as a vertical well.
    if arr.shape[0] == 1:
        md = np.array([md[0], md[0] + 1.0])
        z = np.array([z[0], z[0] + 1.0])
        x = np.array([x[0], x[0]])
        y = np.array([y[0], y[0]])

    order = np.argsort(md)
    if np.polyfit(md[order], z[order], 1)[0] < 0:
        z = -z                            # elevation (positive up) -> TVDSS

    return WellTrack(md=md, x=x, y=y, tvdss=z, source=source)


def load_markers(path_or_buffer, source: str = "") -> list[Marker]:
    """Read a formation-top list of ``MD  name``.

    The name is everything after the depth, so tops with spaces in their names
    -- ``Truncation 1``, ``NMRF (Mid_Mio_Unc)`` -- survive intact.
    """
    markers: list[Marker] = []
    for line in _data_lines(_read_text(path_or_buffer)):
        parts = line.split(None, 1)
        if len(parts) < 2 or not _is_number(parts[0]):
            continue
        name = parts[1].strip()
        if name:
            markers.append(Marker(md=float(parts[0]), name=name))
    if not markers:
        raise ValueError("marker file needs rows of 'MD name'")
    return sorted(markers, key=lambda m: m.md)


def match_well_name(filename: str, well_names: Sequence[str]) -> str | None:
    """Match an auxiliary filename to a loaded well.

    Comparison strips the extension, any known role suffix (``_TD``,
    ``_markers``, ...) and every non-alphanumeric character, so ``F02-1_TD.txt``
    and ``F021_TD.txt`` both match a well called ``F02-1``.
    """
    stem = os.path.splitext(os.path.basename(str(filename)))[0]
    key = stem.lower()
    for suffix in _AUX_SUFFIXES:
        if key.endswith(suffix):
            key = key[: -len(suffix)]
            break
    key = "".join(ch for ch in key if ch.isalnum())
    if not key:
        return None

    normalised = {name: "".join(ch for ch in name.lower() if ch.isalnum()) for name in well_names}
    for name, norm in normalised.items():
        if norm == key:
            return name
    # Fall back to the longest containment match, so "F02-1_sonic_edit" still
    # finds "F02-1" without a shorter well name winning by accident.
    candidates = [n for n, norm in normalised.items() if norm and (norm in key or key in norm)]
    return max(candidates, key=lambda n: len(normalised[n])) if candidates else None


# ==========================================================================
# Folder scan: load a whole well database from a directory tree
# ==========================================================================

# Folder-name hints, used only to break ties -- the contents decide.
_FOLDER_HINTS = {
    "time_depth": ("checkshot", "check shot", "checkshots", "td", "time-depth",
                   "timedepth", "time_depth", "tz", "t-d"),
    "track": ("track", "tracks", "deviation", "deviations", "survey", "path", "dev"),
    "markers": ("tops", "top", "marker", "markers", "picks", "horizons_well"),
    "las": ("las", "lasfiles", "las files", "logs", "wells", "welllogs", "well logs"),
}

# Extensions worth opening during a scan. Anything else is left alone so a
# stray SEG-Y in the tree is never read as a text file.
_SCAN_EXTENSIONS = {".las", ".txt", ".csv", ".dat", ".track", ".asc", ".tz",
                    ".md", ".prn", ".tab", ""}
_MAX_SCAN_BYTES = 64 * 1024 * 1024


@dataclass
class FolderScan:
    """What a directory scan found, and what it managed to do with it."""

    wells: list = field(default_factory=list)
    attached: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    folders: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "wells loaded": len(self.wells),
            "LAS files": self.counts.get("las", 0),
            "checkshots": self.counts.get("time_depth", 0),
            "deviation surveys": self.counts.get("track", 0),
            "marker files": self.counts.get("markers", 0),
            "files skipped": len(self.skipped),
        }


def _folder_role(name: str) -> str | None:
    key = str(name).strip().lower().replace("_", " ")
    for role, hints in _FOLDER_HINTS.items():
        if any(key == h or key.replace(" ", "") == h.replace(" ", "") for h in hints):
            return role
    for role, hints in _FOLDER_HINTS.items():
        if any(h in key for h in hints):
            return role
    return None


def _classify(path: str, folder_role: str | None) -> tuple[str | None, str]:
    """Decide what a file is.  Returns ``(kind, why)``.

    Contents decide; the folder name only breaks a tie.  When the two disagree
    the contents win and the disagreement is reported, because a file in the
    wrong folder is far commoner than a file that lies about its own columns.
    """
    if os.path.splitext(path)[1].lower() == ".las":
        return "las", "LAS extension"
    try:
        sniffed = sniff_well_file(path)
    except Exception:  # noqa: BLE001 - unreadable file, leave it alone
        return None, "could not be read as text"

    if sniffed is None:
        return (folder_role, f"no recognisable columns; taken from the '{folder_role}' folder")             if folder_role in WELL_FILE_KINDS else (None, "no recognisable columns")
    if folder_role in WELL_FILE_KINDS and folder_role != sniffed:
        return sniffed, f"contents look like {sniffed}, not {folder_role} as the folder suggests"
    return sniffed, "from contents"


def scan_well_folder(
    root: str,
    max_depth: int = 3,
    time_unit: str = "auto",
    prefer_checkshot: bool = True,
    sonic_unit_hint: str = "us/ft",
) -> FolderScan:
    """Load a whole well database from a directory tree.

    Built for the layout the F3 demo ships in -- one folder per data type
    (``Lasfiles``, ``Checkshot``, ``Track``, ``Tops``) with one file per well
    inside each -- but it does not depend on those names.  Files are classified
    by contents, folder names only break ties, and wells are keyed off the LAS
    files, since those carry the logs everything else decorates.

    Unrecognised folders and files are reported rather than ignored silently,
    so a file that did not load is visible instead of merely absent.
    """
    root = os.path.abspath(os.path.expanduser(str(root)))
    if not os.path.isdir(root):
        raise ValueError(f"not a folder: {root}")

    scan = FolderScan()
    candidates: list[tuple[str, str | None]] = []

    root_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.rstrip(os.sep).count(os.sep) - root_depth >= max_depth:
            dirnames[:] = []
        role = _folder_role(os.path.basename(dirpath)) if dirpath != root else None
        if dirpath != root:
            scan.folders[os.path.relpath(dirpath, root)] = role or "unrecognised"
        for filename in sorted(filenames):
            if filename.startswith("."):
                continue
            full = os.path.join(dirpath, filename)
            if os.path.splitext(filename)[1].lower() not in _SCAN_EXTENSIONS:
                continue
            try:
                if os.path.getsize(full) > _MAX_SCAN_BYTES:
                    scan.skipped.append(f"{os.path.relpath(full, root)} (too large to scan)")
                    continue
            except OSError:
                continue
            candidates.append((full, role))

    # --- pass 1: LAS files define the wells ------------------------------
    aux: list[tuple[str, str, str]] = []
    for full, role in candidates:
        kind, why = _classify(full, role)
        rel = os.path.relpath(full, root)
        if kind is None:
            scan.skipped.append(f"{rel} ({why})")
            continue
        if kind == "las":
            try:
                well = load_las(full, name=os.path.splitext(os.path.basename(full))[0],
                                sonic_unit=sonic_unit_hint)
                scan.wells.append(well)
                scan.counts["las"] = scan.counts.get("las", 0) + 1
                scan.attached.append(f"{rel} -> well **{well.name}**")
            except Exception as exc:  # noqa: BLE001 - one bad LAS must not stop the scan
                scan.skipped.append(f"{rel} ({exc})")
        else:
            aux.append((full, kind, why))

    if not scan.wells:
        raise ValueError(
            f"no LAS files found under {root}. Expected a folder of .las files "
            "(the scan keys wells off the logs, then matches the other files to them)."
        )

    # --- pass 2: attach everything else to those wells -------------------
    names = [w.name for w in scan.wells]
    for full, kind, why in aux:
        rel = os.path.relpath(full, root)
        target = match_well_name(os.path.basename(full), names)
        if target is None:
            scan.skipped.append(f"{rel} (no well matched; wells are {', '.join(names)})")
            continue
        well = next(w for w in scan.wells if w.name == target)
        try:
            if kind == "time_depth":
                message = well.attach_time_depth(
                    load_time_depth(full, time_unit=time_unit, source=rel), prefer=prefer_checkshot)
            elif kind == "track":
                message = well.attach_track(load_well_track(full, source=rel))
            else:
                message = well.attach_markers(load_markers(full, source=rel))
            scan.counts[kind] = scan.counts.get(kind, 0) + 1
            note = f" ({why})" if why != "from contents" else ""
            scan.attached.append(f"{rel} -> **{target}**: {message}{note}")
        except Exception as exc:  # noqa: BLE001
            scan.skipped.append(f"{rel} ({exc})")

    scan.wells.sort(key=lambda w: w.name)
    return scan


def find_segy_files(root: str, max_depth: int = 3, limit: int = 40) -> list[str]:
    """List SEG-Y files under a folder, largest first.

    Offered alongside a well-folder scan so a survey that keeps its seismic
    beside its wells can be loaded without retyping the path.
    """
    root = os.path.abspath(os.path.expanduser(str(root)))
    if not os.path.isdir(root):
        return []
    found: list[tuple[int, str]] = []
    root_depth = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        if dirpath.rstrip(os.sep).count(os.sep) - root_depth >= max_depth:
            dirnames[:] = []
        for filename in filenames:
            if os.path.splitext(filename)[1].lower() in (".sgy", ".segy"):
                full = os.path.join(dirpath, filename)
                try:
                    found.append((os.path.getsize(full), full))
                except OSError:
                    continue
    found.sort(reverse=True)
    return [path for _, path in found[:limit]]


def load_well_headers(path_or_buffer) -> pd.DataFrame:
    """Read a well-header CSV (name/uwi + X/Y + optional KB).

    Column names are matched case-insensitively against a set of aliases, so a
    header exported from almost any package lands in the right place.
    """
    df = pd.read_csv(path_or_buffer)
    lookup = {str(c).strip().lower(): c for c in df.columns}

    def pick(*aliases):
        for a in aliases:
            if a in lookup:
                return lookup[a]
        return None

    out = pd.DataFrame()
    name_col = pick("well", "well_name", "wellname", "name", "uwi")
    out["well"] = df[name_col].astype(str).str.strip() if name_col else [f"WELL_{i}" for i in range(len(df))]
    for target, aliases in (
        ("x", ("x", "surf_x", "surfx", "easting", "cdp_x", "utmx", "x_coord")),
        ("y", ("y", "surf_y", "surfy", "northing", "cdp_y", "utmy", "y_coord")),
        ("kb", ("kb", "ekb", "kb_elev", "elev", "elevation", "datum")),
    ):
        col = pick(*aliases)
        out[target] = pd.to_numeric(df[col], errors="coerce") if col else np.nan
    return out


def apply_well_headers(wells: Sequence[WellData], headers: pd.DataFrame) -> list[str]:
    """Attach X/Y/KB from a header table onto already-loaded wells (in place).

    Matching is case-insensitive on the well name, then on the UWI.  Returns a
    list of human-readable messages for the UI.
    """
    msgs: list[str] = []
    if headers is None or headers.empty:
        return msgs
    idx = {str(r["well"]).strip().upper(): r for _, r in headers.iterrows()}
    for w in wells:
        # Explicit None checks, never `a or b`: the rows are pandas Series and
        # truth-testing one raises rather than falling through to the UWI.
        row = idx.get(w.name.strip().upper())
        if row is None:
            row = idx.get(str(w.uwi).strip().upper())
        if row is None:
            msgs.append(f"{w.name}: no header row matched")
            continue

        x, y, kb = (_as_float(row.get(k)) for k in ("x", "y", "kb"))
        if x is not None and y is not None:
            w.x, w.y = x, y
        if kb is not None:
            w.kb = kb
        msgs.append(f"{w.name}: located at ({w.x:.1f}, {w.y:.1f})" if w.has_location
                    else f"{w.name}: header row had no usable X/Y")
    return msgs


# ==========================================================================
# Well <-> seismic geometry
# ==========================================================================

@dataclass
class WellTie:
    """Everything needed to compare one well against the seismic."""

    well: str
    il_index: int
    xl_index: int
    iline: int
    xline: int
    distance: float
    seismic: np.ndarray          # IDW-blended extracted trace
    reflectivity: np.ndarray     # well reflectivity on the seismic time axis
    ai: np.ndarray               # well AI on the seismic time axis
    twt: np.ndarray
    n_neighbours: int = 1
    # Set when the trace was sampled down the borehole rather than at the
    # surface location (see extract_well_traces_along_path).
    followed_path: bool = False
    path_step_out: float = 0.0


def extract_well_traces(
    volume: SeismicVolume,
    wells: Sequence[WellData],
    k: int = 4,
    power: float = 2.0,
) -> list[WellTie]:
    """Locate each well on the seismic grid and pull an IDW-blended trace.

    Only wells with a map location participate; a well with no X/Y is skipped
    (the caller surfaces that in the UI rather than silently guessing).
    """
    ties: list[WellTie] = []
    xy = volume.trace_xy()
    live = volume.live_mask()
    flat = volume.flat_data()

    for w in wells:
        if not w.has_location:
            continue
        nb = utils.nearest_live_traces(xy, (w.x, w.y), k=k, live_mask=live, power=power)
        blended = utils.blend_traces(flat[nb.indices, :], nb.weights)
        i, j = volume.flat_to_ij(nb.nearest_index)
        ties.append(
            WellTie(
                well=w.name,
                il_index=i,
                xl_index=j,
                iline=int(volume.iline[i]),
                xline=int(volume.xline[j]),
                distance=float(nb.distances[0]),
                seismic=np.asarray(blended, dtype=float),
                reflectivity=w.reflectivity_on_time_axis(volume.twt),
                ai=w.ai_on_time_axis(volume.twt),
                twt=volume.twt,
                n_neighbours=int(nb.indices.size),
            )
        )
    return ties


def extract_well_traces_along_path(
    volume: SeismicVolume,
    wells: Sequence[WellData],
    k: int = 4,
    power: float = 2.0,
) -> list[WellTie]:
    """Sample the seismic *down the borehole* rather than at the surface point.

    A vertical well and its deviation survey are the same thing, so this falls
    back to :func:`extract_well_traces` when a well has no track or the track is
    vertical.  For a genuinely deviated well they are not the same thing at all:
    a 1,000 m step-out puts the reservoir section tens of bins away from the
    wellhead, and tying the logs to the surface trace ties them to the wrong
    rock.

    Each seismic sample time is handled separately.  The well's own time-depth
    gives the measured depth at that time, the deviation survey gives the map
    position at that depth, and the trace value is taken from there -- so the
    "trace" that comes back is a composite, following the path of the borehole
    through the cube.  Above and below the logged interval the path is held at
    its shallowest and deepest known positions rather than extrapolated.
    """
    xy = volume.trace_xy()
    live = volume.live_mask()
    flat = volume.flat_data()
    twt_axis = np.asarray(volume.twt, dtype=float)

    ties: list[WellTie] = []
    for w in wells:
        if not w.has_location:
            continue
        track = getattr(w, "track", None)
        if track is None or track.is_vertical:
            ties.extend(extract_well_traces(volume, [w], k=k, power=power))
            continue

        good = np.isfinite(w.twt) & np.isfinite(w.md)
        if good.sum() < 2:
            ties.extend(extract_well_traces(volume, [w], k=k, power=power))
            continue

        # TWT -> MD needs a monotonic time axis to invert; sort and dedupe.
        t_log, md_log = w.twt[good], w.md[good]
        order = np.argsort(t_log)
        t_log, md_log = t_log[order], md_log[order]
        keep = np.concatenate(([True], np.diff(t_log) > 0))
        t_log, md_log = t_log[keep], md_log[keep]
        if t_log.size < 2:
            ties.extend(extract_well_traces(volume, [w], k=k, power=power))
            continue

        md_at_t = np.interp(twt_axis, t_log, md_log, left=md_log[0], right=md_log[-1])
        px, py = track.xy_at(md_at_t)

        composite = np.zeros(twt_axis.size, dtype=float)
        nearest_index = None
        distances = []
        # Positions repeat over long stretches, so cache by trace index.
        cache: dict[tuple[float, float], utils.TraceNeighbourhood] = {}
        for s in range(twt_axis.size):
            key = (round(float(px[s]), 1), round(float(py[s]), 1))
            nb = cache.get(key)
            if nb is None:
                nb = utils.nearest_live_traces(xy, key, k=k, live_mask=live, power=power)
                cache[key] = nb
            composite[s] = float(utils.blend_traces(flat[nb.indices, :], nb.weights)[s])
            distances.append(nb.distances[0])
            if s == twt_axis.size // 2:
                nearest_index = nb.nearest_index
        if nearest_index is None:
            nearest_index = 0

        i, j = volume.flat_to_ij(nearest_index)
        tie = WellTie(
            well=w.name, il_index=i, xl_index=j,
            iline=int(volume.iline[i]), xline=int(volume.xline[j]),
            distance=float(np.mean(distances)),
            seismic=composite,
            reflectivity=w.reflectivity_on_time_axis(twt_axis),
            ai=w.ai_on_time_axis(twt_axis),
            twt=twt_axis, n_neighbours=int(k),
        )
        tie.path_step_out = float(np.hypot(px - px[0], py - py[0]).max())
        tie.followed_path = True
        ties.append(tie)
    return ties


# ==========================================================================
# Arbitrary line extraction
# ==========================================================================

@dataclass
class ArbitraryLine:
    """A seismic section sampled along a polyline through the survey."""

    data: np.ndarray                     # (n_points, n_samples)
    distance: np.ndarray                 # cumulative metres along the path
    x: np.ndarray
    y: np.ndarray
    iline: np.ndarray
    xline: np.ndarray
    twt: np.ndarray
    node_distance: np.ndarray = field(default_factory=lambda: np.array([]))
    node_label: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "points": int(self.data.shape[0]),
            "length (m)": f"{self.distance[-1]:.0f}" if self.distance.size else "0",
            "nodes": len(self.node_label),
        }


def extract_arbitrary_line(
    volume,
    points: Sequence[Sequence[float]],
    labels: Sequence[str] | None = None,
    step_m: float | None = None,
) -> ArbitraryLine:
    """Sample the cube along a polyline joining map coordinates.

    Traces are taken by nearest bin rather than interpolated: a seismic section
    should show real traces, and interpolating between them would invent
    amplitudes that the survey never recorded.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2:
        raise ValueError("an arbitrary line needs at least two points")

    xy = volume.trace_xy()
    n_il, n_xl = volume.data.shape[:2]

    # Default sampling: roughly one point per bin along the path.
    if step_m is None:
        spacing = np.median(np.abs(np.diff(volume.cdp_x[:, 0]))) if n_il > 1 else 0.0
        alt = np.median(np.abs(np.diff(volume.cdp_y[0, :]))) if n_xl > 1 else 0.0
        step_m = float(max(min([v for v in (spacing, alt) if v > 0], default=25.0), 1.0))

    seg_lengths = np.hypot(*(pts[1:] - pts[:-1]).T)
    total = float(seg_lengths.sum())
    if total <= 0:
        raise ValueError("the polyline has zero length -- are all the wells at the same location?")
    n_points = int(np.clip(round(total / step_m) + 1, 2, 5000))

    node_distance = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    want = np.linspace(0.0, total, n_points)
    path_x = np.interp(want, node_distance, pts[:, 0])
    path_y = np.interp(want, node_distance, pts[:, 1])

    tree = cKDTree(xy)
    _, flat_idx = tree.query(np.column_stack([path_x, path_y]))
    il_idx, xl_idx = np.divmod(flat_idx, n_xl)

    return ArbitraryLine(
        data=volume.flat_data()[flat_idx, :],
        distance=want,
        x=path_x, y=path_y,
        iline=volume.iline[il_idx],
        xline=volume.xline[xl_idx],
        twt=volume.twt,
        node_distance=node_distance,
        node_label=list(labels) if labels is not None else [],
    )


def line_through_wells(volume, wells: Sequence, order: Sequence[str] | None = None) -> ArbitraryLine:
    """Arbitrary line joining the given wells, in the order supplied."""
    located = {w.name: w for w in wells if getattr(w, "has_location", False)}
    names = [n for n in (order or list(located)) if n in located]
    if len(names) < 2:
        raise ValueError("at least two located wells are needed for a traverse")
    pts = [(located[n].x, located[n].y) for n in names]
    return extract_arbitrary_line(volume, pts, labels=names)


# ==========================================================================
# Synthetic fallback dataset
# ==========================================================================

def make_synthetic_dataset(
    n_iline: int = 40,
    n_xline: int = 40,
    n_samples: int = 400,
    sample_rate_ms: float = 2.0,
    n_wells: int = 4,
    noise: float = 0.08,
    seed: int = 42,
    peak_frequency: float = 28.0,
) -> tuple[SeismicVolume, list[WellData]]:
    """Generate a small 3D cube plus tied pseudo-wells.

    A layered earth model with a gently dipping structure and lateral impedance
    variation is built first; the wells are sampled straight out of that model
    and the seismic is its reflectivity convolved with a Ricker wavelet, so the
    inversion has a known right answer to be checked against.
    """
    rng = np.random.default_rng(seed)
    twt = np.arange(n_samples) * sample_rate_ms
    dt = sample_rate_ms / 1000.0

    # --- layered background: velocity and density trend with depth ---------
    n_layers = 28
    tops = np.sort(rng.uniform(0, n_samples, n_layers - 1)).astype(int)
    layer_id = np.zeros(n_samples, dtype=int)
    for k, t in enumerate(tops):
        layer_id[t:] = k + 1

    depth_trend = 1800.0 + 1.6 * twt                      # m/s, compaction
    layer_vp = rng.normal(0.0, 260.0, n_layers)
    layer_rho = rng.normal(0.0, 90.0, n_layers)

    # --- structure: a dipping, gently folded surface in samples ------------
    ii, jj = np.meshgrid(np.arange(n_iline), np.arange(n_xline), indexing="ij")
    structure = (
        0.55 * ii + 0.30 * jj
        + 9.0 * np.sin(2 * np.pi * ii / max(n_iline, 1))
        + 6.0 * np.cos(2 * np.pi * jj / max(n_xline, 1) * 1.5)
    )
    structure = np.round(structure - structure.mean()).astype(int)

    # --- lateral facies change: a smooth impedance anomaly ------------------
    anomaly = 1.0 + 0.10 * np.exp(
        -(((ii - n_iline * 0.35) / (n_iline * 0.18)) ** 2 + ((jj - n_xline * 0.6) / (n_xline * 0.2)) ** 2)
    )

    ai_cube = np.zeros((n_iline, n_xline, n_samples), dtype=np.float32)
    for i in range(n_iline):
        for j in range(n_xline):
            shift = int(structure[i, j])
            idx = np.clip(np.arange(n_samples) + shift, 0, n_samples - 1)
            vp = depth_trend[idx] + layer_vp[layer_id[idx]]
            rho = 1900.0 + 0.30 * twt[idx] + layer_rho[layer_id[idx]]
            ai_cube[i, j, :] = (vp * rho) * anomaly[i, j]

    # --- seismic = reflectivity * wavelet + noise --------------------------
    from .wavelet import ricker

    wav = ricker(peak_frequency, length_ms=120.0, dt=dt)
    seis = np.zeros_like(ai_cube)
    for i in range(n_iline):
        for j in range(n_xline):
            r = utils.reflectivity_from_ai(ai_cube[i, j, :].astype(float))
            seis[i, j, :] = utils.convolve_same(r, wav)
    scale = float(np.percentile(np.abs(seis), 99)) or 1.0
    seis = seis / scale
    seis += rng.normal(0.0, noise * float(np.std(seis)), seis.shape)

    # --- geometry ----------------------------------------------------------
    bin_size, origin_x, origin_y = 25.0, 500_000.0, 6_000_000.0
    cdp_x = origin_x + ii * bin_size
    cdp_y = origin_y + jj * bin_size

    volume = SeismicVolume(
        data=seis.astype(np.float32),
        iline=np.arange(1000, 1000 + n_iline),
        xline=np.arange(2000, 2000 + n_xline),
        twt=twt,
        cdp_x=cdp_x,
        cdp_y=cdp_y,
        source="synthetic",
        text_header="SYNTHETIC DEMO VOLUME - generated by seismic_inversion_app",
    )

    # --- wells: sampled from the true model, at log rate --------------------
    wells: list[WellData] = []
    pos = _well_positions(n_iline, n_xline, n_wells, rng)
    for k, (i, j) in enumerate(pos):
        shift = int(structure[i, j])
        idx = np.clip(np.arange(n_samples) + shift, 0, n_samples - 1)
        vp_c = depth_trend[idx] + layer_vp[layer_id[idx]]
        rho_c = 1900.0 + 0.30 * twt[idx] + layer_rho[layer_id[idx]]
        vp_c = vp_c * np.sqrt(anomaly[i, j])
        rho_c = rho_c * np.sqrt(anomaly[i, j])

        # Upsample to a plausible log rate and add a little measurement noise.
        fine_twt = np.arange(twt[0], twt[-1], sample_rate_ms / 8.0)
        vp_f = np.interp(fine_twt, twt, vp_c) * (1 + rng.normal(0, 0.006, fine_twt.size))
        rho_f = np.interp(fine_twt, twt, rho_c) * (1 + rng.normal(0, 0.004, fine_twt.size))
        md = np.cumsum(np.gradient(fine_twt) / 1000.0 / 2.0 * vp_f)

        # Carry the same curve/unit machinery as a loaded LAS: sonic in us/ft,
        # density in g/cm3, an explicit TWT curve.  Without this the synthetic
        # wells would be a special case that the log-QC step cannot describe --
        # and "apply assignment" there would wipe logs it could not see.
        curves = {
            "DEPT": md,
            "DT": 1.0e6 / (vp_f * utils.FT_PER_M),
            "RHOB": rho_f / 1000.0,
            "TWT": fine_twt,
        }
        well = WellData(
            name=f"SYN-{k + 1}",
            md=md, twt=fine_twt, vp=vp_f, rho=rho_f,
            x=float(cdp_x[i, j]), y=float(cdp_y[i, j]),
            kb=30.0, uwi=f"SYNTHETIC-{k + 1:03d}",
            curves=curves,
            curve_units={"DEPT": "m", "DT": "us/ft", "RHOB": "g/cm3", "TWT": "ms"},
            curve_descr={"DEPT": "Measured depth", "DT": "Sonic slowness",
                         "RHOB": "Bulk density", "TWT": "Two-way time (already tied)"},
            depth_unit="m",   # built in metres, so QC should not flag it as unlabelled
        )
        well.apply_selection(CurveSelection(
            sonic="DT", sonic_unit="us/ft",
            density="RHOB", density_unit="g/cm3",
            time="TWT", time_unit="ms (TWT)",
        ))
        well.notes.insert(0, "synthetic well sampled from the true impedance model")
        wells.append(well)

    return volume, wells


def _well_positions(n_iline: int, n_xline: int, n_wells: int, rng) -> list[tuple[int, int]]:
    """Spread wells out rather than clustering them, so interpolation is tested."""
    if n_wells <= 0:
        return []
    grid = int(np.ceil(np.sqrt(n_wells)))
    pos: list[tuple[int, int]] = []
    for k in range(n_wells):
        gi, gj = divmod(k, grid)
        fi = (gi + 0.5) / grid
        fj = (gj + 0.5) / grid
        i = int(np.clip(fi * n_iline + rng.integers(-2, 3), 1, n_iline - 2))
        j = int(np.clip(fj * n_xline + rng.integers(-2, 3), 1, n_xline - 2))
        pos.append((i, j))
    return pos
