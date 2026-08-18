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

    def __post_init__(self) -> None:
        self.md = np.asarray(self.md, dtype=float)
        self.twt = np.asarray(self.twt, dtype=float)
        self.vp = np.asarray(self.vp, dtype=float)
        self.rho = np.asarray(self.rho, dtype=float)
        # Snapshot of the as-loaded time axis, so repeated bulk shifts are
        # applied from the original rather than accumulating on each other.
        self._twt_base = self.twt.copy()

    def set_bulk_shift(self, milliseconds: float) -> None:
        """Apply a constant time shift to the whole well (v1 tie adjustment).

        Stretch and squeeze are deliberately out of scope for v1 -- a bulk shift
        can only fix a datum error, not a drifting time-depth relationship.
        """
        self.bulk_shift = float(milliseconds)
        self.twt = self._twt_base + float(milliseconds)

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


def persist_upload(uploaded, suffix: str = ".sgy") -> str:
    """Spill a Streamlit UploadedFile to disk; segyio needs a real path."""
    raw = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(raw)
    tmp.close()
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

    Requires a density curve plus either a sonic curve or ``constant_vp``.
    The time axis is taken from a TWT curve when the LAS carries one (wells are
    assumed tied upstream); otherwise it is built by integrating the sonic from
    the datum, with ``replacement_velocity`` bridging KB to the seismic datum.
    """
    import lasio

    if hasattr(path_or_buffer, "read") and not isinstance(path_or_buffer, str):
        raw = path_or_buffer.getvalue() if hasattr(path_or_buffer, "getvalue") else path_or_buffer.read()
        text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        las = lasio.read(io.StringIO(text))
    else:
        las = lasio.read(path_or_buffer)

    notes: list[str] = []
    md = np.asarray(las.index, dtype=float)

    sonic_name, dt_curve = _first_curve(las, SONIC_ALIASES)
    rho_name, rhob = _first_curve(las, DENSITY_ALIASES)

    if rhob is None:
        raise ValueError(
            f"{name or 'LAS'}: no density curve found (looked for {', '.join(DENSITY_ALIASES)})"
        )

    if dt_curve is not None:
        vp = utils.sonic_to_velocity(dt_curve, unit=sonic_unit)
        notes.append(f"Vp from sonic curve '{sonic_name}' ({sonic_unit})")
    elif constant_vp is not None:
        vp = np.full(md.shape, float(constant_vp))
        notes.append(f"No sonic curve; using constant Vp = {constant_vp:.0f} m/s")
    else:
        raise ValueError(
            f"{name or 'LAS'}: no sonic curve found and no constant Vp supplied "
            f"(looked for {', '.join(SONIC_ALIASES)})"
        )

    # Density: LAS is usually g/cm3; anything above 100 is already kg/m3.
    rho = np.asarray(rhob, dtype=float)
    if np.nanmedian(rho) < 100:
        rho = utils.density_to_si(rho, "g/cm3")
        notes.append("Density converted g/cm3 -> kg/m3")

    time_name, twt_curve = _first_curve(las, TIME_ALIASES)
    if twt_curve is not None and np.isfinite(twt_curve).sum() > 2:
        twt = np.asarray(twt_curve, dtype=float)
        if time_name and time_name.upper() == "OWT":
            twt = twt * 2.0
            notes.append("OWT curve doubled to TWT")
        if np.nanmax(twt) < 20:  # seconds, not milliseconds
            twt = twt * 1000.0
            notes.append("Time curve scaled s -> ms")
        notes.append(f"TWT taken from curve '{time_name}' (well assumed already tied)")
    else:
        twt = integrate_sonic_to_twt(
            md, vp, kb=_header_value(las, ("KB", "EKB", "ELEV"), 0.0),
            replacement_velocity=replacement_velocity, seismic_datum=seismic_datum,
        )
        notes.append("TWT integrated from sonic (no time curve in LAS)")

    if datum_twt:
        twt = twt + float(datum_twt)
        notes.append(f"Bulk shift {datum_twt:+.1f} ms applied")

    well_name = (name or _header_value(las, ("WELL",), "") or "WELL").strip()
    uwi = str(_header_value(las, ("UWI", "API", "WELL"), "") or "")
    x = _header_value(las, ("XCOORD", "X", "SURFX", "EASTING", "LOCX"), None)
    y = _header_value(las, ("YCOORD", "Y", "SURFY", "NORTHING", "LOCY"), None)
    kb = _header_value(las, ("KB", "EKB", "ELEV"), 0.0)

    curves = {c.mnemonic: np.asarray(las[c.mnemonic], dtype=float) for c in las.curves}

    return WellData(
        name=well_name, md=md, twt=twt, vp=vp, rho=rho,
        x=_as_float(x), y=_as_float(y), kb=float(_as_float(kb) or 0.0),
        uwi=uwi, curves=curves, notes=notes,
    )


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

        wells.append(
            WellData(
                name=f"SYN-{k + 1}",
                md=md, twt=fine_twt, vp=vp_f, rho=rho_f,
                x=float(cdp_x[i, j]), y=float(cdp_y[i, j]),
                kb=30.0, uwi=f"SYNTHETIC-{k + 1:03d}",
                notes=["synthetic well sampled from the true impedance model"],
            )
        )

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
