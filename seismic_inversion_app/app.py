"""Post-stack seismic inversion -- Streamlit entry point.

A six-step, sidebar-driven workflow: load data, QC the ties, build a wavelet,
build a background model, invert, inspect and export.  Everything the user
produces is kept in ``st.session_state`` so a step never has to be redone just
because the page changed.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import os
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st

from modules import data_io, inversion, low_freq_model, utils, visualization as viz, wavelet as wvl

st.set_page_config(page_title="Post-Stack Seismic Inversion", page_icon="~", layout="wide")

STEPS = [
    "1 - Data",
    "2 - Log QC",
    "3 - Well tie QC",
    "4 - Wavelet",
    "5 - Low-frequency model",
    "6 - Inversion",
    "7 - Results & export",
]

DEFAULTS = {
    "volume": None,
    "wells": [],
    "ties": [],
    "headers": None,
    "horizons": None,
    "wavelet": None,
    "colour_operator": None,
    "lfm": None,
    "result": None,
    "segy_path": None,
    "gate": None,
    "k_neighbours": 4,
    "job": None,
    "log": [],
    "_flash": None,
}


# ==========================================================================
# Session state plumbing
# ==========================================================================

def init_state() -> None:
    for key, value in DEFAULTS.items():
        st.session_state.setdefault(key, value.copy() if isinstance(value, (list, dict)) else value)


def flash(message: str, kind: str = "success") -> None:
    """Queue a message and rerun immediately.

    The sidebar renders before the page body, so a status change made while
    handling a button would not reach the sidebar until the *next* interaction.
    Queuing the message and rerunning means the sidebar, the page and the
    message all describe the same state.
    """
    st.session_state._flash = (kind, message)
    st.rerun()


def show_flash() -> None:
    item = st.session_state.pop("_flash", None)
    if item is None:
        return
    kind, message = item
    {"success": st.success, "info": st.info, "warning": st.warning, "error": st.error}[kind](message)


def log(message: str) -> None:
    """Append to the running activity log shown in the sidebar."""
    st.session_state.log.append(f"{time.strftime('%H:%M:%S')}  {message}")
    st.session_state.log = st.session_state.log[-40:]


def rebuild_ties(reason: str = "") -> None:
    """Re-extract the seismic trace at every located well.

    Called whenever the volume, the wells or a bulk shift changes -- the ties
    feed the wavelet, the operator and every QC panel, so a stale set would
    quietly poison everything downstream.
    """
    vol, wells = st.session_state.volume, st.session_state.wells
    if vol is None or not wells:
        st.session_state.ties = []
        return
    st.session_state.ties = data_io.extract_well_traces(vol, wells, k=st.session_state.k_neighbours)
    if reason:
        log(f"ties rebuilt ({len(st.session_state.ties)} wells) - {reason}")


def default_gate() -> tuple[float, float]:
    """A sensible analysis gate: the interval the wells actually cover."""
    vol = st.session_state.volume
    ties = st.session_state.ties
    if vol is None:
        return (0.0, 1000.0)
    t_lo, t_hi = float(vol.twt.min()), float(vol.twt.max())
    if ties:
        lows, highs = [], []
        for tie in ties:
            good = np.isfinite(tie.ai) & (tie.ai > 0)
            if good.any():
                lows.append(float(tie.twt[good].min()))
                highs.append(float(tie.twt[good].max()))
        if lows:
            t_lo, t_hi = max(t_lo, min(lows)), min(t_hi, max(highs))
    if t_hi - t_lo < 50:
        t_lo, t_hi = float(vol.twt.min()), float(vol.twt.max())
    return (round(t_lo), round(t_hi))


def get_gate() -> tuple[float, float]:
    if st.session_state.gate is None:
        st.session_state.gate = default_gate()
    return st.session_state.gate


def invalidate(*keys: str) -> None:
    """Clear downstream products so the UI can never show a stale result."""
    for k in keys:
        st.session_state[k] = None


def kv_table(mapping: dict) -> pd.DataFrame:
    """Render a summary dict as a two-column table Streamlit can serialise.

    The summary dicts deliberately mix types -- counts as ints, formatted
    ranges as strings.  Transposing a one-row frame puts all of that in a
    single object column, and Arrow types a column from its first values, so a
    string column carrying an int raises ``ArrowTypeError``.  Streamlit falls
    back to a string cast and the table still renders, but it logs a full
    traceback each time.  Casting here keeps the display identical and the
    console clean.
    """
    rows = {str(k): ("" if v is None else str(v)) for k, v in mapping.items()}
    return pd.DataFrame({"value": pd.Series(rows, dtype="string")})


def show_error(exc: Exception, context: str) -> None:
    st.error(f"**{context}**\n\n{type(exc).__name__}: {exc}")
    with st.expander("Traceback"):
        st.code("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))


# ==========================================================================
# Sidebar
# ==========================================================================

def sidebar() -> str:
    st.sidebar.title("Post-stack inversion")
    step = st.sidebar.radio("Workflow", STEPS, key="step")

    st.sidebar.divider()
    st.sidebar.caption("**Session state**")
    vol = st.session_state.volume
    rows = [
        ("Volume", f"{vol.shape[0]} x {vol.shape[1]} x {vol.shape[2]}" if vol else "-"),
        ("Wells", str(len(st.session_state.wells)) or "-"),
        ("Ties", str(len(st.session_state.ties))),
        ("Wavelet", st.session_state.wavelet.kind if st.session_state.wavelet else "-"),
        ("Colour operator", "designed" if st.session_state.colour_operator else "-"),
        ("Low-freq model", st.session_state.lfm.method if st.session_state.lfm else "-"),
        ("Result", st.session_state.result.method if st.session_state.result else "-"),
    ]
    st.sidebar.dataframe(pd.DataFrame(rows, columns=["item", "status"]),
                         hide_index=True, width="stretch")

    if vol is not None:
        gate = get_gate()
        st.sidebar.caption("**Analysis gate (ms)**")
        new_gate = st.sidebar.slider(
            "Gate", float(vol.twt.min()), float(vol.twt.max()),
            value=(float(gate[0]), float(gate[1])), step=float(vol.sample_rate_ms),
            label_visibility="collapsed",
            help="Time window used for wavelet extraction, operator design and QC statistics.",
        )
        if new_gate != tuple(st.session_state.gate or ()):
            st.session_state.gate = new_gate

    if st.session_state.log:
        with st.sidebar.expander("Activity log"):
            st.code("\n".join(reversed(st.session_state.log[-20:])), language=None)

    st.sidebar.divider()
    if st.sidebar.button("Reset session", width="stretch"):
        for key in DEFAULTS:
            st.session_state.pop(key, None)
        st.rerun()

    return step


# ==========================================================================
# Step 1 -- Data
# ==========================================================================

def page_data() -> None:
    st.header("1 - Data")
    st.caption(
        "Load a 3D post-stack SEG-Y and the wells to calibrate it against. "
        "Wells are assumed already tied upstream; a bulk shift is available on the next step."
    )

    source = st.radio("Data source", ["Synthetic demo dataset", "Upload SEG-Y + LAS"],
                      horizontal=True)

    if source == "Synthetic demo dataset":
        _synthetic_loader()
    else:
        _segy_loader()
        _las_loader()

    vol, wells = st.session_state.volume, st.session_state.wells
    if vol is None:
        st.info("Load a volume to continue.")
        return

    st.divider()
    st.subheader("Volume")
    st.dataframe(kv_table(vol.summary()),
                 width="stretch")
    if vol.text_header:
        with st.expander("Text header"):
            st.code(vol.text_header[:3000], language=None)

    if wells:
        st.subheader("Wells")
        st.dataframe(pd.DataFrame([w.summary() for w in wells]), hide_index=True,
                     width="stretch")
        notes = [f"**{w.name}** - {n}" for w in wells for n in w.notes]
        if notes:
            with st.expander("Loading notes"):
                st.markdown("\n\n".join(notes))

        unlocated = [w.name for w in wells if not w.has_location]
        if unlocated:
            st.warning(
                f"No X/Y for: {', '.join(unlocated)}. These wells cannot be tied to the seismic "
                "and are excluded from wavelet extraction and the low-frequency model. "
                "Upload a well-header CSV below to locate them."
            )

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(viz.basemap_figure(vol, wells, st.session_state.ties),
                            width="stretch")
        with col_b:
            pick = st.selectbox("Well log display", [w.name for w in wells])
            chosen = next(w for w in wells if w.name == pick)
            st.plotly_chart(viz.well_log_figure(chosen), width="stretch")

    _aux_well_files()
    _optional_files()


def _synthetic_loader() -> None:
    st.markdown("A layered earth model with dip, a lateral impedance anomaly and tied pseudo-wells. "
                "The seismic is that model's reflectivity convolved with a Ricker wavelet, so the "
                "inversion has a known right answer.")
    c1, c2, c3, c4 = st.columns(4)
    n_il = c1.number_input("Inlines", 10, 200, 40, 5)
    n_xl = c2.number_input("Crosslines", 10, 200, 40, 5)
    n_s = c3.number_input("Samples", 100, 2000, 400, 50)
    n_w = c4.number_input("Wells", 1, 12, 4, 1)
    c5, c6, c7 = st.columns(3)
    dt_ms = c5.number_input("Sample rate (ms)", 1.0, 8.0, 2.0, 0.5)
    noise = c6.slider("Noise (fraction of RMS)", 0.0, 0.5, 0.08, 0.01)
    freq = c7.number_input("Wavelet peak frequency (Hz)", 5.0, 90.0, 28.0, 1.0)

    if st.button("Generate synthetic dataset", type="primary"):
        with st.spinner("Generating..."):
            vol, wells = data_io.make_synthetic_dataset(
                n_iline=int(n_il), n_xline=int(n_xl), n_samples=int(n_s),
                sample_rate_ms=float(dt_ms), n_wells=int(n_w), noise=float(noise),
                peak_frequency=float(freq),
            )
        st.session_state.volume = vol
        st.session_state.wells = wells
        st.session_state.segy_path = None
        st.session_state.gate = None
        invalidate("wavelet", "colour_operator", "lfm", "result")
        rebuild_ties("synthetic dataset generated")
        st.session_state.gate = default_gate()
        log(f"synthetic dataset {vol.shape} with {len(wells)} wells")
        flash(f"Generated {vol.shape[0]} x {vol.shape[1]} x {vol.shape[2]} with {len(wells)} wells.")


def _segy_loader() -> None:
    st.subheader("SEG-Y volume")

    mode = st.radio(
        "Source", ["Upload a file", "Path on this machine"], horizontal=True,
        help="Browser upload is capped at 1 GB and holds the file in memory while it "
             "transfers. For large volumes, pointing at a path on the machine running "
             "the app skips both.")

    uploaded = None
    local_path = ""
    if mode == "Upload a file":
        uploaded = st.file_uploader("3D post-stack SEG-Y (up to 1 GB)",
                                    type=["sgy", "segy", "SGY", "SEGY"])
    else:
        local_path = st.text_input(
            "Full path to the SEG-Y file",
            placeholder=r"C:\data\survey\poststack.sgy",
            help="Read directly from disk - no size limit and nothing held in memory twice.")

    st.markdown("**Trace header byte positions**")
    c1, c2, c3, c4 = st.columns(4)
    b_il = c1.number_input("Inline", 1, 240, data_io.DEFAULT_BYTES["iline"], 1)
    b_xl = c2.number_input("Crossline", 1, 240, data_io.DEFAULT_BYTES["xline"], 1)
    b_x = c3.number_input("CDP X", 0, 240, data_io.DEFAULT_BYTES["cdp_x"], 1,
                          help="0 to skip; a bin grid is synthesised instead.")
    b_y = c4.number_input("CDP Y", 0, 240, data_io.DEFAULT_BYTES["cdp_y"], 1)

    if mode == "Path on this machine":
        if not local_path:
            return
        if not os.path.isfile(local_path):
            st.error(f"No file at `{local_path}`. Check the path and that the app can reach it.")
            return
        path = local_path
        st.caption(f"{os.path.getsize(path) / 1e6:,.0f} MB on disk")
    else:
        if uploaded is None:
            return
        if st.session_state.get("_upload_name") != uploaded.name:
            with st.spinner("Writing the upload to disk..."):
                st.session_state._upload_path = data_io.persist_upload(uploaded)
            st.session_state._upload_name = uploaded.name
        path = st.session_state._upload_path

    with st.expander("Scan trace headers (helps identify non-standard byte positions)"):
        if st.button("Run header scan"):
            try:
                st.dataframe(data_io.scan_segy_headers(path), width="stretch", height=400)
            except Exception as exc:  # noqa: BLE001
                show_error(exc, "Header scan failed")

    if st.button("Load SEG-Y", type="primary"):
        try:
            with st.spinner("Loading volume (cached on file + byte config)..."):
                digest = data_io.file_digest(path, extra=f"{b_il}-{b_xl}-{b_x}-{b_y}")
                vol = data_io.load_segy_cached(path, (int(b_il), int(b_xl), int(b_x), int(b_y)), digest)
            st.session_state.volume = vol
            st.session_state.segy_path = path
            st.session_state.gate = None
            invalidate("wavelet", "colour_operator", "lfm", "result")
            rebuild_ties("SEG-Y loaded")
            st.session_state.gate = default_gate()
            log(f"loaded {os.path.basename(path)} {vol.shape}")
            flash(f"Loaded {vol.shape[0]} x {vol.shape[1]} x {vol.shape[2]} samples.")
        except Exception as exc:  # noqa: BLE001
            show_error(exc, "SEG-Y load failed - check the inline/crossline byte positions")


def _las_loader() -> None:
    st.subheader("Well logs (LAS)")
    files = st.file_uploader("One or more LAS files", type=["las", "LAS"], accept_multiple_files=True)

    st.caption(
        "Curve roles and units are detected automatically on load and can be corrected on "
        "**step 2 - Log QC**, so a well with unusual mnemonics still loads here.")

    c1, c2, c3 = st.columns(3)
    sonic_unit = c1.selectbox(
        "Sonic unit hint", data_io.SONIC_UNITS,
        help="Only consulted where the LAS unit is blank and the magnitude is ambiguous "
             "(slowness near 130-250). Correct any well individually on step 2.")
    const_vp = c2.number_input("Fallback Vp (m/s) if no sonic", 0.0, 8000.0, 0.0, 100.0,
                               help="0 disables the fallback.")
    repl_v = c3.number_input("Replacement velocity (m/s)", 500.0, 6000.0, 2500.0, 100.0,
                             help="Used only when the LAS has no time curve, to bridge KB to the datum.")

    if files and st.button("Load LAS files", type="primary"):
        wells, errors = [], []
        for f in files:
            try:
                wells.append(data_io.load_las(
                    f, name=os.path.splitext(f.name)[0], sonic_unit=sonic_unit,
                    constant_vp=const_vp if const_vp > 0 else None,
                    replacement_velocity=repl_v,
                ))
            except Exception as exc:  # noqa: BLE001 - report per file, keep the rest
                errors.append(f"**{f.name}**: {exc}")
        st.session_state.wells = wells
        invalidate("wavelet", "colour_operator", "lfm", "result")
        rebuild_ties("LAS files loaded")
        log(f"loaded {len(wells)} wells ({len(errors)} failed)")
        if errors:
            st.error("Some files could not be loaded:\n\n" + "\n\n".join(errors))
        elif wells:
            flagged = sum(1 for w in wells if any(sev == "error" for sev, _ in w.qc_flags()))
            note = f" {flagged} need attention on step 2 - Log QC." if flagged else ""
            flash(f"Loaded {len(wells)} wells.{note}")


def _aux_well_files() -> None:
    """Time-depth, deviation and marker files, routed by content."""
    st.divider()
    st.subheader("Well time-depth, deviation and markers")
    wells = st.session_state.wells
    if not wells:
        st.caption("Load wells first, then drop their auxiliary files here.")
        return

    st.markdown(
        "Drop **all** of a well's files in together - checkshots, deviation surveys and "
        "formation tops are told apart by their contents, and matched to a well by filename "
        "(`F02-1_TD.txt` finds well `F02-1`)."
    )
    with st.expander("Expected formats"):
        st.markdown(
            "| Kind | Columns | Example |\n| --- | --- | --- |\n"
            "| Time-depth / checkshot | `MD  time` | `553.6  0.544` |\n"
            "| Deviation survey | `X  Y  TVDSS  MD` | `606554  6080126  1665  1695` |\n"
            "| Markers / tops | `MD  name` | `1285.09  NMRF (Mid_Mio_Unc)` |\n\n"
            "Whitespace or tab separated, no header needed. Seconds against milliseconds and "
            "one-way against two-way are detected from the implied interval velocity; override "
            "below if a file is read wrongly."
        )

    c1, c2 = st.columns([2, 1])
    with c2:
        time_unit = st.selectbox(
            "Time-depth unit", ["auto"] + list(data_io.TIME_UNITS),
            help="'auto' picks seconds against milliseconds by magnitude, then one-way against "
                 "two-way by which gives a plausible interval velocity.")
        prefer_td = st.checkbox(
            "Adopt checkshot as the time source", value=True,
            help="A measured checkshot beats a LAS time curve and sonic integration.")
    with c1:
        files = st.file_uploader(
            "Time-depth, deviation and marker files",
            type=["txt", "csv", "dat", "track", "asc", "tz", "md"],
            accept_multiple_files=True, key="aux_well_files")

    if not files or not st.button("Attach well files", type="primary"):
        return

    names = [w.name for w in wells]
    applied, problems = [], []
    for handle in files:
        try:
            kind = data_io.sniff_well_file(handle)
            target = data_io.match_well_name(handle.name, names)
            if kind is None:
                problems.append(f"**{handle.name}**: could not identify the file type.")
                continue
            if target is None:
                problems.append(
                    f"**{handle.name}**: no well matched. Wells loaded: {', '.join(names)}.")
                continue

            well = next(w for w in wells if w.name == target)
            if kind == "time_depth":
                td = data_io.load_time_depth(handle, time_unit=time_unit, source=handle.name)
                message = well.attach_time_depth(td, prefer=prefer_td)
            elif kind == "track":
                message = well.attach_track(data_io.load_well_track(handle, source=handle.name))
            else:
                message = well.attach_markers(data_io.load_markers(handle, source=handle.name))
            applied.append(f"**{handle.name}** -> {target}: {message}")
        except Exception as exc:  # noqa: BLE001 - report per file, keep the rest
            problems.append(f"**{handle.name}**: {exc}")

    rebuild_ties("well auxiliary files attached")
    invalidate("wavelet", "colour_operator", "lfm", "result")

    if problems:
        # Rendered here rather than flashed, so the detail survives alongside
        # whatever did attach and the user can see both at once.
        st.error("\n\n".join(f"- {m}" for m in problems))
    if applied:
        log(f"attached {len(applied)} well files")
        detail = "\n\n".join(f"- {m}" for m in applied)
        if problems:
            st.success(detail)
        else:
            # flash() reruns, so the per-file detail has to travel with it or
            # it would be wiped by the redraw.
            flash(f"Attached {len(applied)} well files - check them on step 2 (Log QC).\n\n{detail}")


def _optional_files() -> None:
    st.divider()
    st.subheader("Optional: well headers and horizons")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Well header CSV** - `well, x, y, kb`")
        hdr = st.file_uploader("Well headers", type=["csv"], key="hdr_csv")
        if hdr is not None and st.button("Apply well headers"):
            try:
                table = data_io.load_well_headers(hdr)
                msgs = data_io.apply_well_headers(st.session_state.wells, table)
                st.session_state.headers = table
                rebuild_ties("well headers applied")
                invalidate("wavelet", "colour_operator", "lfm", "result")
                st.success("Headers applied.")
                st.write("\n".join(f"- {m}" for m in msgs))
            except Exception as exc:  # noqa: BLE001
                show_error(exc, "Could not read the well-header CSV")

    with c2:
        st.markdown("**Horizon CSV** - `iline, xline, <horizon columns in ms>`")
        hz = st.file_uploader("Horizons", type=["csv"], key="hz_csv")
        if hz is not None and st.session_state.volume is not None and st.button("Load horizons"):
            try:
                st.session_state.horizons = low_freq_model.load_horizon_csv(hz, st.session_state.volume)
                log(f"loaded horizons: {', '.join(st.session_state.horizons)}")
                st.success(f"Loaded horizons: {', '.join(st.session_state.horizons)}")
            except Exception as exc:  # noqa: BLE001
                show_error(exc, "Could not read the horizon CSV")


# ==========================================================================
# Step 2 -- Log QC and curve assignment
# ==========================================================================

def page_log_qc() -> None:
    st.header("2 - Log QC")
    wells = st.session_state.wells
    vol = st.session_state.volume
    if not wells:
        st.info("Load some wells on step 1 first.")
        return

    st.caption(
        "Check what was actually read from each LAS, and correct it. Curve roles and units "
        "are auto-detected on load, but mnemonics and unit strings vary enough between vendors "
        "that the guess is only a starting point -- a wrong sonic unit scales Vp by 3.28 or "
        "1000 and quietly ruins every impedance downstream."
    )

    names = [w.name for w in wells]
    pick = st.selectbox("Well", names, key="logqc_well")
    well = next(w for w in wells if w.name == pick)

    _log_qc_flags(well, vol)

    _aux_file_status(well)

    st.divider()
    st.subheader("Curve assignment")
    st.caption(f"Currently: {well.selection.describe()}")
    _curve_assignment_form(well, vol)

    st.divider()
    st.subheader("Curves in this file")
    inventory = pd.DataFrame(well.curve_inventory())
    st.dataframe(inventory, hide_index=True, width="stretch")

    preview = st.multiselect(
        "Preview raw curves", list(well.curves), max_selections=6,
        default=[c for c in (well.selection.sonic, well.selection.density) if c],
        help="Shown exactly as stored in the file, before any unit conversion.")
    if preview:
        st.plotly_chart(viz.curve_preview_figure(well, preview), width="stretch")

    st.divider()
    st.subheader("Converted logs")
    st.plotly_chart(viz.log_qc_figure(well), width="stretch")

    st.divider()
    st.subheader("All wells")
    st.dataframe(pd.DataFrame([{
        "well": w.name,
        "Vp curve": w.selection.sonic or (f"constant {w.selection.constant_vp:.0f}"
                                          if w.selection.constant_vp else "-"),
        "Vp unit": w.selection.sonic_unit if w.selection.sonic else "-",
        "density curve": w.selection.density or "-",
        "density unit": w.selection.density_unit if w.selection.density else "-",
        "time curve": w.selection.time or "integrated",
        "time unit": w.selection.time_unit if w.selection.time else "-",
        "usable samples": int(w.valid_mask().sum()),
        "issues": sum(1 for sev, _ in w.qc_flags(vol.twt if vol else None) if sev != "ok"),
    } for w in wells]), hide_index=True, width="stretch")


def _aux_file_status(well) -> None:
    """Show what the time-depth, deviation and marker files contributed."""
    if not (well.time_depth or well.track or well.markers):
        st.caption("No time-depth, deviation or marker file attached "
                   "(upload them on step 1).")
        return

    st.divider()
    st.subheader("Time-depth, deviation and markers")
    tabs = st.tabs(["Time-depth", "Deviation", "Markers"])

    with tabs[0]:
        if well.time_depth is None:
            st.caption("No checkshot attached; the time axis comes from the curve assignment below.")
        else:
            st.dataframe(kv_table(well.time_depth.summary()),
                         width="stretch")
            for message in well.time_depth.warnings():
                st.warning(message)
            in_use = well.selection.time == data_io.TD_SOURCE
            (st.success if in_use else st.warning)(
                "This checkshot is the well's time source."
                if in_use else
                "Attached but not in use - select "
                f"`{data_io.TD_SOURCE}` as the time curve below to adopt it.")
            st.plotly_chart(viz.time_depth_figure(well.time_depth, well.markers), width="stretch")

    with tabs[1]:
        if well.track is None:
            st.caption("No deviation survey attached.")
        else:
            st.dataframe(kv_table(well.track.summary()),
                         width="stretch")
            if not well.track.is_vertical:
                st.info(
                    f"This well deviates up to {well.track.max_deviation:.0f} m from its surface "
                    "location. v1 extracts the seismic trace at the **surface** position; for a "
                    "strongly deviated well that is not where the logs are.")

    with tabs[2]:
        if not well.markers:
            st.caption("No markers attached.")
        else:
            rows = [{"marker": m.name, "MD (m)": round(m.md, 2),
                     "TWT (ms)": round(m.twt, 1) if m.twt is not None else None}
                    for m in well.markers]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            _gate_from_markers(well)


def _gate_from_markers(well) -> None:
    """Let the user set the analysis gate from two formation tops."""
    timed = [m for m in well.markers if m.twt is not None]
    if len(timed) < 2:
        st.caption("At least two markers need a time before a gate can be set from them.")
        return
    names = [f"{m.name}  ({m.twt:.0f} ms)" for m in timed]
    c1, c2, c3 = st.columns([2, 2, 1])
    top = c1.selectbox("Gate top", names, index=0, key="gate_top")
    base = c2.selectbox("Gate base", names, index=len(names) - 1, key="gate_base")
    t0 = timed[names.index(top)].twt
    t1 = timed[names.index(base)].twt
    c3.markdown("&nbsp;")
    if c3.button("Set gate", width="stretch"):
        if t1 <= t0:
            st.error("The base marker must be deeper in time than the top.")
            return
        st.session_state.gate = (round(t0), round(t1))
        flash(f"Analysis gate set to {t0:.0f} - {t1:.0f} ms "
              f"({timed[names.index(top)].name} to {timed[names.index(base)].name}).")


def _log_qc_flags(well, vol) -> None:
    """Render the pass/warn/fail checks for one well."""
    flags = well.qc_flags(vol.twt if vol is not None else None)
    errors = [m for sev, m in flags if sev == "error"]
    warnings = [m for sev, m in flags if sev == "warning"]
    passes = [m for sev, m in flags if sev == "ok"]

    cols = st.columns(3)
    cols[0].metric("Failed checks", len(errors))
    cols[1].metric("Warnings", len(warnings))
    cols[2].metric("Passed", len(passes))

    for message in errors:
        st.error(message)
    for message in warnings:
        st.warning(message)
    if passes and not errors:
        with st.expander(f"{len(passes)} checks passed"):
            for message in passes:
                st.markdown(f"- {message}")


def _time_for(time_source, target, apply_all: bool, is_origin: bool):
    """Resolve the time source for one well when applying an assignment.

    Pushing a role onto another well only makes sense if that well can honour
    it: the checkshot option needs its own time-depth file, and a curve name
    needs that curve to exist.
    """
    if time_source == data_io.TD_SOURCE:
        return data_io.TD_SOURCE if target.time_depth is not None else None
    if time_source is None or time_source in target.curves:
        return time_source
    return None if (apply_all and not is_origin) else time_source


def _curve_assignment_form(well, vol) -> None:
    """Editor for the well name and the curve/unit assignment."""
    if not well.curves:
        st.warning(
            f"**{well.name}** carries no named curves, so there is nothing to reassign. "
            "Its Vp and density were supplied directly rather than read from a LAS.")
        return

    options = ["(none)"] + list(well.curves)
    time_options = list(options)
    if well.time_depth is not None:
        time_options.insert(1, data_io.TD_SOURCE)

    def index_of(mnemonic):
        return options.index(mnemonic) if mnemonic in options else 0

    with st.form(f"assign_{well.name}"):
        new_name = st.text_input(
            "Well name", value=well.name,
            help="LAS headers often carry a blank or inconsistent well name. "
                 "This name is what appears on sections, tables and the base map.")

        c1, c2 = st.columns(2)
        sonic = c1.selectbox("Vp / sonic curve", options, index=index_of(well.selection.sonic))
        sonic_unit = c2.selectbox(
            "Vp unit", data_io.SONIC_UNITS,
            index=data_io.SONIC_UNITS.index(well.selection.sonic_unit)
            if well.selection.sonic_unit in data_io.SONIC_UNITS else 0,
            help="us/ft and us/m are slowness; m/s and ft/s are for a curve that already holds "
                 "velocity rather than sonic travel time.")

        c3, c4 = st.columns(2)
        density = c3.selectbox("Density curve", options, index=index_of(well.selection.density))
        density_unit = c4.selectbox(
            "Density unit", data_io.DENSITY_UNITS,
            index=data_io.DENSITY_UNITS.index(well.selection.density_unit)
            if well.selection.density_unit in data_io.DENSITY_UNITS else 0)

        c5, c6 = st.columns(2)
        time_curve = c5.selectbox(
            "Time curve", time_options,
            index=(time_options.index(well.selection.time)
                   if well.selection.time in time_options else 0),
            help="Leave as (none) to integrate the sonic instead. Wells are assumed tied "
                 "upstream, so a TWT curve in the LAS is preferred where one exists.")
        time_unit = c6.selectbox(
            "Time unit", data_io.TIME_UNITS,
            index=data_io.TIME_UNITS.index(well.selection.time_unit)
            if well.selection.time_unit in data_io.TIME_UNITS else 0,
            help="OWT is doubled to two-way time on load.")

        c7, c8 = st.columns(2)
        constant_vp = c7.number_input(
            "Fallback constant Vp (m/s)", 0.0, 8000.0,
            float(well.selection.constant_vp or 0.0), 100.0,
            help="Used only when no Vp curve is assigned. 0 disables it.")
        repl_v = c8.number_input(
            "Replacement velocity (m/s)", 500.0, 6000.0, 2500.0, 100.0,
            help="Bridges KB to the seismic datum when the time axis is integrated from sonic.")

        apply_all = st.checkbox(
            "Apply this assignment to every well",
            help="Only affects wells that actually contain the named curves.")
        submitted = st.form_submit_button("Apply assignment", type="primary")

    if not submitted:
        return

    selection = data_io.CurveSelection(
        sonic=None if sonic == "(none)" else sonic,
        sonic_unit=sonic_unit,
        density=None if density == "(none)" else density,
        density_unit=density_unit,
        time=None if time_curve == "(none)" else time_curve,
        time_unit=time_unit,
        constant_vp=constant_vp if constant_vp > 0 else None,
    )

    if (selection.sonic is None and not selection.constant_vp) or selection.density is None:
        st.error(
            "Assign both a Vp source (a curve, or a constant) and a density curve before "
            "applying -- otherwise this would clear the logs this well already has.")
        return

    targets = st.session_state.wells if apply_all else [well]
    applied, skipped = [], []
    for target in targets:
        if target is well and new_name.strip() and new_name.strip() != well.name:
            well.name = new_name.strip()
        # Only push a role onto another well if it really has that curve.
        local = data_io.CurveSelection(
            sonic=selection.sonic if selection.sonic in target.curves else (
                None if apply_all and target is not well else selection.sonic),
            sonic_unit=selection.sonic_unit,
            density=selection.density if selection.density in target.curves else (
                None if apply_all and target is not well else selection.density),
            density_unit=selection.density_unit,
            time=_time_for(selection.time, target, apply_all, target is well),
            time_unit=selection.time_unit,
            constant_vp=selection.constant_vp,
        )
        target.apply_selection(local, replacement_velocity=repl_v)
        (applied if target.valid_mask().any() else skipped).append(target.name)

    rebuild_ties("curve assignment changed")
    invalidate("wavelet", "colour_operator", "lfm", "result")
    message = f"Assignment applied to {', '.join(applied)}." if applied else "Assignment applied."
    if skipped:
        message += f" No usable samples for: {', '.join(skipped)}."
    flash(message)


# ==========================================================================
# Step 3 -- Well tie QC
# ==========================================================================

def page_well_tie() -> None:
    st.header("3 - Well tie QC")
    vol, wells, ties = st.session_state.volume, st.session_state.wells, st.session_state.ties
    if vol is None or not wells:
        st.info("Load a volume and at least one well first.")
        return
    if not ties:
        st.warning("No well has an X/Y location, so no seismic trace can be extracted. "
                   "Upload a well-header CSV on step 1.")
        return

    st.caption(
        "This step *verifies* the tie rather than building one: the time-depth relationship is "
        "assumed to come from upstream. A constant bulk shift is available for a datum error; "
        "stretch and squeeze are out of scope for v1."
    )

    gate = get_gate()
    c1, c2 = st.columns([1, 2])
    with c1:
        k = st.number_input("Traces blended per well (IDW)", 1, 16, st.session_state.k_neighbours, 1,
                            help="Nearest live traces around the well, inverse-distance weighted.")
        if k != st.session_state.k_neighbours:
            st.session_state.k_neighbours = int(k)
            rebuild_ties("neighbour count changed")
            st.rerun()

    qc_wavelet = st.session_state.wavelet
    if qc_wavelet is None:
        with c2:
            st.info("No wavelet yet - QC uses a provisional Ricker. Build a wavelet on step 4 "
                    "and return here to see the real fit.")
        freq = st.slider("Provisional Ricker frequency (Hz)", 5, 80, 25)
        qc_samples = wvl.ricker(freq, 128.0, vol.dt)
    else:
        qc_samples = qc_wavelet.samples

    st.subheader("Tie score table")
    table = viz.tie_score_table(ties, qc_samples, gate[0], gate[1])
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.caption(f"Statistics computed over the analysis gate {gate[0]:.0f} - {gate[1]:.0f} ms. "
               "A large 'best lag' means the well is shifted relative to the seismic.")

    st.divider()
    st.subheader("Per-well QC")
    names = [t.well for t in ties]
    pick = st.selectbox("Well", names)
    tie = next(t for t in ties if t.well == pick)
    well = next(w for w in wells if w.name == pick)

    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("**Bulk shift**")
        row = next((r for r in table if r["well"] == pick), None)
        if row:
            st.metric("Suggested shift", f"{row['best lag (ms)']:+.1f} ms",
                      help="Lag of the cross-correlation peak between the extracted trace and "
                           "the synthetic. Applying it lines the two up.")
        shift = st.number_input("Applied shift (ms)", -200.0, 200.0, float(well.bulk_shift), 1.0)
        b1, b2 = st.columns(2)
        if b1.button("Apply", width="stretch"):
            well.set_bulk_shift(shift)
            rebuild_ties(f"{pick} bulk shift {shift:+.1f} ms")
            invalidate("wavelet", "colour_operator", "lfm", "result")
            st.rerun()
        if b2.button("Use suggested", width="stretch") and row:
            well.set_bulk_shift(well.bulk_shift + row["best lag (ms)"])
            rebuild_ties(f"{pick} bulk shift {well.bulk_shift:+.1f} ms")
            invalidate("wavelet", "colour_operator", "lfm", "result")
            st.rerun()
        st.caption("Changing a shift clears the wavelet, model and results, since all of them "
                   "depend on the tie.")

    with c1:
        st.plotly_chart(
            viz.well_tie_figure(tie, qc_samples, gate[0], gate[1],
                                markers=well.markers_in_range(gate[0], gate[1])),
            width="stretch")


# ==========================================================================
# Step 4 -- Wavelet
# ==========================================================================

def page_wavelet() -> None:
    st.header("4 - Wavelet")
    vol, ties = st.session_state.volume, st.session_state.ties
    if vol is None:
        st.info("Load a volume first.")
        return

    gate = get_gate()
    method = st.radio(
        "Estimation method",
        ["Parametric", "Statistical (from seismic spectrum)", "Well-based (least squares)"],
        horizontal=True,
        help="Parametric and statistical assert a phase; only well-based measures it.",
    )

    c1, c2, c3 = st.columns(3)
    length_ms = c1.slider("Wavelet length (ms)", 40, 400, 128, 4)
    taper = c2.slider("Taper fraction", 0.0, 0.8, 0.15, 0.05)
    phase = c3.selectbox("Phase rotation", list(wvl.PHASE_PRESETS), index=0)
    phase_deg = wvl.PHASE_PRESETS[phase]

    cfg: dict = {"length_ms": float(length_ms), "taper": float(taper), "phase": float(phase_deg)}

    if method == "Parametric":
        cfg["method"] = "parametric"
        kind = st.selectbox("Family", list(wvl.WAVELET_TYPES))
        cfg["kind"] = kind
        params: dict = {}
        if kind == "ricker":
            params["freq"] = st.slider("Peak frequency (Hz)", 5.0, 90.0, 25.0, 1.0)
        elif kind == "ormsby":
            p1, p2, p3, p4 = st.columns(4)
            params["f1"] = p1.number_input("f1 (Hz)", 1.0, 100.0, 5.0, 1.0)
            params["f2"] = p2.number_input("f2 (Hz)", 1.0, 120.0, 10.0, 1.0)
            params["f3"] = p3.number_input("f3 (Hz)", 1.0, 200.0, 45.0, 1.0)
            params["f4"] = p4.number_input("f4 (Hz)", 1.0, 250.0, 60.0, 1.0)
        else:
            p1, p2, p3 = st.columns(3)
            params["f_low"] = p1.number_input("Low cut (Hz)", 1.0, 100.0, 8.0, 1.0)
            params["f_high"] = p2.number_input("High cut (Hz)", 2.0, 200.0, 55.0, 1.0)
            params["order"] = p3.number_input("Order", 1, 10, 4, 1)
        cfg.update(params)

    elif method.startswith("Statistical"):
        cfg["method"] = "statistical"
        cfg["max_traces"] = st.slider("Traces sampled", 50, 5000, 500, 50)
        cfg["t_min"], cfg["t_max"] = gate
        st.caption(f"Extracted over the analysis gate {gate[0]:.0f} - {gate[1]:.0f} ms. "
                   "The amplitude spectrum comes from the data; the phase above is asserted.")

    else:
        cfg["method"] = "well-based"
        cfg["twt"] = vol.twt
        cfg["t_min"], cfg["t_max"] = gate
        cfg["prewhitening"] = st.slider("Prewhitening (%)", 0.1, 20.0, 1.0, 0.1,
                                        help="Stabilises the spectral division where the "
                                             "reflectivity spectrum has notches.")
        if not ties:
            st.warning("Well-based extraction needs at least one located, tied well.")

    calibrate = st.checkbox(
        "Calibrate wavelet amplitude against the wells", value=True,
        help="Scales the wavelet so reflectivity * wavelet matches the extracted seismic. "
             "Required for sparse-spike and model-based inversion to recover the correct "
             "impedance contrast - a peak-normalised wavelet gives mis-scaled reflectivity.",
    )

    if st.button("Estimate wavelet", type="primary"):
        try:
            with st.spinner("Estimating..."):
                wav = wvl.wavelet_from_config(cfg, vol.dt, volume=vol, ties=ties)
                if calibrate and ties:
                    wav = wvl.calibrate_amplitude(wav, ties, vol.twt, gate[0], gate[1])
                elif calibrate:
                    st.warning("No located well - amplitude could not be calibrated. "
                               "Absolute impedance will be unreliable.")
            st.session_state.wavelet = wav
            invalidate("result")
            log(f"wavelet: {wav.kind}, {wav.length_ms:.0f} ms, {wav.dominant_frequency():.1f} Hz")
            flash("Wavelet estimated.")
        except Exception as exc:  # noqa: BLE001
            show_error(exc, "Wavelet estimation failed")

    wav = st.session_state.wavelet
    if wav is None:
        st.info("No wavelet yet.")
        return

    st.divider()
    st.subheader("Wavelet QC")
    st.dataframe(kv_table(wav.summary()),
                 width="stretch")
    for note in getattr(wav, "notes", []):
        st.warning(note)
    st.plotly_chart(viz.wavelet_figure(wav), width="stretch")

    if ties:
        spectra = {"wavelet": wav.spectrum()}
        flat = vol.flat_data()
        live = np.flatnonzero(vol.live_mask())
        if live.size:
            pick = live[:: max(live.size // 300, 1)]
            spectra["seismic"] = utils.average_amplitude_spectrum(flat[pick], vol.dt)
        refl = [np.nan_to_num(t.reflectivity) for t in ties]
        if refl:
            spec = [utils.amplitude_spectrum(r, vol.dt)[1] for r in refl]
            freq = utils.amplitude_spectrum(refl[0], vol.dt)[0]
            spectra["well reflectivity"] = (freq, np.mean(np.stack(spec), axis=0))
        st.plotly_chart(
            viz.spectrum_comparison_figure(spectra, max_freq=min(0.5 / vol.dt, 120),
                                           title="Wavelet against seismic and well reflectivity"),
            width="stretch")


# ==========================================================================
# Step 5 -- Low-frequency model
# ==========================================================================

def page_low_freq() -> None:
    st.header("5 - Low-frequency model")
    vol, wells, ties = st.session_state.volume, st.session_state.wells, st.session_state.ties
    if vol is None or not wells:
        st.info("Load a volume and wells first.")
        return

    st.caption(
        "Well impedance, low-pass filtered and interpolated between wells. Model-based inversion "
        "requires this; coloured and sparse-spike use it to turn relative impedance into absolute."
    )

    located = [w for w in wells if w.has_location]
    if not located:
        st.warning("No well has an X/Y location. Upload a well-header CSV on step 1.")
        return

    c1, c2, c3 = st.columns(3)
    cutoff = c1.slider("Low-pass cutoff (Hz)", 2.0, 25.0, 10.0, 0.5,
                       help="Should sit at or below the lowest frequency the seismic carries.")
    method = c2.selectbox("Lateral interpolation", list(low_freq_model.INTERP_METHODS))
    power = c3.slider("IDW power", 0.5, 6.0, 2.0, 0.5, disabled=(method != "idw"))

    c4, c5, c6 = st.columns(3)
    lat_smooth = c4.slider("Lateral smoothing (bins)", 0.0, 10.0, 2.0, 0.5)
    vert_smooth = c5.slider("Vertical smoothing (ms)", 0.0, 100.0, 0.0, 5.0)
    smoothing = c6.slider("RBF smoothing", 0.0, 10.0, 0.0, 0.5, disabled=(method != "rbf"))

    use_horizons = False
    if st.session_state.horizons:
        use_horizons = st.checkbox(
            f"Guide with horizons ({', '.join(st.session_state.horizons)})", value=True,
            help="Interpolates in horizon-flattened time so the trend follows structure.")
    else:
        st.caption("No horizons loaded - interpolation is done on flat time slices, which is only "
                   "defensible where structure is gentle. Load horizons on step 1 to improve this.")

    if st.button("Build low-frequency model", type="primary"):
        bar = st.progress(0.0, text="Interpolating...")
        try:
            model = low_freq_model.build_low_frequency_model(
                vol, wells, cutoff_hz=float(cutoff), method=method, power=float(power),
                smoothing=float(smoothing), lateral_smooth_bins=float(lat_smooth),
                vertical_smooth_ms=float(vert_smooth),
                horizons=st.session_state.horizons if use_horizons else None,
                progress=lambda f: bar.progress(min(f, 1.0), text=f"Interpolating... {f * 100:.0f}%"),
            )
            bar.empty()
            st.session_state.lfm = model
            invalidate("result")
            log(f"low-frequency model: {method}, {cutoff:.1f} Hz, {len(model.wells_used)} wells")
            flash("Model built.")
        except Exception as exc:  # noqa: BLE001
            bar.empty()
            show_error(exc, "Could not build the low-frequency model")

    model = st.session_state.lfm
    if model is None:
        return

    st.divider()
    st.subheader("Model QC")
    st.dataframe(kv_table(model.summary()),
                 width="stretch")
    for note in model.notes:
        st.caption(f"- {note}")

    if ties:
        st.plotly_chart(viz.low_freq_qc_figure(model, ties), width="stretch")

    st.markdown("**Section view**")
    orientation, index, axis_vals = _section_controls(vol, key="lfm")
    st.plotly_chart(
        viz.section_figure(model.ai, model.twt, axis_vals, orientation, index,
                           title=f"Background AI - {orientation} "
                                 f"{(vol.iline if orientation == 'inline' else vol.xline)[index]}",
                           colorscale=viz.IMPEDANCE_SCALE, symmetric=False,
                           colorbar_title="AI",
                           wells=viz.well_overlay_positions(ties, orientation, index, vol)),
        width="stretch")


def _section_controls(vol, key: str, default_orientation: str = "inline"):
    """Shared orientation + line-number picker; returns ``(orientation, index, axis)``."""
    c1, c2 = st.columns([1, 3])
    orientation = c1.selectbox("Orientation", ["inline", "crossline"],
                               index=0 if default_orientation == "inline" else 1, key=f"{key}_orient")
    lines = vol.iline if orientation == "inline" else vol.xline
    line = c2.select_slider(f"{orientation} number", options=[int(v) for v in lines],
                            value=int(lines[len(lines) // 2]), key=f"{key}_line")
    index = int(np.argmin(np.abs(np.asarray(lines) - line)))
    axis_vals = vol.xline if orientation == "inline" else vol.iline
    return orientation, index, axis_vals


# ==========================================================================
# Step 6 -- Inversion
# ==========================================================================

class BackgroundJob:
    """A full-volume run on a worker thread.

    Streamlit re-runs the script on every interaction, so a long inversion in
    the main thread would freeze the page.  The worker writes progress into a
    plain dict (never into ``session_state``, which is not thread-safe) and the
    page polls it, so the sidebar and the stop button stay live throughout.
    """

    def __init__(self, fn, **kwargs) -> None:
        self.progress = {"fraction": 0.0, "message": "starting...", "done": False}
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.started = time.time()

        def _progress(fraction: float, message: str) -> None:
            self.progress["fraction"] = float(fraction)
            self.progress["message"] = message

        self.future = self._executor.submit(fn, progress=_progress, **kwargs)

    @property
    def running(self) -> bool:
        return not self.future.done()

    def result(self):
        try:
            return self.future.result()
        finally:
            self._executor.shutdown(wait=False)


def page_inversion() -> None:
    st.header("6 - Inversion")
    vol, ties, wav, model = (st.session_state.volume, st.session_state.ties,
                             st.session_state.wavelet, st.session_state.lfm)
    if vol is None:
        st.info("Load a volume first.")
        return

    gate = get_gate()
    method_label = st.radio(
        "Method", ["Coloured", "Sparse-spike", "Model-based"], horizontal=True,
        captions=["Fast, wavelet-free, relative impedance",
                  "L1 deconvolution, sparse reflectivity",
                  "Constrained fit to a background model"],
    )
    method = {"Coloured": "coloured", "Sparse-spike": "sparse-spike",
              "Model-based": "model-based"}[method_label]

    params: dict = {"merge_freq": 10.0}
    ready, blockers = True, []

    if method == "coloured":
        params, ready, blockers = _coloured_controls(vol, ties, params)
    elif method == "sparse-spike":
        c1, c2, c3 = st.columns(3)
        params["sparsity"] = c1.slider("Sparsity weight", 0.01, 5.0, 0.15, 0.01,
                                       help="Higher gives fewer, larger spikes and a looser data fit.")
        params["n_iter"] = c2.slider("IRLS iterations", 2, 40, 12, 1)
        params["merge_freq"] = c3.slider("Merge frequency (Hz)", 2.0, 25.0, 10.0, 0.5,
                                         help="Where the inverted band is spliced onto the model.")
        if wav is None:
            ready, blockers = False, blockers + ["a wavelet (step 4)"]
        if model is None:
            st.info("Without a low-frequency model the output is relative impedance only.")
    else:
        c1, c2, c3 = st.columns(3)
        params["model_weight"] = c1.slider("Model constraint weight", 0.0, 2.0, 0.10, 0.01,
                                           help="Higher pulls the answer toward the background model.")
        params["roughness_weight"] = c2.slider("Smoothness weight", 0.0, 1.0, 0.02, 0.01)
        params["max_iter"] = c3.slider("Max iterations", 10, 300, 60, 10)
        c4, c5 = st.columns(2)
        params["max_change"] = c4.slider("Max log-AI departure from model", 0.05, 1.5, 0.35, 0.05,
                                         help="Hard bound per sample; 0.35 is roughly +/-42% in impedance.")
        params["merge_freq"] = c5.slider("Merge frequency (Hz)", 2.0, 25.0, 10.0, 0.5)
        if wav is None:
            ready, blockers = False, blockers + ["a wavelet (step 4)"]
        if model is None:
            ready, blockers = False, blockers + ["a low-frequency model (step 5)"]

    if not ready:
        st.warning("This method still needs: " + ", ".join(blockers) + ".")
        return

    wav_samples = wav.samples if wav is not None else None

    st.divider()
    st.subheader("Preview on a subset")
    st.caption("Invert a small block first. Coloured runs in seconds on a full volume; "
               "sparse-spike and model-based do not.")

    n_il, n_xl = vol.shape[0], vol.shape[1]
    c1, c2 = st.columns(2)
    il_lo, il_hi = c1.slider("Inline index range", 0, n_il, (max(0, n_il // 2 - 5), min(n_il, n_il // 2 + 5)))
    xl_lo, xl_hi = c2.slider("Crossline index range", 0, n_xl, (0, n_xl))

    c1, c2, c3 = st.columns(3)
    if c1.button("Run preview", type="primary", width="stretch"):
        _run_inline(vol, method, wav_samples, model, params,
                    il_range=(il_lo, il_hi), xl_range=(xl_lo, xl_hi), label="preview")

    if c2.button("Estimate full-volume runtime", width="stretch"):
        with st.spinner("Timing a few traces..."):
            secs = inversion.estimate_runtime(vol, method, wav_samples, model, **params)
        st.info(f"Estimated full volume ({vol.n_traces:,} traces): "
                f"**{secs:.0f} s** ({secs / 60:.1f} min).")

    if c3.button("Run full volume", width="stretch"):
        st.session_state.job = BackgroundJob(
            inversion.run_volume, volume=vol, method=method, wavelet=wav_samples,
            low_freq_model=model, **params)
        log(f"full-volume {method} started")
        st.rerun()

    _poll_job()
    _show_single_trace_qc(vol, ties, method, wav_samples, model, params, gate)


def _coloured_controls(vol, ties, params):
    """Design (and calibrate) the coloured operator, then expose it as a param."""
    gate = get_gate()
    c1, c2, c3 = st.columns(3)
    nyq = 0.5 / vol.dt
    f_low = c1.number_input("Design band low (Hz)", 1.0, float(nyq) - 1, 8.0, 1.0)
    f_high = c2.number_input("Design band high (Hz)", 2.0, float(nyq), min(60.0, nyq - 1), 1.0)
    op_len = c3.slider("Operator length (ms)", 60, 600, 200, 20)
    c4, c5 = st.columns(2)
    white = c4.slider("White noise (%)", 0.1, 20.0, 2.0, 0.1,
                      help="Stabilises the spectral division; higher is safer but flatter.")
    params["merge_freq"] = c5.slider("Merge frequency (Hz)", 2.0, 25.0, 10.0, 0.5)

    if not ties:
        st.warning("Coloured inversion fits its target spectrum to well reflectivity, so it needs "
                   "at least one located, tied well.")
        return params, False, ["a located, tied well (step 1)"]

    if st.button("Design operator"):
        try:
            with st.spinner("Designing..."):
                op = inversion.design_colour_operator(
                    vol, ties, f_low=float(f_low), f_high=float(f_high),
                    operator_length_ms=float(op_len), white_noise_pct=float(white))
                op = inversion.calibrate_colour_operator(op, vol, ties, gate[0], gate[1])
            st.session_state.colour_operator = op
            invalidate("result")
            log(f"colour operator designed (beta={op.exponent:+.3f})")
            flash("Operator designed.")
        except Exception as exc:  # noqa: BLE001
            show_error(exc, "Operator design failed")

    op = st.session_state.colour_operator
    if op is None:
        return params, False, ["a designed operator (button above)"]

    st.dataframe(kv_table(op.summary()), width="stretch")
    st.plotly_chart(viz.colour_operator_figure(op), width="stretch")
    params["operator"] = op
    return params, True, []


def _run_inline(vol, method, wav_samples, model, params, il_range, xl_range, label: str) -> None:
    """Foreground run for the (small) preview block, with a progress bar."""
    bar = st.progress(0.0, text="Inverting...")
    try:
        result = inversion.run_volume(
            vol, method, wav_samples, model, il_range=il_range, xl_range=xl_range,
            progress=lambda f, m: bar.progress(min(f, 1.0), text=f"Inverting... {m}"), **params)
        bar.empty()
        st.session_state.result = result
        log(f"{label} {method}: {result.summary()['traces inverted']} traces in {result.elapsed_s:.1f}s")
        flash(f"{label.capitalize()} complete in {result.elapsed_s:.1f} s. "
              "See step 7 for sections, crossplot and export.")
    except Exception as exc:  # noqa: BLE001
        bar.empty()
        show_error(exc, f"{label.capitalize()} failed")


def _poll_job() -> None:
    """Render the background job's progress and collect its result when done."""
    job = st.session_state.job
    if job is None:
        return

    if job.running:
        p = job.progress
        st.progress(min(p["fraction"], 1.0),
                    text=f"Full volume: {p['message']}  ({time.time() - job.started:.0f} s elapsed)")
        st.caption("Running on a worker thread - the rest of the app stays usable.")
        time.sleep(0.6)
        st.rerun()
        return

    st.session_state.job = None
    try:
        result = job.result()
    except Exception as exc:  # noqa: BLE001
        show_error(exc, "Full-volume run failed")
        return
    st.session_state.result = result
    log(f"full volume complete in {result.elapsed_s:.1f}s")
    flash(f"Full volume complete in {result.elapsed_s:.1f} s. "
          "See step 7 for sections, crossplot and export.")


def _show_single_trace_qc(vol, ties, method, wav_samples, model, params, gate) -> None:
    """Invert one trace at a chosen well and show the fit -- the cheapest QC there is."""
    if not ties:
        return
    st.divider()
    st.subheader("Single-trace QC at a well")
    pick = st.selectbox("Well", [t.well for t in ties], key="qc_well")
    tie = next(t for t in ties if t.well == pick)
    trace = vol.trace_at(tie.il_index, tie.xl_index)
    lf_trace = model.trace(tie.il_index, tie.xl_index) if model is not None else None

    try:
        res = inversion.invert(trace, wav_samples, lf_trace, method=method,
                               dt=vol.dt, **params)
    except Exception as exc:  # noqa: BLE001
        show_error(exc, "Single-trace inversion failed")
        return

    st.plotly_chart(
        viz.trace_comparison_figure(res, vol.twt, trace, lf_trace, tie.ai),
        width="stretch")

    cols = st.columns(4)
    cols[0].metric("Correlation", f"{res['correlation']:.3f}")
    cols[1].metric("Misfit", "n/a" if not np.isfinite(res["misfit"]) else f"{res['misfit']:.3f}")
    if res.get("absolute_ai") is not None:
        good = np.isfinite(tie.ai) & (tie.ai > 0)
        i0 = int(np.searchsorted(vol.twt, gate[0]))
        i1 = int(np.searchsorted(vol.twt, gate[1]))
        sel = np.zeros_like(good)
        sel[i0:i1] = True
        sel &= good
        if sel.any():
            cols[2].metric("AI vs well (r)",
                           f"{utils.normalised_correlation(res['absolute_ai'][sel], tie.ai[sel]):.3f}")
    if "iterations" in res:
        cols[3].metric("Iterations", res["iterations"])


# ==========================================================================
# Step 7 -- Results and export
# ==========================================================================

def page_results() -> None:
    st.header("7 - Results & export")
    vol, ties, result, model = (st.session_state.volume, st.session_state.ties,
                                st.session_state.result, st.session_state.lfm)
    if vol is None or result is None:
        st.info("Run an inversion on step 6 first.")
        return

    st.dataframe(kv_table(result.summary()),
                 width="stretch")
    if result.is_subset:
        st.warning("This is a subset (preview) run. Sections below cover only the inverted block.")

    available = {"Relative impedance": result.relative_ai, "Reflectivity": result.reflectivity}
    if result.absolute_ai is not None:
        available = {"Absolute impedance": result.absolute_ai, **available}
    if result.method != "coloured":
        available["Residual"] = result.residual
    if model is not None and not result.is_subset:
        available["Low-frequency model"] = model.ai

    st.divider()
    st.subheader("Section viewer")
    which = st.selectbox("Attribute", list(available))
    cube = available[which]
    is_amplitude_like = which in ("Reflectivity", "Residual", "Relative impedance")

    sub_vol = _subset_volume(vol, result)
    orientation, index, axis_vals = _section_controls(sub_vol, key="res")
    wells_overlay = viz.well_overlay_positions(
        _shift_ties(ties, result), orientation, index, sub_vol)

    st.plotly_chart(
        viz.dual_section_figure(
            sub_vol.data, cube, sub_vol.twt, axis_vals, orientation, index,
            titles=("Seismic amplitude", which),
            impedance_scale=viz.SEISMIC_SCALE if is_amplitude_like else viz.IMPEDANCE_SCALE,
            wells=wells_overlay),
        width="stretch")

    with st.expander("Time slice"):
        t_ms = st.slider("TWT (ms)", float(vol.twt.min()), float(vol.twt.max()),
                         float(vol.twt[len(vol.twt) // 2]), float(vol.sample_rate_ms))
        st.plotly_chart(
            viz.time_slice_figure(
                cube, sub_vol.twt, sub_vol.iline, sub_vol.xline, t_ms, title=f"{which} at {t_ms:.0f} ms",
                colorscale=viz.SEISMIC_SCALE if is_amplitude_like else viz.IMPEDANCE_SCALE,
                symmetric=is_amplitude_like,
                wells=[{"name": t.well, "iline": int(vol.iline[t.il_index]),
                        "xline": int(vol.xline[t.xl_index])} for t in ties]),
            width="stretch")

    with st.expander("Per-trace QC maps"):
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(viz.qc_map_figure(result.correlation, sub_vol.iline, sub_vol.xline,
                                              "Correlation (synthetic vs seismic)"),
                            width="stretch")
        with c2:
            if np.isfinite(result.misfit).any():
                st.plotly_chart(viz.qc_map_figure(result.misfit, sub_vol.iline, sub_vol.xline,
                                                  "Normalised RMS misfit"), width="stretch")
            else:
                st.caption("Coloured inversion forms no wavelet synthetic, so there is no data misfit "
                           "to map. The correlation map compares the shaped trace to the input.")

    _crossplot_section(vol, ties, result)
    _export_section(vol, result, model)


def _subset_volume(vol, result):
    """A view of the seismic matching the inverted block, so sections line up."""
    if not result.is_subset:
        return vol
    il_sl, xl_sl = result.il_slice, result.xl_slice
    return data_io.SeismicVolume(
        data=vol.data[il_sl, xl_sl, :], iline=vol.iline[il_sl], xline=vol.xline[xl_sl],
        twt=vol.twt, cdp_x=vol.cdp_x[il_sl, xl_sl], cdp_y=vol.cdp_y[il_sl, xl_sl],
        source=vol.source)


def _shift_ties(ties, result):
    """Re-index the ties into the subset's coordinate frame for overlays."""
    if not result.is_subset:
        return ties
    il0 = result.il_slice.start or 0
    xl0 = result.xl_slice.start or 0
    out = []
    for t in ties:
        i, j = t.il_index - il0, t.xl_index - xl0
        if 0 <= i < result.relative_ai.shape[0] and 0 <= j < result.relative_ai.shape[1]:
            shifted = data_io.WellTie(**{**t.__dict__})
            shifted.il_index, shifted.xl_index = i, j
            out.append(shifted)
    return out


def _crossplot_section(vol, ties, result) -> None:
    st.divider()
    st.subheader("Crossplot QC - inverted vs well impedance")
    if not ties:
        st.caption("No located wells, so there is nothing to crossplot against.")
        return

    gate = get_gate()
    use_abs = st.checkbox("Use absolute impedance", value=result.absolute_ai is not None,
                          disabled=result.absolute_ai is None)
    crossplot = inversion.crossplot_at_wells(result, vol, ties, use_absolute=use_abs,
                                             t_min=gate[0], t_max=gate[1])
    if not crossplot:
        st.warning("No well falls inside the inverted block (or the gate contains no valid log "
                   "samples). Widen the subset on step 6 or the gate in the sidebar.")
        return

    missing = [t.well for t in ties if t.well not in crossplot]
    if missing:
        st.caption(f"Outside the inverted block or gate: {', '.join(missing)}")
    st.plotly_chart(
        viz.crossplot_figure(crossplot,
                             y_label="Inverted AI" if use_abs else "Inverted relative impedance"),
        width="stretch")


def _export_section(vol, result, model) -> None:
    st.divider()
    st.subheader("Export")

    options = {"Relative impedance": result.relative_ai, "Reflectivity": result.reflectivity}
    if result.absolute_ai is not None:
        options = {"Absolute impedance": result.absolute_ai, **options}
    if model is not None and not result.is_subset:
        options["Low-frequency model"] = model.ai

    c1, c2 = st.columns(2)
    which = c1.selectbox("Cube to export", list(options), key="export_cube")
    fmt = c2.selectbox("Format", ["SEG-Y", "NumPy (.npy)", "NetCDF (xarray)"])
    cube = options[which]

    slug = which.lower().replace(" ", "_")
    sub_vol = _subset_volume(vol, result)

    if st.button("Prepare download", type="primary"):
        try:
            with st.spinner("Writing..."):
                payload, filename, mime = _write_export(cube, sub_vol, result, fmt, slug)
            st.session_state._export = (payload, filename, mime)
            st.success(f"{filename} ready ({len(payload) / 1e6:.1f} MB).")
        except Exception as exc:  # noqa: BLE001
            show_error(exc, "Export failed")

    if st.session_state.get("_export"):
        payload, filename, mime = st.session_state._export
        st.download_button(f"Download {filename}", payload, file_name=filename, mime=mime)

    st.caption(
        "SEG-Y export reuses the original file as a template when one is loaded, which preserves "
        "every trace header byte; otherwise headers are rebuilt from the inline/crossline geometry. "
        "A subset run exports only the inverted block."
    )


def _write_export(cube, sub_vol, result, fmt: str, slug: str):
    """Serialise a cube to bytes for the download button."""
    out_vol = sub_vol.with_data(cube, source=f"{result.method} {slug}")

    if fmt == "NumPy (.npy)":
        path = os.path.join(tempfile.mkdtemp(), f"{slug}.npy")
        np.save(path, np.asarray(cube, dtype=np.float32))
        return open(path, "rb").read(), f"{slug}.npy", "application/octet-stream"

    if fmt == "NetCDF (xarray)":
        path = os.path.join(tempfile.mkdtemp(), f"{slug}.nc")
        out_vol.to_xarray(name=slug).to_netcdf(path)
        return open(path, "rb").read(), f"{slug}.nc", "application/x-netcdf"

    path = os.path.join(tempfile.mkdtemp(), f"{slug}.sgy")
    template = st.session_state.segy_path if not result.is_subset else None
    data_io.write_segy(out_vol, path, template_path=template)
    return open(path, "rb").read(), f"{slug}.sgy", "application/octet-stream"


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    init_state()
    step = sidebar()
    show_flash()
    {
        STEPS[0]: page_data,
        STEPS[1]: page_log_qc,
        STEPS[2]: page_well_tie,
        STEPS[3]: page_wavelet,
        STEPS[4]: page_low_freq,
        STEPS[5]: page_inversion,
        STEPS[6]: page_results,
    }[step]()


if __name__ == "__main__":
    main()
