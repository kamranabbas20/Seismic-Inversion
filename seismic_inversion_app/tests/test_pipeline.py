"""End-to-end checks on the synthetic dataset.

The synthetic cube is built from a known impedance model, so these tests can
assert on more than "it ran": every inversion engine must recover more of the
true band-limited impedance than the low-frequency model alone does.  That is
the property that actually matters -- an inversion that merely reproduces its
own background model is worthless, and would pass a smoke test.

Run with ``pytest`` or directly: ``python tests/test_pipeline.py``.
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import data_io, inversion, low_freq_model, utils  # noqa: E402
from modules import visualization as viz  # noqa: E402
from modules import wavelet as wvl  # noqa: E402

GATE = (60.0, 520.0)


def build_case(n_iline=16, n_xline=16, n_samples=300, n_wells=4, seed=7):
    """Synthetic cube + ties + calibrated wavelet + background model."""
    vol, wells = data_io.make_synthetic_dataset(
        n_iline=n_iline, n_xline=n_xline, n_samples=n_samples, n_wells=n_wells, seed=seed)
    ties = data_io.extract_well_traces(vol, wells, k=4)
    wav = wvl.multi_well_wavelet(ties, vol.dt, vol.twt, 128, t_min=GATE[0], t_max=GATE[1])
    wav = wvl.calibrate_amplitude(wav, ties, vol.twt, *GATE)
    lfm = low_freq_model.build_low_frequency_model(vol, wells, cutoff_hz=10.0)
    return vol, wells, ties, wav, lfm


def band_score(inverted_ai, well_ai, dt, f_lo=10.0, f_hi=60.0):
    """Correlation of the *seismic-band* log-impedance -- the part inversion adds.

    Comparing full-bandwidth impedance would be dominated by the low-frequency
    model both traces share, and would look good even for a broken inversion.
    """
    good = np.isfinite(well_ai) & (well_ai > 0) & np.isfinite(inverted_ai) & (inverted_ai > 0)
    if good.sum() < 32:
        return 0.0
    lw = utils.bandpass(np.log(utils.fill_nan_1d(np.where(good, well_ai, np.nan))), dt, f_lo, f_hi)
    li = utils.bandpass(np.log(np.maximum(inverted_ai, 1e-9)), dt, f_lo, f_hi)
    return utils.normalised_correlation(li[good], lw[good])


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def test_impedance_reflectivity_roundtrip():
    rng = np.random.default_rng(0)
    ai = 5_000_000 * np.exp(np.cumsum(rng.normal(0, 0.01, 400)))
    r = utils.reflectivity_from_ai(ai)
    back = utils.ai_from_reflectivity(r, ai0=ai[0])
    assert np.allclose(back, ai, rtol=1e-6), "recursion must invert the reflectivity exactly"


def test_convolve_same_length():
    """A wavelet longer than the trace must not change the trace length."""
    rng = np.random.default_rng(1)
    for n_x, n_w in [(200, 31), (200, 32), (50, 151), (60, 61)]:
        x, w = rng.normal(size=n_x), rng.normal(size=n_w)
        out = utils.convolve_same(x, w)
        assert out.size == n_x, f"{n_x}/{n_w} gave {out.size}"
        if n_x >= n_w:
            assert np.allclose(out, np.convolve(x, w, mode="same"))


def test_constant_phase_estimator():
    for true_phase in (0, -90, 30, 180):
        w = wvl.make_parametric_wavelet("ricker", 0.002, 160, phase=true_phase, freq=25)
        est = wvl.estimate_constant_phase(w.samples)
        err = ((est - true_phase + 180) % 360) - 180
        assert abs(err) <= 2, f"phase {true_phase}: estimated {est}"


def test_parametric_wavelet_families():
    for kind in wvl.WAVELET_TYPES:
        w = wvl.make_parametric_wavelet(kind, 0.002, 128)
        assert w.samples.size % 2 == 1, "wavelets must be odd-length so they have a centre"
        assert np.isfinite(w.samples).all()
        assert 3.0 < w.dominant_frequency() < 90.0, f"{kind}: {w.dominant_frequency()}"


def test_wellbased_wavelet_recovers_the_true_wavelet():
    """The synthetic is made with a 28 Hz Ricker; extraction should find it."""
    vol, wells = data_io.make_synthetic_dataset(n_iline=12, n_xline=12, n_samples=300,
                                                n_wells=3, peak_frequency=28.0, seed=3)
    ties = data_io.extract_well_traces(vol, wells)
    wav = wvl.multi_well_wavelet(ties, vol.dt, vol.twt, 128, t_min=GATE[0], t_max=GATE[1])
    assert abs(wav.dominant_frequency() - 28.0) < 5.0, wav.dominant_frequency()
    assert wav.quality["mean correlation"] > 0.9, wav.quality


def test_amplitude_calibration_is_applied():
    _, _, ties, wav, _ = build_case(n_iline=12, n_xline=12, n_wells=3)
    scalar = wav.quality.get("amplitude scalar")
    assert scalar and scalar > 0
    assert wav.quality["tie correlation"] > 0.9


def test_low_frequency_model_tracks_the_wells():
    vol, _, ties, _, lfm = build_case(n_iline=12, n_xline=12, n_wells=3)
    for tie in ties:
        lf = lfm.trace(tie.il_index, tie.xl_index)
        good = np.isfinite(tie.ai) & (tie.ai > 0)
        assert utils.normalised_correlation(lf[good], tie.ai[good]) > 0.85, tie.well
        assert np.all(lf[good] > 0), "log-domain filtering must keep impedance positive"


# --------------------------------------------------------------------------
# The three engines
# --------------------------------------------------------------------------

def _engine_beats_background(method, extra=None):
    vol, _, ties, wav, lfm = build_case()
    params = dict(extra or {})
    if method == "coloured":
        op = inversion.design_colour_operator(vol, ties, 8.0, 60.0, 200.0)
        params["operator"] = inversion.calibrate_colour_operator(op, vol, ties, *GATE)

    tie = ties[0]
    trace = vol.trace_at(tie.il_index, tie.xl_index)
    lf = lfm.trace(tie.il_index, tie.xl_index)

    res = inversion.invert(trace, wav.samples, lf, method=method, dt=vol.dt, **params)
    assert res["absolute_ai"] is not None, f"{method} should produce absolute AI given an LFM"

    baseline = band_score(lf, tie.ai, vol.dt)
    score = band_score(res["absolute_ai"], tie.ai, vol.dt)
    assert score > baseline + 0.1, (
        f"{method}: band correlation {score:.3f} vs background {baseline:.3f} -- "
        "the inversion is adding no information over its own low-frequency model")
    return score, baseline


def test_coloured_beats_background():
    _engine_beats_background("coloured")


def test_sparse_spike_beats_background():
    _engine_beats_background("sparse-spike", {"sparsity": 0.15, "n_iter": 12})


def test_model_based_beats_background():
    _engine_beats_background("model-based", {"model_weight": 0.1, "max_iter": 80})


def test_sparsity_slider_controls_sparsity():
    """Raising the weight must produce a sparser reflectivity and a looser fit."""
    vol, _, ties, wav, lfm = build_case(n_iline=8, n_xline=8, n_wells=3)
    tie = ties[0]
    trace = vol.trace_at(tie.il_index, tie.xl_index)
    lf = lfm.trace(tie.il_index, tie.xl_index)
    low = inversion.invert(trace, wav.samples, lf, method="sparse-spike",
                           dt=vol.dt, sparsity=0.02, n_iter=12)
    high = inversion.invert(trace, wav.samples, lf, method="sparse-spike",
                            dt=vol.dt, sparsity=2.0, n_iter=12)
    assert high["sparsity_ratio"] < low["sparsity_ratio"], "higher weight must be sparser"
    assert high["misfit"] > low["misfit"], "higher weight must fit the data less closely"


def test_model_based_requires_a_background_model():
    vol, _, _, wav, _ = build_case(n_iline=6, n_xline=6, n_wells=2)
    try:
        inversion.invert(vol.trace_at(0, 0), wav.samples, None, method="model-based", dt=vol.dt)
    except ValueError as exc:
        assert "low-frequency" in str(exc)
    else:
        raise AssertionError("model-based must refuse to run without a low-frequency model")


def test_model_based_honours_the_hard_bound():
    vol, _, ties, wav, lfm = build_case(n_iline=8, n_xline=8, n_wells=3)
    tie = ties[0]
    bound = 0.05
    res = inversion.invert(vol.trace_at(tie.il_index, tie.xl_index), wav.samples,
                           lfm.trace(tie.il_index, tie.xl_index), method="model-based",
                           dt=vol.dt, max_change=bound, max_iter=60)
    departure = np.abs(res["log_impedance"] - res["background_log_impedance"])
    assert departure.max() <= bound + 1e-6, departure.max()


def test_common_interface_is_common():
    """Every engine must return the same keys, so the QC code can be written once."""
    vol, _, ties, wav, lfm = build_case(n_iline=8, n_xline=8, n_wells=3)
    op = inversion.calibrate_colour_operator(
        inversion.design_colour_operator(vol, ties, 8.0, 60.0, 200.0), vol, ties, *GATE)
    tie = ties[0]
    trace = vol.trace_at(tie.il_index, tie.xl_index)
    lf = lfm.trace(tie.il_index, tie.xl_index)

    required = {"method", "reflectivity", "relative_ai", "absolute_ai",
                "synthetic", "residual", "misfit", "correlation"}
    for method, extra in [("coloured", {"operator": op}),
                          ("sparse-spike", {"sparsity": 0.15}),
                          ("model-based", {})]:
        res = inversion.invert(trace, wav.samples, lf, method=method, dt=vol.dt, **extra)
        assert required <= set(res), f"{method} missing {required - set(res)}"
        for key in ("reflectivity", "relative_ai", "absolute_ai"):
            assert np.asarray(res[key]).size == trace.size, f"{method}/{key} changed the trace length"


# --------------------------------------------------------------------------
# Volume runner and I/O
# --------------------------------------------------------------------------

def test_run_volume_subset_matches_trace_by_trace():
    vol, _, ties, wav, lfm = build_case(n_iline=10, n_xline=10, n_wells=3)
    res = inversion.run_volume(vol, "model-based", wav.samples, lfm,
                               il_range=(2, 5), xl_range=(1, 6), max_iter=40)
    assert res.is_subset
    assert res.relative_ai.shape[:2] == (3, 5)

    single = inversion.invert(vol.trace_at(2, 1), wav.samples, lfm.trace(2, 1),
                              method="model-based", dt=vol.dt, max_iter=40)
    assert np.allclose(res.absolute_ai[0, 0, :], single["absolute_ai"], rtol=1e-4)


def test_dead_traces_are_skipped_not_crashed():
    vol, wells = data_io.make_synthetic_dataset(n_iline=8, n_xline=8, n_samples=200, n_wells=2, seed=4)
    vol.data[0, :, :] = 0.0                      # a dead inline
    ties = data_io.extract_well_traces(vol, wells)
    wav = wvl.calibrate_amplitude(
        wvl.multi_well_wavelet(ties, vol.dt, vol.twt, 128, t_min=60, t_max=340),
        ties, vol.twt, 60, 340)
    lfm = low_freq_model.build_low_frequency_model(vol, wells, cutoff_hz=10.0)
    res = inversion.run_volume(vol, "sparse-spike", wav.samples, lfm, sparsity=0.2, n_iter=6)
    assert np.all(np.isnan(res.correlation[0, :])), "dead traces should be left unset"
    assert np.isfinite(res.correlation[1:, :]).any(), "live traces should still be inverted"


def test_segy_roundtrip_preserves_geometry_and_samples():
    vol, _ = data_io.make_synthetic_dataset(n_iline=8, n_xline=7, n_samples=150, n_wells=1, seed=1)
    out = os.path.join(tempfile.mkdtemp(), "roundtrip.sgy")
    data_io.write_segy(vol, out)
    back = data_io.load_segy(out)
    assert back.shape == vol.shape
    assert np.array_equal(back.iline, vol.iline) and np.array_equal(back.xline, vol.xline)
    assert np.allclose(back.twt, vol.twt)
    assert np.allclose(back.cdp_x, vol.cdp_x, atol=0.01)
    assert np.max(np.abs(back.data - vol.data)) < 1e-4


def test_segy_template_writeback_preserves_headers():
    vol, _ = data_io.make_synthetic_dataset(n_iline=6, n_xline=6, n_samples=100, n_wells=1, seed=5)
    tmp = tempfile.mkdtemp()
    original = os.path.join(tmp, "original.sgy")
    derived = os.path.join(tmp, "derived.sgy")
    data_io.write_segy(vol, original)
    data_io.write_segy(vol.with_data(vol.data * 2.0), derived, template_path=original)
    back = data_io.load_segy(derived)
    assert np.allclose(back.data, vol.data * 2.0, atol=1e-3)
    assert np.array_equal(back.iline, vol.iline)


def test_demo_files_round_trip_through_the_real_readers():
    """The demo generator must produce files the *file readers* accept.

    The in-app synthetic mode bypasses SEG-Y and LAS parsing entirely, so
    without this the upload path is untested -- which is how the well-header
    matching bug got in.
    """
    import make_demo_data

    outdir = tempfile.mkdtemp()
    sys.argv = ["make_demo_data.py", "--outdir", outdir, "--inlines", "8",
                "--crosslines", "8", "--samples", "120", "--wells", "3"]
    assert make_demo_data.main() == 0

    vol = data_io.load_segy(os.path.join(outdir, "demo_seismic.sgy"),
                            iline=189, xline=193, cdp_x=181, cdp_y=185)
    assert vol.shape == (8, 8, 120)

    import glob
    wells = [data_io.load_las(p) for p in sorted(glob.glob(os.path.join(outdir, "*.las")))]
    assert len(wells) == 3
    assert all(w.valid_mask().sum() > 100 for w in wells)
    # Assert on the resolved selection rather than note prose, which is UI text.
    assert all(w.selection.time is not None for w in wells), "LAS should carry a TWT curve"
    assert all(w.selection.sonic == "DT" for w in wells)
    assert all(w.selection.density == "RHOB" for w in wells)
    assert all(w.selection.sonic_unit == "us/ft" for w in wells), \
        [w.selection.sonic_unit for w in wells]
    assert all(w.selection.density_unit == "g/cm3" for w in wells), \
        [w.selection.density_unit for w in wells]

    # Drop the locations so the header CSV has real work to do.
    for w in wells:
        w.x = w.y = None
    headers = data_io.load_well_headers(os.path.join(outdir, "well_headers.csv"))
    data_io.apply_well_headers(wells, headers)
    assert all(w.has_location for w in wells), "every well should be located by the header CSV"

    horizons = low_freq_model.load_horizon_csv(os.path.join(outdir, "horizons.csv"), vol)
    assert set(horizons) == {"TOP_RESERVOIR", "BASE_RESERVOIR"}
    assert all(h.shape == (8, 8) and np.isfinite(h).all() for h in horizons.values())

    ties = data_io.extract_well_traces(vol, wells)
    assert len(ties) == 3 and all(t.distance < 40 for t in ties)


def test_horizon_guided_model_differs_from_flat_interpolation():
    vol, wells = data_io.make_synthetic_dataset(n_iline=10, n_xline=10, n_samples=200,
                                                n_wells=3, seed=6)
    n_il, n_xl = vol.shape[:2]
    ii, jj = np.meshgrid(np.arange(n_il), np.arange(n_xl), indexing="ij")
    horizon = {"H1": 300.0 + 4.0 * ii + 2.0 * jj}

    flat = low_freq_model.build_low_frequency_model(vol, wells, cutoff_hz=10.0)
    guided = low_freq_model.build_low_frequency_model(vol, wells, cutoff_hz=10.0, horizons=horizon)
    assert any("flattened" in n for n in guided.notes), guided.notes
    assert not np.allclose(flat.ai, guided.ai), "horizon guidance should change the model"
    assert np.all(guided.ai > 0)


def test_short_trace_with_long_wavelet_survives():
    """The regression that motivated ``convolve_same``."""
    vol, wells = data_io.make_synthetic_dataset(n_iline=5, n_xline=5, n_samples=50, n_wells=2, seed=2)
    ties = data_io.extract_well_traces(vol, wells)
    wav = wvl.make_parametric_wavelet("ricker", vol.dt, 300.0, freq=20)   # longer than the trace
    assert wav.samples.size > vol.shape[2]
    lfm = low_freq_model.build_low_frequency_model(vol, wells, cutoff_hz=10.0)
    for method, extra in [("sparse-spike", {"sparsity": 0.2, "n_iter": 6}),
                          ("model-based", {"max_iter": 20})]:
        res = inversion.run_volume(vol, method, wav.samples, lfm, **extra)
        assert res.relative_ai.shape == vol.shape
    assert ties


def _synthetic_las(curves: dict, units: list, name: str = "TEST"):
    """Build an in-memory LAS with the given curves and unit strings."""
    import io as _io

    import lasio

    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem("WELL", value=name)
    dept = np.arange(1000.0, 1200.0, 0.5)
    las.append_curve("DEPT", dept, unit="m")
    for (mnemonic, value), unit in zip(curves.items(), units):
        las.append_curve(mnemonic, np.full(dept.shape, float(value)), unit=unit)
    buf = _io.StringIO()
    las.write(buf, version=2.0)
    buf.seek(0)
    return buf


def test_curve_autodetection_across_unit_conventions():
    """Every combination must land on ~3000 m/s and ~2450 kg/m3."""
    cases = [
        ("us/ft + g/cm3", {"DT": 100.0, "RHOB": 2.45}, ["us/ft", "g/cm3"]),
        ("us/m + kg/m3", {"DT": 328.084, "RHOB": 2450.0}, ["us/m", "kg/m3"]),
        ("velocity in m/s", {"VP": 3000.0, "RHOB": 2.45}, ["m/s", "g/cm3"]),
        ("velocity in ft/s", {"VP": 9842.5, "RHOB": 2.45}, ["ft/s", "g/cm3"]),
        ("blank units, us/ft magnitude", {"DTCO": 100.0, "RHOZ": 2.45}, ["", ""]),
        ("blank units, us/m magnitude", {"DTCO": 328.084, "RHOZ": 2450.0}, ["", ""]),
    ]
    for label, curves, units in cases:
        well = data_io.load_las(_synthetic_las(curves, units), name="W")
        vp = float(np.nanmedian(well.vp))
        rho = float(np.nanmedian(well.rho))
        assert abs(vp - 3000) < 60, f"{label}: Vp {vp:.0f}"
        assert abs(rho - 2450) < 20, f"{label}: Rho {rho:.0f}"


def test_las_unit_string_beats_magnitude():
    """An explicit LAS unit must win over the magnitude heuristic."""
    well = data_io.load_las(_synthetic_las({"DT": 328.084, "RHOB": 2.45}, ["us/ft", "g/cm3"]), name="W")
    assert well.selection.sonic_unit == "us/ft"
    assert abs(float(np.nanmedian(well.vp)) - 929) < 20, "should honour the (wrong) stated unit"


def test_sonic_hint_only_breaks_genuine_ties():
    """The hint must not override evidence -- only settle the ambiguous band."""
    from modules.data_io import _guess_sonic_unit

    values = np.full(50, 95.0)      # unambiguously us/ft
    assert _guess_sonic_unit(values, "", hint="us/m") == "us/ft"
    values = np.full(50, 310.0)     # unambiguously us/m
    assert _guess_sonic_unit(values, "", hint="us/ft") == "us/m"
    values = np.full(50, 160.0)     # genuinely ambiguous
    assert _guess_sonic_unit(values, "", hint="us/m") == "us/m"
    assert _guess_sonic_unit(values, "", hint=None) == "us/ft"


def test_reassigning_curves_recomputes_the_logs():
    """The whole point of the QC step: fix a unit, get corrected impedance."""
    well = data_io.load_las(_synthetic_las({"DT": 100.0, "RHOB": 2.45}, ["us/ft", "g/cm3"]), name="W")
    original_vp = float(np.nanmedian(well.vp))

    # Reading us/ft slowness as us/m scales velocity *up* by ft-per-metre.
    wrong = data_io.CurveSelection(sonic="DT", sonic_unit="us/m",
                                   density="RHOB", density_unit="g/cm3")
    well.apply_selection(wrong)
    assert abs(float(np.nanmedian(well.vp)) - original_vp * utils.FT_PER_M) < 5

    fixed = data_io.CurveSelection(sonic="DT", sonic_unit="us/ft",
                                   density="RHOB", density_unit="g/cm3")
    well.apply_selection(fixed)
    assert abs(float(np.nanmedian(well.vp)) - original_vp) < 1e-6, "must be fully reversible"


def test_qc_flags_catch_a_unit_mistake():
    well = data_io.load_las(_synthetic_las({"DT": 100.0, "RHOB": 2.45}, ["us/ft", "g/cm3"]), name="W")
    assert not [m for sev, m in well.qc_flags() if sev == "error"], "correct units should pass"

    well.apply_selection(data_io.CurveSelection(sonic="DT", sonic_unit="us/m",
                                                density="RHOB", density_unit="g/cm3"))
    errors = [m for sev, m in well.qc_flags() if sev == "error"]
    assert errors, "10,000 m/s should be flagged as implausible"
    assert any("Vp" in m for m in errors), errors

    well.apply_selection(data_io.CurveSelection(sonic="DT", sonic_unit="us/ft",
                                                density="RHOB", density_unit="kg/m3"))
    errors = [m for sev, m in well.qc_flags() if sev == "error"]
    assert any("Density" in m for m in errors), errors


def test_qc_flags_catch_no_seismic_overlap():
    well = data_io.load_las(
        _synthetic_las({"DT": 100.0, "RHOB": 2.45, "TWT": 4000.0}, ["us/ft", "g/cm3", "ms"]), name="W")
    flags = well.qc_flags(seismic_twt=np.arange(0.0, 1000.0, 2.0))
    assert any(sev == "error" and "overlap" in m for sev, m in flags), flags


def test_bulk_shift_survives_reassignment():
    well = data_io.load_las(
        _synthetic_las({"DT": 100.0, "RHOB": 2.45, "TWT": 1500.0}, ["us/ft", "g/cm3", "ms"]), name="W")
    well.set_bulk_shift(25.0)
    before = float(np.nanmedian(well.twt))
    well.apply_selection(well.selection)
    assert abs(float(np.nanmedian(well.twt)) - before) < 1e-6, "reassignment must not drop the shift"


def test_curve_inventory_lists_every_curve():
    well = data_io.load_las(
        _synthetic_las({"DT": 100.0, "RHOB": 2.45, "GR": 60.0}, ["us/ft", "g/cm3", "gAPI"]), name="W")
    rows = {r["curve"]: r for r in well.curve_inventory()}
    assert {"DEPT", "DT", "RHOB", "GR"} <= set(rows)
    assert rows["DT"]["role"] == "Vp (sonic)"
    assert rows["RHOB"]["role"] == "Density"
    assert rows["GR"]["role"] == "", "an unassigned curve should still be listed"
    assert rows["GR"]["LAS unit"] == "gAPI"
    assert rows["DT"]["valid %"] == 100.0


def test_well_without_density_still_loads():
    """A missing curve is a QC flag to resolve in the UI, not a lost well."""
    well = data_io.load_las(_synthetic_las({"DT": 100.0}, ["us/ft"]), name="W")
    assert well.selection.density is None
    assert "DT" in well.curves
    assert any(sev == "error" for sev, _ in well.qc_flags())


def test_log_qc_figures_build():
    well = data_io.load_las(
        _synthetic_las({"DT": 100.0, "RHOB": 2.45, "GR": 60.0}, ["us/ft", "g/cm3", "gAPI"]), name="W")
    assert len(viz.log_qc_figure(well).data) > 0
    assert len(viz.curve_preview_figure(well, ["DT", "RHOB"]).data) > 0
    viz.curve_preview_figure(well, [])          # must not raise on an empty selection


def test_upload_spill_is_byte_exact():
    import hashlib
    import io as _io

    payload = os.urandom(3_000_000)
    buf = _io.BytesIO(payload)
    path = data_io.persist_upload(buf, chunk=64 * 1024)
    try:
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == \
            hashlib.sha256(payload).hexdigest()
        assert buf.tell() == 0, "buffer should be rewound for reuse"
    finally:
        os.unlink(path)


def test_synthetic_wells_carry_curves_like_a_real_las():
    """The demo path must exercise the same curve machinery as a loaded LAS.

    Without this the log-QC step could not describe a synthetic well, and
    applying an assignment there would clear logs it could not see.
    """
    vol, wells = data_io.make_synthetic_dataset(n_iline=6, n_xline=6, n_samples=150,
                                                n_wells=2, seed=3)
    for well in wells:
        assert {"DT", "RHOB", "TWT"} <= set(well.curves)
        assert well.selection.sonic == "DT" and well.selection.sonic_unit == "us/ft"
        assert well.selection.density == "RHOB" and well.selection.density_unit == "g/cm3"
        assert well.selection.time == "TWT"
        assert well.curve_units["DT"] == "us/ft"
        assert not [m for sev, m in well.qc_flags(vol.twt) if sev == "error"]

        # Round-tripping the stored curves must reproduce the logs exactly.
        vp_before = well.vp.copy()
        well.apply_selection(well.selection)
        assert np.allclose(well.vp, vp_before, rtol=1e-9)


def test_bulk_shift_is_not_cumulative():
    _, wells = data_io.make_synthetic_dataset(n_iline=5, n_xline=5, n_samples=80, n_wells=1)
    well = wells[0]
    base = well.twt.copy()
    well.set_bulk_shift(12.0)
    well.set_bulk_shift(-5.0)
    assert np.allclose(well.twt, base - 5.0), "shifts must apply from the original time axis"


def test_figures_build():
    """Every plotly figure must construct on real inputs."""
    vol, wells, ties, wav, lfm = build_case(n_iline=8, n_xline=8, n_wells=3)
    op = inversion.calibrate_colour_operator(
        inversion.design_colour_operator(vol, ties, 8.0, 60.0, 200.0), vol, ties, *GATE)
    res = inversion.run_volume(vol, "model-based", wav.samples, lfm, max_iter=30)
    overlay = viz.well_overlay_positions(ties, "inline", ties[0].il_index, vol, tolerance=3)
    crossplot = inversion.crossplot_at_wells(res, vol, ties, t_min=GATE[0], t_max=GATE[1])

    figures = [
        viz.section_figure(vol.data, vol.twt, vol.xline, "inline", 2, wells=overlay),
        viz.dual_section_figure(vol.data, res.absolute_ai, vol.twt, vol.xline, "inline", 2),
        viz.time_slice_figure(res.absolute_ai, vol.twt, vol.iline, vol.xline, 300.0),
        viz.wavelet_figure(wav),
        viz.colour_operator_figure(op),
        viz.well_tie_figure(ties[0], wav.samples, *GATE),
        viz.crossplot_figure(crossplot),
        viz.basemap_figure(vol, wells, ties),
        viz.well_log_figure(wells[0]),
        viz.low_freq_qc_figure(lfm, ties),
        viz.qc_map_figure(res.correlation, vol.iline, vol.xline),
    ]
    assert all(len(f.data) > 0 for f in figures)
    assert len(viz.tie_score_table(ties, wav.samples, *GATE)) == len(ties)


# --------------------------------------------------------------------------

def _main() -> int:
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith("test_") and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
