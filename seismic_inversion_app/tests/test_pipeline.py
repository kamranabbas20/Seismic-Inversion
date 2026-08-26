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
                          ("model-based", {}),
                          ("bayesian", {})]:
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


def _las_with_depth_unit(depth_unit: str, kb: float = 100.0):
    """LAS whose index is 1000-1200 in *depth_unit*, with a KB in the same system."""
    import io as _io

    import lasio

    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem("WELL", value="UNITS")
    las.well["KB"] = lasio.HeaderItem("KB", value=kb)
    dept = np.arange(1000.0, 1200.0, 0.5)
    las.append_curve("DEPT", dept, unit=depth_unit)
    las.append_curve("DT", np.full(dept.shape, 100.0), unit="us/ft")
    las.append_curve("RHOB", np.full(dept.shape, 2.45), unit="g/cm3")
    buf = _io.StringIO()
    las.write(buf, version=2.0)
    buf.seek(0)
    return buf


def test_depth_index_in_feet_is_converted_to_metres():
    """A LAS indexed in feet must not land 3.28x too deep.

    ``WellData.md`` is metres everywhere downstream -- ``integrate_sonic_to_twt``
    divides it by a velocity in m/s -- so an unconverted foot index does not
    fail loudly, it just puts the well far below where it belongs.
    """
    ft = data_io.load_las(_las_with_depth_unit("FT"), name="W")
    m = data_io.load_las(_las_with_depth_unit("M"), name="W")
    assert ft.depth_unit == "ft"
    assert m.depth_unit == "m"
    assert abs(ft.md[0] - 1000.0 / utils.FT_PER_M) < 1e-6, f"md {ft.md[0]}"
    assert abs(ft.md[0] - m.md[0] / utils.FT_PER_M) < 1e-6
    # KB is quoted in the same system as the index, so it converts with it.
    assert abs(ft.kb - 100.0 / utils.FT_PER_M) < 1e-6, f"kb {ft.kb}"
    assert any("feet" in n for n in ft.notes), ft.notes


def test_foot_indexed_well_ties_at_the_same_time_as_its_metric_twin():
    """The same well in feet and in metres must produce the same TWT."""
    ft = data_io.load_las(_las_with_depth_unit("FT"), name="W")
    m_ft = ft.md * utils.FT_PER_M          # back to the raw numbers in the file
    metric = data_io.load_las(_las_with_depth_unit("M"), name="W")
    # Same numbers on the page, different unit: the foot well is the shallower
    # one, so its TWT must be smaller by very nearly the foot/metre ratio.
    t_ft = float(np.nanmin(ft.twt))
    t_m = float(np.nanmin(metric.twt))
    assert t_ft < t_m, f"feet {t_ft:.1f} ms should be shallower than metres {t_m:.1f} ms"
    assert abs(t_m / t_ft - utils.FT_PER_M) < 0.05, f"ratio {t_m / t_ft:.3f}"
    assert abs(m_ft[0] - 1000.0) < 1e-6


def test_unlabelled_depth_index_is_assumed_metres_and_flagged():
    """Depth magnitude cannot separate feet from metres, so QC must say so.

    Written by hand rather than through lasio, which substitutes ``M`` for a
    blank index unit on write -- real files in the wild are not so tidy.
    """
    import io as _io

    text = "\n".join([
        "~Version", "VERS. 2.0 :", "WRAP. NO :",
        "~Well", "WELL. NOWHERE :WELL", "NULL. -999.25 :NULL",
        "~Curve",
        "DEPT.      :1 Depth",          # <- no unit at all
        "DT   .US/F :2 Sonic",
        "RHOB .G/CC :3 Density",
        "~Ascii",
        "1000.0 100.0 2.45",
        "1000.5 100.0 2.45",
        "1001.0 100.0 2.45",
        "1001.5 100.0 2.45",
        "",
    ])
    well = data_io.load_las(_io.StringIO(text), name="W")
    assert well.depth_unit == "", f"got {well.depth_unit!r}"
    assert abs(well.md[0] - 1000.0) < 1e-6, "unlabelled must pass through untouched"
    assert any("assumed metres" in n for n in well.notes), well.notes
    flags = well.qc_flags()
    assert any(lvl == "warning" and "assumed metres" in msg for lvl, msg in flags), flags


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


# --------------------------------------------------------------------------
# Auxiliary well files (time-depth, deviation, markers) -- F3 demo layout
# --------------------------------------------------------------------------

# Real F02-1 values from the F3 demo survey, in the file layouts it ships with:
# tab-separated, no header, times in seconds, markers with spaces in the name,
# and a track that mixes tabs and spaces on the same row.
F3_TD = "\n".join([
    "30\t0", "553.6\t0.544", "612.9\t0.607", "683.31\t0.675", "716.65\t0.712",
    "748.49\t0.748", "795.18\t0.794", "927.28\t0.932", "1025.42\t1.031",
    "1285.09\t1.242", "1695\t1.67", "1872\t1.861", "2636\t2.682", "3150\t3.234",
])
F3_MARKERS = "\r\n".join([
    "30\tSeasurface", "553.6\tMFS11", "1025.42\tTruncation 1",
    "1048.84\tLower Low Sonic", "1285.09\tNMRF (Mid_Mio_Unc)", "1695\tCKGR",
])
F3_TRACK = "606554\t6080126  -30       0\n606554\t6080126\t1665\t1695\n"


def _buf(text: str):
    import io as _io
    return _io.StringIO(text)


def test_sniffing_tells_the_three_file_kinds_apart():
    assert data_io.sniff_well_file(_buf(F3_TD)) == "time_depth"
    assert data_io.sniff_well_file(_buf(F3_MARKERS)) == "markers"
    assert data_io.sniff_well_file(_buf(F3_TRACK)) == "track"


def test_time_depth_reads_seconds_as_two_way():
    td = data_io.load_time_depth(_buf(F3_TD))
    assert td.was_seconds and not td.was_one_way
    assert abs(td.twt.max() - 3234.0) < 1e-6, "3.234 s should become 3234 ms"
    assert abs(td.datum_md - 30.0) < 1e-6, "time zero sits at MD 30 (the KB)"

    vi = td.interval_velocity()
    vi = vi[np.isfinite(vi)]
    assert 1400 < vi.min() and vi.max() < 3200, f"implausible interval velocity {vi.min()}-{vi.max()}"

    # Interpolation must reproduce the table exactly at its own knots.
    assert abs(float(td.to_twt(np.array([553.6]))[0]) - 544.0) < 1e-6


def test_one_way_time_depth_is_detected_when_it_can_be():
    """Detection only fires where the two-way reading is physically impossible."""
    # A fast section (~4600 m/s) recorded one-way: read as two-way it implies
    # ~9200 m/s, which no rock does, so the one-way reading is forced.
    fast_one_way = "\n".join(f"{md}\t{md / 4600.0:.6f}" for md in (0, 500, 1500, 3000))
    td = data_io.load_time_depth(_buf(fast_one_way))
    assert td.was_one_way
    vi = td.interval_velocity()
    assert abs(float(np.median(vi[np.isfinite(vi)])) - 4600) < 50


def test_ambiguous_one_way_defaults_to_two_way_and_warns():
    """Halving the F3 checkshot gives 3850 m/s -- a plausible rock velocity.

    Velocity alone cannot resolve this, so the loader must default to two-way
    (the commoner convention) rather than guess, and say so loudly enough that
    the user can override it.
    """
    halved = "\n".join(f"{md}\t{t / 2:.6f}" for md, t in
                       [(30, 0.0), (553.6, 0.544), (1285.09, 1.242), (3150, 3.234)])
    td = data_io.load_time_depth(_buf(halved))
    assert not td.was_one_way, "must not guess when the reading is plausible"
    assert any("fast for shallow section" in w for w in td.warnings()), td.warnings()

    forced = data_io.load_time_depth(_buf(halved), time_unit="s (OWT)")
    assert forced.was_one_way and abs(forced.twt.max() - 3234.0) < 1e-3


def test_time_depth_warnings_stay_quiet_on_a_good_table():
    assert data_io.load_time_depth(_buf(F3_TD)).warnings() == []


def test_time_depth_unit_can_be_forced():
    td = data_io.load_time_depth(_buf(F3_TD), time_unit="s (OWT)")
    assert td.was_one_way and abs(td.twt.max() - 6468.0) < 1e-6, "explicit unit must win"


def test_track_gives_location_kb_and_geometry():
    track = data_io.load_well_track(_buf(F3_TRACK))
    assert track.surface_xy == (606554.0, 6080126.0)
    assert abs(track.kb - 30.0) < 1e-6, "TVDSS -30 at MD 0 means KB is 30 m above the datum"
    assert track.is_vertical
    assert abs(track.tvdss_at(np.array([1695.0]))[0] - 1665.0) < 1e-6


def test_track_flips_an_elevation_convention():
    """Z decreasing with MD is an elevation, not a depth."""
    elevation = "600000 6080000 30 0\n600000 6080000 -1665 1695\n"
    track = data_io.load_well_track(_buf(elevation))
    assert track.tvdss[-1] > track.tvdss[0], "should be flipped to positive-down"
    assert abs(track.kb - 30.0) < 1e-6


def test_markers_keep_names_containing_spaces():
    markers = data_io.load_markers(_buf(F3_MARKERS))
    names = [m.name for m in markers]
    assert "Truncation 1" in names
    assert "NMRF (Mid_Mio_Unc)" in names
    assert "Lower Low Sonic" in names
    assert markers == sorted(markers, key=lambda m: m.md)


def test_attaching_aux_files_drives_time_location_and_markers():
    _, wells = data_io.make_synthetic_dataset(n_iline=6, n_xline=6, n_samples=100, n_wells=1)
    well = wells[0]

    well.attach_track(data_io.load_well_track(_buf(F3_TRACK)))
    assert well.has_location and abs(well.x - 606554.0) < 1e-6
    assert abs(well.kb - 30.0) < 1e-6

    well.attach_markers(data_io.load_markers(_buf(F3_MARKERS)))
    well.attach_time_depth(data_io.load_time_depth(_buf(F3_TD)))
    assert well.selection.time == data_io.TD_SOURCE, "a checkshot should be adopted by default"

    times = {m.name: m.twt for m in well.markers}
    assert abs(times["MFS11"] - 544.0) < 1e-6, "marker time must come from the checkshot"
    assert abs(times["CKGR"] - 1670.0) < 1e-6


def test_marker_times_follow_a_bulk_shift_and_stay_idempotent():
    _, wells = data_io.make_synthetic_dataset(n_iline=6, n_xline=6, n_samples=100, n_wells=1)
    well = wells[0]
    well.attach_markers(data_io.load_markers(_buf(F3_MARKERS)))
    well.attach_time_depth(data_io.load_time_depth(_buf(F3_TD)))

    base = {m.name: m.twt for m in well.markers}
    well.set_bulk_shift(20.0)
    assert abs({m.name: m.twt for m in well.markers}["MFS11"] - (base["MFS11"] + 20.0)) < 1e-6

    for _ in range(3):
        well.refresh_marker_times()
    assert abs({m.name: m.twt for m in well.markers}["MFS11"] - (base["MFS11"] + 20.0)) < 1e-6, \
        "refreshing must not accumulate the shift"

    well.set_bulk_shift(0.0)
    assert abs({m.name: m.twt for m in well.markers}["MFS11"] - base["MFS11"]) < 1e-6


def test_aux_filename_matching_tolerates_punctuation():
    names = ["F02-1", "F03-2", "F03-4"]
    assert data_io.match_well_name("F021_TD.txt", names) == "F02-1"
    assert data_io.match_well_name("F02-1_markers.txt", names) == "F02-1"
    assert data_io.match_well_name("F021.track", names) == "F02-1"
    assert data_io.match_well_name("F03-4_checkshot.dat", names) == "F03-4"
    assert data_io.match_well_name("UNRELATED.txt", names) is None


def test_checkshot_time_axis_feeds_the_inversion():
    """A well timed by checkshot must tie and invert like any other."""
    vol, wells = data_io.make_synthetic_dataset(n_iline=10, n_xline=10, n_samples=300,
                                                n_wells=2, seed=11)
    well = wells[0]
    # Re-time the well from its own curve, expressed as a coarse checkshot table.
    md = well.md[::200]
    twt = well.twt[::200]
    table = "\n".join(f"{m}\t{t / 1000.0:.6f}" for m, t in zip(md, twt))
    well.attach_time_depth(data_io.load_time_depth(_buf(table)))
    assert well.selection.time == data_io.TD_SOURCE

    ties = data_io.extract_well_traces(vol, wells)
    assert len(ties) == 2
    tie = next(t for t in ties if t.well == well.name)
    assert np.isfinite(tie.ai).sum() > 50, "checkshot timing should still land the log in the cube"


def test_time_depth_figure_builds():
    td = data_io.load_time_depth(_buf(F3_TD))
    markers = data_io.load_markers(_buf(F3_MARKERS))
    for marker in markers:
        marker.twt = float(td.to_twt(np.array([marker.md]))[0])
    assert len(viz.time_depth_figure(td, markers).data) > 0


def test_marker_labels_are_thinned_without_dropping_lines():
    """Every top gets a line; only the text is thinned where they crowd."""
    items = [(30.0, "Seasurface"), (553.6, "MFS11"), (612.9, "FS11"),
             (683.3, "MFS10"), (716.6, "MFS9"), (3150.0, "SLCL")]
    thinned = viz._thin_labels(items, span=3120.0)
    assert len(thinned) == len(items), "no marker line may be dropped"
    labelled = [n for _, n in thinned if n]
    assert 0 < len(labelled) < len(items), "crowded labels should be thinned"
    assert "Seasurface" in labelled and "SLCL" in labelled


def test_marker_annotations_never_render_placeholder_text():
    """Plotly turns annotation_text=None into the literal string 'new text'."""
    td = data_io.load_time_depth(_buf(F3_TD))
    markers = data_io.load_markers(_buf(F3_MARKERS))
    for marker in markers:
        marker.twt = float(td.to_twt(np.array([marker.md]))[0])
    texts = [a.text for a in viz.time_depth_figure(td, markers).layout.annotations]
    assert "new text" not in texts, texts


def test_every_summary_is_arrow_serialisable():
    """Streamlit serialises tables through Arrow, which types a column once.

    The summary dicts mix ints and formatted strings, so a naive transpose
    produces an object column that Arrow rejects with ArrowTypeError. Streamlit
    recovers by casting to string, so the table still renders -- and a full
    traceback is logged for every redraw. This asserts the summaries convert
    cleanly instead of relying on that fallback.
    """
    import pandas as pd
    import pyarrow as pa

    def kv(mapping):
        rows = {str(k): ("" if v is None else str(v)) for k, v in mapping.items()}
        return pd.DataFrame({"value": pd.Series(rows, dtype="string")})

    vol, wells, ties, wav, lfm = build_case(n_iline=8, n_xline=8, n_wells=3)
    op = inversion.calibrate_colour_operator(
        inversion.design_colour_operator(vol, ties, 8.0, 60.0, 200.0), vol, ties, *GATE)
    result = inversion.run_volume(vol, "model-based", wav.samples, lfm, max_iter=25)

    well = wells[0]
    well.attach_time_depth(data_io.load_time_depth(_buf(F3_TD)))
    well.attach_track(data_io.load_well_track(_buf(F3_TRACK)))
    well.attach_markers(data_io.load_markers(_buf(F3_MARKERS)))

    tables = {
        "volume": kv(vol.summary()),
        "wavelet": kv(wav.summary()),
        "low-frequency model": kv(lfm.summary()),
        "colour operator": kv(op.summary()),
        "inversion result": kv(result.summary()),
        "time-depth": kv(well.time_depth.summary()),
        "track": kv(well.track.summary()),
        "wells": pd.DataFrame([w.summary() for w in wells]),
        "curve inventory": pd.DataFrame(well.curve_inventory()),
        "tie scores": pd.DataFrame(viz.tie_score_table(ties, wav.samples, *GATE)),
    }
    for name, frame in tables.items():
        try:
            pa.Table.from_pandas(frame, preserve_index=True)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{name} is not Arrow-serialisable: {exc}") from exc


def _build_well_folder(root: str, wells=("F02-1", "F03-2", "F03-4")) -> str:
    """Create an F3-demo-shaped well folder on disk."""
    import lasio

    for sub in ("Lasfiles", "Checkshot", "Track", "Tops", "Notes"):
        os.makedirs(os.path.join(root, sub), exist_ok=True)

    rng = np.random.default_rng(5)
    for k, name in enumerate(wells):
        md = np.arange(30.0 + 50 * k, 1800.0, 0.5)
        vp = 1800 + 0.3 * md + rng.normal(0, 20, md.size)
        rho = 1900 + 0.17 * md + rng.normal(0, 10, md.size)
        las = lasio.LASFile()
        las.well["WELL"] = lasio.HeaderItem("WELL", value=name)
        las.append_curve("DEPT", md, unit="m")
        las.append_curve("DT", 1e6 / (vp * utils.FT_PER_M), unit="us/ft")
        las.append_curve("RHOB", rho / 1000.0, unit="g/cm3")
        las.write(os.path.join(root, "Lasfiles", f"{name}.las"), version=2.0)

        depths = np.array([30.0, 500.0, 1000.0, 1500.0, 1800.0])
        times = np.interp(depths, [30, 1800], [0.0, 1.85])
        with open(os.path.join(root, "Checkshot", f"{name}_TD.txt"), "w") as fh:
            fh.writelines(f"{d}\t{t:.4f}\n" for d, t in zip(depths, times))
        with open(os.path.join(root, "Track", f"{name}.track"), "w") as fh:
            fh.write(f"{606000 + k * 900}\t{6080000 + k * 1200}  -30       0\n")
            fh.write(f"{606000 + k * 900}\t{6080000 + k * 1200}\t1770\t1800\n")
        with open(os.path.join(root, "Tops", f"{name}_markers.txt"), "w") as fh:
            fh.write("420\tMFS11\r\n760\tTruncation 1\r\n1180\tFS 3\r\n")

    with open(os.path.join(root, "Notes", "readme.txt"), "w") as fh:
        fh.write("F3 Demo well data\nProvided by dGB Earth Sciences\n")
    return root


def test_folder_scan_loads_a_whole_well_database():
    root = _build_well_folder(tempfile.mkdtemp())
    scan = data_io.scan_well_folder(root)

    assert len(scan.wells) == 3
    assert scan.counts == {"las": 3, "markers": 3, "track": 3, "time_depth": 3}
    for well in scan.wells:
        assert well.has_location, f"{well.name} should be located by its track"
        assert abs(well.kb - 30.0) < 1e-6
        assert well.time_depth is not None
        assert well.selection.time == data_io.TD_SOURCE
        assert len(well.markers) == 3
        assert all(m.twt is not None for m in well.markers)
        assert not [m for sev, m in well.qc_flags() if sev == "error"]

    assert scan.folders["Lasfiles"] == "las"
    assert scan.folders["Checkshot"] == "time_depth"
    assert scan.folders["Track"] == "track"
    assert scan.folders["Tops"] == "markers"


def test_folder_scan_reports_what_it_could_not_use():
    """A file that did not load must be visible, not merely absent."""
    root = _build_well_folder(tempfile.mkdtemp())
    scan = data_io.scan_well_folder(root)
    assert any("readme.txt" in m for m in scan.skipped), scan.skipped
    assert len(scan.attached) == 12, scan.attached


def test_folder_scan_trusts_contents_over_folder_name():
    """A checkshot filed under Track must still be read as a checkshot."""
    root = _build_well_folder(tempfile.mkdtemp(), wells=("F02-1",))
    misfiled = os.path.join(root, "Track", "F02-1_TD.txt")
    os.rename(os.path.join(root, "Checkshot", "F02-1_TD.txt"), misfiled)

    scan = data_io.scan_well_folder(root)
    assert scan.counts.get("time_depth") == 1, "contents should win over the folder name"
    assert scan.wells[0].time_depth is not None
    assert any("not track as the folder suggests" in m for m in scan.attached), scan.attached


def test_folder_scan_survives_an_unreadable_las():
    root = _build_well_folder(tempfile.mkdtemp(), wells=("F02-1", "F03-2"))
    with open(os.path.join(root, "Lasfiles", "BROKEN.las"), "w") as fh:
        fh.write("this is not a LAS file at all\n")
    scan = data_io.scan_well_folder(root)
    assert len(scan.wells) == 2, "the good wells should still load"
    assert any("BROKEN.las" in m for m in scan.skipped), scan.skipped


def test_folder_scan_needs_at_least_one_las():
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "Checkshot"))
    with open(os.path.join(root, "Checkshot", "F02-1_TD.txt"), "w") as fh:
        fh.write("30\t0\n1800\t1.85\n")
    try:
        data_io.scan_well_folder(root)
    except ValueError as exc:
        assert "no LAS files" in str(exc)
    else:
        raise AssertionError("a folder with no logs should be rejected, not silently empty")


def test_find_segy_files_orders_by_size():
    root = _build_well_folder(tempfile.mkdtemp(), wells=("F02-1",))
    small_vol, _ = data_io.make_synthetic_dataset(n_iline=5, n_xline=5, n_samples=60, n_wells=1)
    big_vol, _ = data_io.make_synthetic_dataset(n_iline=10, n_xline=10, n_samples=200, n_wells=1)
    small = os.path.join(root, "small.sgy")
    big = os.path.join(root, "Lasfiles", "big.segy")
    data_io.write_segy(small_vol, small)
    data_io.write_segy(big_vol, big)
    found = data_io.find_segy_files(root)
    assert found[0] == big and small in found, found


# --------------------------------------------------------------------------
# Seismic viewer and well correlation
# --------------------------------------------------------------------------

def _correlation_case(n_wells: int = 3):
    """A cube plus wells carrying markers, positioned inside the survey."""
    vol, wells = data_io.make_synthetic_dataset(n_iline=24, n_xline=24, n_samples=400,
                                                n_wells=n_wells, seed=9)
    for k, well in enumerate(wells):
        well.attach_markers([
            data_io.Marker(md=400.0 + 60 * k, name="TOP_A"),
            data_io.Marker(md=800.0 + 60 * k, name="TOP_B"),
            data_io.Marker(md=1200.0 + 60 * k, name=f"LOCAL_{k}"),
        ])
    ties = data_io.extract_well_traces(vol, wells)
    return vol, wells, ties


def test_arbitrary_line_follows_the_well_path():
    vol, wells, _ = _correlation_case(4)
    order = [w.name for w in wells]
    line = data_io.line_through_wells(vol, wells, order)

    assert line.data.shape[1] == vol.shape[2]
    assert line.node_label == order
    assert np.all(np.diff(line.distance) > 0), "distance must increase along the path"
    assert line.distance[-1] > 0

    # Each node must land on the bin the well itself ties to.
    ties = {t.well: t for t in data_io.extract_well_traces(vol, wells)}
    for k, name in enumerate(order):
        i = int(np.argmin(np.abs(line.distance - line.node_distance[k])))
        assert int(line.iline[i]) == int(vol.iline[ties[name].il_index])
        assert int(line.xline[i]) == int(vol.xline[ties[name].xl_index])


def test_arbitrary_line_needs_two_distinct_points():
    vol, wells, _ = _correlation_case(2)
    for well in wells:
        well.x, well.y = 500_000.0, 6_000_000.0      # all at one spot
    try:
        data_io.line_through_wells(vol, wells, [w.name for w in wells])
    except ValueError as exc:
        assert "zero length" in str(exc) or "two located" in str(exc)
    else:
        raise AssertionError("a degenerate traverse should be rejected")


def test_correlation_panel_respects_the_requested_order():
    vol, wells, ties = _correlation_case(3)
    order = [w.name for w in wells][::-1]
    fig = viz.correlation_figure(wells, order, ties)
    assert list(fig.layout.xaxis.ticktext) == order, "wells must appear in the chosen order"
    assert len(fig.data) > 0


def test_correlation_flattening_aligns_the_datum():
    """Flattening must put the chosen top at one time in every well."""
    vol, wells, ties = _correlation_case(3)
    raw = [next(m.twt for m in w.markers if m.name == "TOP_A") for w in wells]
    assert max(raw) - min(raw) > 1.0, "the fixture should have a dipping datum"

    order = [w.name for w in wells]
    flat = viz.correlation_figure(wells, order, ties, flatten_marker="TOP_A",
                                  connect_markers=False)
    # The TOP_A traces are the horizontal ticks drawn in each well's slot.
    tops = [t.y[0] for t in flat.data if t.name == "TOP_A"]
    assert len(tops) == len(order)
    assert max(tops) - min(tops) < 1e-6, f"TOP_A should be flat, spread {max(tops) - min(tops)}"


def test_correlation_only_offers_correlatable_markers():
    vol, wells, ties = _correlation_case(3)
    order = [w.name for w in wells]
    shared = viz.common_markers(wells, order)
    assert "TOP_A" in shared and "TOP_B" in shared
    assert not any(n.startswith("LOCAL_") for n in shared), \
        "a marker in one well only cannot be correlated"


def test_correlation_curves_are_those_common_to_every_well():
    vol, wells, ties = _correlation_case(3)
    wells[0].curves["EXTRA"] = np.zeros_like(wells[0].md)
    order = [w.name for w in wells]
    curves = viz.common_curves(wells, order)
    assert curves[:3] == ["AI", "Vp", "Rho"]
    assert "DT" in curves and "RHOB" in curves
    assert "EXTRA" not in curves, "a curve in one well only is not comparable"
    assert "DEPT" not in curves


def test_correlation_variants_all_build():
    vol, wells, ties = _correlation_case(3)
    order = [w.name for w in wells]
    variants = [
        dict(),
        dict(show_seismic=False),
        dict(show_logs=False),
        dict(curve="GR") if "GR" in viz.common_curves(wells, order) else dict(curve="Vp"),
        dict(flatten_marker="TOP_B"),
        dict(marker_names=["TOP_A"]),
        dict(t_min=200.0, t_max=600.0),
        dict(gain=3.0),
    ]
    for kwargs in variants:
        assert len(viz.correlation_figure(wells, order, ties, **kwargs).data) > 0, kwargs
    # and with no ties at all
    assert len(viz.correlation_figure(wells, order, None).data) > 0


def test_correlation_rejects_an_empty_selection():
    vol, wells, ties = _correlation_case(2)
    try:
        viz.correlation_figure(wells, [], ties)
    except ValueError as exc:
        assert "no wells" in str(exc).lower()
    else:
        raise AssertionError("an empty well selection should be rejected")


def test_section_gain_widens_the_colour_range():
    vol, _, _ = _correlation_case(2)
    plain = viz.section_figure(vol.data, vol.twt, vol.xline, "inline", 3, gain=1.0)
    loud = viz.section_figure(vol.data, vol.twt, vol.xline, "inline", 3, gain=4.0)
    assert loud.data[0].zmax < plain.data[0].zmax, "more gain means a tighter colour range"


def test_arbitrary_line_figure_builds():
    vol, wells, _ = _correlation_case(3)
    line = data_io.line_through_wells(vol, wells, [w.name for w in wells])
    fig = viz.arbitrary_line_figure(line)
    assert len(fig.data) > 0
    assert [a.text for a in fig.layout.annotations] == [w.name for w in wells]


# --------------------------------------------------------------------------
# Bayesian linear inversion
# --------------------------------------------------------------------------

def test_roughness_operator_ignores_constants_and_ramps():
    """A curvature penalty must be blind to level and slope.

    Left with truncated first and last rows the operator reads ``[-2, 1]`` at
    the edges, so a flat trace scores a large "roughness" and any penalty built
    on it drags the level of log-impedance toward zero rather than smoothing
    it.  That silently biased both the model-based and Bayesian engines.
    """
    L = inversion.second_difference_matrix(24).toarray()
    constant = np.full(24, 15.3)
    ramp = np.linspace(10.0, 20.0, 24)
    curved = np.arange(24.0) ** 2

    assert float(np.sum((L @ constant) ** 2)) < 1e-18, "a constant has no curvature"
    assert float(np.sum((L @ ramp) ** 2)) < 1e-18, "a linear ramp has no curvature"
    assert float(np.sum((L @ curved) ** 2)) > 1.0, "a parabola does"


def test_bayesian_prior_mean_is_the_background_model():
    """With the data ignored the posterior must land exactly on the background.

    This is the property the roughness-operator bug broke: the penalty pulled
    the prior mean off the background, so the "prior" the user configured was
    not the model they built.
    """
    vol, wells, ties, wav, lfm = build_case(n_wells=3)
    tie = ties[0]
    lf = lfm.trace(tie.il_index, tie.xl_index)
    for smoothness in (0.0, 0.05, 1.0):
        res = inversion.invert(vol.trace_at(tie.il_index, tie.xl_index), wav.samples, lf,
                               method="bayesian", dt=vol.dt, noise_pct=1e6,
                               smoothness=smoothness, uncertainty=False)
        drift = np.max(np.abs(res["log_impedance"] - res["background_log_impedance"]))
        assert drift < 1e-3, f"smoothness={smoothness} drifted {drift:.3g} from the background"


def test_bayesian_matches_the_dense_closed_form():
    """The banded solve must reproduce the textbook Gaussian posterior.

    Everything else about this engine rests on the banded assembly being the
    same matrix as the dense one, so it is checked directly rather than
    inferred from the answer looking plausible.
    """
    rng = np.random.default_rng(0)
    n, dt = 60, 0.002
    w = wvl.ricker(30, 80, dt)
    G = (inversion.convolution_matrix(w, n) @ inversion.difference_matrix(n)).toarray()
    m_true = np.log(5e6) + np.cumsum(rng.normal(0, 0.02, n))
    d = G @ m_true + rng.normal(0, 0.002, n)
    m0 = np.full(n, np.log(5e6))

    prior_std, smoothness, noise_pct = 0.09, 0.07, 12.0
    sigma_d = float(np.sqrt(np.mean(d ** 2))) * noise_pct / 100.0

    L = inversion.second_difference_matrix(n).toarray()
    Q = (np.eye(n) + smoothness * (L.T @ L)) / prior_std ** 2
    A = G.T @ G / sigma_d ** 2 + Q
    rhs = G.T @ d / sigma_d ** 2 + m0 / prior_std ** 2
    m_dense = np.linalg.solve(A, rhs)
    var_dense = np.diag(np.linalg.inv(A))

    res = inversion.bayesian_inversion(d, w, np.exp(m0), dt=dt, prior_std=prior_std,
                                       smoothness=smoothness, noise_pct=noise_pct)
    assert np.max(np.abs(res["log_impedance"] - m_dense)) < 1e-8
    assert np.max(np.abs(res["posterior_std"] ** 2 - var_dense) / var_dense) < 1e-8


def test_bayesian_beats_the_background_model():
    vol, wells, ties, wav, lfm = build_case()
    tie = ties[0]
    lf = lfm.trace(tie.il_index, tie.xl_index)
    res = inversion.invert(vol.trace_at(tie.il_index, tie.xl_index), wav.samples, lf,
                           method="bayesian", dt=vol.dt)
    baseline = band_score(lf, tie.ai, vol.dt)
    score = band_score(res["absolute_ai"], tie.ai, vol.dt)
    assert score > baseline + 0.1, f"bayesian {score:.3f} vs background {baseline:.3f}"


def test_bayesian_recovers_absolute_impedance_at_least_as_well_as_model_based():
    """The point of the closed form is a better-calibrated answer, not just a faster one."""
    vol, wells, ties, wav, lfm = build_case(n_wells=4)
    def rmse(engine, **kw):
        out = []
        for tie in ties:
            lf = lfm.trace(tie.il_index, tie.xl_index)
            ai = inversion.invert(vol.trace_at(tie.il_index, tie.xl_index), wav.samples, lf,
                                  method=engine, dt=vol.dt, **kw)["absolute_ai"]
            good = np.isfinite(tie.ai) & (tie.ai > 0) & np.isfinite(ai) & (ai > 0)
            lw = np.log(utils.fill_nan_1d(np.where(good, tie.ai, np.nan)))
            out.append(float(np.sqrt(np.mean((np.log(ai)[good] - lw[good]) ** 2))))
        return float(np.mean(out))
    bayes = rmse("bayesian", uncertainty=False)
    mb = rmse("model-based", model_weight=0.1, max_iter=200)
    assert bayes <= mb, f"bayesian log-AI RMSE {bayes:.4f} worse than model-based {mb:.4f}"


def test_bayesian_posterior_lies_between_prior_and_data():
    """Turning the noise up must walk the answer back to the prior, and down must fit."""
    vol, wells, ties, wav, lfm = build_case(n_wells=3)
    tie = ties[0]
    lf = lfm.trace(tie.il_index, tie.xl_index)
    trace = vol.trace_at(tie.il_index, tie.xl_index)

    loud = inversion.invert(trace, wav.samples, lf, method="bayesian", dt=vol.dt,
                            noise_pct=1e6, uncertainty=False)
    assert np.max(np.abs(loud["log_impedance"] - loud["background_log_impedance"])) < 1e-3, \
        "with the data all but ignored the posterior should collapse onto the prior mean"
    assert loud["prior_drift"] < 1e-3

    quiet = inversion.invert(trace, wav.samples, lf, method="bayesian", dt=vol.dt,
                             noise_pct=1.0, uncertainty=False)
    mid = inversion.invert(trace, wav.samples, lf, method="bayesian", dt=vol.dt,
                           noise_pct=20.0, uncertainty=False)
    assert quiet["misfit"] < mid["misfit"] < loud["misfit"], \
        (quiet["misfit"], mid["misfit"], loud["misfit"])


def test_bayesian_uncertainty_is_reduced_by_the_data():
    vol, wells, ties, wav, lfm = build_case(n_wells=3)
    tie = ties[0]
    res = inversion.invert(vol.trace_at(tie.il_index, tie.xl_index), wav.samples,
                           lfm.trace(tie.il_index, tie.xl_index), method="bayesian",
                           dt=vol.dt, prior_std=0.08)
    std = res["posterior_std"]
    assert np.all(std > 0)
    assert np.all(std <= 0.08 + 1e-9), "the posterior can never be less certain than the prior"
    assert 0.0 < res["uncertainty_reduction"] < 1.0
    # Louder data assumptions must leave more uncertainty behind.
    noisy = inversion.invert(vol.trace_at(tie.il_index, tie.xl_index), wav.samples,
                             lfm.trace(tie.il_index, tie.xl_index), method="bayesian",
                             dt=vol.dt, prior_std=0.08, noise_pct=40.0)
    assert np.mean(noisy["posterior_std"]) > np.mean(std)


def test_bayesian_quantiles_bracket_the_mean():
    vol, wells, ties, wav, lfm = build_case(n_wells=2)
    tie = ties[0]
    res = inversion.invert(vol.trace_at(tie.il_index, tie.xl_index), wav.samples,
                           lfm.trace(tie.il_index, tie.xl_index), method="bayesian", dt=vol.dt)
    assert np.all(res["ai_p10"] < res["absolute_ai"])
    assert np.all(res["absolute_ai"] < res["ai_p90"])
    assert np.all(res["ai_p10"] > 0), "log-normal quantiles must stay positive"


def test_bayesian_requires_a_background_model():
    vol, wells, ties, wav, lfm = build_case(n_wells=2)
    try:
        inversion.invert(vol.trace_at(0, 0), wav.samples, None, method="bayesian", dt=vol.dt)
    except ValueError as exc:
        assert "low-frequency" in str(exc)
    else:
        raise AssertionError("Bayesian inversion must refuse to run without a prior mean")


def test_bayesian_volume_run_carries_the_uncertainty_cube():
    vol, wells, ties, wav, lfm = build_case(n_wells=3)
    res = inversion.run_volume(vol, "bayesian", wav.samples, lfm)
    assert res.posterior_std is not None
    assert res.posterior_std.shape == res.relative_ai.shape
    assert np.all(res.posterior_std[np.isfinite(res.correlation)] > 0)
    assert res.summary()["uncertainty"] == "yes"

    off = inversion.run_volume(vol, "bayesian", wav.samples, lfm, uncertainty=False)
    assert off.posterior_std is None
    assert off.summary()["uncertainty"] == "no"


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
