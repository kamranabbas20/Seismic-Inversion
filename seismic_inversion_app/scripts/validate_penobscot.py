"""Validate the inversion engines against real, openly-licensed field data.

Everything in ``tests/`` is synthetic: the answer is known, so the engines can
be scored exactly.  That is necessary but not sufficient -- synthetic data has
a perfect time-depth relationship, a stationary wavelet and no drift, so it
cannot tell you whether the app copes with a real well.  This script closes
that gap using the Penobscot 3D survey, offshore Nova Scotia.

Data (both fetched by ``--fetch``, neither redistributed here):

  seismic  crossline 1155 of the Penobscot 3D, from the SEG tutorial repo
           ``seg/tutorials-2014`` (1410_Phase).  Owned by the Nova Scotia
           Department of Energy, distributed by permission via dGB's Open
           Seismic Repository.
  well     Penobscot L-30, from the same repo (1406_Make_a_synthetic), the
           data for Evan Bianco's Leading Edge tutorial "How to make a
           synthetic".  Logs digitised by Neil Watson; released by CNSOPB for
           non-commercial knowledge sharing.

Why this pair: L-30 sits ~40 m off crossline 1155, and it is the only
open well/seismic combination the author is aware of that ships together with
published tie parameters, so the geometry can be checked against a third
party rather than asserted.

What is and is not blind
------------------------
With a single well the low-frequency model is built from the same log the
result is scored against, so the full-band correlation is not a blind test and
is reported only for context.  The honest number is the correlation in the
10-60 Hz band, above the low-frequency model's cutoff: the background model
carries no information there, so anything the inversion scores in that band it
recovered from the seismic.  The seismic residual is blind everywhere -- it
measures whether the inverted model reproduces the recorded trace, at all 601
traces, with no reference to the well at all.

Usage::

    python scripts/validate_penobscot.py --fetch      # clone the two repos
    python scripts/validate_penobscot.py              # run the comparison
    python scripts/validate_penobscot.py --figure out.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import data_io, inversion, low_freq_model as lfmod, utils, wavelet as wv  # noqa: E402

REPOS = {"seg/tutorials-2014": "tutorials-2014"}
SEISMIC = "tutorials-2014/1410_Phase/data/penobscot_xl1155.sgy"
WELL = "tutorials-2014/1406_Make_a_synthetic/L-30.las"

# Published in the SEG tutorial notebook; used here only to check our own
# reading of the LAS header, never as an input.
TUTORIAL = {"kb_m": 30.175, "water_m": 137.46, "rock_m": 179.83}

# L-30 surface location.  The LAS truncates the longitude seconds field
# ("60 04' 0..."), so the last digit is resolved by closest approach to the
# line -- the well is known to sit on it.
L30_LAT = 44 + 9 / 60 + 43.558 / 3600
L30_LON_MIN = -(60 + 4 / 60)

WATER_VELOCITY = 1480.0      # m/s, tutorial value
REPLACEMENT_VELOCITY = 1600.0


def fetch(root: str) -> None:
    for repo, dest in REPOS.items():
        path = os.path.join(root, dest)
        if os.path.isdir(path):
            print(f"  {dest}: already present")
            continue
        print(f"  cloning {repo} ...")
        subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                        f"https://github.com/{repo}", path], check=True)
        subprocess.run(["git", "-C", path, "sparse-checkout", "set",
                        "1406_Make_a_synthetic", "1410_Phase/data"], check=True)


def utm20n(lat_deg: float, lon_deg: float) -> tuple[float, float]:
    """Forward UTM (WGS84, zone 20N) -- the survey's own coordinate system."""
    a, f = 6378137.0, 1 / 298.257223563
    e2 = f * (2 - f)
    ep2 = e2 / (1 - e2)
    k0 = 0.9996
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    lon0 = np.radians(20 * 6 - 183)
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    T = np.tan(lat) ** 2
    C = ep2 * np.cos(lat) ** 2
    A = (lon - lon0) * np.cos(lat)
    M = a * ((1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256) * lat
             - (3 * e2 / 8 + 3 * e2 ** 2 / 32 + 45 * e2 ** 3 / 1024) * np.sin(2 * lat)
             + (15 * e2 ** 2 / 256 + 45 * e2 ** 3 / 1024) * np.sin(4 * lat)
             - (35 * e2 ** 3 / 3072) * np.sin(6 * lat))
    east = k0 * N * (A + (1 - T + C) * A ** 3 / 6
                     + (5 - 18 * T + T ** 2 + 72 * C - 58 * ep2) * A ** 5 / 120) + 500000.0
    north = k0 * (M + N * np.tan(lat) * (A ** 2 / 2 + (5 - T + 9 * C + 4 * C ** 2) * A ** 4 / 24
                  + (61 - 58 * T + T ** 2 + 600 * C - 330 * ep2) * A ** 6 / 720))
    return float(east), float(north)


def _las_header_float(path: str, mnemonic: str, default: float) -> float:
    """One numeric value out of a LAS ~Well section."""
    import lasio

    for item in lasio.read(path).well:
        if item.mnemonic.upper() == mnemonic.upper():
            try:
                return float(item.value)
            except (TypeError, ValueError):
                break
    return default


def crop(volume, t0: float, t1: float):
    keep = (volume.twt >= t0) & (volume.twt <= t1)
    return data_io.SeismicVolume(
        data=volume.data[:, :, keep], iline=volume.iline, xline=volume.xline,
        twt=volume.twt[keep], cdp_x=volume.cdp_x, cdp_y=volume.cdp_y,
        source=volume.source, coord_scalar=volume.coord_scalar)


def head(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("PENOBSCOT_DIR", "penobscot_data"))
    ap.add_argument("--fetch", action="store_true", help="clone the public data repos first")
    ap.add_argument("--figure", default="", help="write a PNG summary here")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    if args.fetch:
        head("FETCHING OPEN DATA")
        os.makedirs(args.data, exist_ok=True)
        fetch(args.data)
    seis_path, las_path = os.path.join(args.data, SEISMIC), os.path.join(args.data, WELL)
    for p in (seis_path, las_path):
        if not os.path.isfile(p):
            print(f"missing {p}\nrun with --fetch first", file=sys.stderr)
            return 2

    head("1. REAL SEISMIC  --  Penobscot 3D, crossline 1155")
    vol = crop(data_io.load_segy(seis_path), 300.0, 3100.0)
    dt, twt = vol.dt, vol.twt
    print(f"  {vol.data.shape[0]} traces (inline {vol.iline[0]}-{vol.iline[-1]}), "
          f"{twt.size} samples, dt {dt * 1000:.0f} ms, {twt[0]:.0f}-{twt[-1]:.0f} ms")

    head("2. REAL WELL  --  Penobscot L-30, and a check of our header reading")
    well = data_io.load_las(las_path, name="PENOBSCOT L-30",
                            replacement_velocity=REPLACEMENT_VELOCITY)
    # Seafloor elevation, quoted in the same units as the depth index.
    gl_ft = _las_header_float(las_path, "GL", -451.0)
    water_m = abs(gl_ft) / utils.FT_PER_M
    top_m = well.md[0] - well.kb
    rock_m = top_m - water_m
    print(f"  depth index read as {well.depth_unit or 'unlabelled'!r} -> "
          f"md {well.md.min():.0f}-{well.md.max():.0f} m")
    for label, got, want in (("KB (m)", well.kb, TUTORIAL["kb_m"]),
                             ("water column (m)", water_m, TUTORIAL["water_m"]),
                             ("rock to top of log (m)", rock_m, TUTORIAL["rock_m"])):
        flag = "OK" if abs(got - want) < 0.5 else "MISMATCH"
        print(f"  {label:<24s} {got:9.3f}   SEG tutorial {want:8.3f}   {flag}")

    # Bridge the datum: water at 1480 m/s, rock above the log at 1600 m/s.
    # The app integrates the whole gap at one replacement velocity, so the
    # difference goes in as the initial bulk shift.
    t_true = 2.0 * (water_m / WATER_VELOCITY + rock_m / REPLACEMENT_VELOCITY) * 1000.0
    t_app = 2.0 * (top_m / REPLACEMENT_VELOCITY) * 1000.0
    well.set_bulk_shift(t_true - t_app)

    head("3. LOCATING L-30 ON THE LINE")
    xy = vol.trace_xy()
    best_d, best_sec = None, 0
    for sec in range(10):
        x, y = utm20n(L30_LAT, L30_LON_MIN - sec / 3600)
        d = float(np.hypot(xy[:, 0] - x, xy[:, 1] - y).min())
        if best_d is None or d < best_d:
            best_d, best_sec = d, sec
    well.x, well.y = utm20n(L30_LAT, L30_LON_MIN - best_sec / 3600)
    print(f"  lat/lon {L30_LAT:.6f} N, 60 04' {best_sec:02d}\" W  ->  "
          f"UTM20N {well.x:,.1f} E {well.y:,.1f} N")
    print(f"  closest trace is {best_d:.1f} m away")

    head("4. WELL TIE  --  bulk shift, then bulk shift + sonic drift")
    base = well._twt_base.copy()
    t_ref = float(np.nanmin(base))
    probe = wv.ricker(25.0, 120.0, dt)   # fixed, so the tie cannot hide inside the wavelet

    def place(t0: float, alpha: float):
        well.twt = well._twt_base = t_ref + t0 + alpha * (base - t_ref)
        return data_io.extract_well_traces(vol, [well], k=4)[0]

    def score(tie, gate):
        syn = utils.convolve_same(np.nan_to_num(tie.reflectivity), probe)
        return utils.normalised_correlation(syn[gate], tie.seismic[gate])

    # Broad gate for the tie search, from the well as first placed; the final
    # gate is re-derived from the tied well below.
    _ai0 = well.ai_on_time_axis(twt)
    wide = np.isfinite(_ai0) & (_ai0 > 0)
    shift_only = max(((score(place(t0, 1.0), wide), t0) for t0 in np.arange(-60, 61, 4.0)))
    best = max(((score(place(t0, al), wide), t0, al)
                for t0 in np.arange(shift_only[1] - 30, shift_only[1] + 31, 2.0)
                for al in np.arange(0.86, 1.15, 0.01)))
    tie = place(best[1], best[2])
    print(f"  bulk shift only         corr {shift_only[0]:+.3f}  ({shift_only[1]:+.0f} ms)")
    print(f"  + sonic-drift stretch   corr {best[0]:+.3f}  ({best[1]:+.0f} ms, {best[2]:.3f}x)")
    print(f"  drift correction is worth {best[0] - shift_only[0]:+.3f}; the app's tie step offers")
    print("  bulk shift only, so this had to be done outside it.")

    # Gate: where the *tied* well has Vp and density together.
    ok = np.isfinite(tie.ai) & (tie.ai > 0)
    t0, t1 = float(twt[ok].min()), float(twt[ok].max())
    gate = (twt >= t0) & (twt <= t1)
    print(f"  analysis gate {t0:.0f}-{t1:.0f} ms ({int(gate.sum())} samples, "
          "set by the density log, which starts below the sonic)")

    wave = wv.wiener_wavelet(tie.seismic[gate], np.nan_to_num(tie.reflectivity)[gate],
                             dt, length_ms=120.0, prewhitening=1.0)
    wave = wv.calibrate_amplitude(wave, [tie], twt, t_min=t0, t_max=t1)
    print(f"  Wiener wavelet: tie corr {wave.quality.get('tie correlation', float('nan')):+.3f}, "
          f"constant phase {wv.estimate_constant_phase(wave.samples):+.0f} deg, "
          f"amplitude scalar {wave.quality['amplitude scalar']:.4g}")

    head("5. LOW-FREQUENCY MODEL AND COLOUR OPERATOR")
    lf = lfmod.build_low_frequency_model(vol, [well], cutoff_hz=10.0, method="idw")
    op = inversion.design_colour_operator(vol, [tie], f_low=8.0, f_high=60.0)
    op = inversion.calibrate_colour_operator(op, vol, [tie], t_min=t0, t_max=t1)
    print(f"  LFM median AI {np.nanmedian(lf.ai):,.0f}   "
          f"colour operator beta {op.exponent:+.3f}, scalar {op.scalar:.4g}")

    head(f"6. INVERTING THE REAL LINE  ({vol.data.shape[0]} traces x {twt.size} samples)")
    res = {}
    for method in inversion.METHODS:
        started = time.time()
        extra = {"operator": op} if method == "coloured" else {}
        res[method] = inversion.run_volume(vol, method, wavelet=wave.samples,
                                           low_freq_model=lf, **extra)
        el = time.time() - started
        print(f"  {method:<13s}{el:7.1f} s  ({el / vol.data.shape[0] * 1000:6.1f} ms/trace)")

    head(f"7. SCORES AT L-30  ({int(gate.sum())} samples, {t0:.0f}-{t1:.0f} ms)")
    k = tie.il_index
    good = gate & ok
    band = lambda x: utils.bandpass(x, dt, 10.0, 60.0)  # noqa: E731
    seis_rms = float(np.sqrt(np.mean(vol.data[:, 0, gate] ** 2)))
    print(f"  {'':<20s}{'corr 10-60Hz':>14s}{'corr full':>11s}{'RMSE lnAI':>11s}{'seis resid':>12s}")
    print("  " + "-" * 66)
    rows = [("low-freq model only", lf.ai[k, 0, :], None)]
    rows += [(m, res[m].absolute_ai[k, 0, :], res[m]) for m in inversion.METHODS]
    for name, ai, rr in rows:
        cb = utils.normalised_correlation(band(ai)[good], band(tie.ai)[good])
        cf = utils.normalised_correlation(ai[good], tie.ai[good])
        rmse = float(np.sqrt(np.mean((np.log(np.clip(ai[good], 1, None))
                                      - np.log(tie.ai[good])) ** 2)))
        if rr is not None and rr.method != "coloured":
            resid = float(np.sqrt(np.mean(rr.residual[:, 0, gate] ** 2)) / seis_rms * 100)
            resid_s = f"{resid:11.1f}%"
        else:
            resid_s = f"{'n/a':>12s}"
        print(f"  {name:<20s}{cb:14.3f}{cf:11.3f}{rmse:11.4f}{resid_s}")
    print("\n  corr 10-60 Hz is the blind number: the low-frequency model carries no")
    print("  information above 10 Hz, so that band is recovered from the seismic alone.")
    print("  corr full and RMSE lnAI are NOT blind -- with one well the background model")
    print("  is built from the same log -- and are shown only for context.")

    if args.figure:
        _figure(args.figure, vol, tie, lf, res, good, band, k)
        print(f"\n  figure written to {args.figure}")
    return 0


def _figure(path, vol, tie, lf, res, good, band, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    twt = vol.twt
    t = twt[good]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [2, 2, 1.3]})
    ext = [vol.iline[0], vol.iline[-1], t.max(), t.min()]
    sl = slice(int(np.argmax(good)), int(len(good) - np.argmax(good[::-1])))
    s = vol.data[:, 0, sl].T
    c = np.percentile(np.abs(s), 99)
    axes[0].imshow(s, cmap="RdBu_r", aspect="auto", extent=ext, vmin=-c, vmax=c)
    axes[0].set_title("Real seismic: Penobscot crossline 1155")
    axes[0].set_ylabel("TWT (ms)")
    a = res["bayesian"].absolute_ai[:, 0, sl].T
    v0, v1 = np.percentile(a, [2, 98])
    im = axes[1].imshow(a / 1e6, cmap="viridis", aspect="auto", extent=ext,
                        vmin=v0 / 1e6, vmax=v1 / 1e6)
    axes[1].set_title("Bayesian inversion: absolute impedance")
    plt.colorbar(im, ax=axes[1]).set_label("AI (10$^6$ m/s·kg/m³)")
    for ax in axes[:2]:
        ax.axvline(vol.iline[k], color="k", ls="--", lw=1)
        ax.set_xlabel("inline")
    axes[2].plot(band(tie.ai)[good] / 1e6, t, "k", lw=1.8, label="L-30 well", zorder=5)
    for m, col in [("coloured", "#2ca02c"), ("sparse-spike", "#ff7f0e"),
                   ("model-based", "#1f77b4"), ("bayesian", "#d62728")]:
        axes[2].plot(band(res[m].absolute_ai[k, 0, :])[good] / 1e6, t, col, lw=1, label=m)
    axes[2].invert_yaxis()
    axes[2].set_title("10-60 Hz at the well")
    axes[2].set_xlabel("band-limited AI (10$^6$)")
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    raise SystemExit(main())
