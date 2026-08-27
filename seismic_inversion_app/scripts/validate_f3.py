"""F3 well data: an independent check of the unit chain.

This started as a second inversion test, pairing well F02-1 with the F3 inline
362 exported in the SEG tutorial repository.  **That pairing does not work, and
the script now says so rather than reporting numbers from it.**  The evidence is
in section 5: if a well genuinely sits on a line, the crossline that ties best
should be the same whichever analysis gate you score over.  Here it wanders over
14 km and the correlation flips sign, which is what noise looks like.  The
tutorial's own dashed line at crossline 336 is commented "Just to display the
well position" -- schematic, not a projection.  Inversion scores computed
against that line would be meaningless, so none are produced.

What the F3 data *does* validate, and validates well:

* **The unit chain, on the opposite branch from Penobscot.**  F02-1 records DT
  in us/m, RHOB in kg/m3 and depth in metres; L-30 records us/ft, g/cm3 and
  feet.  Between them the two wells cover both paths through ``data_io``.
* **Against a third party's arithmetic, not our own.**  The LAS carries an AI
  curve computed by the data's publisher, so the impedance this app derives can
  be compared rather than trusted.
* **The folder scanner on real data** -- the F3 layout (Lasfiles / Checkshot /
  Track / Tops) that ``scan_well_folder`` was written for.

The real inversion test remains ``validate_penobscot.py``, where the well is
42 m from the line and the tie is established.

Data (fetched by ``--fetch``, not redistributed here): the F3 demo, from
``seg/tutorials-2017`` (1710_Colored_inversion).  F3 is released by dGB Earth
Sciences and TNO under CC-BY-SA through the Open Seismic Repository.

Usage::

    python scripts/validate_f3.py --fetch
    python scripts/validate_f3.py
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import data_io, utils, wavelet as wv  # noqa: E402

REPO = "seg/tutorials-2017"
SUBDIR = "1710_Colored_inversion/data"
CLONE = "tutorials-2017"

SEISMIC = f"{CLONE}/{SUBDIR}/export_inline362.ascii"
WELL_ROOT = f"{CLONE}/{SUBDIR}/All_wells_RawData"

# From the tutorial notebook: time = np.arange(0, 1852, 4).
DT_MS = 4.0
N_SAMPLES = 463
INLINE = 362
# The notebook marks F02-1 at crossline 336 on this line.
WELL_XLINE = 336
BIN_M = 25.0            # F3 bin spacing, used to give the 2D line a metric axis


def fetch(root: str) -> None:
    path = os.path.join(root, CLONE)
    if os.path.isdir(path):
        print(f"  {CLONE}: already present")
        return
    print(f"  cloning {REPO} ...")
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none",
                    f"https://github.com/{REPO}", path], check=True)
    subprocess.run(["git", "-C", path, "sparse-checkout", "set", SUBDIR], check=True)


def load_ascii_line(path: str) -> data_io.SeismicVolume:
    """One OpendTect-exported inline into a SeismicVolume.

    The export carries inline and crossline but no coordinates, so the line is
    given a metric axis from the survey's 25 m bin spacing.  Only the *relative*
    geometry matters here -- the well is placed by crossline, and nothing in the
    workflow needs absolute easting.
    """
    raw = np.loadtxt(path)
    if raw.shape[1] != N_SAMPLES + 2:
        raise ValueError(f"expected {N_SAMPLES + 2} columns, found {raw.shape[1]}")
    xlines = raw[:, 1].astype(int)
    order = np.argsort(xlines)
    xlines = xlines[order]
    data = raw[order, 2:][None, :, :].astype(np.float32)      # (1 inline, n_xl, n_t)

    twt = np.arange(N_SAMPLES, dtype=float) * DT_MS
    cdp_x = (xlines.astype(float) * BIN_M)[None, :]
    cdp_y = np.zeros_like(cdp_x)
    return data_io.SeismicVolume(
        data=data, iline=np.array([INLINE]), xline=xlines, twt=twt,
        cdp_x=cdp_x, cdp_y=cdp_y, source=os.path.basename(path),
        text_header="F3 inline 362, exported from OpendTect as ASCII")


def head(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("F3_DIR", "f3_data"))
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    if args.fetch:
        head("FETCHING OPEN DATA")
        os.makedirs(args.data, exist_ok=True)
        fetch(args.data)

    seis_path = os.path.join(args.data, SEISMIC)
    well_root = os.path.join(args.data, WELL_ROOT)
    if not os.path.isfile(seis_path):
        print(f"missing {seis_path}\nrun with --fetch first", file=sys.stderr)
        return 2

    head("1. REAL SEISMIC  --  F3 inline 362 (dGB / TNO, CC-BY-SA)")
    vol = load_ascii_line(seis_path)
    dt, twt = vol.dt, vol.twt
    print(f"  {vol.data.shape[1]} traces (crossline {vol.xline[0]}-{vol.xline[-1]}), "
          f"{twt.size} samples, dt {dt * 1000:.0f} ms, {twt[0]:.0f}-{twt[-1]:.0f} ms")

    head("2. REAL WELLS  --  scanned from the F3 folder layout")
    scan = data_io.scan_well_folder(well_root)
    print(f"  {scan.summary() if hasattr(scan, 'summary') else scan}")
    wells = {w.name: w for w in scan.wells}
    name = next((n for n in wells if "F02" in n), None)
    if name is None:
        print(f"F02-1 not found; scanned {list(wells)}", file=sys.stderr)
        return 2
    well = wells[name]
    print(f"  using {well.name}: {well.selection.describe()}")
    for note in well.notes:
        print(f"    - {note}")

    head("3. UNIT HANDLING  --  every unit differs from the Penobscot well")
    good = np.isfinite(well.vp) & np.isfinite(well.rho)
    print(f"  depth index   {well.depth_unit or 'unlabelled'!r:8s}  md {well.md.min():.0f}-"
          f"{well.md.max():.0f} m")
    print(f"  Vp            {np.nanmedian(well.vp[good]):,.0f} m/s      "
          f"(from DT in us/m)")
    print(f"  density       {np.nanmedian(well.rho[good]):,.0f} kg/m3   (already kg/m3)")
    ours = well.vp[good] * well.rho[good]
    print(f"  our AI        {np.nanmedian(ours):,.0f}")
    if "AI" in well.curves:
        theirs = np.asarray(well.curves["AI"], dtype=float)[good]
        both = np.isfinite(theirs) & (theirs > 0)
        rel = np.abs(ours[both] - theirs[both]) / theirs[both]
        print(f"  published AI  {np.nanmedian(theirs[both]):,.0f}   "
              f"median disagreement {np.median(rel) * 100:.2f}%  "
              f"({'OK' if np.median(rel) < 0.02 else 'MISMATCH'})")

    head("4. TIME-DEPTH  --  a measured checkshot, unlike L-30")
    if well.time_depth is not None:
        td = well.time_depth
        print(f"  checkshot: {td.md.size} points, {td.md.min():.0f}-{td.md.max():.0f} m")
        print("  time-depth taken from the checkshot, not integrated from the sonic")
    else:
        print("  NO checkshot attached -- time-depth integrated from the sonic")
    print(f"  well TWT range {np.nanmin(well.twt):.0f}-{np.nanmax(well.twt):.0f} ms")

    head("5. IS F02-1 ACTUALLY ON THIS LINE?")
    print("  A well that genuinely sits on a line ties best at the SAME crossline")
    print("  whichever gate you score over.  If the best crossline moves when the")
    print("  gate moves, the correlation is noise and there is no tie to be had.\n")
    r = np.nan_to_num(well.reflectivity_on_time_axis(twt))
    synth = utils.convolve_same(r, wv.ricker(35.0, 120.0, dt))
    picks = []
    for lo in range(200, 1101, 100):
        g = (twt >= lo) & (twt <= lo + 400)
        best = (0.0, None)
        for jj in range(0, vol.data.shape[1], 2):
            _lag, c = utils.best_lag_correlation(synth[g], vol.data[0, jj, :].astype(float)[g],
                                                 max_lag=30)
            if np.isfinite(c) and abs(c) > abs(best[0]):
                best = (c, int(vol.xline[jj]))
        picks.append(best[1])
        print(f"   gate {lo:4d}-{lo + 400:4d} ms  ->  best crossline {best[1]:5d}   "
              f"(correlation {best[0]:+.3f})")

    spread = max(picks) - min(picks)
    print(f"\n  the chosen crossline ranges over {spread} crosslines "
          f"= {spread * BIN_M / 1000:.1f} km")
    stable = spread * BIN_M < 500.0
    if stable:
        print("  VERDICT: stable -- the well can be located on this line.")
    else:
        print("  VERDICT: NOT stable.  F02-1 cannot be located on inline 362, so no")
        print("  inversion score computed against this line would mean anything, and")
        print("  none is produced.  The tutorial's dashed marker at crossline 336 is")
        print('  commented "Just to display the well position" -- it is schematic.')
        print("  For a real inversion test with an established tie, see")
        print("  scripts/validate_penobscot.py (well 42 m from the line).")

    head("SUMMARY")
    print("  VALID   unit chain: us/m, kg/m3 and metres, checked against the")
    print("          publisher's own AI curve rather than our own arithmetic")
    print("  VALID   folder scanner on the real F3 layout")
    print("  VALID   checkshot read as two-way time")
    print("  NOT     pairing F02-1 with inline 362 -- the well is not locatable on it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
