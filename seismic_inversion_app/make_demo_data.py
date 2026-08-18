#!/usr/bin/env python3
"""Write a synthetic demo dataset to disk as real SEG-Y / LAS / CSV files.

The app has a built-in synthetic mode, but that path skips the file readers.
This script writes the same model out as actual files so a reviewer can
exercise the *upload* workflow -- byte-position config, LAS parsing, header
matching, horizon gridding -- without proprietary data.

    python make_demo_data.py --outdir demo_data

Produces:

    demo_seismic.sgy     3D post-stack volume (inline 189, xline 193, X 181, Y 185)
    SYN-1.las ...        one LAS per well, with DT, RHOB and a TWT curve
    well_headers.csv     well, x, y, kb
    horizons.csv         iline, xline, and two interpreted horizons in ms
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import data_io, utils  # noqa: E402


def write_las(well, path: str) -> None:
    """Write a well out as LAS 2.0 with DT, RHOB and a TWT curve.

    A TWT curve is included deliberately: the app treats wells as tied upstream,
    and this is how that tie arrives in a real project.
    """
    import lasio

    las = lasio.LASFile()
    las.well["WELL"] = lasio.HeaderItem("WELL", value=well.name)
    las.well["UWI"] = lasio.HeaderItem("UWI", value=well.uwi)
    las.well["KB"] = lasio.HeaderItem("KB", unit="m", value=float(well.kb))
    if well.has_location:
        las.params["XCOORD"] = lasio.HeaderItem("XCOORD", unit="m", value=float(well.x))
        las.params["YCOORD"] = lasio.HeaderItem("YCOORD", unit="m", value=float(well.y))

    good = well.valid_mask()
    md = well.md[good]
    # Sonic is stored the way logging companies store it: slowness in us/ft.
    dt_us_ft = 1.0e6 / (well.vp[good] * utils.FT_PER_M)

    las.append_curve("DEPT", md, unit="m", descr="Measured depth")
    las.append_curve("DT", dt_us_ft, unit="us/ft", descr="Sonic slowness")
    las.append_curve("RHOB", well.rho[good] / 1000.0, unit="g/cm3", descr="Bulk density")
    las.append_curve("TWT", well.twt[good], unit="ms", descr="Two-way time (well already tied)")
    las.write(path, version=2.0)


def write_headers_csv(wells, path: str) -> None:
    import pandas as pd

    pd.DataFrame([
        {"well": w.name, "x": w.x, "y": w.y, "kb": w.kb}
        for w in wells if w.has_location
    ]).to_csv(path, index=False)


def write_horizons_csv(volume, path: str) -> None:
    """Two smooth horizons that follow the model's structure.

    They are derived from the same dip and fold used to build the cube, so
    horizon-guided interpolation in the low-frequency model has something real
    to follow.
    """
    import pandas as pd

    n_il, n_xl = volume.data.shape[:2]
    ii, jj = np.meshgrid(np.arange(n_il), np.arange(n_xl), indexing="ij")
    structure = (0.55 * ii + 0.30 * jj
                 + 9.0 * np.sin(2 * np.pi * ii / max(n_il, 1))
                 + 6.0 * np.cos(2 * np.pi * jj / max(n_xl, 1) * 1.5))
    structure = structure - structure.mean()
    dt_ms = volume.sample_rate_ms
    span = float(volume.twt.max() - volume.twt.min())

    rows = {
        "iline": np.repeat(volume.iline, n_xl),
        "xline": np.tile(volume.xline, n_il),
        "TOP_RESERVOIR": (volume.twt.min() + 0.35 * span + structure * dt_ms).ravel(),
        "BASE_RESERVOIR": (volume.twt.min() + 0.60 * span + structure * dt_ms).ravel(),
    }
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="demo_data")
    ap.add_argument("--inlines", type=int, default=40)
    ap.add_argument("--crosslines", type=int, default=40)
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--sample-rate", type=float, default=2.0, help="ms")
    ap.add_argument("--wells", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.08, help="fraction of RMS")
    ap.add_argument("--frequency", type=float, default=28.0, help="wavelet peak frequency, Hz")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Generating {args.inlines} x {args.crosslines} x {args.samples} "
          f"with {args.wells} wells...")
    volume, wells = data_io.make_synthetic_dataset(
        n_iline=args.inlines, n_xline=args.crosslines, n_samples=args.samples,
        sample_rate_ms=args.sample_rate, n_wells=args.wells, noise=args.noise,
        peak_frequency=args.frequency, seed=args.seed)

    segy_path = os.path.join(args.outdir, "demo_seismic.sgy")
    data_io.write_segy(volume, segy_path)
    print(f"  {segy_path}  ({os.path.getsize(segy_path) / 1e6:.1f} MB)")

    for well in wells:
        las_path = os.path.join(args.outdir, f"{well.name}.las")
        write_las(well, las_path)
        print(f"  {las_path}")

    headers = os.path.join(args.outdir, "well_headers.csv")
    write_headers_csv(wells, headers)
    print(f"  {headers}")

    horizons = os.path.join(args.outdir, "horizons.csv")
    write_horizons_csv(volume, horizons)
    print(f"  {horizons}")

    print("\nLoad in the app with byte positions: inline 189, crossline 193, "
          "CDP X 181, CDP Y 185.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
