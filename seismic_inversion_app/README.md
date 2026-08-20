# Post-Stack Seismic Inversion

A Streamlit application that loads 3D post-stack SEG-Y volumes and multi-well
LAS logs, then inverts for acoustic impedance by one of three methods:
**coloured inversion**, **sparse-spike inversion**, and **model-based
inversion**. Results are viewable as sections, time slices and crossplots, and
exportable as SEG-Y, NumPy or NetCDF.

---

## Install and run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

Python 3.10 or newer is required (the code uses `X | None` annotations).

### Large volumes

Browser upload is capped at **1 GB** (set in `.streamlit/config.toml`; raise
`maxUploadSize` if you need more). Uploads are streamed to disk in chunks
rather than buffered whole, so peak memory is not doubled.

For volumes at the top of that range, step 1 also accepts a **path on the
machine running the app**, which skips the upload entirely — no size limit and
nothing held in memory twice. Note that the loaded cube still has to fit in
RAM: budget roughly the SEG-Y's own size again for the float32 array.

The app opens on **step 1 - Data**. The fastest way to see it work is to pick
**Synthetic demo dataset** and press *Generate*, then walk down the sidebar:
wavelet → low-frequency model → inversion → results. No data of your own is
needed.

### Demo files for the upload path

The in-app synthetic mode builds the volume in memory and skips the file
readers. To exercise the actual upload workflow — byte-position config, LAS
parsing, header matching, horizon gridding — write the same model out as real
files:

```bash
python make_demo_data.py --outdir demo_data
```

This produces `demo_seismic.sgy`, one `.las` per well, `well_headers.csv` and
`horizons.csv`. Load the SEG-Y with the standard byte positions (inline 189,
crossline 193, CDP X 181, CDP Y 185).

### Tests

```bash
python tests/test_pipeline.py          # or: pytest tests/
```

The tests run the whole pipeline on the synthetic cube. Because that cube is
built from a *known* impedance model, they assert something stronger than "it
ran": every engine must recover more of the true band-limited impedance than
the low-frequency model alone does. An inversion that merely reproduces its own
background model fails.

---

## Workflow

| Step | What it does |
| --- | --- |
| **1 - Data** | Three sources: a synthetic dataset, a **well folder** scanned in one go (F3 demo layout), or file-by-file upload. SEG-Y with byte positions and a header scan, multi-well LAS, time-depth / deviation / marker files, optional well-header and horizon CSVs. |
| **2 - Seismic viewer** | Inline, crossline, time slice and an arbitrary traverse through chosen wells, with gain, clip and colour-scale controls and well overlays. |
| **3 - Log QC** | Curve inventory per well, assign which curve is Vp / density / TWT and in what unit, rename wells, pass-fail sanity checks, checkshot / deviation / marker review. |
| **4 - Well correlation** | Wells side by side in a chosen order with logs, tops and the seismic trace at each, correlation lines between tops, and flattening on a datum. |
| **5 - Well tie QC** | Per-well synthetic-vs-extracted overlay, tie score table, constant bulk shift. |
| **6 - Wavelet** | Parametric, statistical or well-based extraction, with amplitude/phase spectrum QC. |
| **7 - Low-frequency model** | Well AI low-pass filtered and interpolated between wells, optionally guided by horizons. |
| **8 - Inversion** | Four methods (coloured, sparse-spike, model-based, Bayesian), method-specific parameters, single-trace QC, preview on a subset, full-volume run. |
| **9 - Results & export** | Section viewer, time slice, per-trace QC maps, posterior uncertainty and P10/P90 where available, crossplot against well logs, export. |

Step numbers are derived from one list in `app.py`, and every "see step N" in the
help text is generated from it, so inserting a page cannot leave stale numbering
behind.

Everything is held in `st.session_state`, so moving between steps never reloads
data. Changing something upstream (a bulk shift, the wells, the volume) clears
the products that depend on it rather than leaving a stale result on screen.

### Analysis gate

The sidebar carries one **analysis gate** — the time window used for wavelet
extraction, operator design, amplitude calibration and every QC statistic. It
defaults to the interval the wells actually cover. Narrowing it to the zone of
interest usually improves the wavelet.

---

## The three inversion methods

All three share one interface:

```python
invert(trace, wavelet, low_freq_trace=None, method=..., **params) -> dict
```

returning `reflectivity`, `relative_ai`, `absolute_ai`, `synthetic`,
`residual`, `misfit` and `correlation`. `run_volume()` drives any of them over
the cube (or a subset) trace by trace, in chunks, with a progress callback.

### 1. Coloured inversion

**What it does.** Estimates the operator that reshapes the seismic amplitude
spectrum to match a power-law fit of the well reflectivity spectrum
(`|R(f)| ∝ f^β`, the Lancaster & Whitcombe approach), applies it as a spectral
filter, then integrates to relative impedance.

**Assumptions.**
- The earth's reflectivity spectrum follows a power law over the design band.
- One operator is valid for the whole volume — no lateral or vertical
  variation in the wavelet.
- The seismic is zero-phase, or close enough that a zero-phase operator is
  appropriate. Coloured inversion does not correct phase.

**Limitations.**
- Output is **relative** impedance. It carries no absolute level of its own;
  absolute AI comes entirely from splicing onto the low-frequency model.
- No data misfit is defined — there is no wavelet and so no synthetic to
  compare against. The QC panels report the correlation between the shaped and
  input traces and leave misfit as `n/a` rather than printing a meaningless
  number.
- Amplitude is arbitrary until calibrated. The app fits one scalar against the
  band-limited well log-impedance; without a located well that calibration
  cannot happen and absolute impedance will be mis-scaled.

**Use it first.** It is fast (a full volume in seconds), needs no wavelet, and
is stable on noisy data. Run it before spending time on the other two.

> **Implementation note.** The classic single-step formulation folds a −90°
> rotation and a `1/f` ramp into the operator so it maps seismic directly to
> relative impedance. This implementation instead applies a zero-phase
> amplitude-shaping operator and then integrates explicitly. The two are
> mathematically equivalent — integration *is* a −90° rotation plus `1/f` — and
> splitting them makes the operator itself easier to QC.

### 2. Sparse-spike inversion

**What it does.** Writes the trace as `s = W r` (`W` the wavelet's convolution
matrix) and solves for a sparse `r` by L1-regularised deconvolution, using
iteratively reweighted least squares. Reflectivity is then integrated and
merged with the low-frequency model.

**Assumptions.**
- The earth is **blocky**: a small number of significant reflection
  coefficients separated by near-zero intervals. This is a genuine prior, and
  it is wrong for gradational sequences.
- The wavelet is known, stationary and correctly scaled.
- Noise is Gaussian; only the reflectivity is treated as sparse.

**Limitations.**
- The sparsity weight is a real trade-off, not a tuning nuisance: too low and
  it fits the noise, too high and it deletes real thin beds. Watch the misfit
  and the spike count as you move it, and remember that a misfit well below the
  data's noise level means you are inverting noise.
- IRLS converges to a local solution; it is not a global L1 solver.
- Absolute impedance is only as good as the low-frequency model beneath the
  merge frequency.

**Solver.** Each IRLS pass is a banded Cholesky solve (`scipy.linalg.solveh_banded`)
on the normal equations, whose bandwidth is set by the wavelet. That is roughly
14× faster than the conjugate-gradient equivalent on the reweighted — and
increasingly ill-conditioned — system, with identical results. Conjugate
gradient and a sparse direct solve remain available via the `solver` argument.

### 4. Bayesian linear inversion

**What it does.** Treats the same linear problem probabilistically. With a
linear operator, a Gaussian prior and Gaussian noise the posterior is Gaussian
and available in closed form, so there is no iteration:

```
A     = G'G / sigma_d^2 + Q          posterior precision
m     = A^-1 (G'd / sigma_d^2 + Q_amp m0)
Cpost = A^-1
```

The prior `Q = (I + smoothness * L'L) / prior_std^2` says two things at once:
log-impedance stays within `prior_std` of the background model, and its
curvature is penalised. Both terms are banded, so the whole posterior solves in
`O(n b^2)` — it is the *fastest* of the four engines despite being the only one
that returns a distribution.

**What you get back.** A posterior mean *and* a posterior standard deviation,
and from those P10/P90 impedance volumes. Impedance is log-normal here, so the
quantiles are exponentials of the Gaussian ones, not `mean ± k·sd`.

**Assumptions.**
- The forward problem is linear in log-impedance — true for the convolutional
  model, and the reason a closed form exists at all.
- Prior and noise are Gaussian. Real reflectivity is heavier-tailed than that;
  sparse-spike exists precisely because of it.
- `noise_pct` is an honest estimate. Set it too low and the posterior fits
  noise, and the reported uncertainty will be too tight to believe. The
  returned misfit should land near the noise level you claimed — if it lands
  well below, the claim was wrong.

**Limitations.**
- The posterior mean is an MMSE estimate: deliberately conservative, and
  shrunk toward the prior. It is not trying to be the sharpest-looking section.
- The uncertainty is conditional on the prior, the wavelet and the background
  model all being right. It measures how much the *data* constrained the answer
  within those assumptions, not whether the assumptions hold.
- Requires a low-frequency model, which serves as the prior mean.
- Computing the posterior variance costs roughly 3× the mean alone (a banded
  back-substitution per sample), and can be switched off.

Measured against the four synthetic wells it recovers absolute impedance about
as well as the model-based engine (log-impedance RMSE 0.068 against 0.069) at
roughly half the runtime, and carries an uncertainty estimate the other engines
cannot provide.

---

### 3. Model-based inversion

**What it does.** Perturbs log-impedance until `wavelet * reflectivity(impedance)`
matches the trace, regularised toward the background model (Russell & Hampson
style). The objective

```
J(m) = ½‖C D m − s‖²  +  ½ μ‖m − m₀‖²  +  ½ η‖L m‖²
```

is quadratic in `m = ln(AI)`, so the analytic gradient is exact and L-BFGS-B
converges in tens of iterations. A per-sample bound keeps the answer within a
chosen departure from the background model.

**Assumptions.**
- The low-frequency model is right. This method is regularised *toward* it, so
  a bad background model produces a confidently bad answer.
- The wavelet is known, stationary and correctly scaled.
- Impedance is smooth enough for the roughness penalty to be appropriate.

**Limitations.**
- **Requires** a low-frequency model; it will refuse to run without one.
- The model-constraint weight trades data fit against departure from the
  background. Raising it makes the result look more like the model you already
  had — the inversion appears well-behaved precisely as it stops telling you
  anything new. The crossplot QC at wells is the check that matters.
- Non-uniqueness is not quantified. There is one answer, not a posterior.

---

## Why amplitude calibration matters

Sparse-spike and model-based inversion both solve `W r = s` for `r`. If the
wavelet `W` is off by a factor `a`, every recovered reflection coefficient is
off by `1/a`, and so is the impedance contrast after integration. A
peak-normalised wavelet — the natural thing for *display* — is exactly such a
mis-scaled operator.

The app therefore fits one least-squares scalar so that
`reflectivity * wavelet` matches the extracted seismic amplitude, accumulated
over every usable well. The checkbox is on by default on step 3. Turning it off,
or running with no located well, leaves absolute impedance unreliable, and the
app says so.

The same reasoning applies to the coloured operator, which gets its own scalar
fitted against the band-limited well log-impedance.

---

## Log QC and curve assignment

LAS files are not consistent. Mnemonics vary (`DT`, `DTC`, `DTCO`, `AC`, or
something a vendor invented), unit strings are often blank or wrong, and some
files carry velocity where you expect slowness. A sonic read as `us/m` when it
is really `us/ft` scales Vp by 3.28; a density read as `g/cm3` when it is
`kg/m3` is out by 1000. Either mistake produces impedance that is confidently,
silently wrong.

**Step 2** exists to catch that before it propagates:

- **Curve inventory** — every curve in the file with its LAS unit, description,
  valid percentage and min/mean/max. You can see that `DTCO` runs 55–140 and
  `RHOZ` runs 1.9–2.7 before deciding which is which.
- **Assignment** — pick which curve fills each role (Vp, density, TWT) and its
  unit, per well or applied across all wells at once. Well names are editable
  here too, since LAS headers are frequently blank or inconsistent.
- **Sanity checks** — Vp, density and AI are range-checked, the time curve is
  checked for monotonicity, and the logged interval is checked against the
  seismic time range. Ranges are wide on purpose: the job is to catch a unit
  mistake, not to police unusual rocks.
- **Displays** — raw curves exactly as stored (for deciding what a curve *is*),
  and the converted Vp / density / AI tracks with out-of-range samples in red.

Units offered per role:

| Role | Units |
| --- | --- |
| Vp / sonic | `us/ft`, `us/m`, `m/s`, `ft/s` |
| Density | `g/cm3`, `kg/m3` |
| Time | `ms (TWT)`, `s (TWT)`, `ms (OWT)`, `s (OWT)` |

`m/s` and `ft/s` are there because plenty of files carry Vp directly rather
than slowness. OWT curves are doubled to two-way time on load.

Auto-detection on load resolves the roles and units in that order of evidence:
the **LAS unit string** first, then the curve's **magnitude** (the ranges
barely overlap), and only for the genuinely ambiguous 130–250 band does the
app-wide hint on step 1 break the tie. Everything is overridable — the guess
only has to be a good default.

A well missing a density curve still loads, and shows up as a failed check to
resolve here, rather than being rejected at the door and lost.

---

## Viewing the seismic

**Step 2** browses the input volume before any inversion touches it:

- **Inline / crossline** sections with a line selector, and wells projected onto
  the section only when they are within a chosen number of bins — projecting a
  well from far away would misrepresent where it is, so the distance is yours
  to set.
- **Time slice** in map view with the wells marked.
- **Traverse through wells** — an arbitrary line following a polyline through
  the wells you pick, in the order you pick them. Traces are taken at the
  nearest bin rather than interpolated, so the section shows real traces the
  survey recorded, and the horizontal axis is true distance in metres rather
  than trace number.

Gain, clip percentile and colour scale are display-only; none of them touch the
data. A per-line amplitude summary (RMS, p99, dead-trace count) sits under the
section.

---

## Well correlation

**Step 4** is a conventional correlation panel: the wells you select, left to
right in the order you select them, each showing its log, its formation tops
and the seismic trace extracted at its location.

- **Order** follows your selection order — clear and re-pick to rearrange.
- **Log curve** is chosen from the curves present in *every* selected well
  (plus the derived AI, Vp and Rho), and is scaled on a range shared across all
  of them, so a thickening or a sharpening really is comparable between wells.
- **Seismic** is drawn as a wiggle with the positive lobes filled — the
  convention that lets the eye follow peaks across the panel — from the same
  IDW-blended trace the well tie uses.
- **Tops** are drawn per well and joined between adjacent wells by correlation
  lines. Only tops present in at least two selected wells are offered, since a
  pick in a single well cannot be correlated.
- **Flatten on** hangs the panel on a chosen top instead of two-way time; wells
  missing that top are named rather than silently left unflattened.

The panel is drawn on a single axes with one slot per well rather than as a
subplot grid, because the correlation lines have to run *between* wells — in a
grid that means paper-coordinate shapes that drift out of alignment whenever
the layout changes. The same traverse used by the seismic viewer is available
in an expander underneath, so the wells and the section between them can be
read together.

---

## Loading a whole well database from a folder

Step 1 offers **Well folder (F3 demo layout)**: point at the folder that
*contains* the per-type subfolders and every well loads at once, complete with
its checkshot, deviation survey and tops.

```
F3Demo/
  Lasfiles/    F02-1.las   F03-2.las   F03-4.las   F06-1.las
  Checkshot/   F02-1_TD.txt        ...
  Track/       F02-1.track         ...
  Tops/        F02-1_markers.txt   ...
```

Wells are keyed off the **LAS files**, since those carry the logs everything
else decorates; the rest are matched to them by filename. Folder names are only
a tie-breaker — each file is classified by its *contents*, so a checkshot filed
under `Track` is still read as a checkshot (and the app says so). Extra folders
do no harm, and the layout above is a convention rather than a requirement.

The scan reports what it **skipped** and why, not just what it loaded, so a
file that failed to parse is visible rather than merely absent. Any SEG-Y found
under the same folder is offered in a dropdown that fills in the volume path.

A folder with no LAS files is rejected outright rather than loading as an empty
well list.

---

## Time-depth, deviation and marker files

Beyond the LAS, a well usually arrives with three companion files. Step 1
accepts all of them together and tells them apart **by their contents**, not by
their extension, then matches each to a well by filename (`F02-1_TD.txt` finds
well `F02-1`; punctuation and role suffixes are ignored, so `F021_TD.txt` works
too).

| Kind | Columns | Example row |
| --- | --- | --- |
| Time-depth / checkshot | `MD  time` | `553.6  0.544` |
| Deviation survey | `X  Y  TVDSS  MD` | `606554  6080126  1665  1695` |
| Markers / tops | `MD  name` | `1285.09  NMRF (Mid_Mio_Unc)` |

Whitespace or tab separated, no header required. Marker names may contain
spaces — everything after the depth is the name.

**The checkshot becomes the well's time source** when attached, in preference
to a LAS time curve or sonic integration, since it is the only one of the three
that was measured. The deviation survey supplies the surface location and KB,
so a LAS with no X/Y in its header still ties. Markers are carried in time and
drawn on the log tracks and the tie panel; two of them can set the analysis
gate directly.

### Reading the time column

Seconds against milliseconds is decided by magnitude. One-way against two-way
is decided from the implied interval velocity — but **only where that is
decidable**. Reading the F3 F02-1 checkshot as two-way gives 1,925 m/s, so it
is two-way; halving it gives 3,850 m/s, which is a perfectly ordinary rock
velocity, so a one-way table over a slow section is genuinely ambiguous from
the numbers alone.

The loader therefore defaults to two-way (much the commoner convention) rather
than guessing, reports how it read the file in the summary, offers an explicit
override, and warns when the first 1.5 s averages above 3,000 m/s — fast for
shallow section, and the signature of exactly that mistake. The check is
windowed in time rather than depth so a table that starts below the datum is
not judged as though it were shallow.

### Deviated wells

The deviation survey is parsed and its geometry reported, and a `Z` column that
decreases with MD is treated as an elevation and flipped to TVDSS. But v1
extracts the seismic trace at the **surface** location; for a strongly deviated
well that is not where the logs are, so the app says so on step 2 rather than
quietly tying the wrong trace. Extracting along the deviated path is an
extension point, not a v1 feature.

---

## Well ties

The app assumes wells are **already tied** — the time-depth relationship, and
any stretch or squeeze, are expected to come from upstream (a well-tie
notebook or an interpretation package). Step 2 exists to *verify* that tie, not
to build one:

- A LAS carrying a `TWT` (or `TIME`/`TWTT`/`OWT`) curve is used as-is.
- A LAS without one gets a sonic-integrated time-depth, bridged from KB to the
  seismic datum by a replacement velocity. With no checkshot to calibrate
  against, absolute time is only as good as the sonic — treat it as a starting
  point.
- A **checkshot** attached on step 1 takes precedence over both, and is the
  preferred route (see above).
- A constant **bulk shift** per well is available for a datum error; markers
  move with it. Stretch and squeeze are out of scope for v1 (see *Not in this
  version*).

Each well is located against the seismic grid by KD-tree lookup of the nearest
*live* traces, blended by inverse-distance weighting. Restricting the search to
live traces means a well near the survey edge still gets real amplitudes rather
than a dead trace.

---

## Low-frequency model

Well AI is low-pass filtered (default 10 Hz) and interpolated laterally between
wells, one time sample at a time, by IDW, RBF or nearest neighbour.

Filtering is done on `log(AI)`, not `AI`. Impedance is a positive, roughly
log-normal quantity: a low-pass in the log domain cannot produce a negative
background, and it preserves relative contrasts.

With horizons loaded, interpolation happens in horizon-flattened time and the
result is restored to structure, so the trend follows the geology instead of
cutting across it. Without horizons it is a flat time-slice interpolation,
which is only defensible where structure is gentle — the app says so on the
page. v1 uses the first horizon as the flattening datum.

---

## Performance

Measured on a 40 × 40 × 400 synthetic volume (1,600 traces, 400 samples each)
on a single core:

| Method | Time | Per trace | Notes |
| --- | --- | --- | --- |
| Coloured | 2.6 s | 1.6 ms | One convolution per trace |
| Bayesian | 18.3 s | 5.5 ms | Closed form; 1.7 ms without the variance |
| Sparse-spike | 31.8 s | 19.9 ms | 12 IRLS passes, banded Cholesky each |
| Model-based | 43.6 s | 27.3 ms | L-BFGS-B, up to 60 iterations |

Per-trace cost also grows with trace length and wavelet length, so these are
indicative rather than a formula.

Both slow methods scale linearly in trace count, so a production volume is
hours, not seconds. The app is built around that:

- **Estimate full-volume runtime** times a handful of real traces and
  extrapolates, before you commit.
- **Preview on a subset** inverts a block of inlines/crosslines in seconds.
- **Run full volume** runs on a worker thread with a live progress bar, so the
  page stays responsive. Progress is passed through a plain dict rather than
  `session_state`, which is not thread-safe.

---

## Project layout

```
seismic_inversion_app/
├── app.py                    # Streamlit entry point, sidebar routing, session state
├── make_demo_data.py         # writes demo SEG-Y / LAS / CSVs for the upload path
├── requirements.txt
├── README.md
├── modules/
│   ├── data_io.py            # SEG-Y + LAS loading, containers, synthetic generator
│   ├── wavelet.py            # parametric, statistical and well-based extraction
│   ├── inversion.py          # the three engines, volume runner, crossplot QC
│   ├── low_freq_model.py     # background impedance model
│   ├── visualization.py      # plotly sections, spectra, tie QC, crossplots
│   └── utils.py              # units, filtering, geometry, impedance algebra
└── tests/
    └── test_pipeline.py      # end-to-end checks on the synthetic dataset
```

No module under `modules/` requires Streamlit to be running — `data_io` imports
it defensively for caching only. The numeric core can be driven from a notebook
or a script unchanged.

---

## Not in this version

Deferred deliberately, and flagged where the UI would otherwise imply
otherwise:

- **Pre-stack, AVO and simultaneous inversion.** Post-stack only; there is no
  angle-gather handling and no elastic (Vs / density) output.
- **Stretch and squeeze.** Bulk shift only. A bulk shift can fix a datum error;
  it cannot fix a drifting time-depth relationship.
- **Non-stationary wavelets.** One wavelet for the whole volume. No spatially
  or temporally varying extraction, and no Q compensation.
- **Uncertainty quantification.** Each engine returns one answer, not a
  posterior distribution.
- **Multi-horizon flattening.** With several horizons loaded, v1 flattens on the
  first rather than doing proportional layer-cake flattening between them.
- **Deviated-well trace extraction.** Deviation surveys are read and reported,
  but the seismic trace is taken at the surface location, not along the
  borehole path.
