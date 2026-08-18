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
| **1 - Data** | SEG-Y + byte positions (with a header scan to identify non-standard ones), multi-well LAS, optional well-header and horizon CSVs. Or generate a synthetic dataset. |
| **2 - Well tie QC** | Per-well synthetic-vs-extracted overlay, tie score table, constant bulk shift. |
| **3 - Wavelet** | Parametric, statistical or well-based extraction, with amplitude/phase spectrum QC. |
| **4 - Low-frequency model** | Well AI low-pass filtered and interpolated between wells, optionally guided by horizons. |
| **5 - Inversion** | Method selection, method-specific parameters, single-trace QC, preview on a subset, full-volume run. |
| **6 - Results & export** | Section viewer, time slice, per-trace QC maps, crossplot against well logs, export. |

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
- A constant **bulk shift** per well is available for a datum error. Stretch
  and squeeze are out of scope for v1 (see *Not in this version*).

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
