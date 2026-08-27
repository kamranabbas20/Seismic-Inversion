# Post-Stack Seismic Inversion

A Streamlit application that loads 3D post-stack SEG-Y volumes and multi-well
LAS logs, then inverts for acoustic impedance by one of four methods:
**coloured inversion**, **sparse-spike inversion**, **model-based inversion**
and **Bayesian linear inversion**. Impedance can then be turned into a rock
property, either through a calibrated transform or by multi-attribute
prediction from the seismic itself. Results are viewable as sections, time
slices and crossplots, and exportable as SEG-Y, NumPy or NetCDF.

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

### Validation on real field data

Synthetic tests can only prove the algebra is right. They cannot tell you
whether the app survives a real well, because a synthetic well has a perfect
time-depth relationship, a stationary wavelet and no drift.

`scripts/validate_penobscot.py` runs the whole workflow against the Penobscot
3D survey, offshore Nova Scotia — real seismic and a real well, both openly
licensed, neither redistributed in this repository:

```bash
python scripts/validate_penobscot.py --fetch     # clone the public data
python scripts/validate_penobscot.py --figure penobscot.png
```

- **Seismic** — crossline 1155 of the Penobscot 3D (601 traces, 4 ms). Owned by
  the Nova Scotia Department of Energy, distributed by permission through dGB's
  Open Seismic Repository, via the SEG tutorial repository `seg/tutorials-2014`.
- **Well** — Penobscot L-30, Schlumberger logs digitised by Neil Watson and
  released by CNSOPB for non-commercial knowledge sharing, from the same repo
  (the data for Evan Bianco's *Leading Edge* tutorial "How to make a synthetic").

L-30 sits 42 m off that line, which is what makes the pair usable. The script
first checks its own reading of the LAS header against the three geometry
numbers published in that tutorial, so the datum is verified against a third
party rather than asserted:

| | this app | SEG tutorial |
|---|---|---|
| KB elevation | 30.175 m | 30.175 m |
| water column | 137.465 m | 137.46 m |
| rock above the top of log | 179.832 m | 179.83 m |

Scores at the well, 996–2836 ms (461 samples, the interval where the sonic and
density logs overlap):

| | corr 10–60 Hz | corr full band | RMSE ln(AI) | seismic residual |
|---|---|---|---|---|
| low-frequency model only | 0.333 | 0.942 | 0.0795 | — |
| coloured | 0.490 | 0.948 | 0.0748 | — |
| sparse-spike | 0.374 | 0.872 | 0.1441 | 10.1% |
| model-based | 0.494 | 0.845 | 0.1605 | 15.7% |
| **Bayesian** | **0.499** | 0.871 | 0.1442 | 16.3% |

**Read the first column, not the second.** With one well the low-frequency
model is built from the same log the result is scored against, so the full-band
correlation is not a blind test — it mostly measures the background model, which
is why "low-frequency model only" appears to win it. Above the model's 10 Hz
cutoff the background carries no information, so the 10–60 Hz column is what the
inversion actually recovered from the seismic. The seismic residual is blind
everywhere: it is measured over all 601 traces without reference to the well.

### A second dataset that did *not* work

`scripts/validate_f3.py` was written to be a second inversion test, pairing F3
well F02-1 with the inline 362 exported in `seg/tutorials-2017`. It is kept
because the attempt is instructive, but **it produces no inversion scores**: the
well cannot be located on that line.

The test that settles it costs nothing. A well genuinely on a line ties best at
the same crossline whichever gate you score over. Here the best crossline
wanders over 14.2 km as the gate moves, and the correlation flips sign — the
signature of no tie at all. The tutorial's own marker at crossline 336 is
commented *"Just to display the well position"*: schematic, not a projection.

Had that check been skipped, the run would have reported a full set of
plausible-looking numbers — every method scoring *below* the background model,
sparse-spike diverging to a 130% residual — and they would have meant nothing.
A poor score and an absent tie look identical in the results table.

What the F3 data does validate is the unit chain, on the opposite branch from
Penobscot: F02-1 records DT in µs/m, RHOB in kg/m³ and depth in metres, against
L-30's µs/ft, g/cm³ and feet. Its LAS also carries an AI curve computed by the
publisher, so the impedance this app derives is checked against a third party's
arithmetic rather than its own — **median disagreement 0.00%**. The folder
scanner and checkshot reader are exercised on the real F3 layout at the same
time.

---

The method ranking matches the synthetic study — Bayesian best, then
model-based — and every engine now beats the background it was regularised
toward.

**The tie dominates everything.** L-30 has no checkshot in the open release, so
its time-depth comes from integrating the sonic, and a sonic-derived time-depth
drifts. Allowing a stretch on top of the bulk shift is worth **+0.330 of tie
correlation** here (0.295 → 0.626), which lifts the extracted wavelet's tie from
0.45 to 0.67 and drops its constant phase from +74° to +48°. Every score in the
table above moved with it:

| | bulk shift only | with the stretch |
|---|---|---|
| low-frequency model | 0.319 | 0.333 |
| coloured | 0.337 | 0.490 |
| sparse-spike | 0.293 | 0.374 |
| model-based | 0.384 | 0.494 |
| Bayesian | 0.395 | 0.499 |

No engine changed. The tie did. On data without a checkshot, tie quality is
worth more than the choice of inversion method — which is the single most
useful thing this validation has produced.

**Sparse-spike needs its regularisation chosen, not guessed.** Left at a default
sparsity it drove the seismic residual to 0.5% — an almost exact fit — and
scored 0.231, *below* the background model: everything it added above the
background was noise. Choosing the weight by the discrepancy principle instead
(residual matched to a 10% noise level) moved it to 0.374, above the background,
with the residual landing on 10.1% as intended. It remains the engine most
sensitive to tie and wavelet quality.

---

## Workflow

| Step | What it does |
| --- | --- |
| **1 - Data** | Three sources: a synthetic dataset, a **well folder** scanned in one go (F3 demo layout), or file-by-file upload. SEG-Y with byte positions and a header scan, multi-well LAS, time-depth / deviation / marker files, optional well-header and horizon CSVs. |
| **2 - Seismic viewer** | Inline, crossline, time slice and an arbitrary traverse through chosen wells, with gain, clip and colour-scale controls and well overlays. |
| **3 - Log QC** | Curve inventory per well, assign which curve is Vp / density / TWT and in what unit, rename wells, pass-fail sanity checks, checkshot / deviation / marker review. |
| **4 - Well correlation** | Wells side by side in a chosen order with logs, tops and the seismic trace at each, correlation lines between tops, and flattening on a datum. |
| **5 - Well tie QC** | Per-well synthetic-vs-extracted overlay, tie score table, bulk shift, and an optimiser that adds stretch and squeeze. Optional sampling down the borehole for deviated wells. |
| **6 - Wavelet** | Parametric, statistical or well-based extraction with spectrum QC, plus bulk Q estimation and a wavelet extracted per time window to show how far it drifts. |
| **7 - Low-frequency model** | Well AI low-pass filtered and interpolated between wells, optionally flattened on one horizon or proportionally between several. |
| **8 - Inversion** | Four methods (coloured, sparse-spike, model-based, Bayesian), method-specific parameters, sparsity chosen from the noise level, optional lateral coupling, stochastic realisations, single-trace QC, preview on a subset, full-volume run. |
| **9 - Blind validation** | Leave-one-out cross-validation: each well held out, the background rebuilt without it, and the inversion scored against a log it has never seen. |
| **10 - Rock property** | Two routes to a log-measurable property: a rock-physics transform from impedance, or multi-attribute prediction straight from the seismic. Either way a cut-off becomes a probability and an expected net thickness. |
| **11 - Results & export** | Section viewer, time slice, per-trace QC maps, posterior uncertainty and P10/P90 where available, crossplot against well logs, export. |

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

## The four inversion methods

All four share one interface:

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

## Blind validation

Scoring an inversion at a well that helped build its own background model
measures the background model. The full-band correlation in the Penobscot table
above is 0.94 for the background *alone* — it looks like a triumph and means
nothing, because the background is the well.

Step 9 runs the honest experiment instead. Each well is held out, the
low-frequency model is rebuilt from the others, the held-out trace is inverted,
and the result is scored against a log the model has never seen. Only that
well's own trace is inverted, so a fold costs one trace and the whole sweep
costs about as much as inverting a handful — whatever the size of the survey.

Two numbers are reported side by side: the blind score for the inversion, and
the blind score for the background model alone. **An engine that cannot beat its
own background at a blind well has added nothing**, however good the section
looks. The difference is reported as the uplift.

On the synthetic case, where the background's blind score collapses to ~0.00
once its well is removed, all four engines beat it at all four wells. That
collapse — from 0.31 when the well is included to 0.00 when it is not — is the
clearest possible statement of why a non-blind score should not be quoted.

Blind validation needs at least two located wells. With one, holding it out
leaves nothing to build a background from, which is precisely why a single-well
score cannot be blind — and why the Penobscot section above says so explicitly
rather than quoting the flattering number.

---

## Uncertainty, coupling and realisations

Three things follow from the Bayesian engine returning a *distribution* rather
than a number.

### Posterior variance without forming the inverse

The obvious way to get the posterior standard deviation is to back-substitute
against the identity, but that computes all `n²` entries of the inverse to keep
`n` of them: `O(n² b)` flops and `O(n²)` memory, so it goes quadratic exactly
when traces get long. The **Takahashi recursion** computes only the entries
inside the factor's band, which is `O(n b²)`. Measured on the 31-band system:

| trace length | dense identity | selected inversion | speed-up |
|---|---|---|---|
| 701 | 18.0 ms | 6.6 ms | 2.7× |
| 1501 | 174.8 ms | 14.4 ms | 12.2× |
| 3001 | 1058.9 ms | 27.9 ms | 37.9× |

Identical to the dense answer to 8e-16 — it is the same number, computed without
the wasted work.

### Lateral coupling

Every engine here treats each trace as an independent 1D problem, so nothing
stops two neighbours disagreeing by more than the seismic can justify. That is
what vertical striping in a noisy inverted section actually is: independent
estimation errors, side by side.

Adding a lateral roughness term to the joint prior gives a precision matrix that
is far too large to factor directly — it is (traces × samples) square — but it
is block-sparse, so a **block Gauss-Seidel sweep** solves it one trace at a time
against the current estimate of its neighbours. Each block solve is the same
banded Cholesky as the 1D engine, so a run costs `n_sweeps` times a 1D run, and
because the sweep is coordinate descent on a convex quadratic it converges
monotonically.

A lateral weight of zero reproduces the independent result *exactly* — asserted
in the tests, because a "generalisation" that changes the answer at its identity
setting is not one.

### Stochastic realisations

The point estimate is the posterior *mean*: smooth, band-limited, and by
construction the one model no realisation looks like. Sampling the same
posterior gives models that are each consistent with the seismic and the prior,
and whose spread *is* the uncertainty — which is what you want before quoting a
thickness or a contact.

Sampling is almost free once the factorisation exists. With `A = UᵀU` and `z`
standard normal, `U⁻¹z` has covariance `A⁻¹`, so each realisation is one banded
triangular solve: `m_sample = m_map + solve(U, z)`. No Gibbs sampler, no burn-in,
no convergence to argue about — these are independent exact draws, because the
posterior really is Gaussian. 250 draws on a 400-sample trace take about 60 ms.

---

## From impedance to a rock property

Impedance is an intermediate quantity — nobody drills on it. Step 10 fits a
transform from `ln(AI)` to a well curve, applies it to the cube, and carries the
uncertainty through.

Fitting in `ln(AI)` rather than `AI` is deliberate: impedance is a positive,
roughly log-normal quantity, most petrophysical trends are closer to straight in
the log domain, and it makes the uncertainty propagation exact rather than
approximate, because the inversion's posterior is Gaussian in `ln(AI)`.

Two independent uncertainties are combined, because dropping either flatters the
answer:

```
sigma_P² = (dP/dln(AI) · sigma_lnAI)²  +  sigma_fit²
```

The first term is what the seismic did not resolve; the second is the scatter of
the wells about the transform, which stays whatever the seismic does. Given a
deterministic cube only the second is available, and the app says so rather than
letting it pass as the full uncertainty.

That combined sigma is what makes a **cut-off** answerable. Asking "is porosity
above 12% here?" of a single number gives a yes or a no, both overconfident.
Asking it of a distribution gives a probability, and summing that probability
down a trace gives an expected net thickness that already accounts for how well
the seismic resolved the impedance. Where the seismic constrained it tightly the
probability sits near 0 or 1; where it did not, it drifts toward 0.5 instead of
pretending to know.

---

## Multi-attribute prediction

The transform above goes through impedance, so it can only carry what the
inversion resolved. Step 10 offers a second route that goes straight from the
seismic to the log: the classical multi-attribute method of Hampson, Schultz &
Fehler (2001), in `modules/multiattribute.py`.

Thirteen attributes are derived from the volume — amplitude, envelope,
instantaneous phase and frequency, amplitude-weighted frequency, the integrated
trace, first and second derivatives, four band-passed copies, and time itself —
plus any external cube, for which the inverted impedance is the obvious
candidate. Each is read through a short **convolutional operator** rather than
sample by sample, because a log responds to a bed while a seismic sample
responds to everything within a wavelet of it; the two only line up over a
window.

Attributes are then added one at a time by **forward stepwise selection**, and
this is the part that matters:

- at each step every remaining attribute is tried, and the one that lowers the
  **validation** error most is kept;
- validation error is measured leave-one-**well**-out — the model is refitted
  without a well and scored against that well's log;
- selection stops when validation error stops improving, whatever the training
  error is doing.

Training error can only fall as attributes are added. It is not evidence of
anything, and a method that chose its size by training error would happily fit
noise until it ran out of attributes. The app plots both curves against the
number of attributes, with the spread of the target curve drawn across them:
that dashed line is the error you would get by predicting the mean everywhere,
so a validation curve that never drops clearly below it means the seismic is not
carrying this property, however good the training fit looks.

Three guards are stated in words rather than left for the user to spot:

- validation error no better than predicting the mean — *"this model has learned
  nothing that generalises to a well it has not seen"*;
- validation error still falling when the attribute limit was reached — the
  limit, not the data, chose the model size;
- validation error far above training error — the extra freedom is being spent
  on memorising.

A **neural network** option is offered: one hidden tanh layer trained with Adam,
fitted on the attributes the linear selection already chose, so the two are
compared on equal footing. If it validates worse than the linear model, the app
says so instead of presenting the more elaborate answer as the better one. Fits
are refused outright with fewer than two wells, because with one well there is
nothing to validate against.

The uncertainty carried into the cut-off is the blind-well validation error,
held constant over the cube. It is what the predictor missed at a well it had
never seen, and it is the only error bar a data-driven fit has earned — unlike
the impedance route, there is no posterior to propagate.

Depth and time index curves (`DEPT`, `TWT`, `MD` …) are kept out of the target
list on both routes. They are predicted almost perfectly by the `time`
attribute and say nothing about the rock, so offering one invites a fit that
looks excellent and means nothing.

---

## Absorption

One wavelet for a three-second volume is an assumption, not a measurement. The
earth is anelastic: high frequencies are absorbed faster than low ones, so a
wavelet at 2,500 ms is narrower-band and more phase-rotated than the same
wavelet at 800 ms. Step 6 measures that two ways.

**A wavelet per time window**, extracted from the wells in overlapping windows
and blended linearly between window centres. A drift of a few Hz across the
volume is normal and the stationary wavelet is fine; tens of Hz means the deep
section is being inverted with a wavelet it does not have.

**A bulk Q** from the classical spectral-ratio method: between two windows the
amplitude spectra differ by `exp(-π f Δt / Q)`, so `ln(A_deep/A_shallow)` is
linear in `f` with slope `-π Δt / Q`.

The band matters more than anything else here. Outside the seismic's own
bandwidth both spectra are near zero, their ratio is numerical noise near one,
and including that flattens the slope and inflates Q. Measured against synthetic
data with a known Q of 40, a band of 8–70 Hz on 35 Hz data returned **127**;
narrowing to 18–40 Hz returned **52**. So the band is taken from the data by
default — where the shallow window still holds 60% of its peak amplitude — and
`R²` is the honest guide either way: the badly-biased run scored 0.56, the good
one 0.99.

Even with a sensible band this is a rough number, biased high under strong
absorption (Q 150 → 158, Q 80 → 86, Q 40 → 62). Treat it as an order of
magnitude for sizing an inverse-Q gain, not as a rock property.

Inverse-Q filtering is applied window by window, since absorption is
non-stationary and cannot be one filter. The trace is reflect-padded, split into
overlapping Hann-tapered windows, each filtered with the constant-Q operator for
its own centre time, and summed. The gain is capped (default 20 dB) because
otherwise inverse Q amplifies the highest frequencies without bound — and since
those are mostly noise, the result is an unusable trace with a beautiful
spectrum.

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

The *depth index* is the same trap one level up. `WellData.md` is metres
everywhere downstream — `integrate_sonic_to_twt` divides it by a velocity in
m/s — so a LAS indexed in feet has to be converted on the way in. Left alone it
does not fail loudly; it just places the well 3.28× too deep. The index unit is
read from the header (the first curve, falling back to `STRT`) and the KB
elevation is converted with it, since it is quoted in the same system. Depth
magnitude cannot disambiguate the two — a 4,000 m well and a 4,000 ft well are
both ordinary — so an *unlabelled* index is taken as metres and flagged in QC
rather than guessed at. This is the defect the Penobscot run above surfaced:
L-30 is indexed in feet, and its KB now reads 30.175 m against the tutorial's
published 30.175 m.

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
  move with it.
- **Stretch and squeeze** is available on top of that, as a piecewise-linear
  warp over a handful of knots. See below.

Each well is located against the seismic grid by KD-tree lookup of the nearest
*live* traces, blended by inverse-distance weighting. Restricting the search to
live traces means a well near the survey edge still gets real amplitudes rather
than a dead trace. For a **deviated** well the tie step can instead sample the
seismic *down the borehole*: at each time sample the well's time-depth gives the
measured depth, the deviation survey gives the map position there, and the trace
value is taken from that bin. A 1,000 m step-out puts the reservoir tens of bins
from the wellhead, and tying those logs to the surface trace ties them to the
wrong rock. Vertical wells are unaffected — the two routes return byte-identical
traces, which is asserted in the tests.

### Stretch and squeeze

A bulk shift fixes a datum error. It cannot fix a time-depth that *drifts*,
which is what you get whenever the time-depth comes from integrating a sonic
rather than from a checkshot. The optimiser searches a bulk shift first, then a
piecewise-linear warp over a few knots, subject to two constraints:

- **The wavelet is held fixed while the warp is searched.** A wavelet
  re-estimated inside the loop absorbs timing error — a matching filter has
  enough freedom to fit almost any misalignment — so the reported improvement
  would be measuring the wavelet, not the tie.
- **The warp must stay monotonic.** Time cannot run backwards; a warp that would
  fold the log is rejected rather than quietly applied.

If the stretch does not beat the bulk shift it is discarded, so a well with a
good checkshot is left exactly as it was. On Penobscot L-30 — a real well with
no checkshot — it was worth **+0.330** of tie correlation, and lifted every
inversion engine with it (see *Validation on real field data*).

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
page.

**One horizon** gives a constant per-trace datum shift. **Two or more** give
proportional (layer-cake) flattening: each interval is stretched onto the
interval between those horizons' mean times, so the trend follows thickness
variation and not just structure. Horizons are sorted into stratigraphic order
by mean time, so the order they arrive in does not matter, and crossing picks
are forced apart by one sample rather than being allowed to fold the warp.

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
indicative rather than a formula. The Bayesian figure is now roughly flat in
trace length rather than quadratic: on the real 701-sample Penobscot line it
runs at 9 ms/trace with uncertainty on, against 14 ms before the posterior
variance was moved onto the Takahashi recursion.

Optional extras cost roughly:

| | Cost |
| --- | --- |
| Lateral coupling | `n_sweeps` × a 1D Bayesian run (default 3) |
| Stochastic realisations | one triangular solve each — 250 draws ≈ 60 ms per trace |
| Blind cross-validation | one background rebuild and one trace per well |
| Auto-sparsity | 8–14 trial solves on a sample of traces, once per run |

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
│   ├── inversion.py          # the four engines, volume runner, crossplot QC
│   ├── low_freq_model.py     # background impedance model, horizon flattening
│   ├── welltie.py            # bulk shift + stretch/squeeze optimiser
│   ├── crossval.py           # blind leave-one-out validation
│   ├── rockphysics.py        # impedance -> property, with uncertainty
│   ├── multiattribute.py     # attribute selection by blind-well error, linear + MLP
│   ├── visualization.py      # plotly sections, spectra, tie QC, crossplots
│   └── utils.py              # units, filtering, geometry, impedance algebra
├── scripts/
│   └── validate_penobscot.py # end-to-end run on real open field data
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
  angle-gather handling and no elastic (Vs / density) output. This is the one
  large capability still missing, and it is a different tool rather than a
  setting: new data model, Aki-Richards, angle-dependent wavelets.
- **Interval Q.** Absorption is estimated as a single bulk Q between two
  windows and applied as a constant-Q operator. There is no layer-by-layer Q
  profile, and the estimate is biased high under strong absorption.
- **Anisotropy.** Vertical velocity only.
- **Automatic horizon interpretation.** Horizons are read from a CSV; nothing
  here picks them.

Since the first version, six of the original non-goals have been closed and
are now documented above: stretch and squeeze, uncertainty quantification,
non-stationary wavelets, multi-horizon proportional flattening, deviated-well
trace extraction, and multi-attribute / machine-learning prediction.
