"""Well-tie optimisation: bulk shift, then stretch and squeeze.

A bulk shift fixes a datum error.  It cannot fix a *drifting* time-depth
relationship, which is what you get whenever the time-depth comes from
integrating a sonic rather than from a checkshot -- and on real data that drift
is usually the largest single error in the whole workflow.  Measured on
Penobscot L-30 (``scripts/validate_penobscot.py``), allowing a linear stretch
on top of the bulk shift is worth +0.045 of tie correlation, and every engine
downstream inherits that.

Two things keep this honest:

* **The wavelet is held fixed while the warp is searched.**  A wavelet
  re-estimated inside the loop will happily absorb a timing error -- the match
  filter has enough freedom to fit almost any misalignment -- so the reported
  improvement would be measuring the wavelet, not the tie.  The wavelet is
  re-estimated once, afterwards.
* **The warp is constrained.**  Knot corrections are bounded and the resulting
  time axis must stay monotonic, so the optimiser cannot buy correlation by
  folding the log back on itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import data_io, utils

# A warp free to move every knot by an arbitrary amount will always find
# correlation, and it will be meaningless.  These are the defaults the UI
# exposes; they are deliberately tighter than what the maths allows.
MAX_BULK_SHIFT_MS = 60.0
MAX_DRIFT_MS = 30.0
DEFAULT_KNOTS = 4


@dataclass
class TieSolution:
    """The tie the optimiser settled on, and what it was worth."""

    bulk_shift: float
    knots_ms: np.ndarray
    shifts_ms: np.ndarray
    correlation: float
    correlation_start: float
    correlation_bulk_only: float
    n_knots: int
    gate: tuple[float, float]
    notes: list[str] = field(default_factory=list)

    @property
    def drift_gain(self) -> float:
        """Correlation the stretch added over the best bulk shift alone."""
        return float(self.correlation - self.correlation_bulk_only)

    def summary(self) -> dict:
        lo, hi = (float(np.min(self.shifts_ms)), float(np.max(self.shifts_ms))) \
            if self.n_knots > 1 else (0.0, 0.0)
        return {
            "gate (ms)": f"{self.gate[0]:.0f} - {self.gate[1]:.0f}",
            "bulk shift (ms)": f"{self.bulk_shift:+.1f}",
            "knots": self.n_knots,
            "drift range (ms)": f"{lo:+.1f} to {hi:+.1f}" if self.n_knots > 1 else "none",
            "correlation before": f"{self.correlation_start:+.3f}",
            "correlation, bulk only": f"{self.correlation_bulk_only:+.3f}",
            "correlation, with stretch": f"{self.correlation:+.3f}",
            "stretch was worth": f"{self.drift_gain:+.3f}",
        }


def analysis_gate(well, twt_axis: np.ndarray) -> tuple[float, float]:
    """Time range over which the well actually has Vp *and* density.

    The sonic and density tools rarely start at the same depth, and scoring a
    tie over an interval where the well contributes zero reflectivity drags the
    answer toward whatever the seismic happens to do there.
    """
    ai = well.ai_on_time_axis(np.asarray(twt_axis, dtype=float))
    ok = np.isfinite(ai) & (ai > 0)
    if ok.sum() < 8:
        t = np.asarray(twt_axis, dtype=float)
        return float(t[0]), float(t[-1])
    t = np.asarray(twt_axis, dtype=float)[ok]
    return float(t.min()), float(t.max())


def _correlation(well, volume, wavelet, gate_mask, k_neighbours: int) -> float:
    tie = data_io.extract_well_traces(volume, [well], k=k_neighbours)
    if not tie:
        return float("nan")
    tie = tie[0]
    synthetic = utils.convolve_same(np.nan_to_num(tie.reflectivity), wavelet)
    return utils.normalised_correlation(synthetic[gate_mask], tie.seismic[gate_mask])


def optimise_tie(
    well,
    volume,
    wavelet: np.ndarray,
    n_knots: int = DEFAULT_KNOTS,
    max_shift_ms: float = MAX_BULK_SHIFT_MS,
    max_drift_ms: float = MAX_DRIFT_MS,
    gate: tuple[float, float] | None = None,
    k_neighbours: int = 4,
    coarse_step_ms: float = 4.0,
    sweeps: int = 3,
    progress=None,
) -> TieSolution:
    """Find the bulk shift, then the stretch, that best ties this well.

    The well is left carrying the winning tie.  ``n_knots <= 1`` searches the
    bulk shift only, which is the previous behaviour and remains the right
    choice for a well with a trustworthy checkshot.
    """
    wavelet = np.asarray(wavelet, dtype=float)
    if wavelet.size < 3:
        raise ValueError("the tie optimiser needs a wavelet")
    if not getattr(well, "has_location", False):
        raise ValueError(f"{well.name}: no X/Y location, so it cannot be tied to the seismic")

    twt = np.asarray(volume.twt, dtype=float)
    t0, t1 = gate if gate is not None else analysis_gate(well, twt)
    gate_mask = (twt >= t0) & (twt <= t1)
    if gate_mask.sum() < 16:
        raise ValueError(f"{well.name}: analysis gate {t0:.0f}-{t1:.0f} ms is too short to tie on")

    # Start from a clean time model so re-running is idempotent.
    start_shift = float(well.bulk_shift)
    well.set_time_warp(None, None)
    well.set_bulk_shift(start_shift)
    corr_start = _correlation(well, volume, wavelet, gate_mask, k_neighbours)

    notes: list[str] = []

    # --- 1. bulk shift, coarse then fine ---------------------------------
    def score_shift(value: float) -> float:
        well.set_bulk_shift(value)
        return _correlation(well, volume, wavelet, gate_mask, k_neighbours)

    grid = np.arange(start_shift - max_shift_ms, start_shift + max_shift_ms + 1e-9, coarse_step_ms)
    scores = []
    for i, value in enumerate(grid):
        scores.append(score_shift(float(value)))
        if progress is not None:
            progress(0.4 * (i + 1) / grid.size, "searching bulk shift")
    best_i = int(np.nanargmax(scores))
    fine = np.arange(grid[best_i] - coarse_step_ms, grid[best_i] + coarse_step_ms + 1e-9, 1.0)
    fine_scores = [score_shift(float(v)) for v in fine]
    best_shift = float(fine[int(np.nanargmax(fine_scores))])
    corr_bulk = float(np.nanmax(fine_scores))
    well.set_bulk_shift(best_shift)
    notes.append(f"bulk shift {best_shift:+.1f} ms (correlation {corr_bulk:+.3f})")

    if n_knots <= 1 or max_drift_ms <= 0:
        return TieSolution(bulk_shift=best_shift, knots_ms=np.array([]), shifts_ms=np.array([]),
                           correlation=corr_bulk, correlation_start=corr_start,
                           correlation_bulk_only=corr_bulk, n_knots=0, gate=(t0, t1),
                           notes=notes + ["stretch not attempted"])

    # --- 2. stretch, by coordinate descent over the knots -----------------
    # Knots live on the *unshifted* well axis, spread across the gate.
    base_gate = np.array([t0, t1], dtype=float) - best_shift
    knots = np.linspace(base_gate[0], base_gate[1], int(n_knots))
    shifts = np.zeros(int(n_knots), dtype=float)

    def try_shifts(candidate: np.ndarray) -> float:
        try:
            well.set_time_warp(knots, candidate)
        except ValueError:
            return -np.inf          # non-monotonic: reject outright
        return _correlation(well, volume, wavelet, gate_mask, k_neighbours)

    best_corr = try_shifts(shifts)
    step = max(max_drift_ms / 4.0, 1.0)
    for sweep in range(int(sweeps)):
        improved = False
        for j in range(shifts.size):
            for delta in (step, -step):
                trial = shifts.copy()
                trial[j] = float(np.clip(trial[j] + delta, -max_drift_ms, max_drift_ms))
                if trial[j] == shifts[j]:
                    continue
                corr = try_shifts(trial)
                if corr > best_corr + 1e-6:
                    shifts, best_corr, improved = trial, corr, True
                    break
            if progress is not None:
                progress(0.4 + 0.6 * (sweep * shifts.size + j + 1) / (sweeps * shifts.size),
                         "searching stretch")
        if not improved:
            if step <= 1.0:
                break
            step = max(step / 2.0, 1.0)     # refine rather than stop

    if best_corr <= corr_bulk + 1e-6:
        # The stretch bought nothing; do not leave the well carrying one.
        well.set_time_warp(None, None)
        well.set_bulk_shift(best_shift)
        notes.append("stretch rejected: it did not improve on the bulk shift")
        return TieSolution(bulk_shift=best_shift, knots_ms=np.array([]), shifts_ms=np.array([]),
                           correlation=corr_bulk, correlation_start=corr_start,
                           correlation_bulk_only=corr_bulk, n_knots=0, gate=(t0, t1), notes=notes)

    well.set_time_warp(knots, shifts)
    notes.append(f"stretch over {shifts.size} knots, {shifts.min():+.1f} to {shifts.max():+.1f} ms "
                 f"(correlation {best_corr:+.3f}, {best_corr - corr_bulk:+.3f} over bulk shift alone)")
    return TieSolution(bulk_shift=best_shift, knots_ms=knots, shifts_ms=shifts,
                       correlation=float(best_corr), correlation_start=corr_start,
                       correlation_bulk_only=corr_bulk, n_knots=int(shifts.size),
                       gate=(t0, t1), notes=notes)


def optimise_all(wells, volume, wavelet, progress=None, **kwargs) -> dict[str, TieSolution]:
    """Tie every located well, reporting failures rather than raising."""
    out: dict[str, TieSolution] = {}
    located = [w for w in wells if getattr(w, "has_location", False)]
    for i, well in enumerate(located):
        try:
            out[well.name] = optimise_tie(well, volume, wavelet, **kwargs)
        except ValueError as exc:
            out[well.name] = TieSolution(
                bulk_shift=float(well.bulk_shift), knots_ms=np.array([]), shifts_ms=np.array([]),
                correlation=float("nan"), correlation_start=float("nan"),
                correlation_bulk_only=float("nan"), n_knots=0, gate=(np.nan, np.nan),
                notes=[f"not tied: {exc}"])
        if progress is not None:
            progress((i + 1) / max(len(located), 1), f"tied {well.name}")
    return out
