"""Multi-attribute prediction of a well property from seismic.

Inversion asks what impedance is consistent with the seismic.  This asks a
looser question: given a handful of *attributes* derived from the seismic, what
linear or non-linear combination of them best reproduces a log at the wells?
Nothing here is constrained by physics -- the model is whatever fits.

That freedom is the whole danger.  With enough attributes and a convolutional
operator, a regression will reproduce any log at any small number of wells
perfectly, and predict nothing away from them.  Hampson, Schultz & Fehler (2001)
built the defence into the method, and it is implemented here rather than left
to the user:

* **Attributes are selected on validation error, not training error.**  Forward
  stepwise selection adds the attribute that best reduces the *training* error,
  but the number of attributes to keep is decided by *leave-one-well-out*
  error, which rises again once the model starts memorising.
* **Validation holds out whole wells, never samples.**  Neighbouring samples in
  one well are strongly correlated, so leaving out random samples measures
  interpolation between points a few metres apart and always looks excellent.
* **The error curves are returned, not just the answer**, so the gap between
  training and validation is visible.  A model whose validation error never
  improves is reported as such rather than quietly used.

With three or four wells this method can still fool you, and the honest reading
of a small-well-count result is usually "not established".  The machinery below
makes that visible; it cannot make it go away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import hilbert

from . import utils

# Attribute names are the keys used everywhere else in the module.
BASE_ATTRIBUTES = (
    "amplitude",
    "envelope",
    "instantaneous phase",
    "instantaneous frequency",
    "integrated trace",
    "derivative",
    "second derivative",
    "amplitude-weighted frequency",
    "time",
    "filtered 5-15 Hz",
    "filtered 15-30 Hz",
    "filtered 30-45 Hz",
    "filtered 45-60 Hz",
)


# --------------------------------------------------------------------------
# Attributes
# --------------------------------------------------------------------------

def compute_attributes(volume, external: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Derive the standard attribute cubes from a seismic volume.

    ``external`` adds cubes computed elsewhere -- an inverted impedance volume
    is the usual one, and is what turns this from "predict a log from seismic"
    into "predict a log from seismic *and* the inversion".  They are treated
    exactly like the internal attributes by the selection.
    """
    data = np.asarray(volume.data, dtype=float)
    dt = float(volume.dt)
    n_il, n_xl, n_t = data.shape
    flat = data.reshape(-1, n_t)

    analytic = hilbert(flat, axis=1)
    envelope = np.abs(analytic)
    phase = np.unwrap(np.angle(analytic), axis=1)
    # d(phase)/dt in Hz; the raw derivative is noisy where the envelope is small,
    # which is why the amplitude-weighted version is offered alongside it.
    inst_freq = np.gradient(phase, dt, axis=1) / (2.0 * np.pi)

    out: dict[str, np.ndarray] = {
        "amplitude": flat,
        "envelope": envelope,
        "instantaneous phase": np.angle(analytic),
        "instantaneous frequency": inst_freq,
        "integrated trace": np.cumsum(flat, axis=1) * dt,
        "derivative": np.gradient(flat, dt, axis=1),
        "second derivative": np.gradient(np.gradient(flat, dt, axis=1), dt, axis=1),
        "amplitude-weighted frequency": inst_freq * envelope,
        "time": np.broadcast_to(np.asarray(volume.twt, dtype=float), flat.shape).copy(),
    }
    for lo, hi in ((5, 15), (15, 30), (30, 45), (45, 60)):
        if hi < 0.5 / dt:
            out[f"filtered {lo}-{hi} Hz"] = utils.bandpass(flat, dt, float(lo), float(hi))

    cubes = {k: v.reshape(n_il, n_xl, n_t) for k, v in out.items()}
    for name, cube in (external or {}).items():
        cube = np.asarray(cube, dtype=float)
        if cube.shape != data.shape:
            raise ValueError(f"external attribute '{name}' has shape {cube.shape}, "
                             f"expected {data.shape}")
        cubes[name] = cube
    return cubes


def _operator_matrix(trace: np.ndarray, half: int) -> np.ndarray:
    """``(n_samples, 2*half+1)`` of a trace at lags -half..+half.

    The convolutional operator is what lets a *point* attribute predict a log
    that responds to a whole wavelet: the value at one sample rarely explains
    the property there, but a short window around it often does.  Edges hold
    the end value rather than wrapping, so a lag never brings the other end of
    the trace into view.
    """
    if half <= 0:
        return trace[:, None]
    padded = np.pad(trace, half, mode="edge")
    n = trace.size
    return np.stack([padded[k:k + n] for k in range(2 * half + 1)], axis=1)


def _design(traces: dict[str, np.ndarray], names, half: int) -> np.ndarray:
    """Design matrix for one location: lags of each named attribute, plus a bias."""
    blocks = [_operator_matrix(np.nan_to_num(traces[n]), half) for n in names]
    blocks.append(np.ones((len(next(iter(traces.values()))), 1)))
    return np.hstack(blocks)


# --------------------------------------------------------------------------
# Training data
# --------------------------------------------------------------------------

@dataclass
class TrainingData:
    """Attribute traces and the target log, gathered at the wells."""

    per_well: dict[str, dict[str, np.ndarray]]     # well -> attribute -> trace
    targets: dict[str, np.ndarray]                 # well -> target on the seismic axis
    masks: dict[str, np.ndarray]                   # well -> usable samples
    attribute_names: list[str]
    target_name: str

    @property
    def wells(self) -> list[str]:
        return list(self.per_well)

    def n_samples(self) -> int:
        return int(sum(m.sum() for m in self.masks.values()))


def gather_training(volume, wells, target_curve: str, attributes: dict[str, np.ndarray],
                    gate: tuple[float, float] | None = None, ties=None) -> TrainingData:
    """Pull each attribute and the target log at every located well."""
    from . import data_io, rockphysics

    twt = np.asarray(volume.twt, dtype=float)
    located = [w for w in wells if getattr(w, "has_location", False)]
    if not located:
        raise ValueError("multi-attribute training needs at least one located well")
    if ties is None:
        ties = data_io.extract_well_traces(volume, located, k=4)
    by_name = {t.well: t for t in ties}

    per_well, targets, masks = {}, {}, {}
    for well in located:
        tie = by_name.get(well.name)
        if tie is None:
            continue
        y = rockphysics.curve_on_time_axis(well, target_curve, twt)
        good = np.isfinite(y)
        if gate is not None:
            good &= (twt >= gate[0]) & (twt <= gate[1])
        if good.sum() < 16:
            continue
        i, j = tie.il_index, tie.xl_index
        per_well[well.name] = {name: np.asarray(cube[i, j, :], dtype=float)
                               for name, cube in attributes.items()}
        targets[well.name] = y
        masks[well.name] = good

    if not per_well:
        raise ValueError(
            f"no well provided usable '{target_curve}' samples -- check the curve name, "
            "the analysis gate, and that the wells are tied")
    return TrainingData(per_well=per_well, targets=targets, masks=masks,
                        attribute_names=list(attributes), target_name=target_curve)


# --------------------------------------------------------------------------
# Linear model
# --------------------------------------------------------------------------

def _fit_linear(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    """Least squares with a touch of ridge, so a collinear pair cannot blow up."""
    XtX = X.T @ X
    XtX.flat[:: XtX.shape[0] + 1] += ridge * max(float(np.trace(XtX)) / XtX.shape[0], 1e-12)
    return np.linalg.solve(XtX, X.T @ y)


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _stack(train: TrainingData, names, half: int, wells) -> tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for w in wells:
        m = train.masks[w]
        X = _design(train.per_well[w], names, half)[m]
        Xs.append(X)
        ys.append(train.targets[w][m])
    return np.vstack(Xs), np.concatenate(ys)


@dataclass
class MultiAttributeModel:
    """A fitted multi-attribute predictor, and the evidence for how many attributes."""

    target: str
    attributes: list[str]                  # in selection order, truncated to n_used
    operator_half: int
    coefficients: np.ndarray
    kind: str = "linear"
    training_rms: list[float] = field(default_factory=list)     # index k = k+1 attributes
    validation_rms: list[float] = field(default_factory=list)
    order: list[str] = field(default_factory=list)              # full selection order
    wells: list[str] = field(default_factory=list)
    target_std: float = 1.0
    net: "MLP | None" = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_used(self) -> int:
        return len(self.attributes)

    @property
    def overfitting_gap(self) -> float:
        """Validation minus training error at the chosen attribute count."""
        k = self.n_used - 1
        if not self.validation_rms or k >= len(self.validation_rms):
            return float("nan")
        return float(self.validation_rms[k] - self.training_rms[k])

    def summary(self) -> dict:
        k = self.n_used - 1
        val = self.validation_rms[k] if k < len(self.validation_rms) else float("nan")
        train = self.training_rms[k] if k < len(self.training_rms) else float("nan")
        return {
            "target": self.target,
            "model": self.kind,
            "attributes used": self.n_used,
            "operator length": f"{2 * self.operator_half + 1} samples",
            "training RMS": f"{train:.4g}",
            "validation RMS": f"{val:.4g}",
            "validation / target spread": f"{val / max(self.target_std, 1e-12):.2f}",
            "wells": ", ".join(self.wells),
            "selected": ", ".join(self.attributes),
        }

    def predict_traces(self, traces: dict[str, np.ndarray]) -> np.ndarray:
        X = _design(traces, self.attributes, self.operator_half)
        if self.kind == "mlp" and self.net is not None:
            return self.net.predict(X[:, :-1])          # the net carries its own bias
        return X @ self.coefficients


def fit_multi_attribute(
    train: TrainingData,
    operator_half: int = 2,
    max_attributes: int = 8,
    kind: str = "linear",
    hidden: int = 8,
    epochs: int = 400,
    seed: int = 0,
) -> MultiAttributeModel:
    """Forward stepwise selection, with the attribute count set by validation.

    Selection adds whichever attribute most reduces training error -- that is
    what stepwise means -- but the count that is *kept* is the one minimising
    leave-one-well-out error.  The two curves are returned so the gap between
    them can be seen.

    ``kind="mlp"`` trains a small network on the attributes the linear pass
    selected, which is how Hampson et al. use one: the network supplies
    non-linearity, not attribute selection.
    """
    wells = train.wells
    if len(wells) < 2:
        raise ValueError(
            "multi-attribute prediction needs at least two wells: with one there is "
            "nothing to validate against, and the training error of a free-form "
            "regression is meaningless")

    all_y = np.concatenate([train.targets[w][train.masks[w]] for w in wells])
    target_std = float(np.std(all_y)) or 1.0

    candidates = list(train.attribute_names)
    chosen: list[str] = []
    train_rms: list[float] = []
    val_rms: list[float] = []

    for _ in range(min(int(max_attributes), len(candidates))):
        best = (np.inf, None)
        for cand in candidates:
            names = chosen + [cand]
            X, y = _stack(train, names, operator_half, wells)
            err = _rms(X @ _fit_linear(X, y), y)
            if err < best[0]:
                best = (err, cand)
        if best[1] is None:
            break
        chosen.append(best[1])
        candidates.remove(best[1])
        train_rms.append(best[0])

        # Leave one WHOLE well out -- never individual samples.
        errs = []
        for held in wells:
            others = [w for w in wells if w != held]
            Xtr, ytr = _stack(train, chosen, operator_half, others)
            Xte, yte = _stack(train, chosen, operator_half, [held])
            errs.append(_rms(Xte @ _fit_linear(Xtr, ytr), yte))
        val_rms.append(float(np.mean(errs)))

    n_used = int(np.argmin(val_rms)) + 1
    used = chosen[:n_used]
    X, y = _stack(train, used, operator_half, wells)
    coeffs = _fit_linear(X, y)

    model = MultiAttributeModel(
        target=train.target_name, attributes=used, operator_half=int(operator_half),
        coefficients=coeffs, kind="linear", training_rms=train_rms,
        validation_rms=val_rms, order=chosen, wells=wells, target_std=target_std)

    if val_rms[n_used - 1] >= target_std:
        model.notes.append(
            "validation error is no better than simply predicting the mean: this model "
            "has learned nothing that generalises to a well it has not seen")
    if n_used == len(val_rms) and len(val_rms) > 1:
        model.notes.append(
            "validation error was still falling at the attribute limit; raise the limit "
            "to see where it turns")
    if model.overfitting_gap > 0.5 * target_std:
        model.notes.append(
            "validation error greatly exceeds training error -- the classic signature of "
            "too few wells for the number of free parameters")

    if kind == "mlp":
        net = MLP(hidden=hidden, seed=seed)
        net.fit(X[:, :-1], y, epochs=epochs)
        model.net = net
        model.kind = "mlp"
        mlp_val = []
        for held in wells:
            others = [w for w in wells if w != held]
            Xtr, ytr = _stack(train, used, operator_half, others)
            Xte, yte = _stack(train, used, operator_half, [held])
            n = MLP(hidden=hidden, seed=seed)
            n.fit(Xtr[:, :-1], ytr, epochs=epochs)
            mlp_val.append(_rms(n.predict(Xte[:, :-1]), yte))
        model.training_rms = train_rms[:n_used] + [_rms(net.predict(X[:, :-1]), y)]
        model.validation_rms = val_rms[:n_used] + [float(np.mean(mlp_val))]
        model.attributes = used
        if model.validation_rms[-1] > val_rms[n_used - 1]:
            model.notes.append(
                "the network validates worse than the linear model on the same "
                "attributes; the extra freedom is being spent on memorising")
    return model


# --------------------------------------------------------------------------
# A small network, written out rather than pulled in
# --------------------------------------------------------------------------

class MLP:
    """One hidden layer, tanh, trained with Adam.

    Deliberately small.  The failure mode of this whole approach is a model with
    more freedom than the wells can constrain, so the default width is eight
    units -- enough for a curved relationship, not enough to memorise a log.
    Written in numpy to avoid adding a dependency for sixty lines of arithmetic.
    """

    def __init__(self, hidden: int = 8, seed: int = 0, lr: float = 0.01):
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.lr = float(lr)
        self.w1 = self.b1 = self.w2 = self.b2 = None
        self.x_mean = self.x_std = self.y_mean = self.y_std = None

    def _standardise(self, X: np.ndarray) -> np.ndarray:
        return (X - self.x_mean) / self.x_std

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 400) -> "MLP":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        self.x_mean = X.mean(axis=0)
        self.x_std = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
        self.y_mean = float(y.mean())
        self.y_std = float(y.std()) or 1.0
        Xs = self._standardise(X)
        ys = (y - self.y_mean) / self.y_std

        rng = np.random.default_rng(self.seed)
        n_in = Xs.shape[1]
        self.w1 = rng.normal(0, np.sqrt(1.0 / n_in), (n_in, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(0, np.sqrt(1.0 / self.hidden), (self.hidden, 1))
        self.b2 = np.zeros(1)

        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1_, b2_, eps = 0.9, 0.999, 1e-8
        n = Xs.shape[0]
        for t in range(1, int(epochs) + 1):
            h = np.tanh(Xs @ self.w1 + self.b1)
            out = h @ self.w2 + self.b2
            d_out = 2.0 * (out - ys) / n
            g_w2 = h.T @ d_out
            g_b2 = d_out.sum(axis=0)
            d_h = (d_out @ self.w2.T) * (1.0 - h ** 2)
            g_w1 = Xs.T @ d_h
            g_b1 = d_h.sum(axis=0)

            for i, g in enumerate((g_w1, g_b1, g_w2, g_b2)):
                m[i] = b1_ * m[i] + (1 - b1_) * g
                v[i] = b2_ * v[i] + (1 - b2_) * g * g
                mh = m[i] / (1 - b1_ ** t)
                vh = v[i] / (1 - b2_ ** t)
                params[i] -= self.lr * mh / (np.sqrt(vh) + eps)
        self.w1, self.b1, self.w2, self.b2 = params
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._standardise(np.asarray(X, dtype=float))
        h = np.tanh(Xs @ self.w1 + self.b1)
        return (h @ self.w2 + self.b2).ravel() * self.y_std + self.y_mean


# --------------------------------------------------------------------------
# Prediction over a cube
# --------------------------------------------------------------------------

def predict_cube(model: MultiAttributeModel, attributes: dict[str, np.ndarray],
                 progress=None) -> np.ndarray:
    """Apply a fitted model trace by trace."""
    missing = [a for a in model.attributes if a not in attributes]
    if missing:
        raise ValueError(f"attributes missing from the cube: {missing}")
    shape = attributes[model.attributes[0]].shape
    n_il, n_xl, n_t = shape
    out = np.empty(shape, dtype=np.float32)
    total = n_il * n_xl
    done = 0
    for i in range(n_il):
        for j in range(n_xl):
            traces = {a: attributes[a][i, j, :] for a in model.attributes}
            out[i, j, :] = model.predict_traces(traces)
            done += 1
        if progress is not None:
            progress(done / total, f"{done:,} / {total:,} traces")
    return out


def validation_curve(model: MultiAttributeModel) -> dict:
    """The two error curves, for the plot that shows where overfitting starts."""
    return {
        "n_attributes": list(range(1, len(model.training_rms) + 1)),
        "training": list(model.training_rms),
        "validation": list(model.validation_rms),
        "chosen": model.n_used,
        "target_std": model.target_std,
    }
