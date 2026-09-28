"""CADENCE: a label-free drift monitor built on empty conformal prediction sets,
with an alarm-gated recalibration loop for deployed gas-sensor classifiers.

Monitor: split conformal prediction runs on top of any probabilistic
classifier. A prediction set is empty when even the predicted class has
nonconformity above the calibrated threshold. Under exchangeability the
empty-set event implies miscoverage, and under i.i.d. sampling the number
of empty sets in a window of W disjoint test points is stochastically
dominated by a
Beta-Binomial(W, l, n+1-l) law (l = floor(alpha(n+1)), n = calibration
size), which accounts for finite-calibration variance. The alarm fires when
the count reaches the Beta-Binomial critical value at level delta: the
per-window false-alarm probability is at most delta, and at most T*delta/W
over a stream of length T by a union bound.

Cost per sample: one floating-point subtract, one comparison, a
conditional counter increment, and a window-position count.

Closed loop: when the alarm fires, labels are queried for a small budget B
of samples from the window that raised the alarm, the classifier is
retrained, and qhat is recomputed on the installation-time (batch-1)
calibration set, which is the permanent reliability reference and is never
trained on.

Data: UCI Gas Sensor Array Drift dataset (Vergara et al., 2012):
10 batches over 36 months, 16 MOX sensors, 128 features, 6 gases.

All linear algebra is forced to a single thread so that every run is
bitwise reproducible; multi-threaded BLAS changes the floating-point path of
the larger logistic-regression fits and their third decimals.
"""

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import numpy as np
from dataclasses import dataclass, field
from scipy.special import betaln, gammaln
from scipy.stats import binom
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

GASES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


# ----------------------------- data ---------------------------------

def load_batches(data_dir):
    """Load the ten batches of the UCI Gas Sensor Array Drift dataset.
    Returns a list of (X, y) with y in 0..5."""
    batches = []
    for b in range(1, 11):
        X, y = load_svmlight_file(f"{data_dir}/batch{b}.dat", n_features=128)
        batches.append((np.asarray(X.todense()), y.astype(int) - 1))
    return batches


def signed_log(X):
    """Compress the dynamic range of the MOX features (about 1e-2 to 1e5)."""
    return np.sign(X) * np.log1p(np.abs(X))


def make_model(seed):
    return LogisticRegression(max_iter=2000, C=1.0, random_state=seed)


def split_batch1(batches, seed, f_stream=0.2, f_cal=0.3):
    """Split batch 1 into train / calibration / stationary-control stream."""
    rng = np.random.RandomState(seed)
    X1, y1 = batches[0]
    idx = rng.permutation(len(y1))
    n = len(y1)
    n_str, n_cal = int(f_stream * n), int(f_cal * n)
    strm = idx[:n_str]
    cal = idx[n_str:n_str + n_cal]
    tr = idx[n_str + n_cal:]
    scaler = StandardScaler().fit(signed_log(X1[tr]))
    T = lambda Z: scaler.transform(signed_log(Z))
    return (rng, scaler,
            T(X1[tr]), y1[tr],
            T(X1[cal]), y1[cal],
            T(X1[strm]), y1[strm])


# ------------------------- conformal core ----------------------------

def conformal_qhat(model, X_cal, y_cal, alpha=0.10):
    """Split-conformal threshold for the score s(x,y) = 1 - p_y(x).

    Computed as the ceil((n+1)(1-alpha))/n empirical quantile of the
    calibration scores with the 'higher' method (Angelopoulos and Bates,
    2023). With r = ceil((n+1)(1-alpha)) < n this rule returns the (r+1)-th
    smallest score, and the n-th when r = n, so the returned value is never
    smaller than the r-th order statistic. That direction is conservative:
    the realized miscoverage is stochastically dominated by Beta(l, n+1-l)
    with l = floor(alpha(n+1)), which is what the Beta-Binomial certificate
    in bb_critical() assumes.
    """
    p = model.predict_proba(X_cal)
    scores = 1.0 - p[np.arange(len(y_cal)), y_cal]
    n = len(scores)
    q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, q_level, method="higher"))


def betabinom_sf(k, W, a, b):
    """P(X >= k) for X ~ BetaBinomial(W, a, b)."""
    j = np.arange(k, W + 1)
    logpmf = (gammaln(W + 1) - gammaln(j + 1) - gammaln(W - j + 1)
              + betaln(j + a, W - j + b) - betaln(a, b))
    return float(np.exp(logpmf).sum())


def bb_critical(W, n_cal, alpha, delta):
    """Smallest k with P(BetaBinomial(W, l, n+1-l) >= k) <= delta."""
    l = max(1, int(np.floor(alpha * (n_cal + 1))))
    a, b = l, n_cal + 1 - l
    k = int(np.ceil(W * alpha))
    while k <= W and betabinom_sf(k, W, a, b) > delta:
        k += 1
    return k


def naive_binomial_critical(W, alpha, delta):
    """Smallest k with P(Binomial(W, alpha) >= k) <= delta.

    This is the threshold obtained when the randomness of qhat is ignored,
    that is, when the per-sample empty-set probability is taken to be
    exactly alpha. It understates the calibration-conditional risk; see
    run_naive_control() for the empirical companion.
    """
    k = int(np.ceil(W * alpha))
    while k <= W and binom.sf(k - 1, W, alpha) > delta:
        k += 1
    return k


@dataclass
class Monitor:
    """Empty-set drift monitor over disjoint windows.

    Per-sample state that changes: the empty-set count and the window
    position. Stored constants: qhat, W and the critical value k_crit.
    """
    qhat: float
    n_cal: int
    W: int = 50
    alpha: float = 0.10
    delta: float = 1e-3
    count: int = 0
    pos: int = 0
    k_crit: int = 0

    def __post_init__(self):
        self.k_crit = bb_critical(self.W, self.n_cal, self.alpha, self.delta)

    def step(self, p_max):
        """Feed the maximum softmax probability of one sample. Returns True
        when the window that this sample closes reaches the critical value."""
        self.count += int((1.0 - p_max) > self.qhat)
        self.pos += 1
        if self.pos == self.W:
            alarm = self.count >= self.k_crit
            self.count = 0
            self.pos = 0
            return alarm
        return False

    def reset(self, qhat, n_cal):
        self.qhat, self.n_cal = qhat, n_cal
        self.k_crit = bb_critical(self.W, n_cal, self.alpha, self.delta)
        self.count, self.pos = 0, 0


# --------------------------- experiment runs -------------------------

@dataclass
class RunResult:
    name: str
    batch_acc: list = field(default_factory=list)    # accuracy, batches 2..10
    labels_used: int = 0
    alarms: list = field(default_factory=list)       # (batch_no, sample_idx)


class WindowPool:
    """Samples of the window currently being counted by a Monitor.

    The pool follows the monitor's window, not the batch structure, so a
    window that straddles a batch boundary offers all of its samples for
    labeling; the loop therefore uses no batch-boundary information.
    """

    def __init__(self):
        self.X, self.y, self.s = [], [], []

    def add(self, x, y, score):
        self.X.append(x); self.y.append(y); self.s.append(score)

    def clear(self):
        self.X, self.y, self.s = [], [], []

    def query(self, rng, budget, ranked):
        """Return (X, y) of up to `budget` samples: the most nonconforming
        when ranked, otherwise a uniform random subset."""
        k = min(budget, len(self.y))
        if ranked:
            pick = np.argsort(np.asarray(self.s))[::-1][:k]
        else:
            pick = rng.choice(len(self.y), size=k, replace=False)
        return np.asarray(self.X)[pick], np.asarray(self.y)[pick]


def _retrain(seed, X_src, y_src, pool_X, pool_y, pool_weight=5.0):
    """Retrain on installation data plus all queried labels; every queried
    label is weighted pool_weight relative to the installation data."""
    model = make_model(seed)
    if pool_X:
        Xtr = np.vstack([X_src] + pool_X)
        ytr = np.concatenate([y_src] + pool_y)
        w = np.concatenate([np.ones(len(y_src)),
                            pool_weight * np.ones(sum(len(p) for p in pool_y))])
        model.fit(Xtr, ytr, sample_weight=w)
    else:
        model.fit(X_src, y_src)
    return model


def run_static(batches, seed):
    _, scaler, Xtr, ytr, *_ = split_batch1(batches, seed)
    model = make_model(seed).fit(Xtr, ytr)
    res = RunResult(name="static")
    for b in range(1, 10):
        Xb = scaler.transform(signed_log(batches[b][0]))
        res.batch_acc.append(float((model.predict(Xb) == batches[b][1]).mean()))
    return res


def run_periodic(batches, seed, budget=20):
    """Boundary-privileged baseline: knows batch boundaries; queries `budget`
    random labels at the start of every batch. Queried samples are excluded
    from its evaluation (the gated loop, by contrast, is scored prequentially
    on every sample)."""
    rng, scaler, Xtr, ytr, *_ = split_batch1(batches, seed)
    pool_X, pool_y = [], []
    res = RunResult(name=f"periodic-{budget}")
    for b in range(1, 10):
        Xb = scaler.transform(signed_log(batches[b][0]))
        yb = batches[b][1]
        pick = rng.choice(len(yb), size=min(budget, len(yb)), replace=False)
        pool_X.append(Xb[pick]); pool_y.append(yb[pick])
        res.labels_used += len(pick)
        model = _retrain(seed, Xtr, ytr, pool_X, pool_y)
        mask = np.ones(len(yb), bool); mask[pick] = False
        res.batch_acc.append(float((model.predict(Xb[mask]) == yb[mask]).mean()))
    return res


def run_oracle(batches, seed):
    """Full relabeling: before each batch, retrain on every label of all
    earlier batches."""
    _, scaler, Xtr, ytr, *_ = split_batch1(batches, seed)
    res = RunResult(name="oracle")
    accX, accy = [Xtr], [ytr]
    for b in range(1, 10):
        model = make_model(seed).fit(np.vstack(accX), np.concatenate(accy))
        Xb = scaler.transform(signed_log(batches[b][0]))
        yb = batches[b][1]
        res.batch_acc.append(float((model.predict(Xb) == yb).mean()))
        res.labels_used += len(yb)
        accX.append(Xb); accy.append(yb)
    return res


def run_cadence(batches, seed, budget=20, alpha=0.10, W=50, delta=1e-3,
                ranked=False, name="cadence"):
    """Closed loop: certified alarm, then B labels queried from the alarmed
    window, retraining on installation data plus all queried labels, and
    qhat recomputed on the original batch-1 calibration set under the new
    model (the permanent reliability reference; it is never trained on).
    Prequential: every prediction is committed before any label from its own
    window can influence the model. Windows run across batch boundaries."""
    rng, scaler, Xtr, ytr, Xcal, ycal, _, _ = split_batch1(batches, seed)
    model = make_model(seed).fit(Xtr, ytr)
    qhat = conformal_qhat(model, Xcal, ycal, alpha)
    mon = Monitor(qhat=qhat, n_cal=len(ycal), W=W, alpha=alpha, delta=delta)

    train_X, train_y = [], []   # all queried labels; used only for retraining
    window = WindowPool()
    res = RunResult(name=name)

    for b in range(1, 10):
        Xb = scaler.transform(signed_log(batches[b][0]))
        yb = batches[b][1]
        P = model.predict_proba(Xb)
        correct = 0
        for i in range(len(yb)):
            p = P[i]
            correct += int(np.argmax(p) == yb[i])   # commit prediction first
            window.add(Xb[i], yb[i], 1.0 - p.max())
            if mon.step(p.max()):
                res.alarms.append((b + 1, i))
                qX, qy = window.query(rng, budget, ranked)
                train_X.append(qX); train_y.append(qy)
                res.labels_used += len(qy)
                model = _retrain(seed, Xtr, ytr, train_X, train_y)
                qhat = conformal_qhat(model, Xcal, ycal, alpha)
                mon.reset(qhat, len(ycal))
                if i + 1 < len(yb):
                    P[i + 1:] = model.predict_proba(Xb[i + 1:])
            if mon.pos == 0:
                window.clear()
        res.batch_acc.append(correct / len(yb))
    return res


def run_detect_only(batches, seed, W=50, alpha=0.10, delta=1e-3):
    """Detection metrics with a frozen batch-1 model (no adaptation):
    first-alarm sample index per batch, plus per-window empty counts."""
    _, scaler, Xtr, ytr, Xcal, ycal, _, _ = split_batch1(batches, seed)
    model = make_model(seed).fit(Xtr, ytr)
    qhat = conformal_qhat(model, Xcal, ycal, alpha)
    k_crit = bb_critical(W, len(ycal), alpha, delta)
    out = {"k_crit": k_crit, "batches": []}
    for b in range(1, 10):
        Xb = scaler.transform(signed_log(batches[b][0]))
        p = model.predict_proba(Xb)
        empty = ((1.0 - p.max(axis=1)) > qhat).astype(int)
        nwin = len(empty) // W
        counts = empty[:nwin * W].reshape(nwin, W).sum(axis=1)
        hits = np.where(counts >= k_crit)[0]
        out["batches"].append({
            "first_alarm": int((hits[0] + 1) * W) if len(hits) else None,
            "counts": counts.tolist(),
            "empty_rate_all": float(empty.mean()),
            "acc": float((model.predict(Xb) == batches[b][1]).mean()),
        })
    return out


def run_stationary_control(batches, seed, T=2000, W=50, alpha=0.10, delta=1e-3):
    """False-alarm test on an exchangeable stream drawn iid from held-out
    batch-1 data. Fresh split, model, and calibration per seed. Returns
    (alarm windows, total windows); the stream-level union bound on the
    probability of any alarm is T*delta/W."""
    rng, scaler, Xtr, ytr, Xcal, ycal, Xstr, ystr = split_batch1(batches, seed)
    model = make_model(seed).fit(Xtr, ytr)
    qhat = conformal_qhat(model, Xcal, ycal, alpha)
    k_crit = bb_critical(W, len(ycal), alpha, delta)
    Xctl = Xstr[rng.choice(len(Xstr), T, replace=True)]
    p = model.predict_proba(Xctl)
    empty = ((1.0 - p.max(axis=1)) > qhat).astype(int)
    counts = empty[:(T // W) * W].reshape(T // W, W).sum(axis=1)
    return int((counts >= k_crit).sum()), T // W


def run_naive_control(batches, seed=0, n_streams=20, T=2000, W=50,
                      alpha=0.10, delta=1e-3):
    """Stationarity control against the naive binomial critical value, with
    one calibration draw shared by all streams. Empirical companion to the
    analytical comparison of the two thresholds (Section IV-B of the
    letter): the naive threshold's believed per-window level is
    P(Bin(W,alpha) >= k), but its worst-case level under the Beta-Binomial
    law is larger, and because every window shares the same realized qhat, a single
    unlucky calibration draw concentrates the excess into many windows.

    Returns a dict with both thresholds, their exact per-window levels,
    and the observed alarm counts over the shared-draw streams.
    """
    rng0, scaler, Xtr, ytr, Xcal, ycal, Xstr, ystr = split_batch1(batches, seed)
    model = make_model(seed).fit(Xtr, ytr)
    qhat = conformal_qhat(model, Xcal, ycal, alpha)
    l = max(1, int(np.floor(alpha * (len(ycal) + 1))))
    k_naive = naive_binomial_critical(W, alpha, delta)
    k_bb = bb_critical(W, len(ycal), alpha, delta)
    alarm_windows = streams_with_alarm = 0
    for s in range(n_streams):
        rng = np.random.RandomState(10_000 + s)
        Xctl = Xstr[rng.choice(len(Xstr), T, replace=True)]
        p = model.predict_proba(Xctl)
        empty = ((1.0 - p.max(axis=1)) > qhat).astype(int)
        counts = empty[:(T // W) * W].reshape(T // W, W).sum(axis=1)
        a = int((counts >= k_naive).sum())
        alarm_windows += a
        streams_with_alarm += int(a > 0)
    return {
        "k_naive": k_naive,
        "k_betabinom": k_bb,
        "l": l,
        "betabinom_mean": float(W * l / (len(ycal) + 1)),
        "naive_believed_level": float(binom.sf(k_naive - 1, W, alpha)),
        "naive_true_level": betabinom_sf(k_naive, W, l, len(ycal) + 1 - l),
        "certified_true_level": betabinom_sf(k_bb, W, l, len(ycal) + 1 - l),
        "n_streams": n_streams,
        "windows_per_stream": T // W,
        "alarm_windows": alarm_windows,
        "streams_with_alarm": streams_with_alarm,
    }
