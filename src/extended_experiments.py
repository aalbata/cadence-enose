"""Extended experiment suite. Writes extended_results.json.

Second classifier family (32-unit multilayer perceptron); full-relabeling
variants with recency weighting and a sliding window; classical feature-space
monitors (Hotelling T^2 and PCA squared prediction error) under the same
window test, with closed-loop versions; sensitivity to calibration-set size
and nonconformity score; an interleaved-gas stream obtained by shuffling
each batch internally; and a post-adaptation false-alarm measurement on the
stationarity control.
"""
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cadence import (Monitor, RunResult, WindowPool, bb_critical,
                     conformal_qhat, load_batches, make_model, signed_log,
                     split_batch1)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Dataset")
SEEDS = range(10)


def make_mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(32,), max_iter=800,
                         random_state=seed, alpha=1e-3)


# ------------------------------------------------------------------ generic loop
def retrain(factory, seed, X_src, y_src, pool_X, pool_y, w=5.0):
    """Retrain on installation data plus all queried labels, queried labels
    weighted w times. Estimators without sample_weight support receive the
    queried samples replicated round(w) times instead."""
    model = factory(seed)
    if pool_X:
        Xtr = np.vstack([X_src] + pool_X)
        ytr = np.concatenate([y_src] + pool_y)
        try:
            model.fit(Xtr, ytr, sample_weight=np.concatenate(
                [np.ones(len(y_src)), w * np.ones(sum(len(p) for p in pool_y))]))
        except TypeError:
            reps = max(1, int(round(w)))
            Xr = np.vstack([X_src] + pool_X * reps)
            yr = np.concatenate([y_src] + pool_y * reps)
            model.fit(Xr, yr)
    else:
        model.fit(X_src, y_src)
    return model


def run_loop(batches, seed, factory, budget=10, alpha=0.10, W=50, delta=1e-3,
             shuffle_within_batch=False, ranked=False):
    """Closed loop parameterised by classifier factory and stream order.
    Windows run across batch boundaries (see cadence.WindowPool)."""
    rng, scaler, Xtr, ytr, Xcal, ycal, _, _ = split_batch1(batches, seed)
    model = retrain(factory, seed, Xtr, ytr, [], [])
    qhat = conformal_qhat(model, Xcal, ycal, alpha)
    mon = Monitor(qhat=qhat, n_cal=len(ycal), W=W, alpha=alpha, delta=delta)
    train_X, train_y = [], []
    window = WindowPool()
    res = RunResult(name="loop")
    order_rng = np.random.RandomState(777 + seed)
    for b in range(1, 10):
        Xb = scaler.transform(signed_log(batches[b][0]))
        yb = batches[b][1]
        if shuffle_within_batch:
            perm = order_rng.permutation(len(yb))
            Xb, yb = Xb[perm], yb[perm]
        P = model.predict_proba(Xb)
        correct = 0
        for i in range(len(yb)):
            correct += int(np.argmax(P[i]) == yb[i])
            window.add(Xb[i], yb[i], 1.0 - P[i].max())
            if mon.step(P[i].max()):
                res.alarms.append((b + 1, i))
                qX, qy = window.query(rng, budget, ranked)
                train_X.append(qX); train_y.append(qy)
                res.labels_used += len(qy)
                model = retrain(factory, seed, Xtr, ytr, train_X, train_y)
                qhat = conformal_qhat(model, Xcal, ycal, alpha)
                mon.reset(qhat, len(ycal))
                if i + 1 < len(yb):
                    P[i + 1:] = model.predict_proba(Xb[i + 1:])
            if mon.pos == 0:
                window.clear()
        res.batch_acc.append(correct / len(yb))
    return res


def run_static_f(batches, seed, factory):
    _, scaler, Xtr, ytr, *_ = split_batch1(batches, seed)
    model = retrain(factory, seed, Xtr, ytr, [], [])
    return [float((model.predict(scaler.transform(signed_log(batches[b][0])))
                   == batches[b][1]).mean()) for b in range(1, 10)]


def detect_f(batches, seed, factory, alpha=0.10, W=50, delta=1e-3):
    _, scaler, Xtr, ytr, Xcal, ycal, _, _ = split_batch1(batches, seed)
    model = retrain(factory, seed, Xtr, ytr, [], [])
    qhat = conformal_qhat(model, Xcal, ycal, alpha)
    k = bb_critical(W, len(ycal), alpha, delta)
    delays, fired = [], 0
    for b in range(1, 10):
        Xb = scaler.transform(signed_log(batches[b][0]))
        p = model.predict_proba(Xb)
        empty = ((1.0 - p.max(axis=1)) > qhat).astype(int)
        nwin = len(empty) // W
        counts = empty[:nwin * W].reshape(nwin, W).sum(axis=1)
        hits = np.where(counts >= k)[0]
        fired += int(len(hits) > 0)
        delays.append(int((hits[0] + 1) * W) if len(hits) else None)
    return fired, delays


def control_f(batches, seed, factory, T=2000, W=50, alpha=0.10, delta=1e-3):
    rng, scaler, Xtr, ytr, Xcal, ycal, Xstr, _ = split_batch1(batches, seed)
    model = retrain(factory, seed, Xtr, ytr, [], [])
    qhat = conformal_qhat(model, Xcal, ycal, alpha)
    k = bb_critical(W, len(ycal), alpha, delta)
    Xctl = Xstr[rng.choice(len(Xstr), T, replace=True)]
    p = model.predict_proba(Xctl)
    empty = ((1.0 - p.max(axis=1)) > qhat).astype(int)
    counts = empty[:(T // W) * W].reshape(T // W, W).sum(axis=1)
    return int((counts >= k).sum()), T // W


# --------------------------------------------------- relabeling baselines
def run_relabel(batches, seed, mode, recency_w=5.0):
    """Full relabeling variants. mode 'recency' weights the latest batch's
    labels recency_w times; mode 'window' trains on batch 1 plus the previous
    batch only."""
    _, scaler, Xtr, ytr, *_ = split_batch1(batches, seed)
    res = RunResult(name=f"relabel-{mode}")
    accX, accy = [], []
    for b in range(1, 10):
        if mode == "recency":
            model = make_model(seed)
            if accX:
                Xall = np.vstack([Xtr] + accX)
                yall = np.concatenate([ytr] + accy)
                wts = [np.ones(len(ytr))]
                for j, yj in enumerate(accy):
                    wts.append((recency_w if j == len(accy) - 1 else 1.0)
                               * np.ones(len(yj)))
                model.fit(Xall, yall, sample_weight=np.concatenate(wts))
            else:
                model.fit(Xtr, ytr)
        else:  # window
            if accX:
                model = make_model(seed).fit(np.vstack([Xtr, accX[-1]]),
                                             np.concatenate([ytr, accy[-1]]))
            else:
                model = make_model(seed).fit(Xtr, ytr)
        Xb = scaler.transform(signed_log(batches[b][0]))
        yb = batches[b][1]
        res.batch_acc.append(float((model.predict(Xb) == yb).mean()))
        res.labels_used += len(yb)
        accX.append(Xb); accy.append(yb)
    return res


# ------------------------------------------- classical feature-space monitors
class FeatureMonitor:
    """Hotelling T^2 or PCA squared prediction error (SPE) with the same
    window test as the conformal monitor.

    The per-sample event is 'statistic exceeds the empirical (1-alpha)
    calibration quantile', taken at the same quantile rank as the conformal
    threshold, and the window count is compared with the same beta-binomial
    critical value. The principal components, the per-component scale and
    the threshold are all fitted on the calibration set, so the calibration
    statistics are in-sample and the exchangeability argument behind the
    conformal certificate does not transfer to these monitors; their
    false-alarm behavior is reported empirically from the stationary
    control. The comparison isolates the choice of monitored statistic
    (feature-space distance versus classifier confidence)."""

    def __init__(self, kind, Xcal, alpha=0.10, var_keep=0.95):
        self.kind = kind
        self.pca = PCA(n_components=var_keep, svd_solver="full").fit(Xcal)
        Z = self.pca.transform(Xcal)
        self.scale = Z.std(axis=0, ddof=1) + 1e-12
        stat = self._stat(Xcal)
        n = len(stat)
        r = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        self.thr = float(np.quantile(stat, r, method="higher"))

    def _stat(self, X):
        Z = self.pca.transform(X)
        if self.kind == "t2":
            return ((Z / self.scale) ** 2).sum(axis=1)
        R = X - self.pca.inverse_transform(Z)
        return (R ** 2).sum(axis=1)

    def exceed(self, X):
        return (self._stat(X) > self.thr).astype(int)


def feature_monitor_run(batches, seed, kind, budget=10, W=50, alpha=0.10,
                        delta=1e-3, closed_loop=True):
    rng, scaler, Xtr, ytr, Xcal, ycal, Xstr, _ = split_batch1(batches, seed)
    fm = FeatureMonitor(kind, Xcal, alpha)
    k = bb_critical(W, len(ycal), alpha, delta)

    # control false alarms
    Xctl = Xstr[rng.choice(len(Xstr), 2000, replace=True)]
    e = fm.exceed(Xctl)
    ctl_counts = e[:(2000 // W) * W].reshape(-1, W).sum(axis=1)
    ctl_alarms = int((ctl_counts >= k).sum())
    ctl_windows = int(len(ctl_counts))

    # detection with frozen reference
    fired, delays = 0, []
    for b in range(1, 10):
        Xb = scaler.transform(signed_log(batches[b][0]))
        e = fm.exceed(Xb)
        nwin = len(e) // W
        counts = e[:nwin * W].reshape(nwin, W).sum(axis=1)
        hits = np.where(counts >= k)[0]
        fired += int(len(hits) > 0)
        delays.append(int((hits[0] + 1) * W) if len(hits) else None)

    out = {"ctl_alarms": ctl_alarms, "ctl_windows": ctl_windows, "fired": fired, "delays": delays}
    if not closed_loop:
        return out

    # closed loop gated by the feature alarm; same query and retraining policy;
    # the feature monitor itself is model-free and is never refitted
    model = make_model(seed).fit(Xtr, ytr)
    train_X, train_y = [], []
    res = RunResult(name=f"{kind}-loop")
    window = WindowPool()
    count = pos = 0
    for b in range(1, 10):
        Xb = scaler.transform(signed_log(batches[b][0]))
        yb = batches[b][1]
        P = model.predict_proba(Xb)
        exceed = fm.exceed(Xb)
        correct = 0
        for i in range(len(yb)):
            correct += int(np.argmax(P[i]) == yb[i])
            window.add(Xb[i], yb[i], 0.0)
            count += exceed[i]; pos += 1
            if pos == W:
                if count >= k:
                    res.alarms.append((b + 1, i))
                    qX, qy = window.query(rng, budget, ranked=False)
                    train_X.append(qX); train_y.append(qy)
                    res.labels_used += len(qy)
                    model = retrain(make_model, seed, Xtr, ytr, train_X, train_y)
                    if i + 1 < len(yb):
                        P[i + 1:] = model.predict_proba(Xb[i + 1:])
                count = pos = 0
                window.clear()
        res.batch_acc.append(correct / len(yb))
    out["loop_acc"] = float(np.mean(res.batch_acc))
    out["loop_labels"] = res.labels_used
    return out


# ------------------------------------------------------- score / n sensitivity
def margin_scores(P, y=None):
    top2 = np.sort(P, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    if y is None:                       # deployment statistic for the top class
        return 1.0 - margin
    is_top = (np.argmax(P, axis=1) == y)
    s = np.where(is_top, 1.0 - margin, 1.0 + margin)  # non-top classes worse
    return s


def sensitivity_run(batches, seed, f_cal=0.3, score="pmax",
                    W=50, alpha=0.10, delta=1e-3):
    rng, scaler, Xtr, ytr, Xcal, ycal, Xstr, _ = split_batch1(
        batches, seed, f_cal=f_cal)
    model = make_model(seed).fit(Xtr, ytr)
    Pc = model.predict_proba(Xcal)
    if score == "pmax":
        s_cal = 1.0 - Pc[np.arange(len(ycal)), ycal]
    else:
        s_cal = margin_scores(Pc, ycal)
    n = len(s_cal)
    r = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    qhat = float(np.quantile(s_cal, r, method="higher"))
    k = bb_critical(W, n, alpha, delta)

    Xctl = Xstr[rng.choice(len(Xstr), 2000, replace=True)]

    def empty(X):
        P = model.predict_proba(X)
        stat = (1.0 - P.max(axis=1)) if score == "pmax" else margin_scores(P)
        return (stat > qhat).astype(int)

    e = empty(Xctl)
    ctl = int((e[:(2000 // W) * W].reshape(-1, W).sum(axis=1) >= k).sum())
    fired, delays = 0, []
    for b in range(1, 10):
        Xb = scaler.transform(signed_log(batches[b][0]))
        e = empty(Xb)
        nwin = len(e) // W
        counts = e[:nwin * W].reshape(nwin, W).sum(axis=1)
        hits = np.where(counts >= k)[0]
        fired += int(len(hits) > 0)
        delays.append(int((hits[0] + 1) * W) if len(hits) else None)
    return {"n_cal": n, "k_crit": k, "ctl_alarms": ctl, "fired": fired,
            "median_delay": float(np.median([d for d in delays if d]))}


# ------------------------------------------------ post-adaptation false alarms
def post_adapt_control(batches, seed, T=2000, W=50, alpha=0.10, delta=1e-3,
                       budget=10, n_recal=1):
    """Stationary stream; n_recal benign recalibrations are forced at equally
    spaced points (each queries `budget` labels from the W samples preceding
    it, as an alarm would), then false alarms are counted in the windows
    after the last recalibration."""
    rng, scaler, Xtr, ytr, Xcal, ycal, Xstr, ystr = split_batch1(batches, seed)
    model = make_model(seed).fit(Xtr, ytr)
    pick_stream = rng.choice(len(Xstr), T, replace=True)
    Xctl, yctl = Xstr[pick_stream], ystr[pick_stream]
    train_X, train_y = [], []
    recal_at = [int((j + 1) * T / (n_recal + 1)) for j in range(n_recal)]
    for t in recal_at:
        widx = np.arange(max(0, t - W), t)
        pick = rng.choice(widx, size=budget, replace=False)
        train_X.append(Xctl[pick]); train_y.append(yctl[pick])
    model = retrain(make_model, seed, Xtr, ytr, train_X, train_y)
    qhat = conformal_qhat(model, Xcal, ycal, alpha)
    k = bb_critical(W, len(ycal), alpha, delta)
    start = recal_at[-1]
    e = ((1.0 - model.predict_proba(Xctl[start:]).max(axis=1)) > qhat).astype(int)
    nwin = len(e) // W
    counts = e[:nwin * W].reshape(nwin, W).sum(axis=1)
    return int((counts >= k).sum()), nwin


# ----------------------------------------------------------------------- main
def summarize(name, accs, labels):
    a = np.array(accs)
    print(f"{name:26s} acc={a.mean():.3f}+-{a.std():.3f} "
          f"labels={np.mean(labels):.0f}", flush=True)
    return {"acc_per_seed": a.tolist(), "labels": float(np.mean(labels))}


def main():
    t0 = time.time()
    batches = load_batches(DATA)
    out = {}

    print("== classical feature-space monitors ==", flush=True)
    for kind in ("t2", "spe"):
        det_f_, ctl, n_ctl, loops, labl, delays_all = [], 0, 0, [], [], []
        for s in SEEDS:
            r = feature_monitor_run(batches, s, kind)
            det_f_.append(r["fired"]); ctl += r["ctl_alarms"]
            n_ctl += r["ctl_windows"]
            loops.append(r["loop_acc"]); labl.append(r["loop_labels"])
            delays_all.append(r["delays"])
        out[f"fm-{kind}"] = {
            "fired_batches_mean": float(np.mean(det_f_)),
            "ctl_alarm_windows_total": ctl,
            "ctl_windows": n_ctl,
            "loop": summarize(f"{kind}-gated loop", loops, labl),
            "delays": delays_all}
        print(f"  {kind}: fired {np.mean(det_f_):.1f}/9 batches, "
              f"control alarms {ctl}/{n_ctl}", flush=True)

    print("== relabeling baselines ==", flush=True)
    for mode in ("recency", "window"):
        runs = [run_relabel(batches, s, mode) for s in SEEDS]
        out[f"relabel-{mode}"] = summarize(f"relabel-{mode}",
                                           [np.mean(r.batch_acc) for r in runs],
                                           [r.labels_used for r in runs])

    print("== sensitivity (score, calibration size) ==", flush=True)
    sens = {}
    for f_cal, score in [(0.3, "pmax"), (0.3, "margin"),
                         (0.15, "pmax"), (0.45, "pmax")]:
        rows = [sensitivity_run(batches, s, f_cal, score) for s in SEEDS]
        key = f"fcal{f_cal}-{score}"
        sens[key] = {
            "n_cal": rows[0]["n_cal"], "k_crit": rows[0]["k_crit"],
            "fired_mean": float(np.mean([r["fired"] for r in rows])),
            "ctl_total": int(np.sum([r["ctl_alarms"] for r in rows])),
            "median_delay_mean": float(np.mean([r["median_delay"] for r in rows]))}
        print(f"  {key}: {sens[key]}", flush=True)
    out["sensitivity"] = sens

    print("== post-adaptation stationarity control ==", flush=True)
    for n_recal in (1, 3):
        alarms = wins = 0
        for s in range(50):
            a, w = post_adapt_control(batches, s, n_recal=n_recal)
            alarms += a; wins += w
        out[f"post-adapt-{n_recal}"] = {"alarm_windows": alarms, "windows": wins,
                                        "streams": 50}
        print(f"  {n_recal} recal: {alarms}/{wins} windows alarmed", flush=True)

    print("== interleaved-gas stream (within-batch shuffle) ==", flush=True)
    for ranked in (False, True):
        accs, labl = [], []
        for s in SEEDS:
            r = run_loop(batches, s, make_model, budget=20,
                         shuffle_within_batch=True, ranked=ranked)
            accs.append(np.mean(r.batch_acc)); labl.append(r.labels_used)
        out[f"interleaved-B20-{'ranked' if ranked else 'random'}"] = \
            summarize(f"interleaved {'ranked' if ranked else 'random'}",
                      accs, labl)

    print("== second classifier (MLP, 32 hidden units) ==", flush=True)
    st = [np.mean(run_static_f(batches, s, make_mlp)) for s in SEEDS]
    out["mlp-static"] = summarize("mlp-static", st, [0])
    det = [detect_f(batches, s, make_mlp) for s in SEEDS]
    out["mlp-detect"] = {
        "fired_batches_mean": float(np.mean([d[0] for d in det])),
        "delays": [d[1] for d in det]}
    print(f"  detection: fired {out['mlp-detect']['fired_batches_mean']:.1f}/9",
          flush=True)
    ctl = [control_f(batches, s, make_mlp) for s in SEEDS]
    out["mlp-control"] = {"alarms": int(np.sum([c[0] for c in ctl])),
                          "windows": int(np.sum([c[1] for c in ctl])),
                          "streams_with_alarm": int(np.sum([c[0] > 0 for c in ctl])),
                          "streams": len(ctl)}
    print(f"  control: {out['mlp-control']}", flush=True)
    accs, labl = [], []
    for s in SEEDS:
        r = run_loop(batches, s, make_mlp, budget=10)
        accs.append(np.mean(r.batch_acc)); labl.append(r.labels_used)
    out["mlp-cadence-10"] = summarize("mlp-cadence-10", accs, labl)
    out["runtime_seconds"] = round(time.time() - t0)

    with open(os.path.join(ROOT, "results", "extended_results.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"written extended_results.json in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
