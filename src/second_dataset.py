"""Detection check on a second, recent MOX dataset. Writes enose_results.json.

Data: the one-year electronic-nose drift dataset of Wörner, Eimler and
Pein-Hackelbusch (Scientific Data, 2025), Zenodo record 15681119, CC BY 4.0.
The archive (141 MB) is downloaded on first use and its MD5 checked. Only
the authors' feature table (extracted_features.csv, four features per
sensor for 62 sensors plus temperature and humidity channels, 624 labeled
recordings on 36 measurement days between 16 February 2024 and
5 February 2025) is used; the raw time series are not read.

Protocol, fixed before the data were examined for drift: the first six
measurement days are the installation period and are split per seed into
50 percent training, 30 percent calibration and 20 percent stationary
control, exactly as batch 1 of the primary benchmark. A window is one
measurement day (its W samples in recording order), so the critical value
is computed per day from W, n, alpha and delta with the same beta-binomial
rule. The frozen installation-time classifier is then replayed over the
remaining days; the stationarity control resamples the held-out installation
samples into synthetic days; and the closed loop queries B labels per alarm.
The two feature-space monitors of the primary study (Hotelling T^2 and
PCA-SPE) are run on the same day windows for comparison.
"""
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import hashlib
import io
import json
import sys
import urllib.request
import zipfile

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cadence import (Monitor, WindowPool, _retrain, bb_critical,
                     conformal_qhat, make_model, signed_log)
from extended_experiments import FeatureMonitor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "enose")
URL = "https://zenodo.org/records/15681119/files/Dataset.zip?download=1"
EXPECTED_MD5 = "9937678cac4118c53287276009172a74"
INSTALL_DAYS = 6
ALPHA, DELTA, BUDGET = 0.10, 1e-3, 3
SEEDS = range(10)
CONTROL_SEEDS = 50
CONTROL_DAYS = 100          # synthetic days per control stream
CONTROL_W = 18              # samples per synthetic day (the protocol's day length)


def ensure_data():
    path = os.path.join(DATA, "Dataset.zip")
    if not os.path.exists(path):
        os.makedirs(DATA, exist_ok=True)
        print("downloading the one-year e-nose dataset (141 MB) ...", flush=True)
        with urllib.request.urlopen(URL) as r, open(path, "wb") as f:
            f.write(r.read())
    digest = hashlib.md5(open(path, "rb").read()).hexdigest()
    if digest != EXPECTED_MD5:
        sys.exit(f"e-nose archive md5 mismatch: {digest}")
    return path


def load_features(path):
    """Return X (float), y (0..4), day index (0..35) and the date strings."""
    z = zipfile.ZipFile(path)
    rows = [ln.split() for ln in io.TextIOWrapper(z.open("extracted_features.csv"),
                                                  encoding="utf-8")]
    header, body = rows[0], rows[1:]
    # class names contain a space ("Diacetyl 0.1ppm"): the label is every token
    # between the numeric features and the trailing date token
    n_feat = len(header) - 2
    X = np.array([[float(v) for v in r[:n_feat]] for r in body])
    labels = [" ".join(r[n_feat:-1]).strip('"') for r in body]
    dates = [r[-1] for r in body]                       # yy-mm-dd
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    days_sorted = sorted(set(dates))                    # yy-mm-dd sorts chronologically
    day = np.array([days_sorted.index(d) for d in dates])
    return X, y, day, days_sorted, classes


def split_install(X, y, day, seed, f_stream=0.2, f_cal=0.3):
    rng = np.random.RandomState(seed)
    inst = np.where(day < INSTALL_DAYS)[0]
    idx = rng.permutation(inst)
    n = len(idx)
    n_str, n_cal = int(f_stream * n), int(f_cal * n)
    strm, cal, tr = idx[:n_str], idx[n_str:n_str + n_cal], idx[n_str + n_cal:]
    scaler = StandardScaler().fit(signed_log(X[tr]))
    T = lambda Z: scaler.transform(signed_log(Z))
    return rng, T, T(X[tr]), y[tr], T(X[cal]), y[cal], T(X[strm]), y[strm]


def detect(X, y, day, seed):
    """Frozen installation model over the post-installation days."""
    _, T, Xtr, ytr, Xcal, ycal, Xstr, ystr = split_install(X, y, day, seed)
    model = make_model(seed).fit(Xtr, ytr)
    qhat = conformal_qhat(model, Xcal, ycal, ALPHA)
    out = []
    pmax_post = []
    for d in range(INSTALL_DAYS, day.max() + 1):
        idx = np.where(day == d)[0]
        Xd, yd = T(X[idx]), y[idx]
        p = model.predict_proba(Xd)
        pmax_post.extend(p.max(axis=1).tolist())
        empties = int(((1.0 - p.max(axis=1)) > qhat).sum())
        k = bb_critical(len(idx), len(ycal), ALPHA, DELTA)
        out.append({"day": d, "W": len(idx), "k": k, "empties": empties,
                    "alarm": bool(empties >= k),
                    "acc": float((model.predict(Xd) == yd).mean())})
    stats = {"n_cal": len(ycal),
             "install_acc": float((model.predict(Xstr) == ystr).mean()),
             "pmax_cal": float(model.predict_proba(Xcal).max(axis=1).mean()),
             "pmax_post": float(np.mean(pmax_post))}
    return out, stats


def control(X, y, day, seed):
    """Stationarity control: synthetic days resampled from the held-out
    installation samples under the same frozen model and threshold."""
    rng, _, Xtr, ytr, Xcal, ycal, Xstr, _ = split_install(X, y, day, seed)
    model = make_model(seed).fit(Xtr, ytr)
    qhat = conformal_qhat(model, Xcal, ycal, ALPHA)
    k = bb_critical(CONTROL_W, len(ycal), ALPHA, DELTA)
    Xctl = Xstr[rng.choice(len(Xstr), CONTROL_DAYS * CONTROL_W, replace=True)]
    e = ((1.0 - model.predict_proba(Xctl).max(axis=1)) > qhat).astype(int)
    counts = e.reshape(CONTROL_DAYS, CONTROL_W).sum(axis=1)
    return int((counts >= k).sum()), CONTROL_DAYS, k


def feature_monitor(X, y, day, seed, kind):
    """Hotelling T^2 or PCA-SPE on the same day windows and the same
    stationarity control, fitted on the installation calibration set."""
    rng, T, Xtr, ytr, Xcal, ycal, Xstr, _ = split_install(X, y, day, seed)
    fm = FeatureMonitor(kind, Xcal, ALPHA)
    alarm_days, first = 0, None
    for d in range(INSTALL_DAYS, day.max() + 1):
        idx = np.where(day == d)[0]
        e = int(fm.exceed(T(X[idx])).sum())
        if e >= bb_critical(len(idx), len(ycal), ALPHA, DELTA):
            alarm_days += 1
            first = first or (d - INSTALL_DAYS + 1)
    k = bb_critical(CONTROL_W, len(ycal), ALPHA, DELTA)
    Xctl = Xstr[rng.choice(len(Xstr), CONTROL_DAYS * CONTROL_W, replace=True)]
    counts = fm.exceed(Xctl).reshape(CONTROL_DAYS, CONTROL_W).sum(axis=1)
    return alarm_days, first, int((counts >= k).sum())


def closed_loop(X, y, day, seed):
    """Alarm-gated recalibration over the post-installation days with
    day-aligned windows; prequential accuracy per day."""
    rng, T, Xtr, ytr, Xcal, ycal, _, _ = split_install(X, y, day, seed)
    model = make_model(seed).fit(Xtr, ytr)
    qhat = conformal_qhat(model, Xcal, ycal, ALPHA)
    train_X, train_y, accs, labels, alarms = [], [], [], 0, 0
    for d in range(INSTALL_DAYS, day.max() + 1):
        idx = np.where(day == d)[0]
        Xd, yd = T(X[idx]), y[idx]
        W = len(idx)
        mon = Monitor(qhat=qhat, n_cal=len(ycal), W=W, alpha=ALPHA, delta=DELTA)
        window = WindowPool()
        P = model.predict_proba(Xd)
        correct = 0
        for i in range(W):
            correct += int(np.argmax(P[i]) == yd[i])
            window.add(Xd[i], yd[i], 1.0 - P[i].max())
            if mon.step(P[i].max()):
                alarms += 1
                qX, qy = window.query(rng, BUDGET, ranked=False)
                train_X.append(qX); train_y.append(qy); labels += len(qy)
                model = _retrain(seed, Xtr, ytr, train_X, train_y)
                qhat = conformal_qhat(model, Xcal, ycal, ALPHA)
        accs.append(correct / W)
    return accs, labels, alarms


def main():
    path = ensure_data()
    X, y, day, days, classes = load_features(path)
    n_days = len(days)
    dets, stats = [], []
    for s in SEEDS:
        out, st = detect(X, y, day, s)
        dets.append(out); stats.append(st)
    n_cal = stats[0]["n_cal"]
    fm = {kind: [feature_monitor(X, y, day, s, kind) for s in range(CONTROL_SEEDS)]
          for kind in ("t2", "spe")}
    eval_days = n_days - INSTALL_DAYS
    alarm_days = [sum(r["alarm"] for r in out) for out in dets]
    first = [next((r["day"] - INSTALL_DAYS + 1 for r in out if r["alarm"]), None)
             for out in dets]
    static_acc = [float(np.mean([r["acc"] for r in out])) for out in dets]
    ctl = [control(X, y, day, s) for s in range(CONTROL_SEEDS)]
    loops = [closed_loop(X, y, day, s) for s in SEEDS]

    def d2date(d):
        yy, mm, dd = days[d].split("-")
        return f"20{yy}-{mm}-{dd}"

    res = {
        "source": "Wörner, Eimler and Pein-Hackelbusch, Sci. Data 12:1628 (2025); "
                  "Zenodo 10.5281/zenodo.15681119; extracted_features.csv",
        "n_samples": int(len(y)), "n_features": int(X.shape[1]),
        "n_sensors": 62, "n_classes": len(classes), "classes": classes,
        "n_days": n_days, "first_date": d2date(0), "last_date": d2date(n_days - 1),
        "span_days": int((np.datetime64(d2date(n_days - 1)) - np.datetime64(d2date(0)))
                         .astype(int)),
        "install_days": INSTALL_DAYS, "install_samples": int((day < INSTALL_DAYS).sum()),
        "n_cal": n_cal, "alpha": ALPHA, "delta": DELTA, "budget": BUDGET,
        "W": CONTROL_W, "k_crit": bb_critical(CONTROL_W, n_cal, ALPHA, DELTA),
        "eval_days": eval_days, "seeds": len(SEEDS),
        "install_acc_mean": float(np.mean([st["install_acc"] for st in stats])),
        "pmax_cal_mean": float(np.mean([st["pmax_cal"] for st in stats])),
        "pmax_post_mean": float(np.mean([st["pmax_post"] for st in stats])),
        "static_acc_mean": float(np.mean(static_acc)),
        "static_acc_per_seed": static_acc,
        "alarm_days_per_seed": alarm_days,
        "alarm_days_median": float(np.median(alarm_days)),
        "alarm_days_min": int(min(alarm_days)), "alarm_days_max": int(max(alarm_days)),
        "first_alarm_day_per_seed": first,
        "first_alarm_day_median": float(np.median([f for f in first if f is not None]))
        if any(f is not None for f in first) else None,
        "per_day_seed0": dets[0],
        "ctl_alarms": int(sum(c[0] for c in ctl)),
        "ctl_windows": int(sum(c[1] for c in ctl)),
        "ctl_streams": CONTROL_SEEDS,
        "ctl_union_bound": float(DELTA * CONTROL_DAYS),
        "fm": {kind: {"alarm_days_median": float(np.median([r[0] for r in v[:len(SEEDS)]])),
                      "alarm_days_min": int(min(r[0] for r in v[:len(SEEDS)])),
                      "alarm_days_max": int(max(r[0] for r in v[:len(SEEDS)])),
                      "first_alarm_day_median": float(np.median([r[1] for r in v[:len(SEEDS)] if r[1]]))
                      if any(r[1] for r in v[:len(SEEDS)]) else None,
                      "ctl_alarms": int(sum(r[2] for r in v)),
                      "ctl_windows": CONTROL_SEEDS * CONTROL_DAYS}
               for kind, v in fm.items()},
        "loop_acc_mean": float(np.mean([np.mean(a) for a, _, _ in loops])),
        "loop_acc_per_seed": [float(np.mean(a)) for a, _, _ in loops],
        "loop_labels_mean": float(np.mean([l for _, l, _ in loops])),
        "loop_alarms_mean": float(np.mean([a for _, _, a in loops])),
        "loop_acc_per_day_seed0": loops[0][0],
    }
    with open(os.path.join(ROOT, "results", "enose_results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("per_day_seed0", "loop_acc_per_day_seed0")},
                     indent=1))
    print("day-by-day, seed 0 (day, W, k, empties, alarm, frozen acc):")
    for r in dets[0]:
        print(f"  {d2date(r['day'])} W={r['W']:2d} k={r['k']:2d} "
              f"empties={r['empties']:2d} alarm={int(r['alarm'])} acc={r['acc']:.2f}")


if __name__ == "__main__":
    main()
