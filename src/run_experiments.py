"""Primary experiment suite. Writes results.json."""

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import json, sys, time
import numpy as np
from cadence import (load_batches, split_batch1, make_model, run_static,
                     run_periodic, run_cadence, run_oracle, run_detect_only,
                     run_stationary_control, run_naive_control)

SEEDS = list(range(10))


def summarize(runs):
    acc = np.array([r.batch_acc for r in runs])
    return {
        "per_batch_mean": acc.mean(axis=0).tolist(),
        "per_batch_std": acc.std(axis=0).tolist(),
        "mean_acc": float(acc.mean(axis=1).mean()),
        "std_acc": float(acc.mean(axis=1).std()),
        "labels_mean": float(np.mean([r.labels_used for r in runs])),
        "labels_std": float(np.std([r.labels_used for r in runs])),
        "alarms_mean": float(np.mean([len(r.alarms) for r in runs])),
    }


def label_runs(y):
    """Number of maximal runs of consecutive identical labels."""
    return int(1 + np.sum(y[1:] != y[:-1])) if len(y) else 0


def main(data_dir, out_path):
    batches = load_batches(data_dir)
    t0 = time.time()
    results = {"batch_sizes": [len(b[1]) for b in batches],
               "label_runs": [label_runs(b[1]) for b in batches]}

    # split sizes and classifier size (identical for every seed)
    _, _, Xtr, ytr, Xcal, ycal, Xstr, _ = split_batch1(batches, 0)
    model = make_model(0).fit(Xtr, ytr)
    results["split"] = {"n_train": len(ytr), "n_cal": len(ycal),
                        "n_stream": len(Xstr)}
    results["classifier"] = {"n_features": int(model.coef_.shape[1]),
                             "n_classes": int(model.coef_.shape[0]),
                             "n_params": int(model.coef_.size + model.intercept_.size)}

    configs = [
        ("static", lambda s: run_static(batches, s)),
        ("periodic-10", lambda s: run_periodic(batches, s, budget=10)),
        ("periodic-20", lambda s: run_periodic(batches, s, budget=20)),
        ("periodic-40", lambda s: run_periodic(batches, s, budget=40)),
        ("cadence-10", lambda s: run_cadence(batches, s, budget=10)),
        ("cadence-20", lambda s: run_cadence(batches, s, budget=20)),
        ("cadence-20-ranked", lambda s: run_cadence(batches, s, budget=20,
                                                    ranked=True)),
        ("oracle", lambda s: run_oracle(batches, s)),
    ]
    for name, fn in configs:
        results[name] = summarize([fn(s) for s in SEEDS])
        r = results[name]
        print(f"{name:18s} acc={r['mean_acc']:.3f}+-{r['std_acc']:.3f} "
              f"labels={r['labels_mean']:.0f} alarms={r['alarms_mean']:.1f}",
              flush=True)

    # delta and window ablations for the closed loop (B=10)
    for delta in [1e-2, 1e-3, 1e-4]:
        rs = [run_cadence(batches, s, budget=10, delta=delta) for s in SEEDS]
        results[f"delta-{delta}"] = summarize(rs)
        print(f"delta={delta}: acc={results[f'delta-{delta}']['mean_acc']:.3f} "
              f"labels={results[f'delta-{delta}']['labels_mean']:.0f}", flush=True)
    for W in [25, 100]:
        rs = [run_cadence(batches, s, budget=10, W=W) for s in SEEDS]
        results[f"W-{W}"] = summarize(rs)
        print(f"W={W}: acc={results[f'W-{W}']['mean_acc']:.3f} "
              f"labels={results[f'W-{W}']['labels_mean']:.0f}", flush=True)

    # detection with frozen model, all seeds
    det = [run_detect_only(batches, s) for s in SEEDS]
    results["detect"] = {
        "k_crit": det[0]["k_crit"],
        "first_alarm": [[d["batches"][b]["first_alarm"] for d in det]
                        for b in range(9)],
        "frozen_acc": [[d["batches"][b]["acc"] for d in det]
                       for b in range(9)],
        "counts_seed0": [d0["counts"] for d0 in det[0]["batches"]],
        "empty_rate_all_seed0": [d0["empty_rate_all"] for d0 in det[0]["batches"]],
    }
    fa = np.array(results["detect"]["first_alarm"], dtype=float)
    med = np.nanmedian(fa, axis=1)
    print("detection first-alarm (median over seeds):", med.tolist(), flush=True)
    print(f"  median = 50 samples in {int((med == 50).sum())} of 9 batches; "
          f"max median = {int(np.nanmax(med))} samples", flush=True)

    # stationarity control against the certified threshold, 50 independent seeds
    ctl = [run_stationary_control(batches, s) for s in range(50)]
    results["control"] = {"alarm_windows": int(sum(c[0] for c in ctl)),
                          "total_windows": int(sum(c[1] for c in ctl)),
                          "streams_with_alarm": int(sum(c[0] > 0 for c in ctl)),
                          "n_streams": 50, "T": 2000, "W": 50, "delta": 1e-3}
    print(f"control: {results['control']}", flush=True)

    # the same control against the naive binomial threshold, shared draw
    results["naive_control"] = run_naive_control(batches)
    print(f"naive control: {results['naive_control']}", flush=True)

    # one-seed closed-loop trace (kept for inspection; not used by figures)
    tr = run_cadence(batches, 0, budget=10)
    results["trace"] = {"alarms": tr.alarms, "batch_acc": tr.batch_acc,
                        "labels": tr.labels_used}
    results["runtime_seconds"] = round(time.time() - t0)

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"done in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
