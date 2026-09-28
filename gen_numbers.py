"""Generate numbers.tex, the LaTeX macros for every reported number, from the
experiment-output JSON files.

Usage: python gen_numbers.py [results_dir] [output.tex]
"""
import json
import os
import sys

import numpy as np
from scipy import stats
from scipy.special import betaln, gammaln

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "numbers.tex")
R = json.load(open(os.path.join(BASE, "extended_results.json")))
M = json.load(open(os.path.join(BASE, "results.json")))
S = json.load(open(os.path.join(BASE, "seed_accs.json")))
E_PATH = os.path.join(BASE, "enose_results.json")
E = json.load(open(E_PATH)) if os.path.exists(E_PATH) else None


def pm(d, nd=3):
    a = np.array(d["acc_per_seed"])
    return f"${a.mean():.{nd}f}\\pm{a.std():.{nd}f}$"


def pmc(d, nd=3):
    return f"${d['mean_acc']:.{nd}f}\\pm{d['std_acc']:.{nd}f}$"


def short(d, nd=3):
    return f"{np.mean(d['acc_per_seed']):.{nd}f}"


def pval(a, b):
    p = stats.wilcoxon(np.array(a), np.array(b), method="exact").pvalue
    if p >= 0.10:
        return f"{p:.2f}"
    return f"{p:.3f}" if p >= 0.001 else "<0.001"


def thousands(x):
    return f"{x:,.0f}"


def sci(x):
    """LaTeX scientific notation with three significant digits, e.g. 2.85\\times10^{-4}."""
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    return f"{m:.2f}\\times10^{{{e}}}"


def sci_mantissa(x):
    e = int(np.floor(np.log10(x)))
    return round(x / 10 ** e, 2) * 10 ** e


def words(n):
    return ["zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten"][n] if 0 <= n <= 10 else str(n)


# consistency guards: seed_accs must agree with results.json
assert abs(np.mean(S["cadence-10"]) - M["cadence-10"]["mean_acc"]) < 5e-3
assert abs(np.mean(S["oracle"]) - M["oracle"]["mean_acc"]) < 5e-3
assert abs(np.mean(S["periodic-10"]) - M["periodic-10"]["mean_acc"]) < 5e-3

cad = S["cadence-10"]
c10, c20 = M["cadence-10"], M["cadence-20"]
post_alarms = R["post-adapt-1"]["alarm_windows"] + R["post-adapt-3"]["alarm_windows"]
post_windows = R["post-adapt-1"]["windows"] + R["post-adapt-3"]["windows"]
nc = M["naive_control"]
ctl = M["control"]
det = M["detect"]
fa = np.array(det["first_alarm"], dtype=float)
med = np.nanmedian(fa, axis=1)
frozen = np.array(det["frozen_acc"]).mean(axis=1)
least_degraded = np.argsort(-frozen)[:2] + 2          # batch numbers
rates = [sum(c) / (len(c) * ctl["W"]) for c in det["counts_seed0"]]
runs = M["label_runs"]
grouped = [b + 1 for b in range(1, 10) if runs[b] < 15]   # streamed batches with long runs
spe_delays = [d for seed in R["fm-spe"]["delays"] for d in seed]
spe_one_window = sum(1 for d in spe_delays if d == ctl["W"])
mlp_ctl = R["mlp-control"]
n_cal = M["split"]["n_cal"]
clf = M["classifier"]

m = {
    # significance
    "pFullRelabel": pval(cad, S["oracle"]),
    "pPrivTen": pval(cad, S["periodic-10"]),
    "pRecency": pval(cad, R["relabel-recency"]["acc_per_seed"]),
    # data and split
    "totalSamples": thousands(sum(M["batch_sizes"])),
    "batchOneSize": str(M["batch_sizes"][0]),
    "nCal": str(n_cal),
    "nTrain": str(M["split"]["n_train"]),
    "nStream": str(M["split"]["n_stream"]),
    "nParams": str(clf["n_params"]),
    "nMacs": str(clf["n_features"] * clf["n_classes"]),
    "kBytes": f"{clf['n_params'] * 4 / 1000:.1f}",
    "groupedBatchFirst": str(grouped[0]),
    "groupedBatchLast": str(grouped[-1]),
    "groupedBatchCount": words(len(grouped)),
    "groupedRunsMax": str(max(runs[b - 1] for b in grouped)),
    # certificate arithmetic
    "kCrit": str(det["k_crit"]),
    "kNaive": str(nc["k_naive"]),
    "wAlpha": f"{ctl['W'] * 0.1:.0f}",
    "betaBinMean": f"{nc['betabinom_mean']:.2f}",
    "naiveTail": sci(nc["naive_believed_level"]),
    "bbTail": sci(nc["naive_true_level"]),
    "tailRatio": f"{sci_mantissa(nc['naive_true_level']) / sci_mantissa(nc['naive_believed_level']):.1f}",
    "naiveStreamsAlarm": str(nc["streams_with_alarm"]),
    "naiveStreams": str(nc["n_streams"]),
    # detection
    "rateMin": f"{min(rates):.2f}",
    "rateMax": f"{max(rates):.2f}",
    "oneWindowBatches": words(int((med == ctl["W"]).sum())),
    "maxMedianDelay": f"{np.nanmax(med):.0f}",
    "maxMedianDelayBatch": str(int(np.nanargmax(med)) + 2),
    "leastDegradedA": str(min(least_degraded)),
    "leastDegradedB": str(max(least_degraded)),
    "controlStreams": str(ctl["n_streams"]),
    "controlWindows": thousands(ctl["total_windows"]),
    "controlAlarms": str(ctl["alarm_windows"]),
    "controlT": thousands(ctl["T"]),
    "unionBound": f"{ctl['delta'] * ctl['T'] / ctl['W']:.2f}",
    # static, full relabeling, privileged
    "staticAcc": pmc(M["static"]),
    "staticShort": f"{M['static']['mean_acc']:.3f}",
    "fullRelabelAcc": pmc(M["oracle"]),
    "fullRelabelShort": f"{M['oracle']['mean_acc']:.3f}",
    "fullRelabelLabels": thousands(M["oracle"]["labels_mean"]),
    "privTenAcc": pmc(M["periodic-10"]),
    "privTenShort": f"{M['periodic-10']['mean_acc']:.3f}",
    "privTenLabels": f"{M['periodic-10']['labels_mean']:.0f}",
    "privFortyAcc": pmc(M["periodic-40"]),
    "privFortyLabels": f"{M['periodic-40']['labels_mean']:.0f}",
    # central closed-loop numbers
    "cadTenAcc": pmc(c10),
    "cadTenAccShort": f"{c10['mean_acc']:.3f}",
    "cadTenLabels": f"{c10['labels_mean']:.0f}",
    "cadTenAlarms": f"{c10['alarms_mean']:.0f}",
    "cadTwentyAcc": pmc(c20),
    "cadTwentyAccShort": f"{c20['mean_acc']:.3f}",
    "cadTwentyLabels": f"{c20['labels_mean']:.0f}",
    # derived comparisons
    "fullRelabelGapPts": f"{(round(c10['mean_acc'], 3)-round(M['oracle']['mean_acc'], 3))*100:.1f}",
    "privGapPts": f"{(round(M['periodic-40']['mean_acc'], 3)-round(c10['mean_acc'], 3))*100:.1f}",
    "labelRatio": f"{M['oracle']['labels_mean']/c10['labels_mean']:.0f}",
    "speRatio": f"{R['fm-spe']['loop']['labels']/c10['labels_mean']:.1f}",
    # ablations
    "deltaLooseAcc": f"{M['delta-0.01']['mean_acc']:.3f}",
    "deltaLooseLabels": f"{M['delta-0.01']['labels_mean']:.0f}",
    "deltaTightAcc": f"{M['delta-0.0001']['mean_acc']:.3f}",
    "deltaTightLabels": f"{M['delta-0.0001']['labels_mean']:.0f}",
    "wSmallAcc": f"{M['W-25']['mean_acc']:.3f}",
    "wSmallLabels": f"{M['W-25']['labels_mean']:.0f}",
    "wLargeAcc": f"{M['W-100']['mean_acc']:.3f}",
    "wLargeLabels": f"{M['W-100']['labels_mean']:.0f}",
    "rankedAcc": f"{M['cadence-20-ranked']['mean_acc']:.3f}",
    # per-batch comparison for the accuracy-figure caption
    "cadWinsBatches": str(sum(1 for a, b in zip(M["cadence-10"]["per_batch_mean"],
                                                M["oracle"]["per_batch_mean"])
                              if a > b)),
    # relabeling baselines
    "relRecencyAcc": pm(R["relabel-recency"]),
    "relWindowAcc": pm(R["relabel-window"]),
    "relRecencyShort": short(R["relabel-recency"]),
    "relWindowShort": short(R["relabel-window"]),
    # feature monitors
    "tTwoFired": f"{R['fm-t2']['fired_batches_mean']:.1f}",
    "speFired": f"{R['fm-spe']['fired_batches_mean']:.1f}",
    "speCtl": str(R["fm-spe"]["ctl_alarm_windows_total"]),
    "tTwoCtl": str(R["fm-t2"]["ctl_alarm_windows_total"]),
    "fmCtlWindows": str(R["fm-spe"]["ctl_windows"]),
    "speOneWindowPairs": str(spe_one_window),
    "speDelayPairs": str(len(spe_delays)),
    "tTwoAcc": pm(R["fm-t2"]["loop"]),
    "tTwoAccShort": short(R["fm-t2"]["loop"]),
    "speAcc": pm(R["fm-spe"]["loop"]),
    "tTwoLabels": f"{R['fm-t2']['loop']['labels']:.0f}",
    "speLabels": thousands(R["fm-spe"]["loop"]["labels"]),
    "pcaVar": "95\\%",
    # MLP
    "mlpFired": f"{R['mlp-detect']['fired_batches_mean']:.1f}",
    "mlpCtlAlarms": str(mlp_ctl["alarms"]),
    "mlpCtlWindows": str(mlp_ctl["windows"]),
    "mlpCtlStreamsAlarm": words(mlp_ctl["streams_with_alarm"]),
    "mlpCtlStreams": words(mlp_ctl["streams"]),
    "mlpAcc": pm(R["mlp-cadence-10"]),
    "mlpAccShort": short(R["mlp-cadence-10"]),
    "mlpLabels": f"{R['mlp-cadence-10']['labels']:.0f}",
    # sensitivity
    "nCalSmall": str(R["sensitivity"]["fcal0.15-pmax"]["n_cal"]),
    "nTrainSmall": str(M["batch_sizes"][0] - M["split"]["n_stream"]
                       - R["sensitivity"]["fcal0.15-pmax"]["n_cal"]),
    "kCritSmall": str(R["sensitivity"]["fcal0.15-pmax"]["k_crit"]),
    "nCalLarge": str(R["sensitivity"]["fcal0.45-pmax"]["n_cal"]),
    "nTrainLarge": str(M["batch_sizes"][0] - M["split"]["n_stream"]
                       - R["sensitivity"]["fcal0.45-pmax"]["n_cal"]),
    "sensSmallFired": f"{R['sensitivity']['fcal0.15-pmax']['fired_mean']:.1f}",
    "sensSmallDelay": f"{R['sensitivity']['fcal0.15-pmax']['median_delay_mean']:.0f}",
    "sensDefaultDelay": f"{R['sensitivity']['fcal0.3-pmax']['median_delay_mean']:.0f}",
    # interleaved
    "intRandomAcc": short(R["interleaved-B20-random"]),
    "intRankedAcc": short(R["interleaved-B20-ranked"]),
    "intRandomLabels": f"{R['interleaved-B20-random']['labels']:.0f}",
    "intRankedLabels": f"{R['interleaved-B20-ranked']['labels']:.0f}",
    # post-adaptation
    "postAdaptAlarms": str(post_alarms),
    "postAdaptWindows": thousands(post_windows),
    "postAdaptOne": thousands(R["post-adapt-1"]["windows"]),
    "postAdaptThree": str(R["post-adapt-3"]["windows"]),
    "postAdaptStreams": str(R["post-adapt-1"]["streams"]),
    "postAdaptExpected": f"{1e-3*post_windows:.1f}",
}

if E is not None:
    fm_t2 = E["fm"]["t2"]
    m.update({
        "enoseSamples": str(E["n_samples"]),
        "enoseSensors": str(E["n_sensors"]),
        "enoseDays": str(E["n_days"]),
        "enoseClasses": words(E["n_classes"]),
        "enoseInstallDays": words(E["install_days"]),
        "enoseNCal": str(E["n_cal"]),
        "enoseW": str(E["W"]),
        "enoseKCrit": str(E["k_crit"]),
        "enoseEvalDays": str(E["eval_days"]),
        "enoseCtlAlarms": str(E["ctl_alarms"]),
        "enoseCtlWindows": thousands(E["ctl_windows"]),
        "enoseInstallAcc": f"{E['install_acc_mean']:.2f}",
        "enoseStaticAcc": f"{E['static_acc_mean']:.2f}",
        "enosePmaxCal": f"{E['pmax_cal_mean']:.2f}",
        "enosePmaxPost": f"{E['pmax_post_mean']:.2f}",
        "enoseSeeds": words(E["seeds"]),
        "enoseTtwoAlarmDays": f"{fm_t2['alarm_days_median']:g}",
        "enoseTtwoCtl": str(fm_t2["ctl_alarms"]),
    })

with open(OUT, "w", newline="\n") as fh:
    for k, v in m.items():
        assert k.isalpha(), k
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(json.dumps(m, indent=1))
