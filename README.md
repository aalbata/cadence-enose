# CADENCE: label-free drift alarms with certified false-alarm control

Code and results for the letter "Certified Label-Free Drift Alarms From
Empty Conformal Prediction Sets for Metal-Oxide Gas Sensor Arrays"
(accepted for publication in IEEE Sensors Letters). Repository: https://github.com/aalbata/cadence-enose

A gas classifier can lose accuracy under drift while remaining confident.
CADENCE monitors the number of empty conformal prediction sets over
disjoint windows of 50 samples. Under i.i.d. stationarity that count is
stochastically dominated by a beta-binomial law that accounts for the
finite calibration set, so the alarm threshold carries a finite-sample
per-window false-alarm bound for the pre-adaptation epoch; after the
loop retrains, the bound no longer applies and the false-alarm rate is
reported empirically. Cost per sample: one subtraction, one comparison,
a conditional counter increment, and a window-position count. When the
alarm fires, the evaluated loop requests a small budget of reference
labels, retrains, and resumes monitoring.

Headline numbers on the 36-month UCI Gas Sensor Array Drift benchmark
(mean over 10 seeds, batches 2 to 10):

| method | boundary info | labels | accuracy |
|---|---|---|---|
| static | no | 0 | 0.520 |
| full relabeling | yes | 13,465 | 0.816 |
| privileged, B=40 | yes | 360 | 0.958 |
| CADENCE, B=10 | no | 337 | 0.872 |

Column definitions: "boundary info" says whether the method is told when a
new batch begins; "labels" is the number of reference labels consumed over
the nine streamed batches (mean over seeds); "accuracy" is the mean over
batches 2 to 10 of the per-batch accuracy, prequential for CADENCE.

Detection (frozen batch-1 model): alarms in all 9 drifted batches for all
10 seeds; median first-alarm delay is 50 samples (one window) in four of
the nine batches and at most 600 samples in the rest. Stationarity
control: 0 alarms in 2,000 windows across 50 independent streams (union
bound 0.04 per stream). The naive binomial threshold (k=14 instead of the
certified k=15) has a believed per-window level of 2.85e-4 but a
worst-case level of 1.81e-3 under the beta-binomial law, 6.4 times the
believed level; `results.json["naive_control"]`
holds the corresponding shared-calibration empirical control (false
alarms in 3 of 20 streams).

Additional check, not part of the letter (described in the response to
reviewers): `src/second_dataset.py` applies the same alarm, under a
protocol fixed in advance, to the one-year e-nose dataset of Wörner,
Eimler and Pein-Hackelbusch (Scientific Data 12, 1628, 2025; Zenodo
10.5281/zenodo.15681119, CC BY 4.0), using the authors' feature table.
With day-aligned windows the frozen classifier's accuracy falls from
0.90 to 0.50 while the empty-set alarm never fires (0 of 30 days in all
10 seeds), because the mean top-class probability does not fall (0.88 at
calibration, 0.92 afterwards); the stationarity control gives 2 alarms
in 5,000 synthetic day windows, and Hotelling T^2 on the same day windows
fires on a median of 20.5 of 30 days with 4 control alarms. Its output is
`results/enose_results.json`.

## Reproduce

Python 3.11 or newer (the pinned numpy requires it); the results were
produced with Python 3.12.10 on Windows 11 (Intel Core Ultra 7 155U).

```
pip install -r requirements.txt
python reproduce.py
python reproduce.py --with-enose
```

`reproduce.py` downloads the UCI dataset (about 10 MB; SHA-256 checked
against the value used for the reported results), regenerates
`results/results.json`, `results/extended_results.json` and
`results/seed_accs.json`, writes `numbers.tex` (the LaTeX macros behind
every number in the letter) and rebuilds the two figures and the
graphical abstract in `figures/`. The whole run takes about ten minutes
on one core. The `--with-enose` option also downloads the 141 MB
second dataset and writes `results/enose_results.json`.

All linear algebra is forced to a single thread (`OMP_NUM_THREADS=1` and
its OpenBLAS and MKL equivalents are set before NumPy is imported). This
makes every numerical result bitwise reproducible on the same platform; with a
multi-threaded BLAS the larger logistic-regression fits follow a different
floating-point path and their third decimals can differ.

## Layout

- `src/cadence.py` core library: conformal machinery, beta-binomial
  critical values, the `Monitor` and `WindowPool` classes, and the
  primary experiment runners
- `src/run_experiments.py` primary experiment suite (writes `results.json`)
- `src/extended_experiments.py` second classifier family (a one-hidden-layer
  MLP with 32 ReLU units, Adam, L2 penalty 1e-3, up to 800 iterations,
  queries weighted 5x), relabeling baselines, feature-space monitors,
  sensitivity analyses, interleaved stream, and post-adaptation control
  (writes `extended_results.json`)
- `src/seed_accs.py` per-seed values used by the Wilcoxon tests
  (writes `seed_accs.json`)
- `src/second_dataset.py` the additional check on the one-year e-nose
  dataset (writes `enose_results.json`; not part of the letter)
- `src/make_figures.py` the two letter figures from `results/results.json`
- `src/make_ga.py` graphical abstract from `results/results.json`
- `gen_numbers.py` LaTeX macros (`numbers.tex`) from the results files
- `reproduce.py` one-shot entry point (download, run, plot)
- `results/` the shipped JSON results behind the letter
- `requirements.txt`, `CITATION.cff`, `LICENSE`

Key names in the results files: `oracle` is the full-relabeling baseline
(Table I, row 2); `periodic-B` is the boundary-privileged baseline with
B labels per batch boundary; `cadence-B` is the alarm-gated loop with B
labels per alarm; `fm-t2` and `fm-spe` are the Hotelling T^2 and PCA-SPE
monitors.

## Data

UCI Gas Sensor Array Drift dataset (Vergara et al., 2012): 13,910
measurements, 16 MOX sensors, 128 features, 6 gases, 10 chronological
batches over 36 months (batch 1 covers months 1 and 2; batches 2 to 10
cover months 3 to 36). Downloaded automatically; not redistributed here.
Licensed CC BY 4.0 by the UCI Machine Learning Repository (dataset DOI
10.24432/C5RP6W).

Direct inspection of the batch files shows strong gas grouping. Batches 2
to 6 contain 14, 8, 7, 5, and 11 contiguous class-label runs, whereas
batches 7 to 10 contain 1823, 145, 54 and 3600. This matters for the
ranked-selection ablation discussed in the letter. Dennler et al. (2022)
document a temporally clustered recording order in a related MOX
benchmark; they do not analyze this UCI dataset.

## Reproducibility notes

- All randomness is seeded and linear algebra is single-threaded, so the
  suite is deterministic given the pinned package versions in
  `requirements.txt`. If your environment differs, re-run `reproduce.py`
  and use the regenerated JSONs, macros, figures and graphical abstract
  together.
- Figure 1 shows the seed-0 trace of the frozen-model detection run; the
  first-alarm statistics in the letter aggregate all 10 seeds.
- The boundary-privileged baseline is scored on the non-queried remainder
  of each batch; the gated loop is scored prequentially on every sample.
- Alarm windows run across batch boundaries; the query pool for an alarm
  is the whole window that raised it, so the loop uses no batch-boundary
  information.
- `results.json["trace"]` is a single-seed closed-loop trace kept for
  inspection; no figure or table depends on it.

## License

MIT
