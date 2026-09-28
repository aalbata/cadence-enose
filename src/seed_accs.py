"""Per-seed mean accuracies for the paired significance tests.
Writes seed_accs.json."""
import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import json
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cadence import load_batches, run_cadence, run_oracle, run_periodic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "Dataset")


def main():
    batches = load_batches(DATA)
    out = {}
    for name, fn in [
        ("cadence-10", lambda s: run_cadence(batches, s, budget=10)),
        ("periodic-10", lambda s: run_periodic(batches, s, budget=10)),
        ("oracle", lambda s: run_oracle(batches, s)),
    ]:
        accs = [float(np.mean(fn(s).batch_acc)) for s in range(10)]
        out[name] = accs
        print(name, [round(a, 4) for a in accs], flush=True)
    with open(os.path.join(ROOT, "results", "seed_accs.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
