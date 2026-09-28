"""One-shot reproduction: download the data, run every experiment, and build
the figures.

Usage:  python reproduce.py               (about ten minutes on one laptop core)
        python reproduce.py --with-enose  also runs the second-dataset check
                                          (downloads a 141 MB archive)

Linear algebra is forced to a single thread before NumPy is imported, which
makes every result bitwise reproducible across runs on the same platform.
The SHA-256 of the UCI download is checked against the value used for all
reported results and recorded in data/SHA256SUM.
"""

import os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_var] = "1"

import hashlib, io, subprocess, sys, time, urllib.request, zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
URL = ("https://archive.ics.uci.edu/static/public/224/"
       "gas+sensor+array+drift+dataset.zip")

# SHA-256 of the UCI archive used for all reported results; a mismatching
# download aborts the run. Set to None to accept any download while still
# recording its digest.
EXPECTED_SHA256 = ("91e8f466f202e7a093d657673ce47311c3"
                   "e90416f7df3057966058961c351fe4")


def ensure_data():
    target = os.path.join(DATA, "Dataset", "batch10.dat")
    if os.path.exists(target):
        print("dataset already present")
        return
    os.makedirs(DATA, exist_ok=True)
    print("downloading UCI Gas Sensor Array Drift dataset (about 10 MB) ...")
    with urllib.request.urlopen(URL) as r:
        raw = r.read()
    digest = hashlib.sha256(raw).hexdigest()
    print(f"sha256: {digest}")
    with open(os.path.join(DATA, "SHA256SUM"), "w") as f:
        f.write(f"{digest}  gas+sensor+array+drift+dataset.zip\n")
    if EXPECTED_SHA256 is not None and digest != EXPECTED_SHA256:
        sys.exit(f"dataset hash mismatch: expected {EXPECTED_SHA256}, "
                 f"got {digest}; refusing to run on unverified data")
    zipfile.ZipFile(io.BytesIO(raw)).extractall(DATA)
    print("done")


def run(*script_and_args):
    subprocess.check_call([sys.executable, *script_and_args], env=os.environ)


def main():
    t0 = time.time()
    ensure_data()
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    sys.path.insert(0, os.path.join(ROOT, "src"))
    import run_experiments
    run_experiments.main(os.path.join(DATA, "Dataset"),
                         os.path.join(ROOT, "results", "results.json"))
    run(os.path.join(ROOT, "src", "extended_experiments.py"))
    run(os.path.join(ROOT, "src", "seed_accs.py"))
    if "--with-enose" in sys.argv:
        run(os.path.join(ROOT, "src", "second_dataset.py"))
    run(os.path.join(ROOT, "gen_numbers.py"),
        os.path.join(ROOT, "results"), os.path.join(ROOT, "numbers.tex"))
    run(os.path.join(ROOT, "src", "make_figures.py"),
        os.path.join(ROOT, "results"), os.path.join(ROOT, "figures"))
    run(os.path.join(ROOT, "src", "make_ga.py"),
        os.path.join(ROOT, "results", "results.json"),
        os.path.join(ROOT, "figures"))
    print(f"all done in {(time.time()-t0)/60:.0f} min: results JSONs, "
          f"numbers.tex, figures, and graphical abstract")


if __name__ == "__main__":
    main()
