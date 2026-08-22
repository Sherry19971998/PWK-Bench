#!/usr/bin/env python3
"""Build a lightweight, redistributable `data/sphere/` from a full SPHERE download.

>>> SYNTHETIC DATA <<<
SPHERE (Stanford ADRC SPHERE release) is a FULLY SYNTHETIC cohort generated from
the real ADRC cohort, intended by its authors for methods development, pilot
analysis, grant applications and teaching -- NOT for clinical findings. Subject
ids are literally `Synthetic537`, `Synthetic236`, ... Nothing computed from it
supports any claim about Alzheimer's disease; it is used here only to show that
the benchmark's machinery TRANSFERS to a second, heterogeneous-cost domain.

WHAT THIS SCRIPT DOES (and why)
-------------------------------
The raw download is ~1.2 GB, dominated by two files that the acquisition-curve
experiment does not need in full:

  * `scrna.csv` (1.1 GB) -- dropped entirely. It is not one of the analysed
    channels.
  * `proteomics_plasma.csv` (48 MB, ~6900 cols) and `proteomics_csf.csv`
    (13 MB, ~5284 cols) -- truncated to the top-`--n-proteins` columns BY
    VARIANCE, computed over all rows of that file.

The truncation is recorded here and in the loader docstring because it is part
of the reproducibility contract: a different column count gives different
numbers. Variance ranking is label-free -- it never looks at
`diagnosis_consensus` -- so it cannot leak the outcome into the retained
features. (A supervised filter, e.g. top-t-statistic, WOULD leak, which is why
variance and not a label-aware criterion is used.)

USAGE
-----
    python3 scripts/sphere/prep_sphere.py \
        --src /path/to/ADRC_SPHERE_all_data \
        --out data/sphere
"""

import os as _os, sys as _sys
# Make the package importable when the script is run directly from a
# checkout (scripts/<block>/x.py -> repo root is three dirnames up).
# Without this the script only works under `PYTHONPATH=.` or an
# installed package, which silently looks like the layout is broken.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(
    _os.path.dirname(_os.path.abspath(__file__)))))

import argparse, os, shutil
import pandas as pd

# Files copied through untouched -- all are small.
PASSTHROUGH = ["demographics_diagnosis.csv", "cognitive_scores.csv",
               "biomarkers.csv", "wgs.csv",
               "imaging_amyloid.csv", "imaging_tau.csv"]
# Wide files truncated to the top-variance columns.
TRUNCATE = ["proteomics_plasma.csv", "proteomics_csf.csv"]
# Never shipped.
SKIP = ["scrna.csv"]

ID = "adrc_id"


def truncate_by_variance(path: str, n_keep: int) -> pd.DataFrame:
    """Keep `adrc_id` + the n_keep highest-variance numeric columns.

    Variance is computed on the file's own rows with no reference to any label,
    so this is a label-free dimensionality cut, not feature selection.
    """
    df = pd.read_csv(path)
    if ID not in df.columns:
        raise SystemExit(f"{path}: expected an '{ID}' column, got {df.columns[:5].tolist()}")
    feats = df.drop(columns=[ID]).select_dtypes("number")
    if feats.shape[1] <= n_keep:
        keep = list(feats.columns)
    else:
        keep = feats.var(numeric_only=True).nlargest(n_keep).index.tolist()
    return df[[ID] + keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="directory holding the 9 SPHERE CSVs")
    ap.add_argument("--out", default="data/sphere")
    ap.add_argument("--n-proteins", type=int, default=500,
                    help="columns retained per proteomics file (default 500)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for f in PASSTHROUGH:
        s = os.path.join(args.src, f)
        if not os.path.exists(s):
            raise SystemExit(f"missing required file: {s}")
        shutil.copy2(s, os.path.join(args.out, f))
        print(f"  copied     {f:32s} {os.path.getsize(s)/1e6:8.2f} MB")

    for f in TRUNCATE:
        s = os.path.join(args.src, f)
        if not os.path.exists(s):
            raise SystemExit(f"missing required file: {s}")
        d = truncate_by_variance(s, args.n_proteins)
        dst = os.path.join(args.out, f)
        d.to_csv(dst, index=False)
        print(f"  truncated  {f:32s} {os.path.getsize(s)/1e6:8.2f} MB -> "
              f"{os.path.getsize(dst)/1e6:.2f} MB ({d.shape[1]-1} cols)")

    for f in SKIP:
        print(f"  SKIPPED    {f:32s} (not an analysed channel)")

    total = sum(os.path.getsize(os.path.join(args.out, f))
                for f in os.listdir(args.out))
    print(f"\n{args.out}: {total/1e6:.1f} MB total")
    print("NOTE: SYNTHETIC cohort -- methods-development use only, no clinical claim.")


if __name__ == "__main__":
    main()
