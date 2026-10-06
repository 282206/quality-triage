"""Phase 6 - inter-rater agreement between two human labellers.

On the images labelled by BOTH raters we report, separately for sharpness and
exposure:
  * percent agreement           - share of identical labels
  * Cohen's kappa (unweighted)  - agreement corrected for chance
  * Cohen's kappa (quadratic)   - also gives partial credit for near misses
                                  (good vs borderline is a smaller error than
                                  good vs bad), suitable for ordinal labels.
Rough reading (Landis & Koch): <0.2 slight, 0.2-0.4 fair, 0.4-0.6 moderate,
0.6-0.8 substantial, >0.8 almost perfect.

Run:  python -m src.agreement      -> results/agreement.json
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.common import CLASSES, RATER2_LABELS, REAL_LABELS, RESULTS_DIR


def interpret(k: float) -> str:
    for lim, name in ((0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k < lim:
            return name
    return "almost perfect"


def agreement(r1: pd.DataFrame, r2: pd.DataFrame) -> dict:
    m = r1.merge(r2, on="filename", suffixes=("_r1", "_r2"))
    out = {"n_overlap": int(len(m))}
    for col in ("sharpness", "exposure"):
        sub = m.dropna(subset=[f"{col}_r1", f"{col}_r2"])
        a, b = sub[f"{col}_r1"], sub[f"{col}_r2"]
        if len(sub) == 0:
            out[col] = {"n": 0}
            continue
        k = cohen_kappa_score(a, b, labels=CLASSES)
        kq = cohen_kappa_score(a, b, labels=CLASSES, weights="quadratic")
        out[col] = {"n": int(len(sub)), "percent_agreement": round(float((a == b).mean() * 100), 1),
                    "kappa": round(float(k), 3), "kappa_quadratic": round(float(kq), 3),
                    "interpretation": interpret(kq),
                    "confusion (rows=rater1, cols=rater2)":
                        pd.crosstab(a, b).reindex(index=CLASSES, columns=CLASSES, fill_value=0).values.tolist()}
    return out


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    if not (REAL_LABELS.exists() and RATER2_LABELS.exists()):
        print(f"Need both {REAL_LABELS.name} and {RATER2_LABELS.name} - skipping agreement.")
        return None
    res = agreement(pd.read_csv(REAL_LABELS), pd.read_csv(RATER2_LABELS))
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "agreement.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
