"""Phase 4 - classical baseline: threshold selection + simple classifiers.

SHARPNESS
    One scalar sharpness feature x (higher = sharper). Two thresholds t1 < t2:
        x < t1 -> bad,   t1 <= x < t2 -> borderline,   x >= t2 -> good
    (t1, t2) are chosen by GRID SEARCH over quantiles of x on the SYNTHETIC
    TRAIN split, maximising macro-F1 against the known labels.
    Two versions are fitted:
        raw  : x = log10(Laplacian variance)            (the classic method)
        norm : x = the best content-aware metric from Phase 3, chosen by
               train macro-F1 among masked_lap, masked_ten, reblur,
               reblur_masked, reblur_dir, fft_hf (chosen on TRAIN only)

EXPOSURE (same for both versions)
    A multinomial logistic regression on the luminance statistics
    (mean, |mean-0.5|, std, skew, p05, p95, clip_low, clip_high, 8-bin
    histogram) gives class
    probabilities; exposure score e = 0*P(bad) + 0.5*P(borderline) + 1*P(good).
    Two thresholds on e are then grid-searched exactly like sharpness.

SCORES AND OVERALL QUALITY (for ranking)
    Each feature is mapped to [0,1] by a piecewise-linear function through
        (train 1st percentile -> 0), (t1 -> 1/3), (t2 -> 2/3), (train 99th pct -> 1)
    so score < 1/3 means "bad" and > 2/3 means "good" for every metric.
    Overall quality  Q = sqrt(s_sharp * s_exposure)   (geometric mean: an
    image that is bad on EITHER axis gets a low Q).

Run:  python -m src.baseline
Out:  models/classical_raw.json, models/classical_norm.json,
      results/synthetic_baseline.csv
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from src.common import (CLASS_TO_IDX, CLASS_VALUE, CLASSES, MODELS_DIR, RESULTS_DIR,
                        load_image, seed_everything)
from src.features import (EXPOSURE_FEATURES, exposure_features, gray_from_bgr,
                          sharpness_feature)

NORM_CANDIDATES = ["masked_lap", "masked_ten", "reblur", "reblur_masked", "reblur_dir", "fft_hf"]
LOG_FEATURES = {"lap_var", "tenengrad", "masked_lap", "masked_ten"}


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def transform(name: str, x) -> np.ndarray:
    """Monotone transform so thresholds/scores live on a sensible scale."""
    x = np.asarray(x, dtype=float)
    return np.log10(x + 1e-9) if name in LOG_FEATURES else x


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int = 3) -> float:
    """Fast macro-F1 via a confusion matrix (classes absent from both are skipped)."""
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)               # = 2TP + FP + FN
    present = denom > 0
    return float((2 * tp[present] / denom[present]).mean())


def apply_thresholds(x: np.ndarray, t1: float, t2: float) -> np.ndarray:
    return np.where(x < t1, 0, np.where(x < t2, 1, 2))


def grid_search_thresholds(x: np.ndarray, y: np.ndarray, n_grid: int = 80) -> tuple[float, float, float]:
    """Exhaustive search over (t1 < t2) taken from quantiles of x."""
    cands = np.unique(np.quantile(x, np.linspace(0.005, 0.995, n_grid)))
    best = (-1.0, cands[0], cands[-1])
    for i, t1 in enumerate(cands):
        for t2 in cands[i:]:
            f = macro_f1(y, apply_thresholds(x, t1, t2))
            if f > best[0]:
                best = (f, float(t1), float(t2))
    return best


def score_from_thresholds(x: np.ndarray, th: dict) -> np.ndarray:
    """Piecewise-linear map to [0,1]: lo->0, t1->1/3, t2->2/3, hi->1 (clipped)."""
    eps = 1e-9
    t1, t2 = th["t1"], th["t2"]
    lo, hi = min(th["lo"], t1 - eps), max(th["hi"], t2 + eps)
    if t2 <= t1:
        t2 = t1 + eps
    return np.interp(x, [lo, t1, t2, hi], [0, 1 / 3, 2 / 3, 1])


def fit_sharp_feature(name: str, train: pd.DataFrame) -> dict:
    x = transform(name, train[name])
    y = train.sharp_label.map(CLASS_TO_IDX).to_numpy()
    f1, t1, t2 = grid_search_thresholds(x, y)
    return {"t1": t1, "t2": t2, "lo": float(np.percentile(x, 1)),
            "hi": float(np.percentile(x, 99)), "train_macro_f1": f1}


def exposure_design(df: pd.DataFrame) -> np.ndarray:
    X = df[EXPOSURE_FEATURES].to_numpy(float)
    return np.column_stack([X, np.abs(df.lum_mean.to_numpy() - 0.5)])


def fit_exposure(train: pd.DataFrame) -> dict:
    from sklearn.linear_model import LogisticRegression

    X = exposure_design(train)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    y = train.exposure_label.map(CLASS_TO_IDX).to_numpy()
    lr = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    lr.fit((X - mu) / sd, y)
    model = {"mu": mu.tolist(), "sd": sd.tolist(), "coef": lr.coef_.tolist(),
             "intercept": lr.intercept_.tolist(), "classes": lr.classes_.tolist()}
    e = exposure_expected_value(model, X)
    f1, t1, t2 = grid_search_thresholds(e, y)
    model.update(t1=t1, t2=t2, lo=0.0, hi=1.0, train_macro_f1=f1)
    return model


def exposure_proba(model: dict, X: np.ndarray) -> np.ndarray:
    """Softmax of the logistic-regression logits (no sklearn needed at inference)."""
    z = ((X - np.array(model["mu"])) / np.array(model["sd"])) @ np.array(model["coef"]).T \
        + np.array(model["intercept"])
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    full = np.zeros((len(X), 3))
    full[:, model["classes"]] = p
    return full


def exposure_expected_value(model: dict, X: np.ndarray) -> np.ndarray:
    return exposure_proba(model, X) @ CLASS_VALUE


# ----------------------------------------------------------------------------
# Public API used by evaluate.py and the app
# ----------------------------------------------------------------------------
def load_classical(version: str) -> dict:
    return json.loads((MODELS_DIR / f"classical_{version}.json").read_text())


def sharp_score_for_feature(model: dict, name: str, values) -> np.ndarray:
    """[0,1] sharpness score for ANY candidate feature (used by the demo)."""
    return score_from_thresholds(transform(name, values), model["sharp_all"][name])


def predict_from_features(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Vectorised prediction for a DataFrame of precomputed features."""
    name = model["sharp_feature"]
    th = model["sharp_all"][name]
    xs = transform(name, df[name])
    e = exposure_expected_value(model["exposure"], exposure_design(df))
    ex = model["exposure"]
    out = pd.DataFrame(index=df.index)
    out["sharp_score"] = score_from_thresholds(xs, th)
    out["sharp_pred"] = [CLASSES[i] for i in apply_thresholds(xs, th["t1"], th["t2"])]
    out["exposure_score"] = score_from_thresholds(e, ex)
    out["exposure_pred"] = [CLASSES[i] for i in apply_thresholds(e, ex["t1"], ex["t2"])]
    out["quality"] = np.sqrt(out.sharp_score * out.exposure_score)
    return out


def predict_path(path, model: dict) -> dict:
    """Single-image end-to-end prediction (load + ONLY the needed features).
    This is what we time in evaluate.py."""
    gray = gray_from_bgr(load_image(path))
    row = exposure_features(gray)
    row[model["sharp_feature"]] = sharpness_feature(gray, model["sharp_feature"])
    return predict_from_features(pd.DataFrame([row]), model).iloc[0].to_dict()


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    seed_everything()
    from src.features import SHARPNESS_FEATURES, synthetic_features

    df = synthetic_features()
    train, val = df[df.split == "train"], df[df.split == "val"]
    print(f"Synthetic train {len(train)} / val {len(val)} images")

    t = time.perf_counter()
    sharp_all = {n: fit_sharp_feature(n, train) for n in SHARPNESS_FEATURES}
    exposure = fit_exposure(train)
    print(f"Grid search done in {time.perf_counter() - t:.1f}s")

    # Validation macro-F1 of every candidate sharpness feature
    yv = val.sharp_label.map(CLASS_TO_IDX).to_numpy()
    rows = []
    for n, th in sharp_all.items():
        pv = apply_thresholds(transform(n, val[n]), th["t1"], th["t2"])
        rows.append(dict(task="sharpness", feature=n, train_macro_f1=th["train_macro_f1"],
                         val_macro_f1=macro_f1(yv, pv)))
    ev = exposure_expected_value(exposure, exposure_design(val))
    rows.append(dict(task="exposure", feature="logreg(lum stats)",
                     train_macro_f1=exposure["train_macro_f1"],
                     val_macro_f1=macro_f1(val.exposure_label.map(CLASS_TO_IDX).to_numpy(),
                                           apply_thresholds(ev, exposure["t1"], exposure["t2"]))))
    table = pd.DataFrame(rows)

    best_norm = max(NORM_CANDIDATES, key=lambda n: sharp_all[n]["train_macro_f1"])
    MODELS_DIR.mkdir(exist_ok=True)
    for version, feat in (("raw", "lap_var"), ("norm", best_norm)):
        model = {"version": version, "sharp_feature": feat, "sharp_all": sharp_all,
                 "exposure": exposure,
                 "quality_formula": "Q = sqrt(sharp_score * exposure_score)"}
        (MODELS_DIR / f"classical_{version}.json").write_text(json.dumps(model, indent=2))

    RESULTS_DIR.mkdir(exist_ok=True)
    table.to_csv(RESULTS_DIR / "synthetic_baseline.csv", index=False)
    print("\nPer-feature macro-F1 on SYNTHETIC data (thresholds fitted on train):")
    print(table.round(3).to_string(index=False))
    print(f"\nRaw version uses: lap_var.   Normalised version uses: {best_norm} "
          f"(best train macro-F1 among {NORM_CANDIDATES})")

    # Where does the raw metric fail? Sharpness accuracy per degradation type.
    acc = {}
    for version in ("raw", "norm"):
        p = predict_from_features(val, load_classical(version))
        acc[version] = (p.sharp_pred == val.sharp_label).groupby(val.degradation_type).mean()
    print("\nVal sharpness accuracy by degradation type (raw vs normalised):")
    print(pd.DataFrame(acc).round(2).to_string())


if __name__ == "__main__":
    main()
