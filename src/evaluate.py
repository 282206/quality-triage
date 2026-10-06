"""Phase 7 - evaluation on the REAL labelled set only.

Synthetic data is used only for training and threshold selection; every
number here comes from data/raw/ + data/labels/real_labels.csv.

For each method (classical raw, classical texture-normalised, CNN):
  * accuracy, macro precision / recall / F1, per-class P/R/F1
    (sharpness and exposure separately)
  * binary "bad vs not-bad" precision / recall / F1 (what matters for deletion)
    per task, plus "delete?" = bad on EITHER axis
  * confusion matrices
  * CPU inference time per image, end to end (JPEG decode + resize + features
    / CNN forward + prediction), mean and p95, after 3 warm-up images

Run:  python -m src.evaluate
Out:  results/comparison_table.{csv,md}, results/per_class.csv,
      results/accuracy_vs_speed.png, results/confusion_<method>.png,
      results/predictions_<method>.csv, results/summary.md
"""
from __future__ import annotations

import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from src.common import CLASSES, MODELS_DIR, RAW_DIR, REAL_LABELS, RESULTS_DIR, seed_everything

N_WARMUP = 3
METHOD_NAMES = {"classical_raw": "Classical (raw Laplacian)",
                "classical_norm": "Classical (texture-normalised)",
                "cnn": "CNN (MobileNetV3-Small)"}
COLORS = {"classical_raw": "#2a78d6", "classical_norm": "#eb6834", "cnn": "#1baf7a"}
INK, MUTED = "#1f1f1e", "#6b6a64"


# ----------------------------------------------------------------------------
# Running each method with timing
# ----------------------------------------------------------------------------
def get_predictors() -> dict:
    """name -> function(path) -> dict with sharp_pred, exposure_pred, quality..."""
    from src import baseline

    preds = {}
    for v in ("raw", "norm"):
        if (MODELS_DIR / f"classical_{v}.json").exists():
            m = baseline.load_classical(v)
            preds[f"classical_{v}"] = (lambda p, m=m: baseline.predict_path(p, m))
    try:
        from src import cnn
        model = cnn.load_cnn()
        preds["cnn"] = lambda p: cnn.predict_path(model, p)
    except FileNotFoundError as e:
        print(f"  (skipping CNN: {e})")
    return preds


def run_timed(fn, paths) -> tuple[pd.DataFrame, np.ndarray]:
    for p in paths[:N_WARMUP]:      # warm-up: caches, lazy init, thread pools
        fn(p)
    rows, ms = [], []
    for p in paths:
        t = time.perf_counter()
        r = fn(p)
        ms.append((time.perf_counter() - t) * 1000)
        rows.append({"filename": p.name, **r})
    return pd.DataFrame(rows), np.array(ms)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def task_metrics(y_true, y_pred) -> tuple[dict, list[dict]]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES,
                                                 average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, y_pred, labels=CLASSES,
                                                     average=None, zero_division=0)
    yt, yp = np.asarray(y_true) == "bad", np.asarray(y_pred) == "bad"
    bp, br, bf, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
    summary = {"acc": accuracy_score(y_true, y_pred), "macro_P": p, "macro_R": r, "macro_F1": f,
               "bad_P": bp, "bad_R": br, "bad_F1": bf}
    per_class = [{"class": c, "precision": pc[i], "recall": rc[i], "f1": fc[i], "support": int(sc[i])}
                 for i, c in enumerate(CLASSES)]
    return summary, per_class


def plot_confusions(name: str, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, task, pred in zip(axes, ("sharpness", "exposure"), ("sharp_pred", "exposure_pred")):
        cm = confusion_matrix(df[task], df[pred], labels=CLASSES)
        ax.imshow(cm, cmap="Blues", vmin=0)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=12,
                        color="white" if cm[i, j] > cm.max() * 0.6 else INK)
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("predicted", color=MUTED)
        ax.set_ylabel("true (human label)", color=MUTED)
        ax.set_title(task, color=INK)
    fig.suptitle(METHOD_NAMES[name], color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_{name}.png", dpi=120)
    plt.close(fig)


def plot_speed(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for _, r in table.iterrows():
        ax.scatter(r.ms_mean, r.mean_macro_F1, s=110, color=COLORS[r.method], zorder=3,
                   edgecolor="white", linewidth=2)
        ax.annotate(f"{METHOD_NAMES[r.method]}\n{r.ms_mean:.1f} ms, F1 {r.mean_macro_F1:.2f}",
                    (r.ms_mean, r.mean_macro_F1), textcoords="offset points",
                    xytext=(10, -30) if r.method == "classical_raw" else (10, 6),
                    fontsize=9, color=INK)
    ax.set_xscale("log")
    lo, hi = table.ms_mean.min(), table.ms_mean.max()
    ax.set_xlim(lo / 2, hi * 6)
    ax.set_ylim(0, 1)
    ax.set_xlabel("CPU time per image, ms (log scale, incl. decode + features)", color=MUTED)
    ax.set_ylabel("macro-F1 (mean of sharpness & exposure)", color=MUTED)
    ax.set_title("Accuracy vs speed on the real labelled set", color=INK)
    ax.grid(True, which="major", color="#e6e5df", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_vs_speed.png", dpi=120)
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table writer (avoids the optional `tabulate` dependency)."""
    cols = list(df.columns)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_summary(table: pd.DataFrame, n: int, dummy: bool) -> str:
    t = table.set_index("method")
    best = t.mean_macro_F1.idxmax()
    fastest = t.ms_mean.idxmin()
    lines = ["# Results summary", ""]
    if dummy:
        lines += ["> **WARNING: these numbers come from the DUMMY test set** (procedurally generated "
                  "images labelled by rule). They only prove the pipeline runs. Replace with real "
                  "photos and human labels before quoting any result.", ""]
    lines += [f"Test set: {n} real labelled images (never used for training or threshold selection).", "",
              to_markdown(table[["method", "sharp_macro_F1", "exp_macro_F1", "mean_macro_F1",
                                 "sharp_bad_P", "sharp_bad_R", "delete_P", "delete_R",
                                 "ms_mean", "ms_p95"]]), ""]
    if "classical_raw" in t.index and "classical_norm" in t.index:
        d = t.loc["classical_norm", "sharp_macro_F1"] - t.loc["classical_raw", "sharp_macro_F1"]
        lines.append(f"* **Content-aware fix:** texture-normalised sharpness changes sharpness macro-F1 by "
                     f"{d:+.3f} vs the raw Laplacian ({t.loc['classical_raw','sharp_macro_F1']:.3f} -> "
                     f"{t.loc['classical_norm','sharp_macro_F1']:.3f}) at "
                     f"{t.loc['classical_norm','ms_mean']:.1f} vs {t.loc['classical_raw','ms_mean']:.1f} ms/image.")
    if "cnn" in t.index:
        ratio = t.loc["cnn", "ms_mean"] / t.loc[fastest, "ms_mean"]
        lines.append(f"* **Speed/accuracy trade-off:** the CNN takes {t.loc['cnn','ms_mean']:.1f} ms/image "
                     f"(p95 {t.loc['cnn','ms_p95']:.1f}), {ratio:.1f}x the fastest method "
                     f"({METHOD_NAMES[fastest]}), for a mean macro-F1 of {t.loc['cnn','mean_macro_F1']:.3f} vs "
                     f"{t.loc[fastest,'mean_macro_F1']:.3f}.")
    lines.append(f"* **Best overall mean macro-F1:** {METHOD_NAMES[best]} ({t.loc[best,'mean_macro_F1']:.3f}).")
    lines.append("* For deletion, look at `delete_P` (of images flagged bad, how many really are - "
                 "low precision means good photos get binned) and `delete_R` (how many truly bad photos "
                 "are caught). Because the app only MOVES files to `_trash/`, a recall-leaning operating "
                 "point is acceptable.")
    lines += ["", "Small test sets give noisy estimates: one image changes accuracy by "
              f"{100 / max(n, 1):.1f} percentage points.", ""]
    text = "\n".join(lines)
    (RESULTS_DIR / "summary.md").write_text(text)
    return text


# ----------------------------------------------------------------------------
def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    seed_everything()
    RESULTS_DIR.mkdir(exist_ok=True)
    if not REAL_LABELS.exists():
        raise SystemExit(f"{REAL_LABELS} not found - label photos in the app (Label page) first.")
    labels = pd.read_csv(REAL_LABELS).dropna(subset=["sharpness", "exposure"])
    labels = labels[labels.filename.map(lambda f: (RAW_DIR / f).exists())].reset_index(drop=True)
    if labels.empty:
        raise SystemExit("No labelled images found in data/raw.")
    paths = [RAW_DIR / f for f in labels.filename]
    dummy = labels.filename.str.startswith("dummy_").any()
    print(f"Evaluating on {len(paths)} real labelled images{' (DUMMY SET)' if dummy else ''}")

    rows, per_class_rows = [], []
    for name, fn in get_predictors().items():
        pred, ms = run_timed(fn, paths)
        df = labels.merge(pred, on="filename")
        df.to_csv(RESULTS_DIR / f"predictions_{name}.csv", index=False)
        row = {"method": name}
        for task, col, key in (("sharpness", "sharp_pred", "sharp"), ("exposure", "exposure_pred", "exp")):
            s, pc = task_metrics(df[task], df[col])
            row.update({f"{key}_{k}": v for k, v in s.items()})
            per_class_rows += [{"method": name, "task": task, **r} for r in pc]
        # "delete?" decision = bad on either axis
        yt = (df.sharpness == "bad") | (df.exposure == "bad")
        yp = (df.sharp_pred == "bad") | (df.exposure_pred == "bad")
        dp, dr, df1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        row.update(delete_P=dp, delete_R=dr, delete_F1=df1,
                   mean_macro_F1=(row["sharp_macro_F1"] + row["exp_macro_F1"]) / 2,
                   ms_mean=float(ms.mean()), ms_p95=float(np.percentile(ms, 95)))
        rows.append(row)
        plot_confusions(name, df)
        print(f"  {name:15s} sharp F1 {row['sharp_macro_F1']:.3f}  exp F1 {row['exp_macro_F1']:.3f}  "
              f"delete P/R {dp:.2f}/{dr:.2f}  {row['ms_mean']:.1f} ms (p95 {row['ms_p95']:.1f})")

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(RESULTS_DIR / "per_class.csv", index=False)
    show = table[["method", "sharp_acc", "sharp_macro_P", "sharp_macro_R", "sharp_macro_F1", "sharp_bad_P",
                  "sharp_bad_R", "exp_acc", "exp_macro_P", "exp_macro_R", "exp_macro_F1", "exp_bad_P",
                  "exp_bad_R", "delete_P", "delete_R", "ms_mean", "ms_p95"]]
    (RESULTS_DIR / "comparison_table.md").write_text(to_markdown(show) + "\n")
    plot_speed(table)
    agreement = RESULTS_DIR / "agreement.json"
    if agreement.exists():
        print("  inter-rater agreement:", {k: v.get("kappa_quadratic") for k, v in
                                          json.loads(agreement.read_text()).items() if isinstance(v, dict)})
    print("\n" + write_summary(table, len(paths), dummy))


if __name__ == "__main__":
    main()
