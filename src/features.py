"""Phase 2 - classical per-image features, with a CSV cache.

Sharpness : Laplacian variance, Tenengrad (+ the content-aware metrics from
            src/texture_norm.py, so one pass computes everything)
Exposure  : mean / std / skew of luminance, 5th & 95th percentile,
            % of pixels clipped near 0 and near 255, coarse 8-bin histogram
Duplicates: 64-bit perceptual hash (pHash) as a hex string

Everything is computed on a grayscale copy with long side ANALYSIS_SIDE (512).

Run:  python -m src.features           (synthetic set + data/raw)
Cache: data/cache/features_<name>.csv  (rows reused if path+mtime+size match)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import skew

from src import texture_norm as tn
from src.common import (CACHE_DIR, RAW_DIR, SYNTH_DIR, SYNTH_METADATA, list_images,
                        load_image)

SHARPNESS_FEATURES = ["lap_var", "tenengrad", "masked_lap", "masked_ten",
                      "reblur", "reblur_masked", "reblur_dir", "fft_hf"]
HIST_BINS = 8
EXPOSURE_FEATURES = ["lum_mean", "lum_std", "lum_skew", "lum_p05", "lum_p95",
                     "clip_low", "clip_high"] + [f"hist_{i}" for i in range(HIST_BINS)]

CLIP_LOW, CLIP_HIGH = 5 / 255, 250 / 255


def gray_from_bgr(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def _safe_skew(flat: np.ndarray) -> float:
    """Skewness, defined as 0 for (nearly) constant images."""
    if flat.std() < 1e-4:
        return 0.0
    return float(skew(flat))


def exposure_features(gray: np.ndarray) -> dict:
    """Luminance histogram statistics (gray in [0,1])."""
    flat = gray.ravel()
    p05, p95 = np.percentile(flat, [5, 95])
    f = {"lum_mean": float(flat.mean()), "lum_std": float(flat.std()),
         "lum_skew": _safe_skew(flat), "lum_p05": float(p05), "lum_p95": float(p95),
         "clip_low": float((flat <= CLIP_LOW).mean()),
         "clip_high": float((flat >= CLIP_HIGH).mean())}
    hist = np.histogram(flat, bins=HIST_BINS, range=(0, 1))[0] / flat.size
    f.update({f"hist_{i}": float(h) for i, h in enumerate(hist)})
    return f


def sharpness_feature(gray: np.ndarray, name: str) -> float:
    """Compute ONE sharpness feature (used for honest per-method timing)."""
    if name == "lap_var":
        return tn.laplacian_variance(gray)
    if name == "tenengrad":
        return tn.tenengrad(gray)
    if name in ("masked_lap", "masked_ten"):
        return tn.masked_metrics(gray)[name]
    if name == "reblur":
        return tn.reblur_ratio(gray)
    if name == "reblur_masked":
        return tn.reblur_ratio(gray, keep=tn.texture_tile_mask(gray))
    if name == "reblur_dir":
        return tn.reblur_directional(gray, keep=tn.texture_tile_mask(gray))
    if name == "fft_hf":
        return tn.fft_highfreq_ratio(gray)
    raise KeyError(name)


def phash_hex(img_bgr: np.ndarray) -> str:
    rgb = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    return str(imagehash.phash(rgb))


def features_from_image(img_bgr: np.ndarray) -> dict:
    """All features for an already-loaded (analysis-size) BGR image."""
    gray = gray_from_bgr(img_bgr)
    f = {"lap_var": tn.laplacian_variance(gray), "tenengrad": tn.tenengrad(gray)}
    f.update(tn.normalised_metrics(gray))
    f.update(exposure_features(gray))
    f["phash"] = phash_hex(img_bgr)
    f["height"], f["width"] = gray.shape
    return f


def features_from_path(path: Path | str) -> dict:
    return features_from_image(load_image(path))


def extract_features(paths: list[Path], cache_csv: Path | None = None,
                     progress=None) -> pd.DataFrame:
    """Features for many images, reusing cached rows when the file is unchanged.

    progress: optional callback(i, n) - used by the Streamlit progress bar.
    """
    cached = {}
    if cache_csv is not None and cache_csv.exists():
        old = pd.read_csv(cache_csv, dtype={"phash": str})
        # Ignore the cache if it was written by an older feature set.
        if set(SHARPNESS_FEATURES + EXPOSURE_FEATURES + ["phash"]) <= set(old.columns):
            cached = {r["path"]: r for r in old.to_dict("records")}

    rows, n_new = [], 0
    for i, p in enumerate(paths):
        st = p.stat()
        c = cached.get(str(p))
        if c is not None and int(c["mtime"]) == st.st_mtime_ns and c["size"] == st.st_size:
            rows.append(c)
        else:
            f = features_from_path(p)
            f.update(path=str(p), filename=p.name, mtime=st.st_mtime_ns, size=st.st_size)
            rows.append(f)
            n_new += 1
        if progress:
            progress(i + 1, len(paths))
    df = pd.DataFrame(rows)
    if cache_csv is not None and len(df):
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
    print(f"  features: {len(df)} images ({n_new} computed, {len(df) - n_new} from cache)")
    return df


def synthetic_features() -> pd.DataFrame:
    """Features for the synthetic set, merged with its metadata/labels."""
    meta = pd.read_csv(SYNTH_METADATA)
    paths = [SYNTH_DIR / f for f in meta.filename]
    feats = extract_features(paths, CACHE_DIR / "features_synthetic.csv")
    return meta.merge(feats.drop(columns=["path"]), on="filename")


def raw_features(folder: Path = RAW_DIR) -> pd.DataFrame:
    return extract_features(list_images(folder), CACHE_DIR / "features_raw.csv")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)
    t = time.perf_counter()
    syn = synthetic_features()
    print(f"Synthetic: {len(syn)} rows in {time.perf_counter() - t:.1f}s")
    raw = raw_features()
    print(f"Raw: {len(raw)} rows")
    pd.set_option("display.width", 200)
    print("\nMean sharpness features per synthetic sharp_label:")
    print(syn.groupby("sharp_label")[SHARPNESS_FEATURES].median().round(4))
    print("\nMean exposure features per synthetic exposure_label:")
    print(syn.groupby("exposure_label")[EXPOSURE_FEATURES].median().round(3))


if __name__ == "__main__":
    main()
