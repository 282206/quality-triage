"""Phase 1 - synthetic degradation generator.

For every sharp photo in data/sharp_source/ we create degraded copies with
KNOWN parameters, so the ground-truth label of every synthetic image is
exactly known (no human labelling needed for training / threshold tuning).

Run:  python -m src.synth
Output: data/synthetic/*.jpg  and  data/synthetic/metadata.csv
"""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import pandas as pd

from src.common import (SEED, SHARP_SOURCE_DIR, SYNTH_DIR, SYNTH_METADATA,
                        exposure_label_from_contrast, exposure_label_from_ev,
                        list_images, seed_everything, sharp_label_from_sigma)

GEN_LONG_SIDE = 1024

# Degradation grid (the strengths requested in the spec).
GAUSS_SIGMAS = [0.5, 1, 2, 3, 5]
MOTION_LENGTHS = [5, 9, 15, 25]
UNDER_GAINS = [0.8, 0.6, 0.4, 0.25]     # EV  -0.32, -0.74, -1.32, -2.0
OVER_GAINS = [1.25, 1.6, 2.0, 2.8]      # EV  +0.32, +0.68, +1.0,  +1.49
CONTRAST_ALPHAS = [0.8, 0.6, 0.4, 0.25]


# ----------------------------------------------------------------------------
# Degradation operators (all take/return RGB or BGR uint8; channel order
# does not matter for any of them).
# ----------------------------------------------------------------------------
def resize_long_side(img: np.ndarray, long_side: int = GEN_LONG_SIDE) -> np.ndarray:
    """Downscale so the long side is `long_side`. Never upscale: upscaling
    would itself blur the 'sharp' source and corrupt the labels."""
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s >= 1.0:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma,
                            borderType=cv2.BORDER_REFLECT)


def motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """A normalised straight-line kernel of `length` px at `angle_deg`."""
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0                      # horizontal line
    rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle_deg, 1.0)
    k = cv2.warpAffine(k, rot, (length, length), flags=cv2.INTER_LINEAR)
    return k / k.sum()


def motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    return cv2.filter2D(img, -1, motion_kernel(length, angle_deg),
                        borderType=cv2.BORDER_REFLECT)


def brightness(img: np.ndarray, gain: float) -> np.ndarray:
    """Multiply pixel values by `gain` (exposure change of log2(gain) stops),
    clipping at 0/255 like a real sensor/JPEG would."""
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def contrast(img: np.ndarray, alpha: float) -> np.ndarray:
    """Pull every pixel towards the image mean: out = mean + alpha*(in-mean)."""
    f = img.astype(np.float32)
    m = f.mean()
    return np.clip(m + alpha * (f - m), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Variant list for one source image
# ----------------------------------------------------------------------------
def make_variants(img: np.ndarray, rng: np.random.Generator) -> list[dict]:
    """Return a list of dicts: {img, degradation_type, strength, sharp_label,
    exposure_label, angle}. Labels follow the rules in src/common.py."""
    out = [dict(img=img, degradation_type="clean", strength=0.0,
                sharp_label="good", exposure_label="good", angle=np.nan)]
    for s in GAUSS_SIGMAS:
        out.append(dict(img=gaussian_blur(img, s), degradation_type="gaussian", strength=s,
                        sharp_label=sharp_label_from_sigma(s), exposure_label="good", angle=np.nan))
    for L in MOTION_LENGTHS:
        ang = float(rng.uniform(0, 180))
        out.append(dict(img=motion_blur(img, L, ang), degradation_type="motion", strength=L,
                        sharp_label=sharp_label_from_sigma(L / math.sqrt(12)),
                        exposure_label="good", angle=round(ang, 1)))
    for g in UNDER_GAINS:
        out.append(dict(img=brightness(img, g), degradation_type="under", strength=g,
                        sharp_label="good", exposure_label=exposure_label_from_ev(math.log2(g)),
                        angle=np.nan))
    for g in OVER_GAINS:
        out.append(dict(img=brightness(img, g), degradation_type="over", strength=g,
                        sharp_label="good", exposure_label=exposure_label_from_ev(math.log2(g)),
                        angle=np.nan))
    for a in CONTRAST_ALPHAS:
        out.append(dict(img=contrast(img, a), degradation_type="contrast", strength=a,
                        sharp_label="good", exposure_label=exposure_label_from_contrast(a),
                        angle=np.nan))
    return out


def split_sources(sources: list[str], val_frac: float = 0.2, seed: int = SEED) -> dict[str, str]:
    """Split by SOURCE image so that all variants of one photo end up in the
    same split (prevents leakage of the same content into train and val)."""
    rng = np.random.default_rng(seed)
    order = list(sources)
    rng.shuffle(order)
    n_val = max(1, round(len(order) * val_frac)) if len(order) > 1 else 0
    return {s: ("val" if i < n_val else "train") for i, s in enumerate(order)}


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="only use the first N sources")
    args = ap.parse_args(argv)

    seed_everything()
    rng = np.random.default_rng(SEED)
    sources = list_images(SHARP_SOURCE_DIR)[: args.limit]
    if not sources:
        raise SystemExit(f"No images in {SHARP_SOURCE_DIR}. Add photos or run `python -m src.make_dummy`.")

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    for old in SYNTH_DIR.glob("*.jpg"):          # regenerate from scratch
        old.unlink()

    split = split_sources([p.stem for p in sources])
    rows = []
    for p in sources:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            print(f"  ! skipping unreadable {p.name}")
            continue
        img = resize_long_side(img)
        for v in make_variants(img, rng):
            fname = f"{p.stem}__{v['degradation_type']}_{v['strength']:g}.jpg"
            cv2.imwrite(str(SYNTH_DIR / fname), v["img"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(dict(filename=fname, source=p.stem,
                             degradation_type=v["degradation_type"], strength=v["strength"],
                             angle=v["angle"], sharp_label=v["sharp_label"],
                             exposure_label=v["exposure_label"], split=split[p.stem]))
        print(f"  {p.name}: {img.shape[1]}x{img.shape[0]} -> {len(rows)} images so far")

    df = pd.DataFrame(rows)
    df.to_csv(SYNTH_METADATA, index=False)
    print(f"\nWrote {len(df)} images from {len(sources)} sources -> {SYNTH_METADATA}")
    print("\nSources per split:", df.groupby("split")["source"].nunique().to_dict())
    print("\nSharpness labels:\n", pd.crosstab(df.degradation_type, df.sharp_label))
    print("\nExposure labels:\n", pd.crosstab(df.degradation_type, df.exposure_label))
    return df


if __name__ == "__main__":
    main()
