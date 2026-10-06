"""Shared constants and helpers used by every phase.

Keeping paths, label rules and image loading in ONE place guarantees that the
synthetic generator, the classical features, the CNN and the app all see
images in exactly the same way (same resolution, same colour handling).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_DIR = DATA / "raw"
SHARP_SOURCE_DIR = DATA / "sharp_source"
SYNTH_DIR = DATA / "synthetic"
LABELS_DIR = DATA / "labels"
CACHE_DIR = DATA / "cache"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"

SYNTH_METADATA = SYNTH_DIR / "metadata.csv"
REAL_LABELS = LABELS_DIR / "real_labels.csv"
RATER2_LABELS = LABELS_DIR / "rater2_labels.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ----------------------------------------------------------------------------
# Labels. Ordinal order matters (used for weighted kappa and for scores).
# ----------------------------------------------------------------------------
CLASSES = ["bad", "borderline", "good"]          # index 0 = worst
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
# Value of each class when turning class probabilities into a [0,1] score.
CLASS_VALUE = np.array([0.0, 0.5, 1.0])

# All analysis happens on a copy whose LONG side is this many pixels.
# Synthetic images are generated at 1024 px, so analysis sees them at half
# size; real phone photos (3000-4000 px) are reduced much more. This keeps
# every method fast and consistent. (See README "Limitations".)
ANALYSIS_SIDE = 512

SEED = 42


def seed_everything(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def list_images(folder: Path | str) -> list[Path]:
    """Non-recursive, sorted list of image files (skips hidden files/_trash)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    )


def load_image(path: Path | str, max_side: int = ANALYSIS_SIDE) -> np.ndarray:
    """Load an image as BGR uint8 with its long side reduced to `max_side`.

    For big JPEGs we ask OpenCV to decode at 1/2, 1/4 or 1/8 scale directly
    (libjpeg DCT scaling), which is much faster than decoding 12 MP and then
    shrinking. A final INTER_AREA resize hits the exact target size.
    Images that are already smaller are NOT upscaled (upscaling adds blur).
    """
    path = str(path)
    with Image.open(path) as im:          # reads only the header
        w, h = im.size
    long_side = max(w, h)
    flag = cv2.IMREAD_COLOR
    for factor, f in ((8, cv2.IMREAD_REDUCED_COLOR_8),
                      (4, cv2.IMREAD_REDUCED_COLOR_4),
                      (2, cv2.IMREAD_REDUCED_COLOR_2)):
        if long_side / factor >= max_side:
            flag = f
            break
    img = cv2.imread(path, flag)
    if img is None:  # fallback for formats OpenCV can't read (e.g. some webp)
        img = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def to_gray_float(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> luminance (Rec.601) float32 in [0, 1]."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


# ----------------------------------------------------------------------------
# Label rules for synthetic degradations (documented in README).
# ----------------------------------------------------------------------------
def sharp_label_from_sigma(sigma_equiv: float) -> str:
    """Blur strength -> sharpness label.

    sigma_equiv is the Gaussian sigma (in pixels at 1024 px long side) or, for
    motion blur, the std-dev of the box kernel = length / sqrt(12).
        sigma <= 0.75        -> good       (gauss 0.5)
        0.75 < sigma <= 2.5  -> borderline (gauss 1, 2; motion 5)
        sigma > 2.5          -> bad        (gauss 3, 5; motion 9, 15, 25)
    """
    if sigma_equiv <= 0.75:
        return "good"
    if sigma_equiv <= 2.5:
        return "borderline"
    return "bad"


def exposure_label_from_ev(ev: float) -> str:
    """Brightness gain in stops (EV = log2(gain)) -> exposure label.
        |EV| < 0.5        -> good
        0.5 <= |EV| < 1.0 -> borderline
        |EV| >= 1.0       -> bad
    """
    a = abs(ev)
    if a < 0.5:
        return "good"
    if a < 1.0:
        return "borderline"
    return "bad"


def exposure_label_from_contrast(alpha: float) -> str:
    """Contrast factor alpha (1 = unchanged, 0 = flat grey) -> exposure label.
        alpha >= 0.75        -> good
        0.5 <= alpha < 0.75  -> borderline
        alpha < 0.5          -> bad
    """
    if alpha >= 0.75:
        return "good"
    if alpha >= 0.5:
        return "borderline"
    return "bad"


def worst_label(*labels: str) -> str:
    """Combine labels by taking the worst one (used when two effects stack)."""
    return CLASSES[min(CLASS_TO_IDX[l] for l in labels)]
