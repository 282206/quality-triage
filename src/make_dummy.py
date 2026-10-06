"""Create a tiny DUMMY dataset so the whole pipeline runs before real photos exist.

It writes:
  data/sharp_source/dummy_*.png   sharp "source" images for Phase 1
  data/raw/dummy_*.jpg            fake "real" photos, degraded with RANDOM
                                  continuous parameters (different from the
                                  synthetic grid) + JPEG + noise
  data/labels/real_labels.csv     labels derived from those parameters
  data/labels/rater2_labels.csv   a noisy second rater on a subset

Images are procedural (shapes, text, textures, low-texture walls/documents)
plus a few photos bundled with scikit-image. Every file is prefixed `dummy_`
so `python -m src.make_dummy --clean` removes exactly these and nothing else.

THESE NUMBERS ARE NOT REAL RESULTS - replace with real photos + labels.
"""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import pandas as pd

from src.common import (CLASSES, LABELS_DIR, RATER2_LABELS, RAW_DIR, REAL_LABELS,
                        SEED, SHARP_SOURCE_DIR, exposure_label_from_contrast,
                        exposure_label_from_ev, sharp_label_from_sigma, worst_label)
from src.synth import brightness, contrast, gaussian_blur, motion_blur

W, H = 1024, 768


# ----------------------------------------------------------------------------
# Procedural scene generators (all return BGR uint8, H x W x 3)
# ----------------------------------------------------------------------------
def _gradient_bg(rng, h=H, w=W):
    c0, c1 = rng.uniform(40, 220, 3), rng.uniform(40, 220, 3)
    t = np.linspace(0, 1, w)[None, :, None] * rng.uniform(0.3, 1) + \
        np.linspace(0, 1, h)[:, None, None] * rng.uniform(0, 0.7)
    t = t / t.max()
    return (c0 * (1 - t) + c1 * t).astype(np.uint8)


def _fractal_texture(rng, h=H, w=W):
    """Sum of noise at several scales -> gravel / foliage-like texture."""
    tex = np.zeros((h, w), np.float32)
    for scale, amp in ((2, 1.0), (4, 0.8), (8, 0.6), (16, 0.4), (64, 0.3)):
        small = rng.normal(0, 1, (h // scale + 1, w // scale + 1)).astype(np.float32)
        tex += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    tex = (tex - tex.min()) / (tex.max() - tex.min())
    tint = rng.uniform(0.5, 1.0, 3)
    return (tex[..., None] * 200 * tint + 25).astype(np.uint8)


def scene_shapes(rng):
    img = _gradient_bg(rng)
    for _ in range(rng.integers(8, 18)):
        col = tuple(float(c) for c in rng.uniform(0, 255, 3))
        kind = rng.integers(3)
        if kind == 0:
            x, y = rng.integers(0, W), rng.integers(0, H)
            cv2.rectangle(img, (int(x), int(y)), (int(x + rng.integers(40, 300)), int(y + rng.integers(40, 300))), col, -1, cv2.LINE_AA)
        elif kind == 1:
            cv2.circle(img, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(20, 150)), col, -1, cv2.LINE_AA)
        else:
            pts = rng.integers(0, [W, H], size=(int(rng.integers(3, 6)), 2)).astype(np.int32)
            cv2.fillPoly(img, [pts], col, cv2.LINE_AA)
    for _ in range(3):
        cv2.putText(img, "TRIAGE %d" % rng.integers(1000), (int(rng.integers(0, W - 300)), int(rng.integers(40, H))),
                    cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.8, 2)), (20, 20, 20), 2, cv2.LINE_AA)
    return img


def scene_texture(rng):
    img = _fractal_texture(rng)
    for _ in range(rng.integers(2, 6)):          # a few objects on the texture
        cv2.circle(img, (int(rng.integers(0, W)), int(rng.integers(0, H))), int(rng.integers(20, 90)),
                   tuple(float(c) for c in rng.uniform(0, 255, 3)), -1, cv2.LINE_AA)
    return img


def scene_lowtex(rng):
    """Plain wall / sky: smooth gradient + only a few sharp edges."""
    img = _gradient_bg(rng)
    for _ in range(rng.integers(1, 4)):
        p1 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        p2 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        cv2.line(img, p1, p2, tuple(float(c) for c in rng.uniform(0, 255, 3)), int(rng.integers(2, 6)), cv2.LINE_AA)
    if rng.random() < 0.5:   # e.g. a light switch / window frame
        x, y = int(rng.integers(100, W - 200)), int(rng.integers(100, H - 200))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(30, 150)), y + int(rng.integers(30, 150))),
                      tuple(float(c) for c in rng.uniform(0, 255, 3)), int(rng.integers(2, 5)), cv2.LINE_AA)
    return img


def scene_document(rng):
    """Flat document: mostly white page with sparse lines of text."""
    img = np.full((H, W, 3), int(rng.uniform(215, 245)), np.uint8)
    y = int(rng.integers(60, 120))
    while y < H - 60 and rng.random() < 0.9:
        words = " ".join("".join(chr(int(c)) for c in rng.integers(97, 123, rng.integers(2, 9))) for _ in range(6))
        cv2.putText(img, words, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (30, 30, 30), 2, cv2.LINE_AA)
        y += int(rng.integers(40, 90))
    return img


SCENES = {"shapes": scene_shapes, "texture": scene_texture,
          "lowtex": scene_lowtex, "document": scene_document}


def finish(img, rng, noise=1.5):
    """Mild sensor-like noise, then uint8."""
    img = img.astype(np.float32) + rng.normal(0, noise, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def skimage_photos():
    """Photos bundled with scikit-image (no download needed). Returns BGR list."""
    out = {}
    try:
        from skimage import data
        for name in ["astronaut", "coffee", "chelsea", "rocket", "camera", "immunohistochemistry",
                     "hubble_deep_field", "coins", "moon", "page", "text", "grass", "gravel", "brick"]:
            try:
                im = getattr(data, name)()
            except Exception:
                continue
            if im.ndim == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            else:
                im = cv2.cvtColor(im[..., :3], cv2.COLOR_RGB2BGR)
            out[name] = im
    except ImportError:
        pass
    return out


# ----------------------------------------------------------------------------
# Random "real-like" degradation for the dummy raw set
# ----------------------------------------------------------------------------
def random_degrade(img, rng):
    """Apply ONE random degradation (sometimes blur + exposure) with continuous
    parameters, and return (image, sharp_label, exposure_label, description)."""
    sharp, expo, desc = "good", "good", []
    r = rng.random()
    if r < 0.45:  # blur
        if rng.random() < 0.5:
            s = float(rng.choice([0, 0.4, 0.8, 1.5, 2.2, 3.5, 4.5, 6]))
            if s > 0:
                img = gaussian_blur(img, s)
            sharp, desc = sharp_label_from_sigma(s), [f"gauss{s}"]
        else:
            L = int(rng.choice([3, 7, 11, 18, 30]))
            img = motion_blur(img, L, float(rng.uniform(0, 180)))
            sharp, desc = sharp_label_from_sigma(L / math.sqrt(12)), [f"motion{L}"]
    if r >= 0.35:  # exposure (overlaps with blur for 0.35<=r<0.45)
        kind = rng.choice(["under", "over", "contrast", "none"])
        if kind in ("under", "over"):
            ev = float(rng.choice([0.2, 0.6, 0.9, 1.3, 2.0])) * (-1 if kind == "under" else 1)
            img = brightness(img, 2 ** ev)
            expo = exposure_label_from_ev(ev)
            desc.append(f"ev{ev:+.1f}")
        elif kind == "contrast":
            a = float(rng.choice([0.85, 0.65, 0.45, 0.3]))
            img = contrast(img, a)
            expo = exposure_label_from_contrast(a)
            desc.append(f"contrast{a}")
    return img, sharp, expo, "+".join(desc) or "clean"


def flip_adjacent(label, rng):
    """Move a label one step up or down the ordinal scale (simulated rater noise)."""
    i = CLASSES.index(label)
    j = i + (rng.choice([-1, 1]) if 0 < i < 2 else (1 if i == 0 else -1))
    return CLASSES[j]


def clean():
    n = 0
    for d in (SHARP_SOURCE_DIR, RAW_DIR):
        for p in d.glob("dummy_*"):
            p.unlink()
            n += 1
    for f in (REAL_LABELS, RATER2_LABELS):
        if f.exists():
            df = pd.read_csv(f)
            df = df[~df.filename.str.startswith("dummy_")]
            if df.empty:
                f.unlink()
            else:
                df.to_csv(f, index=False)
    print(f"Removed {n} dummy images and dummy label rows.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", action="store_true", help="delete all dummy_* files and label rows")
    ap.add_argument("--n-sources", type=int, default=24, help="procedural sharp sources")
    ap.add_argument("--n-raw", type=int, default=60, help="dummy 'real' photos")
    args = ap.parse_args(argv)
    if args.clean:
        return clean()

    rng = np.random.default_rng(SEED)
    for d in (SHARP_SOURCE_DIR, RAW_DIR, LABELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # --- sharp sources: procedural scenes + half of the skimage photos -------
    kinds = list(SCENES)
    photos = skimage_photos()
    names = sorted(photos)
    src_photos, raw_photos = names[::2], names[1::2]   # disjoint: no leakage into the "real" set
    for i in range(args.n_sources):
        k = kinds[i % len(kinds)]
        cv2.imwrite(str(SHARP_SOURCE_DIR / f"dummy_{k}_{i:02d}.png"), finish(SCENES[k](rng), rng))
    for n in src_photos:
        cv2.imwrite(str(SHARP_SOURCE_DIR / f"dummy_sk_{n}.png"), photos[n])

    # --- dummy "real" photos with random continuous degradations -----------
    rows = []
    bases = []
    for i in range(args.n_raw):
        if raw_photos and i % 4 == 3:
            name = raw_photos[(i // 4) % len(raw_photos)]
            base = photos[name].copy()
        else:
            k = kinds[i % len(kinds)]
            base = finish(SCENES[k](rng), rng, noise=2.5)
        bases.append(base)
        img, s, e, desc = random_degrade(base, rng)
        fname = f"dummy_raw_{i:03d}.jpg"
        cv2.imwrite(str(RAW_DIR / fname), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(dict(filename=fname, sharpness=s, exposure=e, rater="dummy_rules", note=desc))

    # --- a few near-duplicates (burst-shot style: tiny shift + re-encode) --
    for j in range(5):
        src = bases[j * 3]
        dx, dy = int(rng.integers(-12, 12)), int(rng.integers(-12, 12))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        dup = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]), borderMode=cv2.BORDER_REFLECT)
        dup = brightness(dup, float(rng.uniform(0.92, 1.08)))
        fname = f"dummy_raw_{j * 3:03d}_dup.jpg"
        cv2.imwrite(str(RAW_DIR / fname), dup, [cv2.IMWRITE_JPEG_QUALITY, 80])
        rows.append(dict(filename=fname, sharpness="good", exposure="good",
                         rater="dummy_rules", note="near-duplicate of clean base"))

    real = pd.DataFrame(rows)
    real.to_csv(REAL_LABELS, index=False)

    # --- rater 2: 30 images, ~15% of labels moved one class ----------------
    sub = real.sample(n=min(30, len(real)), random_state=SEED).copy()
    for col in ("sharpness", "exposure"):
        mask = rng.random(len(sub)) < 0.15
        sub.loc[mask, col] = [flip_adjacent(l, rng) for l in sub.loc[mask, col]]
    sub["rater"] = "dummy_rater2"
    sub[["filename", "sharpness", "exposure", "rater"]].to_csv(RATER2_LABELS, index=False)

    print(f"Sharp sources: {len(list(SHARP_SOURCE_DIR.glob('dummy_*')))} "
          f"({args.n_sources} procedural + {len(src_photos)} skimage photos)")
    print(f"Raw dummy photos: {len(real)}  -> {REAL_LABELS}")
    print(f"Rater-2 subset: {len(sub)}  -> {RATER2_LABELS}")
    print("Real label counts:\n", real.sharpness.value_counts().to_dict(), real.exposure.value_counts().to_dict())


if __name__ == "__main__":
    main()
