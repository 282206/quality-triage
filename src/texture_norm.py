"""Phase 3 - content-aware sharpness metrics (the key contribution).

THE PROBLEM
    Laplacian variance (and Tenengrad) measure "how much edge energy is in the
    image". That depends on BLUR but also on CONTENT and CONTRAST:
      * a perfectly sharp photo of a plain wall / sky / document has few edges
        -> low score -> wrongly flagged "blurry";
      * a slightly blurred photo of gravel / foliage still has lots of edges
        -> high score -> wrongly passed as "sharp";
      * an under-exposed or low-contrast (but sharp) photo has weaker edges
        -> low score -> wrongly flagged "blurry".

FIX (a) TEXTURE MASKING
    Split the image into tiles, rank tiles by gradient energy and measure
    sharpness ONLY on the top-K% most textured tiles. Flat regions (wall, sky)
    no longer dilute the score; we judge blur where there is something to see.

FIX (b) SPECTRAL / RE-BLUR NORMALISATION
    Use a RATIO, so the absolute amount of content and contrast cancels out:
      * reblur ratio: blur the image again with a known Gaussian and measure
        the fraction of gradient energy that is destroyed.
            R = 1 - Tenengrad(blur(I)) / Tenengrad(I)
        A sharp image loses a lot (R near 1), an already-blurred image barely
        changes (R near 0). Contrast scales numerator and denominator equally,
        so it cancels.
      * FFT high-frequency ratio: share of spectral power above a cut-off
        frequency. Blur is a low-pass filter, so the ratio drops with blur.
    `reblur_masked` combines both ideas (ratio computed on textured tiles only).
      * directional re-blur (`reblur_dir`): MOTION blur only removes gradient
        energy ALONG its direction - edges perpendicular to the motion stay
        sharp, so an isotropic ratio is fooled. We compute the masked re-blur
        ratio separately for 4 gradient orientations (0, 45, 90, 135 deg) and
        keep the MINIMUM (the most-blurred direction).

Run the failure-case demo:  python -m src.texture_norm
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

TILE = 32          # tile size at ANALYSIS_SIDE=512 (== 64 px at 1024 px)
TOP_K = 0.2        # keep the 20% most textured tiles
REBLUR_SIGMA = 1.0
FFT_CUTOFF = 0.25  # fraction of Nyquist above which power counts as "high"


# ----------------------------------------------------------------------------
# Basic building blocks
# ----------------------------------------------------------------------------
def gradient_energy(gray: np.ndarray) -> np.ndarray:
    """Per-pixel squared Sobel gradient magnitude (the Tenengrad integrand)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return gx * gx + gy * gy


def _tiles(a: np.ndarray, tile: int) -> np.ndarray:
    """Reshape an HxW array into (n_tiles, tile*tile), dropping ragged edges."""
    h, w = (a.shape[0] // tile) * tile, (a.shape[1] // tile) * tile
    a = a[:h, :w]
    return a.reshape(h // tile, tile, w // tile, tile).swapaxes(1, 2).reshape(-1, tile * tile)


def texture_tile_mask(gray: np.ndarray, tile: int = TILE, top_k: float = TOP_K) -> np.ndarray:
    """Boolean vector over tiles: True for the top_k most textured tiles.

    Texture is measured on a slightly blurred copy so that pixel noise does
    not make a flat tile look textured.
    """
    energy = _tiles(gradient_energy(cv2.GaussianBlur(gray, (0, 0), 1.0)), tile).mean(1)
    k = max(1, int(round(len(energy) * top_k)))
    keep = np.zeros(len(energy), bool)
    keep[np.argsort(energy)[-k:]] = True
    return keep


# ----------------------------------------------------------------------------
# Raw (content-dependent) metrics - the baseline
# ----------------------------------------------------------------------------
def laplacian_variance(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def tenengrad(gray: np.ndarray) -> float:
    return float(gradient_energy(gray).mean())


# ----------------------------------------------------------------------------
# Fix (a): texture-masked metrics
# ----------------------------------------------------------------------------
def masked_metrics(gray: np.ndarray, tile: int = TILE, top_k: float = TOP_K) -> dict:
    """Laplacian variance and Tenengrad computed only on high-texture tiles."""
    keep = texture_tile_mask(gray, tile, top_k)
    lap_t = _tiles(cv2.Laplacian(gray, cv2.CV_32F, ksize=3), tile)[keep]
    ten_t = _tiles(gradient_energy(gray), tile)[keep]
    return {"masked_lap": float(lap_t.var(axis=1).mean()),
            "masked_ten": float(ten_t.mean())}


# ----------------------------------------------------------------------------
# Fix (b): ratio / spectral metrics
# ----------------------------------------------------------------------------
def reblur_ratio(gray: np.ndarray, sigma: float = REBLUR_SIGMA, keep: np.ndarray | None = None,
                 tile: int = TILE) -> float:
    """Fraction of gradient energy destroyed by re-blurring (1 = very sharp)."""
    e0 = gradient_energy(gray)
    e1 = gradient_energy(cv2.GaussianBlur(gray, (0, 0), sigma))
    if keep is not None:   # restrict to the textured tiles
        e0, e1 = _tiles(e0, tile)[keep], _tiles(e1, tile)[keep]
    return float(1.0 - e1.sum() / (e0.sum() + 1e-12))


def directional_energies(gray: np.ndarray) -> list[np.ndarray]:
    """Squared directional derivatives at 0, 90, 45 and 135 degrees."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return [gx * gx, gy * gy, (gx + gy) ** 2 / 2, (gx - gy) ** 2 / 2]


def reblur_directional(gray: np.ndarray, keep: np.ndarray | None = None,
                       sigma: float = REBLUR_SIGMA, tile: int = TILE) -> float:
    """Minimum over 4 orientations of the (masked) re-blur ratio."""
    e0 = directional_energies(gray)
    e1 = directional_energies(cv2.GaussianBlur(gray, (0, 0), sigma))
    ratios = []
    for a, b in zip(e0, e1):
        if keep is not None:
            a, b = _tiles(a, tile)[keep], _tiles(b, tile)[keep]
        ratios.append(1.0 - b.sum() / (a.sum() + 1e-12))
    return float(min(ratios))


def fft_highfreq_ratio(gray: np.ndarray, cutoff: float = FFT_CUTOFF) -> float:
    """Share of (non-DC) spectral power above `cutoff` x Nyquist."""
    h, w = gray.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)   # avoid edge leakage
    g = (gray - gray.mean()) * win
    p = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None] / 0.5   # 1.0 == Nyquist
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :] / 0.5
    r = np.sqrt(fx ** 2 + fy ** 2)
    total = p[r > 0.02].sum()
    return float(p[r > cutoff].sum() / (total + 1e-12))


def normalised_metrics(gray: np.ndarray) -> dict:
    """All content-aware sharpness metrics for one grayscale [0,1] image."""
    keep = texture_tile_mask(gray)
    out = masked_metrics(gray)
    out["reblur"] = reblur_ratio(gray)
    out["reblur_masked"] = reblur_ratio(gray, keep=keep)
    out["reblur_dir"] = reblur_directional(gray, keep=keep)
    out["fft_hf"] = fft_highfreq_ratio(gray)
    return out


# ----------------------------------------------------------------------------
# Failure-case demo: raw score vs fixed scores, side by side
# ----------------------------------------------------------------------------
DEMO_METRICS = ["lap_var", "masked_lap", "fft_hf", "reblur", "reblur_masked", "reblur_dir"]


def _demo_images():
    """Synthetic failure cases (we have no real ones yet - stated in README).

    Low-texture SHARP images: plain wall, sky gradient, flat document,
    low-contrast sharp scene.  Textured BLURRY images: gravel/foliage texture
    with Gaussian blur sigma 3 and motion blur 15 (both 'bad' by our rules).
    """
    from src.make_dummy import scene_document, scene_lowtex, scene_shapes, scene_texture, finish
    from src.synth import contrast, gaussian_blur, motion_blur

    rng = np.random.default_rng(7)
    sharp = [("wall (sharp)", finish(scene_lowtex(rng), rng)),
             ("sky (sharp)", finish(scene_lowtex(rng), rng)),
             ("document (sharp)", finish(scene_document(rng), rng)),
             ("low-contrast (sharp)", contrast(finish(scene_shapes(rng), rng), 0.35))]
    tex1, tex2 = finish(scene_texture(rng), rng), finish(scene_texture(rng), rng)
    blurry = [("gravel, gauss s=3 (blurry)", gaussian_blur(tex1, 3)),
              ("foliage, motion 15 (blurry)", motion_blur(tex2, 15, 30)),
              ("gravel, gauss s=5 (blurry)", gaussian_blur(tex2, 5))]
    return sharp, blurry


def run_demo(out_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.baseline import load_classical, sharp_score_for_feature
    from src.common import ANALYSIS_SIDE, MODELS_DIR, RESULTS_DIR
    from src.features import features_from_image

    out_path = out_path or RESULTS_DIR / "failure_case_before_after.png"
    sharp, blurry = _demo_images()
    items = [(n, im, "sharp") for n, im in sharp] + [(n, im, "blurry") for n, im in blurry]

    # Analyse each at the same resolution as everything else (resized the same
    # way load_image does).
    feats = []
    for name, im, truth in items:
        s = ANALYSIS_SIDE / max(im.shape[:2])
        small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im
        feats.append(features_from_image(small))

    # Convert each raw metric into a [0,1] sharpness score using the thresholds
    # learned on the synthetic train split (Phase 4): <1/3 = bad, >2/3 = good.
    model = load_classical("norm") if (MODELS_DIR / "classical_norm.json").exists() else None
    scores = np.zeros((len(items), len(DEMO_METRICS)))
    for j, m in enumerate(DEMO_METRICS):
        vals = np.array([f[m] for f in feats])
        if model is not None:
            scores[:, j] = sharp_score_for_feature(model, m, vals)
        else:  # fall back to min-max on log scale (only for ordering)
            v = np.log10(vals + 1e-9) if m in ("lap_var", "masked_lap") else vals
            scores[:, j] = (v - v.min()) / (np.ptp(v) + 1e-12)

    fig = plt.figure(figsize=(16, 8.5))
    n = len(items)
    for i, (name, im, truth) in enumerate(items):
        ax = fig.add_subplot(2, n, i + 1)
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=8, color="tab:green" if truth == "sharp" else "tab:red")
        ax.axis("off")
    ax = fig.add_subplot(2, 1, 2)
    width = 0.8 / len(DEMO_METRICS)
    colors = ["#888888", "#4C72B0", "#8172B2", "#55A868", "#C44E52", "#DD8452"]
    labels = {"lap_var": "raw Laplacian var (baseline)", "masked_lap": "(a) texture-masked Laplacian",
              "fft_hf": "(b) FFT high-freq ratio", "reblur": "(b) re-blur ratio",
              "reblur_masked": "(a+b) masked re-blur ratio",
              "reblur_dir": "(a+b) masked directional re-blur"}
    x = np.arange(n)
    for j, m in enumerate(DEMO_METRICS):
        ax.bar(x + (j - len(DEMO_METRICS) / 2 + 0.5) * width, scores[:, j], width,
               label=labels[m], color=colors[j])
    if model is not None:
        ax.axhline(1 / 3, ls="--", c="k", lw=0.8)
        ax.axhline(2 / 3, ls=":", c="k", lw=0.8)
        ax.text(n - 0.5, 1 / 3 + 0.01, "bad | borderline", ha="right", fontsize=8)
        ax.text(n - 0.5, 2 / 3 + 0.01, "borderline | good", ha="right", fontsize=8)
        ax.set_ylabel("sharpness score (thresholds from synthetic train)")
    else:
        ax.set_ylabel("sharpness score (min-max within demo set)")
    ax.set_xticks(x)
    ax.set_xticklabels([nm for nm, _, _ in items], rotation=12, fontsize=8)
    ax.axvline(len(sharp) - 0.5, c="k", lw=1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.set_title("Left: SHARP low-texture images should score HIGH.   "
                 "Right: BLURRY textured images should score LOW.", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    table = {nm: {m: round(float(scores[i, j]), 3) for j, m in enumerate(DEMO_METRICS)}
             for i, (nm, _, _) in enumerate(items)}
    (out_path.parent / "failure_case_scores.json").write_text(json.dumps(table, indent=2))
    print(f"Saved {out_path}")
    hdr = f"{'image':32s}" + "".join(f"{m:>15s}" for m in DEMO_METRICS)
    print(hdr)
    for nm, row in table.items():
        print(f"{nm:32s}" + "".join(f"{row[m]:15.3f}" for m in DEMO_METRICS))
    return table


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    run_demo()
