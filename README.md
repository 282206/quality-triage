# Blur and Quality Triage

Scan a folder of photos, rank them by quality, and flag **blurred**, **badly
exposed** and **near-duplicate** images so you can clean them up. Nothing is
ever deleted: the app moves selected photos into a `_trash/` subfolder, and you
can undo or restore from there.

The project compares three methods:

| Method | Sharpness signal | Exposure signal |
|---|---|---|
| Classical (raw) | log Laplacian variance, 2 grid-searched thresholds | logistic regression on luminance stats + histogram |
| **Classical (texture-normalised)**, my contribution | content-aware metric (picked automatically, see Phase 3) | same as above |
| CNN | MobileNetV3-Small fine-tuned, 6-output head | same network |

> **Status: the numbers below come from the DUMMY dataset.** They show the
> pipeline works end to end. They are *not* results to quote. See
> [Using your real photos](#using-your-real-photos).

---

## Setup

```bash
cd quality-triage
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ (tested on 3.13, torch 2.11, OpenCV 5.0, Streamlit 1.65).
The first CNN run downloads the ImageNet MobileNetV3-Small weights (~10 MB) to
`~/.cache/torch/hub/checkpoints/`. If that download fails, `src/cnn.py` stops
with a clear message. It never silently trains from scratch; you have to pass
`--from-scratch` explicitly.

## Quick start

```bash
python run_all.py --dummy          # dummy data -> synth -> features -> baseline -> demo -> CNN -> agreement -> evaluate
python run_all.py --dummy --quick  # same, but a 1-minute CNN smoke test
streamlit run app.py               # the app
```

## Running each part

All modules are run from the `quality-triage/` folder as `python -m src.<name>`.

| Step | Command | Output |
|---|---|---|
| Dummy data (optional) | `python -m src.make_dummy` (`--clean` removes it) | `data/sharp_source/dummy_*`, `data/raw/dummy_*`, `data/labels/*.csv` |
| 1. Synthetic set | `python -m src.synth` | `data/synthetic/*.jpg`, `data/synthetic/metadata.csv` |
| 2. Features | `python -m src.features` | `data/cache/features_{synthetic,raw}.csv` (cached) |
| 2. Duplicates | `python -m src.dedup --threshold 8` | prints groups |
| 4. Classical baseline | `python -m src.baseline` | `models/classical_{raw,norm}.json`, `results/synthetic_baseline.csv` |
| 3. Failure-case demo | `python -m src.texture_norm` (after baseline) | `results/failure_case_before_after.png` |
| 5. CNN | `python -m src.cnn` (`--quick` for a smoke test) | `models/cnn_best.pt`, `models/cnn_history.csv` |
| 6. Agreement | `python -m src.agreement` | `results/agreement.json` |
| 7. Evaluation | `python -m src.evaluate` | `results/comparison_table.{csv,md}`, `per_class.csv`, `confusion_*.png`, `accuracy_vs_speed.png`, `summary.md` |
| 8. App | `streamlit run app.py` | — |

Everything is seeded (`SEED = 42` in `src/common.py`).

---

## How it works

### Shared conventions (`src/common.py`)
* Every method analyses the image with its **long side reduced to 512 px**
  (`ANALYSIS_SIDE`). Big JPEGs are decoded directly at 1/2, 1/4 or 1/8 scale
  (libjpeg DCT scaling), which is fast. Smaller images are never upscaled.
* Labels are ordinal: `bad < borderline < good`.

### Phase 1: synthetic data (`src/synth.py`)
Each sharp source is resized to a 1024 px long side (downscale only) and
turned into **22 variants** with known parameters:

| Degradation | Strengths | Label rule |
|---|---|---|
| clean | — | sharp good, exposure good |
| Gaussian blur | σ = 0.5, 1, 2, 3, 5 | uses σ |
| Motion blur (random angle) | length L = 5, 9, 15, 25 | uses σ_equiv = L/√12 (std of a box kernel) |
| Under-exposure (× gain) | 0.8, 0.6, 0.4, 0.25 | uses EV = log2(gain) |
| Over-exposure (× gain, clipped) | 1.25, 1.6, 2.0, 2.8 | uses EV |
| Contrast reduction | α = 0.8, 0.6, 0.4, 0.25 | uses α |

**Sharpness:** σ ≤ 0.75 → good, 0.75 < σ ≤ 2.5 → borderline, σ > 2.5 → bad
(so Gaussian 0.5 is good, 1 and 2 are borderline, 3 and 5 are bad; motion 5 is
borderline, 9 and above are bad).
**Exposure:** |EV| < 0.5 → good, 0.5–1 → borderline, ≥ 1 → bad. Contrast:
α ≥ 0.75 → good, 0.5–0.75 → borderline, < 0.5 → bad.
Exposure variants are labelled *sharp = good*: they are not blurred, and that
is exactly where the raw Laplacian fails.
The train/val split (80/20) is **by source image**, so variants of one photo
never leak across splits.

### Phase 2: classical features (`src/features.py`, `src/dedup.py`)
* Sharpness: Laplacian variance; Tenengrad (mean squared Sobel magnitude).
* Exposure: luminance mean, std, skew, 5th/95th percentile, % clipped (≤ 5 or
  ≥ 250), and an 8-bin histogram.
* pHash (`imagehash.phash`). Images with Hamming distance ≤ 8 (of 64 bits)
  are linked, and **union-find** merges the links into groups.
* Results are cached in CSV, keyed by path + mtime + size.

### Phase 3: content-aware sharpness (`src/texture_norm.py`), the key contribution
**Problem:** Laplacian variance measures *how much edge energy* an image has.
That depends on content and contrast as well as blur. A sharp photo of a wall
scores "blurry", while blurry gravel scores "sharp".

* **(a) Texture masking:** split the image into 32×32 tiles (= 64×64 at
  1024 px), rank the tiles by gradient energy, and measure only the top 20%
  (`masked_lap`, `masked_ten`).
* **(b) Ratio / spectral normalisation:**
  * `fft_hf`: share of spectral power above 0.25 × Nyquist (Hann-windowed).
  * `reblur`: **R = 1 − Tenengrad(blur(I)) / Tenengrad(I)**. This is the fraction
    of gradient energy destroyed by re-blurring with σ = 1. A sharp image loses
    a lot; an already-blurred image barely changes. Contrast multiplies the
    numerator and denominator equally, so it cancels.
  * `reblur_masked`: the same ratio, computed on the textured tiles only, (a)+(b).
  * `reblur_dir`: motion blur only removes energy *along* its direction, so
    the masked ratio is computed for 4 gradient orientations (0/45/90/135°)
    and the **minimum** is kept. This was added after the first run showed
    motion blur at only 21% val accuracy with the isotropic ratio.

The normalised classical method picks automatically whichever candidate has the
best **train** macro-F1. On the dummy data that is `reblur_dir`.

**Failure-case figure.** I had no real photos of plain walls or textured blurry
scenes yet, so the demo images are **synthesised** (smooth gradients with a few
sharp lines, a flat document, a low-contrast scene, and blurred procedural
gravel/foliage textures):

![failure case](results/failure_case_before_after.png)

### Phase 4: classical baseline (`src/baseline.py`)
* Thresholds `t1 < t2` are found by exhaustive grid search over 80 quantiles of
  the feature on the **synthetic train split**, maximising macro-F1.
* Exposure: a multinomial logistic regression on the luminance features gives
  P(bad, borderline, good). The score is e = 0·P(bad) + 0.5·P(bl) + 1·P(good),
  then two thresholds on e are grid-searched the same way.
* **Scores in [0, 1]:** a piecewise-linear map through (train 1st pct → 0),
  (t1 → 1/3), (t2 → 2/3), (train 99th pct → 1). So score < 1/3 always means
  "bad" and score > 2/3 always means "good".
* **Overall quality Q = √(s_sharp · s_exposure)**, a geometric mean: a photo
  that is bad on *either* axis ranks low. The CNN uses the same formula, with
  s = expected class value.

### Phase 5: CNN (`src/cnn.py`)
MobileNetV3-Small (ImageNet) with its last layer replaced by a 6-output head
(3 sharpness + 3 exposure logits) and a class-weighted CE loss on each head.
Input is a **224 crop of the 512 px image, not a resize**. Shrinking the whole
photo to 224 would erase the fine blur we want to detect. Augmentation is only
random crop and h/v flips; there is no blur or brightness augmentation, because
that would corrupt the labels. Test time averages 5 crops. Training schedule:
2 epochs with a frozen backbone, then 4 epochs with the last 4 blocks unfrozen
(LR / 5). The best epoch by val macro-F1 is saved.

### Phase 6: agreement (`src/agreement.py`)
Percent agreement, plus Cohen's κ unweighted and quadratic-weighted, on the
images that both raters labelled.

### Phase 7: evaluation (`src/evaluate.py`)
Uses the **real labelled set only**. Timing is end to end per image (JPEG
decode + resize + features or CNN forward), measured after 3 warm-up images.
The "delete?" decision means *bad on either axis*.

### Phase 8: app (`app.py`)
* **Triage:** folder path, method (classical-normalised / CNN), scan with a
  progress bar (per-file `st.cache_data`, so re-scans are instant), a
  worst-first thumbnail grid with coloured badges, one expander per duplicate
  group with the best photo marked **KEEP**, filters (only bad / only
  duplicates), checkboxes plus "select all bad" and "select dups", then
  **Move to `_trash/`** → confirm. In the sidebar, the Trash panel offers
  *Undo last move*, *Restore all*, and *Restore selected*. A
  `_trash/manifest.csv` remembers original locations, so undo works after a
  restart.
* **Label:** one photo at a time, rater-name field, and a choice of target CSV
  (rater 1 → `real_labels.csv`, rater 2 → `rater2_labels.csv`). The CSV is
  saved on **every click**, and the app moves to the next photo once both
  labels are set. Keys: `1/2/3` sharpness good/borderline/bad, `Q/W/E`
  exposure, `→`/`N` next, `←`/`B` back. Tab + Enter also works.

---

## Results

All numbers come from the **dummy** set: 65 procedurally generated "real" photos,
labelled by the degradation rules. Re-run on your own photos before quoting them.

**Test set (data/raw, never used for training or thresholds)**

| method | sharp_macro_F1 | exp_macro_F1 | mean_macro_F1 | sharp_bad_P | sharp_bad_R | delete_P | delete_R | ms_mean | ms_p95 |
|---|---|---|---|---|---|---|---|---|---|
| classical_raw | 0.427 | 0.481 | 0.454 | 0.344 | 1.000 | 0.462 | 0.947 | 12.791 | 16.562 |
| classical_norm | 0.646 | 0.481 | 0.563 | 0.800 | 0.727 | 0.536 | 0.789 | 14.819 | 19.036 |
| cnn | 0.604 | 0.333 | 0.469 | 0.533 | 0.727 | 0.429 | 0.789 | 187.680 | 190.759 |

`delete_P/R` is the precision/recall of the "bad on either axis → candidate
for deletion" decision. The full table (accuracy and macro P/R per task) is in
[results/comparison_table.md](results/comparison_table.md); per-class numbers
are in `results/per_class.csv`.

![accuracy vs speed](results/accuracy_vs_speed.png)

Confusion matrices: [raw](results/confusion_classical_raw.png) ·
[normalised](results/confusion_classical_norm.png) · [CNN](results/confusion_cnn.png)

**Synthetic val split: sharpness metric comparison (Phase 3/4)**

| feature | train macro-F1 | val macro-F1 |
|---|---|---|
| lap_var | 0.509 | 0.345 |
| tenengrad | 0.450 | 0.375 |
| masked_lap | 0.517 | 0.438 |
| masked_ten | 0.421 | 0.186 |
| reblur | 0.665 | 0.578 |
| reblur_masked | 0.684 | 0.582 |
| reblur_dir | 0.724 | 0.625 |
| fft_hf | 0.596 | 0.474 |
| logreg(lum stats) | 0.594 | 0.497 |

**Inter-rater agreement (dummy rater 2, 30 images):** sharpness κ = 0.737 (quadratic 0.872),
exposure κ = 0.934 (quadratic 0.963).

**Reading it honestly (dummy data):**
* The content-aware fix is the clearest win. Sharpness macro-F1 rises from
  0.43 to 0.65 for about +2 ms per image. The raw Laplacian flags every
  low-contrast or flat photo as blurry (bad-precision 0.34); the normalised
  metric reaches 0.80.
* The CNN is about **14× slower** (~190 ms per image on CPU, 5 crops) and, on
  this dummy set, no more accurate. Its exposure head generalises badly from
  synthetic to "real" images (F1 0.33). With 25 training sources it overfits
  the source images' brightness.
* Motion blur on highly textured scenes is still the hardest case for every
  classical metric (see the failure-case figure).
* The CNN trained in 11.9 min on an 8-core CPU (6 epochs, best epoch 3).


---

## Using your real photos

1. `python -m src.make_dummy --clean`. This removes every `dummy_*` image and
   dummy label row, and nothing else.
2. Put 30+ **sharp, well-exposed** photos (varied content: walls, documents,
   foliage, people) in `data/sharp_source/`.
3. Put your real phone photos in `data/raw/` (JPEG/PNG; convert HEIC first).
4. `streamlit run app.py` → **Label** page. Label everything as rater 1, and ask
   a friend to label a subset (30–50) as rater 2.
5. `python run_all.py` (about 10–15 min, mostly the CNN). Then copy the new
   `results/comparison_table.md` into this README.

## Limitations (be ready for these in the viva)
* Analysis happens at 512 px. Slight blur on a 12 MP photo can vanish at that
  scale. That roughly matches "does it look blurry on a phone screen", but it
  is a design choice (`ANALYSIS_SIDE`).
* Synthetic exposure labels are *relative* to the source image, so a naturally
  dark night photo looks "under-exposed" to absolute luminance statistics.
* Synthetic blur is spatially uniform. Real photos have depth-of-field blur
  (sharp subject, blurred background), which the texture mask partly handles
  by looking at the most detailed tiles.
* The val split has few sources, so the synthetic val numbers are noisy.
