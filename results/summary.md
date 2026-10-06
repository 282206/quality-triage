# Results summary

> **WARNING: these numbers come from the DUMMY test set** (procedurally generated images labelled by rule). They only prove the pipeline runs. Replace with real photos and human labels before quoting any result.

Test set: 65 real labelled images (never used for training or threshold selection).

| method | sharp_macro_F1 | exp_macro_F1 | mean_macro_F1 | sharp_bad_P | sharp_bad_R | delete_P | delete_R | ms_mean | ms_p95 |
|---|---|---|---|---|---|---|---|---|---|
| classical_raw | 0.427 | 0.481 | 0.454 | 0.344 | 1.000 | 0.462 | 0.947 | 12.791 | 16.562 |
| classical_norm | 0.646 | 0.481 | 0.563 | 0.800 | 0.727 | 0.536 | 0.789 | 14.819 | 19.036 |
| cnn | 0.604 | 0.333 | 0.469 | 0.533 | 0.727 | 0.429 | 0.789 | 187.680 | 190.759 |

* **Content-aware fix:** texture-normalised sharpness changes sharpness macro-F1 by +0.219 vs the raw Laplacian (0.427 -> 0.646) at 14.8 vs 12.8 ms/image.
* **Speed/accuracy trade-off:** the CNN takes 187.7 ms/image (p95 190.8), 14.7x the fastest method (Classical (raw Laplacian)), for a mean macro-F1 of 0.469 vs 0.454.
* **Best overall mean macro-F1:** Classical (texture-normalised) (0.563).
* For deletion, look at `delete_P` (of images flagged bad, how many really are - low precision means good photos get binned) and `delete_R` (how many truly bad photos are caught). Because the app only MOVES files to `_trash/`, a recall-leaning operating point is acceptable.

Small test sets give noisy estimates: one image changes accuracy by 1.5 percentage points.
