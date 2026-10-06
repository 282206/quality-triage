"""Run the whole pipeline in order:

    [dummy data] -> synth -> features -> baseline -> failure-case demo
                 -> cnn -> agreement -> evaluate

Usage:
    python run_all.py                 # full run on whatever data is present
    python run_all.py --dummy         # create the dummy dataset first if data/sharp_source is empty
    python run_all.py --quick         # CNN smoke test (tiny subset, 2 epochs)
    python run_all.py --skip-cnn      # classical methods only
"""
from __future__ import annotations

import argparse
import time

from src import agreement, baseline, cnn, evaluate, features, make_dummy, synth, texture_norm
from src.common import SHARP_SOURCE_DIR, list_images, seed_everything


def step(name, fn, *args):
    print(f"\n{'=' * 70}\n>>> {name}\n{'=' * 70}", flush=True)
    t = time.perf_counter()
    fn(*args)
    print(f"<<< {name} done in {time.perf_counter() - t:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dummy", action="store_true", help="generate dummy data if no sharp sources exist")
    ap.add_argument("--quick", action="store_true", help="quick CNN training (smoke test)")
    ap.add_argument("--skip-cnn", action="store_true")
    args = ap.parse_args()
    seed_everything()

    if not list_images(SHARP_SOURCE_DIR):
        if not args.dummy:
            raise SystemExit("data/sharp_source/ is empty. Add photos, or run with --dummy.")
        step("0. dummy dataset", make_dummy.main, [])
    step("1. synthetic degradations", synth.main, [])
    step("2. classical features", features.main, [])
    step("4. classical baseline (thresholds)", baseline.main, [])
    step("3. failure-case demo figure", texture_norm.run_demo)
    if not args.skip_cnn:
        step("5. CNN fine-tuning", cnn.main, ["--quick"] if args.quick else [])
    step("6. inter-rater agreement", agreement.main, [])
    step("7. evaluation on real labelled set", evaluate.main, [])
    print("\nAll done. See results/ and `streamlit run app.py`.")


if __name__ == "__main__":
    main()
