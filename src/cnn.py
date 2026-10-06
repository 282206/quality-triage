"""Phase 5 - MobileNetV3-Small fine-tuned to predict sharpness + exposure.

Architecture
    torchvision MobileNetV3-Small with ImageNet weights. The final 1000-way
    layer is replaced by ONE linear layer with 6 outputs:
        logits[:, 0:3] -> sharpness class (bad, borderline, good)
        logits[:, 3:6] -> exposure  class (bad, borderline, good)
    Loss = CE(sharpness) + CE(exposure), each class-weighted (labels are
    imbalanced: most synthetic images are sharp).
    Scores: s = E[class value] = 0*P(bad) + 0.5*P(borderline) + 1*P(good)
            Q = sqrt(s_sharp * s_exposure)   (same formula as classical)

Input
    The image is loaded at long side 512 (same as the classical features) and
    a 224x224 CROP is taken - NOT a resize of the whole photo, because
    shrinking 512 -> 224 would erase exactly the fine blur we want to detect.
    Train: random crop + horizontal/vertical flips (they don't change blur or
    exposure). NO blur / brightness / colour augmentation: that would silently
    change the ground-truth labels.
    Test : 5 crops (4 corners + centre), class probabilities averaged.

Training schedule (CPU-friendly)
    epochs 1..F   : backbone frozen, only the classifier head trains
    epochs F+1..E : last 4 backbone blocks unfrozen, lower learning rate
    Best epoch by validation macro-F1 (mean of the two heads) is saved.

Run:  python -m src.cnn            (full, ~6 epochs)
      python -m src.cnn --quick    (tiny subset, 2 epochs - smoke test)
"""
from __future__ import annotations

import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.common import (CLASS_TO_IDX, CLASS_VALUE, CLASSES, MODELS_DIR, SEED, SYNTH_DIR,
                        SYNTH_METADATA, load_image, seed_everything)

CROP = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)   # ImageNet normalisation
STD = np.array([0.229, 0.224, 0.225], np.float32)
MODEL_PATH = MODELS_DIR / "cnn_best.pt"

torch.set_num_threads(os.cpu_count() or 4)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_rgb(path) -> np.ndarray:
    img = cv2.cvtColor(load_image(path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if h < CROP or w < CROP:   # pad tiny images (reflect, so no fake edges)
        img = cv2.copyMakeBorder(img, 0, max(0, CROP - h), 0, max(0, CROP - w), cv2.BORDER_REFLECT)
    return img


def to_tensor(crop: np.ndarray) -> torch.Tensor:
    x = (crop.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def five_crops(img: np.ndarray) -> torch.Tensor:
    h, w = img.shape[:2]
    cy, cx = (h - CROP) // 2, (w - CROP) // 2
    coords = [(0, 0), (0, w - CROP), (h - CROP, 0), (h - CROP, w - CROP), (cy, cx)]
    return torch.stack([to_tensor(img[y:y + CROP, x:x + CROP]) for y, x in coords])


class SynthDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, seed: int = SEED):
        self.paths = [SYNTH_DIR / f for f in df.filename]
        self.ys = df.sharp_label.map(CLASS_TO_IDX).to_numpy()
        self.ye = df.exposure_label.map(CLASS_TO_IDX).to_numpy()
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = load_rgb(self.paths[i])
        if self.train:
            h, w = img.shape[:2]
            y, x = self.rng.integers(0, h - CROP + 1), self.rng.integers(0, w - CROP + 1)
            crop = img[y:y + CROP, x:x + CROP]
            if self.rng.random() < 0.5:
                crop = crop[:, ::-1]
            if self.rng.random() < 0.5:
                crop = crop[::-1, :]
            x = to_tensor(crop)
        else:
            x = five_crops(img)                     # (5, 3, 224, 224)
        return x, self.ys[i], self.ye[i]


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def build_model(pretrained: bool = True, allow_scratch: bool = False) -> nn.Module:
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = mobilenet_v3_small(weights=weights)
    except Exception as e:  # network / SSL / disk problems
        if not allow_scratch:
            raise SystemExit(
                "\n*** Could not download the pretrained MobileNetV3-Small ImageNet weights ***\n"
                f"    reason: {e}\n"
                "    Fix your internet connection (weights go to ~/.cache/torch/hub/checkpoints/)\n"
                "    or re-run with --from-scratch to train WITHOUT pretraining (expect much\n"
                "    worse results; you must report this in the write-up).\n")
        print("WARNING: training from scratch (no pretrained weights) as requested.")
        model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 6)
    return model


def set_trainable(model: nn.Module, unfreeze_last_blocks: int) -> None:
    """Freeze the whole backbone, then unfreeze the last N feature blocks."""
    for p in model.features.parameters():
        p.requires_grad = False
    if unfreeze_last_blocks > 0:
        for block in model.features[-unfreeze_last_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True


def train_mode(model: nn.Module, unfrozen_blocks: int) -> None:
    """model.train(), but keep frozen blocks in eval mode so their BatchNorm
    running statistics (learned on ImageNet) are not overwritten."""
    model.train()
    n = len(model.features)
    for i, block in enumerate(model.features):
        if i < n - unfrozen_blocks:
            block.eval()


# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
def probs_from_logits(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    ps = torch.softmax(logits[:, :3], 1).numpy()
    pe = torch.softmax(logits[:, 3:], 1).numpy()
    return ps, pe


@torch.no_grad()
def predict_crops(model: nn.Module, crops: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """crops: (B, 5, 3, H, W) -> per-image class probabilities averaged over crops."""
    b = crops.shape[0]
    ps, pe = probs_from_logits(model(crops.flatten(0, 1)))
    return ps.reshape(b, 5, 3).mean(1), pe.reshape(b, 5, 3).mean(1)


def result_row(ps: np.ndarray, pe: np.ndarray) -> dict:
    s, e = float(ps @ CLASS_VALUE), float(pe @ CLASS_VALUE)
    return {"sharp_pred": CLASSES[int(ps.argmax())], "exposure_pred": CLASSES[int(pe.argmax())],
            "sharp_score": s, "exposure_score": e, "quality": float(np.sqrt(s * e)),
            **{f"p_sharp_{c}": float(p) for c, p in zip(CLASSES, ps)},
            **{f"p_exp_{c}": float(p) for c, p in zip(CLASSES, pe)}}


def load_cnn(path=MODEL_PATH) -> nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - train first with `python -m src.cnn`.")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu")["state_dict"])
    return model.eval()


def predict_path(model: nn.Module, path) -> dict:
    """Single image end-to-end (load + 5 crops + forward). Timed in evaluate.py."""
    ps, pe = predict_crops(model, five_crops(load_rgb(path))[None])
    return result_row(ps[0], pe[0])


def predict_paths(model: nn.Module, paths, progress=None) -> pd.DataFrame:
    rows = []
    for i, p in enumerate(paths):
        rows.append({"path": str(p), **predict_path(model, p)})
        if progress:
            progress(i + 1, len(paths))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(np.float32)
    w = counts.sum() / (3 * np.maximum(counts, 1))
    return torch.tensor(w)


def evaluate_loader(model, loader) -> tuple[float, float]:
    from sklearn.metrics import f1_score

    model.eval()
    ps_all, pe_all, ys_all, ye_all = [], [], [], []
    for x, ys, ye in loader:
        ps, pe = predict_crops(model, x)
        ps_all.append(ps.argmax(1)); pe_all.append(pe.argmax(1))
        ys_all.append(ys.numpy()); ye_all.append(ye.numpy())
    f = lambda a, b: f1_score(np.concatenate(a), np.concatenate(b), average="macro",
                              labels=[0, 1, 2], zero_division=0)
    return f(ys_all, ps_all), f(ye_all, pe_all)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny subset, 2 epochs (smoke test)")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--freeze-epochs", type=int, default=2)
    ap.add_argument("--unfreeze-blocks", type=int, default=4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--from-scratch", action="store_true",
                    help="allow training without pretrained weights if the download fails")
    args = ap.parse_args(argv)

    seed_everything()
    meta = pd.read_csv(SYNTH_METADATA)
    train_df, val_df = meta[meta.split == "train"], meta[meta.split == "val"]
    if args.quick:
        train_df = train_df.sample(n=min(64, len(train_df)), random_state=SEED)
        val_df = val_df.sample(n=min(32, len(val_df)), random_state=SEED)
        args.epochs, args.freeze_epochs = 2, 1
    print(f"Train {len(train_df)} / val {len(val_df)} images, {args.epochs} epochs "
          f"({args.freeze_epochs} frozen), batch {args.batch}, {torch.get_num_threads()} CPU threads")

    g = torch.Generator().manual_seed(SEED)
    train_dl = DataLoader(SynthDataset(train_df, train=True), batch_size=args.batch, shuffle=True,
                          generator=g, num_workers=0)
    val_dl = DataLoader(SynthDataset(val_df, train=False), batch_size=8, num_workers=0)

    model = build_model(pretrained=True, allow_scratch=args.from_scratch)
    ce_s = nn.CrossEntropyLoss(weight=class_weights(train_dl.dataset.ys))
    ce_e = nn.CrossEntropyLoss(weight=class_weights(train_dl.dataset.ye))

    best, history = -1.0, []
    MODELS_DIR.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        n_unfrozen = 0 if frozen else args.unfreeze_blocks
        if epoch == 1 or epoch == args.freeze_epochs + 1:   # (re)build optimiser at phase change
            set_trainable(model, n_unfrozen)
            lr = args.lr if frozen else args.lr / 5
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr,
                                    weight_decay=1e-4)
        train_mode(model, n_unfrozen)
        t0, tot, n = time.perf_counter(), 0.0, 0
        for x, ys, ye in train_dl:
            logits = model(x)
            loss = ce_s(logits[:, :3], ys) + ce_e(logits[:, 3:], ye)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            n += len(x)
        f_s, f_e = evaluate_loader(model, val_dl)
        score = (f_s + f_e) / 2
        history.append(dict(epoch=epoch, frozen=frozen, loss=tot / n, val_f1_sharp=f_s,
                            val_f1_exposure=f_e, val_f1_mean=score))
        mark = ""
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_f1_mean": score,
                        "args": vars(args)}, MODEL_PATH)
            mark = "  <- saved best"
        print(f"epoch {epoch} [{'frozen' if frozen else f'last {n_unfrozen} blocks'}] "
              f"loss {tot / n:.3f}  val macro-F1 sharp {f_s:.3f} exposure {f_e:.3f} "
              f"({time.perf_counter() - t0:.0f}s){mark}")
    pd.DataFrame(history).to_csv(MODELS_DIR / "cnn_history.csv", index=False)
    print(f"Done in {(time.perf_counter() - t_start) / 60:.1f} min. Best val mean macro-F1 {best:.3f} "
          f"-> {MODEL_PATH}")


if __name__ == "__main__":
    main()
