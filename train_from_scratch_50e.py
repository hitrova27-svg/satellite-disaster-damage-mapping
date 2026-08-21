"""Train the from-scratch Siamese CNN on the exact pretrained-model patch arrays.

This is the comparison baseline for the project.  It deliberately reuses the
same manifest, pre/post arrays, augmentation, class weights, split boundaries,
and metrics as the pretrained ResNet-18 experiment.  The only substantive
change is the randomly initialized compact CNN encoder.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


CLASSES = ["no-damage", "minor-damage", "major-damage", "destroyed"]
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PatchPairDataset(Dataset):
    """Memory-mapped pre/post pairs with identical augmentation per branch."""

    def __init__(self, manifest: pd.DataFrame, patch_dir: Path, augment: bool = False):
        self.rows = manifest["row"].to_numpy(dtype=np.int64)
        self.labels = manifest["label"].to_numpy(dtype=np.int64)
        self.pre_path = patch_dir / "pre.npy"
        self.post_path = patch_dir / "post.npy"
        self.augment = augment
        self._pre = None
        self._post = None

    def __len__(self) -> int:
        return len(self.rows)

    def _arrays(self):
        if self._pre is None:
            self._pre = np.load(self.pre_path, mmap_mode="r")
            self._post = np.load(self.post_path, mmap_mode="r")
        return self._pre, self._post

    @staticmethod
    def _normalize(image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

    def __getitem__(self, index: int):
        pre_array, post_array = self._arrays()
        row = self.rows[index]
        pre = np.asarray(pre_array[row])
        post = np.asarray(post_array[row])

        if self.augment:
            if np.random.rand() < 0.5:
                pre, post = pre[:, ::-1], post[:, ::-1]
            if np.random.rand() < 0.5:
                pre, post = pre[::-1], post[::-1]
            rotations = np.random.randint(4)
            if rotations:
                pre, post = np.rot90(pre, rotations), np.rot90(post, rotations)
            if np.random.rand() < 0.3:
                gain = np.float32(np.random.uniform(0.85, 1.15))
                pre = np.clip(pre.astype(np.float32) * gain, 0, 255).astype(np.uint8)
                post = np.clip(post.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        return self._normalize(pre), self._normalize(post), int(self.labels[index])


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )


class SiameseScratchCNN(nn.Module):
    """Shared compact CNN + classifier over [pre, post, absolute difference]."""

    def __init__(self, num_classes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        joined = torch.cat(
            [pre_features, post_features, torch.abs(pre_features - post_features)], dim=1
        )
        return self.head(joined)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp_enabled):
    model.eval()
    total_loss = 0.0
    count = 0
    true_parts = []
    pred_parts = []
    probability_parts = []
    for pre, post, labels in loader:
        pre = pre.to(device, non_blocking=True)
        post = post.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(pre, post)
            loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        count += labels.size(0)
        true_parts.append(labels.cpu().numpy())
        pred_parts.append(logits.argmax(1).cpu().numpy())
        probability_parts.append(torch.softmax(logits.float(), 1).cpu().numpy())
    return (
        total_loss / max(count, 1),
        np.concatenate(true_parts),
        np.concatenate(pred_parts),
        np.concatenate(probability_parts),
    )


def save_curves(history: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history.epoch, history.train_loss, label="train")
    axes[0].plot(history.epoch, history.val_loss, label="val")
    axes[0].set(title="Loss", xlabel="epoch")
    axes[0].legend()
    axes[1].plot(history.epoch, history.val_acc, color="darkorange")
    axes[1].set(title="Validation accuracy", xlabel="epoch")
    axes[2].plot(history.epoch, history.val_f1, color="seagreen")
    axes[2].set(title="Validation macro-F1", xlabel="epoch")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--patch-dir", type=Path, default=Path("work-pretrained-50e/patches")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("work-scratch-50e"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    manifest_path = args.patch_dir / "manifest.csv"
    for required in [manifest_path, args.patch_dir / "pre.npy", args.patch_dir / "post.npy"]:
        if not required.is_file():
            raise FileNotFoundError(required)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")

    manifest = pd.read_csv(manifest_path)
    expected_splits = {"train", "val", "test_id", "test_ood"}
    if set(manifest.split.unique()) != expected_splits:
        raise RuntimeError("Manifest does not contain the four expected splits")

    datasets = {
        split: PatchPairDataset(
            manifest[manifest.split == split], args.patch_dir, augment=(split == "train")
        )
        for split in ["train", "val", "test_id", "test_ood"]
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }
    print("split sizes:", {split: len(dataset) for split, dataset in datasets.items()})

    counts = np.bincount(datasets["train"].labels, minlength=len(CLASSES)).astype(np.float64)
    weights = counts.sum() / (len(CLASSES) * np.maximum(counts, 1))
    class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print("train class counts:", counts.astype(int).tolist())
    print("loss weights:", np.round(weights, 3).tolist())

    model = SiameseScratchCNN(num_classes=len(CLASSES)).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"parameters={parameter_count:,} trainable={trainable_count:,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history = []
    start_epoch = 1
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    if args.resume and last_path.is_file():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        history = state["history"]
        start_epoch = state["epoch"] + 1
        best_f1 = state["best_f1"]
        best_epoch = state["best_epoch"]
        stale = state["stale"]
        print(f"resuming after epoch {state['epoch']}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        running_loss = 0.0
        seen = 0
        for pre, post, labels in loaders["train"]:
            pre = pre.to(device, non_blocking=True)
            post = post.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss = criterion(model(pre, post), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)
        scheduler.step()

        val_loss, y_true, y_pred, _ = evaluate(
            model, loaders["val"], criterion, device, amp_enabled
        )
        val_acc = float((y_true == y_pred).mean())
        val_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        record = {
            "epoch": epoch,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - start,
        }
        history.append(record)

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "arch": "SiameseScratchCNN-32-64-128-256",
                    "classes": CLASSES,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_f1,
                    "parameter_count": parameter_count,
                },
                best_path,
            )
        else:
            stale += 1

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "best_f1": best_f1,
                "best_epoch": best_epoch,
                "stale": stale,
            },
            last_path,
        )
        history_frame = pd.DataFrame(history)
        history_frame.to_csv(args.output_dir / "training_history.csv", index=False)
        save_curves(history_frame, args.output_dir)
        marker = " <- best" if improved else ""
        print(
            f"epoch {epoch:02d}/{args.epochs} train={record['train_loss']:.4f} "
            f"val={val_loss:.4f} acc={val_acc:.4f} macroF1={val_f1:.4f} "
            f"time={record['seconds']:.1f}s{marker}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"early stop after {args.patience} stale epochs", flush=True)
            break

    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["state_dict"])
    print(
        f"restored best epoch={best['best_epoch']} "
        f"val_macro_f1={best['best_val_macro_f1']:.4f}"
    )

    overall_rows = []
    per_class_frames = []
    confusion_frames = []
    for split in ["val", "test_id", "test_ood"]:
        _, y_true, y_pred, probabilities = evaluate(
            model, loaders[split], criterion, device, amp_enabled
        )
        accuracy = float((y_true == y_pred).mean())
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        try:
            auc = float(
                roc_auc_score(y_true, probabilities, multi_class="ovr", average="macro")
            )
        except ValueError:
            auc = float("nan")
        overall_rows.append(
            {
                "model": "from_scratch_siamese_cnn",
                "split": split,
                "n": len(y_true),
                "accuracy": round(accuracy, 4),
                "macro_F1": round(macro_f1, 4),
                "weighted_F1": round(weighted_f1, 4),
                "macro_ROC_AUC": round(auc, 4),
            }
        )
        precision, recall, per_f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=range(len(CLASSES)), zero_division=0
        )
        per_class_frames.append(
            pd.DataFrame(
                {
                    "model": "from_scratch_siamese_cnn",
                    "split": split,
                    "class": CLASSES,
                    "precision": precision,
                    "recall": recall,
                    "f1": per_f1,
                    "support": support,
                }
            )
        )
        matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
        confusion_frames.append(
            pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)
            .rename_axis("true_class")
            .reset_index()
            .assign(split=split, model="from_scratch_siamese_cnn")
        )

    overall = pd.DataFrame(overall_rows)
    overall.to_csv(args.output_dir / "results_overall.csv", index=False)
    pd.concat(per_class_frames, ignore_index=True).to_csv(
        args.output_dir / "results_per_class.csv", index=False
    )
    pd.concat(confusion_frames, ignore_index=True).to_csv(
        args.output_dir / "confusion_matrices.csv", index=False
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "patch_dir": str(args.patch_dir),
                "output_dir": str(args.output_dir),
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                "parameter_count": parameter_count,
                "best_epoch": best_epoch,
                "best_val_macro_f1": best_f1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
