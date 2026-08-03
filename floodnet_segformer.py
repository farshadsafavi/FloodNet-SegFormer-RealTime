"""Reusable SegFormer training and evaluation utilities for FloodNet.

This is a cleaned reimplementation of the experimental SegFormer notebooks
used alongside Safavi and Rahnemoonfar (IEEE JSTARS, 2023).
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from transformers import SegformerForSemanticSegmentation


FLOODNET_CLASSES = (
    "background",
    "building-flooded",
    "building-non-flooded",
    "road-flooded",
    "road-non-flooded",
    "water",
    "tree",
    "vehicle",
    "pool",
    "grass",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    image_dir: str
    mask_dir: str
    output_dir: str = "outputs/segformer-b3"
    model_name: str = "nvidia/mit-b3"
    height: int = 704
    width: int = 1056
    batch_size: int = 1
    epochs: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    seed: int = 19
    num_classes: int = 10


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_ids(image_dir: str | Path) -> list[str]:
    image_dir = Path(image_dir)
    ids = sorted(p.stem for p in image_dir.glob("*.jpg"))
    if not ids:
        raise FileNotFoundError(f"No .jpg images found in {image_dir}")
    return ids


def split_ids(ids: Iterable[str], seed: int = 19) -> tuple[list[str], list[str], list[str]]:
    """Reproduce the notebook split: 10% test, then 15% of the remainder validation."""
    ids = list(ids)
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)
    n_test = max(1, int(np.ceil(0.10 * len(ids))))
    test_ids = ids[:n_test]
    trainval = ids[n_test:]
    n_val = max(1, int(np.ceil(0.15 * len(trainval))))
    return trainval[n_val:], trainval[:n_val], test_ids


class FloodNetDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        ids: Iterable[str],
        size: tuple[int, int] = (704, 1056),
        augment: bool = False,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.ids = list(ids)
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_id = self.ids[index]
        image_path = self.image_dir / f"{sample_id}.jpg"
        mask_path = self.mask_dir / f"{sample_id}_lab.png"
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        if self.augment and random.random() < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
        if self.augment and random.random() < 0.5:
            image, mask = TF.vflip(image), TF.vflip(mask)

        image_tensor = TF.normalize(TF.to_tensor(image), IMAGENET_MEAN, IMAGENET_STD)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.int64).copy())
        return image_tensor, mask_tensor


def build_model(model_name: str = "nvidia/mit-b3", num_classes: int = 10) -> nn.Module:
    id2label = {i: name for i, name in enumerate(FLOODNET_CLASSES[:num_classes])}
    label2id = {name: i for i, name in id2label.items()}
    return SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def _logits(model: nn.Module, images: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    logits = model(pixel_values=images).logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


class ConfusionMatrix:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        predictions = logits.argmax(dim=1).detach().cpu().flatten()
        targets = targets.detach().cpu().flatten()
        valid = (targets >= 0) & (targets < self.num_classes)
        bins = self.num_classes * targets[valid] + predictions[valid]
        self.matrix += torch.bincount(
            bins, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, object]:
        matrix = self.matrix.float()
        intersection = matrix.diag()
        union = matrix.sum(1) + matrix.sum(0) - intersection
        present = union > 0
        per_class = torch.full((self.num_classes,), torch.nan)
        per_class[present] = intersection[present] / union[present]
        return {
            "pixel_accuracy": (intersection.sum() / matrix.sum().clamp_min(1)).item(),
            "mean_iou": torch.nanmean(per_class).item(),
            "per_class_iou": per_class.tolist(),
        }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model.eval()
    meter = ConfusionMatrix(model.config.num_labels)
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = _logits(model, images, masks.shape[-2:])
        total_loss += F.cross_entropy(logits, masks).item()
        meter.update(logits, masks)
    metrics = meter.compute()
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


def train(config: TrainConfig) -> dict[str, list[dict[str, object]]]:
    seed_everything(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ids, val_ids, test_ids = split_ids(discover_ids(config.image_dir), config.seed)
    size = (config.height, config.width)
    train_set = FloodNetDataset(config.image_dir, config.mask_dir, train_ids, size, True)
    val_set = FloodNetDataset(config.image_dir, config.mask_dir, val_ids, size, False)
    train_loader = DataLoader(train_set, config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_set, config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_model(config.model_name, config.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, config.learning_rate, epochs=config.epochs, steps_per_epoch=len(train_loader)
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(json.dumps({"train": train_ids, "val": val_ids, "test": test_ids}, indent=2))
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2))

    history: dict[str, list[dict[str, object]]] = {"train": [], "val": []}
    best_miou = -1.0
    for epoch in range(config.epochs):
        model.train()
        meter = ConfusionMatrix(config.num_classes)
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = _logits(model, images, masks.shape[-2:])
            loss = F.cross_entropy(logits, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            meter.update(logits, masks)
        train_metrics = meter.compute()
        train_metrics["loss"] = running_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        print(f"Epoch {epoch + 1:02d}/{config.epochs}: train mIoU={train_metrics['mean_iou']:.4f}, val mIoU={val_metrics['mean_iou']:.4f}")
        if float(val_metrics["mean_iou"]) > best_miou:
            best_miou = float(val_metrics["mean_iou"])
            model.save_pretrained(output_dir / "best")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train SegFormer on FloodNet")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/segformer-b3")
    parser.add_argument("--model-name", default="nvidia/mit-b3")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=704)
    parser.add_argument("--width", type=int, default=1056)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
