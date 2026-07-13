from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from ml_pr.data.dataset import IUXRayDataset
from ml_pr.training.evaluate import binary_metrics, multilabel_metrics


def _require_deep_learning_dependencies():
    try:
        import torch
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Deep training requires torch, torchvision and transformers. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    return torch, DataLoader, transforms, AutoTokenizer


def _resolve_label_columns(metadata_csv: Path, task: str) -> list[str]:
    metadata = pd.read_csv(metadata_csv, nrows=5)
    if task == "binary":
        return ["is_abnormal"]
    return [column for column in metadata.columns if column.startswith("label_")]


def _collate_batch(batch, tokenizer, torch, max_length: int):
    texts = [sample["text"] for sample in batch]
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    images = torch.stack([sample["image"] for sample in batch])
    labels = torch.tensor([sample["labels"] for sample in batch], dtype=torch.float32)
    return {
        "pixel_values": images,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels,
    }


def _move_batch(batch: dict, device):
    return {key: value.to(device) for key, value in batch.items()}


def train_epoch(model, loader, optimizer, criterion, torch, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        loss = criterion(logits, batch["labels"])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(1, len(loader))


def evaluate_epoch(model, loader, torch, device, task: str, threshold: float) -> dict[str, float]:
    model.eval()
    y_true = []
    y_score = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch(batch, device)
            logits = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            scores = torch.sigmoid(logits).detach().cpu().numpy()
            y_score.append(scores)
            y_true.append(batch["labels"].detach().cpu().numpy())

    import numpy as np

    true = np.vstack(y_true)
    score = np.vstack(y_score)
    if task == "binary":
        return binary_metrics(true.reshape(-1), score.reshape(-1), threshold=threshold)
    return multilabel_metrics(true, score, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    torch, DataLoader, transforms, AutoTokenizer = _require_deep_learning_dependencies()
    from functools import partial

    from ml_pr.models.multimodal import MultimodalCXRModel

    metadata_csv = Path(config["data"]["metadata_csv"])
    task = config["data"].get("task", "binary")
    label_columns = config["data"].get("label_columns") or _resolve_label_columns(metadata_csv, task=task)
    image_size = int(config["data"].get("image_size", 384))
    max_length = int(config["data"].get("max_text_length", 128))
    batch_size = int(config["data"].get("batch_size", 8))
    num_workers = int(config["data"].get("num_workers", 0))
    threshold = float(config["training"].get("threshold", 0.5))
    output_dir = Path(config["training"].get("output_dir", "outputs/binary_gated_fusion"))
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["text_encoder"])
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    train_data = IUXRayDataset(metadata_csv=metadata_csv, split="train", label_columns=label_columns, image_transform=transform)
    val_data = IUXRayDataset(metadata_csv=metadata_csv, split="val", label_columns=label_columns, image_transform=transform)
    collate = partial(_collate_batch, tokenizer=tokenizer, torch=torch, max_length=max_length)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)

    model = MultimodalCXRModel(
        num_labels=len(label_columns),
        image_encoder_name=config["model"]["image_encoder"],
        text_encoder_name=config["model"]["text_encoder"],
        hidden_dim=int(config["model"].get("hidden_dim", 512)),
        dropout=float(config["model"].get("dropout", 0.2)),
        pretrained=bool(config["model"].get("pretrained", True)),
        freeze_backbones=bool(config["model"].get("freeze_backbones", False)),
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.01)),
    )

    best_score = -1.0
    history = []
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, torch, device)
        metrics = evaluate_epoch(model, val_loader, torch, device, task=task, threshold=threshold)
        primary = metrics["roc_auc"] if task == "binary" else metrics["micro_roc_auc"]
        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})
        print(json.dumps(history[-1], ensure_ascii=False))

        if primary == primary and primary > best_score:
            best_score = primary
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_columns": label_columns,
                    "config": config,
                    "best_metric": primary,
                },
                output_dir / "best_model.pt",
            )

    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved training history to {output_dir / 'history.json'}")


if __name__ == "__main__":
    main()
