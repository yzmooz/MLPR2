from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ml_pr.features.text import clean_indication
from ml_pr.training.neural_comparison import masked_mean_pool, resolve_device_name


def _batches(indices: np.ndarray, batch_size: int, shuffle: bool, rng: np.random.Generator):
    values = indices.copy()
    if shuffle:
        rng.shuffle(values)
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def _encode(tokenizer, texts: list[str], indices: np.ndarray, max_length: int, device: str):
    return {
        key: value.to(device)
        for key, value in tokenizer(
            [texts[index] for index in indices],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).items()
    }


def _evaluate(model, tokenizer, texts, labels, indices, batch_size, max_length, device):
    import torch

    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch_indices in _batches(indices, batch_size, shuffle=False, rng=np.random.default_rng(0)):
            encoded = _encode(tokenizer, texts, batch_indices, max_length, device)
            logits = model(**encoded).logits
            probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
    scores = np.concatenate(probabilities)
    truth = labels[indices]
    return {
        "roc_auc": float(roc_auc_score(truth, scores)),
        "pr_auc": float(average_precision_score(truth, scores)),
        "objective": float((roc_auc_score(truth, scores) + average_precision_score(truth, scores)) / 2.0),
    }


def _trainable_encoder_state(model) -> dict[str, object]:
    state = {}
    for name, parameter in model.bert.named_parameters():
        if parameter.requires_grad:
            state[name] = parameter.detach().cpu()
    return state


def _extract_adapted_features(model, tokenizer, texts, batch_size, max_length, device, show_progress):
    import torch

    model.eval()
    features: list[np.ndarray] = []
    indices = np.arange(len(texts))
    with torch.no_grad():
        for batch_number, batch_indices in enumerate(
            _batches(indices, batch_size, shuffle=False, rng=np.random.default_rng(0)), start=1
        ):
            encoded = _encode(tokenizer, texts, batch_indices, max_length, device)
            outputs = model.bert(**encoded)
            pooled = masked_mean_pool(
                outputs.last_hidden_state.cpu().numpy(),
                encoded["attention_mask"].cpu().numpy(),
            )
            features.append(pooled.astype(np.float32))
            if show_progress and batch_number % 20 == 0:
                print(f"Adapted ClinicalBERT features: {min(batch_number * batch_size, len(texts))}/{len(texts)}")
    return np.vstack(features)


def finetune_indication_encoder(
    metadata_csv: Path,
    output_dir: Path,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    epochs: int = 6,
    batch_size: int = 16,
    max_length: int = 128,
    learning_rate: float = 2e-5,
    classifier_learning_rate: float = 1e-4,
    patience: int = 2,
    device: str = "auto",
    random_state: int = 42,
    show_progress: bool = True,
) -> dict[str, object]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    resolved_device = resolve_device_name(device, torch_module=torch)
    metadata = pd.read_csv(metadata_csv)
    texts = [clean_indication(value) for value in metadata["indication"].tolist()]
    labels = metadata["is_abnormal"].to_numpy().astype(np.int64)
    train_indices = np.flatnonzero(metadata["split"].eq("train").to_numpy())
    val_indices = np.flatnonzero(metadata["split"].eq("val").to_numpy())
    if not len(train_indices) or not len(val_indices):
        raise ValueError("metadata must contain train and val rows")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(resolved_device)
    # в этом эксперименте дообучаем только последний слой BERT, весь датасет для полной настройки маловат
    for parameter in model.bert.parameters():
        parameter.requires_grad = False
    for parameter in model.bert.encoder.layer[-1].parameters():
        parameter.requires_grad = True
    for parameter in model.bert.pooler.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        [
            {"params": [parameter for parameter in model.bert.parameters() if parameter.requires_grad], "lr": learning_rate},
            {"params": model.classifier.parameters(), "lr": classifier_learning_rate},
        ],
        weight_decay=0.01,
    )
    # на GPU смешанная точность экономит память и ускоряет обучение
    use_amp = resolved_device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(random_state)
    best_objective = -np.inf
    best_encoder_state = None
    best_classifier_state = None
    best_epoch = 0
    history: list[dict[str, float]] = []
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch_indices in _batches(train_indices, batch_size, shuffle=True, rng=rng):
            encoded = _encode(tokenizer, texts, batch_indices, max_length, resolved_device)
            batch_labels = torch.as_tensor(labels[batch_indices], dtype=torch.long, device=resolved_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = model(**encoded, labels=batch_labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        validation = _evaluate(
            model, tokenizer, texts, labels, val_indices, batch_size, max_length, resolved_device
        )
        validation["epoch"] = epoch
        validation["train_loss"] = float(np.mean(losses))
        history.append(validation)
        if show_progress:
            print(
                f"Epoch {epoch}: loss={validation['train_loss']:.4f}, "
                f"val ROC-AUC={validation['roc_auc']:.4f}, val PR-AUC={validation['pr_auc']:.4f}"
            )
        # сохраняем лучшую эпоху и останавливаемся, когда validation больше не улучшается
        if validation["objective"] > best_objective + 1e-4:
            best_objective = validation["objective"]
            best_encoder_state = _trainable_encoder_state(model)
            best_classifier_state = {name: value.detach().cpu() for name, value in model.classifier.state_dict().items()}
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_encoder_state is None or best_classifier_state is None:
        raise RuntimeError("fine-tuning did not produce a checkpoint")
    model.bert.load_state_dict(best_encoder_state, strict=False)
    model.classifier.load_state_dict(best_classifier_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    delta_path = output_dir / "clinicalbert_indication_delta.pt"
    features_path = output_dir / "clinicalbert_indication_features.npz"
    metrics_path = output_dir / "clinicalbert_indication_metrics.json"
    torch.save(
        {
            "model_name": model_name,
            "max_length": max_length,
            "encoder_state_dict": best_encoder_state,
        },
        delta_path,
    )
    features = _extract_adapted_features(
        model, tokenizer, texts, batch_size, max_length, resolved_device, show_progress
    )
    np.savez_compressed(features_path, features=features)
    result = {
        "model_name": model_name,
        "device": resolved_device,
        "trainable_encoder_layer": 11,
        "best_epoch": best_epoch,
        "best_validation_objective": float(best_objective),
        "history": history,
        "artifacts": {"encoder_delta": str(delta_path), "features": str(features_path)},
    }
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the last ClinicalBERT layer on IU X-ray indications.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/iu_xray_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/neural_final"))
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = finetune_indication_encoder(
        metadata_csv=args.metadata,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        show_progress=not args.quiet,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
