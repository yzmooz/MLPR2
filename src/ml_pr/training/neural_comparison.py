from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_pr.models.gated_sklearn import _fit_binary_classifier
from ml_pr.training.evaluate import binary_metrics
from ml_pr.training.external_torchxrayvision_baseline import _load_image_for_xrv, predict_torchxrayvision_scores


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", value).strip("_")


def _fingerprint(values: list[str]) -> str:
    digest = hashlib.sha1()
    for value in values:
        digest.update(value.encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


def resolve_device_name(device: str, torch_module=None) -> str:
    if device != "auto":
        return device
    if torch_module is None:
        import torch as torch_module
    return "cuda" if torch_module.cuda.is_available() else "cpu"


def _progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message, flush=True)


def masked_mean_pool(hidden_states: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    # усредняем только настоящие токены, padding в вектор не попадает
    mask = attention_mask.astype(np.float32)[..., None]
    summed = (hidden_states.astype(np.float32) * mask).sum(axis=1)
    counts = np.maximum(mask.sum(axis=1), 1.0)
    return summed / counts


def _fit_scaled_classifier(features: np.ndarray, y: np.ndarray, random_state: int):
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(np.asarray(features, dtype=np.float32))
    classifier = _fit_binary_classifier(x_scaled, y, random_state=random_state)
    return scaler, classifier


def _predict_scaled(scaler: StandardScaler, classifier, features: np.ndarray) -> np.ndarray:
    x_scaled = scaler.transform(np.asarray(features, dtype=np.float32))
    return classifier.predict_proba(x_scaled)[:, 1]


def _fusion_features(text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
    return np.hstack([np.asarray(text_features, dtype=np.float32), np.asarray(image_features, dtype=np.float32)])


def _gate_features(
    text_features: np.ndarray,
    image_features: np.ndarray,
    text_prob: np.ndarray,
    image_prob: np.ndarray,
) -> np.ndarray:
    text_norm = np.linalg.norm(text_features, axis=1)
    image_norm = np.linalg.norm(image_features, axis=1)
    return np.column_stack(
        [
            text_prob,
            image_prob,
            np.abs(image_prob - text_prob),
            np.maximum(text_prob, 1.0 - text_prob),
            np.maximum(image_prob, 1.0 - image_prob),
            text_norm,
            image_norm,
        ]
    ).astype(np.float32)


def _base_meta_indices(y: np.ndarray, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    if len(y) < 8 or min(np.bincount(y.astype(int), minlength=2)) < 2:
        return indices, indices
    try:
        base_idx, meta_idx = train_test_split(indices, test_size=0.25, random_state=random_state, stratify=y)
        return np.asarray(base_idx), np.asarray(meta_idx)
    except ValueError:
        return indices, indices


def fit_binary_feature_models(
    train_text_features: np.ndarray,
    train_image_features: np.ndarray,
    train_y: np.ndarray,
    test_text_features: np.ndarray,
    test_image_features: np.ndarray,
    test_y: np.ndarray,
    random_state: int = 42,
) -> dict[str, object]:
    train_y = np.asarray(train_y).astype(int)
    test_y = np.asarray(test_y).astype(int)

    text_scaler, text_classifier = _fit_scaled_classifier(train_text_features, train_y, random_state=random_state)
    image_scaler, image_classifier = _fit_scaled_classifier(train_image_features, train_y, random_state=random_state)
    fusion_scaler, fusion_classifier = _fit_scaled_classifier(
        _fusion_features(train_text_features, train_image_features),
        train_y,
        random_state=random_state,
    )

    text_scores = _predict_scaled(text_scaler, text_classifier, test_text_features)
    image_scores = _predict_scaled(image_scaler, image_classifier, test_image_features)
    early_scores = _predict_scaled(fusion_scaler, fusion_classifier, _fusion_features(test_text_features, test_image_features))

    base_idx, meta_idx = _base_meta_indices(train_y, random_state=random_state)
    gate_text_scaler, gate_text_classifier = _fit_scaled_classifier(
        train_text_features[base_idx],
        train_y[base_idx],
        random_state=random_state,
    )
    gate_image_scaler, gate_image_classifier = _fit_scaled_classifier(
        train_image_features[base_idx],
        train_y[base_idx],
        random_state=random_state,
    )
    meta_text_scores = _predict_scaled(gate_text_scaler, gate_text_classifier, train_text_features[meta_idx])
    meta_image_scores = _predict_scaled(gate_image_scaler, gate_image_classifier, train_image_features[meta_idx])
    gate_target = (np.abs(meta_image_scores - train_y[meta_idx]) <= np.abs(meta_text_scores - train_y[meta_idx])).astype(int)
    gate_scaler, gate_classifier = _fit_scaled_classifier(
        _gate_features(train_text_features[meta_idx], train_image_features[meta_idx], meta_text_scores, meta_image_scores),
        gate_target,
        random_state=random_state,
    )

    gated_text_scores = _predict_scaled(gate_text_scaler, gate_text_classifier, test_text_features)
    gated_image_scores = _predict_scaled(gate_image_scaler, gate_image_classifier, test_image_features)
    gate_x = gate_scaler.transform(_gate_features(test_text_features, test_image_features, gated_text_scores, gated_image_scores))
    image_weights = gate_classifier.predict_proba(gate_x)[:, 1]
    gated_scores = image_weights * gated_image_scores + (1.0 - image_weights) * gated_text_scores

    result = {
        "metrics": {
            "text_neural": binary_metrics(test_y, text_scores),
            "image_neural": binary_metrics(test_y, image_scores),
            "early_fusion": binary_metrics(test_y, early_scores),
            "gated_fusion": binary_metrics(test_y, gated_scores),
        },
        "mean_modality_weights": {
            "text": float((1.0 - image_weights).mean()),
            "image": float(image_weights.mean()),
        },
    }
    return _json_ready(result)


def extract_transformer_text_features(
    texts: list[str],
    model_name: str,
    batch_size: int = 16,
    max_length: int = 128,
    device: str = "cpu",
    show_progress: bool = False,
) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = resolve_device_name(device, torch_module=torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    # здесь ClinicalBERT работает как готовый экстрактор признаков
    model.eval()

    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            _progress(show_progress, f"ClinicalBERT text batch {min(start + len(batch_texts), len(texts))}/{len(texts)}")
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            pooled = masked_mean_pool(
                outputs.last_hidden_state.detach().cpu().numpy(),
                encoded["attention_mask"].detach().cpu().numpy(),
            )
            batches.append(pooled.astype(np.float32))
    return np.vstack(batches)


def extract_torchxrayvision_image_embeddings(
    image_paths: list[str],
    weights: str = "densenet121-res224-all",
    batch_size: int = 16,
    image_size: int = 224,
    device: str = "cpu",
    show_progress: bool = False,
) -> tuple[np.ndarray, list[str]]:
    import torch
    import torchxrayvision as xrv

    device = resolve_device_name(device, torch_module=torch)
    model = xrv.models.DenseNet(weights=weights)
    model = model.to(device)
    # готовая DenseNet уже обучена на рентгенах, здесь ее веса не обновляются
    model.eval()
    pathologies = list(model.pathologies)

    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            _progress(
                show_progress,
                f"TorchXRayVision image batch {min(start + len(batch_paths), len(image_paths))}/{len(image_paths)}",
            )
            batch = torch.stack([_load_image_for_xrv(path, image_size=image_size) for path in batch_paths]).to(device)
            # берем внутренний вектор DenseNet, а не ее готовый список вероятностей
            outputs = model.features2(batch).detach().cpu().numpy()
            batches.append(outputs.astype(np.float32))

    return np.vstack(batches), pathologies


def _load_or_create_text_features(
    metadata: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    show_progress: bool,
) -> np.ndarray:
    from ml_pr.features.text import clean_indication

    texts = [clean_indication(value) for value in metadata["indication"].tolist()]
    cache_path = output_dir / f"text_features_{_safe_name(model_name)}_{_fingerprint(texts)}.npz"
    # кэш позволяет не прогонять все тексты через ClinicalBERT при каждом запуске
    if cache_path.exists():
        cached = np.load(cache_path)
        features = cached["features"]
        if len(features) == len(metadata):
            return features

    features = extract_transformer_text_features(
        texts=texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        show_progress=show_progress,
    )
    np.savez_compressed(cache_path, features=features, model_name=model_name, max_length=max_length)
    return features


def _load_or_create_image_features(
    metadata: pd.DataFrame,
    output_dir: Path,
    weights: str,
    batch_size: int,
    device: str,
    feature_kind: str,
    show_progress: bool,
) -> tuple[np.ndarray, list[str]]:
    image_paths = metadata["image_path"].astype(str).tolist()
    cache_path = output_dir / f"image_{feature_kind}_{_safe_name(weights)}_{_fingerprint(image_paths)}.npz"
    # признаки снимков считаются дольше голов, поэтому их тоже сохраняем в кэш
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        features = cached["features"]
        pathologies = [str(item) for item in cached["pathologies"].tolist()]
        if len(features) == len(metadata):
            return features, pathologies

    if feature_kind == "embedding":
        features, pathologies = extract_torchxrayvision_image_embeddings(
            image_paths,
            weights=weights,
            batch_size=batch_size,
            device=device,
            show_progress=show_progress,
        )
    elif feature_kind == "probabilities":
        features, pathologies = predict_torchxrayvision_scores(
            image_paths,
            weights=weights,
            batch_size=batch_size,
            device=resolve_device_name(device),
        )
    else:
        raise ValueError("feature_kind must be `embedding` or `probabilities`")
    features = features.astype(np.float32)
    np.savez_compressed(
        cache_path,
        features=features,
        pathologies=np.asarray(pathologies, dtype=object),
        weights=weights,
        feature_kind=feature_kind,
    )
    return features, pathologies


def _classical_reference(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("binary", {}).get("metrics")


def run_neural_comparison(
    metadata_csv: Path,
    output_dir: Path,
    text_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    image_weights: str = "densenet121-res224-all",
    image_feature_kind: str = "embedding",
    text_batch_size: int = 16,
    image_batch_size: int = 16,
    max_length: int = 128,
    device: str = "cpu",
    random_state: int = 42,
    output_json: Path | None = None,
    show_progress: bool = False,
) -> dict[str, object]:
    metadata = pd.read_csv(metadata_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_features = _load_or_create_text_features(
        metadata=metadata,
        output_dir=output_dir,
        model_name=text_model,
        batch_size=text_batch_size,
        max_length=max_length,
        device=device,
        show_progress=show_progress,
    )
    image_features, pathologies = _load_or_create_image_features(
        metadata=metadata,
        output_dir=output_dir,
        weights=image_weights,
        batch_size=image_batch_size,
        device=device,
        feature_kind=image_feature_kind,
        show_progress=show_progress,
    )

    train_mask = metadata["split"].isin(["train", "val"]).to_numpy()
    test_mask = (metadata["split"] == "test").to_numpy()
    train_y = metadata.loc[train_mask, "is_abnormal"].to_numpy().astype(int)
    test_y = metadata.loc[test_mask, "is_abnormal"].to_numpy().astype(int)

    neural_result = fit_binary_feature_models(
        train_text_features=text_features[train_mask],
        train_image_features=image_features[train_mask],
        train_y=train_y,
        test_text_features=text_features[test_mask],
        test_image_features=image_features[test_mask],
        test_y=test_y,
        random_state=random_state,
    )

    result: dict[str, object] = {
        "metadata_csv": str(metadata_csv),
        "rows": {"train_val": int(train_mask.sum()), "test": int(test_mask.sum())},
        "encoders": {
            "text": text_model,
            "image": f"TorchXRayVision:{image_weights}",
            "image_feature_kind": image_feature_kind,
            "image_outputs": pathologies,
        },
        "neural": neural_result,
        "classical_reference": _classical_reference(Path("outputs/final/final_metrics.json")),
    }
    result = _json_ready(result)

    output_path = output_json or output_dir / "neural_comparison_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen neural text/image encoders with classical baselines.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/iu_xray_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/neural"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--text-model", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--image-weights", default="densenet121-res224-all")
    parser.add_argument("--image-feature-kind", choices=["embedding", "probabilities"], default="embedding")
    parser.add_argument("--text-batch-size", type=int, default=16)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true", help="Hide feature extraction progress messages.")
    args = parser.parse_args()

    result = run_neural_comparison(
        metadata_csv=args.metadata,
        output_dir=args.output_dir,
        text_model=args.text_model,
        image_weights=args.image_weights,
        image_feature_kind=args.image_feature_kind,
        text_batch_size=args.text_batch_size,
        image_batch_size=args.image_batch_size,
        max_length=args.max_length,
        device=args.device,
        random_state=args.seed,
        output_json=args.out,
        show_progress=not args.quiet,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
