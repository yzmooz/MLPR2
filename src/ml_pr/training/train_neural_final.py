from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml_pr.models.neural_sklearn import NeuralCascadeBinaryClassifier, NeuralEarlyFusionMultilabelClassifier
from ml_pr.training.evaluate import binary_metrics, multilabel_metrics
from ml_pr.training.neural_comparison import _load_or_create_image_features, _load_or_create_text_features


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


def _label_names(label_columns: list[str]) -> list[str]:
    return [column.removeprefix("label_").replace("_", " ") for column in label_columns]


def _classical_reference(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("binary", {}).get("metrics")


def _external_reference(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("binary", {}).get("metrics")


def _metric_delta(candidate: dict[str, float], reference: dict[str, float] | None) -> dict[str, float] | None:
    if reference is None:
        return None
    return {
        name: float(candidate[name] - reference[name])
        for name in ("roc_auc", "pr_auc", "f1")
        if name in candidate and name in reference
    }


def train_neural_fusion_models(
    metadata_csv: Path,
    output_dir: Path,
    cache_dir: Path = Path("outputs/neural"),
    text_model: str = "emilyalsentzer/Bio_ClinicalBERT",
    image_weights: str = "densenet121-res224-all",
    image_feature_kind: str = "embedding",
    text_batch_size: int = 16,
    image_batch_size: int = 16,
    max_length: int = 128,
    device: str = "auto",
    random_state: int = 42,
    show_progress: bool = False,
) -> dict[str, object]:
    metadata = pd.read_csv(metadata_csv)
    label_columns = [column for column in metadata.columns if column.startswith("label_")]
    if not label_columns:
        raise ValueError("metadata must contain top pathology columns with `label_` prefix")

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # в финальной версии энкодеры не обучаются, а только переводят входные данные в признаки
    text_features = _load_or_create_text_features(
        metadata=metadata,
        output_dir=cache_dir,
        model_name=text_model,
        batch_size=text_batch_size,
        max_length=max_length,
        device=device,
        show_progress=show_progress,
    )
    image_features, pathologies = _load_or_create_image_features(
        metadata=metadata,
        output_dir=cache_dir,
        weights=image_weights,
        batch_size=image_batch_size,
        device=device,
        feature_kind=image_feature_kind,
        show_progress=show_progress,
    )

    # параметры подбираются на validation, test до итоговой проверки не используется
    train_mask = (metadata["split"] == "train").to_numpy()
    val_mask = (metadata["split"] == "val").to_numpy()
    train_val_mask = metadata["split"].isin(["train", "val"]).to_numpy()
    test_mask = (metadata["split"] == "test").to_numpy()
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("metadata must contain non-empty train, val and test splits")

    train_y = metadata.loc[train_mask, "is_abnormal"].to_numpy().astype(int)
    val_y = metadata.loc[val_mask, "is_abnormal"].to_numpy().astype(int)
    test_y = metadata.loc[test_mask, "is_abnormal"].to_numpy().astype(int)

    # для текста и снимка обучаются отдельные головы, потом их ответы объединяет каскад
    binary_model = NeuralCascadeBinaryClassifier(random_state=random_state)
    binary_model.fit(
        text_features[train_mask],
        image_features[train_mask],
        train_y,
        text_features[val_mask],
        image_features[val_mask],
        val_y,
    )

    test_sources = binary_model.source_probabilities(text_features[test_mask], image_features[test_mask])
    cascade_scores = binary_model.predict_proba(text_features[test_mask], image_features[test_mask])[:, 1]
    modality_weights = binary_model.modality_weights(text_features[test_mask], image_features[test_mask])

    # после выбора архитектуры top-5 модель обучается на train и validation вместе
    train_y_multi = metadata.loc[train_val_mask, label_columns].to_numpy().astype(int)
    test_y_multi = metadata.loc[test_mask, label_columns].to_numpy().astype(int)
    multilabel_model = NeuralEarlyFusionMultilabelClassifier(
        label_names=_label_names(label_columns),
        random_state=random_state,
    )
    multilabel_model.fit(text_features[train_val_mask], image_features[train_val_mask], train_y_multi)
    multilabel_scores = multilabel_model.predict_proba(text_features[test_mask], image_features[test_mask])

    # все варианты проверяются на одном test split, поэтому сравнение получается честным
    text_test_metrics = binary_metrics(test_y, test_sources["text"])
    image_test_metrics = binary_metrics(
        test_y, test_sources["image"], threshold=binary_model.image_decision_threshold_
    )
    cascade_test_metrics = binary_metrics(
        test_y, cascade_scores, threshold=binary_model.decision_threshold_
    )
    external_reference = _external_reference(Path("outputs/external/torchxrayvision_metrics.json"))

    binary_model_path = output_dir / "binary_neural_fusion.joblib"
    multilabel_model_path = output_dir / "multilabel_neural_fusion.joblib"
    labels_path = output_dir / "labels.json"
    metrics_path = output_dir / "neural_final_metrics.json"

    joblib.dump(binary_model, binary_model_path)
    joblib.dump(multilabel_model, multilabel_model_path)
    labels_payload = {
        "binary_label": "is_abnormal",
        "label_columns": label_columns,
        "pathology_labels": _label_names(label_columns),
        "text_encoder": text_model,
        "image_encoder": f"TorchXRayVision:{image_weights}",
        "image_feature_kind": image_feature_kind,
        "image_outputs": pathologies,
        "binary_architecture": "image_first_log_odds_cascade",
        "decision_threshold": binary_model.decision_threshold_,
    }
    labels_path.write_text(json.dumps(_json_ready(labels_payload), ensure_ascii=False, indent=2), encoding="utf-8")

    result: dict[str, object] = {
        "metadata_csv": str(metadata_csv),
        "rows": {"train": int(train_mask.sum()), "validation": int(val_mask.sum()), "test": int(test_mask.sum())},
        "encoders": {
            "text": text_model,
            "image": f"TorchXRayVision:{image_weights}",
            "image_feature_kind": image_feature_kind,
        },
        "binary": {
            "metrics": {
                "text_neural": text_test_metrics,
                "image_neural": image_test_metrics,
                "neural_cascade": cascade_test_metrics,
            },
            "improvement": {
                "cascade_minus_learned_image": _metric_delta(cascade_test_metrics, image_test_metrics),
                "cascade_minus_external_torchxrayvision": _metric_delta(cascade_test_metrics, external_reference),
            },
            "selection": {
                "selected_on": "validation",
                "regularization_image_c": binary_model.image_c_,
                "regularization_text_c": binary_model.text_c_,
                "image_log_odds_weight": binary_model.image_weight_,
                "text_log_odds_weight": binary_model.text_weight_,
                "decision_threshold": binary_model.decision_threshold_,
                "image_decision_threshold": binary_model.image_decision_threshold_,
                "validation_metrics": binary_model.validation_metrics_,
            },
            "mean_modality_weights": {
                "text": float(modality_weights[:, 0].mean()),
                "image": float(modality_weights[:, 1].mean()),
            },
        },
        "multilabel": {
            "label_columns": label_columns,
            "metrics": {
                "neural_early_fusion": multilabel_metrics(test_y_multi, multilabel_scores),
            },
        },
        "classical_reference": _classical_reference(Path("outputs/final/final_metrics.json")),
        "external_torchxrayvision_reference": external_reference,
        "artifacts": {
            "binary_model": str(binary_model_path),
            "multilabel_model": str(multilabel_model_path),
            "labels": str(labels_path),
            "metrics": str(metrics_path),
        },
    }
    result = _json_ready(result)
    metrics_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final neural text/image fusion artifacts.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/iu_xray_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/neural_final"))
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/neural"))
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

    result = train_neural_fusion_models(
        metadata_csv=args.metadata,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        text_model=args.text_model,
        image_weights=args.image_weights,
        image_feature_kind=args.image_feature_kind,
        text_batch_size=args.text_batch_size,
        image_batch_size=args.image_batch_size,
        max_length=args.max_length,
        device=args.device,
        random_state=args.seed,
        show_progress=not args.quiet,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
