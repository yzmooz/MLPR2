from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from ml_pr.features.image import extract_image_features
from ml_pr.features.text import clean_indication
from ml_pr.models.gated_sklearn import GatedLateFusionClassifier, GatedMultilabelClassifier, _fit_binary_classifier, hstack_text_image
from ml_pr.training.evaluate import binary_metrics, multilabel_metrics


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


def _fit_text_baseline(train_texts, train_y, test_texts, random_state: int, text_max_features: int) -> np.ndarray:
    vectorizer = TfidfVectorizer(max_features=text_max_features, ngram_range=(1, 2), min_df=1)
    x_train = vectorizer.fit_transform(train_texts)
    classifier = _fit_binary_classifier(x_train, train_y, random_state=random_state)
    x_test = vectorizer.transform(test_texts)
    return classifier.predict_proba(x_test)[:, 1]


def _fit_image_baseline(train_features, train_y, test_features, random_state: int) -> np.ndarray:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_features)
    classifier = _fit_binary_classifier(x_train, train_y, random_state=random_state)
    x_test = scaler.transform(test_features)
    return classifier.predict_proba(x_test)[:, 1]


def _fit_early_fusion_baseline(train_texts, train_features, train_y, test_texts, test_features, random_state: int, text_max_features: int) -> np.ndarray:
    vectorizer = TfidfVectorizer(max_features=text_max_features, ngram_range=(1, 2), min_df=1)
    scaler = StandardScaler()
    x_train_text = vectorizer.fit_transform(train_texts)
    x_test_text = vectorizer.transform(test_texts)
    x_train_image = scaler.fit_transform(train_features)
    x_test_image = scaler.transform(test_features)
    classifier = _fit_binary_classifier(
        hstack_text_image(x_train_text, x_train_image),
        train_y,
        random_state=random_state,
    )
    return classifier.predict_proba(hstack_text_image(x_test_text, x_test_image))[:, 1]


def _label_names(label_columns: list[str]) -> list[str]:
    return [column.removeprefix("label_").replace("_", " ") for column in label_columns]


def _split_frames(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "split" not in metadata.columns:
        raise ValueError("metadata must contain a `split` column")
    train = metadata.loc[metadata["split"].isin(["train", "val"])].copy()
    test = metadata.loc[metadata["split"] == "test"].copy()
    if train.empty or test.empty:
        raise ValueError("metadata must contain non-empty train/val and test splits")
    return train.reset_index(drop=True), test.reset_index(drop=True)


def train_gated_models(
    metadata_csv: Path,
    output_dir: Path,
    image_size: int = 24,
    text_max_features: int = 5000,
    random_state: int = 42,
) -> dict[str, object]:
    metadata = pd.read_csv(metadata_csv)
    label_columns = [column for column in metadata.columns if column.startswith("label_")]
    if not label_columns:
        raise ValueError("metadata must contain top pathology columns with `label_` prefix")

    train, test = _split_frames(metadata)
    train_texts = train["indication"].map(clean_indication).to_numpy()
    test_texts = test["indication"].map(clean_indication).to_numpy()
    train_features = extract_image_features(train["image_path"].astype(str).to_numpy(), image_size=image_size)
    test_features = extract_image_features(test["image_path"].astype(str).to_numpy(), image_size=image_size)

    train_y = train["is_abnormal"].to_numpy().astype(int)
    test_y = test["is_abnormal"].to_numpy().astype(int)

    text_scores = _fit_text_baseline(train_texts, train_y, test_texts, random_state, text_max_features)
    image_scores = _fit_image_baseline(train_features, train_y, test_features, random_state)
    early_scores = _fit_early_fusion_baseline(
        train_texts,
        train_features,
        train_y,
        test_texts,
        test_features,
        random_state,
        text_max_features,
    )

    binary_model = GatedLateFusionClassifier(text_max_features=text_max_features, random_state=random_state)
    binary_model.fit(train_texts, train_features, train_y)
    gated_scores = binary_model.predict_proba(test_texts, test_features)[:, 1]
    binary_weights = binary_model.modality_weights(test_texts, test_features)

    train_y_multi = train[label_columns].to_numpy().astype(int)
    test_y_multi = test[label_columns].to_numpy().astype(int)
    multilabel_model = GatedMultilabelClassifier(
        label_names=_label_names(label_columns),
        text_max_features=text_max_features,
        random_state=random_state,
    )
    multilabel_model.fit(train_texts, train_features, train_y_multi)
    multilabel_scores = multilabel_model.predict_proba(test_texts, test_features)

    output_dir.mkdir(parents=True, exist_ok=True)
    binary_model_path = output_dir / "binary_gated_fusion.joblib"
    multilabel_model_path = output_dir / "multilabel_gated_fusion.joblib"
    labels_path = output_dir / "labels.json"
    metrics_path = output_dir / "final_metrics.json"

    joblib.dump(binary_model, binary_model_path)
    joblib.dump(multilabel_model, multilabel_model_path)
    labels_payload = {
        "binary_label": "is_abnormal",
        "label_columns": label_columns,
        "pathology_labels": _label_names(label_columns),
        "image_size": image_size,
        "text_max_features": text_max_features,
    }
    labels_path.write_text(json.dumps(labels_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result: dict[str, object] = {
        "metadata_csv": str(metadata_csv),
        "rows": {"train_val": len(train), "test": len(test)},
        "binary": {
            "metrics": {
                "text_only": binary_metrics(test_y, text_scores),
                "image_only": binary_metrics(test_y, image_scores),
                "early_fusion": binary_metrics(test_y, early_scores),
                "gated_fusion": binary_metrics(test_y, gated_scores),
            },
            "mean_modality_weights": {
                "text": float(binary_weights[:, 0].mean()),
                "image": float(binary_weights[:, 1].mean()),
            },
        },
        "multilabel": {
            "label_columns": label_columns,
            "metrics": {
                "gated_fusion": multilabel_metrics(test_y_multi, multilabel_scores),
            },
        },
        "artifacts": {
            "binary_model": str(binary_model_path),
            "multilabel_model": str(multilabel_model_path),
            "labels": str(labels_path),
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return _json_ready(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final sklearn gated-fusion artifacts.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/iu_xray_metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final"))
    parser.add_argument("--image-size", type=int, default=24)
    parser.add_argument("--text-max-features", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = train_gated_models(
        metadata_csv=args.metadata,
        output_dir=args.output_dir,
        image_size=args.image_size,
        text_max_features=args.text_max_features,
        random_state=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
