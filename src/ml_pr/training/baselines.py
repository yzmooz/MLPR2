from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from ml_pr.training.evaluate import binary_metrics, multilabel_metrics

TaskName = Literal["binary", "multilabel"]
ModalityName = Literal["text", "image", "fusion"]


def resolve_label_columns(metadata: pd.DataFrame, task: TaskName) -> list[str]:
    if task == "binary":
        if "is_abnormal" not in metadata.columns:
            raise ValueError("Binary task requires `is_abnormal` column.")
        return ["is_abnormal"]

    label_columns = [column for column in metadata.columns if column.startswith("label_")]
    if not label_columns:
        raise ValueError("Multilabel task requires columns with `label_` prefix.")
    return label_columns


def _split_metadata(metadata: pd.DataFrame, max_samples: int | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = metadata.copy()
    if max_samples is not None and len(data) > max_samples:
        fraction = min(1.0, max_samples / len(data))
        sampled_parts = []
        for _, part in data.groupby(["split", "is_abnormal"], dropna=False):
            sample_size = max(1, int(round(len(part) * fraction)))
            sampled_parts.append(part.sample(n=min(sample_size, len(part)), random_state=seed))
        data = pd.concat(sampled_parts, ignore_index=True)

    if "split" in data.columns:
        train = data.loc[data["split"] == "train"].copy()
        val = data.loc[data["split"] == "val"].copy()
        test = data.loc[data["split"] == "test"].copy()
    else:
        from sklearn.model_selection import train_test_split

        train, temp = train_test_split(data, train_size=0.7, random_state=seed, stratify=data["is_abnormal"])
        val, test = train_test_split(temp, train_size=1 / 3, random_state=seed, stratify=temp["is_abnormal"])

    if train.empty or test.empty:
        raise ValueError("Train and test splits must be non-empty.")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def _image_features(paths: pd.Series, image_size: int = 32) -> np.ndarray:
    features: list[np.ndarray] = []
    for path in paths:
        try:
            image = Image.open(path).convert("L").resize((image_size, image_size))
            pixels = np.asarray(image, dtype=np.float32) / 255.0
            hist, _ = np.histogram(pixels, bins=16, range=(0.0, 1.0), density=True)
            summary = np.array([pixels.mean(), pixels.std(), np.percentile(pixels, 25), np.percentile(pixels, 75)])
            features.append(np.concatenate([pixels.reshape(-1), hist.astype(np.float32), summary.astype(np.float32)]))
        except OSError:
            features.append(np.zeros(image_size * image_size + 20, dtype=np.float32))
    return np.vstack(features)


def _make_classifier(task: TaskName):
    base = LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear")
    if task == "binary":
        return base
    return OneVsRestClassifier(base)


def _predict_scores(model, x, task: TaskName) -> np.ndarray:
    probabilities = model.predict_proba(x)
    if task == "binary":
        return probabilities[:, 1]
    return probabilities


def _prepare_features(
    modality: ModalityName,
    train: pd.DataFrame,
    test: pd.DataFrame,
    text_max_features: int,
    image_size: int,
):
    if modality in {"text", "fusion"}:
        # в простом baseline текст представлен частотами слов и пар слов
        vectorizer = TfidfVectorizer(max_features=text_max_features, ngram_range=(1, 2), min_df=1)
        x_train_text = vectorizer.fit_transform(train["indication"].fillna(""))
        x_test_text = vectorizer.transform(test["indication"].fillna(""))
    else:
        x_train_text = x_test_text = None

    if modality in {"image", "fusion"}:
        # здесь используются уменьшенные пиксели и гистограмма, без нейросетевого энкодера
        scaler = StandardScaler()
        x_train_image = scaler.fit_transform(_image_features(train["image_path"], image_size=image_size))
        x_test_image = scaler.transform(_image_features(test["image_path"], image_size=image_size))
    else:
        x_train_image = x_test_image = None

    if modality == "text":
        return x_train_text, x_test_text
    if modality == "image":
        return x_train_image, x_test_image
    return sparse.hstack([x_train_text, sparse.csr_matrix(x_train_image)]), sparse.hstack(
        [x_test_text, sparse.csr_matrix(x_test_image)]
    )


def run_baselines(
    metadata_csv: Path,
    task: TaskName = "binary",
    modalities: list[ModalityName] | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    text_max_features: int = 5000,
    image_size: int = 32,
) -> dict[str, object]:
    metadata = pd.read_csv(metadata_csv)
    label_columns = resolve_label_columns(metadata, task=task)
    train, val, test = _split_metadata(metadata, max_samples=max_samples, seed=seed)
    selected_modalities = modalities or ["text", "image", "fusion"]

    results: dict[str, object] = {
        "metadata_csv": str(metadata_csv),
        "task": task,
        "label_columns": label_columns,
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "metrics": {},
    }

    y_train = train[label_columns[0]].to_numpy() if task == "binary" else train[label_columns].to_numpy()
    y_test = test[label_columns[0]].to_numpy() if task == "binary" else test[label_columns].to_numpy()

    for modality in selected_modalities:
        x_train, x_test = _prepare_features(
            modality=modality,
            train=train,
            test=test,
            text_max_features=text_max_features,
            image_size=image_size,
        )
        model = _make_classifier(task)
        model.fit(x_train, y_train)
        scores = _predict_scores(model, x_test, task=task)
        if task == "binary":
            results["metrics"][modality] = binary_metrics(y_test, scores)
        else:
            results["metrics"][modality] = multilabel_metrics(y_test, scores)

    return results


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quick classical baselines for IU X-ray project.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/iu_xray_metadata.csv"))
    parser.add_argument("--task", choices=["binary", "multilabel"], default="binary")
    parser.add_argument("--modalities", nargs="+", choices=["text", "image", "fusion"], default=["text", "image", "fusion"])
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("outputs/baselines/baseline_metrics.json"))
    parser.add_argument("--text-max-features", type=int, default=5000)
    parser.add_argument("--image-size", type=int, default=32)
    args = parser.parse_args()

    results = run_baselines(
        metadata_csv=args.metadata,
        task=args.task,
        modalities=args.modalities,
        max_samples=args.max_samples,
        seed=args.seed,
        text_max_features=args.text_max_features,
        image_size=args.image_size,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(_json_ready(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_json_ready(results), ensure_ascii=False, indent=2))
    print(f"Saved metrics to {args.out}")


if __name__ == "__main__":
    main()
