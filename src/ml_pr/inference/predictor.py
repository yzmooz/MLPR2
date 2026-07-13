from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from ml_pr.features.image import extract_image_feature
from ml_pr.features.text import clean_indication


@dataclass(frozen=True)
class ModelArtifacts:
    binary_model: object
    multilabel_model: object
    pathology_labels: list[str]
    image_size: int


def load_artifacts(output_dir: Path = Path("outputs/final")) -> ModelArtifacts:
    labels_path = output_dir / "labels.json"
    binary_path = output_dir / "binary_gated_fusion.joblib"
    multilabel_path = output_dir / "multilabel_gated_fusion.joblib"
    if not labels_path.exists() or not binary_path.exists() or not multilabel_path.exists():
        raise FileNotFoundError(
            "Final model artifacts are missing. Run: python -m ml_pr.training.train_final"
        )

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return ModelArtifacts(
        binary_model=joblib.load(binary_path),
        multilabel_model=joblib.load(multilabel_path),
        pathology_labels=labels["pathology_labels"],
        image_size=int(labels["image_size"]),
    )


def predict_case(
    artifacts: ModelArtifacts,
    image: Image.Image,
    indication: str,
    threshold: float = 0.5,
    top_k: int = 5,
) -> dict[str, object]:
    texts = np.asarray([clean_indication(indication)], dtype=object)
    image_features = extract_image_feature(image, image_size=artifacts.image_size).reshape(1, -1)

    abnormal_probability = float(artifacts.binary_model.predict_proba(texts, image_features)[0, 1])
    modality_weights = artifacts.binary_model.modality_weights(texts, image_features)[0]
    pathology_scores = artifacts.multilabel_model.predict_proba(texts, image_features)[0]
    top_pathologies = sorted(
        [
            {"name": label, "probability": float(score)}
            for label, score in zip(artifacts.pathology_labels, pathology_scores, strict=True)
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )[:top_k]

    return {
        "decision": "abnormal" if abnormal_probability >= threshold else "normal",
        "abnormal_probability": abnormal_probability,
        "threshold": threshold,
        "top_pathologies": top_pathologies,
        "modality_weights": {
            "text": float(modality_weights[0]),
            "image": float(modality_weights[1]),
        },
    }
