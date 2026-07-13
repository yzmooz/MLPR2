from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ml_pr.inference.predictor import load_artifacts, predict_case
from ml_pr.training.evaluate import binary_metrics
from ml_pr.training.external_torchxrayvision_baseline import abnormal_score_from_external, predict_torchxrayvision_scores


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


def blend_scores(project_scores: np.ndarray, external_scores: np.ndarray, external_weight: float) -> np.ndarray:
    weight = float(np.clip(external_weight, 0.0, 1.0))
    return weight * external_scores + (1.0 - weight) * project_scores


def tune_blend_weight(
    y_true: np.ndarray,
    project_scores: np.ndarray,
    external_scores: np.ndarray,
    steps: int = 100,
) -> tuple[float, dict[str, float]]:
    best_weight = 0.0
    best_metrics = binary_metrics(y_true, project_scores)
    best_score = best_metrics["pr_auc"]

    for weight in np.linspace(0.0, 1.0, steps + 1):
        metrics = binary_metrics(y_true, blend_scores(project_scores, external_scores, float(weight)))
        score = metrics["pr_auc"]
        if score > best_score:
            best_weight = float(weight)
            best_metrics = metrics
            best_score = score

    return best_weight, best_metrics


def _project_scores(metadata: pd.DataFrame, artifacts_dir: Path) -> np.ndarray:
    artifacts = load_artifacts(artifacts_dir)
    scores: list[float] = []
    for row in metadata.itertuples(index=False):
        prediction = predict_case(
            artifacts=artifacts,
            image=Image.open(row.image_path).convert("RGB"),
            indication=row.indication,
        )
        scores.append(float(prediction["abnormal_probability"]))
    return np.asarray(scores)


def _external_scores(metadata: pd.DataFrame, weights: str, batch_size: int, device: str) -> np.ndarray:
    scores, _ = predict_torchxrayvision_scores(
        metadata["image_path"].astype(str).tolist(),
        weights=weights,
        batch_size=batch_size,
        device=device,
    )
    return abnormal_score_from_external(scores)


def run_external_fusion_comparison(
    metadata_csv: Path,
    artifacts_dir: Path,
    output_json: Path,
    weights: str = "densenet121-res224-all",
    batch_size: int = 16,
    device: str = "cpu",
) -> dict[str, object]:
    metadata = pd.read_csv(metadata_csv)
    val = metadata.loc[metadata["split"] == "val"].reset_index(drop=True)
    test = metadata.loc[metadata["split"] == "test"].reset_index(drop=True)
    if val.empty or test.empty:
        raise ValueError("metadata must contain non-empty val and test splits")

    val_y = val["is_abnormal"].to_numpy().astype(int)
    test_y = test["is_abnormal"].to_numpy().astype(int)

    val_project = _project_scores(val, artifacts_dir=artifacts_dir)
    test_project = _project_scores(test, artifacts_dir=artifacts_dir)
    val_external = _external_scores(val, weights=weights, batch_size=batch_size, device=device)
    test_external = _external_scores(test, weights=weights, batch_size=batch_size, device=device)

    external_weight, val_blend_metrics = tune_blend_weight(val_y, val_project, val_external)
    test_blend = blend_scores(test_project, test_external, external_weight)

    result: dict[str, object] = {
        "metadata_csv": str(metadata_csv),
        "artifacts_dir": str(artifacts_dir),
        "weights": weights,
        "selection": {
            "split": "val",
            "criterion": "max PR-AUC",
            "external_weight": external_weight,
            "project_weight": 1.0 - external_weight,
            "val_metrics": val_blend_metrics,
        },
        "test": {
            "project_gated": binary_metrics(test_y, test_project),
            "torchxrayvision": binary_metrics(test_y, test_external),
            "project_plus_torchxrayvision": binary_metrics(test_y, test_blend),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return _json_ready(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend project gated model with TorchXRayVision external baseline.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/iu_xray_metadata.csv"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("outputs/final"))
    parser.add_argument("--out", type=Path, default=Path("outputs/external/project_plus_torchxrayvision_metrics.json"))
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = run_external_fusion_comparison(
        metadata_csv=args.metadata,
        artifacts_dir=args.artifacts_dir,
        output_json=args.out,
        weights=args.weights,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
