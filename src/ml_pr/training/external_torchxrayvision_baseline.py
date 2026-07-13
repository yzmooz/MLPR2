from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml_pr.training.evaluate import binary_metrics, multilabel_metrics


EXTERNAL_LABEL_MAP = {
    "label_Opacity": "Lung Opacity",
    "label_Cardiomegaly": "Cardiomegaly",
    "label_Pulmonary_Atelectasis": "Atelectasis",
}


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


def abnormal_score_from_external(external_scores: np.ndarray) -> np.ndarray:
    # для внешней модели считаем снимок abnormal, если она уверена хотя бы в одной патологии
    return np.nanmax(external_scores, axis=1)


def mapped_multilabel_targets(
    metadata: pd.DataFrame,
    external_pathologies: list[str],
) -> tuple[np.ndarray, list[int], list[str]]:
    y_columns: list[str] = []
    external_indices: list[int] = []
    label_names: list[str] = []

    # названия классов в IU и TorchXRayVision отличаются, сопоставляем только общие
    for column, external_label in EXTERNAL_LABEL_MAP.items():
        if column in metadata.columns and external_label in external_pathologies:
            y_columns.append(column)
            external_indices.append(external_pathologies.index(external_label))
            label_names.append(column.removeprefix("label_").replace("_", " "))

    if not y_columns:
        raise ValueError("No overlapping labels found between IU metadata and TorchXRayVision outputs.")

    return metadata[y_columns].to_numpy().astype(int), external_indices, label_names


def _load_image_for_xrv(path: str, image_size: int):
    import torch
    import torchxrayvision as xrv
    from PIL import Image

    image = Image.open(path).convert("L")
    image_arr = np.asarray(image, dtype=np.float32)
    image_arr = xrv.datasets.normalize(image_arr, 255)
    image_arr = image_arr[None, :, :]
    transform = xrv.datasets.XRayResizer(image_size)
    image_arr = transform(image_arr)
    return torch.from_numpy(image_arr).float()


def predict_torchxrayvision_scores(
    image_paths: list[str],
    weights: str = "densenet121-res224-all",
    batch_size: int = 16,
    image_size: int = 224,
    device: str = "cpu",
) -> tuple[np.ndarray, list[str]]:
    import torch
    import torchxrayvision as xrv

    model = xrv.models.DenseNet(weights=weights)
    model = model.to(device)
    model.eval()
    pathologies = list(model.pathologies)

    batches: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch = torch.stack([_load_image_for_xrv(path, image_size=image_size) for path in batch_paths]).to(device)
            outputs = model(batch).detach().cpu().numpy()
            batches.append(outputs)

    return np.vstack(batches), pathologies


def run_external_baseline(
    metadata_csv: Path,
    output_json: Path,
    weights: str = "densenet121-res224-all",
    batch_size: int = 16,
    image_size: int = 224,
    device: str = "cpu",
) -> dict[str, object]:
    metadata = pd.read_csv(metadata_csv)
    # внешний baseline проверяем ровно на том же test split, что и нашу модель
    test = metadata.loc[metadata["split"] == "test"].reset_index(drop=True)
    if test.empty:
        raise ValueError("metadata must contain non-empty test split")

    scores, pathologies = predict_torchxrayvision_scores(
        test["image_path"].astype(str).tolist(),
        weights=weights,
        batch_size=batch_size,
        image_size=image_size,
        device=device,
    )
    binary_scores = abnormal_score_from_external(scores)
    y_binary = test["is_abnormal"].to_numpy().astype(int)
    y_multi, external_indices, mapped_labels = mapped_multilabel_targets(test, pathologies)
    mapped_scores = scores[:, external_indices]

    result: dict[str, object] = {
        "metadata_csv": str(metadata_csv),
        "weights": weights,
        "rows": {"test": len(test)},
        "external_pathologies": pathologies,
        "binary_abnormal_heuristic": "max probability over all TorchXRayVision pathologies",
        "binary": {
            "metrics": binary_metrics(y_binary, binary_scores),
        },
        "mapped_multilabel": {
            "label_mapping": {
                label: pathologies[index] for label, index in zip(mapped_labels, external_indices, strict=True)
            },
            "metrics": multilabel_metrics(y_multi, mapped_scores),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(_json_ready(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return _json_ready(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare project against TorchXRayVision pretrained baseline.")
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/iu_xray_metadata.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/external/torchxrayvision_metrics.json"))
    parser.add_argument("--weights", default="densenet121-res224-all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    result = run_external_baseline(
        metadata_csv=args.metadata,
        output_json=args.out,
        weights=args.weights,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
