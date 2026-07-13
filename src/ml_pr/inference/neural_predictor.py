from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from ml_pr.features.text import clean_indication
from ml_pr.training.neural_comparison import masked_mean_pool, resolve_device_name


@dataclass(frozen=True)
class NeuralModelArtifacts:
    binary_model: object
    multilabel_model: object
    pathology_labels: list[str]
    text_encoder: str
    image_encoder: str
    image_feature_kind: str


def load_neural_artifacts(output_dir: Path = Path("outputs/neural_final")) -> NeuralModelArtifacts:
    labels_path = output_dir / "labels.json"
    binary_path = output_dir / "binary_neural_fusion.joblib"
    multilabel_path = output_dir / "multilabel_neural_fusion.joblib"
    if not labels_path.exists() or not binary_path.exists() or not multilabel_path.exists():
        raise FileNotFoundError(
            "Neural model artifacts are missing. Run: python -m ml_pr.training.train_neural_final"
        )

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    return NeuralModelArtifacts(
        binary_model=joblib.load(binary_path),
        multilabel_model=joblib.load(multilabel_path),
        pathology_labels=labels["pathology_labels"],
        text_encoder=labels["text_encoder"],
        image_encoder=labels["image_encoder"],
        image_feature_kind=labels.get("image_feature_kind", "embedding"),
    )


@lru_cache(maxsize=2)
def _load_text_encoder(model_name: str, device: str):
    import torch
    from transformers import AutoModel, AutoTokenizer

    # модель остается в памяти, иначе Streamlit загружал бы ее при каждом нажатии
    resolved_device = resolve_device_name(device, torch_module=torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


@lru_cache(maxsize=2)
def _load_xray_encoder(image_encoder: str, device: str):
    import torch
    import torchxrayvision as xrv

    resolved_device = resolve_device_name(device, torch_module=torch)
    weights = image_encoder.removeprefix("TorchXRayVision:")
    model = xrv.models.DenseNet(weights=weights).to(resolved_device)
    model.eval()
    return model, resolved_device


def _xray_tensor_from_pil(image: Image.Image, image_size: int = 224):
    import torch
    import torchxrayvision as xrv

    # обработка снимка должна совпадать с форматом, на котором обучалась TorchXRayVision
    image_arr = np.asarray(image.convert("L"), dtype=np.float32)
    image_arr = xrv.datasets.normalize(image_arr, 255)
    image_arr = image_arr[None, :, :]
    image_arr = xrv.datasets.XRayResizer(image_size)(image_arr)
    return torch.from_numpy(image_arr).float()


def extract_text_feature(text: str, model_name: str, device: str = "auto", max_length: int = 128) -> np.ndarray:
    import torch

    tokenizer, model, resolved_device = _load_text_encoder(model_name, device)
    encoded = tokenizer(
        [clean_indication(text)],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
    return masked_mean_pool(
        outputs.last_hidden_state.detach().cpu().numpy(),
        encoded["attention_mask"].detach().cpu().numpy(),
    ).astype(np.float32)


def extract_image_feature(image: Image.Image, image_encoder: str, device: str = "auto") -> np.ndarray:
    import torch

    model, resolved_device = _load_xray_encoder(image_encoder, device)
    batch = _xray_tensor_from_pil(image).unsqueeze(0).to(resolved_device)
    with torch.no_grad():
        features = model.features2(batch).detach().cpu().numpy()
    return features.astype(np.float32)


def extract_case_neural_features(
    image: Image.Image,
    indication: str,
    text_encoder: str,
    image_encoder: str,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    return (
        extract_text_feature(indication, model_name=text_encoder, device=device),
        extract_image_feature(image, image_encoder=image_encoder, device=device),
    )


def predict_neural_case(
    artifacts: NeuralModelArtifacts,
    image: Image.Image,
    indication: str,
    threshold: float | None = None,
    top_k: int = 5,
    device: str = "auto",
) -> dict[str, object]:
    text_features, image_features = extract_case_neural_features(
        image=image,
        indication=indication,
        text_encoder=artifacts.text_encoder,
        image_encoder=artifacts.image_encoder,
        device=device,
    )

    # отдельно сохраняем ответы веток, чтобы их вклад можно было показать в интерфейсе
    source_scores = artifacts.binary_model.source_probabilities(text_features, image_features)
    has_text = bool(clean_indication(indication))
    if has_text:
        abnormal_probability = float(artifacts.binary_model.predict_proba(text_features, image_features)[0, 1])
        modality_weights = artifacts.binary_model.modality_weights(text_features, image_features)[0]
        selected_threshold = float(
            threshold if threshold is not None else getattr(artifacts.binary_model, "decision_threshold_", 0.5)
        )
        model_type = "neural_image_text_cascade"
    else:
        # пустой indication не заменяем искусственным текстом, а оставляем только снимок
        abnormal_probability = float(artifacts.binary_model.predict_image_only_proba(image_features)[0, 1])
        modality_weights = np.asarray([0.0, 1.0])
        selected_threshold = float(
            threshold if threshold is not None else getattr(artifacts.binary_model, "image_decision_threshold_", 0.5)
        )
        model_type = "neural_image_only_fallback"
    pathology_scores = artifacts.multilabel_model.predict_proba(text_features, image_features)[0]
    # top-5 сортируется по вероятности отдельных патологических голов
    top_pathologies = sorted(
        [
            {"name": label, "probability": float(score)}
            for label, score in zip(artifacts.pathology_labels, pathology_scores, strict=True)
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )[:top_k]

    return {
        "model_type": model_type,
        "decision": "abnormal" if abnormal_probability >= selected_threshold else "normal",
        "abnormal_probability": abnormal_probability,
        "threshold": selected_threshold,
        "top_pathologies": top_pathologies,
        "modality_weights": {
            "text": float(modality_weights[0]),
            "image": float(modality_weights[1]),
        },
        "source_probabilities": {
            "text": float(source_scores["text"][0]),
            "image": float(source_scores["image"][0]),
        },
    }
