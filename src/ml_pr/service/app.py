from __future__ import annotations

import json
from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import streamlit as st
from PIL import Image

from ml_pr.data.projection import infer_projection_from_filename, load_projection_table
from ml_pr.inference.neural_predictor import load_neural_artifacts, predict_neural_case


def _load_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    st.set_page_config(page_title="Chest X-ray multimodal diagnosis", layout="wide")
    st.title("Multimodal chest X-ray diagnosis")

    metrics = _load_metrics(Path("outputs/baselines/binary_baseline_metrics.json"))
    if metrics:
        st.sidebar.subheader("Current baseline")
        for name, values in metrics.get("metrics", {}).items():
            st.sidebar.metric(f"{name} ROC-AUC", f"{values.get('roc_auc', 0):.3f}")

    neural_metrics = _load_metrics(Path("outputs/neural_final/neural_final_metrics.json"))
    neural_binary = neural_metrics.get("binary", {}).get("metrics", {}) if neural_metrics else {}
    if neural_binary:
        st.sidebar.subheader("Final neural model")
        image_neural = neural_binary.get("image_neural", {})
        cascade_neural = neural_binary.get("neural_cascade", {})
        st.sidebar.metric("TXRV image PR-AUC", f"{image_neural.get('pr_auc', 0):.3f}")
        st.sidebar.metric("Multimodal cascade PR-AUC", f"{cascade_neural.get('pr_auc', 0):.3f}")
        st.sidebar.metric("Multimodal cascade F1", f"{cascade_neural.get('f1', 0):.3f}")

    image = st.file_uploader("Frontal chest X-ray image", type=["png", "jpg", "jpeg"])
    indication = st.text_area("Clinical indication", placeholder="cough, fever, chest pain")

    if image is not None:
        st.image(image, caption="Input X-ray", width=420)
        # для файлов из IU X-ray проекцию можно проверить по исходной таблице
        projection = infer_projection_from_filename(image.name, load_projection_table())
        if projection and projection != "Frontal":
            st.warning(
                f"Uploaded file is marked as {projection}. This model was trained for Frontal images, "
                "so the prediction can be unreliable."
            )
        elif projection == "Frontal":
            st.caption("Projection check: Frontal")

    if st.button("Predict", disabled=image is None):
        try:
            # интерфейс использует сохраненную финальную модель, а не обучает ее заново
            artifacts = load_neural_artifacts(Path("outputs/neural_final"))
            pil_image = Image.open(image).convert("RGB")
            prediction = predict_neural_case(artifacts=artifacts, image=pil_image, indication=indication)
            st.caption("Model: TorchXRayVision DenseNet121 + ClinicalBERT indication cascade")
            st.subheader(prediction["decision"])
            st.metric("Abnormal probability", f"{prediction['abnormal_probability']:.3f}")
            st.write("Source probabilities")
            st.json(prediction["source_probabilities"])
            st.write("Modality weights")
            st.json(prediction["modality_weights"])
            st.write("Top pathologies")
            st.dataframe(prediction["top_pathologies"], hide_index=True)
        except FileNotFoundError as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
