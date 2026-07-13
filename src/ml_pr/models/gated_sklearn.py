from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class ConstantBinaryClassifier:
    def __init__(self, positive_probability: float) -> None:
        self.positive_probability = float(np.clip(positive_probability, 0.0, 1.0))

    def fit(self, x, y):
        return self

    def predict_proba(self, x) -> np.ndarray:
        n_rows = x.shape[0] if hasattr(x, "shape") else len(x)
        positive = np.full(n_rows, self.positive_probability, dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


def _fit_binary_classifier(x, y, random_state: int):
    y_arr = np.asarray(y).astype(int)
    unique = np.unique(y_arr)
    if len(unique) < 2:
        return ConstantBinaryClassifier(float(y_arr.mean()))
    model = LogisticRegression(max_iter=1500, class_weight="balanced", solver="liblinear", random_state=random_state)
    model.fit(x, y_arr)
    return model


@dataclass
class GatedPrediction:
    probabilities: np.ndarray
    modality_weights: np.ndarray


class GatedLateFusionClassifier:


    def __init__(
        self,
        text_max_features: int = 5000,
        random_state: int = 42,
        meta_fraction: float = 0.25,
    ) -> None:
        self.text_max_features = text_max_features
        self.random_state = random_state
        self.meta_fraction = meta_fraction

    def fit(self, texts, image_features, y):
        texts_arr = np.asarray(texts, dtype=object)
        image_arr = np.asarray(image_features, dtype=np.float32)
        y_arr = np.asarray(y).astype(int)

        base_idx, meta_idx = self._base_meta_indices(y_arr)
        self.text_vectorizer_ = TfidfVectorizer(max_features=self.text_max_features, ngram_range=(1, 2), min_df=1)
        x_text_base = self.text_vectorizer_.fit_transform(texts_arr[base_idx])
        self.text_classifier_ = _fit_binary_classifier(x_text_base, y_arr[base_idx], random_state=self.random_state)

        self.image_scaler_ = StandardScaler()
        x_image_base = self.image_scaler_.fit_transform(image_arr[base_idx])
        self.image_classifier_ = _fit_binary_classifier(x_image_base, y_arr[base_idx], random_state=self.random_state)

        text_prob_meta, image_prob_meta = self._source_probabilities(texts_arr[meta_idx], image_arr[meta_idx])
        gate_target = (np.abs(image_prob_meta - y_arr[meta_idx]) <= np.abs(text_prob_meta - y_arr[meta_idx])).astype(int)
        gate_features = self._gate_features(texts_arr[meta_idx], image_arr[meta_idx], text_prob_meta, image_prob_meta)
        self.gate_classifier_ = _fit_binary_classifier(gate_features, gate_target, random_state=self.random_state)
        return self

    def _base_meta_indices(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        indices = np.arange(len(y))
        if len(y) < 8 or min(np.bincount(y, minlength=2)) < 2:
            return indices, indices
        try:
            base_idx, meta_idx = train_test_split(
                indices,
                test_size=self.meta_fraction,
                random_state=self.random_state,
                stratify=y,
            )
            return np.asarray(base_idx), np.asarray(meta_idx)
        except ValueError:
            return indices, indices

    def _source_probabilities(self, texts, image_features) -> tuple[np.ndarray, np.ndarray]:
        x_text = self.text_vectorizer_.transform(np.asarray(texts, dtype=object))
        text_prob = self.text_classifier_.predict_proba(x_text)[:, 1]
        x_image = self.image_scaler_.transform(np.asarray(image_features, dtype=np.float32))
        image_prob = self.image_classifier_.predict_proba(x_image)[:, 1]
        return text_prob, image_prob

    def _gate_features(self, texts, image_features, text_prob, image_prob) -> np.ndarray:
        image_arr = np.asarray(image_features, dtype=np.float32)
        text_lengths = np.asarray([len(str(text).split()) for text in texts], dtype=np.float32)
        return np.column_stack(
            [
                text_prob,
                image_prob,
                np.abs(image_prob - text_prob),
                np.maximum(text_prob, 1.0 - text_prob),
                np.maximum(image_prob, 1.0 - image_prob),
                text_lengths,
                image_arr.mean(axis=1),
                image_arr.std(axis=1),
            ]
        )

    def modality_weights(self, texts, image_features) -> np.ndarray:
        text_prob, image_prob = self._source_probabilities(texts, image_features)
        gate_features = self._gate_features(texts, image_features, text_prob, image_prob)
        image_weight = self.gate_classifier_.predict_proba(gate_features)[:, 1]
        return np.column_stack([1.0 - image_weight, image_weight])

    def predict_proba(self, texts, image_features) -> np.ndarray:
        text_prob, image_prob = self._source_probabilities(texts, image_features)
        weights = self.modality_weights(texts, image_features)
        positive = weights[:, 1] * image_prob + weights[:, 0] * text_prob
        positive = np.clip(positive, 0.0, 1.0)
        return np.column_stack([1.0 - positive, positive])

    def predict_with_weights(self, texts, image_features) -> GatedPrediction:
        return GatedPrediction(
            probabilities=self.predict_proba(texts, image_features)[:, 1],
            modality_weights=self.modality_weights(texts, image_features),
        )


class GatedMultilabelClassifier:
    def __init__(
        self,
        label_names: list[str],
        text_max_features: int = 5000,
        random_state: int = 42,
        meta_fraction: float = 0.25,
    ) -> None:
        self.label_names = label_names
        self.text_max_features = text_max_features
        self.random_state = random_state
        self.meta_fraction = meta_fraction

    def fit(self, texts, image_features, y):
        y_arr = np.asarray(y).astype(int)
        self.models_ = []
        for label_index, _ in enumerate(self.label_names):
            model = GatedLateFusionClassifier(
                text_max_features=self.text_max_features,
                random_state=self.random_state + label_index,
                meta_fraction=self.meta_fraction,
            )
            model.fit(texts, image_features, y_arr[:, label_index])
            self.models_.append(model)
        return self

    def predict_proba(self, texts, image_features) -> np.ndarray:
        scores = [model.predict_proba(texts, image_features)[:, 1] for model in self.models_]
        return np.vstack(scores).T

    def modality_weights(self, texts, image_features) -> np.ndarray:
        weights = [model.modality_weights(texts, image_features) for model in self.models_]
        return np.mean(np.stack(weights, axis=0), axis=0)


def hstack_text_image(text_matrix, image_features):
    return sparse.hstack([text_matrix, sparse.csr_matrix(image_features)])
