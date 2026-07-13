from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from ml_pr.models.gated_sklearn import ConstantBinaryClassifier, _fit_binary_classifier


def _as_float_array(features) -> np.ndarray:
    return np.asarray(features, dtype=np.float32)


def _fusion_features(text_features, image_features) -> np.ndarray:
    return np.hstack([_as_float_array(text_features), _as_float_array(image_features)])


def _fit_regularized_classifier(features, y, c_value: float, random_state: int):
    y_arr = np.asarray(y).astype(int)
    if len(np.unique(y_arr)) < 2:
        return ConstantBinaryClassifier(float(y_arr.mean()))
    # балансировка нужна, чтобы частый класс abnormal не подавлял normal
    classifier = LogisticRegression(
        C=c_value,
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )
    classifier.fit(features, y_arr)
    return classifier


def _safe_logit(probabilities) -> np.ndarray:
    return logit(np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6))


def _best_f1_threshold(y_true, probabilities) -> float:
    thresholds = np.linspace(0.05, 0.80, 151)
    return float(max(thresholds, key=lambda value: f1_score(y_true, probabilities >= value, zero_division=0)))


class NeuralCascadeBinaryClassifier:


    def __init__(
        self,
        random_state: int = 42,
        regularization_grid: tuple[float, ...] = (0.0003, 0.001, 0.003, 0.01, 0.03),
        image_weight_grid: tuple[float, ...] = (1.0,),
        text_weight_grid: tuple[float, ...] | None = None,
    ) -> None:
        self.random_state = random_state
        self.regularization_grid = regularization_grid
        self.image_weight_grid = image_weight_grid
        self.text_weight_grid = text_weight_grid

    def fit(
        self,
        text_features,
        image_features,
        y,
        validation_text_features,
        validation_image_features,
        validation_y,
    ):
        text_arr = _as_float_array(text_features)
        image_arr = _as_float_array(image_features)
        val_text_arr = _as_float_array(validation_text_features)
        val_image_arr = _as_float_array(validation_image_features)
        y_arr = np.asarray(y).astype(int)
        val_y_arr = np.asarray(validation_y).astype(int)
        if len(np.unique(val_y_arr)) < 2:
            raise ValueError("validation_y must contain both classes for cascade selection")

        # у BERT и DenseNet разные масштабы признаков, поэтому нормализую их отдельно
        self.text_scaler_ = StandardScaler().fit(text_arr)
        self.image_scaler_ = StandardScaler().fit(image_arr)
        train_text = self.text_scaler_.transform(text_arr)
        train_image = self.image_scaler_.transform(image_arr)
        val_text = self.text_scaler_.transform(val_text_arr)
        val_image = self.image_scaler_.transform(val_image_arr)

        # силу регуляризации подбираю отдельно для текста и снимка
        text_candidates = self._candidate_models(train_text, y_arr, val_text, offset=100)
        image_candidates = self._candidate_models(train_image, y_arr, val_image, offset=0)

        weights = self.text_weight_grid or tuple(float(value) for value in np.linspace(0.0, 0.5, 11))

        best = None
        for image_c, image_classifier, image_probability in image_candidates:
            image_logit = _safe_logit(image_probability)
            for text_c, text_classifier, text_probability in text_candidates:
                text_logit = _safe_logit(text_probability)
                for image_weight in self.image_weight_grid:
                    for text_weight in weights:
                        # снимок дает основное решение, а текст его корректирует
                        cascade_probability = expit(image_weight * image_logit + text_weight * text_logit)
                        objective = (
                            roc_auc_score(val_y_arr, cascade_probability)
                            + average_precision_score(val_y_arr, cascade_probability)
                        ) / 2.0
                        candidate = (
                            float(objective),
                            -float(image_weight + text_weight),
                            image_c,
                            text_c,
                            float(image_weight),
                            float(text_weight),
                            image_classifier,
                            text_classifier,
                            image_probability,
                            text_probability,
                            cascade_probability,
                        )
                        if best is None or candidate[:2] > best[:2]:
                            best = candidate

        if best is None:
            raise RuntimeError("cascade model selection produced no candidate")
        (
            self.validation_objective_,
            _,
            self.image_c_,
            self.text_c_,
            self.image_weight_,
            self.text_weight_,
            self.image_classifier_,
            self.text_classifier_,
            val_image_probability,
            val_text_probability,
            val_cascade_probability,
        ) = best
        # порог выбирается по validation и после этого на test уже не меняется
        self.decision_threshold_ = _best_f1_threshold(val_y_arr, val_cascade_probability)
        self.image_decision_threshold_ = _best_f1_threshold(val_y_arr, val_image_probability)
        self.validation_metrics_ = {
            "image_roc_auc": float(roc_auc_score(val_y_arr, val_image_probability)),
            "image_pr_auc": float(average_precision_score(val_y_arr, val_image_probability)),
            "text_roc_auc": float(roc_auc_score(val_y_arr, val_text_probability)),
            "text_pr_auc": float(average_precision_score(val_y_arr, val_text_probability)),
            "cascade_roc_auc": float(roc_auc_score(val_y_arr, val_cascade_probability)),
            "cascade_pr_auc": float(average_precision_score(val_y_arr, val_cascade_probability)),
            "cascade_f1": float(
                f1_score(val_y_arr, val_cascade_probability >= self.decision_threshold_, zero_division=0)
            ),
        }
        return self

    def _candidate_models(self, train_features, y, validation_features, offset: int):
        candidates = []
        for index, c_value in enumerate(self.regularization_grid):
            classifier = _fit_regularized_classifier(
                train_features,
                y,
                c_value=c_value,
                random_state=self.random_state + offset + index,
            )
            probability = classifier.predict_proba(validation_features)[:, 1]
            candidates.append((float(c_value), classifier, probability))
        return candidates

    def source_probabilities(self, text_features, image_features) -> dict[str, np.ndarray]:
        text_probability = self.text_classifier_.predict_proba(
            self.text_scaler_.transform(_as_float_array(text_features))
        )[:, 1]
        image_probability = self.image_classifier_.predict_proba(
            self.image_scaler_.transform(_as_float_array(image_features))
        )[:, 1]
        return {"text": text_probability, "image": image_probability}

    def predict_proba(self, text_features, image_features) -> np.ndarray:
        sources = self.source_probabilities(text_features, image_features)
        positive = expit(
            self.image_weight_ * _safe_logit(sources["image"])
            + self.text_weight_ * _safe_logit(sources["text"])
        )
        return np.column_stack([1.0 - positive, positive])

    def predict_image_only_proba(self, image_features) -> np.ndarray:
        positive = self.image_classifier_.predict_proba(
            self.image_scaler_.transform(_as_float_array(image_features))
        )[:, 1]
        return np.column_stack([1.0 - positive, positive])

    def modality_weights(self, text_features, image_features) -> np.ndarray:
        count = len(_as_float_array(text_features))
        total = self.image_weight_ + self.text_weight_
        weights = np.asarray([self.text_weight_ / total, self.image_weight_ / total], dtype=np.float64)
        return np.tile(weights, (count, 1))


class NeuralEarlyFusionBinaryClassifier:


    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def fit(self, text_features, image_features, y):
        text_arr = _as_float_array(text_features)
        image_arr = _as_float_array(image_features)
        fusion_arr = _fusion_features(text_arr, image_arr)
        y_arr = np.asarray(y).astype(int)

        self.text_dim_ = text_arr.shape[1]
        self.image_dim_ = image_arr.shape[1]

        self.text_scaler_ = StandardScaler()
        self.text_classifier_ = _fit_binary_classifier(
            self.text_scaler_.fit_transform(text_arr),
            y_arr,
            random_state=self.random_state,
        )

        self.image_scaler_ = StandardScaler()
        self.image_classifier_ = _fit_binary_classifier(
            self.image_scaler_.fit_transform(image_arr),
            y_arr,
            random_state=self.random_state,
        )

        self.fusion_scaler_ = StandardScaler()
        self.fusion_classifier_ = _fit_binary_classifier(
            self.fusion_scaler_.fit_transform(fusion_arr),
            y_arr,
            random_state=self.random_state,
        )
        self.modality_weight_ = self._coefficient_modality_weight()
        return self

    def source_probabilities(self, text_features, image_features) -> dict[str, np.ndarray]:
        text_scores = self.text_classifier_.predict_proba(self.text_scaler_.transform(_as_float_array(text_features)))[:, 1]
        image_scores = self.image_classifier_.predict_proba(self.image_scaler_.transform(_as_float_array(image_features)))[:, 1]
        return {"text": text_scores, "image": image_scores}

    def predict_proba(self, text_features, image_features) -> np.ndarray:
        fusion_arr = self.fusion_scaler_.transform(_fusion_features(text_features, image_features))
        return self.fusion_classifier_.predict_proba(fusion_arr)

    def modality_weights(self, text_features, image_features) -> np.ndarray:
        n_rows = len(_as_float_array(text_features))
        return np.tile(self.modality_weight_, (n_rows, 1))

    def _coefficient_modality_weight(self) -> np.ndarray:
        coef = getattr(self.fusion_classifier_, "coef_", None)
        if coef is None:
            return np.asarray([0.5, 0.5], dtype=np.float64)
        coef_arr = np.abs(np.asarray(coef)[0])
        text_strength = float(np.linalg.norm(coef_arr[: self.text_dim_]))
        image_strength = float(np.linalg.norm(coef_arr[self.text_dim_ : self.text_dim_ + self.image_dim_]))
        total = text_strength + image_strength
        if total <= 0.0:
            return np.asarray([0.5, 0.5], dtype=np.float64)
        return np.asarray([text_strength / total, image_strength / total], dtype=np.float64)


class NeuralEarlyFusionMultilabelClassifier:


    def __init__(self, label_names: list[str], random_state: int = 42) -> None:
        self.label_names = label_names
        self.random_state = random_state

    def fit(self, text_features, image_features, y):
        fusion_arr = _fusion_features(text_features, image_features)
        y_arr = np.asarray(y).astype(int)
        self.fusion_scaler_ = StandardScaler()
        x_scaled = self.fusion_scaler_.fit_transform(fusion_arr)
        # для каждой патологии обучается своя бинарная голова, признаки у них общие
        self.classifiers_ = [
            _fit_binary_classifier(x_scaled, y_arr[:, label_index], random_state=self.random_state + label_index)
            for label_index in range(y_arr.shape[1])
        ]
        return self

    def predict_proba(self, text_features, image_features) -> np.ndarray:
        x_scaled = self.fusion_scaler_.transform(_fusion_features(text_features, image_features))
        scores = [classifier.predict_proba(x_scaled)[:, 1] for classifier in self.classifiers_]
        return np.vstack(scores).T
