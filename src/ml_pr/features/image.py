from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def extract_image_feature(image_or_path: Image.Image | str | Path, image_size: int = 32) -> np.ndarray:
    # это простые признаки для классического baseline, финальная модель использует DenseNet
    if isinstance(image_or_path, Image.Image):
        image = image_or_path.convert("L")
    else:
        image = Image.open(image_or_path).convert("L")

    image = image.resize((image_size, image_size))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    hist, _ = np.histogram(pixels, bins=16, range=(0.0, 1.0), density=True)
    summary = np.array(
        [
            pixels.mean(),
            pixels.std(),
            np.percentile(pixels, 10),
            np.percentile(pixels, 25),
            np.percentile(pixels, 50),
            np.percentile(pixels, 75),
            np.percentile(pixels, 90),
            pixels.min(),
            pixels.max(),
            float((pixels > 0.5).mean()),
            float((pixels > 0.7).mean()),
            float((pixels < 0.2).mean()),
            float(np.abs(np.diff(pixels, axis=0)).mean()),
            float(np.abs(np.diff(pixels, axis=1)).mean()),
            float(pixels[: image_size // 2].mean()),
            float(pixels[image_size // 2 :].mean()),
            float(pixels[:, : image_size // 2].mean()),
            float(pixels[:, image_size // 2 :].mean()),
            float(pixels.var()),
            float(np.median(pixels)),
        ],
        dtype=np.float32,
    )
    return np.concatenate([pixels.reshape(-1), hist.astype(np.float32), summary]).astype(np.float32)


def extract_image_features(paths: list[str] | np.ndarray, image_size: int = 32) -> np.ndarray:
    features: list[np.ndarray] = []
    empty = np.zeros(image_size * image_size + 36, dtype=np.float32)
    for path in paths:
        try:
            features.append(extract_image_feature(path, image_size=image_size))
        except OSError:
            # битый снимок заменяется нулевым вектором, чтобы не сдвинуть порядок строк
            features.append(empty.copy())
    return np.vstack(features)
