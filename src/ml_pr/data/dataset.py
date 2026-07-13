from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
from PIL import Image


class IUXRayDataset:


    def __init__(
        self,
        metadata_csv: Path,
        split: str,
        label_columns: list[str],
        image_transform: Callable | None = None,
    ) -> None:
        metadata = pd.read_csv(metadata_csv)
        if "split" in metadata.columns:
            metadata = metadata.loc[metadata["split"] == split].copy()
        self.metadata = metadata.reset_index(drop=True)
        self.label_columns = label_columns
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.metadata.iloc[index]
        # из одной строки возвращаем обе модальности и общую целевую разметку
        image = Image.open(row["image_path"]).convert("RGB")
        if self.image_transform is not None:
            image = self.image_transform(image)

        labels = row[self.label_columns].astype("float32").to_numpy()
        return {
            "image": image,
            "text": row.get("indication", ""),
            "labels": labels,
            "uid": int(row["uid"]),
        }
