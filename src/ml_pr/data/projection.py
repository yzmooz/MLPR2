from __future__ import annotations

from pathlib import Path

import pandas as pd


def infer_projection_from_filename(filename: str, projections: pd.DataFrame) -> str | None:
    if "filename" not in projections.columns or "projection" not in projections.columns:
        return None

    basename = Path(filename).name
    matches = projections.loc[projections["filename"].astype(str) == basename, "projection"]
    if matches.empty:
        return None
    return str(matches.iloc[0])


def load_projection_table(path: Path = Path("indiana_projections.csv")) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["filename", "projection"])
    return pd.read_csv(path)
