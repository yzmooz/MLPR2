from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_pr.features.text import clean_indication
from ml_pr.data.labels import extract_problem_terms, is_abnormal_terms, sanitize_label_name, top_problem_terms


@dataclass(frozen=True)
class MetadataBuildResult:
    metadata_csv: Path
    labels_txt: Path
    rows: int
    pathology_labels: list[str]


def _normalise_indication(value: object) -> str:
    return clean_indication(value)


def build_metadata_frame(
    reports: pd.DataFrame,
    projections: pd.DataFrame,
    images_dir: Path,
    pathology_labels: list[str] | None = None,
    projection: str = "Frontal",
) -> pd.DataFrame:
    required_reports = {"uid", "Problems", "indication"}
    required_projections = {"uid", "filename", "projection"}
    missing_reports = required_reports - set(reports.columns)
    missing_projections = required_projections - set(projections.columns)
    if missing_reports:
        raise ValueError(f"reports csv is missing columns: {sorted(missing_reports)}")
    if missing_projections:
        raise ValueError(f"projections csv is missing columns: {sorted(missing_projections)}")

    # top-5 выбирается по частоте в Problems, чтобы для каждого класса хватило примеров
    labels = pathology_labels or top_problem_terms(reports, top_k=5)
    # один отчет может относиться к нескольким снимкам одного исследования
    merged = projections.merge(
        reports[["uid", "Problems", "indication"]],
        on="uid",
        how="inner",
        validate="many_to_one",
    )

    merged = merged.loc[merged["projection"] == projection].copy()
    merged["problem_terms"] = merged["Problems"].map(lambda value: "|".join(extract_problem_terms(value)))
    merged["is_abnormal"] = merged["Problems"].map(lambda value: is_abnormal_terms(extract_problem_terms(value)))
    merged["indication"] = merged["indication"].map(_normalise_indication)
    merged["image_path"] = merged["filename"].map(lambda filename: str(images_dir / str(filename)))

    for label in labels:
        column = f"label_{sanitize_label_name(label)}"
        merged[column] = merged["Problems"].map(lambda value, label=label: int(label in extract_problem_terms(value)))

    keep_columns = [
        "uid",
        "filename",
        "projection",
        "image_path",
        "indication",
        "problem_terms",
        "is_abnormal",
    ]
    keep_columns.extend(f"label_{sanitize_label_name(label)}" for label in labels)
    return merged[keep_columns].reset_index(drop=True)


def add_split_column(
    metadata: pd.DataFrame,
    seed: int = 42,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
) -> pd.DataFrame:
    if round(train_size + val_size + test_size, 6) != 1.0:
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    # деление по uid, чтобы снимки одного исследования не оказались в разных частях выборки
    uid_labels = metadata.groupby("uid", as_index=False)["is_abnormal"].max()
    train_uids, temp_uids = train_test_split(
        uid_labels,
        train_size=train_size,
        random_state=seed,
        # сохранение примерно одинаковую долю normal и abnormal в каждом split
        stratify=uid_labels["is_abnormal"],
    )
    relative_val = val_size / (val_size + test_size)
    val_uids, test_uids = train_test_split(
        temp_uids,
        train_size=relative_val,
        random_state=seed,
        stratify=temp_uids["is_abnormal"],
    )

    split_by_uid = {int(uid): "train" for uid in train_uids["uid"]}
    split_by_uid.update({int(uid): "val" for uid in val_uids["uid"]})
    split_by_uid.update({int(uid): "test" for uid in test_uids["uid"]})

    result = metadata.copy()
    result["split"] = result["uid"].map(split_by_uid)
    return result


def build_metadata_from_csv(
    reports_csv: Path,
    projections_csv: Path,
    images_dir: Path,
    out_csv: Path,
    top_k: int = 5,
    projection: str = "Frontal",
    seed: int = 42,
) -> MetadataBuildResult:
    reports = pd.read_csv(reports_csv)
    projections = pd.read_csv(projections_csv)
    labels = top_problem_terms(reports, top_k=top_k)
    metadata = build_metadata_frame(
        reports=reports,
        projections=projections,
        images_dir=images_dir,
        pathology_labels=labels,
        projection=projection,
    )
    metadata = add_split_column(metadata, seed=seed)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(out_csv, index=False, encoding="utf-8")
    labels_txt = out_csv.with_suffix(".labels.txt")
    labels_txt.write_text("\n".join(labels) + "\n", encoding="utf-8")

    return MetadataBuildResult(
        metadata_csv=out_csv,
        labels_txt=labels_txt,
        rows=len(metadata),
        pathology_labels=labels,
    )
