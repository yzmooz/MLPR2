from __future__ import annotations

import argparse
from pathlib import Path

from ml_pr.data.iu_xray import build_metadata_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IU X-ray metadata from Kaggle CSV files.")
    parser.add_argument("--reports-csv", type=Path, default=Path("indiana_reports.csv"))
    parser.add_argument("--projections-csv", type=Path, default=Path("indiana_projections.csv"))
    parser.add_argument("--images-dir", type=Path, default=Path("images/images_normalized"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/iu_xray_metadata.csv"))
    parser.add_argument("--projection", default="Frontal")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build_metadata_from_csv(
        reports_csv=args.reports_csv,
        projections_csv=args.projections_csv,
        images_dir=args.images_dir,
        out_csv=args.out,
        top_k=args.top_k,
        projection=args.projection,
        seed=args.seed,
    )
    print(f"Wrote {result.rows} rows to {result.metadata_csv}")
    print(f"Top-{len(result.pathology_labels)} labels: {', '.join(result.pathology_labels)}")
    print(f"Labels saved to {result.labels_txt}")


if __name__ == "__main__":
    main()
