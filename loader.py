from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.helpers import normalize_boolean_columns, coerce_numeric_columns


@dataclass(frozen=True)
class DatasetPaths:
    dataset_csv: Path


def get_dataset_paths(base_dir: Optional[str] = None) -> DatasetPaths:
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
    dataset_csv = base / "dataset" / "fmcg_business_dataset.csv"
    return DatasetPaths(dataset_csv=dataset_csv)


def load_dataset(base_dir: Optional[str] = None) -> pd.DataFrame:
    """Load and normalize the FMCG dataset.

    Notes:
    - We keep this deterministic and purely Pandas-based.
    - All downstream analytics assume cleaned dtypes.
    """
    paths = get_dataset_paths(base_dir=base_dir)
    if not paths.dataset_csv.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {paths.dataset_csv}. "
            f"Please ensure dataset/fmcg_business_dataset.csv is present."
        )

    df = pd.read_csv(paths.dataset_csv)

    # Normalize dtypes
    if "week_start_date" in df.columns:
        df["week_start_date"] = pd.to_datetime(df["week_start_date"], errors="coerce")

    normalize_boolean_columns(
        df,
        bool_columns=["promotion_flag", "stockout_flag"],
        default_false=True,
    )

    # promotion_type can be blank when no promo
    if "promotion_type" in df.columns:
        df["promotion_type"] = df["promotion_type"].fillna("").astype(str)
        df.loc[df["promotion_type"].str.strip() == "", "promotion_type"] = "No Promotion"

    # numeric coercions
    numeric_cols = [
        "units_sold",
        "revenue",
        "discount_pct",
        "opening_stock",
        "units_received",
        "closing_stock",
    ]
    coerce_numeric_columns(df, numeric_cols=numeric_cols)

    # Basic validation
    required = {
        "week_start_date",
        "product_id",
        "product_name",
        "category",
        "region",
        "store_id",
        "store_name",
        "units_sold",
        "revenue",
        "promotion_flag",
        "promotion_type",
        "discount_pct",
        "opening_stock",
        "units_received",
        "closing_stock",
        "stockout_flag",
    }
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Drop rows where critical date is missing
    df = df.dropna(subset=["week_start_date"]).reset_index(drop=True)

    return df

