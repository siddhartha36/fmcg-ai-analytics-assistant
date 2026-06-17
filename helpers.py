from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


TRUTHY = {"true", "1", "yes", "y", "t"}
FALSY = {"false", "0", "no", "n", "f"}


def normalize_boolean_columns(
    df: pd.DataFrame,
    bool_columns: Sequence[str],
    default_false: bool = False,
) -> None:
    """Normalize boolean-ish columns into actual Python bool.

    Handles values like: True/False, 1/0, "True"/"False", "true"/"false".
    """

    for col in bool_columns:
        if col not in df.columns:
            continue

        def _norm(v):
            if pd.isna(v):
                return bool(default_false)
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, np.integer)):
                return bool(int(v) == 1)
            if isinstance(v, (float, np.floating)):
                if np.isnan(v):
                    return bool(default_false)
                return bool(int(v) == 1)
            s = str(v).strip().lower()
            if s in TRUTHY:
                return True
            if s in FALSY:
                return False
            # Fallback: treat unknown as default
            return bool(default_false)

        df[col] = df[col].map(_norm).astype(bool)


def coerce_numeric_columns(df: pd.DataFrame, numeric_cols: Sequence[str]) -> None:
    for col in numeric_cols:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")


def latest_month_window(df: pd.DataFrame) -> pd.Period:
    """Return the latest month period available in week_start_date."""
    if "week_start_date" not in df.columns:
        raise ValueError("week_start_date missing")
    latest = df["week_start_date"].max()
    if pd.isna(latest):
        return pd.Period(pd.Timestamp.today(), freq="M")
    return pd.Period(pd.Timestamp(latest), freq="M")


def apply_common_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply basic filters derived from LLM intent.

    Supported keys (optional):
    - regions: List[str]
    - region: str (single)
    - categories: List[str]
    - category: str
    - products: List[str] (matches product_name)
    - product_name: str
    - stores: List[str] (matches store_name)
    - store_name: str
    - time_granularity: 'month' (optional)
    - timeframe: 'this_month' | 'all' | 'latest'
    """

    out = df

    def _apply_list(field: str, key: str):
        nonlocal out
        if key in filters and filters[key]:
            vals = filters[key]
            if isinstance(vals, str):
                vals = [vals]
            out = out[out[field].isin(vals)]

    _apply_list("region", "regions")
    _apply_list("region", "region")
    _apply_list("category", "categories")
    _apply_list("category", "category")
    _apply_list("product_name", "products")
    _apply_list("product_name", "product_name")
    _apply_list("store_name", "stores")
    _apply_list("store_name", "store_name")

    timeframe = (filters.get("timeframe") or filters.get("time_window") or "latest").lower()

    if timeframe in {"this_month", "this month", "month", "current_month"}:
        month = latest_month_window(out)
        start = month.start_time
        end = month.end_time
        out = out[(out["week_start_date"] >= start) & (out["week_start_date"] <= end)]

    return out.copy()


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0 or pd.isna(denominator):
        return default
    return float(numerator) / float(denominator)


def dataframe_to_records_for_llm(df: pd.DataFrame, max_rows: int = 30) -> List[Dict]:
    """Convert small dataframe to list-of-dicts for LLM summarization.

    This keeps prompts bounded.
    """
    if df is None or df.empty:
        return []

    limited = df.head(max_rows)
    return limited.to_dict(orient="records")

