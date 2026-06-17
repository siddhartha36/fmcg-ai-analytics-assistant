from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.express as px

from utils.helpers import apply_common_filters


def inventory_analysis(df: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Inventory analysis.

    Supports:
    - Which stores experienced stockouts?
    - Stockout detection and counts.

    Uses stockout_flag and closing_stock / opening_stock as context.
    """

    filtered = apply_common_filters(df, filters)

    if filtered.empty:
        return {"answer_text": "No data found for the given filters.", "table": [], "chart": None}

    stockouts = filtered[filtered["stockout_flag"] == True].copy()

    if stockouts.empty:
        # Provide chart with zeros (still deterministic)
        store_counts = (
            filtered.groupby("store_name", as_index=False)
            .agg(stockout_events=("stockout_flag", "sum"))
            .sort_values("stockout_events", ascending=False)
            .head(10)
        )
        fig = px.bar(
            store_counts,
            x="store_name",
            y="stockout_events",
            title="Stockout Analysis (Top Stores by Stockout Events)",
        )
        fig.update_layout(xaxis_title="Store", yaxis_title="Stockout Events")
        return {
            "stockouts": {
                "total_stockout_rows": 0,
                "total_rows": int(len(filtered)),
            },
            "top_stores": store_counts.to_dict(orient="records"),
            "chart": fig,
        }

    # Stockout counts by store
    store_counts = (
        stockouts.groupby("store_name", as_index=False)
        .agg(stockout_events=("stockout_flag", "sum"), stockout_revenue=("revenue", "sum"))
        .sort_values("stockout_events", ascending=False)
        .head(10)
    )

    fig = px.bar(
        store_counts,
        x="store_name",
        y="stockout_events",
        title="Stockout Analysis (Top Stores by Stockout Events)",
    )
    fig.update_layout(xaxis_title="Store", yaxis_title="Stockout Events")

    # Extra: stockout share
    total_rows = len(filtered)
    total_stockout_rows = int(stockouts.shape[0])
    share = float(total_stockout_rows) / float(total_rows) if total_rows else 0.0

    # Aggregate by region too (for narrative)
    region_counts = (
        stockouts.groupby("region", as_index=False)
        .agg(stockout_events=("stockout_flag", "sum"), revenue_affected=("revenue", "sum"))
        .sort_values("stockout_events", ascending=False)
        .head(5)
    )

    return {
        "stockouts": {
            "total_stockout_rows": total_stockout_rows,
            "total_rows": int(total_rows),
            "stockout_share_pct": share * 100.0,
        },
        "top_stores": store_counts.to_dict(orient="records"),
        "top_regions": region_counts.to_dict(orient="records"),
        "chart": fig,
    }

