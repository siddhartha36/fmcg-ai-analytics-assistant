from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.express as px

from utils.helpers import apply_common_filters


def top_products_analysis(df: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Top products analysis.

    Supports:
    - 'Show top performing products this month'
    - If no timeframe: top overall by revenue.
    """

    filtered = apply_common_filters(df, filters)
    if filtered.empty:
        return {"answer_text": "No data found for the given filters.", "table": [], "chart": None}

    # Ensure timeframe is already applied in apply_common_filters.
    top = (
        filtered.groupby(["product_id", "product_name"], as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
        .sort_values("revenue", ascending=False)
        .head(10)
    )

    fig = px.bar(top, x="product_name", y="revenue", title="Top Products by Revenue")
    fig.update_layout(xaxis_title="Product", yaxis_title="Revenue")

    return {
        "top_products": top.to_dict(orient="records"),
        "chart": fig,
    }


def product_performance_analysis(df: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Product performance + underperformers.

    Heuristic:
    - If filters include products/category/region, compute within those.
    - Underperformers = bottom 5 by revenue.
    """

    filtered = apply_common_filters(df, filters)

    if filtered.empty:
        return {"answer_text": "No data found for the given filters.", "table": [], "chart": None}

    perf = (
        filtered.groupby(["product_id", "product_name", "category"], as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
    )

    under = perf.sort_values("revenue", ascending=True).head(5)
    fig = px.bar(
        under,
        x="product_name",
        y="revenue",
        title="Underperforming Products (Lowest Revenue)",
    )
    fig.update_layout(xaxis_title="Product", yaxis_title="Revenue")

    return {
        "underperforming_products": under.to_dict(orient="records"),
        "chart": fig,
    }

