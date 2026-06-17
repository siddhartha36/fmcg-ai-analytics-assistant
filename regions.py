from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.express as px

from utils.helpers import apply_common_filters


def revenue_analysis(df: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Revenue analysis (generic) - supports top by product or category depending on filters."""
    # Reuse sales.py logic by importing locally to avoid circular imports
    from analytics.sales import revenue_analysis as _revenue_analysis

    return _revenue_analysis(df, filters)


def regional_comparison_analysis(df: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Compare regions (North vs South, etc.)."""

    filtered = apply_common_filters(df, filters)

    regions = filters.get("regions")
    if regions and len(regions) >= 2:
        filtered = filtered[filtered["region"].isin(regions)]
    else:
        # If not specified, compare top 2 regions by revenue
        top_regions = (
            filtered.groupby("region", as_index=False)
            .agg(revenue=("revenue", "sum"))
            .sort_values("revenue", ascending=False)
            .head(2)["region"]
            .tolist()
        )
        filtered = filtered[filtered["region"].isin(top_regions)]

    if filtered.empty:
        return {"answer_text": "No data found for the given filters.", "table": [], "chart": None}

    comp = (
        filtered.groupby("region", as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
        .sort_values("revenue", ascending=False)
    )

    fig = px.bar(
        comp,
        x="region",
        y="revenue",
        title="Revenue Comparison by Region",
        color="region",
    )
    fig.update_layout(xaxis_title="Region", yaxis_title="Revenue")

    return {
        "regions": comp["region"].tolist(),
        "revenue_by_region": comp.to_dict(orient="records"),
        "chart": fig,
    }

