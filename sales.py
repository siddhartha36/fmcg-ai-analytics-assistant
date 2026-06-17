from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import plotly.express as px

from utils.helpers import apply_common_filters, latest_month_window, safe_divide


def revenue_by_region_chart(df: pd.DataFrame):
    agg = (
        df.groupby("region", as_index=False)
        .agg(revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False)
    )
    fig = px.bar(
        agg,
        x="region",
        y="revenue",
        title="Revenue by Region",
    )
    fig.update_layout(xaxis_title="Region", yaxis_title="Revenue")
    return fig, agg


def sales_trends_analysis(df: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Sales trends over time.

    Interprets:
    - timeframe='this_month' -> filter to latest month
    - otherwise aggregates all time
    """

    filtered = apply_common_filters(df, filters)

    # Ensure sorted
    trend_df = (
        filtered.groupby("week_start_date", as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
        .sort_values("week_start_date")
    )

    if trend_df.empty:
        return {"answer_text": "No data found for the given filters.", "table": [], "chart": None}

    fig = px.line(
        trend_df,
        x="week_start_date",
        y="revenue",
        markers=True,
        title="Sales Trend Over Time (Revenue)",
    )
    fig.update_layout(xaxis_title="Week", yaxis_title="Revenue")

    latest_week = trend_df["week_start_date"].max()
    latest = trend_df[trend_df["week_start_date"] == latest_week]
    prev = trend_df[trend_df["week_start_date"] < latest_week]

    mom_change = None
    if not latest.empty and not prev.empty:
        prev_last = prev[prev["week_start_date"] == prev["week_start_date"].max()]
        if not prev_last.empty:
            mom_change = float(latest["revenue"].iloc[0] - prev_last["revenue"].iloc[0])

    return {
        "metric": "revenue",
        "time_range": {
            "start": str(trend_df["week_start_date"].min().date()),
            "end": str(trend_df["week_start_date"].max().date()),
        },
        "latest_week_revenue": float(latest["revenue"].iloc[0]),
        "mom_change_revenue": mom_change,
        "table": trend_df.to_dict(orient="records"),
        "chart": fig,
    }


def revenue_analysis(df: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Revenue analysis.

    Supports:
    - Which product generated the highest revenue?
    - Which category generated the most revenue?
    - Generic revenue totals
    """

    filtered = apply_common_filters(df, filters)

    if filtered.empty:
        return {"answer_text": "No data found for the given filters.", "table": [], "chart": None}

    # Category/product intent is inferred from filters
    product_name: Optional[str] = filters.get("product_name")
    category: Optional[str] = filters.get("category")

    if product_name:
        agg = (
            filtered.groupby("product_name", as_index=False)
            .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
        )
        row = agg.sort_values("revenue", ascending=False).head(1)
        fig = px.bar(row, x="product_name", y="revenue", title="Revenue for Selected Product")
        fig.update_layout(xaxis_title="Product", yaxis_title="Revenue")
        return {
            "top_entity": product_name,
            "revenue": float(row["revenue"].iloc[0]),
            "units_sold": float(row["units_sold"].iloc[0]),
            "table": agg.to_dict(orient="records"),
            "chart": fig,
        }

    if category:
        agg = (
            filtered.groupby("category", as_index=False)
            .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
        )
        row = agg.sort_values("revenue", ascending=False).head(1)
        fig = px.bar(row, x="category", y="revenue", title="Revenue for Selected Category")
        fig.update_layout(xaxis_title="Category", yaxis_title="Revenue")
        return {
            "top_entity": category,
            "revenue": float(row["revenue"].iloc[0]),
            "units_sold": float(row["units_sold"].iloc[0]),
            "table": agg.to_dict(orient="records"),
            "chart": fig,
        }

    # Default: highest product by revenue
    top_products = (
        filtered.groupby(["product_id", "product_name"], as_index=False)
        .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"))
        .sort_values("revenue", ascending=False)
    )

    top = top_products.head(1)
    fig = px.bar(
        top_products.head(10),
        x="product_name",
        y="revenue",
        title="Top Products by Revenue",
    )
    fig.update_layout(xaxis_title="Product", yaxis_title="Revenue")

    return {
        "top_product": {
            "product_name": str(top["product_name"].iloc[0]),
            "product_id": str(top["product_id"].iloc[0]),
            "revenue": float(top["revenue"].iloc[0]),
            "units_sold": float(top["units_sold"].iloc[0]),
        },
        "table": top_products.head(10).to_dict(orient="records"),
        "chart": fig,
    }

