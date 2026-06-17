from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import plotly.express as px

from utils.helpers import apply_common_filters


def promotion_effectiveness_analysis(df: pd.DataFrame, filters: Dict[str, Any]) -> Dict[str, Any]:
    """Promotion effectiveness analysis.

    Compares metrics where promotion_flag=True vs False.
    Also aggregates by promotion_type for ranking.
    """

    filtered = apply_common_filters(df, filters)

    if filtered.empty:
        return {"answer_text": "No data found for the given filters.", "table": [], "chart": None}

    promo = filtered[filtered["promotion_flag"] == True].copy()
    no_promo = filtered[filtered["promotion_flag"] == False].copy()

    def _agg(d: pd.DataFrame):
        return {
            "revenue": float(d["revenue"].sum()),
            "units_sold": float(d["units_sold"].sum()),
            "rows": int(len(d)),
        }

    promo_agg = _agg(promo)
    nopromo_agg = _agg(no_promo)

    revenue_uplift = None
    if nopromo_agg["revenue"] != 0:
        revenue_uplift = (promo_agg["revenue"] - nopromo_agg["revenue"]) / nopromo_agg["revenue"]

    # Ranking by promotion_type
    if not promo.empty:
        by_type = (
            promo.groupby("promotion_type", as_index=False)
            .agg(revenue=("revenue", "sum"), units_sold=("units_sold", "sum"), discount_avg=("discount_pct", "mean"))
            .sort_values("revenue", ascending=False)
        )
        top_types = by_type.head(5)
        fig = px.bar(
            top_types,
            x="promotion_type",
            y="revenue",
            title="Promotion Impact (Top Promotion Types by Revenue)",
        )
        fig.update_layout(xaxis_title="Promotion Type", yaxis_title="Revenue")
        top_types_records = by_type.head(5).to_dict(orient="records")
    else:
        fig = None
        top_types_records = []

    compare_df = pd.DataFrame(
        [
            {"promotion": "Promotion", **promo_agg},
            {"promotion": "No Promotion", **nopromo_agg},
        ]
    )

    chart_compare = px.bar(
        compare_df,
        x="promotion",
        y="revenue",
        title="Promotion vs No Promotion (Revenue)",
    )
    chart_compare.update_layout(xaxis_title="", yaxis_title="Revenue")

    return {
        "promotion": {
            "promo_revenue": promo_agg["revenue"],
            "no_promo_revenue": nopromo_agg["revenue"],
            "revenue_uplift_pct": revenue_uplift * 100 if revenue_uplift is not None else None,
            "promo_units_sold": promo_agg["units_sold"],
            "no_promo_units_sold": nopromo_agg["units_sold"],
        },
        "top_promotion_types": top_types_records,
        "table": compare_df.to_dict(orient="records"),
        "chart": chart_compare,
        "type_chart": fig,
    }

