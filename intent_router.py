from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from analytics.inventory import inventory_analysis
from analytics.products import product_performance_analysis, top_products_analysis
from analytics.promotions import promotion_effectiveness_analysis
from analytics.regions import regional_comparison_analysis, revenue_analysis
from analytics.sales import sales_trends_analysis
from llm.gemini_helper import build_gemini, llm_extract_intent, llm_summarize


AnalyticsFn = Callable[[pd.DataFrame, Dict[str, Any]], Dict[str, Any]]


@dataclass
class RoutedResult:
    intent_json: Dict[str, Any]
    analytics_payload: Dict[str, Any]
    summary: str


INTENT_TO_ANALYTICS: Dict[str, AnalyticsFn] = {
    "Top Products": top_products_analysis,
    "Revenue Analysis": revenue_analysis,
    "Regional Comparison": regional_comparison_analysis,
    "Promotion Effectiveness": promotion_effectiveness_analysis,
    "Inventory Analysis": inventory_analysis,
    "Stockout Detection": inventory_analysis,
    "Product Performance": product_performance_analysis,
    "Sales Trends": sales_trends_analysis,
}


def route_and_compute(
    df: pd.DataFrame,
    user_query: str,
    gemini_model: str = "gemini-1.5-flash",
    override_intent_json: Optional[Dict[str, Any]] = None,
) -> RoutedResult:
    """LLM-safe router.

    Steps:
    1) Use Gemini to extract intent JSON.
    2) Route to deterministic Pandas analytics function.
    3) Use Gemini only for summarization of already computed results.
    """

    gemini = build_gemini(model=gemini_model)

    intent_json = override_intent_json or llm_extract_intent(gemini, user_query)
    intent = (intent_json.get("intent") or "Revenue Analysis").strip()
    filters = intent_json.get("filters") or {}

    fn = INTENT_TO_ANALYTICS.get(intent)
    if fn is None:
        # Fallback: default revenue analysis
        fn = revenue_analysis

    payload = fn(df, filters)

    summary = llm_summarize(gemini, user_query, intent_json, payload)

    return RoutedResult(
        intent_json=intent_json,
        analytics_payload=payload,
        summary=summary,
    )

