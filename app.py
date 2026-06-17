from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

from analytics.inventory import inventory_analysis
from analytics.products import product_performance_analysis, top_products_analysis
from analytics.promotions import promotion_effectiveness_analysis
from analytics.regions import regional_comparison_analysis, revenue_analysis
from analytics.sales import sales_trends_analysis
from data.loader import load_dataset
from llm.gemini_helper import build_gemini, llm_extract_intent, llm_summarize
from llm.intent_router import route_and_compute


st.set_page_config(page_title="FMCG AI Analytics Assistant", page_icon="📊", layout="wide")


@st.cache_data(show_spinner=False)
def get_df() -> pd.DataFrame:
    return load_dataset()


def _render_chat() -> None:
    df = get_df()

    st.title("📊 FMCG Business Analytics Assistant")
    st.caption("Ask questions about sales, promotions, inventory, regions, and products.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Seed assistant message
    if not st.session_state.messages:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Ask me about revenue, promotions, inventory/stockouts, regional performance, or product performance.",
            }
        )

    # Display chat
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Type your business question...")

    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking & analyzing..."):
            try:
                routed = route_and_compute(df=df, user_query=user_query)
                analytics_payload = routed.analytics_payload

                # Render optional chart
                chart = analytics_payload.get("chart")
                if chart is not None:
                    st.plotly_chart(chart, use_container_width=True)

                # Also show small tables when present
                for key in ("table", "top_products", "underperforming_products", "top_stores", "top_regions", "top_promotion_types"):
                    if key in analytics_payload and isinstance(analytics_payload[key], list):
                        data_list = analytics_payload[key]
                        if data_list:
                            st.dataframe(pd.DataFrame(data_list).head(10), use_container_width=True)

                st.write(routed.summary)

                # Save assistant message
                st.session_state.messages.append({"role": "assistant", "content": routed.summary})

            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})


def _render_sales_dashboard() -> None:
    df = get_df()

    st.header("Sales Dashboard")
    # Revenue by Region
    from analytics.sales import revenue_by_region_chart

    region_fig, region_agg = revenue_by_region_chart(df)
    st.plotly_chart(region_fig, use_container_width=True)

    # Sales Trend
    st.subheader("Sales Trend Over Time")
    trend_payload = sales_trends_analysis(df, filters={})
    chart = trend_payload.get("chart")
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)

    if "latest_week_revenue" in trend_payload:
        st.metric("Latest Revenue", value=trend_payload["latest_week_revenue"])


def _render_promotion_dashboard() -> None:
    df = get_df()
    st.header("Promotion Dashboard")
    payload = promotion_effectiveness_analysis(df, filters={})

    chart = payload.get("chart")
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("Promotion vs No Promotion")
    promo = payload.get("promotion", {})
    if promo:
        st.write(promo)


def _render_inventory_dashboard() -> None:
    df = get_df()
    st.header("Inventory Dashboard")
    payload = inventory_analysis(df, filters={})

    chart = payload.get("chart")
    if chart is not None:
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("Stockout Summary")
    st.write(payload.get("stockouts", {}))


def _render_regional_dashboard() -> None:
    df = get_df()
    st.header("Regional Dashboard")

    payload = regional_comparison_analysis(df, filters={})
    fig = payload.get("chart")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Region Revenue Ranking")
    if "revenue_by_region" in payload:
        st.dataframe(pd.DataFrame(payload["revenue_by_region"]).head(10), use_container_width=True)


def _render_product_dashboard() -> None:
    df = get_df()
    st.header("Product Performance Dashboard")

    st.subheader("Top Products")
    top_payload = top_products_analysis(df, filters={"timeframe": "this_month"})
    fig = top_payload.get("chart")
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Underperforming Products")
    under_payload = product_performance_analysis(df, filters={"timeframe": "this_month"})
    fig2 = under_payload.get("chart")
    if fig2 is not None:
        st.plotly_chart(fig2, use_container_width=True)


def main() -> None:
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Go to",
            [
                "Chat Assistant",
                "Sales Dashboard",
                "Promotion Dashboard",
                "Inventory Dashboard",
                "Regional Dashboard",
            ],
            index=0,
        )

    if page == "Chat Assistant":
        _render_chat()
    elif page == "Sales Dashboard":
        _render_sales_dashboard()
    elif page == "Promotion Dashboard":
        _render_promotion_dashboard()
    elif page == "Inventory Dashboard":
        _render_inventory_dashboard()
    elif page == "Regional Dashboard":
        _render_regional_dashboard()


if __name__ == "__main__":
    main()

