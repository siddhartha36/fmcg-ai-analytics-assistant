from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from langchain_google_genai import ChatGoogleGenerativeAI


def _get_google_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError(
            "Missing GOOGLE_API_KEY. Set it via .env or Streamlit Cloud environment variables."
        )
    return key


def build_gemini(model: str = "gemini-1.5-flash") -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=_get_google_api_key(),
        temperature=0.0,
    )


def robust_json_loads(text: str) -> Dict[str, Any]:
    """Best-effort JSON extraction from LLM responses."""
    text = text.strip()

    # If the model wrapped JSON in markdown, remove fences.
    if text.startswith("```"):
        text = text.strip("`")
        # Remove leading language tags if any
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                text = rest

    # Find first '{' and last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start : end + 1]
    else:
        chunk = text

    return json.loads(chunk)


def llm_extract_intent(gemini: ChatGoogleGenerativeAI, user_query: str) -> Dict[str, Any]:
    """Ask Gemini to extract an intent JSON.

    Output schema:
      {"intent": "", "filters": {}, "metric": ""}

    Safety note: Gemini is used only for intent parsing.
    """

    system = (
        "You are an FMCG analytics assistant.\n"
        "Your task is to convert the user's natural-language question into a JSON object that matches this schema exactly:\n\n"
        "{\n  \"intent\": \"Top Products | Revenue Analysis | Regional Comparison | Promotion Effectiveness | "
        "Inventory Analysis | Stockout Detection | Product Performance | Sales Trends\",\n  \"filters\": {\n    \"regions\": [string],\n    \"region\": string,\n    \"categories\": [string],\n    \"category\": string,\n    \"products\": [string],\n    \"product_name\": string,\n    \"stores\": [string],\n    \"store_name\": string,\n    \"timeframe\": string,\n    \"time_window\": string\n  },\n  \"metric\": string\n}\n\n"
        "Rules:\n"
        "- Select exactly ONE intent from the list.\n"
        "- If the user asks for 'this month', 'current', or similar, set filters.timeframe='this_month'.\n"
        "- For 'compare X vs Y', set filters.regions=[X,Y] and intent='Regional Comparison'.\n"
        "- For promotions effectiveness, set intent='Promotion Effectiveness' and include any promo_type keywords in filters (if obvious).\n"
        "- If no filters are specified, use an empty object {}.\n"
        "- metric should be a short label (e.g., 'revenue', 'units_sold', 'stockouts').\n"
        "Return ONLY valid JSON. No extra keys."
    )

    prompt = f"User question: {user_query}\n\nReturn intent JSON now."

    resp = gemini.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ])

    data = robust_json_loads(resp.content)

    # Minimal cleanup / ensure keys exist
    if "intent" not in data:
        data["intent"] = "Revenue Analysis"
    if "filters" not in data or data["filters"] is None:
        data["filters"] = {}
    if "metric" not in data or data["metric"] is None:
        data["metric"] = "revenue"

    return data


def llm_summarize(gemini: ChatGoogleGenerativeAI, user_query: str, intent_json: Dict[str, Any], result_payload: Dict[str, Any]) -> str:
    """Summarize computed analytics results.

    This function must NOT be used to compute metrics; it only describes results.
    """

    system = (
        "You are a business analytics assistant for FMCG. "
        "You will be given a user's question, an intent JSON, and a deterministic result payload computed from data. "
        "Write a concise, clear business response. "
        "Do not introduce numbers that are not present in the result payload. "
        "If the result payload indicates empty/no data, say so."
    )

    prompt = {
        "user_query": user_query,
        "intent": intent_json,
        "result_payload": result_payload,
        "instructions": [
            "Prefer bullet points when comparing multiple regions/products.",
            "Explain what the numbers mean in plain business language.",
        ],
    }

    resp = gemini.invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ])

    return resp.content.strip()

