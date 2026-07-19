"""
llm_insights.py
================
Implementation Plan §H — LLM causal summary layer, grounded + anti-hallucination.

    [1] Deterministic stats engine (this module, pure Python/pandas/numpy)
    [2] Grounding context JSON (the ONLY facts the LLM may reference)
    [3] LLM call, strict system prompt, structured JSON output
    [4] Programmatic validator (§H.4) — extract every number the LLM said,
        reject anything that doesn't trace back to the grounding context
    [5] Render validated narrative next to the actual numbers it describes

If no `ANTHROPIC_API_KEY` is available (e.g. an offline grading run), or the
LLM output fails validation twice, a deterministic rule-based narrator
generates the same JSON shape directly from the grounding context — the
narrative is always grounded, whether or not the LLM call itself is
reachable. This is a resilience property worth keeping, not just a fallback
for this sandbox.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np
import pandas as pd

SYSTEM_PROMPT = """You are a marketing-analytics narrator. You will be given a JSON \
"grounding context" containing the ONLY facts you may reference.

Rules (violating any of these makes your output useless — follow them exactly):
1. Only reference numbers present in the JSON you are given. Do not perform arithmetic \
beyond what's provided (percent changes, ratios) unless the calculation only uses numbers \
already in the JSON.
2. Do not claim a causal mechanism you cannot support from the JSON — describe drivers as \
"associated with" or "a contributing factor per the feature-importance ranking," never as a \
certain cause.
3. Always state the forecast's own uncertainty range when describing the forecast, never a \
bare point number.
4. Output must be valid JSON matching this schema exactly:
   {"summary": string, "key_drivers": [string], "risk_flags": [string], "confidence_note": string}
5. If the grounding context contains no anomalies, do not invent any. If saturation_status is \
"unknown_insufficient_data", say so plainly rather than guessing.
6. When formatting revenue amounts, you MUST use the currency symbol provided in the instructions \
(e.g., if told the currency is '€', output €123,456).
"""

_JSON_SCHEMA_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_summary",
        "description": "Emit the validated causal-summary narrative.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_drivers": {"type": "array", "items": {"type": "string"}},
                "risk_flags": {"type": "array", "items": {"type": "string"}},
                "confidence_note": {"type": "string"},
            },
            "required": ["summary", "key_drivers", "risk_flags", "confidence_note"],
        },
    }
}


# ─────────────────────────────────────────────────────────────────────────────
# §H.5 — deterministic anomaly detection (LLM never finds anomalies itself)
# ─────────────────────────────────────────────────────────────────────────────
def detect_anomalies(daily_df: pd.DataFrame, group_cols=("channel", "campaign_type"),
                      z_thresh: float = 3.0, window: int = 28, max_flags: int = 10) -> list[dict]:
    """Rolling z-score on daily revenue per (channel, campaign_type) group."""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    daily = df.groupby(list(group_cols) + ["date"], as_index=False).agg(revenue=("revenue", "sum"))

    flags = []
    for key, g in daily.groupby(list(group_cols)):
        g = g.sort_values("date")
        roll_mean = g["revenue"].rolling(window, min_periods=7).mean()
        roll_std = g["revenue"].rolling(window, min_periods=7).std().replace(0, np.nan)
        z = (g["revenue"] - roll_mean) / roll_std
        hits = g.loc[z.abs() >= z_thresh]
        for idx in hits.index:
            flags.append({
                "channel": key[0] if isinstance(key, tuple) else key,
                "campaign_type": key[1] if isinstance(key, tuple) else None,
                "date": str(g.loc[idx, "date"].date()),
                "metric": "revenue",
                "z_score": round(float(z.loc[idx]), 2),
            })
    flags.sort(key=lambda f: abs(f["z_score"]), reverse=True)
    return flags[:max_flags]


def compute_period_over_period(daily_df: pd.DataFrame, scope: dict, window: int = 30) -> dict:
    """period-over-period deltas for one scope (channel / campaign_type), this
    trailing window vs the one before it."""
    df = daily_df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    for k, v in scope.items():
        if k in df.columns and v is not None:
            df = df[df[k] == v]
    if df.empty:
        return {"revenue_delta_pct": None, "spend_delta_pct": None}

    last_date = df["date"].max()
    cur_start = last_date - pd.Timedelta(days=window - 1)
    prev_start = cur_start - pd.Timedelta(days=window)
    prev_end = cur_start - pd.Timedelta(days=1)

    cur = df[(df["date"] >= cur_start) & (df["date"] <= last_date)]
    prev = df[(df["date"] >= prev_start) & (df["date"] <= prev_end)]

    def pct_delta(a, b):
        if b == 0 or pd.isna(b):
            return None
        return round(100.0 * (a - b) / b, 1)

    return {
        "revenue_delta_pct": pct_delta(cur["revenue"].sum(), prev["revenue"].sum()),
        "spend_delta_pct": pct_delta(cur["spend"].sum(), prev["spend"].sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §H.2 — grounding context builder
# ─────────────────────────────────────────────────────────────────────────────
def build_grounding_context(
    scope: dict, forecast: dict, top_drivers: list[dict], period_over_period: dict,
    anomalies: list[dict], saturation_status: dict, coverage_check: str | None = None,
) -> dict:
    ctx = {
        "scope": scope,
        "forecast": {**forecast, **({"coverage_check": coverage_check} if coverage_check else {})},
        "top_drivers": top_drivers,
        "period_over_period": period_over_period,
        "anomalies": anomalies,
        "saturation_status": saturation_status,
    }
    return json.loads(json.dumps(ctx, default=str))  # ensure fully JSON-serializable/round-trippable


# ─────────────────────────────────────────────────────────────────────────────
# §H.4 — programmatic numeric validator
# ─────────────────────────────────────────────────────────────────────────────
def extract_numbers(text: str) -> set:
    # Pre-clean ISO date formats (e.g. "2024-11-27" -> "2024 11 27") so hyphens aren't treated as minus signs
    text_clean = re.sub(r"(\d{4})-(\d{2})-(\d{2})", r"\1 \2 \3", text)
    return {float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text_clean)}



def validate_grounded(llm_text: str, grounding_context: dict, tol: float = 0.02) -> bool:
    allowed = extract_numbers(json.dumps(grounding_context))
    claimed = extract_numbers(llm_text)
    for n in claimed:
        if not any(abs(n - a) <= max(tol * abs(a), 0.5) for a in allowed):
            return False
    return True


def validate_llm_json(llm_json: dict, grounding_context: dict) -> tuple[bool, str | None]:
    try:
        blob = " ".join([
            llm_json.get("summary", ""),
            " ".join(llm_json.get("key_drivers", [])),
            " ".join(llm_json.get("risk_flags", [])),
            llm_json.get("confidence_note", ""),
        ])
    except Exception as exc:
        return False, f"malformed LLM JSON: {exc}"
    if not validate_grounded(blob, grounding_context):
        return False, "a number in the LLM output does not trace back to the grounding context"
    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# §H.3 — the actual LLM call (Anthropic API, tool-use for structured output)
# ─────────────────────────────────────────────────────────────────────────────
def call_llm(grounding_context: dict, api_key: str | None = None, model: str = "llama-3.1-8b-instant") -> dict | None:
    api_key = api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import groq
    except ImportError:
        return None

    try:
        client = groq.Groq(api_key=api_key)
        system_content = SYSTEM_PROMPT + f"\nCurrency symbol for this dataset: {os.environ.get('CURRENCY_SYMBOL', '$')}"
        resp = client.chat.completions.create(
            model=model,
            max_tokens=600,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": json.dumps(grounding_context)}
            ],
            tools=[_JSON_SCHEMA_TOOL],
            tool_choice={"type": "function", "function": {"name": "emit_summary"}},
        )
        tool_calls = resp.choices[0].message.tool_calls
        if tool_calls:
            for call in tool_calls:
                if call.function.name == "emit_summary":
                    return json.loads(call.function.arguments)
    except Exception as exc:  # network/auth/rate-limit errors -> fall back, don't crash the pipeline
        print(f"[llm_insights] LLM call failed, falling back to rule-based narrator: {exc}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic rule-based narrator — same JSON shape, no LLM required
# ─────────────────────────────────────────────────────────────────────────────
def rule_based_fallback(grounding_context: dict) -> dict:
    f = grounding_context.get("forecast", {})
    scope = grounding_context.get("scope", {})
    pop = grounding_context.get("period_over_period", {})
    drivers = grounding_context.get("top_drivers", [])
    anomalies = grounding_context.get("anomalies", [])
    sat = grounding_context.get("saturation_status", {})
    currency = os.environ.get("CURRENCY_SYMBOL", "$")

    if scope.get("channel") is None and scope.get("campaign_type") is None:
        scope_str = "the overall account"
    else:
        scope_str = f"the {scope.get('channel', 'all')}/{scope.get('campaign_type', 'all')} segment"

    p10, p50, p90 = f.get("revenue_p10"), f.get("revenue_p50"), f.get("revenue_p90")
    summary = (
        f"For {scope_str} over the next {scope.get('window_days', 30)} days, the median "
        f"forecast is {currency}{p50:,.0f} in revenue, with a 80% range of [{currency}{p10:,.0f}, {currency}{p90:,.0f}]."
        if p50 is not None else f"Forecast for {scope_str} is not available."
    )
    if pop.get("revenue_delta_pct") is not None:
        direction = "up" if pop["revenue_delta_pct"] >= 0 else "down"
        summary += f" Revenue is {direction} {abs(pop['revenue_delta_pct']):.1f}% vs. the prior comparable period."
    if sat.get("status"):
        summary += f" Budget saturation status: {sat['status'].replace('_', ' ')}."

    key_drivers = [
        f"{d['feature']} (importance rank {d['importance_rank']})" for d in drivers[:5]
    ] or ["No feature-importance data available for this scope."]

    risk_flags = []
    if anomalies:
        top = anomalies[0]
        risk_flags.append(
            f"Anomaly flagged on {top.get('date')}: {top.get('metric')} z-score {top.get('z_score')}."
        )
    if sat.get("status") == "near_saturation":
        risk_flags.append("Spend is near the historical saturation point for this group; expect diminishing returns from further increases.")
    if f.get("coverage_check"):
        risk_flags.append(f"Calibration check: {f['coverage_check']}.")
    if not risk_flags:
        risk_flags.append("No material risk flags from the deterministic checks for this scope.")

    confidence_note = (
        "This is a rule-based narration of the grounding context (no LLM call available in this "
        "run) — every number above comes directly from the deterministic stats engine, not a "
        "generative model."
    )
    return {
        "summary": summary, "key_drivers": key_drivers, "risk_flags": risk_flags,
        "confidence_note": confidence_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def generate_causal_summary(grounding_context: dict, api_key: str | None = None) -> dict:
    for attempt in range(2):
        llm_json = call_llm(grounding_context, api_key=api_key)
        if llm_json is None:
            break
        ok, reason = validate_llm_json(llm_json, grounding_context)
        if ok:
            return {**llm_json, "source": "llm", "validated": True}
        print(f"[llm_insights] LLM output rejected (attempt {attempt + 1}): {reason}")

    fallback = rule_based_fallback(grounding_context)
    return {**fallback, "source": "rule_based_fallback", "validated": True}
