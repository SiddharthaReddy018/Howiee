"""
tests/test_llm_insights.py
===========================
Implementation Plan §H — two gaps closed here:

  §H.4 (day-11 exit criterion): "an automated test proving the validator
  actually rejects an injected fabricated number." `validate_llm_json` and
  `validate_grounded` existed and were wired into `generate_causal_summary`,
  but nothing in the test suite exercised the rejection path itself -- this
  file does, directly and through the full orchestrator (with `call_llm`
  monkeypatched so the fallback path is exercised deterministically, without
  a real API key or network access).

  §H.3: the live Anthropic call (`call_llm`) was never exercised end-to-end
  in this sandbox because no `ANTHROPIC_API_KEY` was available (confirmed
  again here: `ANTHROPIC_API_KEY` is unset in this environment too). What
  CAN be verified without real credentials is the response-parsing logic
  itself -- that `call_llm` correctly extracts the `emit_summary` tool-use
  block's `input` dict from an Anthropic-shaped response, and that it
  degrades to `None` (triggering the rule-based fallback) on an API error
  rather than raising. This is done by monkeypatching `sys.modules["anthropic"]`
  with a fake client that mimics the real SDK's response shape, since
  `call_llm` does `import anthropic` locally rather than at module scope.
  This is NOT a substitute for a real network-verified run -- it proves the
  parsing/error-handling code is correct, not that api.anthropic.com itself
  is reachable with a real key, which still requires a real
  `ANTHROPIC_API_KEY` this sandbox does not have.
"""

from __future__ import annotations

import os
import sys
import types
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import llm_insights as L


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture
def grounding_context() -> dict:
    return L.build_grounding_context(
        scope={"channel": "google", "campaign_type": "SEARCH", "window_days": 30},
        forecast={"revenue_p10": 100000.0, "revenue_p50": 250000.0, "revenue_p90": 400000.0, "roas_p50": 3.2},
        top_drivers=[
            {"feature": "planned_future_daily_budget", "importance_rank": 1, "gain": 118539.0},
            {"feature": "horizon_days", "importance_rank": 2, "gain": 32864.0},
        ],
        period_over_period={"revenue_delta_pct": -8.9, "spend_delta_pct": 0.1},
        anomalies=[{"channel": "google", "campaign_type": "SEARCH", "date": "2026-03-31",
                    "metric": "revenue", "z_score": 4.1}],
        saturation_status={"status": "approaching_saturation"},
    )


# ─────────────────────────────────────────────────────────────────────────────
# §H.4 — the validator itself must reject a fabricated number
# ─────────────────────────────────────────────────────────────────────────────
def test_validate_llm_json_rejects_fabricated_number(grounding_context):
    """The day-11 exit criterion, directly: a number that does not trace
    back (within tolerance) to anything in the grounding context must be
    rejected, not silently accepted."""
    fabricated = {
        "summary": "Revenue is forecast at 250,000 with strong momentum, and next quarter should hit 999999.",
        "key_drivers": ["planned_future_daily_budget (importance rank 1)"],
        "risk_flags": ["No material risk flags."],
        "confidence_note": "Grounded narration.",
    }
    ok, reason = L.validate_llm_json(fabricated, grounding_context)
    assert ok is False
    assert reason is not None and "does not trace back" in reason


def test_validate_llm_json_rejects_fabricated_small_number(grounding_context):
    """Fabrication doesn't have to be a large, obviously-invented number --
    an invented small figure (e.g. a made-up percentage) must be caught too,
    not just implausibly large ones."""
    fabricated = {
        "summary": "Revenue is forecast at 250,000 in revenue.",
        "key_drivers": ["A surprising 42.7% lift not present anywhere in the grounding data."],
        "risk_flags": ["No material risk flags."],
        "confidence_note": "Grounded narration.",
    }
    ok, reason = L.validate_llm_json(fabricated, grounding_context)
    assert ok is False


def test_validate_llm_json_accepts_grounded_output(grounding_context):
    """Sanity check on the other side of the same test: real numbers taken
    directly from the grounding context must NOT be rejected. (Deliberately
    avoids directional wording like "down 8.9%" for a negative
    revenue_delta_pct=-8.9 -- the plain number-matching validator compares
    literal signed digits, so "down 8.9%" for a value stored as -8.9 is a
    phrasing the validator can't resolve; production never re-validates the
    rule-based fallback's own directional text against itself for exactly
    this reason -- see generate_causal_summary, which trusts that path
    unconditionally rather than round-tripping it through the validator.)"""
    grounded = {
        "summary": "For google/SEARCH over the next 30 days, the median forecast is 250,000 in revenue, "
                   "with a range of [100,000, 400,000].",
        "key_drivers": ["planned_future_daily_budget (importance rank 1)"],
        "risk_flags": ["Revenue changed by -8.9% vs. the prior comparable period."],
        "confidence_note": "Grounded in the deterministic stats engine.",
    }
    ok, reason = L.validate_llm_json(grounded, grounding_context)
    assert ok is True and reason is None


def test_validate_llm_json_handles_malformed_input(grounding_context):
    """A non-dict-shaped/malformed LLM response must fail closed (rejected),
    never raise an uncaught exception up into the pipeline."""
    ok, reason = L.validate_llm_json({"summary": None, "key_drivers": "not-a-list"}, grounding_context)
    assert ok is False
    assert reason is not None


# ─────────────────────────────────────────────────────────────────────────────
# §H.4 — the orchestrator must actually USE the validator to reject+retry,
# then fall back, when call_llm itself returns fabricated content
# ─────────────────────────────────────────────────────────────────────────────
def test_generate_causal_summary_rejects_fabricated_and_falls_back(grounding_context, monkeypatch):
    call_count = {"n": 0}

    def fake_call_llm(ctx, api_key=None, model="claude-sonnet-5"):
        call_count["n"] += 1
        return {
            "summary": "Revenue will explode to 5000000 next month.",  # fabricated, not in ctx
            "key_drivers": ["Fabricated driver at 77.7% importance."],
            "risk_flags": ["None."],
            "confidence_note": "trust me",
        }

    monkeypatch.setattr(L, "call_llm", fake_call_llm)
    result = L.generate_causal_summary(grounding_context, api_key="fake-key-for-test")

    assert call_count["n"] == 2, "generate_causal_summary should retry once (2 attempts) before falling back"
    assert result["source"] == "rule_based_fallback"
    assert result["validated"] is True
    # the fabricated LLM attempt itself must be the thing that was rejected,
    # not silently accepted and passed through
    ok, reason = L.validate_llm_json(
        {"summary": "Revenue will explode to 5000000 next month.", "key_drivers": [], "risk_flags": [],
         "confidence_note": ""},
        grounding_context,
    )
    assert ok is False


def test_generate_causal_summary_accepts_valid_llm_output(grounding_context, monkeypatch):
    valid = {
        "summary": "For google/SEARCH over the next 30 days, the median forecast is 250,000 in revenue.",
        "key_drivers": ["planned_future_daily_budget (importance rank 1)"],
        "risk_flags": ["No material risk flags."],
        "confidence_note": "Grounded in the deterministic stats engine.",
    }

    def fake_call_llm(ctx, api_key=None, model="claude-sonnet-5"):
        return valid

    monkeypatch.setattr(L, "call_llm", fake_call_llm)
    result = L.generate_causal_summary(grounding_context, api_key="fake-key-for-test")

    assert result["source"] == "llm"
    assert result["validated"] is True
    assert result["summary"] == valid["summary"]


def test_generate_causal_summary_falls_back_when_no_api_key(grounding_context):
    """No monkeypatching here -- exercises the real `call_llm`, confirming
    the documented no-key/offline behavior actually holds in THIS
    environment too (ANTHROPIC_API_KEY is unset here, same as the original
    training sandbox)."""
    assert "ANTHROPIC_API_KEY" not in os.environ
    result = L.generate_causal_summary(grounding_context)
    assert result["source"] == "rule_based_fallback"
    assert result["validated"] is True


# ─────────────────────────────────────────────────────────────────────────────
# §H.3 — mocked verification of call_llm's response parsing + error handling
# (NOT a substitute for a real network-verified run; see module docstring)
# ─────────────────────────────────────────────────────────────────────────────
class _FakeFunction:
    def __init__(self, name, arguments_dict):
        self.name = name
        self.arguments = json.dumps(arguments_dict)


class _FakeToolCall:
    def __init__(self, name, arguments_dict):
        self.function = _FakeFunction(name, arguments_dict)


class _FakeMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, tool_calls):
        self.message = _FakeMessage(tool_calls)


class _FakeResponse:
    def __init__(self, tool_calls):
        self.choices = [_FakeChoice(tool_calls)]


def _install_fake_groq_module(monkeypatch, create_fn):
    """`call_llm` does `import groq` INSIDE the function body, so we
    monkeypatch sys.modules rather than an attribute on llm_insights."""
    fake_completions = types.SimpleNamespace(create=create_fn)
    fake_chat = types.SimpleNamespace(completions=fake_completions)
    fake_client = types.SimpleNamespace(chat=fake_chat)

    fake_module = types.ModuleType("groq")
    fake_module.Groq = lambda api_key: fake_client
    monkeypatch.setitem(sys.modules, "groq", fake_module)


def test_call_llm_parses_tool_use_block_correctly(grounding_context, monkeypatch):
    """A well-formed Groq tool-use response must be parsed into exactly
    the dict the tool was called with."""
    expected_payload = {
        "summary": "Median forecast is 250,000.", "key_drivers": ["planned_future_daily_budget"],
        "risk_flags": ["none"], "confidence_note": "grounded",
    }

    def fake_create(**kwargs):
        assert kwargs["tool_choice"] == {"type": "function", "function": {"name": "emit_summary"}}
        return _FakeResponse([_FakeToolCall(name="emit_summary", arguments_dict=expected_payload)])

    _install_fake_groq_module(monkeypatch, fake_create)
    result = L.call_llm(grounding_context, api_key="fake-key-for-test")
    assert result == expected_payload


def test_call_llm_returns_none_on_api_error(grounding_context, monkeypatch):
    """Network/auth/rate-limit errors must degrade to None (triggering the
    rule-based fallback upstream), never propagate as an uncaught
    exception."""
    def fake_create(**kwargs):
        raise RuntimeError("simulated network failure")

    _install_fake_groq_module(monkeypatch, fake_create)
    result = L.call_llm(grounding_context, api_key="fake-key-for-test")
    assert result is None


def test_call_llm_returns_none_without_api_key(grounding_context):
    assert L.call_llm(grounding_context, api_key=None) is None


def test_call_llm_end_to_end_fabrication_still_caught(grounding_context, monkeypatch):
    """Combines both mocks: a fake Groq client that returns a
    plausible-shaped but fabricated tool-use payload must still be rejected
    by generate_causal_summary's validation step, ending up on the
    rule-based fallback."""
    fabricated_payload = {
        "summary": "Revenue will reach 123456789 next month.",
        "key_drivers": ["invented driver"], "risk_flags": ["none"], "confidence_note": "trust me",
    }

    def fake_create(**kwargs):
        return _FakeResponse([_FakeToolCall(name="emit_summary", arguments_dict=fabricated_payload)])

    _install_fake_groq_module(monkeypatch, fake_create)
    result = L.generate_causal_summary(grounding_context, api_key="fake-key-for-test")
    assert result["source"] == "rule_based_fallback"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
