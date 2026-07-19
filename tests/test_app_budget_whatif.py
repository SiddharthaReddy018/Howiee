"""
tests/test_app_budget_whatif.py
================================
Regression test for the §C.3/§G.3 gap in `src/app.py`'s Budget What-If tab:
the live slider's headline "Median revenue @ Xx" metric used to come from a
fresh, unclamped, un-isotonic-corrected `score_scenario` call -- so neither
the monotonic-budget-response guarantee nor the ROAS plausibility clamp
(both real for the static `predict.py` output) actually held for the
specific number shown on the interactive tab.

Uses Streamlit's `AppTest` harness (`streamlit.testing.v1`) to actually
execute `src/app.py` end-to-end against the repo's own `pickle/model.pkl` +
`./data`, drive the real slider widget, and read the real rendered metric --
this exercises the exact code path a user hits, not a reimplementation of it.

Requires `pickle/model.pkl` to already exist (`python src/train.py` first).
"""

from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

HERE = os.path.dirname(__file__)
REPO_ROOT = os.path.join(HERE, "..")
APP_PATH = os.path.join(REPO_ROOT, "src", "app.py")
MODEL_PATH = os.path.join(REPO_ROOT, "pickle", "model.pkl")

MULTIPLIERS = [0.25, 0.55, 0.85, 1.0, 1.37, 1.83, 2.15, 2.5]


def _median_metric_value(at) -> float:
    metrics = {m.label: m.value for m in at.metric}
    key = next(k for k in metrics if k.startswith("Median revenue"))
    return float(re.sub(r"[^\d.]", "", metrics[key]))


def _run_sweep(select_channel: str | None, select_type: str | None):
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at = at.run()
    assert not at.exception, f"app failed to load: {list(at.exception)}"

    if select_channel:
        sel = next(s for s in at.selectbox if s.label == "Channel")
        at = sel.set_value(select_channel).run()
    if select_type:
        sel = next(s for s in at.selectbox if s.label == "Campaign type")
        at = sel.set_value(select_type).run()

    values = []
    for m in MULTIPLIERS:
        # AppTest widget proxies are snapshots of one run -- must be
        # re-fetched from `at` after every `.run()`, not reused, or later
        # `.set_value()` calls silently apply to a stale element tree.
        slider = next(s for s in at.slider if "multiplier" in s.label.lower())
        at = slider.set_value(m).run()
        assert not at.exception, f"app raised at multiplier={m}: {list(at.exception)}"
        values.append(_median_metric_value(at))
    return values, at


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="pickle/model.pkl not built yet")
def test_budget_whatif_monotonic_all_scope():
    """§C.3 exit criterion: dragging the live slider up must never decrease
    the displayed median revenue, for the default (all-campaigns) scope."""
    values, _ = _run_sweep(None, None)
    assert all(values[i] <= values[i + 1] + 1e-6 for i in range(len(values) - 1)), values


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="pickle/model.pkl not built yet")
def test_budget_whatif_monotonic_small_flagged_scope():
    """The monotonic guarantee has to hold even for a small, heavily
    ROAS-clamped scope (bing/Audience -- zero revenue on every historical
    day, per docs/technical_documentation.md §10) where per-campaign clamp
    flags can flip on and off between budget levels. This is the scope that
    exposed the original bug: injecting the live multiplier into the
    isotonic fit on every rerun could make two independently-fit points
    inconsistent near the edges for exactly this kind of small/noisy scope."""
    values, at = _run_sweep("bing", "Audience")
    assert all(values[i] <= values[i + 1] + 1e-6 for i in range(len(values) - 1)), values

    md_text = " ".join(m.value for m in at.markdown)
    assert "historical envelope" in md_text, (
        "expected a §G.3 plausibility caveat to render for a scope with a known clamp violation"
    )


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="pickle/model.pkl not built yet")
def test_budget_whatif_baseline_delta_is_zero_at_1x():
    """At exactly the 1.0x baseline, the delta shown next to the metric
    must be (numerically) zero -- a basic sanity check that the interpolated
    "current" and "baseline" reads are the same underlying curve."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at = at.run()
    slider = next(s for s in at.slider if "multiplier" in s.label.lower())
    at = slider.set_value(1.0).run()
    metrics = {m.label: (m.value, m.delta) for m in at.metric}
    key = next(k for k in metrics if k.startswith("Median revenue"))
    _, delta_str = metrics[key]
    delta = float(re.sub(r"[^\d.\-]", "", delta_str.split(" ")[0]))
    assert abs(delta) < 1.0, f"expected ~0 delta at 1.0x baseline, got {delta_str!r}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
