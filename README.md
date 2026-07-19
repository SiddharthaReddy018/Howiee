# AIgnition 3.0 — Probabilistic Multi-Channel Ad Revenue Forecasting

**Team TechBlazers** · NetElixir AIgnition 3.0 Hackathon

Campaign-level Google Ads / Meta Ads / Bing Ads data in → calibrated
30/60/90-day probabilistic revenue forecasts, reconciled across a
campaign → campaign_type → channel → total hierarchy, with a budget
what-if layer and a grounded AI narrative, out.

**Start here:** 
- [`docs/plain_english_summary.md`](docs/plain_english_summary.md) — a non-technical overview of what this project does and why it matters.
- [`docs/summary.md`](docs/summary.md) — a 1-2 page executive summary (methodology, headline numbers, top limitations). 

The full writeup (every design decision's reasoning, real numbers from this repo's own data, assumptions, and the complete known-limitations list) is [`docs/technical_documentation.md`](docs/technical_documentation.md); the module map is [`docs/architecture.md`](docs/architecture.md).

## Quickstart

```bash
# Requires Python 3.10+ (Tested with Python 3.12.x)
pip install -r requirements.txt --break-system-packages   # or use a venv


# 1. Train (offline; produces pickle/model.pkl) -- ~950s on a normal machine;
#    run in the foreground, not backgrounded, and expect it to take a while --
#    §8c's per-level calibration re-audit alone reconciles every holdout
#    origin_date/horizon snapshot (see the dev-speed flag below if you're
#    iterating and don't need the full number every time).
python3 src/train.py --data-dir ./data --model-out ./pickle/model.pkl

# 1b. Faster dev-only iteration (NOT for final numbers -- both flags default
#     to the full/unreduced behavior if omitted; every number in these docs
#     was produced with them omitted):
python3 src/train.py --data-dir ./data --model-out ./pickle/model.pkl \
    --origin-stride 2 --calibration-max-snapshots 20

# 2. Run inference (what the grading harness calls)
bash run.sh ./data ./pickle/model.pkl ./output/predictions.csv

# 3. Explore the frontend
streamlit run src/app.py

# 4. Run the schema-robustness test suite
pytest tests/ -v

# 5. Quick data-health check on any directory, without training
python3 src/validate.py --data-dir ./data
```

**This round adds:** CatBoost as a second point-model family compared
honestly against LightGBM Tweedie/hurdle (§5b) — an equal-weight blend of
all three currently wins; a genuine learned-scorecaster D-term for
Conformal PID (§6b), alongside the original derivative-of-error-signal
proxy; an MPC-style rolling-horizon budget-reallocation backtest (§9b); and
four extended funnel features — CPM, CPA, `reach`, and Google's
`video_views`, both previously ingested and then silently dropped
(§4). Every number below was produced by a full, unreduced training run
(`python3 src/train.py --data-dir ./data --model-out ./pickle/model.pkl`,
no dev-speed flags) — this is the true headline WAPE/coverage/ROAS state of
the shipped `pickle/model.pkl`, not a reduced-settings verification pass.

`run.sh` needs no network access and no API key — the LLM summary layer
(`src/llm_insights.py`) falls back to a deterministic, equally-grounded
rule-based narrator whenever `ANTHROPIC_API_KEY` isn't set. If you do want
live LLM narration, copy `.env.example` to `.env` and set the key, or
`export ANTHROPIC_API_KEY=...` before running — `predict.py`/`run.sh` will
pick it up automatically with no code changes. To confirm the live path is
actually working (not just falling back) before a demo, run
`python3 src/verify_live_llm.py` after exporting the key: it makes one real
call and reports `source` (`llm` vs. `rule_based_fallback`), whether the
grounding validator accepted it, and latency.

## What's in `output/` after a run

| File | Contents |
|---|---|
| `data_quality_report.txt` | This run's schema-mapping/ingestion + Pandera validation report (`validate.py`, run.sh step 0) |
| `predictions.csv` | Per (campaign, horizon) quantile forecast, ROAS, saturation status, plausibility flag |
| `reconciled_hierarchy.csv` | Per (node, horizon) reconciled forecast — campaign → campaign_type → channel → total, with the coherence-error check |
| `data_health.json` | This run's live schema-mapping/ingestion report |
| `reliability.json` | CV protocol comparison + final-holdout metrics + calibration reliability diagram |
| `hill_curves.json` | Fitted Hill saturation curves per (channel, campaign_type), with fit R² and rejection reason where applicable |
| `mpc_reallocation_backtest.json` | MPC vs. open-loop rolling-horizon budget-reallocation backtest (§9b) |
| `causal_summary.json` | Grounded AI narrative for the account total + each channel |

## Headline results (this repo's data, see `docs/technical_documentation.md` for full detail)


- Final holdout (never touched during tuning): **36.5% WAPE** on the median
  30/60/90-day forecast, calibrated interval coverage **90.5%/94.9%** at
  nominal 80%/90% (slightly conservative, the safer direction for budget
  planning). §8c separately checks whether that same coverage survives
  hierarchical reconciliation — it doesn't uniformly (channel and
  individual-campaign bands under-cover post-reconciliation; account-total
  and campaign-type bands stay conservative) — see the technical
  documentation for the measured numbers at each level.
- **Skill vs. a naive baseline (§7a of the technical documentation):** a trivial "continue at the recent
  28-day pace" forecast — never trained, just trailing daily revenue rate ×
  horizon — scores **126.8% WAPE** on this same holdout. The production
  model cuts that error by **71.2%** (62.2% at the 30-day horizon, rising to
  75.7% at 90 days, since a flat extrapolation of recent pace gets worse,
  not better, further out while the model actually accounts for the planned
  budget change). Reported because "36.5% WAPE" means little on its own —
  this is the number that answers "compared to what?"

- Hierarchical reconciliation coherence error: **0.000000** — channel
  totals sum exactly to the grand total. Interval reconciliation now uses
  the real `hierarchicalforecast.methods.Conformal` reconciler wherever
  this project's own data supports its joint-alignment requirement
  (total + channel), and a marginal per-node split-conformal correction
  elsewhere (§8, §E.1).
- The full `run.sh` pipeline was verified end-to-end against a synthetically
  mangled version of the Bing file (renamed/dropped/shuffled/junk columns) —
  identical output shape, zero crashes, every degradation logged, **and**
  quantitatively near-identical CRPS/WAPE vs. the clean file (§G.2.3). See
  `tests/test_schema_robustness.py`, `tests/test_cv_mangled_schema.py`, and
  §3 of the technical documentation.
- Numeric fields also survive real-world **value** formatting, not just
  column renaming: currency symbols, thousands separators, accounting-style
  `(123.45)` negatives, and a trailing `%` are cleaned before parsing, so a
  correctly-named-but-currency-formatted `spend`/`revenue` column can no
  longer silently collapse to an all-zero, Pandera-passing "success" — see
  `tests/test_numeric_format_robustness.py` and §3 of the technical
  documentation for the bug this replaced.
- Adstock/carryover spend features (a standard MMM technique) were built,
  retrained, and measured — real holdout WAPE went 36.1% → 36.8%, so they
  were **not** shipped. Documented as a negative result rather than deleted;
  see §4 of the technical documentation.
- Each channel's AI causal summary now surfaces its **own** SHAP-ranked
  drivers instead of one account-wide ranking pasted into every scope (the
  previous behavior — real numbers, but identical across `total`/`bing`/
  `google`/`meta`); see §5c and `tests/test_shap_per_channel_drivers.py`.
- **Extended funnel features (§4):** two real columns already present in
  this project's own raw files — Meta's `reach`, Google's
  `metrics_video_views` — were being ingested and then silently thrown away
  (`ignored_columns`) with no canonical slot to land in. Four features now
  build on them: `cpm_roll_28`/`cpa_roll_28` (cost-efficiency, from
  already-tracked spend/impressions/conversions), and
  `frequency_roll_28`/`video_view_rate_roll_28` (reach- and video-based
  funnel-stage signals — `frequency_roll_28` in particular is the concrete
  signal that separates brand-awareness spend from performance spend in a
  way `campaign_type` alone can't). `ctr_roll_28` and `video_view_rate_roll_28`
  both land in the top 10 features by mean |SHAP| on this run (ranks 9 and
  10). Finding and fixing these also surfaced a
  real gap-fill bug in the daily-reindex step (a channel that never reports
  a field at all could still get that field zero-filled on reindexed
  calendar gap days) — invisible before because no pre-existing optional
  field was ever whole-file-missing in this project's own data; fixed and
  covered by `tests/test_funnel_features.py`.
- **Ensemble diversity: CatBoost as a second point-model family (§5b).**
  The Tweedie-vs-hurdle ablation is now a 3-way comparison — LightGBM
  Tweedie, LightGBM hurdle, and CatBoost's own independent Tweedie
  implementation (different library: ordered boosting, native
  ordered-target-statistic categoricals) — plus a 4th, implicit candidate:
  the simple equal-weight blend of all three, the concrete test of whether
  these models' errors are uncorrelated enough for averaging to help.


  On this run: Tweedie pinball(q0.5)=6,112, hurdle=5,947, CatBoost=5,895,
  **equal-weight blend of all three=5,846 — the blend wins**, by +4.4% vs.
  Tweedie alone — concrete evidence the three candidates' errors are
  uncorrelated enough for averaging to help, not just an assumption that
  it does.

- Adaptive Conformal Inference (§D.2, optional) shows real measured value on
  this project's own holdout timeline: static CQR runs a bit conservative
  post-warmup, ACI tracks it back toward nominal with a correspondingly
  narrower interval — reported for comparison, not the shipped default
  (§6a).
- **Conformal PID control (§6b), now with a genuine learned scorecaster.**
  Beyond the original derivative-of-error-signal proxy D-term
  (Angelopoulos, Candès & Tibshirani, NeurIPS 2023's own P+I terms, a
  scoped-down D term), `run_conformal_pid_control_learned_scorecaster` adds
  the paper's actual D-term design: a small ridge-AR model refit online, at
  every step, to forecast the next nonconformity score directly — the two
  variants are now compared honestly on the identical holdout sequence.


  Measured, not assumed, and now with two D-term variants compared
  honestly: derivative-proxy PID gets coverage 89.9% at mean width 48,997;
  the genuine learned-scorecaster variant matches that same 89.9% coverage
  but at nearly double the width (94,269) on this real holdout — a
  legitimate, mixed finding (the more sophisticated D-term is not
  automatically the more efficient one here), reported as such. Both stay
  optional, reported for comparison, not shipped as the default, exactly
  like ACI itself.

- **A cross-channel budget allocator (§9a),** built entirely on the
  already-fitted Hill curves — not a retrain, pure post-processing. Given a
  fixed total budget, it recommends the split across channels that
  maximizes predicted revenue, solved with dynamic programming (not a naive
  greedy walk, since the fitted curves are S-shaped, not concave — see
  `tests/test_budget_optimizer.py`). Hill curves are recency-weighted
  (§9, §F re-audit — a daily point 45 days old carries half the weight of
  today's) — including, as of this round, the flat-fallback line a group
  falls back to when it doesn't clear the real-fit gates, not just the
  curve-fit path (a real gap found and fixed while building §9b below).


  On this run: reallocating the account's current 5,757/day according to
  the fitted curves instead of today's actual split is predicted to raise
  daily revenue **+56.3%** (30,475 → 47,642) — with no extra budget at
  all. It also reports **marginal ROAS** per group (the three funded
  Google groups cluster at ~4.4–4.5x marginal ROAS, textbook marginal-return
  equalization; the three funded `meta` groups instead sit at ~97–98% of
  their 4x-historical-spend safety cap, correctly held there rather than
  extrapolated further), supports an optional **minimum-blended-ROAS
  floor** (reusing the DP's own already-computed frontier, not a
  re-solve), and reports an approximate uncertainty band on the predicted
  revenue rather than a bare point estimate.

- **MPC-style rolling-horizon budget reallocation backtest (§9b), new this
  round.** The full closed-loop extension of the Hill-curve recency
  weighting above (Pathak, Shyamal, Mhasker & Swartz 2026, "Learning to
  Spend"): does periodically re-fitting the curves and re-solving the
  allocation, as real historical data arrives, earn back more revenue than
  deciding once and running that plan for the whole horizon? Backtested
  honestly against a frozen open-loop baseline on this account's own
  historical timeline, with planned-vs-realized spend execution noise
  modeled directly, and both methods scored against an identical
  retrospective ground truth so only the allocation differs.


  On this run: re-fitting the curves and re-solving the allocation every
  30 days, backtested across a real 90-day historical window, beats a
  frozen one-shot plan by **+5.3%** on average (13,318 → 14,026/day realized
  revenue) — a genuine, mixed result: it wins in 2 of the 3 backtested
  windows and loses in the third (11,081 vs. 11,314), reported as-is rather
  than smoothed over.

- **New: reconciled-interval calibration checked at every hierarchy level
  (§8c),** not just the base per-campaign-row level §6 already covers.
  Coherence (exact, every run) and calibration are separate properties
  (Principato et al. 2024) — this was a genuinely open question until now.
  `total` and `campaign_type` stay conservative post-reconciliation, but
  `channel` and the reconciled `campaign` level measurably under-cover — a
  real, quantified gap between this project's point and interval
  reconciliation, reported rather than assumed away. See
  `src/reconciliation.py::evaluate_reconciled_calibration` and
  `tests/test_reconciled_calibration_by_level.py`.

## Repository layout

```
src/
  schema_mapper.py         §A  schema robustness / ingestion
  feature_engineering.py   §B  origin-based feature engineering
  modeling.py              §C/D/G/H.1  Tweedie + quantile ensemble, hurdle-model
                                ablation, CV, CQR, SHAP, metrics
  adaptive_conformal.py    §D.2 (optional)  Adaptive Conformal Inference
  reconciliation.py        §E  hierarchical reconciliation (two-tier probabilistic)
  budget_curves.py         §F  Hill saturation curves
  sanity_clamps.py         §G.3  business-plausibility bounds
  llm_insights.py          §H  grounded LLM causal summary
  train.py                 offline training entry point
  generate_features.py     run.sh stage 1
  predict.py                run.sh stage 2
  validate.py               run.sh stage 0 / standalone data-health CLI
  app.py                    §I  Streamlit frontend (data health -> forecast ->
                                 budget what-if -> breakdown -> AI summary -> reliability)
tests/
  test_schema_robustness.py       §A / §G.2.3 (qualitative mangled-file pass)
  test_cv_mangled_schema.py       §G.2.3 (quantitative CRPS/WAPE comparison)
  test_app_budget_whatif.py       §C.3/§G.3 live what-if monotonicity + clamps
  test_llm_insights.py            §H hallucination-rejection + mocked call_llm
  test_modeling_extensions.py     §C.1/§H.1 hurdle model + SHAP unit tests
  test_adaptive_conformal.py      §D.2 ACI, incl. a distribution-shift scenario
  test_reconciliation_probabilistic.py  §E.1 two-tier probabilistic reconciliation
docs/
  technical_documentation.md
  architecture.md
data/                      provided CSVs
pickle/model.pkl           trained model bundle (produced by train.py)
output/                    run.sh artifacts (produced by predict.py)
run.sh
requirements.txt
```
