# Architecture

## Data flow

```
data/*.csv (any vendor schema)
      │
      ▼
validate.py  (run.sh stage 0)             standalone Pandera/ingestion report,
      │                                    also runnable on its own —
      │                                    data_quality_report.txt
      ▼
schema_mapper.ingest_directory()          §A  — alias mapping, fuzzy fallback,
      │                                        degrade-not-crash, Pandera validation
      ▼
canonical_df  (channel, campaign_id, campaign_name, date, spend, revenue,
               conversions, clicks, impressions, reach, video_views,
               campaign_type, daily_budget)
      │
      ├─────────────────────────────────────────────────────────────┐
      ▼                                                             ▼
feature_engineering.build_training_frame()      feature_engineering.build_latest_snapshot()
   (offline, train.py only)                        (online, generate_features.py, stage 1)
      │  one row per (campaign_id, origin_date,       │  one row per campaign_id,
      │  horizon) with target_revenue                 │  at the current forecast origin
      ▼                                                ▼
modeling.py                                       predict.py (run.sh stage 2)
  - Tweedie point model (§C.1)                        │  loads pickle/model.pkl
  - quantile ensemble (§C.2)                           ▼
  - walk-forward + grouped CV (§C.6)              quantile + point predictions
  - CQR calibration (§D.1)                             │
  - honest final-holdout eval (§G.2.4)                 ├──► sanity_clamps (§G.3) → display value + caveat
      │                                                ├──► budget_curves (§F) → saturation status
      ▼                                                ├──► reconciliation (§E) → coherent hierarchy
pickle/model.pkl (bundle: models, hurdle                └──► llm_insights (§H) → grounded summary
  ablation, SHAP importance, cqr                             │
  corrections, hill curves, roas                             ▼
  bounds, oof predictions, cv reports,             output/
  ACI comparison)
                                                        data_quality_report.txt
                                                        predictions.csv
                                                        reconciled_hierarchy.csv
                                                        data_health.json
                                                        reliability.json
                                                        hill_curves.json
                                                        mpc_reallocation_backtest.json
                                                        causal_summary.json
                                                            │
                                                            ▼
                                                   app.py (Streamlit, §I)
```

## Module responsibilities

| Module | Responsibility |
|---|---|
| `schema_mapper.py` | §A. Raw CSV → canonical schema. Alias table, fuzzy fallback, revenue/conversions degrade path, campaign_type regex inference, Pandera validation, `reach`/`video_views` optional-field mapping (§A.9). The only place raw vendor column names are ever referenced. |
| `feature_engineering.py` | §B. Per-campaign daily reindexing (gap-safe, including the whole-file-missing-field fix for `reach`/`video_views`), lag/rolling/calendar/expanding features, funnel-efficiency and extended funnel ratios (§B.6/§B.7 — CTR/CVR, CPM/CPA, reach/video signals), origin-based aggregate-window training-frame construction, live snapshot construction. |
| `modeling.py` | §C/§D/§G/§H.1. Tweedie point model, quantile ensemble, monotonic-constraint handling (+ its LightGBM limitation and isotonic workaround), quantile-crossing fix, walk-forward/grouped CV, CQR calibration, all evaluation metrics, the point-model ablation across three candidates — LightGBM Tweedie, LightGBM hurdle (classifier x Gamma), CatBoost Tweedie (§C.1c) — plus an equal-weight blend (§5b), real SHAP feature importance. |
| `adaptive_conformal.py` | §D.2/§D.2b/§D.2c (optional). Adaptive Conformal Inference (Gibbs & Candès 2021) — online-updating miscoverage target — and Conformal PID control (Angelopoulos, Candès & Tibshirani 2023) with two honestly-compared D-term variants: a derivative-of-error-signal proxy, and a genuine learned scorecaster (a ridge-AR model refit online at every step, §D.2c). All reported alongside static CQR for comparison, not the shipped default calibration. |
| `reconciliation.py` | §E. Hierarchy definition (total → channel → campaign_type → campaign), `hierarchicalforecast`-based MinTrace/BottomUp point reconciliation, two-tier probabilistic reconciliation: genuine `Conformal` reconciler at total/channel (full calibration-date alignment), marginal per-node split-conformal at campaign_type/campaign (alignment doesn't hold there), rescale fallback only for the sparsest campaigns; reconciled-interval calibration audit across every hierarchy level (§8c). |
| `budget_curves.py` | §F. Hill saturation curve fitting (`scipy.optimize.curve_fit`) per (channel, campaign_type), recency-weighted on both the curve-fit AND flat-fallback paths, with an R²/boundary-hit fit-quality gate; a DP cross-channel budget allocator (§F.3/§9a) with marginal ROAS and an optional ROAS floor; an MPC-style rolling-horizon reallocation backtest (§F.4/§9b) against a frozen open-loop baseline, with planned-vs-realized spend execution noise modeled directly; saturation status and divergence sanity check against the pooled model. |
| `sanity_clamps.py` | §G.3. Per-(channel, campaign_type) historical ROAS envelopes; flags and (display-only) clips implausible forecasts without hiding the raw number. Applied to both the static `predict.py` output and the live Streamlit what-if slider. |
| `llm_insights.py` | §H. Deterministic stats engine (anomalies, period-over-period), grounding-context builder (fed by real SHAP importance when available), Anthropic API call with structured tool-use output, programmatic numeric validator (with an automated hallucination-rejection test), deterministic rule-based fallback narrator. |
| `train.py` | Offline orchestrator: ingest → features → chronological train/calibration/holdout split → CV → 3-way point-model ablation + blend → fit production models (conditionally refitting whichever candidates the winner needs) → calibrate → evaluate once → SHAP → ACI + both Conformal PID variants → OOF for reconciliation → reconciled-calibration audit → fit Hill curves/ROAS bounds → serialize bundle. |
| `generate_features.py` | `run.sh` stage 1: ingest + build the live multi-horizon snapshot, write `features.parquet`. |
| `predict.py` | `run.sh` stage 2: score (whichever point-model candidate or blend won its ablation), apply CQR + clamps + Hill sanity check, reconcile the hierarchy, backtest MPC-vs-open-loop budget reallocation (§9b), write all `output/` artifacts (predictions, reconciliation, data-health, reliability, hill curves, MPC backtest, causal summaries). |
| `validate.py` | `run.sh` stage 0, also runnable standalone: run the schema layer against a data directory and print the ingestion/Pandera-validation report (`data_quality_report.txt`). |
| `app.py` | §I. Streamlit frontend, six views in ingest→forecast→simulate→insights order: data health, forecast, budget what-if (now plausibility-clamped + monotonic at the exact slider value, not just the chart; includes the MPC backtest expander, §9b), breakdown, AI summary, model reliability (CV, ACI + both Conformal PID variants, 3-way point-model ablation + blend, SHAP + gain importance). |

## Why campaign-level, not channel×type-level

The original approach modeled at (channel × campaign_type) grain and
distributed down to campaigns by a static historical share — a campaign that
changes trajectory relative to its peers would never be reflected. This
version's bottom modeling unit *is* the campaign (`build_training_frame`
keys off `campaign_id`), with `channel`/`campaign_type` as pooling
categoricals rather than the modeling grain itself. The hierarchy in §E then
aggregates campaign-level forecasts *up* to campaign_type/channel/total,
rather than distributing a channel-level forecast *down* — a strictly more
information-preserving direction, and the one that makes genuine
hierarchical reconciliation (rather than a fixed share table) meaningful.
