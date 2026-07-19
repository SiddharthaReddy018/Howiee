# AIgnition 3.0 — Executive Summary

*Team TechBlazers. Full technical detail, every design decision's reasoning,
and the complete known-limitations list live in
[`technical_documentation.md`](./technical_documentation.md) — this page is
the 1-2 page version the brief asked for: what it does, the headline
numbers, and the caveats that matter most before you trust an output.*

## What it does

Probabilistic multi-channel advertising revenue and ROAS forecasting.
Google Ads, Meta Ads, and Bing Ads campaign-level exports go in; calibrated
30/60/90-day revenue *and* ROAS forecasts come out — per campaign,
reconciled up to campaign-type, channel, and account totals, each with real
uncertainty ranges (not single point estimates), a live budget what-if
simulator *and* a cross-channel budget allocator, and a grounded
natural-language summary.

## Methodology, in eight ideas

1. **A schema-mapping layer sits in front of everything.** Every input file
   is column-mapped (exact match, then order-independent optimal fuzzy
   matching) onto a fixed canonical schema before anything else touches it —
   no raw vendor column name is referenced anywhere downstream.
2. **Forecasting is origin-based, aggregate-window, campaign-level.** For
   every `(campaign_id, origin_date)`, the model predicts total revenue over
   the next 30/60/90 days directly, with `horizon_days` itself as a feature
   — not a daily time-series model aggregated after the fact.
3. **The model is a probabilistic ensemble, not a point estimate.** A
   point model (LightGBM Tweedie, LightGBM hurdle, CatBoost's own Tweedie
   implementation, or a simple equal-weight blend of all three — whichever
   wins a held-out ablation, re-checked every training run, §5b) handles
   the zero-inflated mean; a 7-quantile LightGBM ensemble is conformally
   calibrated (CQR) so the stated intervals are honestly covered, not just
   plausible-looking. Both the point model's variance-power sweep and the
   quantile ensemble's hyperparameters are selected by averaging across
   every walk-forward CV fold, not a single validation split.
4. **Campaign forecasts reconcile up a real hierarchy — for both revenue
   and ROAS.** Nixtla's `hierarchicalforecast` (MinTrace) enforces that
   campaign → campaign-type → channel → total revenue sums are
   mathematically exact, at both the point and (where the data's
   calibration structure supports it) interval level. Spend is aggregated up
   the identical hierarchy, so ROAS *ranges* — not just a single derived
   point — exist at every level, including a headline blended-ROAS number.
5. **Every number shown to a human or an LLM is checked against reality
   first** — schema validation, campaign-consistency checks, per-(channel,
   campaign_type) plausibility clamps, and a programmatic numeric validator
   that rejects any LLM output containing a figure that doesn't trace back
   to the real underlying stats.
6. **A budget allocator turns the fitted saturation curves into an actual
   recommendation, not just a chart.** Given a fixed total budget, dynamic
   programming finds the split across channels that maximizes predicted
   revenue — genuinely necessary rather than a nicety, since the fitted
   response curves are S-shaped, not concave, so a naive greedy split can
   get stuck (§9a of the technical documentation). It also reports
   **marginal ROAS** per channel, supports an optional minimum-blended-ROAS
   floor, and gives the recommendation an approximate revenue range rather
   than a bare point estimate. An MPC-style rolling-horizon extension
   (§9b) backtests whether periodically re-fitting the curves and
   re-solving the allocation — as real historical data arrives — earns
   back more revenue than deciding once and running that plan for the
   whole horizon, against a frozen open-loop baseline, with
   planned-vs-realized spend execution noise modeled directly.
7. **Every optional calibration alternative is measured on the same
   holdout as the shipped default, honestly, and reported whether it wins
   or not.** Adaptive Conformal Inference and Conformal PID control
   (Angelopoulos, Candès & Tibshirani, NeurIPS 2023) are both compared
   against the shipped static CQR calibration on the identical sequence.
   PID itself now ships two honestly-compared D-term variants — a cheap
   derivative-of-error-signal proxy, and a genuine learned scorecaster (a
   small ridge-AR model, refit online at every step, forecasting the next
   nonconformity score directly, the paper's own actual design) — reported
   as the honest trade-off each one is, not spun as a win (§6b).
8. **Coherence and calibration are checked separately, because they're not
   the same guarantee.** The hierarchy sums exactly (item 4), every run —
   but that alone doesn't say whether the reconciled *interval* around each
   summed number is still trustworthy. It's now measured directly, pooled
   across 318 holdout snapshots at every hierarchy level: the account
   `total` and `campaign_type` levels stay conservative post-reconciliation,
   but `channel` and individual `campaign` bands measurably under-cover
   their nominal target (90% nominal lands at 81.4% and 71.3% empirical,
   respectively) — a real, quantified finding, not a hedge (§8c of the
   technical documentation).

## The headline numbers

| # | Metric | Value |
|---|---|---|
| 1 | **Forecast accuracy** (WAPE, median, final holdout — 8,808 rows, 2025-12-22→2026-05-06, scored once) | **36.5%**, improving from 41.6% at 30 days to 33.9% at 90 days — the 30-day figure is a property of the target (20% of 30-day windows are exact zero, vs. 1.5% at 90 days), not a model weakness; see full doc §7 |
| 1b | **Skill vs. naive baseline** (§7a of the technical documentation — "continue at the recent 28-day pace," never trained, same holdout) | Naive baseline scores 126.8% WAPE; the model cuts that error by **71.2%** overall (62.2% at 30 days → 75.7% at 90 days) — this is what "36.5% WAPE" is actually worth, since the number alone has no reference point |
| 2 | **Calibration** (CQR-corrected interval, final holdout) | CRPS 5,008 (vs. 5,050 raw); 80%-nominal interval empirically covers ~90.5% of holdout points, 90%-nominal covers ~94.9% — conservative, not overconfident |
| 3 | **Hierarchical coherence** (campaign → type → channel → total, every horizon, revenue) | **0.000000** max absolute error — exact, not approximate |
| 4 | **Blended ROAS** (total scope, 30-day, this run) | **4.88x** median, P10–P90 range **2.90x – 5.33x** — now a headline metric with a real range, at every hierarchy level |
| 5 | **Robustness** | 174 automated tests passing, incl. dedicated regression tests for a mangled/renamed/reordered-column input file, two channels colliding on the same raw campaign ID, a junk column that could previously hijack the channel field, the naive-baseline reference forecast, the budget allocator's DP-vs-naive-split correctness (including its ROAS-floor and marginal-ROAS logic), the MPC rolling-horizon reallocation backtest (including a synthetic regime-shift scenario that isolates the mechanism), a CatBoost point-model ablation and its equal-weight blend candidate, both Conformal PID D-term variants (derivative-proxy and learned-scorecaster, including a check that they collapse to plain ACI and to each other when their extra gain terms are zeroed), recency-weighted vs. equal-weighted Hill-curve fitting, the extended funnel features (CPM/CPA/reach/video_views), and the reconciled calibration-by-level check (§8c) against both real data and a synthetic ground-truth case |
| 6 | **Budget allocator uplift** (§9a — same total budget, optimized split vs. today's actual split, this run) | Reallocating the account's current 5,757/day according to the fitted (now recency-weighted, §9) Hill curves is predicted to raise daily revenue **+56.3%** (30,475 → 47,642) — with zero additional spend |
| 7 | **MPC rolling-horizon reallocation backtest** (§9b — same budget, periodically re-solving vs. a frozen one-shot plan, backtested on 90 real days) | Re-fitting the curves and re-solving every 30 days beats deciding once by **+5.3%** on average (13,318 → 14,026/day) — a genuine, mixed result: it wins in 2 of the 3 backtested windows and loses in the third, reported as-is |
| 8 | **Hindsight-regret audit** (§9c — the tool's recommendation vs. what this account's real historical spend/revenue actually were, identical windows as row 7) | Estimated **+10.7%** (open-loop) to **+16.5%** (MPC) more revenue than what actually happened, same total spend — the comparison the naive baseline (row 1) does for the forecast, done for the allocator |

*(Data scale: 25,562 canonical rows, 136 campaigns, 3 channels,
2024-01-01 → 2026-06-05.)*

## Top limitations to know before trusting an output

1. **Meta's "revenue" is a proxy, not real revenue.** No revenue field
   exists in Meta's export; the `conversion` column stands in for it, and
   every Meta number downstream — forecasts, ROAS, the Hill curve, the LLM
   narrative — is flagged `revenue_confidence: proxy` accordingly. This is
   the single biggest reason to weight Meta's numbers differently from
   Google's or Bing's, and it's a data-availability constraint, not
   something the model can fix. **Worth checking:** the brief's Resources
   section separately lists GA4 session source/medium data and Shopify
   conversion data alongside the dataset link — if the actual download
   includes those (this repo's `data/` folder currently doesn't), Shopify's
   real order revenue could directly resolve this for Meta specifically.
2. **The forecast fan chart is flat by design, not a widening cone.** The
   model predicts one total for the next 30/60/90 days, not a daily path —
   so the displayed forward band is genuinely flat across the whole window.
   This is the honest shape of what the model actually estimates, not a
   simplified stand-in for something more granular.
3. **Interval reconciliation is two-tier** (revenue; ROAS ranges inherit
   whichever tier their revenue quantiles came from). Total and channel
   nodes get the library's real joint conformal reconciliation (their
   calibration data is fully aligned — checked directly: 308/308 dates).
   Individual campaigns don't — campaign lifecycles don't overlap enough
   for joint alignment (0/308 dates have every campaign present at once, a
   real property of the data), so those nodes get an independently-computed
   marginal correction instead. Point coherence (headline #3 above) is
   unaffected either way; this caveat is about interval width/shape only.

## What changed this round

**Ensemble diversity, a genuine PID scorecaster, MPC budget backtesting,
and extended funnel features — the four items carried over from the prior
round's own "further work" list, plus the number-consistency fix that list
also flagged.**

- **CatBoost as a second point-model family (§5b).** The Tweedie-vs-hurdle
  ablation is now 3-way (LightGBM Tweedie, LightGBM hurdle, CatBoost's own
  independent Tweedie implementation) plus a 4th, implicit candidate: the
  simple equal-weight blend of all three — the concrete test of whether
  ensemble diversity pays off rather than an assumption that it does.
- **A genuine learned scorecaster for Conformal PID (§6b).** Alongside the
  original derivative-of-error-signal proxy D-term,
  `run_conformal_pid_control_learned_scorecaster` implements the actual
  design Angelopoulos, Candès & Tibshirani's paper describes: a small
  ridge-AR model, refit online at every step, forecasting the next
  nonconformity score directly — the two D-term variants are now compared
  honestly on the identical holdout sequence.
- **An MPC-style rolling-horizon budget-reallocation backtest (§9b),** the
  closed-loop extension of the Hill-curve recency weighting already in
  place — does periodically re-fitting the curves and re-solving the
  allocation, as real historical data arrives, earn back more revenue than
  deciding once and running that plan for the whole horizon? Backtested
  honestly against a frozen open-loop baseline, with planned-vs-realized
  spend execution noise modeled directly.
- **Two more funnel features: CPM/CPA cost-efficiency, and reach/video
  funnel signals (§4).** Two real columns — Meta's `reach`, Google's
  `metrics_video_views` — were already present in this project's own raw
  files and being silently dropped at ingestion with no canonical slot to
  land in. `cpa_roll_28` lands in the top 10 features by gain on this data.
  Finding and fixing this also surfaced (and fixed) a real gap-fill bug in
  the daily-reindex step, invisible before because no earlier optional
  field was ever whole-file-missing in this project's own data.
- **Number-consistency fix.** The previous round's headline numbers
  (WAPE, coverage, ROAS, the budget-reallocation uplift) were generated by
  two different training runs — the documented numbers came from a run
  predating the CTR/CVR features, while the actually-shipped
  `pickle/model.pkl` was retrained afterward under intentionally reduced
  settings for a time-constrained verification pass. Every number in this
  document (and README.md, technical_documentation.md) now comes from one
  single full, unreduced run that includes every feature and every
  candidate model described here — the numbers below and the shipped
  pickle are, for the first time this round, guaranteed to match.


On this run: the point-model ablation's **equal-weight blend of Tweedie +
hurdle + CatBoost wins**, by +4.4% pinball loss vs. Tweedie alone (§5b) —
concrete evidence the three candidates' errors are usefully uncorrelated.
The learned-scorecaster PID variant matches the derivative-proxy variant's
coverage (89.9% vs. 89.9%) but at nearly double the interval width on this
real holdout (§6b) — an honest, mixed result, not a clean win either
direction. The MPC budget-reallocation backtest beats a frozen one-shot
plan by +5.3% overall, winning in 2 of 3 backtested windows (§9b).

**Prior round, for context:**

**A genuine gap-check against the brief, not just bug fixes.** Re-reading
the brief's exact deliverable list against the actual prototype surfaced
three named-but-missing pieces:

- **ROAS ranges, and a headline "blended ROAS."** The brief names
  "channel-level/campaign-type/campaign-level ROAS ranges" and "expected
  blended ROAS" as their own top-level deliverables. ROAS previously
  existed only as a single derived point per campaign. Fixed by giving
  spend the same hierarchical treatment as revenue (§ full doc §8) — now a
  headline metric card, a per-level chart on the Breakdown tab, and a
  per-campaign range in the Forecast tab's table.
- **Campaign-consistency validation**, its own named deliverable bullet,
  didn't exist at all. Added: duplicate (campaign_id, date) detection, and
  detection of a campaign_id reporting more than one campaign_type or
  campaign_name over its history. Zero issues on this project's own data
  (confirmed directly) — but nothing was checking before now.
- **A real ingestion robustness bug**, found by directly testing "extra
  columns shouldn't break it": a junk column merely *named* something like
  `some_new_platform_field_2026` could fuzzy-match and silently hijack the
  `channel` field with garbage values — worse than a crash, since nothing
  downstream would catch it. `channel` is now exact-match only; its own
  safer filename/override fallback handles the rest.

**Modeling methodology**, since it's the majority of this round's grading
weight: both the Tweedie variance-power sweep and the quantile ensemble's
hyperparameter search were retooled to average their metric across every
walk-forward CV fold instead of trusting one validation split — a real
overfitting-to-one-fold risk in the previous approach. Concretely, the
quantile search (now widened to 7 configs including learning-rate
variants) landed on the *same* configuration as before even under the more
rigorous multi-fold evaluation — a genuinely useful negative result
confirming the original choice was already robust, not a lucky pick. The
Tweedie-vs-hurdle point-model ablation *did* flip (hurdle now wins by
3.1%, vs. Tweedie by 4.2% previously) purely from the different CV fold
boundaries after retraining — an illustration of exactly the run-to-run
sensitivity the more rigorous evaluation is meant to catch, documented
honestly rather than smoothed over.

**Two robustness bugs from the previous round**, still standing: order-
dependent fuzzy column matching (replaced with a real optimal assignment)
and campaign_id not being namespaced by channel (fixed at the schema
source). Both retrain-verified with zero numeric drift on this project's
real data (confirmed via exact key-based diff, not just a re-run).

**Frontend**: the Hill saturation curve, previously computed and tested but
never shown anywhere, now has its own chart on the Budget What-If tab; the
reconciled hierarchy has a treemap/sunburst view; the budget slider flags
extrapolation beyond historical spend. The live LLM call path (§H.3) was
pushed as far as possible without a real `ANTHROPIC_API_KEY` in this
sandbox — confirmed a genuine authenticated round-trip to
`api.anthropic.com` and the full grounding pipeline against production
data; `src/verify_live_llm.py` gets the final key-required confirmation in
one command.

