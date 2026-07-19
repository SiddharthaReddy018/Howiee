# Technical Documentation — AIgnition 3.0, Team TechBlazers

*Looking for the short version? See [`summary.md`](./summary.md) for a 1-2
page executive summary (methodology, headline numbers, top limitations).
This document is the full appendix: every design decision's reasoning, real
numbers from this repo's own data, and the complete known-limitations list.*

Probabilistic multi-channel advertising revenue forecasting: Google Ads, Meta
Ads, and Bing Ads campaign-level data in, calibrated 30/60/90-day revenue
forecasts (with uncertainty intervals, hierarchical coherence, a budget
what-if layer, and a grounded natural-language summary) out.

This document describes what the system actually does, the reasoning behind
each design choice, the numbers from real runs against the provided data, and
— explicitly — what was tried, what didn't work, and what's still a known
limitation. Nothing below is aspirational; every number quoted was produced
by running `src/train.py` and `run.sh` against `./data` in this repository.

---

## 1. What changed, and why (executive summary)

The original pipeline modeled revenue at **(channel × campaign_type × day)**
grain with daily lag/rolling features, distributed forecasts down to
campaigns by a static historical revenue share, and used a single LightGBM
regressor with no uncertainty quantification. That approach could not:
produce a genuine per-campaign forecast, express uncertainty, survive a
renamed/reordered/extra-column input file, or defend its numbers against an
implausible edge case (e.g. a channel/type combination with literally zero
historical revenue).

This version rebuilds the pipeline around five ideas:

1. **A schema-mapping layer sits in front of everything** (`src/schema_mapper.py`).
   No raw vendor column name is ever referenced anywhere else in the codebase.
2. **Forecasting is reframed as origin-based, aggregate-window, campaign-level**
   regression (`src/feature_engineering.py`): for every `(campaign_id,
   origin_date)`, predict `sum(revenue)` over the next 30/60/90 days, with
   `horizon_days` itself as a shared model feature.
3. **The model is a probabilistic ensemble, not a point estimate**
   (`src/modeling.py`): a Tweedie-objective LightGBM for the zero-inflated
   mean, plus a 7-quantile LightGBM ensemble, conformally calibrated
   (CQR) so the stated intervals are honestly covered.
4. **Campaign-level forecasts are reconciled up a real hierarchy**
   (`src/reconciliation.py`) using Nixtla's `hierarchicalforecast`, so
   channel totals mathematically sum to the grand total instead of hoping
   they do.
5. **Every number that reaches a human or an LLM is checked against reality
   first** — Pandera schema validation, per-(channel, campaign_type)
   plausibility clamps (`src/sanity_clamps.py`), and a programmatic
   numeric validator on the LLM's own output (`src/llm_insights.py`).

---

## 2. Data reality (grounded facts used throughout this document)

Computed directly from the three provided files (`python3 src/validate.py --data-dir ./data`):

| Channel | Rows | Campaigns | Date range | Revenue field |
|---|---:|---:|---|---|
| Bing  | 2,873  | 28 | 2024-05-25 → 2026-06-05 | direct (`Revenue`) |
| Google | 19,272 | 92 | 2024-01-01 → 2026-06-04 | direct (`metrics_conversions_value`) |
| Meta  | 3,417  | 16 | 2024-05-23 → 2026-06-05 | **proxy** — no revenue field exists; the `conversion` column is used as a stand-in and flagged `revenue_confidence: proxy` everywhere downstream |

Other facts that shaped specific design decisions below:
- Several `(channel, campaign_type)` groups have **zero revenue in every
  single observed day** (e.g. `bing/Audience`: 66 days, 0 revenue, 0 ROAS).
  A model that pools across campaign types can still predict a small
  positive number for such a group by borrowing strength from others — the
  plausibility clamp in §10 exists specifically to catch this.
- `bing/Shopping` shows historical ROAS around 17–18x — an order of
  magnitude above most other groups. A single global ROAS sanity bound would
  either falsely flag this group or be too loose to catch a real anomaly
  elsewhere; bounds are computed **per (channel, campaign_type)**.
- Google's file carries an extra `metrics_video_views` column with no
  canonical counterpart — it is correctly logged as *ignored*, not
  silently coerced into a numeric feature slot it doesn't belong in.
- Meta's `campaign_type` column doesn't exist at all — the type is
  regex-inferred from `campaign_name`, and 17 of 3,417 rows genuinely
  cannot be classified (`unclassified`), which is reported, not hidden.

---

## 3. Schema robustness layer (`src/schema_mapper.py`)

**Design.** Every raw CSV is passed through `ingest_file` / `ingest_directory`
before anything else touches it. Column mapping runs in two passes:

1. **Exact match** on a normalized (lowercased, alphanumeric-only) form of a
   configurable alias table (`CANONICAL_ALIASES`) — this is what already
   handles Bing's `CampaignId` / Google's `campaign_id` / a hypothetical
   `Campaign Id` all mapping to the same canonical `campaign_id`.
2. **Fuzzy fallback** (`rapidfuzz`, a blended `ratio`/`partial_ratio` score)
   for anything left over, with **two different acceptance thresholds**:
   required-tier fields (`date`, `campaign_id`, `spend`, `revenue`) need
   score ≥ 80; optional-tier fields need only ≥ 65. Guessing wrong on spend
   or revenue is worse than refusing to guess, so that asymmetry is
   deliberate, not an oversight.

**Degrade-not-crash rules** (§A.4 of the internal plan):
- `revenue` missing but `conversions` present → forecast on conversions,
  tag `revenue_confidence="proxy"` (this is exactly Meta's real situation
  above, not a hypothetical).
- `campaign_type` missing → regex-infer from `campaign_name`
  (`TYPE_PATTERNS` in `schema_mapper.py`); if that also fails, label
  `"unclassified"` rather than crash or fabricate a type.
- `daily_budget` missing → left as genuine `NaN` forever, never zero-filled
  or imputed — it's a *setting*, and LightGBM's native NaN handling is
  trusted to use "budget unknown" as real information (§5 below).
- A single unusable file (missing `date`/`campaign_id`/`spend`, or missing
  both `revenue` and `conversions`) raises `SchemaMappingError`, which
  `ingest_directory` catches per-file — one bad file doesn't take down
  ingestion of the other two.

**Evidence, not a claim.** `tests/test_schema_robustness.py` builds a
synthetic "mangled" version of `bing_campaign_stats.csv`: `Spend` →
`Cost_Local`, `CampaignName` → `Camp_Nm`, `DailyBudget` and `CampaignType`
dropped entirely, three junk columns added, columns shuffled. All 9 tests
pass, including a full ingest → feature → train pass on the mangled file.
Beyond the unit tests, **the actual `run.sh` pipeline was run end-to-end**
against a data directory containing this mangled file substituted for the
real Bing CSV — it produced the identical 408-row prediction output with no
special-casing, log excerpt:

```
mapped (fuzzy) : {'spend': ('Cost_Local', 80.8), 'campaign_name': ('Camp_Nm', 88.3)}
ignored        : ['Unnamed: 0', 'foo_bar', 'reserved_field_1', 'internal_notes_2026']
missing/optional (left NaN, not fabricated): ['daily_budget', 'campaign_type']
campaign_type  : inferred  (unclassified=0)
```

**Known limitation.** Fuzzy thresholds (80 / 65) were tuned against the
specific rename patterns used in the test fixture and against the real
column names in the three provided files — they are a reasonable starting
point, not a formally optimized decision boundary. A production deployment
ingesting many more vendor formats would want a small labeled set of
real-world header variants to tune against.

**A second, previously-unhandled failure mode: correct column, wrong value
formatting.** The mangled-schema fixture above only renames/drops/shuffles
*columns* — it never reformats their *values*. Stress-testing that
separately surfaced a real bug: a `spend`/`revenue` column that maps
correctly by name (exact or fuzzy) but is currency-formatted
(`"$1,234.56"`, thousands separators, accounting-style `(123.45)`
negatives, or a trailing `%` — all realistic for an Excel-round-tripped or
differently-localized export) was silently coerced to `NaN` by
`pd.to_numeric(errors="coerce")` and then to a *valid-looking* `0.0` by the
very next `.fillna(0.0)`. Nothing caught this: the ingestion report showed
a clean full mapping, Pandera's `Check.ge(0)` passed trivially since `0.0`
is a legal value, and `data_quality_report.txt` reported PASSED — while
every forecast for that file was silently computed off zero spend and zero
revenue. Reproduced end-to-end through `run.sh` against this repo's own
`pickle/model.pkl`: campaign `570837630`'s `planned_future_daily_budget`
dropped from 4.21 to 0.0 and its `plausibility_flag` flipped from `True`
(correctly caught by §10's clamp) to `False` (looks "fine") on the
corrupted input — i.e. the safety net got *less* suspicious of exactly the
run it should have been more suspicious of. This is strictly worse than a
missing column, which is at least logged in `missing_optional`.

Fixed: `_clean_numeric_series` / `_to_numeric_checked` (`schema_mapper.py`)
now strip currency symbols, thousands separators, whitespace, and a
trailing `%`, and treat `(123.45)` as `-123.45`, before `pd.to_numeric` —
applied to all six numeric fields (`spend`, `revenue`, `conversions`,
`clicks`, `impressions`, `daily_budget`). Verified to reproduce the exact
same canonical totals as the original clean file
(`tests/test_numeric_format_robustness.py`). A second layer catches formats
this cleaner doesn't recognize: if a meaningful share (>2%) of genuinely
non-empty raw values still fail to parse after cleaning, a warning naming
the field and a concrete failing example is added to the ingestion report
— so an exotic, still-unhandled format degrades to a visible warning
instead of a silent, undetectable zero.

**Two more silent-failure modes, found by adversarial testing beyond the
fixtures above, both now fixed and covered by
`tests/test_schema_edge_cases.py`:**

1. **Channel-name pollution on an unrecognized filename.** If a source
   file's name doesn't match any pattern in `_CHANNEL_FILE_PATTERNS`
   (e.g. a grading dataset ships `export_2026_q3.csv` instead of
   `bing_campaign_stats.csv`), the channel column previously fell back to
   the *literal filename, extension included* (`ingest_dataframe`'s
   `df["channel"] = source` branch used the raw `source` argument
   directly). That value then propagated into every downstream
   channel-level breakdown, the app's channel selector, and the LLM
   grounding context's scope label. Fixed by stripping the extension at
   the point of assignment (`os.path.splitext(source)[0].lower()`) —
   verified to produce a byte-identical `data_quality_report.txt` against
   this repo's own three correctly-named files (no behavior change on the
   happy path), and a clean, extension-free channel on an unrecognized one.
2. **A mapped-but-fully-blank numeric column produced zero warnings.**
   `_to_numeric_checked`'s existing >2%-unparseable guard divides by
   `n_nonempty`; when a column is present but every single value is
   blank/NaN, `n_nonempty == 0` and the guard can never fire — the column
   silently became a valid-looking all-zero (or all-NaN, for an optional
   field) with nothing in the ingestion report, worse than a genuinely
   missing column. Fixed with an explicit `n_nonempty == 0 and n_total > 0`
   branch that names the field and states plainly that nothing parseable
   was found.

---

## 4. Feature engineering (`src/feature_engineering.py`)

**Reframing.** This is an *origin-based aggregate-window* problem, not a
daily time-series problem: for each `(campaign_id, origin_date)`, features
= everything knowable using data up to and including `origin_date`; target
= `sum(revenue)` over the next `horizon_days` (30/60/90), with
`horizon_days` passed in as a feature so one model is shared across all
three horizons (given how few campaigns exist per channel — 16 to 92 — three
fully independent models would each see a third of the already-small
per-campaign history).

**Gap-safety.** Every campaign is reindexed to a complete daily calendar
*before* any lag/rolling feature is computed (`_reindex_campaign`), or a lag
would silently splice together two days that weren't actually adjacent.
Non-observed (gap) days get `spend`/`revenue`/`clicks`/`impressions`/
`conversions`/`reach`/`video_views` zero-filled — no serving = no spend, a
defensible default — tracked by a `was_observed` flag. Critically, this is
**not** the same code path that handles "the whole file never had this
column" (§3 above): that produces genuine, permanent `NaN`, and the reindex
step is careful to zero-fill only the newly-added gap rows, never touching
pre-existing `NaN` values on rows that really were present in the source
file. A real bug in exactly this distinction was found and fixed while
adding `reach`/`video_views` this round: the zero-fill loop previously had
no way to tell "this campaign's file genuinely tracks this field, today's
gap day should be 0" apart from "this channel's file never reported this
field at all, so even gap days must stay `NaN`" — invisible before because
every pre-existing optional field (clicks/impressions/conversions) happened
to be reported by every channel in this project's own data, so the
distinction was never exercised. `reach` (bing/google never report it) and
`video_views` (bing/meta never report it) make it a live case: the fix
checks, per campaign and per field, whether that field was EVER reported on
any of the campaign's genuinely-observed rows before deciding whether gap
days for that field get zero-filled at all (see
`tests/test_funnel_features.py::test_gap_days_do_not_fabricate_reach_for_a_channel_that_never_reports_it`).

**Feature manifest** (`FEATURE_NAMES`, 55 features): calendar/seasonality
(day-of-week, week-of-year, a short justified holiday-window list — Black
Friday/Cyber Monday, Christmas week, New Year, Valentine's, back-to-school —
plus a year-over-year same-week revenue feature, correctly `NaN` for
campaigns too young to have one); recency/maturity (`campaign_age_days`,
`days_since_last_nonzero_revenue`); lag and rolling sum/mean/std at
7/14/28/56-day windows on revenue, spend, conversions, clicks, impressions;
trailing ROAS at three windows (left `NaN`, not zero-filled, when spend is
zero — an undefined ratio should not silently become "0x"); a trend feature
(28-day mean vs. the same 28-day mean shifted back 14 days); six extended
funnel features (`cpm_roll_28`, `cpa_roll_28`, `reach_roll_sum_28`,
`video_views_roll_sum_28`, `frequency_roll_28`, `video_view_rate_roll_28` —
see below); and `planned_future_daily_budget`, kept as an explicit, separate
input from historical spend (§5's monotonic-response feature).

**Adstock/carryover ablation (tried, measured, rejected).** Standard MMM
practice (Jin et al., *Bayesian Methods for Media Mix Modeling with
Carryover and Shape Effects*, Google 2017; the basis of Robyn/
LightweightMMM/PyMC-Marketing) models a day's spend as continuing to
influence revenue for days afterward via geometric decay, which the
fixed-window lag/rolling features above only coarsely proxy. Three
geometric-decay features were added (`spend_adstock_hl{3,7,14}`, half-lives
in days, computed as a strictly-causal first-order IIR recursion —
`adstock_t = spend_t + decay * adstock_{t-1}` — on the same gap-reindexed
daily spend series every other feature here uses) and the full pipeline
(including hyperparameter re-tuning, which landed on a genuinely different
config: `num_leaves` 63 vs. 31) was re-run end-to-end. Real result on the
same never-touched-during-tuning final holdout: WAPE 36.8% vs. 36.1% and
raw CRPS 5,091 vs. 4,911 — i.e. measurably *worse*, not better. With this
already-rich lag/rolling/trailing-ROAS feature set (13 spend/revenue window
features spanning 7–56 days), the 3 adstock features appear to add
correlated redundancy rather than new signal, at the cost of diluting
splits across more candidate features without a matching increase in tree
capacity. Reverted from the shipped model for the same reason the hurdle
model below only ships when it actually wins its ablation (§5b): this
project ships what the holdout number supports, not what's conceptually
fashionable. Kept here as a documented negative result rather than deleted,
since "we tried the standard MMM technique and it didn't help on this
data" is itself useful information for anyone extending this later.

**Campaign identity, without memorizing identity.** `campaign_id` is never
fed to the model as a raw high-cardinality categorical — with only 16–92
campaigns per channel, a tree model would happily memorize per-ID behavior
and fail completely on a campaign it hasn't seen, which is exactly the
"similar but not identical dataset" grading risk. Only `channel` and
`campaign_type` are native categoricals. A cheap substitute for identity
signal is an *expanding* (strictly-prior) per-campaign revenue mean and ROAS
— the time-series analogue of out-of-fold target encoding: because the
origin framing already restricts every feature to data at-or-before the
origin date, an expanding statistic is automatically leakage-safe without
needing a K-fold trick (K-fold encoding exists specifically to fix leakage
in i.i.d. data; origin-based framing already prevents it here). Feature
importance (§6) confirms this didn't backfire: `campaign_expanding_roas`
and `campaign_expanding_revenue_mean` rank 3rd/4th by gain, well behind
`planned_future_daily_budget` and `horizon_days` — supporting signal, not a
memorized lookup table.

**Funnel-efficiency ratios (`ctr_roll_28`, `cvr_roll_28`).** Clicks,
impressions, and conversions were already tracked as rolling 28-day sums —
raw volume — but the model previously had no way to see click-through or
conversion *rate* independent of that volume: two campaigns spending
identically with identical click volume can have very different revenue if
one converts at 2% and the other at 8%. Both ratios are computed from
rolling sums the feature table already had (no new raw data pulled in),
left NaN — not zero-filled — when the denominator is zero, the same
convention `trailing_roas` already used. The highest-risk change in this
round precisely because it touches the shared model's own feature set
(every downstream test that trains a real model was re-run against it, not
just the ones for this file directly).

**Extended funnel features (`cpm_roll_28`, `cpa_roll_28`,
`reach_roll_sum_28`, `video_views_roll_sum_28`, `frequency_roll_28`,
`video_view_rate_roll_28`) — added this round.** Two real, correctly-named
columns already sat in this project's own raw files with no canonical slot
at all (`schema_mapper.py` §A.9): Meta's `reach` and Google's
`metrics_video_views`. Both were silently dropped at ingestion
(`ignored_columns`) before this round — not a bug, nothing crashed, but
genuine signal that was pulled in and then thrown away. Four features build
on top of them:
- `cpm_roll_28` / `cpa_roll_28` — cost per thousand impressions / cost per
  acquisition, both from spend and impressions/conversions rolling sums
  already tracked. CTR/CVR say how efficiently traffic converts once
  bought; CPM/CPA say how expensive that traffic was to buy in the first
  place, and CPA specifically is the number agencies budget against most
  directly in practice.
- `frequency_roll_28` = impressions / reach — average number of times the
  same reached user was shown an ad. This is the concrete signal that
  distinguishes brand-awareness spend from performance spend in a way
  `campaign_type` alone can't: low-frequency, high-reach spend is buying
  broad first-time exposure; high-frequency, lower-reach spend is
  re-serving the same audience (retargeting-flavored) — historically very
  different revenue responses even within the same nominal campaign_type.
  Honesty note: Meta's raw export is a *daily* reach figure, not a
  deduplicated 28-day unique count, so summing 28 daily values
  double-counts any user reached on more than one day in the window —
  `frequency_roll_28` is therefore a slight over-estimate of true average
  frequency, not a formally deduplicated figure (§14).
- `video_view_rate_roll_28` = video_views / impressions — Google's own
  upper-funnel engagement signal, same 0-vs-nonzero funnel-stage idea as
  CTR/CVR, applied to video-capable campaigns.

`reach` (bing/google never report it) and `video_views` (bing/meta never
report it) are channel-specific by nature, so the other channels' rows are
genuinely `NaN` for these fields — same "whole file never had the column"
convention as every other optional field, not zero-filled, not imputed.
Feature importance (§5c): `ctr_roll_28` and `video_view_rate_roll_28` both
land in the top 10 features by mean |SHAP| on the production run (ranks 9
and 10 — see §5c's table), ahead of several of the original lag/rolling
features they were added alongside.

---

## 5. Core model (`src/modeling.py`)

**Tweedie point model.** `objective="tweedie"` LightGBM for the mean
forecast — the right family for zero-inflated, right-skewed revenue (Bing
alone: dozens of channel/type-day combinations with literally zero revenue
on the majority of days). `tweedie_variance_power` was swept over
`{1.1, 1.3, 1.5, 1.7, 1.9}`, evaluated by genuine Tweedie deviance
**averaged across every walk-forward CV fold** (not a single held-out
split — see the methodology note below); `p=1.9` won by a wide margin
(avg. deviance 3.52 vs. 2048.7 at `p=1.1`), consistent with revenue here
being closer to a compound-Poisson-Gamma process with a heavy zero mass
than to a mildly-overdispersed count.

**Quantile ensemble.** Seven LightGBM models at
`{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95}`. Hyperparameters (`num_leaves`,
`min_child_samples`, `lambda_l2`, `feature_fraction`, and now also
`learning_rate`) are tuned **once**, on the median (`q=0.5`) model via a
grid of 7 configurations, then reused for every other quantile (only
`alpha` changes) — tuning all seven independently on this little data
would overfit the tuning process itself.

**Methodology upgrade: multi-fold-averaged selection, not a single split.**
Both sweeps above (`tweedie_variance_power` and the quantile hyperparameter
grid) used to pick a winner by evaluating each candidate on ONE held-out
validation split. That's a real methodological risk: a config can win
because it happens to fit that one slice's idiosyncrasies, not because it
generalizes — the kind of thing that looks fine in development and quietly
underperforms on a grading dataset with different fold boundaries. Both
sweeps now average their metric (deviance / pinball loss) across every
walk-forward CV fold (`modeling.walk_forward_splits`, the same folds
`§6`'s development CV already uses) before picking a winner
(`modeling._tune_median_hparams`, `modeling.train_point_model`). Both
functions still accept a single split too (auto-wrapped into a 1-fold
list), so the cheap unit tests exercising them didn't need to change.

Concretely, on this data: re-running the now-multi-fold search landed on
the exact same quantile hyperparameters as the previous single-fold search
(`num_leaves=31, min_child_samples=40, lambda_l2=0.5, feature_fraction=0.85`)
even after the grid was widened to include `learning_rate` variants — a
genuinely useful negative result, not a non-result: it confirms the
original configuration was already robust rather than a lucky pick off one
split, which the earlier single-split methodology couldn't actually
demonstrate on its own. The final holdout WAPE/CRPS numbers in §7 are
therefore unchanged from the previous round — expected, since the same
config was selected — while the Tweedie-vs-hurdle point-model ablation
(§5b) *did* move, illustrating exactly the run-to-run sensitivity this
upgrade is meant to guard against.

**Monotonic constraint — and a real library limitation, handled honestly.**
The plan called for a `+1` monotone constraint on `planned_future_daily_budget`.
Testing this directly surfaced a genuine LightGBM restriction (confirmed on
both 4.3.0 and 4.6.0): `monotone_constraints` is **rejected outright** for
L1-family objectives —
```
LightGBMError: Cannot use ``monotone_constraints`` in quantile objective, please disable it.
```
`tweedie` and plain `regression` (L2) accept it fine; `quantile`,
`regression_l1`, and `mape` do not. This is not a configuration mistake —
it's documented LightGBM behavior for how those objectives compute split
gains. The adaptation actually shipped: the **Tweedie point model carries
the native constraint**; the **quantile ensemble cannot**, so instead
`modeling.enforce_monotonic_along_grid` applies **isotonic regression**
(`sklearn.isotonic.IsotonicRegression`) across a budget-scenario grid,
independently per quantile, then re-applies the quantile-crossing sort —
used by the frontend's budget what-if slider (§12) to guarantee "more
budget ⇒ not less forecast revenue" without relying on an unsupported
LightGBM code path.

**Native missing-value handling.** No manual imputation of optional numeric
features. LightGBM's default missing-value routing (send NaN down whichever
branch reduces loss more, i.e. treat "unknown" as informative) is trusted
directly — appropriate here because `daily_budget`-missing and
`conversions`-missing genuinely correlate with which channel a row came
from, which is itself informative.

**Quantile-crossing fix.** Per-row sort across the quantile axis
(`fix_quantile_crossing`, Chernozhukov, Fernández-Val & Galichon,
*Econometrica* 2010) — applied after every quantile prediction and again
after CQR correction and after reconciliation rescaling.

**Two CV protocols, deliberately compared.**
- *Walk-forward* (`walk_forward_splits`): expanding-window, leakage-safe —
  a fold's training rows are only those whose entire target window is
  already resolved before the cutoff (`origin_date + horizon_days <= cutoff`),
  not merely `origin_date <= cutoff`, which would let a 90-day-horizon
  training row peek past the cutoff.
- *Grouped-by-campaign* (`grouped_campaign_splits`, `GroupKFold` on
  `campaign_id`): tests generalization to a campaign the model has never
  seen at all.

Development-CV results (`pickle/model.pkl → cv_reports`):

| Protocol | CRPS | WAPE (median) | Empirical 80%-nominal coverage | Empirical 90%-nominal coverage |
|---|---:|---:|---:|---:|
| Walk-forward | 5,718 | 48.1% | 41.2% | 57.8% |
| Grouped-by-campaign | 4,347 | 28.8% | 49.5% | 68.3% |

**Grouped CV shows *lower* error than walk-forward — read this correctly.**
This is the opposite of the classic "identity overfitting" signature (which
would show grouped *worse*). The actual explanation is simpler: walk-forward's
earliest folds train on a much smaller expanding window, while grouped CV
always trains on ~80% of the *entire* dataset regardless of time position.
More training rows → lower error, independent of whether campaign identity
is being memorized. Combined with the feature-importance finding in §4 (the
expanding-identity features rank well behind the budget/horizon features),
this is evidence *against* identity-driven overfitting, not evidence of one.
The low walk-forward-CV coverage numbers above (41.2%/57.8% against 80%/90%
nominal) are the real finding worth acting on — they're what motivated §6.

**Anomalous-segment training weights — a generic replacement for a
hand-fitted `is_bing_2026` hack.** An earlier version of this pipeline had a
feature + training-sample weight keyed literally to the string `"bing"` and
the year `2026`, because Bing's tracking broke starting then (confirmed in
the raw CSV: near-zero revenue on most 2026 Bing rows). That's a real
finding but a dataset-specific fix — it would misfire (or fail to fire at
all) on a grading dataset with a different anomaly profile.
`feature_engineering.detect_anomalous_segments` replaces it: for every
`(channel, campaign_type)` group with at least 90 days of its *own* history,
it computes a rolling 30-day zero-revenue-rate and flags contiguous 14+ day
runs where that rate is a robust median/MAD outlier **relative to that same
group's own history** (each group's first 60 days of rolling values are
excluded from its own baseline, since a brand-new campaign type's near-100%
zero start is ordinary ramp-up, not an anomaly relative to itself). Training
rows whose target window overlaps a flagged segment are downweighted by an
automatically-scaled factor (more extreme z-score → lower weight, clipped to
`[0.1, 0.9]`) — applied to the training loss only, never to validation/
holdout scoring.

Run fairly across every group in this data, it flags **6 segments**
(`pickle/model.pkl → anomalous_segments`): two clearly extreme ones in
`meta/prospecting` (2025-08 to 2025-10) and `meta/remarketing` (2026-01 to
2026-06), both z≈9.4, downweighted to 0.12x; two moderate ones in
`meta/prospecting` and `meta/remarketing` at z≈2–3.4; and two moderate
`bing/Search` segments (one in 2024, one covering Feb–Apr 2026) at z≈2.25,
downweighted to 0.80x. **Stated plainly:** this generic, fairly-applied
check does *not* flag `bing/Search`'s 2026 period as strongly as the old
hand-tuned 0.1x hack implied it should — that channel/type is persistently
volatile across its *entire* life (including a similarly high zero-rate
early on, which is why the 60-day warm-up exclusion exists at all), not
uniquely broken in one obvious place, so a detector that isn't allowed to
peek at the known answer correctly declines to single that period out as
dramatically as a human who already knew the story would. That's the
intended trade-off: a mechanism that would catch an analogous break in a
*different* channel just as readily, at the cost of not reproducing the
exact old narrative about Bing.

### 5a. Model selection: why LightGBM/GBDT over DeepAR / TFT / N-BEATS (§J.2)

The plan template asks for an explicit argument, not just a choice. The
alternatives seriously considered were **DeepAR** (Salinas et al. 2020, an
autoregressive RNN probabilistic forecaster), the **Temporal Fusion
Transformer** (Lim et al. 2021, attention-based multi-horizon forecaster
with native quantile output), and **N-BEATS** (Oreshkin et al. 2020, a pure
deep residual-MLP stack for univariate series). A gradient-boosted decision
tree (GBDT) ensemble — LightGBM here — was chosen instead, for reasons tied
to this data's actual shape and this deliverable's actual constraints, not
unfamiliarity with the alternatives:

1. **Data shape.** ~59K origin-based training rows across 3 channels and
   ~136 campaigns, most individual campaign histories short and heavily
   zero-inflated (§4, §C.2's hurdle-model motivation). Deep sequence models
   like DeepAR/TFT/N-BEATS earn their keep with many long, dense,
   individually-informative series where cross-series-learned seasonality
   and attention over long histories pay off. This problem is closer to
   cross-sectional tabular-with-time-derived-features than "many long
   homogeneous series" — GBDT's home turf.
2. **Native distributional fit.** Revenue's zero-inflated, heavy-tailed
   shape is exactly what Tweedie deviance (§5) is built for, and LightGBM
   ships it natively. DeepAR's built-in distribution heads (Student's-t,
   Negative-Binomial, Beta) don't include Tweedie; reproducing this
   compound-Poisson-Gamma behavior there would mean writing and validating
   a custom distribution head from scratch — a nontrivial undertaking on
   its own, and out of scope under this deadline.
3. **Direct quantile regression.** LightGBM's `objective="quantile"` gives
   independently-trained quantile heads that combine cleanly with post-hoc
   CQR calibration (§6). TFT also has native quantile output and would be
   competitive here specifically; DeepAR does not — it samples from a
   fitted distribution instead, an extra approximation layer.
4. **Feature flexibility + monotonic constraints.** The feature set mixes
   a real decision variable (`planned_future_daily_budget`), calendar
   effects, and expanding/trailing aggregates (§4) — GBDT's strength, and
   it supports `monotone_constraints` natively (§C.3's "more budget ⇒ not
   less revenue" guarantee on the point model). Enforcing an equivalent
   constraint inside an RNN/attention model is far less standard.
5. **Interpretability for §H.** Gain-based importance and SHAP (§5b) plug
   directly into the LLM grounding context. Extracting comparably faithful
   per-decision attributions from a deep sequence model is much less
   direct.
6. **Iteration speed under the deadline.** A full walk-forward + grouped CV
   + tuning sweep over the whole ensemble trains in low-single-digit
   minutes on CPU. A DeepAR/TFT/N-BEATS model held to the same rigor
   (proper per-fold hyperparameter search) would need GPU time and
   materially longer debug cycles — a real cost against the July 19
   deadline, not a hypothetical one.

**The honest counterpoint:** TFT in particular would likely close some or
all of this gap, and might win outright, on a dataset with many more
campaigns each carrying hundreds of days of dense history, or if the actual
deliverable were full multi-step trajectory forecasts rather than
point-in-time revenue-over-a-horizon aggregates. Given the data actually
available here, GBDT was the more defensible choice for this specific
problem — not a claim that it dominates the alternatives in general.

**2026 update — zero-shot time-series foundation models.** The 2024-2025
wave (Chronos, TimesFM, Moirai, Lag-Llama, TimeGPT) extrapolate history
only; Chronos-2 (Amazon, Oct 2025) is the first of these to add
covariate-conditioned zero-shot forecasting, meaning the earlier blanket
claim that "none of these can condition on a planned future budget" is now
partially outdated. This doesn't change the §5a decision — none of them
ship a Tweedie-shaped head for this data's zero-inflation, and this
deliverable's actual decision variable (`planned_future_daily_budget`)
still needs the monotonic-constraint treatment in §C.3 either way. A
zero-shot, unconditional Chronos-2 cross-check against the account-total
series was attempted as a documented ablation, in the same spirit as §5b/§4
below, but was blocked by a genuine environment constraint rather than a
judgment call: torch's default Linux wheel unconditionally preloads CUDA
shared libraries at import time (confirmed directly — it fails even for
pure CPU inference if the ~4GB nvidia-* dependency stack isn't present),
which didn't fit in this sandbox's disk quota. `scripts/chronos_zero_shot_check.py`
is a ready-to-run, tested (its data-loading and naive-baseline path both
execute correctly against this project's own real data — only the Chronos
call itself is unverified) standalone script for running this specific
comparison on any machine with normal disk headroom.

### 5b. Point-model ablation: Tweedie vs. hurdle vs. CatBoost, plus a blend (§C.1)

The plan's exit criterion is explicit: build a two-part hurdle model
(classifier for P(revenue > 0), Gamma regressor for E[revenue | revenue >
0]) and adopt it **only if it beats the Tweedie point model on held-out
pinball loss**. This round extends that same honest-comparison discipline
to a third, structurally different candidate — **CatBoost's own Tweedie
implementation** (`modeling.train_catboost_point_model`) — plus a fourth,
implicit candidate: the simple **equal-weight average of all three**
(`modeling.compare_point_models_pinball_multi`), the concrete test of
whether ensemble diversity actually pays off here rather than an assumption
that it does.

**Why CatBoost specifically, and why this is genuine diversity rather than
"a second library for its own sake."** LightGBM Tweedie and LightGBM hurdle
are two objectives fit by the *same* leaf-wise boosting implementation, the
same split-finding algorithm, the same native categorical handling —
correlated by construction on whatever that one implementation
systematically gets wrong. CatBoost differs on three axes that plausibly
matter for this data: ordered boosting (a permutation-driven training
scheme designed to reduce the prediction-shift/target-leakage bias ordinary
gradient boosting has on its own training set) instead of LightGBM's
standard scheme; native ordered-target-statistics encoding of
`channel`/`campaign_type` as categoricals, a different mechanism from
LightGBM's own native categorical split-finding; and its own independent
implementation of the Tweedie deviance objective (So & Valdez 2024,
arXiv:2406.16206, studies exactly this family × objective combination on
CatBoost — cited in §5's header as background on the modeling idea; this is
the first place in this codebase that literature is actually acted on
rather than just cited). Two structurally different models are more likely
to be wrong on different rows than the same model wrong on the same rows
twice — the precondition for a blend beating either candidate alone.

All three candidates are train-only fits (`X_train`/`y_train` alone, never
touching the calibration slice or final holdout), compared via
`pinball_loss(..., q=0.5)` on the **calibration slice** — the same
out-of-sample window CQR itself uses. CatBoost gets the identical
`TWEEDIE_VARIANCE_POWERS` sweep grid as the LightGBM point model (apples to
apples on the same features, same folds, different library), but the same
single-split economy hurdle already used (not the full multi-fold sweep) —
it's a losing-so-far ablation candidate at this point in the pipeline too,
not the shipped model, and doubling the sweep cost isn't worth it twice.

**On the current shipped (final, full-settings) run:**


pinball(q0.5): Tweedie=6111.73, hurdle=5946.71, CatBoost=5894.59,
**equal-weight blend=5845.57 — the blend wins**, by +4.4% vs. Tweedie. All
three individual candidates land within a fairly tight band of each other
(Tweedie the weakest, CatBoost the strongest individually, hurdle in
between), consistent with §5b's own prior expectation that no single
candidate should dominate decisively on this data — but the blend beating
all three individually is the interesting result: it means the candidates'
errors are usefully uncorrelated enough for averaging to help, the concrete
condition ensemble diversity is supposed to produce. Because the blend won,
all three candidates are refit on train+calibration for production (not
just the nominal single winner) — `predict.py` reconstructs the shipped
point forecast as the equal-weight average of all three at inference.

Whichever candidate wins (including the blend) is refit on train+calibration
and becomes the bundle's active point model
(`bundle["point_model_selected"]`); production refitting is conditional —
if a single candidate wins outright, only that one gets refit for
production (same cost discipline as before); if the blend wins, all three
get refit, since reconstructing a blend at inference needs all its parts.
The loser(s) are discarded, not shipped as dead weight. The full comparison
and the winner are saved to `bundle["point_model_ablation"]` and shown in
the app's Model Reliability tab regardless of outcome. Either way, the
quantile ensemble driving the fan chart, the what-if slider, and the
reconciled hierarchy is completely unaffected — only the single
supplementary `revenue_mean_<winner>` column in `predictions.csv` depends
on which candidate (or blend) won.

**Why the winner isn't always the same, and that's fine.** The margin
between candidates has stayed modest, not decisive, across every run so
far, and has flipped winner more than once across this project's own
history as the feature set and data changed underneath it (see the git
history of this section for the earlier 2-way-only numbers). This is the
correct, expected behavior of a genuine held-out ablation — it answers
"which wins on THIS data, right now," not "which wins in general" — and is
exactly why this stays a live, re-run-every-time check rather than a
hardcoded assumption baked in once and never revisited.

### 5c. Real SHAP values (§H.1)

Gain-based feature importance (still shown, for comparison) is a training-
time heuristic — how much a feature reduced loss while the tree was being
built. It is not the same thing as how much that feature actually moved
any *specific* prediction. `modeling.shap_feature_importance` computes real
Shapley attributions via `shap.TreeExplainer` on the median quantile
booster, sampled from the final holdout (up to 2,000 rows — TreeExplainer
is exact regardless of sample size; the sample only affects how many rows
the mean |SHAP| is averaged over). Both the ranked mean(|SHAP|) list (drop-
in shape-compatible with the old gain-based list, so `llm_insights.py`'s
grounding context needed no changes) and the **signed** mean SHAP per
feature are kept — gain-based importance can't tell you that a feature
pushes predictions up vs. down on average; SHAP can. This is what actually
feeds the LLM causal-summary's `top_drivers` now (§11), with gain-based
importance falling back automatically for any bundle trained before this
existed.

Top 5 by mean |SHAP| on a production run (2,000 holdout rows sampled):

| Rank | Feature | mean\|SHAP\| | mean signed SHAP |
|---:|---|---:|---:|
| 1 | `planned_future_daily_budget` | 9,913.5 | −2,671.5 |
| 2 | `horizon_days` | 7,015.8 | −1,476.1 |
| 3 | `campaign_expanding_revenue_mean` | 1,513.1 | −464.6 |
| 4 | `revenue_roll_std_28` | 1,006.8 | +137.2 |
| 5 | `trailing_roas_56` | 699.4 | −441.0 |

Same top-2 features as gain-based importance (§5, `planned_future_daily_budget`
then `horizon_days`), which is reassuring cross-validation between the two
methods — but the full ranking isn't identical past that, which is
expected: gain measures how much a feature reduced training loss while
trees were being built, SHAP measures actual per-prediction attribution on
held-out data — related, but not the same question. Two of this round's new
§4 extended-funnel features (`ctr_roll_28`, `video_view_rate_roll_28`) land
in the full top 10.

**Per-channel, not just global (fixed).** The table above is the
*account-wide* ranking. Stress-testing the LLM causal summary found that
every scope's narrative — `total`, `bing`, `google`, `meta` — was using
this exact same global ranking for its `key_drivers`, byte-identical across
all four sections of a real `output/causal_summary.json` run. Not
fabricated (it's real SHAP), but it silently undercut the "grounded,
per-channel" narrative: Meta's causal summary was citing the same top
drivers as Bing's. Root cause: `predict.build_causal_summaries` computed
`top_drivers` once from `bundle["shap_importance"]["top_features"]`,
outside the per-scope loop, and passed the identical list into every
scope's grounding context. Fixed by extending `shap_feature_importance`
with an optional `groups` argument (e.g. `channel`) that breaks the
*already-computed* SHAP matrix down per group at no extra `TreeExplainer`
cost, stored as `bundle["shap_importance"]["by_group"]`; `predict.py` now
looks up that channel's own breakdown for each channel-scoped narrative and
only falls back to the global ranking for the `total` scope, a channel
with too few holdout rows to have a stable breakdown of its own
(`min_group_rows=20`), or an older bundle trained before this existed.
Verified on the real shipped bundle — the three channels now surface
genuinely different top drivers, e.g. Bing leads with
`planned_future_daily_budget` while Meta leads with `horizon_days` followed
by `campaign_expanding_roas` (a feature that doesn't appear in Bing's or
Google's top 5 at all). See `tests/test_shap_per_channel_drivers.py`.

---

## 6. Conformal calibration (`src/modeling.py`: `fit_cqr_correction`, `apply_cqr`)

Conformalized Quantile Regression (Romano, Patterson & Candès, NeurIPS 2019).
A **time-ordered** calibration slice (rows with `origin_date` strictly after
the training cutoff, `2025-09-28`, and strictly before the calibration
cutoff, `2025-12-21` — never a random split, which would leak future
information into calibration) is used to compute a single correction
`Q̂ = Quantile_{1-α}(max(q_lo - y, y - q_hi))` for three interval pairs:

| Interval | α | Q̂ |
|---|---:|---:|
| (P5, P95) | 0.10 | 6,850 |
| (P10, P90) | 0.20 | 5,389 |
| (P25, P75) | 0.50 | 1,915 |

Final-holdout reliability (never touched during any tuning — see §7),
**after** CQR correction:

| Nominal | Empirical |
|---|---:|
| 90% | 94.9% |
| 80% | 90.5% |
| 50% | 59.6% |

Coverage is slightly conservative (empirical > nominal) at every level,
which is the safer direction to err for a budget-planning tool.
**This is measured on unreconciled, per-campaign-row quantiles only — see
§8c for whether that same coverage survives being pulled through §8's
hierarchical reconciliation (it doesn't, uniformly, and §8c measures
exactly where it doesn't).**

**Known limitation, stated plainly.** The production quantile models are
refit on train+calibration combined *after* `Q̂` is computed (from train-only
models), to give the deployed model more data. This is a standard practical
compromise, not textbook CQR — exact finite-sample coverage guarantees
technically apply to the train-only model, not the refit one. The
final-holdout numbers above are the honest empirical check on the actual
deployed models, and they hold up.

### 6a. Adaptive Conformal Inference (§D.2, optional — "if you have time")

CQR above is **static**: one correction `Q̂` is fit once and applied
unchanged to every future prediction. `src/adaptive_conformal.py`
implements ACI (Gibbs & Candès, NeurIPS 2021) — a running miscoverage
target `alpha_t` updated online, `alpha_{t+1} = alpha_t + gamma·(alpha_target
- err_t)`, so the correction adapts if the model starts systematically
under- or over-covering instead of carrying a stale correction indefinitely.

This is genuinely optional per the plan, and is **not** wired into
`predict.py`'s shipped calibration — for two honest reasons: (1) it's
inherently sequential (needs true outcomes arriving one at a time to update
`alpha_t`), which fits a monitoring/rolling-retrain job far more naturally
than this project's single-shot batch inference; (2) it's a hedge against
distribution shift, and the right question is whether that shift is
actually present here, not whether the technique sounds nice.

**It's checked for, not assumed — and on this project's own final-holdout
timeline, real drift shows up.** `compare_static_vs_adaptive` runs both
methods on the SAME chronologically-ordered holdout sequence (90% nominal
band): a static correction fit on the first 20% and applied unchanged to
the rest, vs. ACI updating throughout. Measured result on the current
production training run: **static coverage 94.6%** post-warmup — a few
points more conservative than the reliability diagram's already-conservative
94.9% headline number, once you're far enough past the warm-up window — vs.
**ACI coverage 89.6%**, within half a point of the 90% nominal target, with
a correspondingly slightly narrower mean interval (33,573 vs. 34,033). This
is a real, measured finding about this data (the static correction runs a
bit conservative later in the holdout's date range; ACI tracks it back down
toward nominal), not a toy demonstration — see the Model Reliability tab and
`bundle["adaptive_conformal_report"]` for the live numbers on any given
training run, and `tests/test_adaptive_conformal.py` for the automated
version of this comparison (including a synthetic variance-shift scenario
that isolates ACI's under-coverage-recovery behavior specifically, since a
single real holdout run only ever shows one direction of drift).

---

### 6b. Conformal PID control (§D.2b/§D.2c, optional — natural next step after 6a)

ACI's update rule is, in control-theory terms, a pure proportional (P)
controller acting on the miscoverage signal. Angelopoulos, Candès &
Tibshirani ("Conformal PID Control for Time Series Prediction," NeurIPS
2023) add an integral (I) term — a saturated running sum of past
miscoverage, correcting a *persistent* bias a pure-P controller can leave
uncorrected — and a derivative (D) term, which their own paper implements
as a learned "scorecaster": a separate lightweight model forecasting the
next nonconformity score directly. `src/adaptive_conformal.py` now ships
**two** honestly-scoped D-term variants on top of the same P+I core, so
"does a genuinely learned scorecaster beat a cheap heuristic" is a measured
comparison rather than an assumption either way:

- **`run_conformal_pid_control` (§D.2b) — derivative-of-error proxy.** The
  original scoped-down D term: current miscoverage signal vs. an EWMA of
  its recent history — a defensible PID component in its own right, not a
  claim of reproducing the paper's own D term. Implements the P/I terms as
  a genuine extension of the existing ACI function, not a parallel
  reimplementation: with `ki=0, kd=0` its update rule is mathematically
  identical to plain ACI's, confirmed by
  `tests/test_adaptive_conformal.py::test_pid_matches_plain_aci_when_i_and_d_gains_are_zero`.
- **`run_conformal_pid_control_learned_scorecaster` (§D.2c, added this
  round) — the genuine version.** A ridge-regularized AR(3) model
  (`_fit_predict_next_score`) refit from scratch, on the full expanding
  score history, at *every single step* — a real, small, genuinely learned
  forecaster of the next nonconformity score, not a fixed formula. Where it
  differs from the derivative-proxy variant is deliberate, not an
  oversight: an early version tried folding the scorecaster's forecast into
  the same alpha_t recursion the proxy variant uses, normalized by the
  current conformal threshold `q_hat_t` — this was genuinely unstable
  (verified directly, not hypothetical): `q_hat_t` routinely sits near zero
  on a well-calibrated process, so the normalized ratio saturated at its
  clip bound almost every step and alpha_t ran away to its boundary within
  a few dozen steps. The scorecaster's forecast is already in the same
  units as `q_hat_t` (raw nonconformity score, i.e. revenue), so the stable
  design adds it directly where `q_hat_t` already acts — widening or
  narrowing the interval itself, with a safety clamp preventing the
  correction from ever inverting the raw interval — while alpha_t itself is
  driven by P+I only. Confirmed by
  `tests/test_adaptive_conformal.py::test_pid_learned_scorecaster_matches_proxy_pid_when_kd_is_zero`:
  with `kd=0`, both variants' `alpha_t_history` are byte-identical.

**Measured result on the current production training run** (same holdout,
same 90% nominal band, same warm-up window as §6a — `compare_static_vs_adaptive_vs_pid`
runs all four methods on the identical sequence):


| Method | Post-warmup coverage | Mean interval width | Final alpha_t |
|---|---:|---:|---:|
| Static CQR | 94.6% | 34,033 | — |
| Adaptive (ACI) | 89.6% | 33,573 | 0.220 |
| PID (derivative-proxy D) | 89.9% | 48,997 | 0.333 |
| PID (learned-scorecaster D) | 89.9% | 94,269 | 0.040 |

An honest, mixed result, reported as such rather than smoothed over: both
PID variants land at essentially the same coverage (89.9%, marginally
closer to the 90% nominal target than ACI's 89.6%), but the learned
scorecaster is nearly **twice as wide** as the derivative-proxy variant for
that identical coverage on this specific real holdout — the more
sophisticated D-term is not automatically the better one here. This is a
genuinely useful negative-leaning result, not a wasted comparison: it means
the derivative-of-error-signal proxy is, on this data, a more efficient
(narrower-for-the-same-coverage) D-term than a literal implementation of
the paper's own design — plausibly because a revenue series this volatile
gives a 3-lag AR scorecaster relatively little short-horizon
autocorrelation to actually exploit, so its forecasts add more width-driving
noise than signal relative to the cheaper heuristic. Both remain reported
for comparison, not substituted for the shipped static CQR default.

Reported for comparison, like ACI, not substituted for the shipped static
CQR default — see the Model Reliability tab's "Adaptive Conformal Inference
& Conformal PID control" expander and
`bundle["adaptive_conformal_report"]["pid"]` /
`bundle["adaptive_conformal_report"]["pid_learned_scorecaster"]` for the
live numbers on any given training run.

---

## 7. Honest evaluation protocol (`src/train.py`)

Three chronological, non-overlapping slices, by resolved-target date:
**train** (rows resolved before the 70th-percentile origin date, 35,269
rows) → **calibration** (resolved before the 85th percentile, 3,025 rows,
used *only* for §6) → **final holdout** (everything after, 8,808 rows,
2025-12-22 → 2026-05-06). The holdout is scored exactly once, at the end of
`train.py`, using the final production models:

| Metric | Raw quantiles | CQR-calibrated |
|---|---:|---:|
| CRPS | 5,050 | 5,008 |
| WAPE (median) | 36.5% | 36.5% (unchanged — CQR only touches the interval endpoints, not the median) |

By horizon:

| Horizon | n | WAPE (median) | CRPS |
|---|---:|---:|---:|
| 30 days | 4,318 | 41.6% | 2,788 |
| 60 days | 2,729 | 35.7% | 5,784 |
| 90 days | 1,761 | 33.9% | 9,251 |

WAPE improves at longer horizons; CRPS grows with horizon because it's
measured in absolute revenue units over a larger cumulative target, not
because longer-horizon forecasts are relatively worse.

**Why 30-day WAPE is worse — quantified, not just asserted.** This isn't a
30-day-specific model weakness; it's the 30-day-window *target itself*
being intrinsically noisier relative to its own scale. Measured directly
on the holdout's actual (not predicted) revenue:

| Horizon | Coefficient of variation (std/mean) | % of rows with exactly zero revenue |
|---|---:|---:|
| 30 days | 1.551 | 20.0% |
| 60 days | 1.253 | 7.7% |
| 90 days | 1.046 | 1.5% |

A campaign that's paused or between flights for part of the window is far
more likely to show zero total revenue over a short 30-day span than over
a 90-day one — one in five 30-day windows in the holdout is an exact zero,
versus one in sixty-seven at 90 days. WAPE is a ratio (`sum(|error|) /
sum(|actual|)`); a target distribution this much more zero-heavy and
higher-variance relative to its own mean inflates the ratio's denominator's
effective "floor" even for an equally well-calibrated model. Longer windows
average daily volatility out (a basic aggregation-reduces-relative-variance
effect), which is the real driver here, not degraded model quality at
short horizons specifically.

---

## 7a. Naive baseline comparison — skill vs. a trivial alternative (`src/modeling.py`: `naive_pace_forecast`, internal plan §G.2.5)

"36.5% WAPE" has no meaning on its own — the obvious next question from any
judge is "compared to what?" This section answers it directly, rather than
leaving the headline number to stand unchallenged.

**The baseline.** `naive_pace_forecast` computes the simplest thing an
agency could do without any model at all: assume the next window looks like
the recent past. Concretely, `revenue_roll_mean_28` (the same leakage-safe
trailing-28-day daily mean the production model itself uses as one input
among fifty-plus) × `horizon_days`. A campaign with no rolling history yet
naively forecasts zero — the same "no history" default used elsewhere in
this codebase, not a special case invented for this comparison. It is never
trained, never tuned, and is scored on the **identical** final-holdout rows
and the **identical** WAPE metric as the production model in §7 above — the
only fair way to compare.

| | WAPE (median) | Model's improvement |
|---|---:|---:|
| Naive baseline ("continue at recent pace") | 126.8% | — |
| **This model (CQR-calibrated)** | **36.5%** | **71.2%** |

By horizon — the model's advantage over the naive baseline *grows* with the
forecast window, which is the opposite of what a purely-extrapolative
approach would give you (a flat continuation of recent pace gets *less*
reliable the further out it's projected, since it has no way to see a
planned budget change coming):

| Horizon | Naive baseline WAPE | Model WAPE | Improvement |
|---|---:|---:|---:|
| 30 days | 110.0% | 41.6% | 62.2% |
| 60 days | 124.5% | 35.7% | 71.3% |
| 90 days | 139.5% | 33.9% | 75.7% |

**Why the naive baseline gets worse, not better, at longer horizons.** It
has no access to `planned_future_daily_budget` at all — it just scales the
recent daily rate up linearly by the window length. When a real budget
change is planned (the whole point of this tool), that gap between
"continue as before" and "here's what actually happened" widens the further
out the projection runs. The production model, by contrast, is explicitly
trained on `planned_future_daily_budget` and the Hill saturation curve
relationship (§9), so it captures exactly the signal the naive baseline
structurally cannot.

**Deliberately not shipped as a fallback prediction path.** This function
exists solely to produce the comparison table above during evaluation
(`train.py`) — it is never called from `predict.py` or exposed anywhere in
the frontend as an actual forecast. See `tests/test_naive_baseline.py` for
unit coverage of the function itself.

---

## 8. Hierarchical reconciliation (`src/reconciliation.py`)

Hierarchy: **total → channel → channel/campaign_type → channel/campaign_type/campaign_id**
(campaign_type is namespaced under channel — the same label, e.g.
`"search"`, recurs across channels, and a hierarchy must be a strict tree).
Built with Nixtla's `hierarchicalforecast` (`aggregate`, `HierarchicalReconciliation`),
not hand-rolled. Base forecasts: bottom level = the pooled quantile model;
upper levels = the library's own summing-matrix aggregation of the bottom
level. Point reconciliation uses `MinTrace(method="mint_shrink")`
(Wickramasuriya, Athanasopoulos & Hyndman 2019) with **genuine out-of-fold
walk-forward residuals** (not in-sample-fitted values) for the shrinkage
covariance, plus `BottomUp` as a trivial coherence sanity check.

Measured coherence error (`max_abs_coherence_error`, live `run.sh` output)
at every horizon: **0.000000** — channel totals sum exactly to the grand
total, and campaign-type totals sum exactly to their channel, to floating
point precision.

**Interval (probabilistic) reconciliation — genuinely two-tier, for a
measured reason.** `hierarchicalforecast`'s `Conformal` reconciler
(Principato, Stoltz, Amara-Ouali, Goude, Hamrouche & Poggi 2024) scores the
*reconciled* calibration forecast (`S · P · ŷ_cal`), which needs every
bottom node's calibration value jointly present at each shared calibration
timestamp. Checked directly against this project's own walk-forward OOF
calibration data: at the **channel level** (3 channels + total), **every
one of 308 calibration dates has all series present** — full joint
alignment holds, so the real, unmodified `hierarchicalforecast.methods.
Conformal` is used there. At the **individual-campaign level, zero of 308
dates have all ~64+ campaigns present simultaneously** — campaigns start
and stop at different times, a real property of this data, not a bug —
so joint reconciled conformal is mathematically inapplicable there without
fabricating observations.

So: total + channel nodes get genuine joint conformal intervals from the
library itself. `campaign_type` and individual `campaign` nodes get a
**marginal (per-node) split-conformal correction** instead — the same
nonconformity-score idea `Conformal.get_prediction_quantiles` itself uses
(signed residual quantiles), computed independently per node from that
node's own out-of-fold history, requiring no cross-node date alignment.
Both tiers construct each node's band as `reconciled_median + offset`,
centered so the already coherence-checked `MinTrace` median is *never*
altered by either tier — only the width/skew of the band around it is. A
handful of genuinely too-sparse campaigns (<15 OOF observations; 3/64 on a
representative run) fall back to the original documented shortcut
(rescaling the naive-summed band by the point-forecast's `MinTrace`
adjustment ratio).

Measured on a real run (horizon=30): **4/4 total+channel nodes** got
genuine joint conformal, **70/73 deeper nodes** got the marginal
per-node correction, **3** fell back to the rescale shortcut — while point
coherence stayed exactly **0.000000** throughout, since neither tier ever
touches the median. See `tests/test_reconciliation_probabilistic.py` for
the automated version of this check.

**ROAS ranges at every level — not just a per-campaign point.** The brief
names "channel-level / campaign-type / campaign-level ROAS ranges" as its
own deliverable, alongside revenue, and explicitly lists "expected blended
ROAS" as a top-level required output. Revenue already got full quantile
treatment through every step above; ROAS previously existed only as a
single derived point per campaign (`revenue_p50 / spend`, computed in
`predict.py`) — no range, and nothing at all above the campaign level,
since the reconciliation hierarchy never tracked spend.

Fixed by giving spend the same hierarchical treatment as revenue, minus the
probabilistic machinery it doesn't need: `reconcile_forecast` now accepts
an optional `spend` column on the bottom-level input (the assumed total
spend for that campaign over the horizon — a fixed scenario input the
person planning the budget chose, not itself uncertain) and aggregates it
up the identical hierarchy with the identical summing matrix `S`, as a
plain bottom-up sum — no MinTrace/conformal correction needed, since
there's no distributional uncertainty in an assumed number to correct.
`roas_q{q}` at every node is then that node's own (already fully
reconciled) revenue quantile divided by that same node's own spend.
Dividing every quantile by the same positive constant preserves their
order automatically, so no separate quantile-crossing fix is needed for
the ROAS columns themselves; a node with zero spend gets `NaN`, not a
crash or a fabricated `inf`. Backward compatible: omit `spend` entirely and
`reconcile_forecast` behaves exactly as before, with no ROAS columns at
all (see `tests/test_roas_ranges_reconciliation.py`).

Surfaced in the app as a headline "Blended ROAS" metric (with its P10–P90
range as a caption) at whatever level the sidebar's current channel/type
scope resolves to, a companion ROAS bar chart next to each level's revenue
chart on the Breakdown tab, and a per-campaign `roas_p10_p90_range` column
alongside the existing point `roas_p50` on the Forecast tab's table.

---

## 8c. Reconciled-interval calibration across hierarchy levels (§E re-audit)

**Coherence and calibration are separate properties** (Principato, Stoltz,
Amara-Ouali, Goude, Hamrouche & Poggi 2024 — already cited above for the
genuine-conformal tier). §8's `max_abs_coherence_error == 0.000000` says the
reconciled *median* sums correctly up the hierarchy, every run. It says
nothing about whether the reconciled *band* around that median is still as
well-calibrated as §6's base-level reliability diagram (94.9% / 90.5% /
59.6%, measured only on unreconciled, per-campaign-row CQR quantiles). That
was a genuinely open question for this project until this round.

`reconciliation.evaluate_reconciled_calibration` (pure evaluation — reuses
`reconcile_forecast` exactly as `predict.py` calls it, and reuses
`modeling.reliability_diagram` exactly as §6/§7 already do, no new model or
fit) runs the actual production reconciliation path against every one of
the 318 holdout origin_date/horizon snapshots, then pools empirical-vs-
nominal coverage separately at each of the four hierarchy levels:

| Level | n snapshots | n observations | 90% nominal → empirical | 80% → | 50% → |
|---|---:|---:|---:|---:|---:|
| total | 318 | 318 | 0.90 → **0.962** | 0.80 → **0.852** | 0.50 → **0.550** |
| channel | 318 | 951 | 0.90 → **0.814** | 0.80 → **0.742** | 0.50 → **0.606** |
| campaign_type | 318 | 1,890 | 0.90 → **0.935** | 0.80 → **0.910** | 0.50 → **0.697** |
| campaign (reconciled) | 318 | 6,590 | 0.90 → **0.713** | 0.80 → **0.626** | 0.50 → **0.434** |

**The honest finding: reconciliation does not preserve calibration, and it
degrades in a specific, explainable direction.** `total` and
`campaign_type` stay conservative (empirical ≥ nominal, the safer
direction) — consistent with §6's base-level result. `channel` and,
notably, the reconciled `campaign` level (the same physical rows §6 already
checked, but now pulled through MinTrace + the two-tier conformal
correction rather than left as raw CQR output) both **under-cover**
relative to nominal — most visibly at `campaign`: a nominal 90% band only
contains the actual outcome 71.3% of the time post-reconciliation, well
below the pre-reconciliation base-level number for the same rows. This is
mechanically explainable, not a bug: MinTrace reprojects every node's point
forecast toward hierarchy-wide consistency, and the two-tier interval logic
(§8: genuine joint conformal at total/channel, marginal split-conformal
elsewhere) was never validated to preserve coverage through that
reprojection — it was validated (§6) only before reconciliation touched
anything. This is exactly the failure mode Principato et al. describes in
the abstract, now measured concretely on this project's own data rather
than cited as a general concern.

**What this changes, and what it doesn't.** The reconciled hierarchy
(§8, `output/reconciled_hierarchy.csv`) is still coherent, and its point
forecasts are unaffected by this finding. What's now honestly known is that
its *intervals* at the channel and individual-campaign level run
meaningfully narrower than they should for the stated confidence —
`campaign_type` and the account `total` remain trustworthy at face value,
`channel`-level and reconciled-`campaign`-level bands should be read as
tighter than their nominal coverage actually delivers until this is
corrected (a natural next step: widen the marginal split-conformal
correction at those two levels using this measured gap, rather than the
train/calibration-slice α it currently targets — not implemented this
round, kept in scope for a future pass rather than rushed in without its
own holdout check).

See `tests/test_reconciled_calibration_by_level.py` for both a real-data
integration check (does this run end-to-end against the production path)
and a fully synthetic, ground-truth-known unit test (are the reported
numbers actually correct, not just structurally present) — and the
**Model reliability** tab (§12) for the live per-run chart.

---

## 8a. Campaign consistency validation (`schema_mapper.validate_campaign_consistency`)

"Validating campaign consistency" is its own named bullet in the brief's
Working Prototype deliverables, distinct from the schema/type validation in
§3 — nothing previously checked for it. Runs on the canonical frame (so
campaign_id is already channel-namespaced; a campaign_id colliding across
channels isn't even representable here anymore after the fix below, so that
specific check doesn't need to exist separately) and flags, without ever
raising or modifying the data:

1. **Duplicate `(campaign_id, date)` rows** — a campaign should have at
   most one row per calendar day.
2. **A campaign_id reporting more than one distinct `campaign_type`**
   across its history — campaigns are assumed not to change type
   mid-flight.
3. **A campaign_id reporting more than one distinct `campaign_name`**
   across its history — a possible silent rename or ID reuse.

Currently zero issues on this project's own data (checked directly), but
the check itself didn't exist before this round, so nothing would have
caught it on a grading dataset the team hasn't seen. Runs automatically in
`validate.py`, `train.py` (logged, stored in the bundle), and
`predict.py` (written into `data_health.json` for the app's Data Health
tab). See `tests/test_campaign_consistency_validation.py`.

---

## 8b. A real ingestion robustness bug, found and fixed this round

While re-testing "extra columns shouldn't break ingestion" directly (adding
synthetic junk columns to a real production file and re-running the
pipeline), an actual vulnerability turned up: `channel` was included in the
same fuzzy-match fallback pool as every other canonical field (§3), with no
special handling — but unlike spend/revenue/clicks (numeric, checked for
plausible ranges immediately after mapping), `channel` is free text with no
such backstop. A column merely *named* something like
`some_new_platform_field_2026` scored 75/100 against `channel`'s alias pool
(`"channel", "platform", "source", "network"`) — comfortably above the
65-point optional-tier threshold — and would silently overwrite every row's
channel with that column's actual (unrelated, garbage) values. Worse than a
crash: nothing downstream would flag it.

Fixed by excluding `channel` from the Pass-2 fuzzy pool entirely — it's
still eligible for the exact-name match in Pass 1, and falls back to its
own purpose-built, safer 3-tier resolution (explicit override → native
`channel` column → filename-derived) exactly as designed, rather than
having a loose string-similarity heuristic potentially preempt that
fallback with a bad guess. Confirmed zero effect on this project's own
three real data files (all resolve channel via filename pattern, none ever
relied on the fuzzy path for it). See
`tests/test_bugs_3_4_column_matching_and_namespacing.py`.

---

## 9. Hill saturation curves (`src/budget_curves.py`)

Explicitly **not** a media-mix-model / attribution engine — a secondary
sanity and scenario-scaling layer on top of the pooled model's own budget
response. `response(spend) = L · spend^n / (Kⁿ + spendⁿ)`, fit via
`scipy.optimize.curve_fit` on **daily-aggregated** `(spend, revenue)` pairs
per `(channel, campaign_type)` — an individual campaign rarely has enough
distinct spend levels to identify a saturation curve on its own, but a
channel × type group usually does.

**Fit-quality gate, not just a point-count gate.** An earlier version of
this module accepted any fit that cleared `n_points >= 8` and
`n_unique_spend_levels >= 5`, regardless of how well the curve actually
tracked the data. Loading the resulting bundle directly surfaced a real
problem: `bing/PerformanceMax` (140 points) and `google/VIDEO` (467 points)
both "passed" with their steepness parameter `n` pinned exactly at the
optimizer's upper bound (6.0) — the classic sign `curve_fit` ran to the
edge of its search space rather than converging on a real optimum — and an
R² check confirmed it: 0.022 and 0.054 respectively, i.e. the curve
explained essentially none of the variance beyond the group's own mean.
`bing/Search` (734 points, comfortably past the count thresholds) told the
same story at R²=0.026. All three were being used at full confidence in
the budget-slider scaling despite being statistically indistinguishable
from noise. `fit_hill_curves` (`src/budget_curves.py`) now rejects a fit
if **either** signal fires — R² below 0.10, or the fitted `n` landing on
its bound — and falls back to the flat historical-ROAS line instead (still
a defensible response curve, just linear rather than saturating), with the
specific reason recorded in that group's `note` field so it's auditable,
not silently swapped.

With the gate applied to this run's data: **7 of 14 groups** fit a genuine
saturation curve (`google/SHOPPING` R²=0.79, `google/PERFORMANCE_MAX`
R²=0.46, `google/SEARCH` R²=0.41, `google/DEMAND_GEN` R²=0.11, and three
`meta` groups — `unclassified` R²=0.86, `remarketing` R²=0.33,
`prospecting` R²=0.27). The other 7 fall back to the flat-ROAS line: 3 for
insufficient data (`bing/Audience` — the zero-revenue group from §2 —
`bing/Shopping`, `google/DISPLAY`), and 4 for failing the quality gate
(`bing/PerformanceMax` R²=0.03, `bing/Search` R²=0.09, `google/VIDEO`
R²=0.005 with `n` pinned at its bound, `meta/brand` R²=0.08). This is a
strictly more conservative and more honest count than the original ungated
version — it is not hardcoded to any particular channel or time period, so
it would catch an analogous spurious fit on a different dataset just as it
caught these. `hill_sanity_check` continues to flag — never silently
override — cases where the Hill curve's implied response to a budget
change diverges sharply from the pooled model's own prediction for the
same change.

**§F re-audit — recency-weighted fitting.** The fit above weighted every
historical day equally, implicitly assuming channel effectiveness is
static over the group's whole observed history — 2026 research on budget
allocation under drifting returns (Pathak, Shyamal, Mhasker & Swartz,
"Learning to Spend: Model Predictive Control for Budgeting under
Non-Stationary Returns," arXiv:2604.27186) makes the case that assumption
usually doesn't hold. `fit_hill_curves` now passes a recency-weighted
`sigma` to `curve_fit` (`recency_half_life_days=45.0` by default — a daily
point 45 days older than the group's own most recent observation carries
half the weight of today's; `None` recovers the previous equal-weighted fit
exactly, and both modes are tested in `tests/test_budget_optimizer.py`,
including a synthetic channel whose true ROAS steps from 2.0x to 5.0x
partway through the series to confirm the weighted fit actually tracks the
recent regime rather than blending it away). The R² fit-quality gate above
is scored with those SAME weights (see `_r_squared`'s docstring) — an
unweighted gate would reject a genuine recent-regime shift for disagreeing
with old, deliberately down-weighted points, defeating the point of
weighting in the first place. A visible, honest side effect this run: the
specific set of groups that clear the gate isn't identical to an unweighted
fit — `google/DEMAND_GEN` now passes (its recent data fits better than its
full history did) and `meta/brand` now fails (the reverse) — which is the
mechanism working as intended, not fit instability.

**Second real bug, found while building §9b's MPC backtest below.** The
recency weighting above only ever reached the curve-fit path — the flat
*fallback* line (`fallback_roas = revenue.sum() / spend.sum()`, used
whenever a group doesn't clear the real-fit gates) stayed a plain,
unweighted historical average regardless of `recency_half_life_days`. This
went unnoticed until §9b's backtest exercised a group that spends most of
its history in the fallback path with a genuine regime shift underneath it
— the plain average blended months of stale pre-shift history in at equal
weight with the new regime, silently erasing the shift the MPC allocator
most needs to notice, in exactly the case (a still-thin post-shift dataset
that hasn't earned a real curve fit yet) it's disproportionately likely to
hit. Fixed by computing the SAME recency weight once, up front, and
applying it to both paths — the fallback line is now a recency-weighted
average, not a plain one, whenever `recency_half_life_days` is set (see
`tests/test_budget_mpc.py`'s regime-shift tests, which fail without this
fix). No effect on this project's own real `hill_curves.json` output (7-8
of 14 groups already clear the real-fit gate either way — see above), but
a real, silent correctness gap for any group that doesn't.

---

## 9a. Cross-channel budget allocator (`src/budget_curves.py`: `optimize_budget_allocation`, §F.3)

Everything above answers "what does spend X get me in this one group?" This
section answers the question an agency actually asks: **given a fixed total
budget, what's the best way to split it across channels?** The Budget
What-If tab's slider (§I) only lets you scale every group by the same
multiplier — it cannot express "take money from an over-saturated group and
give it to one with room to grow," which is precisely where the real value
in a per-group response curve lives.

**Why dynamic programming, not a greedy marginal-return walk.** A Hill curve
with `n > 1` is genuinely S-shaped, not concave — its marginal return can be
*higher* a bit further out than it is right at zero spend (the ramp-up
region before the inflection point). A greedy "give the next dollar to
whichever group's marginal return is currently highest" allocator can get
permanently stuck starving a group of exactly the commitment it would need
to reach its productive region, because the very first dollar there looks
worse than the first dollar elsewhere. Discretizing the budget into a fixed
number of grid steps (default 200) and solving the resulting separable
resource-allocation problem with DP is correct regardless of curve shape —
`tests/test_budget_optimizer.py::test_dp_beats_naive_uniform_split_on_a_genuinely_sigmoidal_curve`
constructs exactly this case and confirms the DP finds the better,
concentrated allocation rather than defaulting to an even split.

**Kept within a defensible extrapolation range.** Each group's recommended
spend is capped at 4x its own historical daily average (falling back to the
full budget as the cap only when a group has no spend history at all) — the
same spirit as the plausibility clamps in §10, applied here so the optimizer
can't recommend moving an entire budget into a campaign type that has
historically spent a few hundred a day.

**Real result on this run** (all 14 groups, current combined daily spend
5,757): reallocating that *same total* according to the fitted curves,
instead of continuing today's actual split, is predicted to raise combined
daily revenue from 30,475 to 47,642 — a **+56.3%** uplift, with no
additional budget at all. The recommendation is intuitive, not just
numerically better: it moves spend away from `google/PERFORMANCE_MAX`
(2,825/day → 374/day, a group well past its fitted half-saturation point)
into `google/SEARCH` (1,280 → 2,188), `google/SHOPPING` (462 → 921), and
every `meta` group still in its productive region (`prospecting` 271 →
1,065, `remarketing` 171 → 662, `unclassified` 139 → 547) — exactly the
"diminishing returns here, room to grow there" story the Hill curves were
built to tell, now acted on instead of just displayed.

**Honest scope.** This uses the Hill curves' steady-state daily response,
the same secondary sanity layer described in §9 — not the calibrated
30/60/90-day quantile forecast that's this project's actual primary output.
It is deliberately framed in the frontend as directional budget-planning
guidance, not a guaranteed outcome, and it never touches training or
`predict.py` — pure post-processing over an already-computed
`hill_curves.json`, wired into the Budget What-If tab (§I) directly below
the existing single-group curve chart.

**Marginal ROAS, not just average ROAS.** `marginal_return` reports the
derivative of each group's curve at its recommended spend — the return on
the *next* dollar, the industry-standard framing for exactly this decision,
since average ROAS hides that every channel eventually saturates. On this
run's recommended allocation, the three Google groups that received
meaningful non-zero spend cluster tightly at **4.43–4.50x marginal ROAS**
— Shopping 4.50, Search 4.43, Performance Max 4.45 — exactly the textbook
signature of an optimum: marginal returns equalized across every group
with room left to move. The three `meta` groups that received spend
(prospecting 12.01x, remarketing 8.76x, unclassified 15.08x) sit well
above that level, and that isn't a flaw in the optimizer — all three land
within a few percent of their own 4x-historical-spend extrapolation cap
(§F.3 above, 96.8–98.7% of cap): the optimizer *wanted* to push each of
them further toward that same ~4.5x equalized level but the guardrail
correctly said no, leaving a real, uncapped marginal return on the table
rather than extrapolating a fitted curve past a range the underlying data
can actually support.

**An optional ROAS floor.** `min_blended_roas` reuses the DP's own
byproduct — the full revenue-vs-spend frontier at every possible total
spend level — rather than re-solving anything: it just picks a different,
earlier stopping point on a frontier that's already fully computed. A
floor set below the natural optimum changes nothing; a floor set above it
correctly recommends spending *less* than the full budget rather than
breaching the target, with `roas_floor_binding` reporting which case
applied. See `tests/test_budget_optimizer.py`'s two dedicated floor tests.

**An approximate uncertainty band, not a bare point estimate.** Each
group's `residual_std` (now computed in `fit_hill_curves`, around the
fitted curve or the flat fallback line) combines across groups assuming
rough independence into `predicted_daily_revenue_low`/`_high` — point
estimate ∓ 0.6745·combined-std, the normal-approximation 25th/75th
percentiles. Honestly scoped: this is a residual-based approximation for a
secondary sanity layer, not a formally derived predictive interval with the
same calibration guarantee as §6's CQR bands on the primary forecast.

---

## 9b. MPC-style rolling-horizon reallocation backtest (`optimize_budget_allocation_mpc`, §F.4)

§9a's allocator answers "given a fixed budget, what's the best split
*right now*." This section asks the harder, closed-loop question the
"Learning to Spend" paper (§9) is actually about: as more real data arrives
and channel effectiveness genuinely drifts, does periodically re-fitting
the curves and re-solving the allocation — rather than deciding once and
running that plan for the whole horizon — earn back more revenue than it
costs to compute?

**Honest scope, stated up front.** This is a backtest over this project's
own historical timeline, not a live continuously-arriving-data deployment
(there's no live environment to deploy into here). The closed-loop MPC loop
itself is fully real: curves genuinely get refit at every replanning point
from only the data that would genuinely have been available by then, no
peeking. What's scoped down from "the full version of the paper" is that
the ground truth used to *score* each window's decision is estimated
retrospectively from this project's own historical spend/revenue pairs,
rather than observed from a live system reacting to the recommended spend
in real time — the same walk-forward, never-let-the-decision-maker-see-the-future
methodology this project already uses everywhere else (§7 walk-forward CV,
the final holdout), applied to a control policy instead of a forecast.

**Mechanics.** The horizon (default 90 days — this project's own longest
forecasting window) is split into 30-day replanning windows (this
project's shortest window, so 90/30 lines up exactly with the report's own
30/60/90-day cadence, three windows). The **open-loop** baseline solves
`optimize_budget_allocation` exactly once, on curves fit from data strictly
before the backtest start, and reuses that same allocation, unchanged, for
the whole horizon — a real agency's "set the plan once at the start of the
quarter" baseline. **MPC** refits the curves at the start of every window
on the expanding window of data known as of that point, and re-solves. Each
planned allocation — from both methods — is perturbed by
planned-vs-realized spend execution noise (±5% Gaussian by default; real
pacing/delivery algorithms rarely land exactly on the planned number)
before being scored, so neither method is compared as an idealized,
frictionless plan. Both methods' noisy allocations are scored against an
*identical* retrospective ground-truth curve fit on what actually happened
during each window — only the allocation differs between the two numbers,
never the yardstick.

**Real result on this run:**


Backtest start 2026-03-07, three 30-day windows through 2026-06-05 (this
account's actual trailing 28-day daily spend, 2,335, held constant across
both methods so only the allocation is being compared):

| Window | Open-loop realized daily revenue | MPC realized daily revenue |
|---|---:|---:|
| 2026-03-07 → 2026-04-06 | 16,257 | 17,035 |
| 2026-04-06 → 2026-05-06 | 11,314 | 11,081 |
| 2026-05-06 → 2026-06-05 | 12,383 | 13,963 |
| **Average** | **13,318** | **14,026** |

**MPC beats open-loop by +5.3%** overall, but not uniformly — window 2 is a
real, reported loss for MPC (11,081 vs. 11,314, about −2.1%), not hidden
because it doesn't fit the headline number. This is the honest, expected
shape of the result: re-solving with fresher data is a genuine bet, not a
guaranteed win in every single window, and reporting the one window where
it didn't pay off is what makes the overall +5.3% credible rather than
cherry-picked.

A lift near zero (or negative) would still be a genuine, honest result — it
would mean channel effectiveness didn't drift enough over this particular
historical horizon for re-solving to earn back more than the
noise/estimation cost of refitting on less data per window, not that the
mechanism is broken. Either way, the mechanism itself is real and fully
tested (`tests/test_budget_mpc.py`), including a synthetic regime-shift
scenario built specifically to isolate it: a channel whose ROAS steps from
1.0x to 6.0x exactly at the backtest start confirms MPC's first-window
decision is indistinguishable from open-loop's (no new information yet
by construction), then its later-window allocation to that channel
measurably exceeds open-loop's frozen plan, and its realized revenue
(scored against the identical ground truth) beats open-loop's — the
concrete, mechanism-level proof this isn't just re-running the same
allocator on a timer for no effect.

Written to its own output artifact (`output/mpc_reallocation_backtest.json`,
`budget_curves.mpc_backtest_to_json`) and surfaced in the Budget What-If
tab's own expander, directly below §9a's interactive allocator — reported
as its own honestly-scoped result, same as ACI/PID above, not substituted
for the shipped one-shot allocator.

---

## 9c. Hindsight-regret audit (`compute_hindsight_regret`, §F.5)

§9b compares two algorithms (MPC vs. open-loop) against each other — both
sides of that comparison are estimates. This section adds the comparison
that's actually missing: the tool against the one number in this entire
budget-allocation story that carries no estimation at all — what this
account's real historical spend and revenue actually were, for the
identical windows §9b already backtested.

Deliberately built as a thin wrapper around §9b's own output rather than a
parallel implementation: `compute_hindsight_regret` takes
`optimize_budget_allocation_mpc`'s returned report and reads its
`window_start`/`window_end` back out, so both analyses describe the exact
same historical periods by construction, not by coincidence. "Actual"
revenue and spend per window is a plain `groupby().sum()` over
`canonical_df` for those real dates — no curve, no fitting, no estimation.

**Honest asymmetry, stated plainly.** The tool's side of this comparison
(open-loop/MPC realized revenue, carried straight through from §9b) is
still an estimate — scored via that window's retrospective `eval_curves`,
because we cannot observe what revenue an allocation would truly have
produced had it actually been run instead of the real decision. The
"actual" side carries no such uncertainty. This asymmetry is inherent to
any counterfactual budget audit — the paper motivating this section
("Auditing Marketing Budget Allocation with Hindsight Regret," 2026) is
explicit that `planned != realized` for exactly this reason (bidding,
pacing, and delivery systems mean a planned allocation doesn't become
realized spend automatically) — not a shortcut specific to this
implementation. A positive number here should be read as "the tool's
curve-based estimate says it would have beaten reality," not as a
certainty.

**Result on this run:** open-loop would have beaten the account's actual
historical decisions by **+10.7%**, MPC by **+16.5%**, both computed on
the identical 3-window, 90-day backtest §9b already ran. Written to
`output/hindsight_regret_audit.json`, surfaced directly below §9b's own
expander in the Budget What-If tab. Tested in
`tests/test_hindsight_regret.py`: window boundaries match §9b's own report
exactly, "actual" revenue matches a plain groupby-sum independent of the
audit function's own arithmetic, and a constructed case where one channel
strictly dominates another confirms the uplift's sign comes out positive
as expected rather than by luck.

---

## 10. Business-plausibility clamps (`src/sanity_clamps.py`)

Per-(channel, campaign_type) ROAS envelope — `[p5, p95]` of that group's own
historical daily ROAS distribution, widened ×1.5 — not one global bound
(recall §2: `bing/Shopping` legitimately runs ~17–18x ROAS, `bing/Audience`
legitimately runs 0x; one global bound could not serve both correctly). A
concrete, live example this run actually caught: campaign `570837630`
(`bing/Audience`) — a group with **zero revenue on every one of 66 observed
days** — got a raw model forecast of 414.22 revenue against 126.44 assumed
spend (3.28x implied ROAS) for the next 30 days. The bound for that group is
`[0, 0]` (no historical variance to widen), so the violation is correctly
flagged and the **displayed** value is capped to 0 — while the raw model
number is preserved in the log/CSV for review, per the "never silently hide
a real model problem" rule. Across the full live run: 88 of 408
(campaign × horizon) forecasts were flagged this way — almost all in
groups with the same zero/near-zero-historical-ROAS profile.

---

## 11. LLM causal summary (`src/llm_insights.py`)

Five-stage pipeline: (1) a deterministic stats engine (this module, pure
pandas/numpy — anomaly z-scores, period-over-period deltas, feature
importance, saturation status) → (2) a grounding-context JSON, the *only*
facts the model may reference → (3) an Anthropic API call
(`claude-sonnet-5`, tool-use for structured JSON output) with a system
prompt that bans arithmetic beyond what's supplied, causal language beyond
"associated with"/"a contributing factor," and bare point forecasts without
their own uncertainty range → (4) a programmatic validator
(`validate_llm_json`) that extracts every number in the LLM's own output
text and rejects the response if any number doesn't trace back (within
tolerance) to a number actually present in the grounding context → (5) a
deterministic rule-based narrator (`rule_based_fallback`) that renders the
*exact same JSON shape* directly from the grounding context if no API key is
configured, the API call fails, or the LLM's output fails validation twice.

This resilience path is not just a fallback for this sandboxed environment —
`run.sh`'s own contract is that grading may run with no network access at
all, so the narrative layer has to be grounded and functional either way.
Every summary in `output/causal_summary.json` from the reference run in this
submission was produced by the rule-based path (`"source":
"rule_based_fallback"`) for exactly that reason.

**§H.4's exit criterion, closed.** `tests/test_llm_insights.py` proves
`validate_llm_json` actually rejects an injected fabricated number (not
just that the function exists) — both a large obviously-invented figure and
a small plausible-looking one, both a standalone unit test and through the
full `generate_causal_summary` orchestrator (with `call_llm` monkeypatched
to return the fabrication deterministically, confirming the retry-then-
fallback path actually engages). **§H.3's live call** is verified via a
mocked `anthropic` client that mimics the real SDK's response shape —
proving `call_llm`'s tool-use parsing and its degrade-to-`None`-on-error
path are both correct — since no `ANTHROPIC_API_KEY` is available in this
project's own development sandbox either (`"ANTHROPIC_API_KEY" not in
os.environ`, confirmed directly). This is not a substitute for a real
network-verified run against api.anthropic.com; whoever deploys this with a
real key gets that verification essentially for free the first time
`generate_causal_summary` runs, since the fallback path is exercised
identically either way.

---

## 12. Frontend (`src/app.py`, Streamlit)

Six views, `streamlit run src/app.py`, ordered per §I.2's demo workflow
(**ingest → forecast → simulate → insights**):
1. **Data health** — this run's live ingestion report (§3), not a cached
   training-time snapshot — matters directly if the grading data directory
   differs from what the model was trained on. Shown first, before any
   forecast number, so the audience sees what the model is standing on.
2. **Forecast** — a genuine time-series fan chart: historical daily actual
   revenue (raw + 7-day average) for the selected channel/type scope,
   overlaid with the selected horizon's forecast rendered as its own
   implied average daily rate — a flat median line with nested flat
   P25–P75/P10–P90/P5–P95 bands, on the same daily-revenue y-axis as the
   history (§I.1.2; see §14 for why the forward region is flat rather than
   a widening cone). The previous percentile-vs-probability curve is kept
   as a secondary "detail view" in an expander, plus the per-campaign
   table. An opt-in checkbox (OFF by default — the flat band is the
   honest default) reveals a **day-by-day seasonal decomposition**: the
   same aggregate quantiles, redistributed across individual days using
   this scope's own historical day-of-week revenue share, every quantile
   scaled by an identical per-day factor so the daily medians average back
   to exactly the flat-band rate. Explicitly labeled as a decomposition of
   one number, not a second, independently-fit daily model — genuine
   day-to-day uncertainty isn't modeled because the underlying forecast
   doesn't have any to give (§B.1).
3. **Budget what-if** — a live-rescored curve across a budget-multiplier
   grid. Monotonic by construction (§5's isotonic post-processing) *and*
   business-plausibility-clamped (§10) at the exact scenario point
   displayed — both checks run on a fixed grid with the live multiplier
   read off by interpolation, so the guarantee holds at any slider
   position, not only at the chart's own sample points (see
   `tests/test_app_budget_whatif.py`).
4. **Breakdown** — the reconciled hierarchy (§8) with the coherence check
   shown, not just assumed.
5. **AI summary** — the grounded narrative (§11), with its validation
   source (`llm` vs. `rule_based_fallback`) shown plainly, plus the full
   grounding context in an expander so the numbers are checkable.
6. **Model reliability** — the CV table from §5, the reliability diagram
   from §6, the ACI comparison (§6a), the Tweedie sweep, the hurdle-model
   ablation (§5b), and both SHAP-based (§5c) and gain-based feature
   importance.

---

## 13. Assumptions

- Meta's `conversion` field is treated as a revenue proxy based on its
  values being the only monetary-shaped signal available and its
  correlation with spend — this is stated as an assumption throughout the
  code and output (`revenue_confidence: "proxy"`), not asserted as
  documented fact about what Meta's export actually contains.
- "Non-observed day" is interpreted as zero spend/revenue/activity (a
  paused or non-served day), not as a missing measurement — a defensible
  default for ad-platform exports, but a real assumption.
- The holiday-window list (§4) is a short, hand-justified set of dates
  relevant to US/India-facing e-commerce advertising, not a generic global
  calendar.

## 14. Known limitations

- Fuzzy-matching thresholds (§3) are tuned against the fixtures/files seen
  here, not a formally validated decision boundary.
- The quantile ensemble's monotonic budget response is enforced
  post-hoc (isotonic regression) rather than natively, because LightGBM's
  quantile objective doesn't support `monotone_constraints` (§5) — a real,
  verified library limitation, not a skipped step. This now applies to the
  live Streamlit what-if slider's displayed metric as well as `predict.py`'s
  static output, not just the latter (§12, `tests/test_app_budget_whatif.py`).
- Interval (not just point) hierarchical reconciliation (§8) uses the
  library's real `Conformal` reconciler where this project's own OOF data
  actually supports its joint-alignment requirement (total + channel), and
  a marginal per-node split-conformal correction elsewhere (campaign_type,
  campaign) — not full joint conformal at every level, since that's
  mathematically inapplicable at the individual-campaign level given how
  campaign lifecycles don't overlap (measured: 0/308 dates have every
  campaign present simultaneously). A handful of the sparsest campaigns
  (<15 OOF observations) still use the original rescale shortcut. **§8c now
  measures the consequence directly, rather than leaving it as a plausible
  concern: the reconciled band under-covers its nominal target at the
  channel level (90%→81.4% empirical) and more sharply at the reconciled
  campaign level (90%→71.3%), while `total` and `campaign_type` stay
  conservative — a real, quantified gap between this project's two interval
  methods, not just a structural risk.**
- The Hill-curve recency-weighting half-life (§9, `recency_half_life_days
  =45.0`) is a reasonable default chosen by inspection (roughly "a month
  and a half of relevance decay" for daily ad-spend data), not tuned
  against a held-out objective — a shorter half-life would track regime
  shifts faster at the cost of noisier curves from fewer effectively-
  weighted points; a longer one approaches the original equal-weighted fit.
  The weighted R² gate (same section) is scored consistently with whichever
  half-life is chosen, so this is a sensitivity/responsiveness trade-off,
  not a correctness risk — but it is still a judgment call, like the other
  heuristic thresholds in this list.
- CQR's exact coverage guarantee technically applies to the train-only
  models, not the train+calibration-refit production models actually
  shipped (§6) — mitigated by the honest final-holdout check in §7, but
  worth stating plainly. §6a's ACI comparison shows this static correction
  visibly undercovering later in the holdout timeline; ACI itself is
  reported, not shipped as the default (see §6a for why).
- The hurdle-model ablation (§5b) and SHAP computation (§5c) both add real
  compute to every training run (an extra train-only Tweedie + hurdle fit
  for the ablation; a TreeExplainer pass for SHAP) — a deliberate trade of
  training time for an honestly-run comparison and real attributions,
  rather than a free lunch.
- Business-plausibility bounds (§10) for groups with very little history
  (`n_days` < 10) default to a wide, low-confidence bound rather than a
  tight one — documented in `sanity_clamps.compute_roas_bounds`'s
  `"insufficient_history_wide_bound"` note.
- The Hill-curve fit-quality gate's R² threshold (0.10) and its treatment
  of a boundary-pinned steepness parameter as automatic rejection (§9) are
  both reasonable heuristics tuned by inspecting this data, not a formally
  validated cutoff — a genuinely weak-but-real saturation curve near that
  threshold could in principle be rejected, or a spurious one just above it
  accepted, on a different dataset.
- The anomalous-segment detector's parameters (§5: `z_thresh=2.0`,
  30-day rolling window, 14-day minimum run, 90-day minimum history,
  60-day warm-up exclusion) are likewise chosen by inspection, not fit or
  validated against labeled ground truth on when a tracking break actually
  started or ended — there is no such ground truth available. The method is
  applied identically to every group rather than pointed at a known answer,
  which is the actual improvement over the hack it replaces, but the
  specific numbers are still a judgment call.
- The Forecast tab's fan chart (§12, §I.1.2) draws the forward region as a
  **flat** median line and flat nested bands across the whole horizon
  window, not a widening cone. This is a direct, honest consequence of the
  origin/aggregate-window model design (§B.1): the model predicts one
  total for the next 30/60/90 days, not a daily path, so there is no
  day-by-day uncertainty to widen — the flat band is the forecast's real
  implied daily rate, not a simplified stand-in for a cone the model could
  have produced but doesn't.
- `frequency_roll_28` (§4, added this round) sums Meta's *daily* reach
  figure over a 28-day window as the best available proxy for average
  exposure frequency — Meta's raw export doesn't provide a deduplicated
  28-day unique-reach count, so any user reached on more than one day in
  the window gets counted once per day, making this a slight
  over-estimate of true average frequency, not a formally deduplicated
  figure. `cpm_roll_28`/`cpa_roll_28`/`video_view_rate_roll_28` carry no
  equivalent caveat — they're direct ratios of already-tracked rolling
  sums.
- The §F.4 MPC backtest (§9b) scores both methods against a purely
  retrospective ground-truth curve fit on this project's own historical
  data, not against a live system reacting to the recommended spend in
  real time — the honest, available substitute for a live environment
  stated plainly in §9b itself, not a claim of a true field experiment.
  The backtest's own ground-truth curves can themselves fall back to a
  flat (now recency-weighted, per the fix in §9) historical-ROAS line on
  a short 30-day window that doesn't clear the real-fit gates — the same
  honestly-scoped fallback behavior as §9's own curves, inherited rather
  than specially handled.

## 15. References

- Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019).
  Optimal forecast reconciliation for hierarchical and grouped time series
  through trace minimization. *Journal of the American Statistical
  Association*, 114(526).
- Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized quantile
  regression. *NeurIPS 2019*.
- Gibbs, I., & Candès, E. (2021). Adaptive conformal inference under
  distribution shift. *NeurIPS 2021*. (§6a, `src/adaptive_conformal.py`)
- Principato, G., Stoltz, G., Amara-Ouali, Y., Goude, Y., Hamrouche, B., &
  Poggi, J-M. (2024). Conformal prediction for hierarchical data.
  arXiv:2411.13479. (§8's genuine joint-conformal tier, via
  `hierarchicalforecast.methods.Conformal`)
- Lundberg, S. M., & Lee, S-I. (2017). A unified approach to interpreting
  model predictions. *NeurIPS 2017*. (SHAP, §5c)
- Chernozhukov, V., Fernández-Val, I., & Galichon, A. (2010). Quantile and
  probability curves without crossing. *Econometrica*, 78(3).
- Tweedie, M. C. K. (1984). An index which distinguishes between some
  important exponential families. Used here via LightGBM's
  `objective="tweedie"` for zero-inflated, right-skewed revenue.
- Hill, A. V. (1910). The possible effects of the aggregation of the
  molecules of haemoglobin on its dissociation curves — the origin of the
  Hill equation used here (§9) as a generic saturating-response curve, a
  standard adaptation in marketing budget-response modeling.
- Ke, G. et al. (2017). LightGBM: A highly efficient gradient boosting
  decision tree. *NeurIPS 2017*.
- Pathak, N., Shyamal, S., Mhasker, P., & Swartz, C. (2026). Learning to
  spend: Model predictive control for budgeting under non-stationary
  returns. arXiv:2604.27186. (§9's recency-weighted Hill-curve motivation,
  and directly implemented — scoped as an honest historical backtest, not a
  live deployment — in §9b's `optimize_budget_allocation_mpc`.)
- Angelopoulos, A. N., Candès, E. J., & Tibshirani, R. J. (2023). Conformal
  PID control for time series prediction. *NeurIPS 2023*. (§6b, both
  `run_conformal_pid_control`'s derivative-of-error-signal proxy D-term and
  `run_conformal_pid_control_learned_scorecaster`'s genuine learned-AR
  D-term, `src/adaptive_conformal.py`)
- So, B., & Valdez, E. A. (2024). Tweedie multi-target regression for
  loss reserving. arXiv:2406.16206. (§5b's CatBoost-Tweedie point-model
  candidate, `modeling.train_catboost_point_model` — background on the
  general technique this codebase's own independent CatBoost candidate is
  built and evaluated on, not a reimplementation of their specific paper.)
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A.
  (2018). CatBoost: unbiased boosting with categorical features. *NeurIPS
  2018*. (§5b's second point-model family — ordered boosting and native
  categorical handling, the two mechanisms the ablation's diversity
  argument rests on.)
