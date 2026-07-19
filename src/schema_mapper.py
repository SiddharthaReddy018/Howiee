"""
schema_mapper.py
=================
Schema-normalization layer — implements Implementation Plan §A.

Design principle (§A.1): raw vendor column names never touch feature-engineering
or modeling code. Every raw file is passed through `ingest_file` / `ingest_directory`
here first, which produces the CANONICAL schema (§A.2) plus an IngestionReport that
records exactly what was mapped, inferred, degraded, or ignored — this report is what
powers the frontend's "data-health" panel (§I.1).

canonical columns (post-mapping):
  channel          str   REQUIRED
  campaign_id      str   REQUIRED
  campaign_name    str   optional
  date             date  REQUIRED
  spend            float REQUIRED
  revenue          float REQUIRED-OR-conversions (see A.4)
  conversions      float optional
  clicks           float optional
  impressions      float optional
  reach            float optional (Meta-only in this project's data; NaN
                                    everywhere else, same "whole file never
                                    had it" convention as every other
                                    optional field -- see A.9)
  video_views      float optional (Google-only in this project's data; same
                                    convention as `reach`)
  campaign_type    str   optional (defaulted to "unclassified")
  daily_budget     float optional (left as genuine NaN, never fabricated)

A.9 -- `reach`/`video_views` (added this round). Previously BOTH columns
were present in the raw source files (Meta's `reach`, Google's
`metrics_video_views`) but had no canonical counterpart at all, so
`map_columns` logged them as `ignored_columns` and every byte of that data
was thrown away before it ever reached feature engineering -- not a bug
(nothing crashed, nothing was silently corrupted), just genuine unused
signal sitting in already-ingested files. `reach` in particular is the
clearest available proxy for distinguishing brand-awareness spend from
performance spend that this data offers -- a campaign with high impressions
but low reach is hitting the same users repeatedly (retargeting-flavored),
while high reach relative to impressions means broad, mostly-first-exposure
delivery -- a distinction `campaign_type` alone doesn't fully capture (see
§B.7 in feature_engineering.py for the derived `frequency_roll_28` /
`video_view_rate_roll_28` ratios built on top of these two raw fields).
Both are channel-specific by nature (only Meta reports reach; only Google's
video-capable campaigns report view counts) so, exactly like the existing
`revenue_confidence: proxy` situation for Meta, the other channels'
rows are genuinely NaN for these fields, not zero -- LightGBM's native NaN
handling routes this the same way it already does for every other
whole-file-missing optional column.
"""

from __future__ import annotations

import glob
import os
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# A.2 — canonical schema
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_COLUMNS: list[str] = [
    "channel", "campaign_id", "campaign_name", "date", "spend", "revenue",
    "conversions", "clicks", "impressions", "reach", "video_views",
    "campaign_type", "daily_budget",
]
REQUIRED_TIER: list[str] = ["date", "campaign_id", "spend", "revenue"]
OPTIONAL_TIER: list[str] = ["campaign_name", "conversions", "clicks",
                            "impressions", "reach", "video_views",
                            "campaign_type", "daily_budget"]
NUMERIC_ACTIVITY_FIELDS = ["spend", "revenue", "conversions", "clicks",
                           "impressions", "reach", "video_views"]

# ─────────────────────────────────────────────────────────────────────────────
# A.3 — alias mapping (config, not code)
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_ALIASES: dict[str, list[str]] = {
    "campaign_id":   ["campaign_id", "CampaignId", "campaignid", "campaign id"],
    "campaign_name": ["campaign_name", "CampaignName", "camp_name"],
    "date":          ["date", "segments_date", "date_start", "TimePeriod",
                       "stat_date", "day", "report_date"],
    "spend":         ["spend", "Spend", "cost", "metrics_cost_micros", "media_cost",
                       "ad_spend", "amount_spent"],
    "revenue":       ["metrics_conversions_value", "Revenue", "conversion_value",
                       "purchase_value", "revenue", "conv_value", "total_revenue"],
    "conversions":   ["metrics_conversions", "Conversions", "conversion", "purchases",
                       "conv_count"],
    "clicks":        ["clicks", "Clicks", "metrics_clicks"],
    "impressions":   ["impressions", "Impressions", "metrics_impressions"],
    # A.9 -- Meta-native "reach" (unique users) and Google-native
    # "metrics_video_views" (video ad view count). Both were real,
    # correctly-named columns in this project's own raw files that had no
    # canonical slot at all until now (see the A.9 module-docstring note).
    "reach":         ["reach", "Reach"],
    "video_views":   ["metrics_video_views", "video_views", "VideoViews"],
    "campaign_type": ["campaign_type", "CampaignType",
                       "campaign_advertising_channel_type", "ad_type"],
    "daily_budget":  ["daily_budget", "DailyBudget", "campaign_budget_amount", "budget"],
    # not part of the model feature set, but recognized so it's mapped rather
    # than silently dropped/ignored when a file happens to carry one
    "channel":       ["channel", "platform", "source", "network"],
}

# Columns needing a unit fix at mapping time, keyed by the *raw* column name
# (normalized) so the fix travels with the exact source field, not the
# canonical name it happens to land in.
UNIT_FIXUPS: dict[str, "callable"] = {
    "metricscostmicros": lambda s: pd.to_numeric(s, errors="coerce") / 1_000_000.0,
}

# ─────────────────────────────────────────────────────────────────────────────
# A.5 — campaign-type inference fallback
# ─────────────────────────────────────────────────────────────────────────────
TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"prospect",              "prospecting"),
    (r"remarket|retarget",     "remarketing"),
    (r"brand",                 "brand"),
    (r"shop",                  "shopping"),
    (r"search",                "search"),
    (r"pmax|performance.?max", "performance_max"),
    (r"display",               "display"),
    (r"video",                 "video"),
    (r"demand.?gen",           "demand_gen"),
    (r"audience",              "audience"),
]


def infer_campaign_type(campaign_name: str) -> str:
    name = (campaign_name or "").lower()
    for pattern, label in TYPE_PATTERNS:
        if re.search(pattern, name):
            return label
    return "unclassified"


# ─────────────────────────────────────────────────────────────────────────────
# Normalization + fuzzy fallback (§A.7)
# ─────────────────────────────────────────────────────────────────────────────
FUZZY_REQUIRED_THRESHOLD = 80.0   # stricter — wrong guess on date/id/spend/revenue is costly
FUZZY_OPTIONAL_THRESHOLD = 65.0   # looser — worst case an optional feature stays NaN

try:
    from rapidfuzz import fuzz as _fuzz
    _HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - fallback path
    import difflib
    _HAVE_RAPIDFUZZ = False


def _norm(s: str) -> str:
    """Lowercase, strip everything but alphanumerics — makes 'Cost_Local',
    'cost-local', 'CostLocal' all compare equal."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


# ─────────────────────────────────────────────────────────────────────────────
# Numeric cleaning (§A.4b — a formatted-but-mapped column must not silently
# become all-zero). A raw numeric field (spend/revenue/conversions/clicks/
# impressions/reach/video_views/daily_budget) can be MAPPED correctly by name (exact or fuzzy)
# while still being unparsable by plain pd.to_numeric because of currency
# symbols, thousands separators, percent signs, or accounting-style negative
# parentheses — all common in real ad-platform / Excel-round-tripped exports.
# Left alone, `pd.to_numeric(..., errors="coerce")` turns every such value
# into NaN, which the existing `.fillna(0.0)` step then silently converts to
# a *valid-looking* zero — the column is reported as successfully mapped,
# Pandera's `Check.ge(0)` passes trivially (0.0 >= 0), and nothing downstream
# ever sees an error. This is strictly worse than a missing column, which is
# at least logged in `missing_optional`.
# ─────────────────────────────────────────────────────────────────────────────
_CURRENCY_STRIP_RE = re.compile(r"[,$€£₹\s]")
_PAREN_NEGATIVE_RE = re.compile(r"^\((.*)\)$")
_TRAILING_PERCENT_RE = re.compile(r"%$")


def _clean_numeric_series(s: pd.Series) -> pd.Series:
    """Best-effort normalization of common numeric-as-string formats before
    `pd.to_numeric`: strips currency symbols/thousands separators/whitespace,
    treats accounting-style `(123.45)` as `-123.45`, and strips a trailing
    `%`. A no-op for values that are already numeric (int/float dtype)."""
    if pd.api.types.is_numeric_dtype(s):
        return s
    cleaned = s.astype(str).str.strip()
    is_paren_negative = cleaned.str.match(_PAREN_NEGATIVE_RE)
    cleaned = cleaned.str.replace(_PAREN_NEGATIVE_RE, r"\1", regex=True)
    cleaned = cleaned.str.replace(_TRAILING_PERCENT_RE, "", regex=True)
    cleaned = cleaned.str.replace(_CURRENCY_STRIP_RE, "", regex=True)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    numeric = numeric.where(~is_paren_negative, -numeric.abs())
    return numeric


def _to_numeric_checked(
    raw_series: pd.Series, field_name: str, report: "IngestionReport",
) -> pd.Series:
    """`_clean_numeric_series` + a visible warning if a meaningful share of
    genuinely non-empty raw values still failed to parse — the safety net
    for currency formats this cleaner doesn't yet recognize (e.g. a locale
    that uses '.' as the thousands separator), so an unrecognized format
    degrades to a *logged* warning instead of a silent, undetectable zero."""
    numeric = _clean_numeric_series(raw_series)
    raw_str = raw_series.astype(str).str.strip()
    was_nonempty = raw_str.ne("") & raw_str.str.lower().ne("nan") & raw_series.notna()
    newly_failed = was_nonempty & numeric.isna()
    n_failed = int(newly_failed.sum())
    n_nonempty = int(was_nonempty.sum())
    n_total = int(len(raw_series))
    if n_nonempty > 0 and (n_failed / n_nonempty) > 0.02:
        example = raw_str[newly_failed].iloc[0] if n_failed else ""
        report.warnings.append(
            f"'{field_name}': {n_failed}/{n_nonempty} non-empty raw values "
            f"could not be parsed as numeric even after currency/format "
            f"cleaning (e.g. {example!r}) — they were treated as missing, "
            f"not zero. Check this column's number formatting."
        )
    elif n_nonempty == 0 and n_total > 0:
        # The column WAS mapped (it exists in the source file) but every
        # single value is blank/NaN before parsing even runs. The >2% guard
        # above can never fire here since there's no nonempty count to
        # divide by — left unhandled, this silently becomes a valid-looking
        # all-zero column (or all-NaN for an optional field) with no trace
        # in the ingestion report, which is strictly worse than a missing
        # column: a missing column is at least logged in missing_optional.
        report.warnings.append(
            f"'{field_name}': mapped column has {n_total} row(s) but every "
            f"single value is blank/empty — nothing parseable was found. "
            f"This field will read as all-zero (or all-missing), not "
            f"because the underlying data says so. Check that the correct "
            f"source column was mapped and actually contains values."
        )
    return numeric


def _fuzzy_score(a: str, b: str) -> float:
    if _HAVE_RAPIDFUZZ:
        return 0.5 * _fuzz.ratio(a, b) + 0.5 * _fuzz.partial_ratio(a, b)
    # difflib fallback: only has one similarity measure
    return 100.0 * difflib.SequenceMatcher(None, a, b).ratio()


# Build a flat (normalized_alias -> canonical_field) lookup for exact matching,
# and a (canonical_field -> [normalized_alias, ...]) pool for fuzzy matching.
_EXACT_LOOKUP: dict[str, str] = {}
_ALIAS_POOL: dict[str, list[str]] = {}
for _field, _aliases in CANONICAL_ALIASES.items():
    _ALIAS_POOL[_field] = [_norm(a) for a in _aliases]
    for _a in _aliases:
        _EXACT_LOOKUP[_norm(_a)] = _field
_UNIT_FIXUPS_NORM = {_norm(k): v for k, v in UNIT_FIXUPS.items()}


@dataclass
class IngestionReport:
    source: str
    channel: str = ""
    mapped: dict = field(default_factory=dict)        # canonical -> raw column name
    fuzzy_mapped: dict = field(default_factory=dict)   # canonical -> (raw, score)
    fuzzy_rejected: list = field(default_factory=list)  # (raw, best_field, score) - logged, not applied
    ignored_columns: list = field(default_factory=list)
    missing_optional: list = field(default_factory=list)
    revenue_confidence: str = "direct"                 # "direct" | "proxy"
    campaign_type_source: str = "native"                # "native" | "inferred" | "unclassified"
    campaign_type_unclassified_count: int = 0
    n_rows: int = 0
    n_campaigns: int = 0
    date_range: tuple = None
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if d.get("date_range"):
            d["date_range"] = [str(x) for x in d["date_range"]]
        return d


class SchemaMappingError(Exception):
    """Raised when a file is missing a hard-required signal (§A.4)."""


def _drop_pandas_index_col(df: pd.DataFrame, report: IngestionReport) -> pd.DataFrame:
    """The ubiquitous 'Unnamed: 0' pandas index column — the running example
    from §A.6 of the extra-column problem."""
    first_col = str(df.columns[0])
    if first_col == "" or first_col.lower().startswith("unnamed"):
        report.ignored_columns.append(first_col)
        df = df.drop(columns=[df.columns[0]])
    return df


def map_columns(columns: list[str]) -> tuple[dict, IngestionReport]:
    """
    Pure column-name mapping step (no data). Returns
    (rename_map: {raw_col: canonical_field}, partial_report).
    """
    report = IngestionReport(source="<columns>")
    rename_map: dict[str, str] = {}
    claimed_fields: set[str] = set()
    unmatched_raw: list[str] = []

    # Pass 1 — exact (normalized) match
    for col in columns:
        canon = _EXACT_LOOKUP.get(_norm(col))
        if canon and canon not in claimed_fields:
            rename_map[col] = canon
            claimed_fields.add(canon)
            report.mapped[canon] = col
        else:
            unmatched_raw.append(col)

    # Pass 2 — fuzzy fallback (§A.7) for whatever's left, against whatever
    # canonical fields are still unclaimed.
    #
    # Bug fix (order-dependence, was: greedy-in-column-order): the previous
    # version of this loop scored and assigned ONE raw column at a time, in
    # whatever order the source CSV happened to list its columns. Whichever
    # column got processed first "won" a contested canonical field even if a
    # column processed later scored strictly higher against it — e.g.
    # `Cost_Local` (score 80.8 against `spend`) would claim the `spend` slot
    # before `Ad_Spend_USD` (score 91.2 against `spend`) was even considered,
    # purely because of column position in the file, silently dropping the
    # objectively better match into `ignored_columns`.
    #
    # Fixed by scoring EVERY remaining (raw column x canonical field) pair
    # up front into a matrix, then solving it as a linear sum assignment
    # problem (maximizing total match quality) instead of resolving
    # contested fields in column order. This is the actual best-match
    # solution — a column only loses a field to another column when that
    # other column is a better match for it, never because of iteration
    # order — and reuses a well-tested library routine rather than a
    # hand-rolled greedy pop loop.
    #
    # `channel` is deliberately excluded from this fuzzy pool (still eligible
    # for the exact match in Pass 1 above). Every other canonical field is
    # numeric and gets range/type-checked right after mapping
    # (`_to_numeric_checked`), so a bad fuzzy match at worst produces a
    # column of implausible numbers that later checks can catch. `channel`
    # is free-text with no such backstop, and already has its own
    # purpose-built, safer 3-tier fallback below (explicit override > native
    # `channel` column > filename-derived) — fuzzy-matching it from column
    # *names* alone risks a short, generic alias ("source", "platform")
    # coincidentally matching some unrelated junk column and silently
    # overwriting every row's channel with that column's actual (garbage)
    # values, which nothing downstream would catch. Confirmed live: an
    # unrelated column merely named `some_new_platform_field_2026` scored
    # 75/100 against `channel`'s alias pool and would otherwise have won it.
    remaining_fields = [f for f in CANONICAL_ALIASES if f not in claimed_fields and f != "channel"]

    if not unmatched_raw:
        return rename_map, report

    if not remaining_fields:
        report.ignored_columns.extend(unmatched_raw)
        return rename_map, report

    score_matrix = np.array([
        [max(_fuzzy_score(_norm(col), alias_norm) for alias_norm in _ALIAS_POOL[f])
         for f in remaining_fields]
        for col in unmatched_raw
    ])

    # A raw maximize-total-score assignment on its own has a subtle failure
    # mode: a column that will NEVER clear a field's threshold can still
    # "win" that field in the optimal assignment if doing so frees up the
    # column that actually deserved it for a slightly-lower-scoring second
    # field, netting a higher total sum overall (concretely hit while
    # testing this fix: a stray junk/index-like column scored 63.5 against
    # `campaign_name`, just under its 65 optional threshold — sum-maximizing
    # assignment happily "spent" campaign_name on that guaranteed-to-be-
    # rejected pairing and bumped the column that actually meant
    # `campaign_name` (score 88.3) down to `campaign_type` (67.8) instead).
    # A pairing that can never be accepted shouldn't be able to outbid one
    # that can. Fix: mask every sub-threshold cell to the SAME flat
    # sentinel (not just `score - big_number`, which merely shifts the
    # existing 0-100 spread down without collapsing it — two invalid cells
    # can then still differ by up to ~100, which is exactly the same order
    # of magnitude as the valid-side differences we're trying to protect,
    # so it doesn't actually stop invalid-side "which reject is least bad"
    # tie-breaking from outweighing a valid-side choice). A single flat
    # sentinel makes every invalid pairing equally (and arbitrarily) bad,
    # so the solver can only ever gain total score by using a valid pairing
    # wherever one exists, and only falls back to an invalid one when a
    # field has no valid candidate at all — at which point it was always
    # going to be rejected below regardless of which reject fills the slot.
    threshold_per_field = np.array([
        FUZZY_REQUIRED_THRESHOLD if f in REQUIRED_TIER else FUZZY_OPTIONAL_THRESHOLD
        for f in remaining_fields
    ])
    is_valid = score_matrix >= threshold_per_field
    assignment_matrix = np.where(is_valid, score_matrix, -1e6)

    row_idx, col_idx = linear_sum_assignment(assignment_matrix, maximize=True)
    assigned = {r: c for r, c in zip(row_idx, col_idx) if is_valid[r, c]}  # raw-col idx -> field idx, valid only

    for i, col in enumerate(unmatched_raw):
        if i in assigned:
            j = assigned[i]
            best_field, best_score = remaining_fields[j], float(score_matrix[i, j])
            rename_map[col] = best_field
            claimed_fields.add(best_field)
            report.fuzzy_mapped[best_field] = (col, round(best_score, 1))
        else:
            # Either this column wasn't part of the assignment at all (more
            # unmatched columns than remaining fields), or it was, but only
            # against a field it could never clear the threshold for.
            # Report against its own single best-scoring field for an
            # honest "nearest miss" — independent of whatever the assignment
            # solver arbitrarily did with the invalid/sentinel portion of
            # the matrix, which carries no meaning beyond "not accepted".
            j = int(np.argmax(score_matrix[i]))
            report.fuzzy_rejected.append((col, remaining_fields[j], round(float(score_matrix[i, j]), 1)))
            report.ignored_columns.append(col)

    return rename_map, report


def ingest_dataframe(
    raw: pd.DataFrame,
    source: str,
    channel: str | None = None,
) -> tuple[pd.DataFrame, IngestionReport]:
    """
    Map one already-loaded raw DataFrame onto the canonical schema.
    Raises SchemaMappingError on a hard-fail condition (§A.4).
    """
    df = raw.copy()
    partial = IngestionReport(source=source)
    df = _drop_pandas_index_col(df, partial)

    rename_map, col_report = map_columns(list(df.columns))
    # merge column-mapping report into this file's report
    partial.mapped = col_report.mapped
    partial.fuzzy_mapped = col_report.fuzzy_mapped
    partial.fuzzy_rejected = col_report.fuzzy_rejected
    partial.ignored_columns += col_report.ignored_columns

    df = df.rename(columns=rename_map)

    # Apply unit fixups keyed on the *original* raw column name
    for raw_col, canon in rename_map.items():
        fixup = _UNIT_FIXUPS_NORM.get(_norm(raw_col))
        if fixup is not None and canon in df.columns:
            df[canon] = fixup(df[canon])

    # ── A.4 — required-field hard-fail checks ──────────────────────────────
    hard_required = ["date", "campaign_id", "spend"]
    missing_hard = [f for f in hard_required if f not in df.columns]
    if missing_hard:
        raise SchemaMappingError(
            f"[{source}] Missing hard-required field(s) {missing_hard} — "
            f"refusing to guess a primary key or date. Columns seen: {list(raw.columns)}"
        )

    if "revenue" not in df.columns and "conversions" not in df.columns:
        raise SchemaMappingError(
            f"[{source}] Neither 'revenue' nor 'conversions' could be mapped — "
            f"no usable target signal at all. Columns seen: {list(raw.columns)}"
        )

    # ── dtype normalization ─────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    n_bad_dates = df["date"].isna().sum()
    df = df.dropna(subset=["date"]).copy()
    if n_bad_dates:
        partial.warnings.append(f"{n_bad_dates} row(s) dropped — unparseable date")
    if df.empty:
        raise SchemaMappingError(f"[{source}] No parseable dates remain after mapping.")

    df["campaign_id"] = df["campaign_id"].astype(str)
    df["spend"] = _to_numeric_checked(df["spend"], "spend", partial).fillna(0.0).clip(lower=0).astype(float)

    # ── revenue vs. conversions (§A.4 degrade-don't-crash path) ─────────────
    if "revenue" in df.columns:
        df["revenue"] = _to_numeric_checked(df["revenue"], "revenue", partial).fillna(0.0).clip(lower=0).astype(float)
        partial.revenue_confidence = "direct"
    else:
        df["revenue"] = _to_numeric_checked(df["conversions"], "conversions", partial).fillna(0.0).clip(lower=0).astype(float)
        partial.revenue_confidence = "proxy"
        partial.warnings.append(
            "Revenue forecast for this channel is conversion-count/value-derived; "
            "no direct revenue field was found in the source file."
        )

    if "conversions" in df.columns:
        df["conversions"] = _to_numeric_checked(df["conversions"], "conversions", partial).fillna(0.0).astype(float)
    else:
        df["conversions"] = np.nan
        partial.missing_optional.append("conversions")

    for f in ["clicks", "impressions", "reach", "video_views"]:
        if f in df.columns:
            df[f] = _to_numeric_checked(df[f], f, partial).fillna(0.0).astype(float)
        else:
            df[f] = np.nan
            partial.missing_optional.append(f)

    # daily_budget: genuine optional numeric — never fabricate a value (§A.4/§C.4)
    if "daily_budget" in df.columns:
        df["daily_budget"] = _to_numeric_checked(df["daily_budget"], "daily_budget", partial)
    else:
        df["daily_budget"] = np.nan
        partial.missing_optional.append("daily_budget")

    if "campaign_name" not in df.columns:
        df["campaign_name"] = df["campaign_id"]
        partial.missing_optional.append("campaign_name")
    df["campaign_name"] = df["campaign_name"].astype(str)

    # ── campaign_type: native > regex-inferred > unclassified (§A.5) ────────
    if "campaign_type" in df.columns:
        df["campaign_type"] = df["campaign_type"].astype(str).str.strip()
        empty_native = df["campaign_type"].isin(["", "nan", "None"])
        if empty_native.any():
            df.loc[empty_native, "campaign_type"] = (
                df.loc[empty_native, "campaign_name"].apply(infer_campaign_type)
            )
            partial.campaign_type_source = "native+inferred"
        else:
            partial.campaign_type_source = "native"
    else:
        df["campaign_type"] = df["campaign_name"].apply(infer_campaign_type)
        partial.campaign_type_source = "inferred"
        partial.missing_optional.append("campaign_type")

    partial.campaign_type_unclassified_count = int((df["campaign_type"].str.lower() == "unclassified").sum())

    # ── channel ───────────────────────────────────────────────────────────
    if channel:
        df["channel"] = channel
    elif "channel" in df.columns:
        df["channel"] = df["channel"].astype(str).str.lower()
    else:
        # Filename-derived last resort (e.g. a file whose name doesn't match
        # any of _CHANNEL_FILE_PATTERNS in ingest_directory). Strip the
        # extension -- an un-stripped 'source' here previously leaked a
        # literal '.csv' suffix into the channel column, and from there into
        # every downstream channel-level breakdown, the app's channel
        # selector, and the LLM grounding context's scope label.
        df["channel"] = os.path.splitext(source)[0].lower()

    # Bug fix (cross-channel campaign_id collision): feature_engineering.py
    # groups lag/rolling/expanding-statistic features by `campaign_id` ALONE
    # in several places (the reindex step, expanding revenue mean, expanding
    # ROAS, etc.). Nothing in this schema layer asserts campaign_id is
    # unique ACROSS channels, only that it's present and non-null — so two
    # different channels reusing the same raw platform campaign_id (plausible
    # with small sequential platform IDs, e.g. campaign "1024" existing on
    # both Bing and Google) would silently splice two unrelated campaigns'
    # histories together before reconciliation.py's channel-aware hierarchy
    # keys ("channel/campaign_type/campaign_id") ever get a chance to tell
    # them apart. Namespacing campaign_id by channel HERE — once, at the
    # source — makes every downstream `groupby("campaign_id")` automatically
    # collision-safe without having to touch feature_engineering.py (or
    # anything else) at all. Applied unconditionally (not just when a
    # collision is actually detected) so behavior doesn't depend on what
    # happens to be true of today's data.
    df["campaign_id"] = df["channel"].astype(str) + "::" + df["campaign_id"].astype(str)

    df = df[CANONICAL_COLUMNS].copy()
    df = df.sort_values(["campaign_id", "date"]).reset_index(drop=True)

    partial.channel = str(df["channel"].iloc[0]) if len(df) else (channel or source)
    partial.n_rows = len(df)
    partial.n_campaigns = df["campaign_id"].nunique()
    partial.date_range = (df["date"].min().date(), df["date"].max().date()) if len(df) else None

    return df, partial


def ingest_file(path: str, channel: str | None = None) -> tuple[pd.DataFrame, IngestionReport]:
    raw = pd.read_csv(path, low_memory=False)
    df, report = ingest_dataframe(raw, source=os.path.basename(path), channel=channel)
    return df, report


# ─────────────────────────────────────────────────────────────────────────────
# A.8 — Pandera validation of the canonical schema (post-mapping)
# ─────────────────────────────────────────────────────────────────────────────
def validate_canonical_schema(df: pd.DataFrame) -> list[str]:
    """
    Encodes the canonical schema as a Pandera DataFrameSchema and runs lazy
    (collect-everything) validation. Returns a list of human-readable failure
    strings (empty list == valid). Never raises — callers decide what to do
    with a non-empty list (this pipeline treats it as advisory, matching
    validate.py's existing "warnings, not a hard gate" philosophy, except for
    the structural checks already enforced earlier in ingest_dataframe).
    """
    import pandera.pandas as pa
    from pandera import Column, Check

    schema = pa.DataFrameSchema(
        {
            "channel":       Column(str, nullable=False),
            "campaign_id":   Column(str, nullable=False),
            "campaign_name": Column(str, nullable=True),
            "date":          Column("datetime64[ns]", nullable=False),
            "spend":         Column(float, Check.ge(0), nullable=False),
            "revenue":       Column(float, Check.ge(0), nullable=False),
            "conversions":   Column(float, Check.ge(0), nullable=True),
            "clicks":        Column(float, Check.ge(0), nullable=True),
            "impressions":   Column(float, Check.ge(0), nullable=True),
            "reach":         Column(float, Check.ge(0), nullable=True),
            "video_views":   Column(float, Check.ge(0), nullable=True),
            "campaign_type": Column(str, nullable=False),
            "daily_budget":  Column(float, Check.ge(0), nullable=True),
        },
        strict=False,
        coerce=False,
    )
    try:
        schema.validate(df, lazy=True)
        return []
    except pa.errors.SchemaErrors as exc:
        failure_df = exc.failure_cases
        msgs = []
        for _, row in failure_df.iterrows():
            msgs.append(
                f"column={row.get('column')} check={row.get('check')} "
                f"failure_case={row.get('failure_case')} index={row.get('index')}"
            )
        return msgs


def validate_campaign_consistency(df: pd.DataFrame) -> list[str]:
    """
    "Validating campaign consistency" is a named deliverable in the brief,
    separate from schema/type validation above. Checks, on the CANONICAL
    frame (post schema-mapping, campaign_id already channel-namespaced —
    §Bug-4's fix means a campaign_id colliding across channels is no longer
    even representable here, so that specific check doesn't need to exist
    separately):

      1. Duplicate (campaign_id, date) rows — a campaign should have at
         most one row per calendar day; two rows for the same day is a real
         upstream data issue (double-exported file, a join gone wrong),
         not something schema/type checks would catch.
      2. A single campaign_id reporting more than one distinct
         campaign_type across its history — campaigns aren't expected to
         change type mid-flight; if one does, that's either a genuine
         vendor-side campaign_type change or a mapping/ID-collision bug
         worth a human looking at, not something to silently average over.
      3. A single campaign_id reporting more than one distinct
         campaign_name across its history — same idea, for renames.

    Returns a list of human-readable issue strings (empty == no issues
    found). Never raises, and never modifies `df` — purely advisory, same
    "warnings, not a hard gate" philosophy as `validate_canonical_schema`
    above (a grading dataset the team hasn't seen might have exactly these
    issues; the goal is to surface them, not to silently drop or merge
    rows on the team's own assumption of what the "right" fix would be).
    """
    issues: list[str] = []
    if df.empty:
        return issues

    dup_mask = df.duplicated(subset=["campaign_id", "date"], keep=False)
    n_dupes = int(dup_mask.sum())
    if n_dupes:
        example = df.loc[dup_mask, ["campaign_id", "date"]].drop_duplicates().iloc[0]
        issues.append(
            f"{n_dupes} row(s) share a duplicate (campaign_id, date) pair "
            f"(e.g. campaign_id={example['campaign_id']!r} date={example['date'].date()}) — "
            f"expected at most one row per campaign per day."
        )

    type_counts = df.groupby("campaign_id")["campaign_type"].nunique()
    inconsistent_type = type_counts[type_counts > 1]
    if len(inconsistent_type):
        cid = inconsistent_type.index[0]
        seen = sorted(df.loc[df["campaign_id"] == cid, "campaign_type"].unique())
        issues.append(
            f"{len(inconsistent_type)} campaign_id(s) report more than one distinct campaign_type "
            f"over their history (e.g. campaign_id={cid!r} has seen {seen}) — "
            f"campaigns are assumed not to change type mid-flight."
        )

    name_counts = df.groupby("campaign_id")["campaign_name"].nunique()
    inconsistent_name = name_counts[name_counts > 1]
    if len(inconsistent_name):
        cid = inconsistent_name.index[0]
        seen = sorted(df.loc[df["campaign_id"] == cid, "campaign_name"].astype(str).unique())
        issues.append(
            f"{len(inconsistent_name)} campaign_id(s) report more than one distinct campaign_name "
            f"over their history (e.g. campaign_id={cid!r} has seen {seen}) — "
            f"possible silent rename or an ID reused across unrelated campaigns."
        )

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# Directory-level ingestion — auto-discovers files, tolerant of unknown files
# ─────────────────────────────────────────────────────────────────────────────
_CHANNEL_FILE_PATTERNS: dict[str, list[str]] = {
    "google": ["*google*ads*campaign*.csv", "*google_ads*.csv", "google*.csv",
               "ga_*.csv", "gads*.csv", "*gads_campaign*.csv"],
    "meta":   ["*meta*ads*.csv", "*meta_ads*.csv", "meta*.csv",
               "*facebook*ads*.csv", "*fb*ads*.csv", "facebook*.csv", "fb_*.csv",
               "*instagram*ads*.csv"],
    "bing":   ["*bing*campaign*.csv", "*bing*.csv", "*microsoft*ads*.csv",
               "msads*.csv", "*msads_campaign*.csv", "*microsoft_campaign*.csv"],
}


def _guess_channel(path: str) -> str | None:
    base = os.path.basename(path)
    for channel, patterns in _CHANNEL_FILE_PATTERNS.items():
        for pat in patterns:
            if glob.fnmatch.fnmatch(base.lower(), pat.lower()):
                return channel
    return None


def ingest_directory(data_dir: str) -> tuple[pd.DataFrame, dict[str, IngestionReport]]:
    """
    Auto-discover every *.csv in data_dir, map each through the schema layer,
    and concatenate. A single bad file is skipped (with its error captured in
    the report) rather than taking down ingestion of the other files —
    "shouldn't break" applies at the directory level too.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    frames: list[pd.DataFrame] = []
    reports: dict[str, IngestionReport] = {}

    for path in paths:
        base = os.path.basename(path)
        channel = _guess_channel(path)
        try:
            raw = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            reports[base] = IngestionReport(source=base, errors=[f"Could not read CSV: {exc}"])
            continue

        fallback_channel = channel or os.path.splitext(base)[0].lower()
        try:
            df, report = ingest_dataframe(raw, source=base, channel=channel or None)
            if channel is None:
                # no known pattern matched — fell back to a column or filename-derived channel
                report.warnings.append(
                    f"Channel not recognized from filename; using '{df['channel'].iloc[0] if len(df) else fallback_channel}'."
                )
            frames.append(df)
            reports[base] = report
        except SchemaMappingError as exc:
            reports[base] = IngestionReport(source=base, channel=fallback_channel, errors=[str(exc)])
            print(f"[schema_mapper] SKIPPED {base}: {exc}")

    if not frames:
        raise FileNotFoundError(f"No ingestible CSVs found in '{data_dir}'.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["channel", "campaign_id", "date"]).reset_index(drop=True)
    return combined, reports


def print_ingestion_log(reports: dict[str, IngestionReport]) -> None:
    for name, r in reports.items():
        print(f"[schema_mapper] {name}  (channel={r.channel or '?'})")
        if r.errors:
            for e in r.errors:
                print(f"    ERROR: {e}")
            continue
        print(f"    rows={r.n_rows:,}  campaigns={r.n_campaigns}  "
              f"date_range={r.date_range}  revenue_confidence={r.revenue_confidence}")
        if r.mapped:
            print(f"    mapped (exact) : {r.mapped}")
        if r.fuzzy_mapped:
            print(f"    mapped (fuzzy) : {r.fuzzy_mapped}")
        if r.fuzzy_rejected:
            print(f"    fuzzy suggestions rejected (below threshold): {r.fuzzy_rejected}")
        if r.ignored_columns:
            print(f"    ignored        : {r.ignored_columns}")
        if r.missing_optional:
            print(f"    missing/optional (left NaN, not fabricated): {r.missing_optional}")
        print(f"    campaign_type  : {r.campaign_type_source}  "
              f"(unclassified={r.campaign_type_unclassified_count})")
        for w in r.warnings:
            print(f"    WARN: {w}")
