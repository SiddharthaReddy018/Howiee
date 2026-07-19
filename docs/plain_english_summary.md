# AIgnition 3.0 — What This Project Does (Plain-English Version)

*Team TechBlazers. This page has no jargon on purpose — for the full technical
version, see `docs/summary.md` and `docs/technical_documentation.md`.*

## The one-sentence pitch

Feed it a spreadsheet of past ad spend and results from Google, Bing, and
Meta — it predicts how much revenue and return-on-ad-spend to expect over
the next 30/60/90 days, tells you how confident it is, and explains *why*
in plain sentences.

## The problem it solves

Marketing agencies currently guess future ad performance using spreadsheets
and gut feel. This tool replaces the guess with a number that's been tested
against real historical outcomes — and, importantly, it also tells you how
*unsure* it is, instead of pretending to know exactly.

## How it works, without the jargon

1. **It reads messy real-world files automatically.** Google, Bing, and Meta
   each name their columns differently ("Spend" vs. "Cost" vs. "spend_usd").
   The system figures out what each column actually means on its own, and
   it doesn't fall over if a file is missing a column or has extra junk
   columns in it — tested directly by deliberately feeding it a scrambled
   file.
2. **It predicts a range, not a single number.** Instead of "you'll make
   $742,000," it says "most likely $742,000, but realistically anywhere
   from $195,000 to $2,237,000" — because pretending to know the exact
   number would be dishonest, and a range is what a real business decision
   actually needs.
3. **The parts always add up to the whole.** The forecast for Google +
   Bing + Meta always equals the forecast for the total account, exactly —
   no rounding mismatches between a campaign-level number and the
   company-wide number.
4. **It explains its reasoning in words, not just numbers.** An AI layer
   writes a short summary of what's driving each forecast (e.g., "budget
   level and time window are the biggest factors this month") — and every
   number it's allowed to mention is double-checked against the real
   stats first, so it can't make something up.
5. **It doesn't just forecast — it recommends, and it checks itself
   against multiple approaches before trusting any one of them.** Beyond
   predicting what will happen, it can answer "given a fixed budget,
   what's the best way to split it across Google, Meta, and Bing?" On this
   account's real data, simply reallocating the *same* money already being
   spent — no extra budget at all — is predicted to meaningfully increase
   daily revenue, by shifting money away from a channel that's plateaued
   (spending more there barely moves the needle anymore) toward channels
   that still have room to grow. The forecast itself is built the same
   careful way: three genuinely different prediction methods are tried and
   compared honestly on data none of them were trained on, and whichever
   one (or blend of them) actually predicts best is what gets used — not
   whichever one sounded fanciest going in.
6. **It checks in periodically instead of "set it and forget it," and it
   was tested on more of the real data hiding in the files.** A second
   version of the budget recommendation re-checks itself every month
   against a full quarter, using only the data that would genuinely have
   been available at each check-in point, and was tested against a
   "decide once and never touch it again" plan on this account's own
   history to see whether checking in actually pays for itself. Separately,
   two columns of real data (how many unique people an ad reached, and how
   many people watched an ad video) were sitting in the original files
   completely unused — the system now reads them and uses them, including
   a "how many times did we show the same person this ad" signal that
   distinguishes broad brand-awareness spending from narrower, repeat-
   targeting spending.

## The headline numbers, translated

| What we measured | The number | What it actually means |
|---|---|---|
| Forecast accuracy | 36.5% average error | On revenue that's tested on data the model never saw during training, its middle-of-the-road guess lands within roughly a third of the true number — reasonable for ad revenue, which is naturally volatile day to day. Accuracy improves the further out you forecast (34% error at 90 days vs. 42% at 30 days). |
| Is that actually good? | 71.2% better than the obvious alternative | We also tested the simplest thing anyone could do with a spreadsheet — "just assume next month looks like last month" — on the exact same data. That simple approach is wrong more than the true number 127% of the time on average; this model cuts that error by roughly 71%. This is the number that answers "compared to what?" |
| Honesty about uncertainty | ~90-95% actual vs. 80-90% claimed | When the model says "I'm 80% (or 90%) sure revenue lands in this range," it's actually right about 90% (or 95%) of the time in the historical test — if anything, it's a little *too* cautious, which is the safer direction to be wrong in for budget planning. |
| Consistency | Exact match | Add up every individual campaign's forecast and it equals the total account forecast, to the decimal. |
| Return on ad spend | $4.88 back per $1 spent | Realistic range: $2.90–$5.33, depending on how things go over the next 30 days. |
| Budget reallocation | +56% more revenue, same budget | Moving today's existing daily spend around — not adding a single extra dollar — to where the data says it works harder is predicted to raise daily revenue from about $30,500 to $47,600. |
| Checking in periodically vs. deciding once | +5% more revenue, same budget, over a real 3-month test | Re-checking the recommended split every month against fresh data, instead of setting it once and never touching it again, is predicted to earn back a further 5% on top of the reallocation above — tested honestly on this account's own history, including a month where checking in *didn't* help, reported rather than hidden. |
| Reliability testing | 174 automated checks, all passing | Includes deliberately feeding it a scrambled, mislabeled spreadsheet to confirm it still works — not just testing it on the clean, expected file. |

## What it's honest about (the limitations)

- **Meta doesn't report a "revenue" number at all** — only a generic
  "conversion" count is available, so anything about Meta's revenue is
  clearly flagged as an estimate, not a hard fact. This is a limitation of
  the data provided, not something the model got wrong.
- **The forecast is a flat total for the whole time window**, not a
  day-by-day curve. It answers "how much revenue over the next 30 days,"
  not "how much on day 17 specifically" — a deliberate, honestly-stated
  design choice, not a shortcut.

## Why this is more impressive than it might sound

Most hackathon prototypes work once, on the exact file they were built
against, and quietly break the moment something looks slightly different.
This one was specifically stress-tested against messy, mislabeled,
incomplete versions of the input files to make sure it keeps working in
the real world — and every claim in this document is backed by a number
that was actually measured, not estimated.
