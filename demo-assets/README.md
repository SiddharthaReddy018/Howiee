# Demo Assets

- `AIgnition_3.0_Interface_Walkthrough.pptx` — the Deliverable #4 slide deck.
  10 slides: title, system overview, one slide per app tab (screenshot +
  what/why), and a closing "built to survive scrutiny" summary.
- `screenshots/` — raw, unannotated screenshots of all 7 tabs, taken from a
  live run of the app against `./data`.
- `screenshots-annotated/` — the same screenshots with arrow-and-label
  callouts pointing at specific UI elements (filters, charts, buttons,
  the Plotly hover toolbar, etc.) — these are what's embedded in the deck.

To regenerate screenshots after further app.py changes: run the app locally
with `streamlit run src/app.py`, and re-screenshot each of the 7 tabs.
