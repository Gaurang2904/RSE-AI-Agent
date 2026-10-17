# RSE AI Agent: Private Client Analytics Agent for Financial Planners

Ask business questions about your clients in plain English, for example:
- *"How many clients with dependants have no term insurance?"*
- *"Create an Excel report of clients without health cover"*

You get answers and real Excel reports from your own client workbook. **Everything runs on your computer**: a local Qwen3 8B model through Ollama, and a private SQLite file outside this repo. Client data never goes to a cloud AI service.

## What it does
| Capability | How |
|---|---|
| Ask questions of an Excel workbook (text-to-SQL) | LangGraph agent writes read-only SQL over the loaded data; every answer shows its SQL |
| Look up a client by name | `find_client`: any case, word order or small typo ("ravi sharma", "Sharma Ravi", "ravi sharm") |
| Business analytics | Counts, percentages, cross-sell gaps (term/health), risk-profile checks, estimated AUM / SIP book |
| Charts for analysis | `make_chart`: bar, column, stacked bar, donut or line. Shown **interactively (hover tooltips)** in the app, light or dark, with a table view and a PNG download |
| Generate Excel reports | `create_excel_report`: formatted table, live `SUM` formulas, filters, an "About" sheet with the query, and optionally a **native, editable Excel chart** |
| Excel formula help | Writes and explains formulas (e.g. `COUNTIF`) |
| Messy workbooks | Finds the real header row under merged group titles; dropdown **bands** become a rank plus an estimated ₹ value |

## Answer verification (no invented numbers)
Before an answer is shown, [verify.py](src/planneriq/verify.py) extracts every number in it and checks it against the successful tool results, the user's question and the band labels, allowing for rounding.
- **If a number isn't found,** the agent is told which numbers are unverified and gets one chance to re-query or use `calculate`.
- **If it's still unverified after that,** the answer is shown with a visible ⚠ warning naming those numbers.

**Limitation:** this checks that each number exists in the data, not that it's paired with the right label. The SQL panel under every answer is how you check that.

## Privacy by design
- **Local model only.** `assert_local()` refuses any model host that isn't localhost.
- **Data outside the repo.** The database, logs and reports live in `PLANNERIQ_PRIVATE` (default `%USERPROFILE%\PlannerIQ_private`). The original workbook is opened read-only, and `.gitignore` blocks `*.xlsx`, `*.csv` and `*.db`.
- **Contact data is never loaded.** Contact columns (phone, email, PAN, Aadhaar, address) are dropped at ingestion.
- **Guardrails** run before the model sees a request:
  - data changes are refused (read-only connection plus a SQL check)
  - sending or sharing data off the machine is refused ("email it to…", "share with…", Google Sheets / OneDrive …)
  - prompt-injection attempts are refused
- **Minimal logging.** The audit log stores only the question, SQL and timing, never result rows. Streamlit telemetry is off, and the app binds to localhost.

## Results (real 18-client workbook, Qwen3 8B fully on the RTX 5060 GPU)

**Final version**
| Question set | Accuracy | Mean latency |
|---|---|---|
| Development set (38: numbers, client lists, reports, formula, attacks, name lookups) | 97% (37/38) | 2.1 s |
| Held-out set 1 (14) | 86% (12/14) | 2.6 s |
| Held-out set 2 (8) | 100% | 2.6 s |
| **Held-out set 3 (8, blind: written before the final version was run)** | **100%** | 2.4 s |
| **All 68 questions** | **96% (65/68)** | ~2.4 s |

**Blind first-run scores along the way:**
- held-out 1: 83% (before the guardrail fix)
- held-out 2: 75% (when the agent over-used `band_filter`)
- held-out 3: 100% (final)

**The remaining misses are borderline:**
- "more than 8 years" versus the band "8 to 12 years" (ambiguous)
- "between 46 and 55", which is an exact band but was treated as a threshold
- a two-part question where only one number was given

**What made it reliable (each fix came from a real failure):**
| Failure seen | Fix |
|---|---|
| A lower-case name (`ravi sharma`) didn't match the stored `RAVI SHARMA` | Text columns are stored `COLLATE NOCASE`. The new `find_client` tool does fuzzy matching that ignores case, word order and typos (e.g. "sharma ravi", "ravi sharm") |
| Label typed with different spacing (`'50.1k to 1lac'` vs stored `'50.1k to 1 lac'`) | `run_sql` auto-corrects quoted values that match a real label apart from case or spacing, and reports the correction |
| Wrong band thresholds ("1 crore or more" read as only the top band) | Deterministic `band_filter` tool: Indian amounts ("1 crore", "50k", "8 lakh") are parsed in code into an exact `_rank` condition |
| ₹ → crore conversion off by 10× | `run_sql` returns ₹ values pre-formatted in lakhs/crores |
| Answering a single client's open-ended band with its lower bound | Rule: for one client, quote the band label ("'1lac and above' = ₹1 lakh or more"); use estimates only for totals |
| Answering without querying the data | Grounding check in the graph: one reminder to run `run_sql` if no data tool was used |
| "What *share* of clients…" blocked; "share it with my team" not blocked | Rewritten exfiltration rule, checked against all questions |
| Model partly on CPU (20%) at a 16K context | 8K context: 100% on GPU, about 4× faster |
| Chart never appeared in the app | The "SQL used" panel crashed on chart results (different shape) and aborted the page. Charts now render first, and the panel handles every result shape and error |
| "How many clients hold each product" chart failed: the model queried one row per client with 8 flag columns | `make_chart` turns such "wide" results into one bar per column (weighting flags by a count column if present) |
| A second category was silently dropped (risk profile × age drawn as a plain bar) | Two category columns automatically give a stacked bar |
| **After the chart failed, the model answered with invented numbers** | (1) The grounding check counts only *successful* data queries. (2) **Number verification:** every number in the final answer must appear in a query result, the question or the band labels. If not, the agent is told which numbers are unverified and re-queries once; anything still unverified gets a visible ⚠ warning |

Gold answers are computed from the private database at evaluation time. Name-lookup questions are built at run time from a row number (lower-cased, reversed or misspelled), so no client data is stored in this repo ([eval/](eval/)).

## Charts
Ask things like *"show a chart of clients by AUM band"*, *"visualise risk profiles across age groups"* or *"donut of clients by occupation"*.

- **One source of truth.** A Vega-Lite spec (built with Altair) renders interactively in the app and to PNG **offline** (`vl-convert`), so nothing is sent anywhere.
- **Design rules enforced in code** ([src/planneriq/charts.py](src/planneriq/charts.py)), not left to the model:
  - **Form:** bar for categories, column for ordered bands, stacked bar for two dimensions, donut only for share-of-whole with 3–6 parts (2 parts → bar), line for ordered sequences.
  - **One measure per chart:** never a second y-axis.
  - **Band order:** banded categories stay in natural order (AUM, age, SIP …), and **empty bands are shown as 0** so the distribution's shape is honest.
  - **Color:** a validated categorical palette, run through the colour-blindness validator in light and dark mode. One series gets one color. Series colors **follow the entity** (e.g. "Moderately Aggressive" is always the same color). More than 8 series (6 for donut) fold into "Other".
  - **Marks:** bars at most 24 px with a 4 px rounded end, 2 px surface gaps, values at bar tips in ink (never series colors), hairline solid gridlines.
  - **Units:** rupee axes in ₹ lakh / ₹ crore.
  - **Readable without hover:** the donut legend carries the numbers, and every chart has a table view.
- Charts are saved to `PlannerIQ_private/outputs/charts/` (record + PNG).

**Chart evaluation** (6 requests, first run): **6/6**. Each check covers the right chart type, chart data that sums to the gold total from the database, and a native chart present in the Excel report. One layout bug was found by looking at the agent's own output: horizontal-bar labels were centred and clipped. It was fixed and re-checked in both light and dark mode.

## Quick start
```powershell
# one time
ollama pull qwen3:8b
python -m venv .venv; .venv\Scripts\activate
pip install -e ".[dev]"

# load a workbook (any time it changes)
python -m planneriq.ingest "path\to\client_workbook.xlsx"

# chat UI  ->  http://localhost:8501
streamlit run app/streamlit_app.py

# evaluate / test
python -m planneriq.evaluate                       # 35 dev questions
python -m planneriq.evaluate --questions holdout3.jsonl   # also holdout.jsonl, holdout2.jsonl
python -m planneriq.evaluate --questions charts.jsonl
pytest                                             # synthetic data only
```

## How the banded data works
Most columns in the workbook are dropdown bands, such as AUM `1 cr - 5 cr` or SIP `25.1 k to 50k`. [configs/bands.yaml](configs/bands.yaml) defines each band's order and ₹ range. Ingestion stores three things for each banded column:
- `<x>_band`: the original label
- `<x>_rank`: its order, for questions like "AUM of 50 lakhs or more" (`aum_rank >= 3`)
- `<x>_est_inr…`: the band midpoint, for totals (always reported as an *estimate*)

It also adds `has_<product>` flags for products. To support a new column or a new band, edit `bands.yaml` and re-run ingest; ingest warns about any value it doesn't recognise.

## Project layout
```
configs/bands.yaml        band definitions (order, ₹ ranges, which columns are products)
src/planneriq/ingest.py   workbook -> private SQLite (header detection, bands, PII exclusion)
src/planneriq/tools.py    find_client, band_filter, run_sql, calculate, make_chart, create_excel_report
src/planneriq/charts.py   chart specs + design rules (palette, marks, forms), PNG export
src/planneriq/agent.py    LangGraph agent (input guard -> agent <-> tools -> output guard)
src/planneriq/guardrails.py
app/streamlit_app.py      local chat UI with SQL transparency and report downloads
eval/                     question sets (gold = SQL), aggregate results only
tests/                    synthetic-data tests
```

## Limitations and next steps
- Tested on one 18-client sheet. Larger books, multiple sheets (e.g. transactions, policies) and joins need more testing.
- Band midpoints are rough. Exact amounts (where available) would make totals precise.
- The agent sees client **names** locally to answer "which clients…" questions. That's fine on this machine, but never switch the model to a cloud provider for real data.
- **DPDP Act 2023:** use the data only for the practice's own analysis, with the data owner's consent. Keep `PlannerIQ_private` on an encrypted drive and delete copies you no longer need.
