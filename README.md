# RSE AI Agent

**A private, local AI business-analyst agent for a financial-planning practice.**

Ask questions in plain English and get grounded answers, charts and Excel reports. The agent works over three sources:
- your **client workbook** (text-to-SQL)
- a **knowledge base** of product definitions, insurance FAQs and practice policies (RAG with citations)
- a **CRM REST API** (authenticated, paginated, with retries)

Everything runs on your own computer: LangGraph and a local Qwen3 8B model through Ollama. Client data never goes to a cloud AI service.

> *"Why might clients with dependants need term insurance, and how many of ours lack it?"* → a cited explanation from the knowledge base, **plus** the exact count from the client data, plus an education-not-advice notice.
>
> *"Which clients have an open health-insurance follow-up in the CRM but no health cover?"* → calls the CRM API, then joins its results with the local client data.

## Capabilities

| Capability | How |
|---|---|
| **Text-to-SQL analytics** over an Excel workbook | Read-only SQL over a private SQLite file built from the workbook. Every answer shows its SQL |
| **RAG with citations** | Retrieve-before-answer over `knowledge/*.md`: `nomic-embed-text` embeddings (local Ollama) and a **FAISS** index. Answers cite `[Document § Section]` and list their sources |
| **REST API integration** | `crm_client.py`: bearer-token auth, timeouts, exponential-backoff retries, `Retry-After` handling for 429, pagination, **Pydantic** response validation, typed errors. `mock_crm.py` is a FastAPI CRM with synthetic data and simulated rate limits, outages, slow responses and schema drift |
| **Multi-step tool use** | One answer can combine CRM data, client data and knowledge passages, e.g. CRM follow-ups joined with local client records |
| **Charts** | Bar, column, stacked bar, donut and line, shown interactively (hover) in the app and saved as PNG, with design rules enforced in code |
| **Excel reports** | Formatted `.xlsx` with live `SUM` formulas, filters, a query "About" sheet, and an optional native Excel chart |
| **Client lookup** | Any case, word order or small typo ("ravi sharma", "Sharma Ravi", "ravi sharm") |
| **Guardrails** | Read-only data; blocks data changes, sending data elsewhere and prompt injection; local model only; masks personal data |
| **Answer verification** | Every number must appear in a tool result. Unverified numbers trigger a re-query, then a visible ⚠ warning |
| **Education vs advice** | Guidance answers always carry a notice that they are educational, not personalised financial advice |

## How a question flows

```
question ─► input guard ─► retrieve (RAG: knowledge passages if the question needs them)
                               │
                               ▼
                      agent (Qwen3 8B) ⇄ tools: run_sql · band_filter · find_client · calculate
                               │               search_knowledge · crm_products · crm_followups
                               │               make_chart · create_excel_report
                               ▼
          grounding check ─► number verification ─► output guard (PII mask, sources, advice notice)
```

## Project structure

```
rse-ai-agent/
├── src/rse_agent/
│   ├── config.py          Settings: private data folder (RSE_PRIVATE), model, knowledge folder, excluded columns
│   ├── ingest.py          Excel workbook → private SQLite: finds headers under merged titles, adds band
│   │                      rank / ₹ estimate / has_<product> columns, never loads contact columns
│   ├── tools.py           The agent's tools: run_sql, band_filter, find_client, calculate,
│   │                      search_knowledge (RAG), crm_products / crm_followups (REST), make_chart,
│   │                      create_excel_report
│   ├── agent.py           LangGraph flow: input guard → retrieve → agent ⇄ tools → grounding check →
│   │                      number verification → output guard; local audit log
│   ├── rag.py             Knowledge base: section chunking, Ollama embeddings, FAISS index, search
│   ├── crm_client.py      REST client: auth, timeouts, retries/backoff, 429 Retry-After, pagination,
│   │                      Pydantic validation, typed errors
│   ├── mock_crm.py        Mock CRM REST API (FastAPI, synthetic data, simulated faults)
│   ├── guardrails.py      Request screening, read-only SQL check, PII masking, local-model enforcement
│   ├── verify.py          Number verification of final answers
│   ├── charts.py          Chart specs and design rules (palette, marks, forms, light/dark), PNG export
│   └── evaluate.py        Evaluation runner; gold answers computed from the data at run time
├── knowledge/             Knowledge base (original educational articles + two synthetic practice policies)
├── configs/bands.yaml     Band definitions: order, ₹ ranges, which columns are products
├── app/streamlit_app.py   Local chat UI: SQL / sources / API panels, interactive charts, downloads;
│                          starts the mock CRM automatically
├── eval/*.jsonl           Evaluation questions (analytics, held-out sets, charts, RAG + CRM)
├── tests/                 Unit and integration tests (synthetic data only)
└── .streamlit/config.toml Telemetry off, bound to localhost
```

## Results

Measured on a real 18-client workbook with Qwen3 8B fully on an RTX 5060 Laptop GPU. Gold answers are computed from the data at evaluation time, and no client data is stored in this repo.

| Evaluation set | Questions | Accuracy |
|---|---|---|
| RAG + CRM (knowledge answers with citations, combined RAG + data, CRM counts / joins / catalogue, out-of-scope refusal) | 13 | **100%** |
| Analytics (numbers, client lists, reports, formula, attacks, name lookups) | 38 | **100%** |
| Charts (right type, data sums to gold, native chart in report) | 6 | **100%** |
| Held-out set written before the final version | 8 | 88% |
| **Total** | **65** | **98%** |

Retrieval: the top passage was the correct section for every knowledge question tested (cosine similarity 0.79–0.90).

**What changed along the way (each fix came from a measured failure):**
- **The model skipped retrieval and answered from memory.** Fix: *retrieve before answering*. Questions phrased as knowledge questions, or scoring ≥ 0.76 against the knowledge base, get passages first; pure data questions don't.
- **Inconsistent citations.** Fix: a deterministic *Sources* footer built from the retrieved passages.
- **Combined questions stopped after the explanation.** Fix: when a question asks about "our clients", the grounding check requires a real data query.
- **Wrong rank filters.** Fix: `run_sql` explains each rank condition in labels (`age_rank >= 3` → `['46-55', '56+']`) so the model can see a mismatch and correct it.
- **Earlier fixes** (from the analytics stage): case-insensitive and fuzzy name lookup, band-threshold helper, lakh/crore formatting, and number verification after a chart failure once led to invented figures.

## Quick start

```powershell
ollama pull qwen3:8b; ollama pull nomic-embed-text
python -m venv .venv; .venv\Scripts\activate
pip install -e ".[dev]"

python -m rse_agent.ingest "path\to\client_workbook.xlsx"   # build the private database
python -m rse_agent.rag build                               # build the knowledge index (auto-rebuilds on change)
streamlit run app/streamlit_app.py                          # http://localhost:8501 (starts the mock CRM too)

python -m rse_agent.mock_crm                                # or run the CRM API on its own (port 8765)
python -m rse_agent.evaluate --questions rag_crm.jsonl      # also questions / holdout* / charts
pytest
```

| Environment variable | Default | Purpose |
|---|---|---|
| `RSE_PRIVATE` | `%USERPROFILE%\RSE_AI_Agent_private` | Database, knowledge index, reports, charts, logs (outside the repo) |
| `RSE_MODEL` | `qwen3:8b` | Local chat model |
| `RSE_CRM_URL` / `RSE_CRM_TOKEN` | `http://127.0.0.1:8765` / `demo-token` | CRM endpoint and bearer token |
| `RSE_CRM_RATE_LIMIT`, `RSE_CRM_FAIL_EVERY`, `RSE_CRM_SLOW_EVERY`, `RSE_CRM_MALFORMED` | off | Mock CRM fault simulation |

## Privacy and responsible use
- **Local model only:** `assert_local()` refuses any model host that isn't localhost, and embeddings are also computed locally.
- **No client data in this repository:** the workbook, database, index, reports and logs live in `RSE_PRIVATE`. Contact columns are never loaded.
- **The CRM receives only filters and row references,** never names or personal data.
- **Guidance is educational.** The knowledge base and answers explain concepts and policies. Personalised investment advice in India is regulated (SEBI for investments, IRDAI for insurance), and recommendations for a specific client require a suitability assessment by a qualified, registered adviser.
- **Use client data responsibly under the DPDP Act 2023:** for the practice's own purposes, with consent, on an encrypted drive, deleting working copies when no longer needed.

## Limitations
- Tested on one 18-client sheet. Larger books and multi-sheet joins need more testing.
- Band midpoints are estimates.
- The knowledge base is small (9 articles) and written for education. It is not a regulatory reference.
- The retrieval routing thresholds were calibrated on the evaluation questions and should be re-checked as the knowledge base grows.
- Number verification checks that each number exists in the data, not that it is paired with the right label. The SQL / sources panels are the final check.
