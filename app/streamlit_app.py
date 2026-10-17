"""RSE AI Agent - private, local client analytics.

    .venv\\Scripts\\streamlit run app/streamlit_app.py

Everything runs on this computer: the workbook is loaded into a private SQLite file
(PLANNERIQ_PRIVATE) and questions are answered by a local Ollama model.
"""
import json
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

from planneriq.charts import build_spec
from planneriq.config import DB_PATH, MODEL, PRIVATE
from planneriq.ingest import ingest

st.set_page_config(page_title="RSE AI Agent", page_icon="📊", layout="wide")

def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # older Streamlit: default to light
        return "light"


def show_chart(path: str, key: str) -> None:
    """Interactive chart (hover tooltips) in the app's theme + table view + PNG download."""
    p = Path(path)
    if not p.exists():
        return
    rec = json.loads(p.read_text(encoding="utf-8"))
    st.vega_lite_chart(build_spec(rec, theme_mode()), theme=None, width="stretch")
    with st.expander("Table view"):
        st.dataframe(pd.DataFrame(rec["data"]), width="stretch", hide_index=True)
    png = p.with_suffix(".png")
    if png.exists():
        st.download_button(f"⬇ {png.name}", png.read_bytes(), file_name=png.name, key=f"png-{key}")


def show_tool_call(c: dict) -> None:
    """'SQL used' panel. Tool results differ in shape (run_sql: rows+columns, make_chart: table,
    errors: error), so this never assumes a field exists and never breaks the page."""
    try:
        res = json.loads(c["result"])
    except (json.JSONDecodeError, TypeError):
        res = {}
    label = "⚠ failed" if isinstance(res, dict) and "error" in res else "SQL used"
    with st.expander(f"🔎 {c['name']}: {label}"):
        st.code(c["args"].get("sql", ""), language="sql")
        if not isinstance(res, dict):
            return
        if "error" in res:
            st.caption(res["error"])
        elif isinstance(res.get("rows"), list) and res.get("columns"):
            st.dataframe(pd.DataFrame(res["rows"], columns=res["columns"]), width="stretch", hide_index=True)
        elif isinstance(res.get("table"), list) and res["table"]:
            st.dataframe(pd.DataFrame(res["table"]), width="stretch", hide_index=True)


EXAMPLES = [
    "Give me a cross-sell summary: how many clients are missing term and health insurance?",
    "Which clients aged 56+ have a moderately aggressive or aggressive risk profile?",
    "Show a chart of how many clients hold each product.",
    "Create an Excel report of clients without health cover with age group, AUM band and risk profile.",
]

with st.sidebar:
    st.header("Data")
    st.caption(f"🔒 Local only · model `{MODEL}` via Ollama · data folder `{PRIVATE}`")
    up = st.file_uploader("Load a client workbook (.xlsx)", type=["xlsx"])
    if up is not None and st.button("Load workbook"):
        dest = PRIVATE / "uploads" / Path(up.name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(up.getvalue())
        info = ingest(dest)
        st.success(f"Loaded {info['clients']} clients ({info['columns']} columns)")
        for w in info["warnings"]:
            st.warning(w)
        st.session_state.pop("graph_ready", None)
    if DB_PATH.exists():
        con = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)
        n = con.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
        holders = pd.read_sql(
            "SELECT SUM(has_term_cover) term, SUM(has_health_cover) health, SUM(has_sip) sip, "
            "SUM(has_swp) swp, SUM(has_pms) pms, SUM(has_las) las, SUM(has_li_premium) li, "
            "SUM(has_ulip) ulip FROM clients", con).T.rename(columns={0: "clients holding"})
        con.close()
        st.metric("Clients loaded", n)
        st.caption("Product holdings")
        st.dataframe(holders, width="stretch")
    if st.button("New conversation"):
        st.session_state.pop("history", None)
        st.session_state.pop("messages", None)

st.title("📊 RSE AI Agent")
st.caption("Ask business questions about your clients in plain English. Every number comes from a SQL query you can inspect.")

if not DB_PATH.exists():
    st.info("Load a client workbook from the sidebar to begin.")
    st.stop()

st.session_state.setdefault("history", [])
st.session_state.setdefault("messages", [])

cols = st.columns(len(EXAMPLES))
clicked = next((ex for c, ex in zip(cols, EXAMPLES) if c.button(ex, width="stretch")), None)

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        for i, path in enumerate(turn.get("charts", [])):
            show_chart(path, key=f"h{id(turn)}-{i}")
        for path in turn.get("reports", []):
            if Path(path).exists():
                st.download_button(f"⬇ {Path(path).name}", Path(path).read_bytes(), file_name=Path(path).name,
                                   key=f"{path}-{id(turn)}")

prompt = st.chat_input("e.g. How many clients with dependants have no term insurance?") or clicked
if prompt:
    from planneriq.agent import ask

    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"), st.spinner("Analysing locally…"):
        r = ask(prompt, history=st.session_state.messages[-12:])
        st.session_state.messages = r["messages"]
        st.markdown(r["answer"])
        for i, path in enumerate(r.get("charts", [])):      # charts first: they are the point
            show_chart(path, key=f"new-{i}")
        for c in r["tool_calls"]:
            if c["name"] in ("run_sql", "create_excel_report", "make_chart"):
                show_tool_call(c)
        for path in r["reports"]:
            st.download_button(f"⬇ Download {Path(path).name}", Path(path).read_bytes(), file_name=Path(path).name)
        st.caption(f"{r['steps']} steps · {r['latency_s']:.1f}s · local model")
    st.session_state.history.append({"role": "assistant", "content": r["answer"], "reports": r["reports"],
                                     "charts": r.get("charts", [])})
