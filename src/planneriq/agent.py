"""Local business-analyst agent (LangGraph + Ollama) over the private client database.

    from planneriq.agent import ask
    ask("How many clients have no health cover?")
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .config import LOGS, MODEL, OLLAMA_HOST
from .guardrails import assert_local, check_request, mask_pii
from .tools import TOOLS, data_dictionary, run_sql
from .verify import fmt, successful, unverified

MAX_STEPS = 6

SYSTEM = """You are RSE AI Agent, a careful business analyst for a financial-planning practice.
You answer questions about the practice's clients by querying the `clients` table with run_sql.

How the data works:
- Most fields are BANDS from dropdowns (e.g. aum_band '1 cr - 5 cr'). Filter with exact band labels
  from the data dictionary, or compare with the *_rank columns (0 = lowest band).
- For a numeric threshold on a banded AMOUNT or AGE column ("1 crore or more", "above 50k",
  "below 25,000", "aged 56+"), call band_filter, then use the condition it returns in run_sql.
  Do NOT use band_filter for exact categories (a specific band such as age_group = '46-55',
  risk_profile, gender, occupation, area, source, dependants, goals_mapped, has_*): filter those
  directly in run_sql.
- Every number in your answer must come from a run_sql result in this conversation.
- For ONE client's amount, quote the stored band label exactly and then explain it in words, e.g.
  "'1lac and above' (₹1 lakh or more per month)". Never present the *_est_* number for one client;
  use *_est_* only for totals/averages across clients.
- Keep money columns in rupees and name them with "_inr" (e.g. SUM(aum_est_inr) AS total_aum_inr);
  run_sql then returns them pre-formatted in lakhs/crores under "inr_formatted". Quote those.
- *_est_* columns are ESTIMATES from band midpoints. Use them only for totals/averages and always
  say the figure is an estimate based on bands. Show money in lakhs or crores (1 cr = 100 lakhs).
- has_<product> = 1 if the client holds that product (term_cover, health_cover, sip, swp, pms, las,
  li_premium, ulip). '0' in a band column means not held.
- risk_profile is the client's risk appetite (Conservative ... Aggressive).

Rules:
- Always get numbers from run_sql (prefer one query that computes the answer, e.g. COUNT, AVG,
  GROUP BY). Use calculate for percentages and conversions. Never invent numbers.
- When the user names a specific client (any spelling or case), call find_client first, then
  query with the exact name it returns. Never conclude a client does not exist from a failed
  exact-match query.
- For "details"/"profile" of a client, select only the original columns (name, tenure_band,
  age_group, gender, marital_status, occupation, income_group, dependants, area, source,
  risk_profile, aum_band, goals_mapped and the *_band product columns), not *_rank/*_est_*/has_*,
  and present a short profile grouped as Profile and Products held (skip products shown as 0).
- When listing clients, give their names. Do not output contact details.
- If a query returns no rows or a warning, fix the filter (check exact labels) and retry.
- create_excel_report only when the user asks for a report, sheet, export or Excel file.
- make_chart when the user asks for a chart, graph, plot or visual (or says "show me" a
  distribution/split). The SQL must return one category column + ONE numeric column (+ one series
  column for stacked_bar), e.g. SELECT aum_band, COUNT(*) AS clients FROM clients GROUP BY aum_band.
  Pick: bar = compare categories; column = distribution over an ordered band (age_group, aum_band,
  sip_band); stacked_bar = two dimensions (e.g. aum_band by risk_profile); donut = share of a whole
  with <= 6 parts; line = an ordered sequence. Never put two different measures in one chart; make
  two charts instead. Give the chart a title that states the takeaway, then summarise the key
  numbers in words. For one bar per product use: SELECT SUM(has_term_cover) AS term_cover,
  SUM(has_health_cover) AS health_cover, SUM(has_sip) AS sip, ... FROM clients with x='product',
  y='clients'. For a split by two dimensions pass the second one as series.
- For Excel formula questions, give the formula and explain it briefly.
- Answer concisely: the key number(s) first, then 1-3 sentences of insight (e.g. cross-sell gaps,
  risk-profile mismatches). Mention the client count behind percentages.

DATA DICTIONARY
{dictionary}
"""

FINALIZE = ("Step budget reached. Answer now using only the results already retrieved. If no query "
            "succeeded, say plainly that you could not retrieve the data and suggest rephrasing. "
            "Never estimate or invent numbers.")
RECHECK = ("Check: these numbers in your answer do not appear in any query result: {nums}. Re-run the "
           "query (or use calculate) to get the exact figures, or remove them. Never invent numbers.")
WARNING = ("\n\n⚠ *Some figures in this answer ({nums}) could not be verified against the data. "
           "Check the SQL below before relying on them.*")
NUDGE = ("Check: you have not run any data query for this question yet. If the answer depends on the "
         "client data, call run_sql now (use band_filter only for amount/age thresholds). If it is a "
         "general question (e.g. an Excel formula), answer it directly.")
DATA_TOOLS = {"run_sql", "create_excel_report", "make_chart"}


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    blocked: str | None
    nudged: bool
    rechecked: bool


def chat_model(model: str = MODEL, seed: int = 42):
    from langchain_ollama import ChatOllama

    assert_local(OLLAMA_HOST)                      # client data never leaves this machine
    # 8K context keeps qwen3:8b fully on an 8 GB GPU (16K spilled ~20% to CPU and was ~4x slower)
    return ChatOllama(model=model, base_url=OLLAMA_HOST, temperature=0, seed=seed, num_ctx=8192,
                      keep_alive="15m", reasoning=False if model.startswith("qwen3") else None)


def build_graph(model: str = MODEL):
    llm = chat_model(model)
    llm_tools = llm.bind_tools(TOOLS)
    system = SYSTEM.format(dictionary=data_dictionary())

    def input_guard(state: State):
        refusal = check_request(state["messages"][-1].content)
        if refusal:
            return {"blocked": "guardrail", "messages": [AIMessage(refusal)]}
        return {"blocked": None, "steps": 0, "nudged": False, "rechecked": False}

    def agent(state: State):
        resp = llm_tools.invoke([SystemMessage(system), *state["messages"]])
        return {"messages": [resp], "steps": state.get("steps", 0) + 1}

    def finalize(state: State):
        return {"messages": [llm.invoke([SystemMessage(system), *state["messages"], HumanMessage(FINALIZE)])]}

    def question_start(msgs) -> int:
        return max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage) and m.name != "nudge")

    def grounded(state: State) -> bool:
        """Did a data tool run SUCCESSFULLY since the user's latest question?"""
        msgs = state["messages"]
        return any(isinstance(m, ToolMessage) and m.name in DATA_TOOLS and successful(m.content)
                   for m in msgs[question_start(msgs):])

    def check_numbers(state: State) -> list[float]:
        msgs = state["messages"]
        sources = [m.content for m in msgs if isinstance(m, ToolMessage) and successful(m.content)]
        sources += [msgs[question_start(msgs)].content, system]   # question + band labels
        return unverified(str(msgs[-1].content), sources)

    def output_guard(state: State):
        last = state["messages"][-1]
        text = mask_pii(str(last.content))
        bad = check_numbers(state)
        if bad:                       # still unverified after the recheck: say so visibly
            text += WARNING.format(nums=fmt(bad))
        return {"messages": [AIMessage(text, id=last.id, usage_metadata=getattr(last, "usage_metadata", None))]}

    def route(state: State):
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools" if state["steps"] < MAX_STEPS else "finalize"
        if not state.get("nudged") and not grounded(state) and state["steps"] < MAX_STEPS:
            return "nudge"            # answer without data: remind once to verify with run_sql
        if not state.get("rechecked") and state["steps"] < MAX_STEPS and check_numbers(state):
            return "recheck"          # numbers not found in any result: one chance to fix
        return "output_guard"

    def recheck(state: State):
        return {"messages": [HumanMessage(RECHECK.format(nums=fmt(check_numbers(state))), name="nudge")],
                "rechecked": True}

    def nudge(state: State):
        return {"messages": [HumanMessage(NUDGE, name="nudge")], "nudged": True}

    g = StateGraph(State)
    g.add_node("input_guard", input_guard)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("finalize", finalize)
    g.add_node("nudge", nudge)
    g.add_node("recheck", recheck)
    g.add_node("output_guard", output_guard)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", lambda s: END if s.get("blocked") else "agent", ["agent", END])
    g.add_conditional_edges("agent", route, ["tools", "finalize", "nudge", "recheck", "output_guard"])
    g.add_edge("nudge", "agent")
    g.add_edge("recheck", "agent")
    g.add_edge("tools", "agent")
    g.add_edge("finalize", "output_guard")
    g.add_edge("output_guard", END)
    return g.compile()


_GRAPHS: dict = {}


def ask(question: str, history: list | None = None, model: str = MODEL) -> dict:
    if model not in _GRAPHS:
        _GRAPHS[model] = build_graph(model)
    t0 = time.perf_counter()
    state = _GRAPHS[model].invoke(
        {"messages": [*(history or []), HumanMessage(question)], "steps": 0, "blocked": None, "nudged": False, "rechecked": False},
        config={"recursion_limit": 4 * MAX_STEPS + 6})
    latency = time.perf_counter() - t0
    msgs = state["messages"][len(history or []):]
    results = {m.tool_call_id: m.content for m in msgs if isinstance(m, ToolMessage)}
    calls, reports, charts = [], [], []
    for m in msgs:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                res = results.get(tc["id"], "")
                calls.append({"name": tc["name"], "args": tc["args"], "result": res})
                try:
                    if tc["name"] == "create_excel_report":
                        reports.append(json.loads(res)["saved_to"])
                    elif tc["name"] == "make_chart":
                        charts.append(json.loads(res)["chart"])
                except (KeyError, json.JSONDecodeError):
                    pass
    _audit(question, calls, latency, state.get("blocked"))
    return {"answer": msgs[-1].content, "tool_calls": calls, "reports": reports, "charts": charts,
            "steps": state.get("steps", 0), "blocked": state.get("blocked"),
            "latency_s": latency, "messages": state["messages"]}


def _audit(question: str, calls: list, latency: float, blocked) -> None:
    """Local audit log: question, SQL and timing only (no result rows)."""
    LOGS.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "question": mask_pii(question),
             "sql": [c["args"].get("sql") for c in calls if "sql" in c["args"]],
             "tools": [c["name"] for c in calls], "latency_s": round(latency, 2), "blocked": blocked}
    with open(LOGS / "audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


__all__ = ["ask", "build_graph", "run_sql"]
