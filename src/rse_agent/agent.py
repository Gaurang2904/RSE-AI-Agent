"""Local business-analyst agent (LangGraph + Ollama) over the private client database.

    from rse_agent.agent import ask
    ask("How many clients have no health cover?")
"""
from __future__ import annotations

import json
import re
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
- When knowledge passages are already in the conversation (search_knowledge results), answer the
  question directly from them in 2-5 sentences with inline citations. Never reply with commentary
  about your process (e.g. "this does not require a query"): give the actual answer.
- run_sql returns "filter_meaning" for every *_rank condition. Check it matches the question
  (e.g. "aged 56+" must mean age_group in ['56+']); if not, fix the SQL and run it again.
- search_knowledge for concepts, product explanations, FAQs and practice policy ("why", "what is",
  "should we", "what does our policy say"). Base the explanation on the passages and cite them
  inline as [Document § Section]. If the question is also about our clients, ALSO query the data
  with run_sql and keep the two parts clearly separate (guidance vs. facts about our clients).
- Explain considerations; never tell anyone to buy or sell a specific product or security, and
  never present guidance as personalised advice.
- crm_products / crm_followups call the CRM REST API (product catalogue, RM follow-up tasks).
  CRM client_ref = clients.sr_no: to combine CRM tasks with client data, take client_refs from
  crm_followups and run_sql ... WHERE sr_no IN (...). If a CRM call returns an error, report it
  plainly (e.g. CRM unavailable) and answer only what the local data supports.
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
         "client data, call run_sql now (use band_filter only for amount/age thresholds). If no data is "
         "needed, reply with your final answer to the user now (the answer itself, not commentary).")
DATA_TOOLS = {"run_sql", "create_excel_report", "make_chart", "search_knowledge", "crm_products",
              "crm_followups"}
# Retrieve-before-answer routing. Calibrated on the evaluation questions: every knowledge question
# scored >= 0.79 and every pure data question <= 0.73 against the knowledge base.
AUTO_RETRIEVE = 0.76          # inject passages whatever the phrasing
INTENT_RETRIEVE = 0.70        # inject passages when the question is phrased as a knowledge question
KNOWLEDGE_INTENT = re.compile(r"\b(why|explain|difference|define|definition|meaning|what (is|are|does)|"
                              r"should|policy|policies|according to|lock-?in|minimum investment|faq)\b", re.I)
DATA_INTENT = re.compile(r"\b(how many|count|number of|which clients|list|chart|plot|graph|total|average|"
                         r"our clients|of ours|clients (with|aged|who|in)|report|excel)\b", re.I)
DISCLAIMER = ("\n\nℹ️ *Educational information, not personalised financial advice. Recommendations for "
              "a specific client require a suitability assessment by a qualified, registered adviser.*")


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

    def retrieve(state: State):
        """Retrieve before answering: knowledge questions get the top passages as context."""
        from .rag import MIN_SCORE, search

        q = str(state["messages"][-1].content)
        know, data = bool(KNOWLEDGE_INTENT.search(q)), bool(DATA_INTENT.search(q))
        try:
            hits = search(q, k=4)
        except Exception:  # noqa: BLE001  (no embeddings available: the model can still call tools)
            return {}
        top = hits[0]["score"] if hits else 0.0
        if top >= AUTO_RETRIEVE or (know and top >= INTENT_RETRIEVE):
            payload = {"passages": [{"source": h["source"], "score": h["score"], "text": h["text"]}
                                    for h in hits if h["score"] >= MIN_SCORE], "cite_as": "[Document § Section]"}
        elif know and not data:
            payload = {"passages": [], "note": "No relevant passage in the knowledge base. If the question is "
                                               "about our clients, use the data tools instead. If it is a general "
                                               "knowledge question, say the knowledge base does not cover it and do "
                                               "not answer from general knowledge."}
        else:
            return {}
        call = {"name": "search_knowledge", "args": {"query": q}, "id": "auto_retrieve", "type": "tool_call"}
        return {"messages": [AIMessage(content="", tool_calls=[call]),
                             ToolMessage(json.dumps(payload, ensure_ascii=False), tool_call_id="auto_retrieve",
                                         name="search_knowledge")]}

    def agent(state: State):
        resp = llm_tools.invoke([SystemMessage(system), *state["messages"]])
        return {"messages": [resp], "steps": state.get("steps", 0) + 1}

    def finalize(state: State):
        return {"messages": [llm.invoke([SystemMessage(system), *state["messages"], HumanMessage(FINALIZE)])]}

    def question_start(msgs) -> int:
        return max(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage) and m.name != "nudge")

    def grounded(state: State) -> bool:
        """Did a data tool run SUCCESSFULLY since the user's latest question? A question that asks
        about our clients ("how many of ours ...") needs a client-data query, not just knowledge."""
        msgs = state["messages"]
        start = question_start(msgs)
        ok = {m.name for m in msgs[start:] if isinstance(m, ToolMessage) and successful(m.content)}
        if DATA_INTENT.search(str(msgs[start].content)):
            return bool(ok & (DATA_TOOLS - {"search_knowledge"}))
        return bool(ok & DATA_TOOLS)

    def check_numbers(state: State) -> list[float]:
        msgs = state["messages"]
        sources = [m.content for m in msgs if isinstance(m, ToolMessage) and successful(m.content)]
        sources += [msgs[question_start(msgs)].content, system]   # question + band labels
        return unverified(str(msgs[-1].content), sources)

    def output_guard(state: State):
        msgs = state["messages"]
        last = msgs[-1]
        # After a reminder, a reply without any tool call just confirms the earlier answer: keep that
        # earlier answer rather than the confirmation text.
        # (Only for the grounding reminder; after a number recheck the new answer is the corrected one.)
        confirmed_after_nudge = (
            len(msgs) >= 3 and isinstance(msgs[-2], HumanMessage) and msgs[-2].content == NUDGE
            and isinstance(msgs[-3], AIMessage) and bool(msgs[-3].content) and not msgs[-3].tool_calls)
        if confirmed_after_nudge:
            last = msgs[-3]
        text = mask_pii(str(last.content))
        bad = check_numbers(state)
        if bad:                       # still unverified after the recheck: say so visibly
            text += WARNING.format(nums=fmt(bad))
        msgs = state["messages"]
        sources = []
        for m in msgs[question_start(msgs):]:
            if isinstance(m, ToolMessage) and m.name == "search_knowledge" and successful(m.content):
                sources += [p_["source"] for p_ in json.loads(m.content).get("passages", [])]
        sources = list(dict.fromkeys(sources))[:4]
        if sources:                   # deterministic citations: the passages the answer was grounded on
            text += "\n\n**Sources (knowledge base):** " + "; ".join(sources)
            if "not personalised financial advice" not in text:
                text += DISCLAIMER    # guidance answers always carry the education-vs-advice notice
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
    g.add_node("retrieve", retrieve)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_node("finalize", finalize)
    g.add_node("nudge", nudge)
    g.add_node("recheck", recheck)
    g.add_node("output_guard", output_guard)
    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", lambda s: END if s.get("blocked") else "retrieve", ["retrieve", END])
    g.add_edge("retrieve", "agent")
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
