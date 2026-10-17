"""Analyst tools: read-only SQL, safe calculator, Excel report generator."""
from __future__ import annotations

import ast
import json
import math
import operator
import re
import sqlite3
from datetime import datetime

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .config import DB_PATH, OUTPUTS
from .guardrails import GuardrailError, check_sql, mask_pii


def _connect():
    return sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True, timeout=5)


def fmt_inr(v: float) -> str:
    """Indian-style money: ₹12.35 cr / ₹25.00 lakh / ₹37,550."""
    if v is None:
        return "n/a"
    a = abs(v)
    if a >= 1e7:
        return f"₹{v / 1e7:,.2f} cr"
    if a >= 1e5:
        return f"₹{v / 1e5:,.2f} lakh"
    return f"₹{v:,.0f}"


def _range_text(low, high, unit: str) -> str:
    if not unit or not unit.startswith("inr") or low is None:
        return ""
    if high is None:
        return f" ({fmt_inr(low)} and above)"
    if high == 0:
        return " (not held)"
    return f" ({fmt_inr(low)} to {fmt_inr(high)})"


def data_dictionary() -> str:
    """Column-by-column description with band labels, built from the database itself."""
    con = _connect()
    cols = pd.read_sql("PRAGMA table_info(clients)", con).name.tolist()
    meta = pd.read_sql("SELECT * FROM columns WHERE column IS NOT NULL", con).set_index("column")
    bands = pd.read_sql("SELECT column, label, rank, low, high, unit FROM bands ORDER BY column, rank", con)
    lines = ["Table `clients` (one row per client). Columns:"]
    for c in cols:
        if c in meta.index:
            m = meta.loc[c]
            desc = m.description or f"original header '{m.source_header}'"
            line = f"- {c}: {desc}"
            if c in set(bands.column):
                labs = bands[bands.column == c]
                line += " | band labels by rank: " + ", ".join(
                    f"{b['rank']}='{b['label']}'{_range_text(b['low'], None if pd.isna(b['high']) else b['high'], b['unit'])}"
                    for _, b in labs.iterrows())
            else:
                vals = [v[0] for v in con.execute(f"SELECT DISTINCT {c} FROM clients LIMIT 12")]
                if c != "name" and len(vals) <= 10:
                    line += f" | values: {vals}"
            lines.append(line)
        elif c.endswith("_rank"):
            lines.append(f"- {c}: integer rank of the band (0 = lowest band; use for comparisons)")
        elif "_est_" in c:
            unit = c.split("_est_")[1]
            lines.append(f"- {c}: ESTIMATED value from the band midpoint ({unit}); open-ended top bands use the lower bound")
        elif c.startswith("has_"):
            lines.append(f"- {c}: 1 if the client holds this product, else 0")
    con.close()
    return "\n".join(lines)


def _loose(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def known_labels() -> dict[str, str]:
    """loose form -> exact stored value, for band labels and short categorical values."""
    con = _connect()
    try:
        out = {_loose(l): l for (l,) in con.execute("SELECT DISTINCT label FROM bands")}
        cols = [r[1] for r in con.execute("PRAGMA table_info(clients)") if r[2].upper().startswith("TEXT")]
        for c in cols:
            if c == "name":
                continue
            for (v,) in con.execute(f"SELECT DISTINCT {c} FROM clients WHERE {c} IS NOT NULL"):
                out.setdefault(_loose(v), v)
    finally:
        con.close()
    return out


def ground_literals(sql: str) -> tuple[str, list[str]]:
    """Replace quoted values that match a stored label apart from case/spacing with the exact label."""
    labels, fixes = known_labels(), []

    def fix(m):
        lit = m.group(1)
        exact = labels.get(_loose(lit))
        if exact is not None and exact != lit:
            fixes.append(f"'{lit}' -> '{exact}'")
            return "'" + exact.replace("'", "''") + "'"
        return m.group(0)

    return re.sub(r"'((?:[^']|'')*)'", fix, sql), fixes


# ---------------------------------------------------------------- band_filter
_UNITS = {"cr": 1e7, "crore": 1e7, "crores": 1e7, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5,
          "l": 1e5, "k": 1e3, "thousand": 1e3}


def parse_amount(text: str) -> float:
    """'1 crore' -> 1e7, '50,000' -> 5e4, '25k' -> 25000, '8.5 lakh' -> 850000, '56' -> 56."""
    t = text.lower().replace(",", "").replace("₹", "").replace("rs.", "").replace("rs", "").strip()
    m = re.match(r"^([\d.]+)\s*([a-z]*)", t)
    if not m:
        raise ValueError(f"cannot read amount '{text}'")
    num, unit = float(m.group(1)), m.group(2)
    return num * _UNITS.get(unit, 1.0)


class BandFilterInput(BaseModel):
    column: str = Field(description="Banded column, e.g. aum_band, sip_band, term_cover_band, income_group, age_group")
    amount: str = Field(description="Threshold as written by the user, e.g. '1 crore', '50,000', '25k', '8 lakh', '56'")
    direction: str = Field(description="at_least | more_than | below | at_most")


@tool("band_filter", args_schema=BandFilterInput)
def band_filter(column: str, amount: str, direction: str) -> str:
    """Turn a threshold on a BANDED column into an exact SQL condition on its _rank column.
    Always use this for questions like 'AUM of 1 crore or more', 'SIP above 50k', 'income below
    24 lakh', 'aged 56+'."""
    try:
        con = _connect()
        try:
            rows = con.execute("SELECT label, rank, low, high, unit FROM bands WHERE column = ? ORDER BY rank",
                               (column,)).fetchall()
            base = re.sub(r"_(band|group)$", "", column)
        finally:
            con.close()
        if not rows:
            return json.dumps({"error": f"'{column}' is not a banded amount column: filter it directly "
                                        f"in run_sql (e.g. WHERE {column} = ...)."})
        x = parse_amount(amount)
        if rows[0][4] == "inr_per_year" and column == "income_group" and x < 1000:
            x *= 1e5                                  # "income above 24" means 24 lakh p.a.
        zero = [r for r in rows if r[0] == "0"]
        bands = [r for r in rows if r[0] != "0"]
        if direction not in ("at_least", "more_than", "below", "at_most"):
            return json.dumps({"error": "direction must be at_least, more_than, below or at_most"})
        if direction in ("at_least", "more_than"):
            keep = [r for r in bands if r[2] >= x]                       # band starts at/above X
        else:
            keep = [r for r in bands if r[3] is not None and r[3] <= x]  # band ends at/below X
        straddle = [r for r in bands if r[2] < x and (r[3] is None or r[3] > x)]
        if not keep:
            return json.dumps({"condition": "0", "labels": [], "note": "no band lies entirely beyond this threshold",
                               "straddling_bands": [r[0] for r in straddle]})
        ranks = [r[1] for r in keep]
        cond = (f"{base}_rank >= {min(ranks)}" if direction in ("at_least", "more_than")
                else f"{base}_rank BETWEEN {min(ranks)} AND {max(ranks)}")
        out = {"condition": cond, "labels": [r[0] for r in keep]}
        if straddle:
            out["straddling_bands"] = [r[0] for r in straddle]
            out["note"] = "these bands contain the threshold and were excluded; mention this if relevant"
        if zero and direction in ("below", "at_most"):
            out["note_zero"] = f"clients without the product ({base}_rank = 0) are excluded; add 'OR {base}_rank = 0' to include them"
        return json.dumps(out, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"band_filter failed: {e}"})


# ---------------------------------------------------------------- run_sql
class SQLInput(BaseModel):
    sql: str = Field(description="One read-only SQLite SELECT query over the `clients` table.")


@tool("run_sql", args_schema=SQLInput)
def run_sql(sql: str) -> str:
    """Run ONE read-only SQLite SELECT over the client database and return columns + rows."""
    try:
        safe = check_sql(sql)
        safe, fixes = ground_literals(safe)
        con = _connect()
        try:
            cur = con.execute(safe)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(200)
        finally:
            con.close()
        out = {"sql": safe, "columns": cols, "row_count": len(rows),
               "rows": [[round(v, 2) if isinstance(v, float) else v for v in r] for r in rows]}
        if fixes:
            out["corrected_values"] = fixes
        money = [j for j, c in enumerate(cols) if "inr" in c.lower()]
        if money and rows:  # pre-formatted lakhs/crores: models are unreliable at this conversion
            out["inr_formatted"] = [{cols[j]: fmt_inr(r[j]) for j in money if isinstance(r[j], (int, float))}
                                    for r in rows[:50]]
        if not rows or all(v is None for r in rows for v in r):
            out["warning"] = ("Query returned no data. Do NOT report 0 or 'none' yet: check the exact "
                              "band labels / values in the data dictionary and retry. If you filtered "
                              "on a person's name, call find_client to get the exact spelling.")
        return mask_pii(json.dumps(out, ensure_ascii=False))
    except GuardrailError as e:
        return json.dumps({"error": f"blocked by guardrail: {e}"})
    except sqlite3.Error as e:
        return json.dumps({"error": f"SQL error: {e}. Check column names in the data dictionary."})


# ---------------------------------------------------------------- find_client
def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def match_names(query: str, names: list[str], limit: int = 5) -> list[tuple[str, float]]:
    """Rank names by similarity to the query, ignoring case, spacing, word order and small typos.

    Every query word is matched to its most similar word in the name (prefixes count, so
    'Ram' finds 'RAMACHANDRAN'); the score is the mean of those word similarities.
    """
    from difflib import SequenceMatcher

    q = _tokens(query)
    if not q:
        return []
    scored = []
    for name in names:
        n = _tokens(name)
        if not n:
            continue
        sims = []
        for qt in q:
            best = 0.0
            for nt in n:
                s = 1.0 if (qt == nt or (len(qt) >= 3 and nt.startswith(qt))) else SequenceMatcher(None, qt, nt).ratio()
                best = max(best, s)
            sims.append(best)
        scored.append((name, sum(sims) / len(sims)))
    scored.sort(key=lambda x: -x[1])
    return [(n, round(s, 2)) for n, s in scored[:limit] if s >= 0.6]


class FindInput(BaseModel):
    name: str = Field(description="Client name or part of it, in any case, e.g. 'ravi sharma' or 'sharma'")


@tool("find_client", args_schema=FindInput)
def find_client(name: str) -> str:
    """Find clients by name, ignoring upper/lower case, extra spaces, word order and small typos.
    Use this FIRST whenever the user mentions a specific person, then query with the exact name
    it returns (WHERE name = '<exact name>')."""
    con = _connect()
    try:
        names = [r[0] for r in con.execute("SELECT name FROM clients WHERE name IS NOT NULL")]
    finally:
        con.close()
    matches = match_names(name, names)
    if not matches:
        return json.dumps({"matches": [], "note": f"No client name resembles '{name}'."})
    exact = matches[0][1] >= 0.95 and (len(matches) == 1 or matches[1][1] < 0.95)
    return json.dumps({"matches": [{"name": n, "score": s} for n, s in matches],
                       "note": "Use the exact name in SQL." + ("" if exact else
                               " Several similar names: confirm with the user if unsure.")},
                      ensure_ascii=False)


# ---------------------------------------------------------------- calculate
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
        ast.UAdd: operator.pos, ast.Mod: operator.mod}
_FUNCS = {"round": round, "abs": abs, "sqrt": math.sqrt, "min": min, "max": max,
          "pct": lambda part, whole: 100.0 * part / whole,
          "lakhs": lambda inr: inr / 1e5, "crores": lambda inr: inr / 1e7,
          "sip_future_value": lambda monthly, annual_rate_pct, years: monthly * (
              ((1 + annual_rate_pct / 1200) ** (12 * years) - 1) / (annual_rate_pct / 1200))
          * (1 + annual_rate_pct / 1200)}


def safe_eval(expr: str) -> float:
    """Arithmetic and whitelisted functions only; Python eval is never used."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in _FUNCS and not node.keywords):
            return _FUNCS[node.func.id](*[ev(a) for a in node.args])
        raise ValueError("unsupported expression element")

    if len(expr) > 300:
        raise ValueError("expression too long")
    return ev(ast.parse(expr.replace("^", "**"), mode="eval"))


class CalcInput(BaseModel):
    expression: str = Field(description="Arithmetic expression. Functions: pct(part, whole), lakhs(inr), "
                                        "crores(inr), sip_future_value(monthly, annual_rate_pct, years), "
                                        "round(x, n). Example: pct(4, 18)")


@tool("calculate", args_schema=CalcInput)
def calculate(expression: str) -> str:
    """Exact arithmetic for percentages, conversions to lakhs/crores and SIP projections."""
    try:
        clean = re.sub(r"(?<=\d),(?=\d)", "", expression)
        return json.dumps({"expression": expression, "result": round(float(safe_eval(clean)), 4)})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"could not evaluate: {e}"})


# ---------------------------------------------------------------- create_excel_report
class ReportInput(BaseModel):
    title: str = Field(description="Report title, e.g. 'Clients without health cover'")
    sql: str = Field(description="Read-only SELECT whose result becomes the report table")
    note: str = Field(default="", description="One-line explanation shown in the report")
    chart: str = Field(default="none", description="Optional native Excel chart: none | bar | column | pie. "
                                                   "Needs a category column and a numeric column, e.g. "
                                                   "SELECT aum_band, COUNT(*) AS clients ... GROUP BY aum_band")


@tool("create_excel_report", args_schema=ReportInput)
def create_excel_report(title: str, sql: str, note: str = "", chart: str = "none") -> str:
    """Create a formatted Excel report (.xlsx) from a SELECT query: table, live SUM formulas for
    numeric columns, filters, and an 'About' sheet with the query. Saved locally."""
    try:
        safe = check_sql(sql, max_rows=5000)
        safe, _ = ground_literals(safe)
        con = _connect()
        try:
            df = pd.read_sql(safe, con)
        finally:
            con.close()
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:40] or "report"
        path = OUTPUTS / f"{datetime.now():%Y%m%d_%H%M%S}_{slug}.xlsx"
        with pd.ExcelWriter(path, engine="xlsxwriter") as xw:
            df.to_excel(xw, sheet_name="Report", startrow=2, index=False)
            wb, ws = xw.book, xw.sheets["Report"]
            ws.write(0, 0, title, wb.add_format({"bold": True, "font_size": 14}))
            head = wb.add_format({"bold": True, "bg_color": "#D9E1F2", "border": 1})
            money = wb.add_format({"num_format": "₹#,##0"})
            for j, col in enumerate(df.columns):
                ws.write(2, j, col, head)
                width = max(len(str(col)), *(len(str(v)) for v in df[col].head(200))) if len(df) else len(col)
                fmt = money if ("_inr" in col or col.endswith("_est")) else None
                ws.set_column(j, j, min(width + 2, 45), fmt)
            n = len(df)
            if n:
                ws.autofilter(2, 0, 2 + n, len(df.columns) - 1)
                total_row = 3 + n
                ws.write(total_row, 0, "Total / count", wb.add_format({"bold": True}))
                for j, col in enumerate(df.columns):
                    if j and pd.api.types.is_numeric_dtype(df[col]):
                        cl = _col_letter(j)
                        ws.write_formula(total_row, j, f"=SUM({cl}4:{cl}{3 + n})",
                                         wb.add_format({"bold": True, **({"num_format": "₹#,##0"} if "_inr" in col else {})}))
            chart_note = _add_excel_chart(wb, ws, df, chart, title) if n else None
            about = wb.add_worksheet("About")
            for i, (k, v) in enumerate([("Report", title), ("Note", note), ("Generated", f"{datetime.now():%d %b %Y %H:%M}"),
                                        ("Rows", n), ("SQL", safe),
                                        ("Caveat", "Columns ending in _est_* are estimates from band midpoints.")]):
                about.write(i, 0, k, head)
                about.write(i, 1, str(v))
            about.set_column(0, 0, 12)
            about.set_column(1, 1, 100)
        out = {"saved_to": str(path), "rows": n, "columns": list(df.columns)}
        if chart_note:
            out["chart"] = chart_note
        return json.dumps(out)
    except GuardrailError as e:
        return json.dumps({"error": f"blocked by guardrail: {e}"})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"report failed: {e}"})


# ---------------------------------------------------------------- make_chart
class ChartInput(BaseModel):
    title: str = Field(description="Chart title stating the takeaway, e.g. 'Most clients hold no health cover'")
    sql: str = Field(description="Read-only SELECT returning one category column, one numeric column and "
                                 "optionally one series column, e.g. SELECT aum_band, COUNT(*) AS clients "
                                 "FROM clients GROUP BY aum_band")
    chart_type: str = Field(description="bar (compare categories) | column (distribution over ordered bands "
                                        "such as age_group or aum_band) | stacked_bar (two dimensions, e.g. "
                                        "AUM band split by risk profile) | donut (share of a whole, <= 6 parts) | line")
    x: str = Field(description="Category column name in the query result")
    y: str = Field(description="Numeric column name in the query result (exactly one measure)")
    series: str | None = Field(default=None, description="Second category column, only for stacked_bar")
    note: str = Field(default="", description="Optional subtitle, e.g. 'AUM values are band-midpoint estimates'")


def _melt_wide(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame | None:
    """A 'wide' result (several numeric columns, no category column x) becomes one row per
    column: x = readable column name, y = column total. Lets 'how many hold each product' work
    whether the model selects per-client flags or pre-summed columns."""
    if x in df.columns:
        return None
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num) < 2 or len(num) != len(df.columns):
        return None
    from .charts import pretty

    def label(c: str) -> str:
        return pretty(re.sub(r"^(has_|n_|num_|count_)", "", c).replace("_cover", " cover"))

    # flags (0/1 columns) grouped with a count column: weight each flag by that count
    weight = y if y in df.columns else None
    flags = [c for c in num if c != weight]
    if weight and all(set(df[c].dropna().unique()) <= {0, 1} for c in flags):
        return pd.DataFrame({x: [label(c) for c in flags],
                             y: [float((df[c] * df[weight]).sum()) for c in flags]})
    return pd.DataFrame({x: [label(c) for c in num], y: [float(df[c].sum()) for c in num]})


def _band_order(values: list, complete: bool = False) -> list | None:
    """If the categories are labels of one banded column, return them in band order
    (complete=True returns every band of that column except the '0' = not-held band)."""
    con = _connect()
    try:
        bands = pd.read_sql("SELECT column, label, rank FROM bands", con)
    finally:
        con.close()
    vals = {str(v) for v in values}
    for _, g in bands.groupby("column"):
        labels = g.sort_values("rank").label.tolist()
        if vals and vals <= set(labels):
            return [lab for lab in labels if lab in vals or (complete and lab != "0")]
    return None


@tool("make_chart", args_schema=ChartInput)
def make_chart(title: str, sql: str, chart_type: str, x: str, y: str, series: str | None = None,
               note: str = "") -> str:
    """Draw a chart from a read-only SELECT (shown in the app with hover tooltips and saved as PNG).
    One measure per chart. Use when the user asks for a chart/graph/plot/visual."""
    from .charts import CHART_TYPES, MAX_DONUT, MAX_SERIES, build_spec, fold_other, to_png

    try:
        if chart_type not in CHART_TYPES:
            return json.dumps({"error": f"chart_type must be one of {list(CHART_TYPES)}"})
        safe = check_sql(sql, max_rows=500)
        safe, _ = ground_literals(safe)
        con = _connect()
        try:
            df = pd.read_sql(safe, con)
        finally:
            con.close()
        wide = _melt_wide(df, x, y)
        if wide is not None:      # e.g. one row per client with has_term_cover, has_sip, ... columns
            df = wide
            notes_pre = [f"summed {len(df)} numeric columns into one bar per column"]
        else:
            notes_pre = []
        missing = [c for c in [x, y] + ([series] if series else []) if c not in df.columns]
        if missing:
            return json.dumps({"error": f"columns {missing} not in query result {list(df.columns)}. "
                                        "The query must return one category column and one numeric column, "
                                        "e.g. SELECT aum_band, COUNT(*) AS clients FROM clients GROUP BY aum_band. "
                                        "For one bar per product/column, select the columns themselves "
                                        "(e.g. SELECT SUM(has_term_cover) AS term_cover, SUM(has_sip) AS sip FROM clients) "
                                        "and pass x='product', y='clients'."})
        if not pd.api.types.is_numeric_dtype(df[y]):
            return json.dumps({"error": f"'{y}' must be numeric (one measure per chart)"})
        if df.empty:
            return json.dumps({"error": "query returned no rows; nothing to chart"})
        notes = list(notes_pre)
        others = [c for c in df.columns if c not in (x, y) and not pd.api.types.is_numeric_dtype(df[c])]
        if not series and len(others) == 1 and chart_type in ("bar", "column", "stacked_bar"):
            # two category columns: show the split instead of silently dropping a dimension
            series = others[0]
            if chart_type != "stacked_bar":
                notes.append(f"result has a second category ('{series}'): drew a stacked bar to keep the split")
            chart_type = "stacked_bar"
        if chart_type == "stacked_bar" and not series:
            chart_type = "bar"
            notes.append("no series column given: drew a plain bar chart")
        if chart_type != "stacked_bar":
            series = None
            df = df.groupby(x, as_index=False, sort=False)[y].sum()
        df[x] = df[x].astype(str)
        if series:
            df[series] = df[series].astype(str)
            df, folded = fold_other(df, series, y, MAX_SERIES)
            if folded:
                notes.append(f"more than {MAX_SERIES} series: smallest folded into 'Other'")
        if chart_type == "donut" and df[x].nunique() < 3:
            chart_type = "bar"   # a 2-slice donut hides the comparison; bars (or the numbers) read better
            notes.append("only 2 parts: drew a bar chart instead of a donut")
        if chart_type == "donut":
            df, folded = fold_other(df, x, y, MAX_DONUT)
            if folded:
                notes.append(f"more than {MAX_DONUT} parts: smallest folded into 'Other'")
            if (df[y] < 0).any():
                return json.dumps({"error": "a donut cannot show negative values; use bar"})
        order = _band_order(df[x].unique().tolist())
        if order and chart_type in ("bar", "column", "line"):
            full = _band_order(df[x].unique().tolist(), complete=True)
            missing = [b for b in full if b not in set(df[x])]
            if missing:   # empty bands are shown as 0 so the distribution's shape is honest
                df = pd.concat([df, pd.DataFrame({x: missing, y: 0})], ignore_index=True)
                order = full
        series_order = _band_order(df[series].unique().tolist()) if series else None
        series_domain = _band_order(df[series].unique().tolist(), complete=True) if series else None
        record = {"title": title, "subtitle": note, "chart_type": chart_type, "x": x, "y": y,
                  "color": series, "order": order, "series_order": series_order,
                  "series_domain": series_domain, "sql": safe,
                  "data": json.loads(df.to_json(orient="records"))}
        charts = OUTPUTS / "charts"
        charts.mkdir(parents=True, exist_ok=True)
        stem = f"{datetime.now():%Y%m%d_%H%M%S}_{re.sub(r'[^a-z0-9]+', '_', title.lower()).strip('_')[:40]}"
        (charts / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        to_png(build_spec(record, "light"), charts / f"{stem}.png")
        return json.dumps({"chart": str(charts / f"{stem}.json"), "png": str(charts / f"{stem}.png"),
                           "chart_type": chart_type, "rows": len(df), "notes": notes,
                           "table": record["data"][:30]}, ensure_ascii=False)
    except GuardrailError as e:
        return json.dumps({"error": f"blocked by guardrail: {e}"})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"chart failed: {e}"})


def _add_excel_chart(wb, ws, df: pd.DataFrame, kind: str, title: str) -> str | None:
    """Native, editable Excel chart next to the table (same palette as the app charts)."""
    from .charts import PALETTE

    if kind not in ("bar", "column", "pie"):
        return None
    cats = [j for j, c in enumerate(df.columns) if not pd.api.types.is_numeric_dtype(df[c])]
    vals = [j for j, c in enumerate(df.columns) if pd.api.types.is_numeric_dtype(df[c]) and j not in cats]
    if not cats or not vals:
        return "chart skipped: needs one text column and one numeric column"
    ci, vi, n = cats[0], vals[0], len(df)
    if kind == "pie" and n > 6:
        kind = "bar"                                   # > 6 slices: a bar reads better
    chart = wb.add_chart({"type": kind})
    series = {"name": str(df.columns[vi]),
              "categories": ["Report", 3, ci, 2 + n, ci],
              "values": ["Report", 3, vi, 2 + n, vi],
              "data_labels": {"value": True, "font": {"color": "#52514e"}}}
    if kind == "pie":
        series["points"] = [{"fill": {"color": PALETTE["light"][i]}, "border": {"color": "#fcfcfb", "width": 1.5}}
                            for i in range(n)]
        series["data_labels"] = {"percentage": True, "category": False, "font": {"color": "#0b0b0b"}}
    else:
        series["fill"] = {"color": PALETTE["light"][0]}
        series["border"] = {"none": True}
        series["gap"] = 80
    chart.add_series(series)
    chart.set_title({"name": title, "name_font": {"size": 12, "bold": True, "color": "#0b0b0b"}})
    if kind != "pie":
        chart.set_legend({"none": True})                 # one series: the title names it
        grid = {"visible": True, "line": {"color": "#e1e0d9", "width": 0.75}}
        axis_line = {"color": "#c3c2b7"}
        value_axis = {"major_gridlines": grid, "line": {"none": True}, "num_font": {"color": "#898781"}}
        cat_axis = {"line": axis_line, "num_font": {"color": "#898781"}}
        chart.set_x_axis(value_axis if kind == "bar" else cat_axis)
        chart.set_y_axis(cat_axis if kind == "bar" else value_axis)
        if kind == "bar":
            chart.set_y_axis({**cat_axis, "reverse": True})  # keep table order top-to-bottom
    chart.set_chartarea({"border": {"none": True}, "fill": {"color": "#fcfcfb"}})
    chart.set_plotarea({"fill": {"color": "#fcfcfb"}})
    chart.set_size({"width": 560, "height": max(300, 28 * n + 90) if kind == "bar" else 320})
    ws.insert_chart(2, len(df.columns) + 1, chart)
    return f"native Excel {kind} chart added"


def _col_letter(j: int) -> str:
    s, j = "", j + 1
    while j:
        j, r = divmod(j - 1, 26)
        s = chr(65 + r) + s
    return s


TOOLS = [find_client, band_filter, run_sql, calculate, make_chart, create_excel_report]
