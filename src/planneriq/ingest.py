"""Load a client workbook (merged multi-row headers, dropdown bands) into a private SQLite DB.

    python -m planneriq.ingest "path/to/client_workbook.xlsx"

Creates three tables in PLANNERIQ_PRIVATE/clients.db:
  clients  - one row per client: the original band labels plus, for banded columns,
             <base>_rank (order of the band) and <base>_est_<unit> (band midpoint estimate),
             and has_<product> flags (1 if the client holds the product)
  bands    - every allowed band: column, label, rank, low, high, estimate, unit
  columns  - data dictionary: column name, original header, header group, description, unit
Contact/identity columns (phone, email, PAN, ...) are never loaded. The workbook is opened
read-only and never modified.
"""
from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import openpyxl
import pandas as pd
import yaml

from .config import BANDS_YAML, DB_PATH, EXCLUDED_COLUMNS, PRIVATE

UNIT_SUFFIX = {"inr": "est_inr", "inr_per_year": "est_inr_year", "inr_per_month": "est_inr_month",
               "years": "est_years"}


def snake(name: str) -> str:
    s = re.sub(r"\(.*?\)", "", str(name)).strip().lower()
    s = re.sub(r"[^0-9a-z]+", "_", s).strip("_")
    return s if s and not s[0].isdigit() else f"c_{s}"


def base_name(column: str) -> str:
    return re.sub(r"_(band|group)$", "", column)


def read_header(ws, scan_rows: int = 10) -> tuple[int, list[str], list[str | None]]:
    """Find the leaf header row beneath merged group titles.

    Returns (header_row, leaf_names, group_titles) with one entry per column.
    """
    n_cols = ws.max_column
    grid = [[ws.cell(r, c).value for c in range(1, n_cols + 1)] for r in range(1, scan_rows + 1)]
    group = [None] * n_cols
    for rng in ws.merged_cells.ranges:
        top = ws.cell(rng.min_row, rng.min_col).value
        if rng.min_row > scan_rows:
            continue
        for r in range(rng.min_row, min(rng.max_row, scan_rows) + 1):
            for c in range(rng.min_col, rng.max_col + 1):
                if rng.max_col > rng.min_col:          # horizontal merge = group title
                    group[c - 1] = top
                    grid[r - 1][c - 1] = None
                else:                                   # vertical merge = leaf header
                    grid[r - 1][c - 1] = top
    counts = [sum(isinstance(v, str) and v.strip() != "" for v in row) for row in grid]
    header_row = max(range(scan_rows), key=lambda i: counts[i]) + 1
    names = [str(v).strip() if v is not None else f"col_{i + 1}" for i, v in enumerate(grid[header_row - 1])]
    return header_row, names, group


def load_bands() -> dict:
    return yaml.safe_load(BANDS_YAML.read_text(encoding="utf-8"))


def norm_label(v) -> str | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)) and float(v) == 0:
        return "0"
    return re.sub(r"\s+", " ", str(v)).strip()


def ingest(xlsx: Path, db: Path = DB_PATH, sheet: str | None = None) -> dict:
    wb = openpyxl.load_workbook(xlsx, read_only=False, data_only=True)
    ws = wb[sheet] if sheet else wb.active
    header_row, headers, groups = read_header(ws)
    spec = load_bands()
    spec_by_header = {h.strip().lower(): (h, s) for h, s in spec.items()}

    rows = []
    for r in range(header_row + 1, ws.max_row + 1):
        vals = [ws.cell(r, c).value for c in range(1, len(headers) + 1)]
        if all(v is None or str(v).strip() == "" for v in vals):
            continue
        rows.append(vals)

    out: dict[str, list] = {}
    dictionary, band_rows, warnings = [], [], []
    for i, header in enumerate(headers):
        key = header.strip().lower()
        if key in EXCLUDED_COLUMNS:
            dictionary.append({"column": None, "source_header": header, "group": groups[i],
                               "description": "EXCLUDED (personal contact data, never loaded)",
                               "unit": None})
            continue
        raw = [row[i] for row in rows]
        if key in spec_by_header:
            _, s = spec_by_header[key]
            col, unit, scale = s["column"], s["unit"], s.get("scale", 1)
            labels = [b[0] for b in s["bands"]]
            lookup = {str(lab).strip().lower(): j for j, lab in enumerate(labels)}
            base = base_name(col)
            out[col] = [norm_label(v) for v in raw]
            ranks, ests = [], []
            for v in out[col]:
                j = lookup.get(v.lower()) if v is not None else None
                if v is not None and j is None:
                    warnings.append(f"{header}: unknown band '{v}'")
                ranks.append(j)
                if j is None:
                    ests.append(None)
                else:
                    _, lo, hi = s["bands"][j]
                    est = lo if hi is None else (lo + hi) / 2
                    ests.append(est * scale)
            out[f"{base}_rank"] = ranks
            if unit in UNIT_SUFFIX:
                out[f"{base}_{UNIT_SUFFIX[unit]}"] = ests
            if s.get("product"):                      # band 0 = product not held
                out[f"has_{base}"] = [None if j is None else int(j > 0) for j in ranks]
            for j, (lab, lo, hi) in enumerate(s["bands"]):
                band_rows.append({"column": col, "label": str(lab), "rank": j,
                                  "low": None if lo is None else lo * scale,
                                  "high": None if hi is None else hi * scale,
                                  "estimate": (lo if hi is None else (lo + hi) / 2) * scale,
                                  "unit": unit, "open_ended": hi is None})
            dictionary.append({"column": col, "source_header": header, "group": groups[i],
                               "description": s.get("description", ""), "unit": unit})
        else:
            col = snake(header)
            series = pd.Series(raw)
            numeric = pd.to_numeric(series, errors="coerce")
            if series.notna().sum() == 0:
                continue                      # completely empty column (e.g. RM/Team)
            if numeric.notna().sum() == series.notna().sum():
                out[col] = numeric.tolist()
            else:
                out[col] = [norm_label(v) for v in raw]
            dictionary.append({"column": col, "source_header": header, "group": groups[i],
                               "description": "", "unit": None})

    clients = pd.DataFrame(out)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    db.parent.mkdir(parents=True, exist_ok=True)
    db.unlink(missing_ok=True)
    con = sqlite3.connect(db)
    # Text columns are case-insensitive, so name = 'ravi sharma' matches 'RAVI SHARMA'.
    text_cols = {c: "TEXT COLLATE NOCASE" for c in clients.columns
                 if clients[c].dropna().map(lambda v: isinstance(v, str)).all() and clients[c].notna().any()}
    clients.to_sql("clients", con, index=False, dtype=text_cols)
    pd.DataFrame(band_rows).to_sql("bands", con, index=False)
    pd.DataFrame(dictionary).to_sql("columns", con, index=False)
    con.commit()
    con.close()
    info = {"db": str(db), "clients": len(clients), "columns": len(clients.columns),
            "header_row": header_row, "warnings": warnings}
    print(f"loaded {len(clients)} clients, {len(clients.columns)} columns -> {db}")
    for w in warnings:
        print("  warning:", w)
    return info


if __name__ == "__main__":
    ingest(Path(sys.argv[1]))
