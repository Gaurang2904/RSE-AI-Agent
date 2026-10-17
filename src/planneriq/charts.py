"""Chart specs (Vega-Lite via Altair): one spec renders interactively in the app (hover
tooltips) and to PNG offline (vl-convert) for downloads.

Design rules enforced here, not left to the model:
- form by job: bar (categories, horizontal for long labels), column (ordered bands), stacked_bar
  (two dimensions / part-to-whole per category), donut (part-to-whole, <= 6 segments), line
- one numeric measure per chart: never a second y-axis
- one series -> one color (slot 1); series colors follow a fixed validated order, never cycled;
  more than 8 series (6 for donut) fold into "Other"
- banded categories (AUM, age, SIP ...) keep their natural band order
- thin marks: bars <= 24px with a 4px rounded data end, 2px surface gaps, 2px lines, 8px dots
  with a surface ring; hairline solid gridlines; text in ink tokens, never series colors
- light and dark are separate validated steps (the light palette passes all hard checks; three
  light slots are below 3:1 contrast, so charts always carry labels/legend and a table view)
"""
from __future__ import annotations

import altair as alt
import pandas as pd

PALETTE = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
}
CHROME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835"},
}
FONT = "system-ui, -apple-system, Segoe UI, sans-serif"
CHART_TYPES = ("bar", "column", "stacked_bar", "donut", "line")
MAX_SERIES = 8
MAX_DONUT = 6


def pretty(name: str) -> str:
    """Column name -> axis/legend title: 'sip_inr_month' -> 'SIP per month', 'risk_profile' -> 'Risk profile'."""
    s = name.replace("_est_inr_month", " per month").replace("_inr_month", " per month")
    s = s.replace("_est_inr_year", " per year").replace("_inr_year", " per year")
    s = s.replace("_est_inr", "").replace("_inr", "").replace("_band", "").replace("_group", "")
    s = s.replace("_", " ").strip()
    words = [w.upper() if w in {"sip", "swp", "pms", "las", "aum", "ulip", "li", "rm"} else w for w in s.split()]
    out = " ".join(words)
    return out[:1].upper() + out[1:]


def scale_inr(df: pd.DataFrame, col: str) -> tuple[pd.DataFrame, str]:
    """Rupee columns are shown in lakhs or crores so axes read '12' not '1.2M'."""
    peak = df[col].abs().max() if len(df) else 0
    if "inr" not in col.lower() or not peak or peak < 1e5:
        return df, ""
    div, unit = (1e7, "₹ crore") if peak >= 1e7 else (1e5, "₹ lakh")
    df = df.copy()
    df[col] = (df[col] / div).round(2)
    return df, unit


def fold_other(df: pd.DataFrame, cat: str, val: str, keep: int) -> tuple[pd.DataFrame, bool]:
    """Keep the `keep - 1` largest categories and sum the rest into 'Other'."""
    totals = df.groupby(cat, sort=False)[val].sum().sort_values(ascending=False)
    if len(totals) <= keep:
        return df, False
    top = set(totals.index[: keep - 1])
    out = df.copy()
    out[cat] = out[cat].where(out[cat].isin(top), "Other")
    group_cols = [c for c in out.columns if c != val]
    return out.groupby(group_cols, as_index=False, sort=False)[val].sum(), True


def _config(chart, mode: str):
    c = CHROME[mode]
    return (chart.configure(background=c["surface"], font=FONT)
            .configure_view(stroke=None)
            .configure_axis(gridColor=c["grid"], gridWidth=1, gridDash=[], domainColor=c["axis"],
                            tickColor=c["axis"], labelColor=c["muted"], titleColor=c["ink2"],
                            labelFontSize=11, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(labelColor=c["ink2"], titleColor=c["ink2"], orient="top",
                              symbolType="square", labelFontSize=11)
            .configure_title(color=c["ink"], fontSize=15, fontWeight=600, anchor="start",
                             subtitleColor=c["muted"], subtitleFontSize=11))


def build_spec(record: dict, mode: str = "light") -> dict:
    """record = {title, subtitle, chart_type, x, y, color, order, data(list of dicts)}"""
    df = pd.DataFrame(record["data"])
    x, y, color = record["x"], record["y"], record.get("color")
    kind, order = record["chart_type"], record.get("order") or None
    series_order = record.get("series_order") or None
    df, unit = scale_inr(df, y)
    y_title = pretty(y) + (f" ({unit})" if unit else "")
    x_title, color_title = pretty(x), pretty(color) if color else None
    pal, c = PALETTE[mode], CHROME[mode]
    title = alt.TitleParams(record["title"], subtitle=record.get("subtitle") or "")
    tip = [alt.Tooltip(x, title=x_title)] + ([alt.Tooltip(color, title=color_title)] if color else []) + \
          [alt.Tooltip(y, title=y_title, format=",.4~g")]
    n_cat = df[x].nunique()

    if kind in ("bar", "column"):
        horizontal = kind == "bar"
        longest = max((len(str(v)) for v in df[x]), default=1)
        fits = longest * 6.5 + 12 <= 560 / max(n_cat, 1)
        thick = min(24, max(10, int(220 / max(n_cat, 1))))
        sort = order if order else alt.EncodingSortField(field=y, order="descending")
        cat_enc = (alt.Y if horizontal else alt.X)(f"{x}:N", sort=sort, title=None,
                                                    axis=alt.Axis(labelLimit=220, labelAngle=0 if horizontal or fits else -30,
                                                                  labelAlign="right" if horizontal or not fits else "center",
                                                                  labelPadding=6))
        val_enc = (alt.X if horizontal else alt.Y)(f"{y}:Q", title=y_title,
                                                    axis=alt.Axis(grid=True, tickCount=5, format="~s"))
        base = alt.Chart(df).encode(cat_enc, val_enc)
        bars = base.mark_bar(size=thick, cornerRadiusEnd=4, color=pal[0]).encode(tooltip=tip)
        layers = [bars]
        if n_cat <= 15:   # value at the bar tip, in ink (never the series color)
            layers.append(base.mark_text(align="left" if horizontal else "center",
                                         baseline="middle" if horizontal else "bottom",
                                         dx=5 if horizontal else 0, dy=0 if horizontal else -5,
                                         color=c["ink2"], fontSize=11)
                          .encode(text=alt.Text(f"{y}:Q", format=",.4~g")))
        height = max(160, n_cat * (thick + 14)) if horizontal else 300
        chart = alt.layer(*layers).properties(title=title, width=560, height=height)

    elif kind == "stacked_bar":
        present = list(dict.fromkeys(df[color].tolist()))
        domain = record.get("series_domain") or series_order or present
        series = [s_ for s_ in domain if s_ in present] + [s_ for s_ in present if s_ not in domain]
        domain = domain + [s_ for s_ in present if s_ not in domain]
        scale = alt.Scale(domain=domain, range=pal[: len(domain)])
        sort = order if order else alt.EncodingSortField(field=y, op="sum", order="descending")
        thick = min(24, max(10, int(220 / max(n_cat, 1))))
        base = alt.Chart(df).encode(alt.Y(f"{x}:N", sort=sort, title=None, axis=alt.Axis(labelLimit=220)))
        bars = base.mark_bar(size=thick, stroke=c["surface"], strokeWidth=2).encode(
            alt.X(f"{y}:Q", stack="zero", title=y_title, axis=alt.Axis(tickCount=5, format="~s")),
            alt.Color(f"{color}:N", scale=scale, sort=series, title=color_title,
                      legend=alt.Legend(values=series)),
            alt.Order("series_order:Q"), tooltip=tip).transform_calculate(
            series_order=f"indexof({series!r}, datum['{color}'])".replace("'", '"'))
        totals = base.transform_aggregate(total=f"sum({y})", groupby=[x]).mark_text(
            align="left", dx=5, color=c["ink2"], fontSize=11).encode(
            alt.X("total:Q"), text=alt.Text("total:Q", format=",.4~g"))
        chart = alt.layer(bars, totals).properties(title=title, width=560,
                                                   height=max(160, n_cat * (thick + 14)))

    elif kind == "donut":
        if not order:
            df = df.sort_values(y, ascending=False)
        total = df[y].sum()
        df = df.assign(share=(df[y] / total * 100).round(1) if total else 0)
        # legend labels carry the numbers, so values are readable without hovering
        df["legend"] = [f"{a}: {v:,.4g} ({p:.0f}%)" for a, v, p in zip(df[x], df[y], df["share"])]
        cats = df["legend"].tolist()
        scale = alt.Scale(domain=cats, range=pal[: len(cats)])
        chart = alt.Chart(df).mark_arc(innerRadius=70, outerRadius=120, stroke=c["surface"], strokeWidth=2).encode(
            alt.Theta(f"{y}:Q", stack=True), alt.Color("legend:N", scale=scale, sort=cats, title=None,
                                                         legend=alt.Legend(orient="right", labelLimit=320)),
            alt.Order("order_idx:Q"),
            tooltip=tip + [alt.Tooltip("share:Q", title="share %")]).transform_calculate(
            order_idx=f"indexof({cats!r}, datum['legend'])".replace("'", '"')).properties(
            title=title, width=280, height=260)

    elif kind == "line":
        sort = order if order else None
        base = alt.Chart(df).encode(alt.X(f"{x}:N", sort=sort, title=None, axis=alt.Axis(labelAngle=0)),
                                    alt.Y(f"{y}:Q", title=y_title, axis=alt.Axis(tickCount=5, format="~s")))
        line = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round", color=pal[0])
        dots = base.mark_point(size=64, filled=True, color=pal[0], stroke=c["surface"], strokeWidth=2,
                               opacity=1).encode(tooltip=tip)
        chart = alt.layer(line, dots).properties(title=title, width=560, height=300)
    else:
        raise ValueError(f"chart_type must be one of {CHART_TYPES}")

    return _config(chart, mode).to_dict()


def to_png(spec: dict, path) -> None:
    import vl_convert as vlc

    path.write_bytes(vlc.vegalite_to_png(spec, scale=2))
