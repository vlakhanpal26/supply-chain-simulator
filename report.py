# report.py
from __future__ import annotations
import os
from datetime import datetime
import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, Flowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ---------- helpers ----------
def _nice_float(x, n=3):
    try:
        return round(float(x), n)
    except Exception:
        return x

def _read_and_clean_kpis(kpi_compare_csv: str) -> pd.DataFrame:
    df = pd.read_csv(kpi_compare_csv)

    # Keep the most useful columns and rename for readability
    keep = [
        "Scenario",
        "FR", "CSL", "AvgInventory", "TotalHoldingCost", "TotalStockoutCost", "TotalCost",
        "StockoutDays", "OrdersPlacedCount",
        "FR_delta", "CSL_delta", "TotalCost_delta",
        "FR_delta_pct", "CSL_delta_pct", "TotalCost_delta_pct",
    ]
    keep = [c for c in keep if c in df.columns]  # tolerate extra cols
    df = df[keep].copy()

    nice = {
        "Scenario": "Scenario",
        "FR": "Fill Rate",
        "CSL": "Service Level",
        "AvgInventory": "Avg Inv",
        "TotalHoldingCost": "Holding $",
        "TotalStockoutCost": "Stockout $",
        "TotalCost": "Total $",
        "StockoutDays": "Stockout Days",
        "OrdersPlacedCount": "Orders Placed",
        "FR_delta": "Δ FR",
        "CSL_delta": "Δ CSL",
        "TotalCost_delta": "Δ Total $",
        "FR_delta_pct": "Δ FR %",
        "CSL_delta_pct": "Δ CSL %",
        "TotalCost_delta_pct": "Δ Total $ %",
    }
    df.rename(columns=nice, inplace=True)

    # Round numerics
    for c in df.columns:
        if c == "Scenario": 
            continue
        df[c] = df[c].apply(lambda v: _nice_float(v, 3))

    # Sort with baseline first (if present), then by Total $
    if "Scenario" in df.columns and (df["Scenario"] == "baseline").any():
        base = df[df["Scenario"] == "baseline"]
        other = df[df["Scenario"] != "baseline"].sort_values(by="Total $", ascending=True)
        df = pd.concat([base, other], ignore_index=True)

    return df

def _scenario_title_from_filename(fname: str) -> tuple[str, str]:
    """
    Returns (scenario, chart_title) from an output PNG filename pattern
    e.g., 'baseline_inventory.png' → ('baseline', 'Inventory Position & Events — baseline')
          'supplier_delay_plus2_cost.png' → ('supplier_delay_plus2', 'Cumulative Cost — supplier_delay_plus2')
    """
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)
    if name.endswith("_inventory"):
        scen = name[:-10]
        return scen, f"Inventory Position & Events — {scen}"
    if name.endswith("_cost"):
        scen = name[:-5]
        return scen, f"Cumulative Cost — {scen}"
    # default
    return name, name

def _inventory_caption(scen: str) -> str:
    return (
        f"{scen}: Inventory (line) falls with demand and jumps on delivery. "
        "Triangles mark orders placed; diamonds mark arrivals. Healthy policies show a repeating saw-tooth."
    )

def _cost_caption(scen: str) -> str:
    return (
        f"{scen}: Cumulative Holding vs Stockout cost. A flatter dotted line means fewer stockouts. "
        "Rising holding cost reflects higher average inventory."
    )

# ---------- main API ----------
def _read_risk_summary(risk_csv: str | None) -> pd.DataFrame:
    if not risk_csv or not os.path.exists(risk_csv):
        return pd.DataFrame()
    df = pd.read_csv(risk_csv)
    keep_stats = {"mean", "std", "q05", "q95", "risk"}
    if "stat" in df.columns:
        df = df[df["stat"].isin(keep_stats)].copy()
    return df


def make_pdf(kpi_compare_csv: str, out_pdf: str, img_dir: str, risk_csv: str | None = None, advisor_txt: str | None = None):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H0", parent=styles["Title"], fontSize=20, leading=24))
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading2"], spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="Caption", parent=styles["BodyText"], fontSize=9, textColor=colors.grey))

    doc = SimpleDocTemplate(out_pdf, pagesize=landscape(A4), leftMargin=28, rightMargin=28, topMargin=24, bottomMargin=24)
    flow: list[Flowable] = []

    # ---- Title ----
    title = "Supply Chain Simulator — Phase 2 Report"
    flow.append(Paragraph(title, styles["H0"]))
    flow.append(Paragraph(datetime.now().strftime("%b %d, %Y %H:%M"), styles["Small"]))
    flow.append(Spacer(1, 8))

    # ---- KPI table (clean) ----
    df = _read_and_clean_kpis(kpi_compare_csv)
    flow.append(Paragraph("KPI Comparison vs Baseline", styles["H1"]))

    # Build table
    table_data = [list(df.columns)] + df.values.tolist()
    t = Table(table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
        ("TEXTCOLOR", (0,0), (-1,0), colors.black),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 10),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
        ("FONTSIZE", (0,1), (-1,-1), 9),
    ]))
    flow.append(t)
    flow.append(Spacer(1, 10))

    # ---- Executive summary bullets (auto from deltas vs baseline) ----
    try:
        base_row = df[df["Scenario"] == "baseline"].iloc[0].to_dict()
        bullets = []
        for _, row in df.iterrows():
            scen = row["Scenario"]
            if scen == "baseline":
                continue
            d_fr = _nice_float(row.get("Δ FR", 0), 3)
            d_cost = _nice_float(row.get("Δ Total $", 0), 0)
            d_cost_pct = _nice_float(row.get("Δ Total $ %", 0)*100, 1) if isinstance(row.get("Δ Total $ %", 0), (int,float)) else 0
            sign_cost = "↓" if d_cost < 0 else "↑"
            bullets.append(f"<b>{scen}</b>: FR {('+' if d_fr>=0 else '')}{d_fr}, Total $ {sign_cost}{abs(d_cost)} ({abs(d_cost_pct)}%).")
        if bullets:
            flow.append(Paragraph("Executive Summary", styles["H1"]))
            for b in bullets:
                flow.append(Paragraph(f"• {b}", styles["Small"]))
            flow.append(Spacer(1, 10))
    except Exception:
        pass

    # ---- Risk (Monte Carlo) summary ----
    risk_df = _read_risk_summary(risk_csv)
    if not risk_df.empty:
        flow.append(Paragraph("Monte Carlo Risk Summary", styles["H1"]))
        cols = [c for c in risk_df.columns if c != "Scenario"]
        cols = ["Scenario", "stat"] + [c for c in risk_df.columns if c not in {"Scenario", "stat"}]
        risk_df = risk_df[[c for c in cols if c in risk_df.columns]]
        risk_df = risk_df.fillna("")
        data = [list(risk_df.columns)] + risk_df.values.tolist()
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 9),
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("FONTSIZE", (0,1), (-1,-1), 8),
        ]))
        flow.append(tbl)
        flow.append(Spacer(1, 10))

    if advisor_txt and os.path.exists(advisor_txt):
        flow.append(Paragraph("Policy Advisor Recommendation", styles["H1"]))
        with open(advisor_txt, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh.readlines() if line.strip()]
        for line in lines:
            if line.startswith("-"):
                flow.append(Paragraph(line, styles["Small"]))
            else:
                flow.append(Paragraph(line, styles["BodyText"]))
        flow.append(Spacer(1, 10))

    # ---- Per-scenario images (two-up layout with captions) ----
    flow.append(Paragraph("Scenario Charts", styles["H1"]))
    flow.append(Spacer(1, 4))

    # Group images by scenario
    imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]
    by_scen: dict[str, dict[str, str]] = {}
    for p in sorted(imgs):
        scen, title_text = _scenario_title_from_filename(p)
        by_scen.setdefault(scen, {})
        if p.endswith("_inventory.png"):
            by_scen[scen]["inv"] = p
        elif p.endswith("_cost.png"):
            by_scen[scen]["cost"] = p

    for scen in sorted(by_scen.keys()):
        sec = by_scen[scen]
        inv = sec.get("inv"); cost = sec.get("cost")
        if not inv and not cost:
            continue

        # Section header
        flow.append(Paragraph(scen, styles["Heading3"]))
        # Lay images side by side if both exist
        cells = []
        captions = []
        if inv:
            cells.append(Image(inv, width=5.5*inch, height=3.2*inch))
            captions.append(Paragraph(_inventory_caption(scen), styles["Caption"]))
        if cost:
            cells.append(Image(cost, width=5.5*inch, height=3.2*inch))
            captions.append(Paragraph(_cost_caption(scen), styles["Caption"]))

        # Images row
        if cells:
            tbl = Table([cells], colWidths=[5.7*inch]*len(cells))
            tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
            flow.append(tbl)
        # Captions row
        if captions:
            cap_tbl = Table([captions], colWidths=[5.7*inch]*len(captions))
            flow.append(cap_tbl)

        flow.append(Spacer(1, 14))

    # ---- Build ----
    doc.build(flow)
    print(f"✅ PDF report saved → {out_pdf}")
