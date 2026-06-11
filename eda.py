"""
eda.py
Exploratory Data Analysis — generates an interactive HTML dashboard
from the cleaned product DataFrame.

Charts:
  1. Product category distribution
  2. Price distribution (box plot)
  3. Top supplier cities
  4. Data quality / completeness
  5. AI extraction confidence
  6. Price range buckets
  7. Supplier concentration
  8. Key insights (auto-generated text)
"""

import json
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EXPORTS_DIR = Path("data/exports")


# ── Insights ──────────────────────────────────────────────────────────────────

def _generate_insights(df: pd.DataFrame) -> list[str]:
    insights = []

    # Total & category
    top_cat = df["category"].value_counts().index[0] if "category" in df.columns and len(df) > 0 else "N/A"
    top_pct = df["category"].value_counts().iloc[0] / len(df) * 100 if len(df) > 0 else 0
    insights.append(f"<strong>{len(df):,} products</strong> collected. Top category: <strong>{top_cat}</strong> ({top_pct:.0f}% of listings).")

    # Price
    priced = df[df["price_midpoint"].notna()]["price_midpoint"]
    if not priced.empty:
        pct_with_price = len(priced) / len(df) * 100
        insights.append(
            f"<strong>{pct_with_price:.0f}%</strong> of products have listed prices. "
            f"Median price: <strong>₹{priced.median():,.0f}</strong> | Range: ₹{priced.min():,.0f} – ₹{priced.max():,.0f}."
        )
        if priced.std() / priced.mean() > 2:
            insights.append("High price spread detected — the dataset likely mixes low-cost commodities with capital equipment.")

    # Geography
    if "supplier_city" in df.columns:
        top_city = df["supplier_city"].dropna().value_counts()
        if not top_city.empty:
            city = top_city.index[0]
            city_pct = top_city.iloc[0] / len(df) * 100
            insights.append(f"Supplier concentration: <strong>{city_pct:.0f}%</strong> of listings originate from <strong>{city}</strong>.")

    # AI extraction
    if "ai_extracted" in df.columns:
        ai_rate = df["ai_extracted"].mean() * 100
        avg_conf = df["ai_confidence"].dropna().mean()
        insights.append(
            f"AI extraction rate: <strong>{ai_rate:.0f}%</strong>. "
            + (f"Average confidence: <strong>{avg_conf:.2f}</strong>." if pd.notna(avg_conf) else "")
        )

    # Data quality
    no_desc = (~df["has_description"]).mean() * 100
    if no_desc > 30:
        insights.append(f"<strong>{no_desc:.0f}%</strong> of products lack a description — typical for B2B where sellers prefer direct inquiry.")

    # Supplier diversity
    n_suppliers = df["supplier_name"].nunique()
    avg_listings = len(df) / max(n_suppliers, 1)
    insights.append(f"<strong>{n_suppliers:,}</strong> unique suppliers, averaging <strong>{avg_listings:.1f}</strong> listings each.")

    return insights


# ── Charts ────────────────────────────────────────────────────────────────────

DARK = {
    "template": "plotly_dark",
    "paper_bgcolor": "#1E293B",
    "plot_bgcolor": "#1E293B",
    "font": {"color": "#F1F5F9", "size": 12},
    "margin": {"t": 50, "b": 40, "l": 40, "r": 20},
    "height": 380,
}


def _chart_category_bar(df: pd.DataFrame) -> str:
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    fig = px.bar(counts, x="category", y="count", color="category",
                 title="Products per Category", color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(**DARK)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_price_box(df: pd.DataFrame) -> str:
    priced = df[df["price_midpoint"].notna()]
    if priced.empty:
        return "<p style='color:#94A3B8;padding:1rem'>No price data to display.</p>"
    fig = px.box(priced, x="category", y="price_midpoint", color="category", log_y=True,
                 title="Price Distribution by Category (log scale ₹)",
                 labels={"price_midpoint": "Price ₹ (log)"})
    fig.update_layout(**DARK)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_top_cities(df: pd.DataFrame) -> str:
    city_counts = df["supplier_city"].dropna().replace("", pd.NA).dropna().infer_objects(copy=False).value_counts().head(15)
    if city_counts.empty:
        return "<p style='color:#94A3B8;padding:1rem'>No location data to display.</p>"
    fig = px.bar(x=city_counts.values, y=city_counts.index, orientation="h",
                 title="Top 15 Supplier Cities",
                 labels={"x": "Listings", "y": "City"},
                 color=city_counts.values, color_continuous_scale="Blues")
    fig.update_layout(**DARK)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_completeness(df: pd.DataFrame) -> str:
    cols = ["title", "description", "price_min", "supplier_name", "supplier_city", "moq", "ai_product_type"]
    existing = [c for c in cols if c in df.columns]
    pct_filled = ((1 - df[existing].isnull().mean()) * 100).round(1).sort_values()
    fig = go.Figure(go.Bar(
        x=pct_filled.values, y=pct_filled.index, orientation="h",
        marker=dict(color=pct_filled.values,
                    colorscale=[[0, "#EF4444"], [0.5, "#F59E0B"], [1, "#10B981"]],
                    showscale=False),
    ))
    fig.update_layout(title="Field Completeness (%)", xaxis_title="% Filled", xaxis_range=[0, 100], **DARK)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_ai_confidence(df: pd.DataFrame) -> str:
    conf = df["ai_confidence"].dropna() if "ai_confidence" in df.columns else pd.Series(dtype=float)
    if conf.empty:
        return "<p style='color:#94A3B8;padding:1rem'>AI confidence data not available.</p>"
    fig = px.histogram(conf, nbins=20, title="AI Extraction Confidence Score",
                       labels={"value": "Confidence", "count": "Products"},
                       color_discrete_sequence=["#7C3AED"])
    fig.add_vline(x=conf.mean(), line_dash="dash", line_color="#F59E0B",
                  annotation_text=f"Mean: {conf.mean():.2f}", annotation_font_color="#F59E0B")
    fig.update_layout(**DARK)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_price_buckets(df: pd.DataFrame) -> str:
    priced = df["price_midpoint"].dropna()
    if priced.empty:
        return "<p style='color:#94A3B8;padding:1rem'>No price data to display.</p>"
    bins   = [0, 500, 2000, 10000, 50000, 200000, float("inf")]
    labels = ["<₹500", "₹500–2K", "₹2K–10K", "₹10K–50K", "₹50K–2L", ">₹2L"]
    bucketed = pd.cut(priced, bins=bins, labels=labels, right=False).value_counts().sort_index()
    fig = px.bar(x=bucketed.index.astype(str), y=bucketed.values,
                 title="Products by Price Range",
                 labels={"x": "Price Range", "y": "Count"},
                 color=bucketed.values, color_continuous_scale="Blues")
    fig.update_layout(**DARK)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _chart_quality_hist(df: pd.DataFrame) -> str:
    if "quality_score" not in df.columns:
        return ""
    fig = px.histogram(df["quality_score"], nbins=10, title="Data Quality Score Distribution",
                       labels={"value": "Quality Score", "count": "Products"},
                       color_discrete_sequence=["#2563EB"])
    fig.update_layout(**DARK)
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ── Summary stats ─────────────────────────────────────────────────────────────

def _summary_stats(df: pd.DataFrame) -> dict:
    priced = df["price_midpoint"].dropna()
    return {
        "total": len(df),
        "unique_suppliers": int(df["supplier_name"].nunique()),
        "categories": int(df["category"].nunique()),
        "with_price": int(df["has_price"].sum()) if "has_price" in df.columns else 0,
        "ai_extracted": int(df["ai_extracted"].sum()) if "ai_extracted" in df.columns else 0,
        "avg_quality": f"{df['quality_score'].mean():.2f}" if "quality_score" in df.columns else "—",
        "median_price": f"₹{priced.median():,.0f}" if not priced.empty else "—",
    }


# ── Build Dashboard ───────────────────────────────────────────────────────────

def build_dashboard(df: pd.DataFrame, output_path: str | Path | None = None) -> Path:
    """
    Generate a self-contained interactive HTML EDA dashboard.

    Args:
        df         : cleaned DataFrame from etl.run_etl()
        output_path: where to save the HTML (default: data/exports/eda_report.html)

    Returns:
        Path to the saved HTML file
    """
    output_path = Path(output_path or EXPORTS_DIR / "eda_report.html")
    print(f"[EDA] Building dashboard for {len(df)} products...")

    stats = _summary_stats(df)
    insights = _generate_insights(df)

    # Render charts
    charts = {
        "category":    _chart_category_bar(df),
        "price_box":   _chart_price_box(df),
        "cities":      _chart_top_cities(df),
        "complete":    _chart_completeness(df),
        "ai_conf":     _chart_ai_confidence(df),
        "price_bkt":   _chart_price_buckets(df),
        "quality":     _chart_quality_hist(df),
    }

    stat_cards = [
        ("📦", "Total Products",    f"{stats['total']:,}"),
        ("🏭", "Unique Suppliers",  f"{stats['unique_suppliers']:,}"),
        ("📂", "Categories",        f"{stats['categories']}"),
        ("💰", "With Price",        f"{stats['with_price']:,}"),
        ("🤖", "AI Extracted",      f"{stats['ai_extracted']:,}"),
        ("⭐", "Avg Quality",       stats['avg_quality']),
        ("📊", "Median Price",      stats['median_price']),
    ]

    cards_html = "\n".join(f"""
        <div class="card">
            <div class="card-icon">{icon}</div>
            <div class="card-val">{val}</div>
            <div class="card-lbl">{label}</div>
        </div>""" for icon, label, val in stat_cards)

    insights_html = "\n".join(
        f'<div class="insight">💡 <span>{text}</span></div>' for text in insights
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Slooze · IndiaMART EDA Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body   {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0F172A; color: #F1F5F9; }}

  /* ── Header ── */
  .header {{ background: linear-gradient(135deg, #1E3A5F, #0F172A);
             padding: 2rem 2.5rem; border-bottom: 1px solid #2563EB44; }}
  .header h1 {{ font-size: 1.6rem; font-weight: 700; }}
  .header h1 span {{ color: #3B82F6; }}
  .header p  {{ color: #94A3B8; margin-top: 0.3rem; font-size: 0.9rem; }}

  /* ── Layout ── */
  .container {{ max-width: 1300px; margin: 0 auto; padding: 2rem 2.5rem; }}
  .section {{ margin-bottom: 2.5rem; }}
  .section-title {{ font-size: 1rem; font-weight: 600; color: #94A3B8;
                    text-transform: uppercase; letter-spacing: 0.06em;
                    margin-bottom: 1rem; padding-bottom: 0.4rem;
                    border-bottom: 1px solid #1E293B; }}

  /* ── Stat cards ── */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(155px, 1fr)); gap: 1rem; }}
  .card  {{ background: #1E293B; border: 1px solid #334155; border-radius: 10px;
            padding: 1.2rem 1rem; text-align: center; }}
  .card:hover {{ border-color: #3B82F6; }}
  .card-icon {{ font-size: 1.5rem; margin-bottom: 0.4rem; }}
  .card-val  {{ font-size: 1.5rem; font-weight: 700; color: #3B82F6; }}
  .card-lbl  {{ font-size: 0.72rem; color: #64748B; margin-top: 0.25rem;
                text-transform: uppercase; letter-spacing: 0.05em; }}

  /* ── Chart grid ── */
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.25rem; }}
  .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.25rem; }}
  .chart-box {{ background: #1E293B; border-radius: 10px; padding: 1rem;
                border: 1px solid #1E293B; overflow: hidden; }}
  .chart-box.full {{ grid-column: 1 / -1; }}

  /* ── Insights ── */
  .insight {{ background: #1E293B; border-left: 3px solid #3B82F6;
              border-radius: 0 8px 8px 0; padding: 0.75rem 1rem;
              margin-bottom: 0.6rem; font-size: 0.88rem;
              color: #CBD5E1; line-height: 1.6; display: flex; gap: 0.6rem; }}

  /* ── Footer ── */
  .footer {{ text-align: center; padding: 1.5rem; color: #475569;
              font-size: 0.8rem; border-top: 1px solid #1E293B; margin-top: 2rem; }}

  @media (max-width: 800px) {{
    .grid-2, .grid-3 {{ grid-template-columns: 1fr; }}
    .container {{ padding: 1rem; }}
  }}
</style>
</head>
<body>

<div class="header">
  <h1>🏭 <span>Slooze</span> · IndiaMART EDA Dashboard</h1>
  <p>B2B Marketplace Data Pipeline · Scraped from IndiaMART · AI-structured with Groq / Llama 3</p>
</div>

<div class="container">

  <!-- Stats -->
  <div class="section">
    <div class="section-title">Dataset Overview</div>
    <div class="cards">{cards_html}</div>
  </div>

  <!-- Insights -->
  <div class="section">
    <div class="section-title">Auto-Generated Insights</div>
    {insights_html}
  </div>

  <!-- Category + Price buckets -->
  <div class="section">
    <div class="section-title">Category & Pricing</div>
    <div class="grid-2">
      <div class="chart-box">{charts['category']}</div>
      <div class="chart-box">{charts['price_bkt']}</div>
    </div>
  </div>

  <!-- Price box plot -->
  <div class="section">
    <div class="section-title">Price Distribution</div>
    <div class="chart-box full">{charts['price_box']}</div>
  </div>

  <!-- Geography -->
  <div class="section">
    <div class="section-title">Regional Supplier Patterns</div>
    <div class="chart-box full">{charts['cities']}</div>
  </div>

  <!-- Data quality -->
  <div class="section">
    <div class="section-title">Data Quality</div>
    <div class="grid-2">
      <div class="chart-box">{charts['complete']}</div>
      <div class="chart-box">{charts['quality']}</div>
    </div>
  </div>

  <!-- AI extraction -->
  <div class="section">
    <div class="section-title">AI Extraction Quality (Groq / Llama 3)</div>
    <div class="chart-box">{charts['ai_conf']}</div>
  </div>

</div>

<div class="footer">
  Slooze B2B Data Engineering Pipeline · IndiaMART · Groq Llama 3 · Plotly
</div>

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[EDA] Dashboard saved -> {output_path}\n")
    return output_path