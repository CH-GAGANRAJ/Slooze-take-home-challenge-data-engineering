"""
main.py
Single entry point for the Slooze B2B Data Engineering Pipeline.

Usage:
  # Full pipeline (scrape -> AI extract -> ETL -> EDA)
  python main.py

  # Control pages per category
  python main.py --pages 3

  # Scrape only specific categories
  python main.py --categories "hydraulic press" "cotton fabric"
"""

import argparse
import json
import sys
from pathlib import Path

from scraper       import scrape_indiamart
from ai_extractor  import extract_fields
from etl           import run_etl
from eda           import build_dashboard

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Default categories to scrape
DEFAULT_CATEGORIES = [
    "hydraulic press machine",
    "cnc lathe machine",
    "cotton fabric",
    "industrial sensor",
    "agricultural tractor",
]


def run_pipeline(categories: list[str], max_pages: int):

    print("=" * 60)
    print("  Slooze B2B Data Engineering Pipeline")
    print("  Source: IndiaMART Directory | AI: Groq / Llama 3")
    print("=" * 60)

    # ── Step 1: Scrape ─────────────────────────────────────────────────────────
    print(f"\n[Step 1/4] Scraping IndiaMART "
          f"({len(categories)} categories, {max_pages} pages each)...")

    raw_products: list[dict] = []
    for category in categories:
        products = scrape_indiamart(category, max_pages=max_pages)
        raw_products.extend(products)

    if not raw_products:
        print("\n[!] No products scraped. Check your network or try fewer categories.")
        sys.exit(1)

    # Save raw data
    raw_path = RAW_DIR / "scraped_products.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_products, f, indent=2, ensure_ascii=False)
    print(f"  Raw data saved -> {raw_path}")
    print(f"  Total raw products: {len(raw_products)}")

    # ── Step 2: AI Extraction ──────────────────────────────────────────────────
    print("\n[Step 2/4] Running AI extraction (Groq / Llama 3)...")
    enriched = extract_fields(raw_products, batch_size=5)

    # ── Step 3: ETL ────────────────────────────────────────────────────────────
    print("[Step 3/4] Running ETL (clean + deduplicate + export)...")
    df = run_etl(enriched)

    if df.empty:
        print("\n[!] ETL produced an empty DataFrame. Nothing to visualise.")
        sys.exit(1)

    # ── Step 4: EDA ────────────────────────────────────────────────────────────
    print("[Step 4/4] Building EDA dashboard...")
    dashboard_path = build_dashboard(df)

    # ── Done ───────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Pipeline complete!")
    print(f"  Products   : {len(df)}")
    print(f"  CSV        : data/exports/products.csv")
    print(f"  JSON       : data/exports/products.json")
    print(f"  Dashboard  : {dashboard_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slooze B2B Data Pipeline")
    parser.add_argument(
        "--categories", nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Search categories/keywords to scrape from IndiaMART",
    )
    parser.add_argument(
        "--pages", type=int, default=2,
        help="Number of search result pages per category (default: 2)",
    )
    args = parser.parse_args()

    run_pipeline(categories=args.categories, max_pages=args.pages)