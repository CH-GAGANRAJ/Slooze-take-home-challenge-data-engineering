"""
etl.py
Clean, normalize, and deduplicate scraped product data.
Input : list of raw dicts (with ai_fields added by ai_extractor.py)
Output: clean pandas DataFrame saved as CSV + JSON
"""

import re
import json
import hashlib
from pathlib import Path

import pandas as pd

EXPORTS_DIR = Path("data/exports")
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_price(raw: str) -> tuple[float | None, float | None]:
    """Extract min and max price from strings like '₹1,500 - ₹3,000'."""
    if not raw:
        return None, None
    raw = raw.replace(",", "")
    nums = re.findall(r"[\d]+(?:\.\d+)?", raw)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), float(nums[0])
    return None, None


def _clean_location(raw: str) -> tuple[str, str]:
    """
    Split 'Mumbai, Maharashtra, India' → (city='Mumbai', state='Maharashtra')
    Returns ('', '') if unparseable.
    """
    if not raw:
        return "", ""
    parts = [p.strip() for p in raw.split(",")]
    city = parts[0] if len(parts) >= 1 else ""
    state = parts[1] if len(parts) >= 2 else ""
    # Strip noise words
    noise = re.compile(r"\b(india|pvt|ltd|private|limited)\b", re.IGNORECASE)
    city = noise.sub("", city).strip()
    state = noise.sub("", state).strip()
    return city, state


def _fingerprint(title: str, supplier: str) -> str:
    """SHA256 hash for exact-duplicate detection."""
    key = f"{title.lower().strip()}::{supplier.lower().strip()}"
    return hashlib.sha256(key.encode()).hexdigest()


def _quality_score(row: pd.Series) -> float:
    """Simple 0–1 completeness score."""
    score = 0.0
    if len(str(row.get("title", ""))) > 10:      score += 0.25
    if row.get("has_description"):                score += 0.20
    if row.get("price_min") is not None:          score += 0.20
    if row.get("supplier_name"):                  score += 0.15
    if row.get("supplier_city"):                  score += 0.10
    if row.get("ai_extracted"):                   score += 0.10
    return round(score, 2)


# ── Main ETL ──────────────────────────────────────────────────────────────────

def run_etl(products: list[dict]) -> pd.DataFrame:
    """
    Clean, normalize, deduplicate, and export the product list.

    Returns a clean pandas DataFrame.
    """
    print(f"[ETL] Processing {len(products)} raw products...")

    rows = []
    for p in products:
        # ── Basic text fields ─────────────────────────────────────────────
        title = p.get("title", "").strip()
        description = p.get("description", "").strip()

        # ── Price ─────────────────────────────────────────────────────────
        price_min, price_max = _parse_price(p.get("price_raw", ""))
        price_mid = ((price_min + price_max) / 2) if price_min and price_max else price_min

        # ── Location ──────────────────────────────────────────────────────
        city, state = _clean_location(p.get("supplier_location", ""))

        # ── AI fields (flatten top-level) ─────────────────────────────────
        ai = p.get("ai_fields", {}) or {}
        ai_extracted = p.get("ai_extracted", False)

        # ── Derived flags ─────────────────────────────────────────────────
        has_desc = len(description.split()) >= 5
        has_price = price_min is not None

        row = {
            # Core
            "title":              title,
            "description":        description,
            "category":           p.get("category", ""),
            "source":             p.get("source", "indiamart"),
            "url":                p.get("url", ""),
            "scraped_at":         p.get("scraped_at", ""),
            # Price
            "price_raw":          p.get("price_raw", ""),
            "price_min":          price_min,
            "price_max":          price_max,
            "price_midpoint":     price_mid,
            # Supplier
            "supplier_name":      p.get("supplier_name", "").strip(),
            "supplier_location":  p.get("supplier_location", "").strip(),
            "supplier_city":      city,
            "supplier_state":     state,
            "moq":                p.get("moq", ""),
            # Flags
            "has_description":    has_desc,
            "has_price":          has_price,
            # AI fields
            "ai_extracted":       ai_extracted,
            "ai_confidence":      ai.get("confidence_score"),
            "ai_product_type":    ai.get("product_type"),
            "ai_industry":        ai.get("industry"),
            "ai_material":        ai.get("material"),
            "ai_application":     ai.get("application"),
            "ai_capacity":        ai.get("capacity"),
            "ai_power_source":    ai.get("power_source"),
            "ai_fiber_content":   ai.get("fiber_content"),
            "ai_purity":          ai.get("purity"),
            # Fingerprint for dedup
            "_fp":                _fingerprint(title, p.get("supplier_name", "")),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Deduplicate ───────────────────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["_fp"]).drop(columns=["_fp"])
    print(f"[ETL] Deduplication: {before} -> {len(df)} ({before - len(df)} removed)")

    # ── Quality score ─────────────────────────────────────────────────────────
    df["quality_score"] = df.apply(_quality_score, axis=1)

    # ── Export ────────────────────────────────────────────────────────────────
    csv_path = EXPORTS_DIR / "products.csv"
    json_path = EXPORTS_DIR / "products.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2, default_handler=str)

    print(f"[ETL] Saved -> {csv_path}")
    print(f"[ETL] Saved -> {json_path}")
    print(f"[ETL] Final: {len(df)} clean products | Avg quality: {df['quality_score'].mean():.2f}\n")

    return df