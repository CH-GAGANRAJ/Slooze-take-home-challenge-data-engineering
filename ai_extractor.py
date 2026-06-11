"""
ai_extractor.py
Uses Groq API (Llama 3) to turn messy product descriptions
into clean, structured JSON fields.

Flow:
    Raw title + description
            ↓
       Groq / Llama 3
            ↓
    Structured JSON fields
            ↓
    Merged back into product record
"""

import json
import time
import os
import re

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Compatibility patch: groq 0.9.0 passes 'proxies' to httpx which removed it in 0.28 ──
try:
    import httpx
    _orig_init = httpx.Client.__init__
    def _patched_init(self, *args, **kwargs):
        kwargs.pop("proxies", None)
        _orig_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_init
except Exception:
    pass


# Category → fields we want to extract
CATEGORY_FIELDS = {
    "industrial machinery": ["product_type", "capacity", "power_source", "voltage", "automation_level", "material", "application", "certifications"],
    "electronics":          ["product_type", "operating_voltage", "current_rating", "interface", "package_type", "operating_temp", "application"],
    "textiles":             ["product_type", "fiber_content", "weave_type", "gsm", "width_cm", "color", "usage"],
    "chemicals":            ["product_type", "purity", "grade", "physical_state", "packaging", "application"],
    "agricultural":         ["product_type", "power_hp", "fuel_type", "working_width", "crop_type", "application"],
}
DEFAULT_FIELDS = ["product_type", "industry", "material", "application", "key_features"]


def _get_fields_for_category(category: str) -> list[str]:
    cat = category.lower()
    for key, fields in CATEGORY_FIELDS.items():
        if key in cat:
            return fields
    return DEFAULT_FIELDS


def _build_prompt(products: list[dict]) -> str:
    """Build a batch prompt — send multiple products in one API call."""
    fields = _get_fields_for_category(products[0].get("category", ""))
    fields_str = ", ".join(fields)

    items_text = ""
    for i, p in enumerate(products, 1):
        title = p.get("title", "")
        desc = p.get("description", "")
        price = p.get("price_raw", "")
        combined = f"{title}. {desc}. Price: {price}".strip(". ")
        items_text += f"{i}. {combined}\n"

    prompt = f"""You are a B2B data extraction assistant.

Extract structured fields from these product descriptions.
Return ONLY a valid JSON array with exactly {len(products)} objects — one per product, in the same order.
No explanation, no markdown, just the JSON array.

Fields to extract for each product:
{fields_str}, confidence_score (float 0.0-1.0)

Set any field you cannot determine to null.

Products:
{items_text}
"""
    return prompt


def _clean_json(text: str) -> list | None:
    """Strip markdown fences and parse JSON."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None


def extract_fields(products: list[dict], batch_size: int = 5) -> list[dict]:
    """
    Run AI extraction on a list of products.
    Sends them in small batches to stay within token limits.

    Args:
        products  : list of raw product dicts from scraper
        batch_size: how many products per Groq API call

    Returns:
        Same list, each product now has an 'ai_fields' key added.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        print("[AI Extractor] GROQ_API_KEY not set - skipping AI extraction")
        for p in products:
            p["ai_fields"] = {}
            p["ai_extracted"] = False
        return products

    client = Groq(api_key=api_key)
    total = len(products)
    print(f"[AI Extractor] Extracting {total} products (batch_size={batch_size})")

    for i in range(0, total, batch_size):
        batch = products[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = -(-total // batch_size)  # ceiling division
        print(f"  Batch {batch_num}/{total_batches} ({len(batch)} products)...")

        try:
            prompt = _build_prompt(batch)

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",  # replaces decommissioned llama3-8b-8192
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,              # low temp = deterministic structured output
                max_tokens=1500,
            )

            raw_text = response.choices[0].message.content
            results = _clean_json(raw_text)

            if results and len(results) == len(batch):
                for product, ai_result in zip(batch, results):
                    product["ai_fields"] = ai_result if isinstance(ai_result, dict) else {}
                    product["ai_extracted"] = bool(product["ai_fields"])
            else:
                print(f"  Warning: unexpected response format for batch {batch_num}")
                for p in batch:
                    p["ai_fields"] = {}
                    p["ai_extracted"] = False

        except Exception as e:
            print(f"  Error in batch {batch_num}: {e}")
            for p in batch:
                p["ai_fields"] = {}
                p["ai_extracted"] = False

        # Small pause between Groq calls
        if i + batch_size < total:
            time.sleep(0.5)

    success = sum(1 for p in products if p.get("ai_extracted"))
    print(f"[AI Extractor] Done. Extracted: {success}/{total}\n")
    return products