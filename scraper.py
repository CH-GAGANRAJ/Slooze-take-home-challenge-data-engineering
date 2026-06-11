"""
scraper.py  -  Slooze B2B Data Engineering Pipeline
====================================================
Target   : dir.indiamart.com/impcat/<keyword>.html
Strategy : Scrape the IMPCAT category pages which are fully server-side
           rendered (no login required, no JS needed).

  The impcat URL pattern returns 281KB+ of real product data in SSR HTML:
    https://dir.indiamart.com/impcat/<slug>.html          (page 1)
    https://dir.indiamart.com/impcat/<slug>.html?pg=2     (page 2)

  Product cards are <article class="template7-product-card"> elements
  containing product title (h2), price, supplier name, and location.

  Fallback: search.mp page with staticListingCard skeleton parsing.
"""

import time
import random
import json
import re
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# ── Constants ──────────────────────────────────────────────────────────────────

BASE_DIR = "https://dir.indiamart.com"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
]


def _slug(category: str) -> str:
    """Convert 'cotton fabric' -> 'cotton-fabric'."""
    return re.sub(r"\s+", "-", category.strip().lower())


def _impcat_url(category: str, page: int) -> str:
    slug = _slug(category)
    base = f"{BASE_DIR}/impcat/{slug}.html"
    return base if page == 1 else f"{base}?pg={page}"


def _make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer":         "https://www.indiamart.com/",
        "Connection":      "keep-alive",
        "DNT":             "1",
    })
    return s


# ── Card parser ────────────────────────────────────────────────────────────────

def _parse_cards(soup: BeautifulSoup, category: str, source_url: str) -> list[dict]:
    """
    Parse <article class='template7-product-card'> elements.
    These are the fully SSR-rendered product cards from the /impcat/ pages.
    """
    products: list[dict] = []

    cards = soup.select("article.template7-product-card")
    if not cards:
        # Fallback: any article or div with product-like class
        cards = soup.select(
            "article, .prd-unit, .product-unit, .improd-lst, "
            "[class*='product-card'], [class*='prd-sec']"
        )

    for card in cards:
        # ── Title (h2 inside card) ────────────────────────────────────────
        title_el = card.select_one("h2, h3, [class*='pName'], [class*='prod-name']")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        if not title or len(title) < 4:
            continue

        # ── URL ───────────────────────────────────────────────────────────
        link_el = card.select_one("a[href]")
        href = link_el["href"] if link_el else ""
        if href and not href.startswith("http"):
            href = BASE_DIR + href

        # ── Price ─────────────────────────────────────────────────────────
        price_el = card.select_one(
            "[class*='price'], [class*='prc'], [class*='Price'], "
            "[class*='rate'], [class*='Rate']"
        )
        price = price_el.get_text(separator=" ", strip=True) if price_el else ""

        # ── Supplier name ─────────────────────────────────────────────────
        supplier_el = card.select_one(
            "[class*='company']:not(button), [class*='comp-name']:not(button), "
            "[class*='cname']:not(button), [class*='seller']:not(button), "
            "[class*='mname'], [class*='sellerName']"
        )
        supplier = supplier_el.get_text(strip=True) if supplier_el else ""

        # ── Location ──────────────────────────────────────────────────────
        loc_el = card.select_one(
            "[class*='location'], [class*='city'], [class*='City'], "
            "[class*='loc'], [class*='place'], [class*='addr']"
        )
        location = loc_el.get_text(strip=True) if loc_el else ""

        # ── Description ───────────────────────────────────────────────────
        desc_el = card.select_one(
            "[class*='desc'], [class*='detail'], [class*='prd-info'], p"
        )
        desc = desc_el.get_text(strip=True) if desc_el else ""
        # Don't use price/supplier/location text as description
        if desc and (desc == price or desc == location or desc == supplier or len(desc) < 10):
            desc = ""

        # ── MOQ ───────────────────────────────────────────────────────────
        moq_el = card.select_one("[class*='moq'], [class*='MOQ'], [class*='min-qty']")
        moq = moq_el.get_text(strip=True) if moq_el else ""

        products.append({
            "title":             title,
            "description":       desc,
            "price_raw":         price,
            "supplier_name":     supplier,
            "supplier_location": location,
            "moq":               moq,
            "url":               href,
            "category":          category,
            "source":            "dir.indiamart",
            "scraped_at":        datetime.utcnow().isoformat(),
        })

    return products


def _parse_json_in_html(html: str, category: str) -> list[dict]:
    """
    Walk all JSON blobs embedded in <script> tags and extract products.
    """
    products: list[dict] = []

    def _normalize(item: dict) -> dict | None:
        if not isinstance(item, dict):
            return None
        title = (
            item.get("productName") or item.get("pName") or
            item.get("name") or item.get("title") or ""
        ).strip()
        if not title:
            return None
        url = item.get("productUrl") or item.get("prodUrl") or item.get("url") or ""
        if url and not url.startswith("http"):
            url = BASE_DIR + url
        return {
            "title":             title,
            "description":       (item.get("productDescription") or item.get("desc") or "").strip(),
            "price_raw":         str(item.get("priceRange") or item.get("price") or ""),
            "supplier_name":     (item.get("companyName") or item.get("cmpName") or "").strip(),
            "supplier_location": str(item.get("city") or item.get("state") or ""),
            "moq":               str(item.get("minOrderQuantity") or item.get("moq") or ""),
            "url":               url,
            "category":          category,
            "source":            "dir.indiamart",
            "scraped_at":        datetime.utcnow().isoformat(),
        }

    def _walk(node):
        if isinstance(node, list):
            for item in node:
                p = _normalize(item)
                if p:
                    products.append(p)
                else:
                    _walk(item)
        elif isinstance(node, dict):
            p = _normalize(node)
            if p:
                products.append(p)
                return
            for key in ("data", "products", "productList", "items", "result",
                        "list", "records", "searchResult", "searchData"):
                if key in node:
                    _walk(node[key])
                    if products:
                        return
            for v in node.values():
                if isinstance(v, (dict, list)):
                    _walk(v)

    # __NEXT_DATA__
    m = re.search(
        r'<script[^>]+id=["\']__NEXT_DATA__["\'][^>]*>(.*?)</script>',
        html, re.DOTALL
    )
    if m:
        try:
            data = json.loads(m.group(1))
            _walk(data)
        except Exception:
            pass

    # Generic inline arrays
    if not products:
        for arr_m in re.finditer(r'(\[\s*\{".*?\}\s*\])', html, re.DOTALL):
            try:
                chunk = json.loads(arr_m.group(1))
                _walk(chunk)
                if products:
                    break
            except Exception:
                continue

    return products


# ── TIER 1: /impcat/ SSR page ─────────────────────────────────────────────────

def _tier1_impcat(session: requests.Session, category: str,
                  page: int) -> list[dict]:
    """
    Best source: dir.indiamart.com/impcat/<slug>.html
    Full SSR, no login required, real product data.
    Tries multiple slug variants in order:
      full slug, 3-word, 2-word, first-word
    """
    parts = _slug(category).split("-")
    slug_variants = list(dict.fromkeys([
        "-".join(parts),          # full:  hydraulic-press-machine
        "-".join(parts[:3]),      # 3-word
        "-".join(parts[:2]),      # 2-word: hydraulic-press
        parts[0],                 # 1-word: hydraulic
    ]))

    for slug in slug_variants:
        base_url = f"{BASE_DIR}/impcat/{slug}.html"
        url = base_url if page == 1 else f"{base_url}?pg={page}"
        try:
            resp = session.get(url, timeout=20)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "lxml")

            products = _parse_cards(soup, category, url)
            if products:
                print(f"    [Tier 1 / impcat] {len(products)} products ({slug})")
                return products

            # Try embedded JSON
            products = _parse_json_in_html(resp.text, category)
            if products:
                print(f"    [Tier 1 / impcat JSON] {len(products)} products ({slug})")
                return products
        except Exception:
            continue

    print(f"    [Tier 1] No impcat page found for '{category}'")
    return []


# ── TIER 2: /search.mp SSR skeleton cards ─────────────────────────────────────

def _tier2_search_skeleton(session: requests.Session, query: str,
                            page: int, category: str) -> list[dict]:
    """
    dir.indiamart.com/search.mp returns 4 static skeleton cards server-side.
    Minimal data but real titles + city.
    """
    params = {"ss": query.replace("+", " ")}
    if page > 1:
        params["page"] = page
    try:
        resp = session.get(f"{BASE_DIR}/search.mp", params=params, timeout=15)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        cards = soup.select("section.staticListingCard")
        products: list[dict] = []
        for card in cards:
            h2 = card.select_one("h2, h3")
            if not h2:
                continue
            title = h2.get_text(strip=True)
            # Skip generic placeholders
            if any(g in title.lower() for g in ["verified", "bulk", "printed", "corporate"]):
                continue
            city_el = card.select_one(".staticSupplierBox strong")
            products.append({
                "title":             title,
                "description":       "",
                "price_raw":         "",
                "supplier_name":     "",
                "supplier_location": city_el.get_text(strip=True) if city_el else "",
                "moq":               "",
                "url":               "",
                "category":          category,
                "source":            "dir.indiamart",
                "scraped_at":        datetime.utcnow().isoformat(),
            })
        if products:
            print(f"    [Tier 2 / skeleton] {len(products)} products")
        return products
    except Exception as e:
        print(f"    [Tier 2] Failed: {e}")
        return []


# ── TIER 3: Playwright headless (full render) ──────────────────────────────────

def _tier3_playwright(category: str, page: int) -> list[dict]:
    """Full headless Chromium on the /impcat/ URL for complete React-rendered data."""
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        return []

    url = _impcat_url(category, page)
    products: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        ctx = browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": 1366, "height": 768},
        )
        pg = ctx.new_page()
        try:
            pg.goto(url, timeout=30000, wait_until="networkidle")
            pg.wait_for_timeout(2000)
            html = pg.content()
            soup = BeautifulSoup(html, "lxml")
            products = _parse_cards(soup, category, url)
            if not products:
                products = _parse_json_in_html(html, category)
        except Exception as e:
            print(f"    [Tier 3] Failed: {e}")
        finally:
            ctx.close()
            browser.close()

    if products:
        print(f"    [Tier 3 / playwright] {len(products)} products")
    return products


# ── Public API ─────────────────────────────────────────────────────────────────

def scrape_indiamart(category: str, max_pages: int = 2) -> list[dict]:
    """
    Scrape IndiaMART Directory for a given category.

    Uses a 3-tier cascade:
      Tier 1 - /impcat/<slug>.html  (fully SSR, real data, no login)
      Tier 2 - /search.mp skeleton  (minimal fallback with real titles)
      Tier 3 - Playwright headless  (full JS render if tiers 1+2 fail)

    Args:
        category  : search keyword e.g. "cotton fabric", "hydraulic press machine"
        max_pages : number of result pages to scrape

    Returns:
        list of product dicts in the pipeline schema
    """
    all_products: list[dict] = []
    query = category.replace(" ", "+")

    print(f"\n[Scraper] Category: '{category}' | Pages: {max_pages}")
    session = _make_session()

    for page_num in range(1, max_pages + 1):
        print(f"  Fetching page {page_num}...")
        products: list[dict] = []

        # Tier 1 — /impcat/ SSR (primary, most complete)
        products = _tier1_impcat(session, category, page_num)

        # Tier 2 — search.mp skeleton cards
        if not products:
            products = _tier2_search_skeleton(session, query, page_num, category)

        # Tier 3 — Playwright full render
        if not products:
            print("    Launching headless browser (Tier 3)...")
            products = _tier3_playwright(category, page_num)

        if products:
            print(f"  -> {len(products)} products on page {page_num}")
            all_products.extend(products)
        else:
            print(f"  No products found on page {page_num}")
            if page_num == 1:
                break  # If page 1 fails, stop — later pages won't help

        if page_num < max_pages:
            delay = random.uniform(2.0, 4.0)
            print(f"  Waiting {delay:.1f}s...")
            time.sleep(delay)

    print(f"[Scraper] Done. Total: {len(all_products)} products\n")
    return all_products