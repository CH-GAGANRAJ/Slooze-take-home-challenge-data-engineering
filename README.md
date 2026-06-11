# Slooze B2B Data Engineering Pipeline

This project scrapes product listings from IndiaMART, optionally enriches the raw product text with Groq/Llama 3, cleans the data with an ETL step, and generates an interactive EDA dashboard.

## Project Files

- `main.py` - single entry point for the full pipeline.
- `scraper.py` - scrapes IndiaMART product listings.
- `ai_extractor.py` - uses Groq to extract structured product fields.
- `etl.py` - cleans, normalizes, deduplicates, and exports product data.
- `eda.py` - builds the HTML dashboard using Plotly.
- `requirements.txt` / `req.txt` - Python dependencies.
- `.env.example` - template for local environment variables.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r req.txt
```

You can also use the conventional filename:

```bash
pip install -r requirements.txt
```

If you want the AI enrichment step, add your Groq key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

If no valid Groq key is present, the pipeline still runs, but AI extraction is skipped.

Do not commit `.env` to GitHub. It should stay local only.

## Run

Run the full pipeline:

```bash
python main.py
```

Scrape fewer or more pages per category:

```bash
python main.py --pages 3
```

Scrape selected categories:

```bash
python main.py --categories "cotton fabric" "hydraulic press machine"
```

## Outputs

The pipeline creates these generated files:

- `data/raw/scraped_products.json` - raw scraped product records.
- `data/exports/products.csv` - cleaned tabular dataset.
- `data/exports/products.json` - cleaned JSON dataset.
- `data/exports/eda_report.html` - interactive EDA dashboard.

These output files are generated at runtime and are not required for the code to run.

## How The IndiaMART Scraper Works

The scraper uses the public IndiaMART directory domain:

```text
https://dir.indiamart.com/impcat/<category-slug>.html
```

For example, `cotton fabric` becomes:

```text
https://dir.indiamart.com/impcat/cotton-fabric.html
```

For page 2 and later, it adds a page query:

```text
https://dir.indiamart.com/impcat/cotton-fabric.html?pg=2
```

The scraping flow is:

1. `main.py` calls `scrape_indiamart(category, max_pages)`.
2. `scraper.py` creates a `requests.Session` with browser-like headers.
3. For each category and page, it first tries IndiaMART `/impcat/` category pages.
4. It parses server-rendered HTML product cards, mainly `article.template7-product-card`.
5. From each card it extracts title, URL, price, supplier name, location, description, MOQ, category, source, and timestamp.
6. If normal product cards are not found, it tries embedded JSON inside script tags.
7. If `/impcat/` fails, it falls back to `dir.indiamart.com/search.mp`.
8. If static scraping fails and Playwright is installed, it can open the page in a headless browser and parse the rendered HTML.

The scraper is intentionally built as a three-tier fallback:

- Tier 1: `/impcat/<slug>.html` server-rendered pages. This is the main source.
- Tier 2: `/search.mp?ss=<query>` skeleton cards. This gives limited fallback data.
- Tier 3: Playwright headless browser rendering. This is used only when the first two tiers fail.

## Data Pipeline

After scraping:

1. `ai_extractor.py` batches products and asks Groq/Llama 3 for structured fields.
2. `etl.py` parses price ranges, cleans locations, removes duplicates, computes data quality, and exports CSV/JSON.
3. `eda.py` creates charts and summary insights in `eda_report.html`.
