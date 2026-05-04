# card_stuff

A collection of standalone Python scripts for managing a sports card selling operation.

---

## Scripts

### `ai_text_me.py`
Sends a daily iMessage via Claude API and macOS Shortcuts. Runs on cron.

```bash
python ai_text_me.py            # send daily text (fact, haiku, history, science)
python ai_text_me.py --card     # send a random card photo with a player fact
python ai_text_me.py --test     # dry run: print card filename + caption, no send
```

**Cron schedule:**
- 7:00am daily — daily text
- 9:15pm Mon/Thu/Sat — card + fact

**Card facts** rotate across 10 categories (early life, season stats, records, quirky stories, etc.) and track the last 20 facts per player in `~/.card_facts_history.json` to avoid repeats. Edit that file directly to seed or remove facts.

Requires: `anthropic`, `typer`
Credentials: `ANTHROPIC_API_KEY` env var or `.anthropic-api-key.txt` in repo root or `~`

---

### `international_shipping.py`
Detects the destination country from a shipping label photo and copies an appropriate iMessage to the clipboard.

```bash
python international_shipping.py              # uses newest image in ~/Downloads
python international_shipping.py photo.heic   # specific file
python international_shipping.py --plural     # use plural phrasing
```

Converts the image via `sips`, sends to OpenAI Vision to detect the country, renames the output as `YYYY-MM-DD_COUNTRY.jpg` in `/Volumes/Dutton 2TB/Sales/shipping/`, and moves the original to `processed/` if confidence > 0.75.

Requires: `openai`, `pyperclip`
Credentials: `OPENAI_API_KEY` env var or `.openai-api-key.txt`

---

### `ebay_tracker.py`
Tracks eBay purchases by scanning Gmail. Persists state to `~/.ebay_tracker.json`.

```bash
python ebay_tracker.py           # interactive mode (offline, reads state file)
python ebay_tracker.py --cron    # fetch Gmail, update state, mark read
python ebay_tracker.py --refresh # fetch then go interactive
python ebay_tracker.py --init    # one-time deep scan of all eBay emails
```

Credentials: Gmail OAuth via `~/.gmail-mcp/gcp-oauth.keys.json`, token cached at `~/.gmail-mcp/ebay-tracker-token.json`

---

### `ebay_api_demo.py`
Minimal starter for the eBay Developer API. Fetches an app token and searches public listings through the Browse API.

```bash
python ebay_api_demo.py token
python ebay_api_demo.py search "Nikola Jokic Select"
python ebay_api_demo.py search "Caitlin Clark Prizm" --limit 10
python ebay_api_demo.py search "Michael Jordan Fleer" --sandbox
```

Credentials: `EBAY_CLIENT_ID` and `EBAY_CLIENT_SECRET` in env vars or `.env`

---

### `collectors_update.py`
Bulk-updates My Cost, Source, My Notes, and Date Acquired on app.collectors.com for cards in a CSV matched by PSA cert number.

```bash
python collectors_update.py            # update all cards in CSV
python collectors_update.py --dry-run  # print plan without saving
```

CSV path is hardcoded to `My Collection YYYYMMDD.csv` in the repo root — update the filename when exporting a new one. Uses Playwright + `playwright-stealth` to bypass Cloudflare.

Requires: `pip install playwright playwright-stealth python-dotenv` and `playwright install chromium`
Credentials: `PSA_USER` and `PSA_PASS` in `.env`

---

### `psa_download.py`
Downloads front and back card images from PSA's public API by cert number.

```bash
python psa_download.py fetch <cert_number>
python psa_download.py fetch <cert> --out-dir /path/to/dir --verbose
```

Uses `curl` by default (bypasses WAF) with `requests` as fallback. Downloads and resizes images to 75% at quality 65.

Credentials: `--token` arg, or `.psa-token.txt` in repo root or `~/.psa-token.txt`

---

### `card_merge.py`
Merges card front/back photos into a side-by-side grid image.

```bash
python card_merge.py card*.jpg    # merge odd-numbered files (fronts)
python card_merge.py -e card*.jpg # merge even-numbered files (backs)
```

Adjusts the merged image's timestamp so it sorts as the newest file.

---

### `cropper.py`
Auto-rotates and crops card images.

```bash
python cropper.py -i "*.jpg"       # process all JPGs
python cropper.py -i file.jpg -o   # overwrite in place
```

---

### `sale.py`
Creates and prints the current month's sales directory path.

```bash
python sale.py                # current month
python sale.py --date 2024-03 # specific month
```

---

### `card-search.html`
A local HTML tool for searching cards. Open directly in a browser.

---

## GUI Apps

All GUI apps are PyQt5 desktop apps with a `.app` wrapper for macOS.

### `Card Namer App/card_namer.py`
Batch-renames card image files using OCR (pytesseract) + OpenAI Vision.

Credentials: `OPENAI_API_KEY` env var or `.openai-api-key.txt`

### `Ebay_Titles/ebay_title_gui.py`
Generates optimized eBay listing titles for cards.

### `Letter Track App/lettertrack.py`
Tracks physical mail with TinyURL-shortened tracking links.

### `Envelope Print App/envelope-print.py`
Prints envelopes with address formatting.

---

## `psa_update/`

Helper scripts for filling in missing purchase cost data on cards.

- **`search_costs.py`** — searches Gmail for eBay purchase emails matching cards in `missingcost.csv`, extracts price + shipping + tax, writes `missingcost_with_email_total.csv`
- **`match_ebay_report.py`** — matches cards in `missingcost_with_email_total.csv` against an eBay purchase history HTML export

---

## Credentials summary

| Script | Credential | Where |
|---|---|---|
| `ai_text_me.py` | `ANTHROPIC_API_KEY` | env var or `.anthropic-api-key.txt` |
| `international_shipping.py`, `Card Namer App` | `OPENAI_API_KEY` | env var or `.openai-api-key.txt` |
| `psa_download.py` | PSA token | `--token` arg or `.psa-token.txt` |
| `ebay_tracker.py`, `psa_update/search_costs.py` | Gmail OAuth | `~/.gmail-mcp/gcp-oauth.keys.json` |
| `collectors_update.py` | `PSA_USER`, `PSA_PASS` | `.env` in repo root |
