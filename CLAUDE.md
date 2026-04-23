# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A collection of standalone Python scripts for managing a sports card selling operation — image processing, PSA grading downloads, eBay purchase tracking, international shipping, and utility tools.

## Running scripts

Each script is standalone and run directly:

```bash
python international_shipping.py              # uses newest ~/Downloads image
python international_shipping.py photo.heic   # specific file
python international_shipping.py --plural     # use plural message

python psa_download.py fetch <cert_number>    # downloads front+back card images
python psa_download.py fetch <cert> --out-dir /path/to/dir --verbose

python ebay_tracker.py                        # interactive mode (offline, reads state file)
python ebay_tracker.py --cron                 # fetch Gmail, update state, mark read
python ebay_tracker.py --refresh              # fetch then go interactive
python ebay_tracker.py --init                 # one-time deep scan of all eBay emails

python sale.py                                # print/create current month's sales dir
python sale.py --date 2024-03                 # specific month

python card_merge.py card*.jpg                # merge card images into grid
python card_merge.py -e card*.jpg             # use even-numbered files

python cropper.py -i "*.jpg"                  # auto-rotate/crop card images
python cropper.py -i file.jpg -o              # overwrite in place

python collectors_update.py                   # update all cards in CSV on collectors.com
python collectors_update.py --dry-run         # print plan without saving changes
```

## API keys / credentials

Scripts look for credentials in this priority order:

- **OpenAI** (`international_shipping.py`, `Card Namer App/card_namer.py`): `OPENAI_API_KEY` env var, or `.openai-api-key.txt` in repo root
- **Anthropic** (`ai_text_me.py`): `ANTHROPIC_API_KEY` env var, or `.anthropic-api-key.txt` in repo/home
- **PSA** (`psa_download.py`): `--token` arg, or `.psa-token.txt` in repo root or `~/.psa-token.txt`
- **Gmail/eBay tracker** (`ebay_tracker.py`): OAuth via `~/.gmail-mcp/gcp-oauth.keys.json`, token cached at `~/.gmail-mcp/ebay-tracker-token.json`
- **Collectors.com** (`collectors_update.py`): `PSA_USER` and `PSA_PASS` in `.env` in repo root

## Architecture notes

**No shared library** — each script is fully self-contained with its own helpers. Don't extract shared utilities across scripts.

**State file** (`~/.ebay_tracker.json`): `ebay_tracker.py` persists purchase state here. The cron mode is meant to run periodically to keep it fresh; interactive mode reads it offline.

**Image pipeline** (`international_shipping.py`):
1. Converts source image (HEIC/PNG/etc) to JPEG via macOS `sips`
2. Sends to OpenAI vision API to detect destination country
3. Renames output as `YYYY-MM-DD_COUNTRY.jpg` in `/Volumes/Dutton 2TB/Sales/shipping/`
4. Copies shipping message to clipboard
5. Moves original HEIC to `processed/` if confidence > 0.75

**PSA downloader** (`psa_download.py`): Hits two PSA public APIs (cert data + images). Uses `curl` by default (bypasses WAF) with `requests` fallback. Downloads front/back, resizes to 75% at quality 65.

**GUI apps** (`Card Namer App/`, `Ebay_Titles/`, `Letter Track App/`): PyQt5 desktop apps. `card_namer.py` uses OCR (pytesseract) + OpenAI to batch-rename card image files. `ebay_title_gui.py` generates eBay listing titles. `lettertrack.py` tracks physical mail via TinyURL-shortened tracking links.

**`card_merge.py`**: Combines front/back card photos into a grid. Defaults to odd-numbered files (fronts); `-e` flag selects even-numbered files (backs). Adjusts timestamps so the merged image sorts as newest.

**`ai_text_me.py`**: Runs via macOS cron, calls Claude API, sends result as iMessage via macOS Shortcuts.
- **Default (no flags)**: picks a random prompt from `PROMPT_QUESTIONS` (birthday, history, science, haiku) and sends via "Send Message" shortcut. Cron: 7am daily.
- **`--card`**: two-step Claude call — (1) identify player name from a random card image in `/Volumes/Dutton 2TB/Cards/Mix`, (2) fetch a categorized fact avoiding the last 20 facts sent for that player. Imports image into Photos "cards" album, runs "Text Latest Image" shortcut to send the photo, then sends the caption via "Send Card Message" shortcut. Falls back to text message if drive is unmounted or Photos fails. Cron: 9:15pm Mon/Thu/Sat.
- **`--test`**: dry run — picks a card and generates a caption, prints `filename\ncaption` to stdout without sending anything. Still saves the fact to history.
- **Fact history**: `~/.card_facts_history.json` stores up to 20 facts per player (keyed by lowercase name). Edit this file directly to seed or remove facts.
- **Fact categories** (`FACT_CATEGORIES`): rotates across early life, rookie season, specific season stats, records, personal life, quirky stories, pre-pro background, overlooked moments, adversity, and trades/signings.

**`collectors_update.py`**: Logs into app.collectors.com via Playwright and bulk-updates My Cost, Source, My Notes, and Date Acquired for cards listed in a CSV (matched by PSA cert number). Login is a two-step flow (email → Verify button → password → submit) at `/signin`. The cert→internal item ID mapping is built by intercepting API responses when the collection page loads — no need to return to the list between cards. Uses `playwright-stealth` to bypass Cloudflare. CSV path is hardcoded to `My Collection YYYYMMDD.csv` in the repo root; update the filename when exporting a new one. Requires `pip install playwright playwright-stealth python-dotenv` and `playwright install chromium`.
