#!/usr/bin/env python3
"""
collectors_update.py

Logs into app.collectors.com and updates My Cost, Source, My Notes, and
Date Acquired for each card in the CSV, matched by PSA cert number.

Requirements:
    pip install playwright python-dotenv
    playwright install chromium

Usage:
    python collectors_update.py
    python collectors_update.py --dry-run              # print plan without making changes
    python collectors_update.py --csv path/to/file.csv # use a different CSV
"""

import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from playwright_stealth import Stealth

_default_csv = Path(__file__).parent / "My Collection 20260406.csv"
CSV_PATH = Path(sys.argv[sys.argv.index("--csv") + 1]) if "--csv" in sys.argv else _default_csv
DATE_ACQUIRED_FALLBACK = date.today().strftime("%Y-%m-%d")
BASE_URL = "https://app.collectors.com"

DRY_RUN = "--dry-run" in sys.argv


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

def login(page, username: str, password: str) -> None:
    Stealth().apply_stealth_sync(page)
    print("Navigating to signin page...")
    page.goto(f"{BASE_URL}/signin", wait_until="load", timeout=60_000)
    print(f"  URL: {page.url}")

    # Step 1: fill email and submit (two-step login flow)
    for selector in ['input[type="email"]', 'input[name="email"]', 'input[name="username"]', "input:first-of-type"]:
        try:
            el = page.locator(selector).first
            el.wait_for(state="visible", timeout=15_000)
            el.fill(username)
            break
        except PWTimeout:
            continue

    # Click Continue/Next or press Enter to advance to the password step
    advanced = False
    for selector in ['button:has-text("Continue")', 'button:has-text("Next")', 'button[type="submit"]', 'input[type="submit"]']:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=2000):
                btn.click()
                advanced = True
                break
        except PWTimeout:
            continue
    if not advanced:
        page.keyboard.press("Enter")

    # Step 2: wait for password field, then fill it
    pw = page.locator('input[type="password"]')
    pw.wait_for(state="visible", timeout=15_000)
    pw.fill(password)

    for selector in ['button:has-text("Verify")', 'button[type="submit"]', 'input[type="submit"]', 'button:has-text("Sign in")', 'button:has-text("Log in")', 'button:has-text("Login")']:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=2000):
                btn.click()
                break
        except PWTimeout:
            continue

    # Wait until we leave the signin page (up to 30s)
    try:
        page.wait_for_url(lambda url: "/signin" not in url, timeout=30_000)
    except PWTimeout:
        print("  WARNING: still on signin page after submit — check credentials or Cloudflare challenge")
    print(f"  Logged in — URL: {page.url}")


# ---------------------------------------------------------------------------
# Build cert → item_id map by intercepting collection API responses
# ---------------------------------------------------------------------------

def capture_cert_to_id(page) -> dict[str, str]:
    """
    Load the collection page and sniff XHR/fetch responses to extract
    cert number → internal item ID mappings.
    """
    cert_to_id: dict[str, str] = {}
    api_hits: list[dict] = []

    def on_response(response):
        url = response.url
        if response.status != 200:
            return
        ct = response.headers.get("content-type", "")
        if "json" not in ct:
            return
        # Capture anything that looks like a collection data endpoint
        lower = url.lower()
        if any(kw in lower for kw in ("collection", "item", "cert", "card", "portfolio")):
            try:
                body = response.json()
                api_hits.append({"url": url, "body": body})
            except Exception:
                pass

    page.on("response", on_response)

    print("Loading collection page (intercepting API calls)...")
    page.goto(f"{BASE_URL}/collection", wait_until="load")

    # Wait for cards to appear in the DOM, then scroll to load all
    try:
        page.wait_for_selector("a[href*='/collection/']", timeout=15_000)
    except PWTimeout:
        pass
    page.wait_for_timeout(3_000)

    # Scroll until no new API responses arrive for 3 consecutive passes
    stale = 0
    while stale < 3:
        prev_hits = len(api_hits)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(6_000)
        if len(api_hits) == prev_hits:
            stale += 1
        else:
            stale = 0
            print(f"  Scrolled — {len(api_hits)} API responses captured so far...")

    page.remove_listener("response", on_response)

    print(f"  Captured {len(api_hits)} JSON API response(s)")
    for hit in api_hits:
        _walk_for_cert_id(hit["body"], cert_to_id)

    print(f"  Mapped {len(cert_to_id)} cert number(s) from API")

    # DOM fallback: scrape cert numbers and item links directly from the page.
    # Each card shows cert as "#149237658" and links to /collection/{item_id}
    if not cert_to_id:
        print("  Falling back to DOM scrape...")
        cert_to_id = _scrape_cert_ids_from_dom(page)
        print(f"  Mapped {len(cert_to_id)} cert number(s) from DOM")

    return cert_to_id


def _scrape_cert_ids_from_dom(page) -> dict[str, str]:
    """Extract cert→item_id by reading links and cert text directly from the page."""
    result: dict[str, str] = {}
    # Each collection item is a link to /collection/{id}
    # The cert number appears nearby as text like "#149237658"
    raw = page.evaluate("""() => {
        const items = [];
        document.querySelectorAll('a[href*="/collection/"]').forEach(a => {
            const href = a.getAttribute('href') || '';
            const match = href.match(/\\/collection\\/(\\d+)/);
            if (!match) return;
            const itemId = match[1];
            // Search this element and its parent subtree for a cert number
            const root = a.closest('li, tr, [class*="card"], [class*="item"], [class*="row"]') || a.parentElement;
            const text = root ? root.innerText : a.innerText;
            const certMatch = text.match(/#(\\d{7,})/);
            if (certMatch) items.push([certMatch[1], itemId]);
        });
        return items;
    }""")
    for cert, item_id in (raw or []):
        result[cert] = item_id
    return result


def _walk_for_cert_id(obj, result: dict[str, str]) -> None:
    """Recursively find objects that have both an ID and a cert number."""
    if isinstance(obj, dict):
        # Collect candidate values from this dict
        id_val = (
            obj.get("id")
            or obj.get("itemId")
            or obj.get("collectionItemId")
            or obj.get("collectionId")
        )
        cert_val = (
            obj.get("certNumber")
            or obj.get("cert_number")
            or obj.get("certNo")
            or obj.get("cert")
            or obj.get("psaCert")
            or obj.get("psa_cert")
        )
        if id_val and cert_val:
            result[str(cert_val).strip()] = str(id_val).strip()
        for v in obj.values():
            _walk_for_cert_id(v, result)
    elif isinstance(obj, list):
        for item in obj:
            _walk_for_cert_id(item, result)


# ---------------------------------------------------------------------------
# Fallback: search for cert on the collection page
# ---------------------------------------------------------------------------

def find_id_via_search(page, cert: str) -> str | None:
    """Try using the site's search box to locate a cert, return item ID."""
    for selector in [
        'input[placeholder*="search" i]',
        'input[type="search"]',
        '[aria-label*="search" i]',
        'input[placeholder*="cert" i]',
    ]:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=1500):
                el.clear()
                el.fill(cert)
                el.press("Enter")
                page.wait_for_load_state("load")
                break
        except PWTimeout:
            continue

    # Look for a /collection/{id} link in the resulting page
    for link in page.locator('a[href*="/collection/"]').all():
        href = link.get_attribute("href") or ""
        text = link.inner_text()
        if cert in text or cert in href:
            parts = [p for p in href.rstrip("/").split("/") if p.isdigit()]
            if parts:
                return parts[-1]
    return None


# ---------------------------------------------------------------------------
# Update a single item
# ---------------------------------------------------------------------------

def _debug_page_elements(page) -> None:
    """Print all interactive elements on the page to help identify selectors."""
    elements = page.evaluate("""() => {
        const results = [];
        document.querySelectorAll('input, textarea, select, button, [role="button"]').forEach(el => {
            if (el.offsetParent === null) return;  // skip hidden
            results.push({
                tag: el.tagName.toLowerCase(),
                type: el.type || '',
                name: el.name || '',
                placeholder: el.placeholder || '',
                ariaLabel: el.getAttribute('aria-label') || '',
                text: el.innerText?.trim().slice(0, 60) || '',
                classes: el.className?.slice(0, 80) || '',
            });
        });
        return results;
    }""")
    print("    DEBUG elements on page:")
    for el in (elements or []):
        print(f"      {el}")


def update_item(page, item_id: str, cost: str, source: str, notes: str, date_acquired: str = '') -> bool:
    url = f"{BASE_URL}/collection/{item_id}"
    print(f"    → {url}")
    page.goto(url, wait_until="load")

    # Wait for the React app to render the card form (dateAcquired is a reliable indicator)
    try:
        page.wait_for_selector('input[name="dateAcquired"], input[name*="cost" i], input[name*="note" i]', timeout=15_000)
    except PWTimeout:
        pass  # Will fall through to debug dump if nothing found

    # Try to click an Edit button / pencil icon
    for selector in [
        'button:has-text("Edit")',
        'a:has-text("Edit")',
        '[aria-label*="edit" i]',
        '[title*="edit" i]',
        '[data-testid*="edit"]',
        'button.edit',
    ]:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=1500):
                btn.click()
                page.wait_for_timeout(1_000)
                break
        except PWTimeout:
            continue

    filled_any = False

    # --- My Cost ---
    for selector in [
        'input[name*="cost" i]',
        'input[placeholder*="cost" i]',
        'input[label*="cost" i]',
        '[data-field*="cost" i] input',
        'input[name*="price" i]',
    ]:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=1500):
                el.fill(cost)
                filled_any = True
                break
        except PWTimeout:
            continue

    # --- Source (may be a text input or a <select>) ---
    for selector in [
        'select[name*="source" i]',
        'input[name*="source" i]',
        'input[placeholder*="source" i]',
        '[data-field*="source" i] input',
        '[data-field*="source" i] select',
    ]:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=1500):
                tag = el.evaluate("el => el.tagName.toLowerCase()")
                if tag == "select":
                    # Try exact match first, then partial
                    try:
                        el.select_option(label=source)
                    except Exception:
                        el.select_option(value=source)
                else:
                    el.fill(source)
                filled_any = True
                break
        except PWTimeout:
            continue

    # --- My Notes ---
    for selector in [
        'textarea[name*="note" i]',
        'input[name*="note" i]',
        'textarea[placeholder*="note" i]',
        '[data-field*="note" i] textarea',
        '[data-field*="note" i] input',
    ]:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=1500):
                el.fill(notes)
                filled_any = True
                break
        except PWTimeout:
            continue

    # --- Date Acquired ---
    _date_val = date_acquired if date_acquired and date_acquired != '-' else DATE_ACQUIRED_FALLBACK
    for selector in [
        'input[name="dateAcquired"]',
        'input[name*="acquired" i]',
        'input[name*="date" i]',
        'input[placeholder*="date" i]',
        '[data-field*="acquired" i] input',
        '[data-field*="date" i] input',
    ]:
        try:
            el = page.locator(selector).first
            if el.is_visible(timeout=1500):
                el.fill(_date_val)
                # Some date pickers need Tab to confirm
                el.press("Tab")
                filled_any = True
                break
        except PWTimeout:
            continue

    if not filled_any:
        _debug_page_elements(page)
        print(f"    WARNING: found no editable fields — page structure may differ from expected")
        return False

    # --- Save / Submit ---
    for selector in [
        'button[type="submit"]',
        'button:has-text("Save")',
        'button:has-text("Update")',
        'button:has-text("Done")',
        'button:has-text("Apply")',
    ]:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=1500):
                if not DRY_RUN:
                    btn.click()
                    page.wait_for_load_state("load")
                else:
                    print(f"    DRY RUN — would click: {selector}")
                return True
        except PWTimeout:
            continue

    print(f"    WARNING: could not find Save button")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()
    username = os.environ.get("PSA_USER") or os.environ.get("COLLECTORS_USER")
    password = os.environ.get("PSA_PASS") or os.environ.get("COLLECTORS_PASS")
    if not username or not password:
        sys.exit("ERROR: PSA_USER and PSA_PASS must be set in .env")

    rows = load_csv(CSV_PATH)
    print(f"Loaded {len(rows)} cards from CSV")
    if DRY_RUN:
        print("DRY RUN mode — no changes will be saved\n")

    results: dict[str, list[str]] = {"ok": [], "missing": [], "error": []}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=150)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()

        login(page, username, password)
        cert_to_id = capture_cert_to_id(page)

        for row in rows:
            cert   = row["Cert Number"].strip()
            cost          = row["My Cost"].strip()
            source        = row["Source"].strip()
            notes         = row["My Notes"].strip()
            date_acquired = row.get("Date Acquired", "").strip()

            print(f"\nCert {cert}  ({row['Item'][:60]})")

            item_id = cert_to_id.get(cert)
            if not item_id:
                print(f"  Not in API response — trying search fallback...")
                page.goto(f"{BASE_URL}/collection", wait_until="load")
                item_id = find_id_via_search(page, cert)

            if not item_id:
                print(f"  SKIP: could not find internal ID")
                results["missing"].append(cert)
                continue

            print(f"  item_id={item_id}  cost={cost}  source={source}  date={date_acquired or DATE_ACQUIRED_FALLBACK}")
            try:
                ok = update_item(page, item_id, cost, source, notes, date_acquired)
                if ok:
                    print(f"  OK")
                    results["ok"].append(cert)
                else:
                    results["error"].append(cert)
            except Exception as exc:
                print(f"  ERROR: {exc}")
                results["error"].append(cert)

        print("\nDone. Keeping browser open for 10 s so you can inspect...")
        page.wait_for_timeout(10_000)
        browser.close()

    print("\n=== Summary ===")
    print(f"Updated:   {len(results['ok'])} card(s)")
    print(f"Not found: {len(results['missing'])} cert(s): {results['missing']}")
    print(f"Errors:    {len(results['error'])} cert(s): {results['error']}")


if __name__ == "__main__":
    main()
