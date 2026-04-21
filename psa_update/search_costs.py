#!/usr/bin/env python3
"""
Search Gmail for purchase emails matching each card in missingcost.csv.
Extracts price + shipping + tax total, rounds up to nearest dollar.
Writes missingcost_with_email_total.csv with a new "Email Total" column.
"""

import base64
import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES     = ['https://www.googleapis.com/auth/gmail.modify']
OAUTH_PATH = os.path.expanduser('~/.gmail-mcp/gcp-oauth.keys.json')
TOKEN_PATH = os.path.expanduser('~/.gmail-mcp/ebay-tracker-token.json')
CUTOFF     = '2023/01/01'

INPUT_CSV  = os.path.join(os.path.dirname(__file__), 'missingcost.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'missingcost_with_email_total.csv')


# ── Auth ──────────────────────────────────────────────────────────────────────

def get_credentials():
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if os.path.exists(TOKEN_PATH):
                os.remove(TOKEN_PATH)
            flow = InstalledAppFlow.from_client_secrets_file(OAUTH_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, 'w') as f:
            f.write(creds.to_json())
    return creds


# ── HTML → plain text ─────────────────────────────────────────────────────────

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ('style', 'script', 'head'):
            self._skip = True
        if tag in ('td', 'tr', 'div', 'p', 'br', 'li'):
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag in ('style', 'script', 'head'):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            s = data.strip()
            if s:
                self.parts.append(s)

    def result(self):
        raw = '\n'.join(self.parts)
        raw = re.sub(r'[\u034f\u200b\u200c\u200d\u00ad\ufeff\u061c\u17b5]', '', raw)
        lines = [re.sub(r'[ \t]+', ' ', l).strip() for l in raw.split('\n')]
        return '\n'.join(l for l in lines if l)


def _decode_part(part):
    data = part.get('body', {}).get('data', '')
    return base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='replace') if data else ''


def extract_text(payload):
    mime = payload.get('mimeType', '')
    if mime == 'text/plain':
        return _decode_part(payload)
    if mime == 'text/html':
        p = _TextExtractor()
        p.feed(_decode_part(payload))
        return p.result()
    plain = html = ''
    for part in payload.get('parts', []):
        t = extract_text(part)
        if not t:
            continue
        if part.get('mimeType') == 'text/plain':
            plain = plain or t
        else:
            html = html or t
    return plain or html


# ── Price parsing ─────────────────────────────────────────────────────────────

_DOLLAR_RE = re.compile(r'^\s*\$\s*([\d,]+\.?\d*)\s*$')
_AMT_RE    = re.compile(r'\$\s*([\d,]+\.?\d*)')


def _parse_dollar(s):
    s = s.replace(',', '')
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        return None


def parse_ebay_order_total(body):
    """
    Parse eBay order confirmation email body.
    eBay puts labels and dollar amounts on separate consecutive lines.

    Returns (total, item_count, note) where total is float or None.
    item_count is the number of items in the order (1 = single item).
    """
    lines = [l.strip() for l in body.split('\n') if l.strip()]

    # Find "Order total:" section and extract subtotal, shipping, tax
    # eBay format (each on its own line):
    #   Order total:
    #   Subtotal (N items)
    #   $X.XX
    #   Shipping
    #   $X.XX
    #   Sales tax
    #   $X.XX
    #   [Spendable funds / eBay Bucks]
    #   [− $X.XX]
    #   Total
    #   $X.XX   ← may be $0 if credits used

    subtotal = shipping = tax = item_count = None

    for i, line in enumerate(lines):
        ll = line.lower()

        # Detect "Subtotal (N items)" or "Subtotal (1 item)"
        m = re.search(r'subtotal\s*\((\d+)\s*items?\)', ll)
        if m:
            item_count = int(m.group(1))
            # Next non-empty line after this should be the dollar amount
            for j in range(i + 1, min(i + 3, len(lines))):
                amt_m = _DOLLAR_RE.match(lines[j])
                if amt_m:
                    subtotal = _parse_dollar(amt_m.group(1))
                    break
            continue

        # Shipping label → next line is amount
        if ll == 'shipping' or ll == 'shipping:':
            for j in range(i + 1, min(i + 3, len(lines))):
                amt_m = _DOLLAR_RE.match(lines[j])
                if amt_m:
                    v = _parse_dollar(amt_m.group(1))
                    if v is not None and v < 50:
                        shipping = v
                    break
            continue

        # "Free shipping" → $0
        if 'free shipping' in ll:
            shipping = 0.0
            continue

        # Sales tax label → next line is amount
        if re.match(r'^sales?\s*tax', ll):
            for j in range(i + 1, min(i + 3, len(lines))):
                amt_m = _DOLLAR_RE.match(lines[j])
                if amt_m:
                    v = _parse_dollar(amt_m.group(1))
                    if v is not None and v < 50:
                        tax = v
                    break
            continue

    if subtotal is not None:
        total = subtotal + (shipping or 0) + (tax or 0)
        note = f'{item_count or "?"}-item order: ${subtotal:.2f} + ship ${shipping or 0:.2f} + tax ${tax or 0:.2f}'
        return total, item_count or 1, note

    # Fallback: look for inline "Total: $X.XX" patterns
    inline_patterns = [
        re.compile(r'order\s+total[:\s]+\$\s*([\d,]+\.?\d*)', re.IGNORECASE),
        re.compile(r'amount\s+paid[:\s]+\$\s*([\d,]+\.?\d*)', re.IGNORECASE),
        re.compile(r'you\s+paid[:\s]+\$\s*([\d,]+\.?\d*)', re.IGNORECASE),
    ]
    for line in lines:
        for pat in inline_patterns:
            m = pat.search(line)
            if m:
                v = _parse_dollar(m.group(1))
                if v and v > 0.50:
                    return v, 1, 'inline total'

    return None, 0, 'no total found'


def count_items_in_body(body):
    """Count 'Item ID:' occurrences as a proxy for items in the order."""
    return len(re.findall(r'Item ID:\s*\d{9,13}', body))


# ── Gmail search ──────────────────────────────────────────────────────────────

def search_messages(service, query, max_results=15):
    ids = []
    page_token = None
    while len(ids) < max_results:
        r = service.users().messages().list(
            userId='me', q=query, pageToken=page_token,
            maxResults=min(max_results - len(ids), 50)
        ).execute()
        ids.extend(r.get('messages', []))
        page_token = r.get('nextPageToken')
        if not page_token:
            break

    msgs = []
    for m in ids[:max_results]:
        try:
            msgs.append(
                service.users().messages().get(userId='me', id=m['id'], format='full').execute()
            )
        except Exception:
            pass  # skip deleted / inaccessible messages
    return msgs


def is_order_confirmation(subject, body):
    combined = (subject + ' ' + body[:300]).lower()
    return bool(re.search(
        r'order.{0,20}confirm|you.ve won|you won|order.{0,10}placed|'
        r'payment.{0,20}received|checkout complete|purchase complete',
        combined, re.IGNORECASE
    ))


def score_match(body, subject_kw, year, cardnum, set_words):
    """Score how well an email body matches the card we're looking for."""
    score = 0
    bl = body.lower()
    if year and year in body:
        score += 2
    if cardnum and re.search(r'#\s*0*' + re.escape(cardnum), body, re.IGNORECASE):
        score += 3
    if subject_kw.lower() in bl:
        score += 2
    # Check set words (e.g. PRIZM, PANINI, WNBA)
    for w in set_words:
        if w.lower() in bl:
            score += 1
    return score


def find_purchase_total(service, card):
    """Search Gmail for purchase emails for this card. Returns (total, item_count, note)."""
    player  = card['Subject']   # e.g. "CAITLIN CLARK"
    item    = card['Item']      # e.g. "2024 PANINI PRIZM WNBA #145 CAITLIN CLARK"

    year_m  = re.search(r'\b(20\d\d|19\d\d)\b', item)
    num_m   = re.search(r'#(\w+)', item)
    year    = year_m.group(1) if year_m else ''
    cardnum = num_m.group(1) if num_m else ''

    # Pick a few distinctive set words (skip generic words)
    skip_words = {'PANINI', 'THE', 'AND', 'OF', 'IN', 'A', 'AN', '#', '-', 'VARIATION'}
    words = [w for w in item.split() if w not in skip_words and not w.startswith('#') and len(w) > 3]
    set_words = words[:6]

    # Search strategies: broad → more specific
    queries = []
    if year and cardnum:
        queries.append(f'from:ebay.com "{player}" "{year}" after:{CUTOFF}')
    if year:
        queries.append(f'from:ebay.com "{player}" after:{CUTOFF}')
    queries.append(f'from:ebay.com "{player}" after:{CUTOFF}')

    best = None  # (total, item_count, date_str, note, score)

    seen_ids = set()
    for query in queries:
        msgs = search_messages(service, query, max_results=15)
        for msg in msgs:
            mid = msg['id']
            if mid in seen_ids:
                continue
            seen_ids.add(mid)

            subj = next((h['value'] for h in msg.get('payload', {}).get('headers', [])
                         if h['name'] == 'Subject'), '')
            body = extract_text(msg.get('payload', {}))

            if not is_order_confirmation(subj, body):
                continue

            sc = score_match(body, player, year, cardnum, set_words)
            if sc < 1:
                continue

            total, n_items, note = parse_ebay_order_total(body)
            if total is None or total <= 0:
                continue

            ts = int(msg.get('internalDate', 0)) / 1000
            date_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime('%Y-%m-%d') if ts else ''

            if best is None or sc > best[4]:
                best = (total, n_items, date_str, f'score={sc} {note}', sc)

    if best:
        return best[0], best[1], best[2], best[3]
    return None, 0, '', 'Not found in email'


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    creds   = get_credentials()
    service = build('gmail', 'v1', credentials=creds)

    with open(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Insert "Email Total" column right after "My Cost"
    cost_idx = fieldnames.index('My Cost')
    new_fields = fieldnames[:cost_idx+1] + ['Email Total'] + fieldnames[cost_idx+1:]

    results = []
    for i, row in enumerate(rows):
        cert = row['Cert Number']
        item = row['Item']
        print(f"[{i+1}/{len(rows)}] {item[:65]}...")
        total, n_items, date_str, note = find_purchase_total(service, row)
        if total is not None:
            per_card = total / n_items if n_items > 1 else total
            rounded = math.ceil(per_card)
            multi_note = f' (÷{n_items} from order total ${total:.2f})' if n_items > 1 else ''
            row['Email Total'] = str(rounded)
            row['Date Acquired'] = date_str
            print(f"         → ${per_card:.2f} → ${rounded}{multi_note}  date={date_str}  [{note}]")
        else:
            row['Email Total'] = ''
            print(f"         → {note}")
        results.append(row)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        writer.writerows(results)

    found = sum(1 for r in results if r['Email Total'])
    print(f"\nDone. {found}/{len(results)} cards found in email.")
    print(f"Output: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
