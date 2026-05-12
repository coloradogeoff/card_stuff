#!/usr/bin/env python3
"""
eBay Purchase Tracker

Modes:
  python ebay_tracker.py            # interactive: review items, mark received
  python ebay_tracker.py --cron     # fetch inbox, update state, mark emails read
  python ebay_tracker.py --refresh  # fetch + update, then go interactive

The cron job keeps the inbox tidy (marks purchase emails read) and updates
~/.ebay_tracker.json.  The interactive mode works offline from that file;
Gmail is only contacted when archiving emails for received items.
"""

import base64
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES     = ['https://www.googleapis.com/auth/gmail.modify']
OAUTH_PATH = os.path.expanduser('~/.gmail-mcp/gcp-oauth.keys.json')
TOKEN_PATH = os.path.expanduser('~/.gmail-mcp/ebay-tracker-token.json')
STATE_PATH = os.path.expanduser('~/.ebay_tracker.json')
DAYS_BACK  = 40

STAGES     = ['CONFIRMED', 'ORDER UPDATE', 'WITH CARRIER', 'OUT FOR DELIVERY', 'DELIVERED']
STAGE_RANK = {s: i for i, s in enumerate(STAGES)}
STAGE_LABEL = {
    'CONFIRMED':         '⏳ Confirmed',
    'ORDER UPDATE':      '🔄 Order Update',
    'WITH CARRIER':      '📦 With Carrier',
    'OUT FOR DELIVERY':  '🚚 Out for Delivery',
    'DELIVERED':         '✅ Delivered',
}
W = 60  # display width


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


# ── State file ────────────────────────────────────────────────────────────────

def load_state():
    """Return list of purchase dicts from state file, or []."""
    if not os.path.exists(STATE_PATH):
        return []
    with open(STATE_PATH) as f:
        data = json.load(f)
    return data.get('purchases', [])


def save_state(purchases):
    with open(STATE_PATH, 'w') as f:
        json.dump({
            'updated': datetime.now().isoformat(timespec='seconds'),
            'purchases': purchases,
        }, f, indent=2)


def state_updated_at():
    if not os.path.exists(STATE_PATH):
        return None
    with open(STATE_PATH) as f:
        return json.load(f).get('updated')


# ── Gmail helpers ─────────────────────────────────────────────────────────────

def fetch_ebay_messages(service, days_back=DAYS_BACK, inbox_only=True):
    """Fetch eBay emails within the last N days. inbox_only=False for deep scan."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    inbox  = 'in:inbox ' if inbox_only else ''
    query  = f'from:ebay.com {inbox}after:{cutoff.strftime("%Y/%m/%d")}'

    ids = []
    page_token = None
    while True:
        r = service.users().messages().list(
            userId='me', q=query, pageToken=page_token, maxResults=500
        ).execute()
        ids.extend(r.get('messages', []))
        page_token = r.get('nextPageToken')
        if not page_token:
            break

    label = 'inbox ' if inbox_only else 'all '
    print(f"Fetching {len(ids)} eBay {label}emails (last {days_back} days)...")
    msgs = []
    for i, m in enumerate(ids):
        if i % 25 == 0 and i > 0:
            print(f"  {i}/{len(ids)}...")
        msgs.append(
            service.users().messages().get(userId='me', id=m['id'], format='full').execute()
        )
    return msgs


def mark_read(service, message_ids):
    """Remove UNREAD label from a list of message IDs."""
    for mid in message_ids:
        try:
            service.users().messages().modify(
                userId='me', id=mid, body={'removeLabelIds': ['UNREAD']}
            ).execute()
        except Exception:
            pass  # already read or missing — skip


def archive_messages(service, message_ids):
    """Remove INBOX label from a list of message IDs."""
    for mid in message_ids:
        try:
            service.users().messages().modify(
                userId='me', id=mid, body={'removeLabelIds': ['INBOX']}
            ).execute()
        except Exception:
            pass  # already archived or missing — skip


def get_header(msg, name):
    for h in msg.get('payload', {}).get('headers', []):
        if h['name'].lower() == name.lower():
            return h['value']
    return ''


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


# ── Email classification & parsing ────────────────────────────────────────────

_SKIP_LINE = re.compile(
    r'^(Price:|Estimated delivery:|Order number:|Seller:|Track|Hi |Check|View|'
    r'Your order|Shipped|Delivered|Carrier:|How useful|© |eBay |Update your|'
    r'Email Reference|If you have|\$[\d.]+|[\d.\-]+$|Pyp:|Pyc:|'
    r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|'
    r'(AU|US|CA|NZ|GB|CAD|USD|AUD|NZD|GBP|EUR|JPY)\s*\$|'
    r'\(\s*\d+\s*x\s*\$|'
    r'The seller |Your message|Buy It Now|Quantity:|Item subtotal|Subtotal:|Shipping:)',
    re.IGNORECASE
)

_EST_DELIVERY_RE = re.compile(r'Estimated delivery:\s*(.+?)(?:\n|$)', re.IGNORECASE)

_NOISE_RE = re.compile(
    r'pay now to receive|automatic checkout is set|best offer|'
    r'you have a question|sent a message|sent a question|'
    r'feedback reminder|seller left feedback|left you feedback|'
    r'invoice from|unpaid item|seller has responded|response from the seller',
    re.IGNORECASE
)


def classify_email(subject, body):
    if _NOISE_RE.search(subject):
        return None
    combined = (subject + ' ' + body[:500]).lower()
    if any(k in combined for k in ["order's been delivered", 'order has arrived',
                                    'has been delivered', 'order delivered']):
        return 'DELIVERED'
    if re.search(r'\bdelivered\b', subject, re.IGNORECASE):
        return 'DELIVERED'
    if any(k in combined for k in ['out for delivery', 'almost there']):
        return 'OUT FOR DELIVERY'
    if re.search(r'out for delivery', subject, re.IGNORECASE):
        return 'OUT FOR DELIVERY'
    if any(k in combined for k in ['with its carrier', 'tracking number',
                                    'package is now with', 'let the tracking begin']):
        return 'WITH CARRIER'
    if re.search(r'(has shipped|your package)', subject, re.IGNORECASE):
        return 'WITH CARRIER'
    if 'order update' in combined or 'packing your order' in combined:
        return 'ORDER UPDATE'
    if any(k in combined for k in ['order is confirmed', 'order has been confirmed',
                                    'order confirmed', 'you won', "you're the winner",
                                    'thanks for shopping']):
        return 'CONFIRMED'
    return None


def parse_items_from_body(body):
    """Return list of (name, item_id, est_delivery) tuples."""
    items = []
    seen  = set()
    for m in re.finditer(r'Item ID:\s*(\d{9,13})', body):
        item_id = m.group(1)
        if item_id in seen:
            continue
        seen.add(item_id)

        before = body[max(0, m.start() - 500) : m.start()]
        lines  = [l.strip() for l in before.split('\n') if l.strip()]
        candidates = [l for l in lines if not _SKIP_LINE.match(l) and len(l) > 8]
        name = candidates[-1] if candidates else ''

        est_m = _EST_DELIVERY_RE.search(before)
        est   = est_m.group(1).strip() if est_m else ''
        if est:
            half = len(est) // 2
            if est[:half] == est[half:]:
                est = est[:half]

        items.append((name, item_id, est))
    return items


def strip_subject_prefix(subject):
    s = re.sub(r'^[\U00010000-\U0010ffff\u2600-\u26ff\u2700-\u27bf!]+\s*', '', subject).strip()
    s = re.sub(
        r'^(your order is confirmed[:\s]*|order update[:\s]*|order confirmed[:\s]*|'
        r'your package is now with its carrier[:\s!]*|'
        r"your order's been delivered[:\s]*|delivered[:\s]*|order delivered[:\s]*|"
        r'you won[:\s]*|your ebay item has shipped[:\s]*|'
        r'out for delivery[:\s]*|tracking update[:\s]*|shipment update[:\s]*|'
        r'almost there[:\s]*|payment.*?[:\s]*|reminder[:\s]*)',
        '', s, flags=re.IGNORECASE
    ).strip()
    return s.rstrip('.').strip()


def msg_date(msg):
    ts = int(msg.get('internalDate', 0)) / 1000
    return datetime.fromtimestamp(ts).strftime('%b %d'), int(msg.get('internalDate', 0))


# ── State merge ───────────────────────────────────────────────────────────────

def emails_to_purchase_map(messages):
    """
    Process a list of Gmail messages into a dict keyed by item_id.
    Each value: { name, order_date, status, status_date, est_delivery,
                  ts, message_ids }
    Also returns a list of all message IDs that were classified (to mark read).
    """
    purchases   = {}   # item_id → dict
    by_name     = {}   # normalized name → dict (fallback)
    classified_ids = []

    def get_or_create_by_id(item_id, name, order_date):
        if item_id not in purchases:
            purchases[item_id] = {
                'item_id': item_id, 'name': name, 'order_date': order_date,
                'status': None, 'status_date': '', 'est_delivery': '',
                'ts': 0, 'message_ids': [],
            }
        return purchases[item_id]

    def apply_update(p, stage, date_str, ts, name='', est='', mid=None):
        if name and not _SKIP_LINE.match(name) and (
            not p['name'] or _SKIP_LINE.match(p['name'])
        ):
            p['name'] = name
        if mid and mid not in p['message_ids']:
            p['message_ids'].append(mid)
        if p['status'] is None or STAGE_RANK[stage] > STAGE_RANK[p['status']]:
            p['status']       = stage
            p['status_date']  = date_str
            p['ts']           = ts
            if est:
                p['est_delivery'] = est
        if stage == 'CONFIRMED' and (not p['order_date'] or p['order_date'] == '?'):
            p['order_date'] = date_str

    for msg in messages:
        subject = get_header(msg, 'Subject')
        body    = extract_text(msg.get('payload', {}))
        stage   = classify_email(subject, body)
        if not stage:
            continue

        mid = msg['id']
        classified_ids.append(mid)
        date_str, ts = msg_date(msg)
        items = parse_items_from_body(body)

        if items:
            subject_name = strip_subject_prefix(subject)
            for name, item_id, est in items:
                if not name and subject_name:
                    name = subject_name
                p = get_or_create_by_id(item_id, name, date_str if stage == 'CONFIRMED' else '?')
                apply_update(p, stage, date_str, ts, name, est, mid)
        else:
            name = strip_subject_prefix(subject)
            key  = name.lower()
            if not key:
                continue
            if key not in by_name:
                by_name[key] = {
                    'item_id': None, 'name': name,
                    'order_date': date_str if stage == 'CONFIRMED' else '?',
                    'status': None, 'status_date': '', 'est_delivery': '',
                    'ts': 0, 'message_ids': [],
                }
            apply_update(by_name[key], stage, date_str, ts, mid=mid)

    # Merge by_name into purchases where names overlap
    for key, p in list(by_name.items()):
        for purchase in purchases.values():
            pname = purchase['name'].lower()
            if key and pname and (key in pname or pname in key):
                if purchase['status'] is None or (
                    p['status'] and STAGE_RANK[p['status']] > STAGE_RANK[purchase['status']]
                ):
                    purchase['status']      = p['status']
                    purchase['status_date'] = p['status_date']
                    purchase['ts']          = p['ts']
                for mid in p['message_ids']:
                    if mid not in purchase['message_ids']:
                        purchase['message_ids'].append(mid)
                del by_name[key]
                break

    new_items = list(purchases.values()) + list(by_name.values())
    return new_items, classified_ids


def merge_with_state(existing, new_items):
    """
    Merge freshly parsed purchase data into the existing state list.
    - Existing items are updated if new emails show a higher stage.
    - New items not in existing state are appended.
    - Existing items with no new email data are preserved as-is.
    """
    state_by_id   = {p['item_id']: p for p in existing if p.get('item_id')}
    state_by_name = {p['name'].lower(): p for p in existing if not p.get('item_id') and p.get('name')}

    for item in new_items:
        iid  = item.get('item_id')
        name = (item.get('name') or '').lower()

        if iid and iid in state_by_id:
            existing_p = state_by_id[iid]
        elif not iid and name in state_by_name:
            existing_p = state_by_name[name]
        else:
            existing.append(item)
            if iid:
                state_by_id[iid] = item
            elif name:
                state_by_name[name] = item
            continue

        # Update status if higher
        if item['status'] and (
            existing_p['status'] is None or
            STAGE_RANK[item['status']] > STAGE_RANK.get(existing_p['status'], -1)
        ):
            existing_p['status']      = item['status']
            existing_p['status_date'] = item['status_date']
            existing_p['ts']          = item['ts']
            if item.get('est_delivery'):
                existing_p['est_delivery'] = item['est_delivery']

        # Update name if we now have a better one (existing missing or looks like noise)
        new_name = item.get('name')
        old_name = existing_p.get('name')
        if new_name and not _SKIP_LINE.match(new_name) and (
            not old_name or _SKIP_LINE.match(old_name)
        ):
            existing_p['name'] = new_name

        # Update order date if we now have one
        if (not existing_p.get('order_date') or existing_p['order_date'] == '?') \
                and item.get('order_date') and item['order_date'] != '?':
            existing_p['order_date'] = item['order_date']

        # Merge message IDs
        for mid in item.get('message_ids', []):
            if mid not in existing_p.get('message_ids', []):
                existing_p.setdefault('message_ids', []).append(mid)

    return existing


# ── Display ───────────────────────────────────────────────────────────────────

def sorted_purchases(state):
    """Return (in_progress, delivered) sorted lists, excluding received/archived."""
    active = [p for p in state if not p.get('received')]
    in_progress = [p for p in active if p.get('status') != 'DELIVERED']
    delivered   = [p for p in active if p.get('status') == 'DELIVERED']
    in_progress.sort(key=lambda p: STAGE_RANK.get(p.get('status') or '', 0), reverse=True)
    delivered.sort(key=lambda p: p.get('ts', 0), reverse=True)
    return in_progress, delivered


def display_state(state, updated_at=None):
    in_progress, delivered = sorted_purchases(state)

    print()
    print('═' * W)
    print(f'  eBay Purchase Tracker')
    if updated_at:
        print(f'  Last sync: {updated_at}')
    print('═' * W)

    if in_progress:
        print('\n── In Progress ──────────────────────────────────────────')
        for i, p in enumerate(in_progress, 1):
            _show_item(p, number=i)
    else:
        print('\n  Nothing in progress.')

    if delivered:
        print('\n── Delivered (not yet confirmed received) ───────────────')
        start = len(in_progress) + 1
        for i, p in enumerate(delivered, start):
            _show_item(p, number=i)

    total = len(in_progress) + len(delivered)
    print()
    print('═' * W)
    print(f'  {len(in_progress)} in progress  |  {len(delivered)} delivered')
    print('═' * W)
    print()


def _show_item(p, number=None):
    label   = STAGE_LABEL.get(p.get('status'), '❓ Unknown')
    name    = p.get('name') or '(unknown)'
    item_id = p.get('item_id')
    id_str  = f'  #{item_id}' if item_id else ''
    ordered = p.get('order_date') or '?'
    status_date = p.get('status_date', '')
    num_str = f'{number:2}. ' if number else '    '
    print(f'\n  {num_str}{label}')
    print(f'      {name}')
    print(f'      Ordered: {ordered}  →  {status_date}{id_str}')
    if p.get('est_delivery') and p.get('status') != 'DELIVERED':
        print(f'      Est. delivery: {p["est_delivery"]}')


# ── Interactive: mark received ────────────────────────────────────────────────

def prompt_and_archive(service, state):
    in_progress, delivered = sorted_purchases(state)
    active = in_progress + delivered
    if not active:
        return

    print("Enter numbers of items you've received (e.g.  1 3 5  or  2-8),")
    print("or press Enter to skip: ", end='', flush=True)
    raw = input().strip()
    if not raw:
        print("Nothing archived.")
        return

    chosen = []
    seen = set()
    for token in re.split(r'[\s,]+', raw):
        m = re.fullmatch(r'(\d+)-(\d+)', token)
        if m:
            nums = range(int(m.group(1)), int(m.group(2)) + 1)
        elif token.isdigit():
            nums = [int(token)]
        else:
            continue
        for n in nums:
            idx = n - 1
            if 0 <= idx < len(active) and idx not in seen:
                seen.add(idx)
                chosen.append(active[idx])

    if not chosen:
        print("No valid numbers entered.")
        return

    print()
    for p in chosen:
        mids = p.get('message_ids', [])
        print(f"  Archiving {len(mids)} email(s): {p.get('name', '?')}")
        archive_messages(service, mids)
        p['received'] = True

    save_state(state)
    print(f"\nDone. Marked {len(chosen)} item(s) received and archived their emails.")


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_fetch_and_update(service):
    """Fetch inbox emails, merge into state, mark emails read. Returns state."""
    state    = load_state()
    messages = fetch_ebay_messages(service, inbox_only=True)

    if messages:
        new_items, classified_ids = emails_to_purchase_map(messages)
        state = merge_with_state(state, new_items)
        # Remove items the user has marked received
        state = [p for p in state if not p.get('received')]
        save_state(state)

        print(f"Marking {len(classified_ids)} emails as read...")
        mark_read(service, classified_ids)
        print(f"State updated: {len(state)} items tracked.")
    else:
        print("No new eBay inbox emails found.")

    return state


def run_init(service):
    """One-time deep scan: all eBay emails (read + unread), no mark-read."""
    print("Deep scan — building initial state from all eBay emails...")
    messages = fetch_ebay_messages(service, inbox_only=False)
    if not messages:
        print("No eBay emails found.")
        return
    new_items, _ = emails_to_purchase_map(messages)
    # Start fresh — don't merge with stale state
    state = [p for p in new_items if not p.get('received')]
    save_state(state)
    print(f"State built: {len(state)} items tracked.\n")
    display_state(state, state_updated_at())


def run_cron(service):
    run_fetch_and_update(service)


def run_interactive(service):
    updated_at = state_updated_at()
    state      = load_state()

    if not state:
        print("No tracking data found. Run with --cron or --refresh first.")
        return

    display_state(state, updated_at)
    prompt_and_archive(service, state)


def run_refresh(service):
    run_fetch_and_update(service)
    print()
    run_interactive(service)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    mode = 'interactive'
    if '--init' in sys.argv:
        mode = 'init'
    elif '--cron' in sys.argv:
        mode = 'cron'
    elif '--refresh' in sys.argv:
        mode = 'refresh'

    creds   = get_credentials()
    service = build('gmail', 'v1', credentials=creds)

    if mode == 'init':
        run_init(service)
    elif mode == 'cron':
        run_cron(service)
    elif mode == 'refresh':
        run_refresh(service)
    else:
        run_interactive(service)


if __name__ == '__main__':
    main()
