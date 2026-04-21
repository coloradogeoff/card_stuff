#!/usr/bin/env python3
"""
Match cards in missingcost_with_email_total.csv against eBay purchase history HTML.
Prints matches for review, then writes updated CSV.

Usage:
    python match_ebay_report.py           # review + write
    python match_ebay_report.py --dry-run # review only
"""

import csv
import math
import re
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path

EBAY_HTML  = Path.home() / 'Downloads/ebayReports/reports/transactionreports/purchaseHistory.html'
INPUT_CSV  = Path(__file__).parent / 'missingcost_with_email_total.csv'
OUTPUT_CSV = Path(__file__).parent / 'missingcost_with_email_total.csv'
GRADING_FEE = 20
DRY_RUN = '--dry-run' in sys.argv


# ── Parse eBay HTML ───────────────────────────────────────────────────────────

class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows = []; self.current_row = []; self.current_cell = ''
        self.in_cell = False; self.headers = []; self.header_done = False

    def handle_starttag(self, tag, attrs):
        if tag in ('td', 'th'):
            self.in_cell = True; self.current_cell = ''

    def handle_endtag(self, tag):
        if tag in ('td', 'th'):
            self.in_cell = False
            self.current_row.append(self.current_cell.strip())
        elif tag == 'tr':
            if self.current_row:
                if not self.header_done:
                    self.headers = self.current_row; self.header_done = True
                else:
                    self.rows.append(self.current_row)
                self.current_row = []

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data


def load_ebay_rows(path):
    with open(path) as f:
        html = f.read()
    p = _TableParser()
    p.feed(html)
    return [dict(zip(p.headers, r)) for r in p.rows]


# ── Matching ──────────────────────────────────────────────────────────────────

def score_match(card_item, card_player, ebay_title):
    """Score how well an eBay listing title matches a card."""
    title = ebay_title.upper()
    score = 0

    # Player name (high value)
    for word in card_player.upper().split():
        if word in title:
            score += 2

    # Year
    year_m = re.search(r'\b(20\d\d|19\d\d)\b', card_item)
    if year_m and year_m.group(1) in title:
        score += 2

    # Card number — most distinctive signal
    num_m = re.search(r'#(\w+)', card_item)
    if num_m:
        cardnum = num_m.group(1).upper()
        if re.search(r'#\s*0*' + re.escape(cardnum), title):
            score += 5

    # Set brand words
    skip = {'PANINI', 'THE', 'AND', 'OF', 'IN', 'A', 'AN', 'CARD', 'VARIATION', 'PRIZM'}
    for word in card_item.upper().split():
        if word not in skip and not word.startswith('#') and len(word) > 3:
            if word in title:
                score += 1

    return score


def find_best_match(card_item, card_player, ebay_rows, min_score=5):
    """Return best matching eBay row or None."""
    best = None
    best_score = 0
    for row in ebay_rows:
        sc = score_match(card_item, card_player, row['Listing Title'])
        if sc > best_score:
            best_score = sc
            best = row
    if best_score >= min_score:
        return best, best_score
    return None, best_score


def parse_date(date_str):
    """Convert 'Apr 12, 2026 10:30 AM' → 'YYYY-MM-DD'."""
    try:
        return datetime.strptime(date_str.strip(), '%b %d, %Y %I:%M %p').strftime('%Y-%m-%d')
    except ValueError:
        try:
            return datetime.strptime(date_str.strip()[:12], '%b %d, %Y').strftime('%Y-%m-%d')
        except ValueError:
            return ''


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ebay_rows = load_ebay_rows(EBAY_HTML)
    print(f'Loaded {len(ebay_rows)} eBay purchase rows\n')

    with open(INPUT_CSV) as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    MURRAY = '132272795'
    LEBRON = '149237668'
    CLARK_145  = '132272752'  # Prizm WNBA #145 base — wrong match (Blue variety too expensive)
    CLARK_72   = '132272751'  # Select WNBA #72 base — wrong match (already-graded PSA 10)
    ARTISTIC_1 = '132272772'  # Artistic Selection AS1 — matched expensive Prizm, not this card
    ARTISTIC_2 = '132272773'  # Artistic Selection AS2 — same

    updates = []
    for row in rows:
        cert   = row['Cert Number']
        item   = row['Item']
        player = row['Subject']

        # Manual overrides
        if cert == MURRAY:
            updates.append((row, None, 0, 30, '', 'Manual ($30 estate sale)'))
            continue
        if cert == LEBRON:
            updates.append((row, None, 0, 8, '', 'Manual ($8)'))
            continue
        if cert == CLARK_145:
            updates.append((row, None, 0, 10, '', 'Manual ($10 base)'))
            continue
        if cert == CLARK_72:
            updates.append((row, None, 0, 10, '', 'Manual ($10 base)'))
            continue
        if cert in (ARTISTIC_1, ARTISTIC_2):
            updates.append((row, None, 0, 5, '', 'Manual ($5)'))
            continue

        match, score = find_best_match(item, player, ebay_rows)
        if match:
            total_price = float(match['Total Price'])
            shipping    = float(match['Transaction Shipping Fee'])
            raw_cost    = total_price + shipping
            date        = parse_date(match['Purchase Date'])
            updates.append((row, match, score, raw_cost, date, ''))
        else:
            updates.append((row, None, score, None, '', 'NO MATCH'))

    # ── Review output ──────────────────────────────────────────────────────────
    print(f"{'CERT':12} {'DATE':12} {'COST+SHIP':>10} {'+GRADE':>7} {'SC':>3}  MATCH TITLE")
    print('─' * 110)
    no_match = []
    for row, match, score, raw_cost, date, note in updates:
        cert = row['Cert Number']
        item = row['Item'][:55]
        if note == 'NO MATCH':
            print(f"{cert:12} {'':12} {'':>10} {'':>7} {score:>3}  *** NO MATCH *** {item}")
            no_match.append(cert)
        elif note.startswith('Manual'):
            final = raw_cost + GRADING_FEE
            print(f"{cert:12} {'':12} {raw_cost:>10.2f} {final:>7}  --  {note}")
        else:
            final = math.ceil(raw_cost) + GRADING_FEE
            title = match['Listing Title'][:60]
            print(f"{cert:12} {date:12} {raw_cost:>10.2f} {final:>7} {score:>3}  {title}")

    print()
    print(f'Matched: {len(updates) - len(no_match)}/{len(updates)}   No match: {len(no_match)}')

    if DRY_RUN:
        print('\n--dry-run: no changes written.')
        return

    # ── Write ──────────────────────────────────────────────────────────────────
    confirm = input('\nWrite these values to CSV? [y/N] ').strip().lower()
    if confirm != 'y':
        print('Aborted.')
        return

    for row, match, score, raw_cost, date, note in updates:
        if raw_cost is None:
            continue
        final = math.ceil(raw_cost) + GRADING_FEE
        row['My Cost']       = str(final)
        row['Date Acquired'] = date
        # Source already set; leave as-is

    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Written to {OUTPUT_CSV}')


if __name__ == '__main__':
    main()
