#!/usr/bin/env python3
"""
Minimal eBay Developer API starter for public listing search.

Examples:
  python ebay_api_demo.py search "Nikola Jokic Select"
  python ebay_api_demo.py search "Caitlin Clark Prizm" --limit 10
  python ebay_api_demo.py search "Michael Jordan Fleer" --sandbox
  python ebay_api_demo.py token

Credentials:
  Set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET in the environment, or add them
  to a .env file in the repo root or home directory.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROD_IDENTITY_URL = "https://api.ebay.com/identity/v1/oauth2/token"
PROD_BROWSE_URL = "https://api.ebay.com/buy/browse/v1/item_summary/search"
SANDBOX_IDENTITY_URL = "https://api.sandbox.ebay.com/identity/v1/oauth2/token"
SANDBOX_BROWSE_URL = "https://api.sandbox.ebay.com/buy/browse/v1/item_summary/search"
DEFAULT_SCOPE = "https://api.ebay.com/oauth/api_scope"


class EbayApiError(RuntimeError):
    """Raised when an eBay API request fails."""


def load_dotenv() -> None:
    """Load simple KEY=VALUE pairs from local .env files if present."""
    candidates = [Path.cwd() / ".env", Path.home() / ".env"]
    for path in candidates:
        if not path.exists():
            continue
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if value:
        return value
    raise SystemExit(
        f"Missing {name}. Set it in the environment or a .env file."
    )


def request_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: bytes | None = None,
) -> dict[str, Any]:
    req = Request(url, method=method, headers=headers or {}, data=data)
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise EbayApiError(f"HTTP {exc.code} for {url}: {body}") from exc
    except URLError as exc:
        raise EbayApiError(f"Network error for {url}: {exc.reason}") from exc


def get_app_token(*, sandbox: bool = False, scope: str = DEFAULT_SCOPE) -> dict[str, Any]:
    client_id = require_env("EBAY_CLIENT_ID")
    client_secret = require_env("EBAY_CLIENT_SECRET")
    identity_url = SANDBOX_IDENTITY_URL if sandbox else PROD_IDENTITY_URL
    basic_auth = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    body = urlencode(
        {
            "grant_type": "client_credentials",
            "scope": scope,
        }
    ).encode("utf-8")
    return request_json(
        identity_url,
        method="POST",
        headers={
            "Authorization": f"Basic {basic_auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data=body,
    )


def search_items(
    query: str,
    *,
    limit: int,
    offset: int,
    sandbox: bool = False,
) -> dict[str, Any]:
    token_payload = get_app_token(sandbox=sandbox)
    browse_url = SANDBOX_BROWSE_URL if sandbox else PROD_BROWSE_URL
    url = f"{browse_url}?{urlencode({'q': query, 'limit': limit, 'offset': offset})}"
    return request_json(
        url,
        headers={
            "Authorization": f"Bearer {token_payload['access_token']}",
            "Accept": "application/json",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        },
    )


def print_token(payload: dict[str, Any]) -> None:
    print("access_token:", payload.get("access_token", "")[:60] + "...")
    print("expires_in:", payload.get("expires_in"))
    print("token_type:", payload.get("token_type"))


def print_search_results(payload: dict[str, Any]) -> None:
    total = payload.get("total")
    summaries = payload.get("itemSummaries", [])
    if total is not None:
        print(f"total: {total}")
    if not summaries:
        print("No items returned.")
        return

    for idx, item in enumerate(summaries, start=1):
        price = item.get("price", {})
        shipping = item.get("shippingOptions", [])
        shipping_cost = ""
        if shipping:
            shipping_cost = shipping[0].get("shippingCost", {})
        print(f"{idx}. {item.get('title', 'Untitled')}")
        print(f"   price: {price.get('value', '?')} {price.get('currency', '')}".rstrip())
        if shipping_cost:
            print(
                "   shipping: "
                f"{shipping_cost.get('value', '?')} {shipping_cost.get('currency', '')}".rstrip()
            )
        print(f"   condition: {item.get('condition', 'Unknown')}")
        print(f"   item id: {item.get('itemId', '')}")
        print(f"   url: {item.get('itemWebUrl', '')}")
        print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Starter client for eBay Developer OAuth and Browse search."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    token_parser = subparsers.add_parser("token", help="Fetch and print an app token summary")
    token_parser.add_argument("--sandbox", action="store_true", help="Use sandbox endpoints")

    search_parser = subparsers.add_parser("search", help="Search public eBay listings")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--offset", type=int, default=0, help="Pagination offset")
    search_parser.add_argument("--sandbox", action="store_true", help="Use sandbox endpoints")

    return parser


def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.command == "token":
            payload = get_app_token(sandbox=args.sandbox)
            print_token(payload)
            return 0

        if args.command == "search":
            if args.limit < 1 or args.limit > 200:
                raise SystemExit("--limit must be between 1 and 200")
            if args.offset < 0:
                raise SystemExit("--offset must be 0 or greater")
            payload = search_items(
                args.query,
                limit=args.limit,
                offset=args.offset,
                sandbox=args.sandbox,
            )
            print_search_results(payload)
            return 0

        parser.print_help()
        return 1
    except EbayApiError as exc:
        print(f"eBay API error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
