#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path


def _parse_date(value: str) -> tuple[int, int]:
    try:
        year_str, month_str = value.split("-", 1)
        year = int(year_str)
        month = int(month_str)
    except ValueError as exc:
        raise ValueError("date must be in YYYY-MM format") from exc
    if not (1 <= month <= 12):
        raise ValueError("month must be 1-12")
    return year, month


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print and ensure the Sales directory for a given month exists."
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("SALES_ROOT") or str(Path.home() / "Sales"),
        help="Sales root directory (default: ~/Sales or $SALES_ROOT)",
    )
    parser.add_argument(
        "--date",
        help="Target month in YYYY-MM format (default: current month)",
    )
    args = parser.parse_args()

    if args.date:
        year, month = _parse_date(args.date)
    else:
        now = dt.date.today()
        year, month = now.year, now.month

    root = Path(args.root).expanduser()
    if root.is_symlink() and not root.exists():
        link_target = os.readlink(root)
        print(
            f"Sales root symlink is broken: {root} -> {link_target}",
            file=sys.stderr,
        )
        return 2

    target = root / f"{year:04d}" / f"{month:02d}"
    target.mkdir(parents=True, exist_ok=True)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
