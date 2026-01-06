#!/usr/bin/env python3
"""
make_envelope_pdf.py

Generates an envelope-style PDF with:
- A rectangle inset 0.5" from all edges
- Return address in the top-left (inside the inset)
- Destination address block centered on the page, left-justified,
  and shifted down by ~1 inch

Printing note (HP/CUPS): Some drivers ignore PDF /Rotate flags and layout based on
page size (MediaBox). If your 9x6 artwork won’t rotate when printing, generate a
true 6x9 PDF (portrait) but rotate the CONTENT 90° so it still looks like a 9x6
envelope. This script does that automatically when --driver-safe is enabled.

Requires:
  pip install reportlab
"""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import black, white
from reportlab.pdfbase import pdfmetrics



def make_envelope_pdf(
    out_path: str | Path,
    return_lines: list[str],
    to_lines: list[str],
    *,
    page_w_in: float = 9.0,
    page_h_in: float = 6.0,
    inset_in: float = 0.5,
    padding_in: float = 0.15,
    return_font: str = "Helvetica",
    return_font_size: int = 11,
    return_leading: int = 13,
    to_font: str = "Helvetica",
    to_font_size: int = 16,
    to_leading: int = 20,
    # Vertical placement controls
    base_center_y_frac: float = 0.43,  # slightly below true center
    to_shift_down_in: float = 1.0,     # move destination block further down
    content_rotation: str = "none",    # "none", "cw", or "ccw"
) -> str:
    W_page, H_page = page_w_in * inch, page_h_in * inch
    out_path = str(out_path)
    c = canvas.Canvas(out_path, pagesize=(W_page, H_page))

    # White background (fill whole page)
    c.setFillColor(white)
    c.rect(0, 0, W_page, H_page, fill=1, stroke=0)

    if content_rotation == "ccw":
        c.saveState()
        c.translate(W_page, 0)
        c.rotate(90)
        W, H = H_page, W_page
    elif content_rotation == "cw":
        c.saveState()
        c.translate(0, H_page)
        c.rotate(-90)
        W, H = H_page, W_page
    else:
        W, H = W_page, H_page

    inset = inset_in * inch
    padding = padding_in * inch

    # Return address (top-left inside inset)
    x_ret = inset + padding
    y_ret_top = H - inset - padding
    c.setFillColor(black)
    c.setFont(return_font, return_font_size)
    for i, line in enumerate(return_lines):
        c.drawString(x_ret, y_ret_top - i * return_leading, line)

    # Destination address:
    # Center the block horizontally, but draw each line left-justified at x_left.
    c.setFont(to_font, to_font_size)
    line_widths = [pdfmetrics.stringWidth(line, to_font, to_font_size) for line in to_lines] or [0]
    block_width = max(line_widths)
    x_left = (W - block_width) / 2.0

    # Compute block height (approx) and place it lower on the page
    block_height = to_leading * (len(to_lines) - 1) + to_font_size
    center_y = (H * base_center_y_frac) - (to_shift_down_in * inch)
    y_block_top = center_y + (block_height / 2.0)

    for i, line in enumerate(to_lines):
        c.drawString(x_left, y_block_top - i * to_leading, line)

    if content_rotation in ("cw", "ccw"):
        c.restoreState()

    c.showPage()
    c.save()
    return out_path



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an envelope-style PDF with addresses. Use --driver-safe to output a true 6x9 portrait PDF with rotated content that looks like a 9x6 envelope."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="envelope_9x6.pdf",
        help="Output PDF filename (default: envelope_9x6.pdf)",
    )

    parser.add_argument(
        "--page",
        choices=["9x6", "6x9"],
        default="9x6",
        help="Page size to generate (default: 9x6). Use 6x9 if your printer feeds the envelope in portrait.",
    )

    # Default: driver-safe ON (use --no-driver-safe to disable)
    if hasattr(argparse, "BooleanOptionalAction"):
        parser.add_argument(
            "--driver-safe",
            default=True,
            action=argparse.BooleanOptionalAction,
            help="Output a driver-safe rotated PDF by swapping page size and rotating content (recommended for some HP/CUPS setups).",
        )
    else:
        # Fallback for very old Python
        parser.add_argument(
            "--driver-safe",
            action="store_true",
            help="Output a driver-safe rotated PDF by swapping page size and rotating content.",
        )
        parser.add_argument(
            "--no-driver-safe",
            action="store_true",
            help="Disable driver-safe output (fallback mode).",
        )

    parser.add_argument(
        "--content-rotation",
        choices=["none", "cw", "ccw"],
        default=None,
        help="Rotate the drawing content by 90 degrees: none, cw (clockwise), or ccw (counterclockwise). Default is 'ccw' when --driver-safe is enabled, else 'none'.",
    )

    args = parser.parse_args()

    return_lines = [
        "NedDog's Stamps & Cards",
        "2644 Ridge Rd",
        "Nederland, CO 80466",
    ]

    to_lines = [
        "Andrew Attard",
        "13 Lady Smith Drive",
        "ABN#64652016681 Code:PAID",
        "Edmondson Park NSW 2174",
        "Australia",
    ]

    out_path = Path(args.output)

    # Determine driver-safe flag (supports old-Python fallback)
    driver_safe = getattr(args, "driver_safe", False)
    if hasattr(args, "no_driver_safe") and args.no_driver_safe:
        driver_safe = False

    # HP/CUPS note: prefer a true MediaBox in the feed orientation.
    # When driver_safe is enabled, force a 6x9 portrait PDF (upright text).
    effective_page = "6x9" if driver_safe else args.page

    if effective_page == "9x6":
        page_w_in, page_h_in = 9.0, 6.0
    else:
        page_w_in, page_h_in = 6.0, 9.0

    # Determine content rotation
    if args.content_rotation is None:
        if effective_page == "6x9":
            content_rotation = "ccw"
        else:
            content_rotation = "none"
    else:
        content_rotation = args.content_rotation

    make_envelope_pdf(
        out_path,
        return_lines,
        to_lines,
        page_w_in=page_w_in,
        page_h_in=page_h_in,
        content_rotation=content_rotation,
    )

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
