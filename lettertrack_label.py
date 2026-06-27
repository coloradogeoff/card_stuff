#!/usr/bin/env python3
"""Print a LetterTrack label PDF on an A7 (5.25×7.25) or 6×9 envelope."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from pypdf import PageObject, PdfReader, PdfWriter, Transformation

PRINTER = "_192_168_86_174"
PRINT_OPTIONS = [
    "MediaType=MidWeight96110",
    "HPPrintQuality=ProRes1200",
]
ENVELOPES = {
    "a7":  (5.25 * 72, 7.25 * 72, "Custom.5.25x7.25in"),
    "6x9": (6.0  * 72, 9.0  * 72, "Custom.6x9in"),
}


def newest_pdf() -> Path:
    downloads = Path.home() / "Downloads"
    pdfs = [p for p in downloads.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"]
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {downloads}")
    return max(pdfs, key=lambda p: p.stat().st_mtime)


def scale_to_envelope(source: Path, output: Path, env_w: float, env_h: float) -> None:
    reader = PdfReader(source)
    if len(reader.pages) != 1:
        raise ValueError(f"Expected a 1-page PDF, got {len(reader.pages)}")

    src = reader.pages[0]
    src_w = float(src.mediabox.width)
    src_h = float(src.mediabox.height)

    scale = min(env_w / src_w, env_h / src_h)
    offset_x = (env_w - src_w * scale) / 2
    offset_y = (env_h - src_h * scale) / 2

    page = PageObject.create_blank_page(width=env_w, height=env_h)
    page.merge_transformed_page(
        src,
        Transformation().scale(scale).translate(offset_x, offset_y),
        over=True,
    )

    writer = PdfWriter()
    writer.add_page(page)
    with output.open("wb") as f:
        writer.write(f)


def print_pdf(pdf: Path, media: str, printer: str) -> None:
    cmd = ["lp", "-d", printer, "-o", f"media={media}"]
    for opt in PRINT_OPTIONS:
        cmd.extend(["-o", opt])
    cmd.append(str(pdf))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown lp error"
        raise RuntimeError(f"lp failed: {detail}")
    if result.stdout.strip():
        print(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path,
        help="LetterTrack label PDF (default: newest PDF in ~/Downloads)",
    )
    parser.add_argument(
        "--envelope", choices=["a7", "6x9"], default="a7",
        help="Envelope size: a7 = 5.25×7.25\" (default), 6x9 = 6×9\"",
    )
    parser.add_argument("--printer", default=PRINTER)
    parser.add_argument(
        "--preview", action="store_true",
        help="Write scaled PDF to /tmp and open it instead of printing",
    )
    args = parser.parse_args()

    pdf = (args.input or newest_pdf()).expanduser().resolve()
    if not pdf.is_file():
        print(f"File not found: {pdf}", file=sys.stderr)
        sys.exit(1)
    if not args.input:
        print(f"Using newest PDF: {pdf.name}")

    env_w, env_h, media = ENVELOPES[args.envelope]

    if args.preview:
        out = Path(f"/tmp/{pdf.stem}-{args.envelope}-preview.pdf")
        scale_to_envelope(pdf, out, env_w, env_h)
        print(f"Preview: {out}")
        subprocess.run(["open", str(out)])
        return

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        scale_to_envelope(pdf, tmp_path, env_w, env_h)
        print(f"Printing {pdf.name} → {args.printer} ({media})…")
        print_pdf(tmp_path, media, args.printer)
        print("Sent to printer.")
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
