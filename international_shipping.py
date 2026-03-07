#!/usr/bin/env python3
import base64
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import typer
from openai import OpenAI

app = typer.Typer(add_completion=False)
DEFAULT_SOURCE_DIR = Path("~/Downloads")
DEFAULT_OUTPUT_DIR = Path("/Volumes/Dutton 2TB/Sales/shipping")
DEFAULT_PROCESSED_DIR = Path("/Volumes/Dutton 2TB/Sales/shipping/processed")
SOURCE_IMAGE_EXTENSIONS = {".heic", ".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}

# -------- Helpers --------

def convert_to_jpeg(input_path: Path, jpeg_path: Path) -> None:
    """
    Uses macOS 'sips' to convert anything (HEIC/PNG/etc) to JPEG.
    """
    jpeg_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["/usr/bin/sips", "-s", "format", "jpeg", str(input_path), "--out", str(jpeg_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"sips failed:\n{proc.stderr}\n{proc.stdout}")

def data_url_for_jpeg(jpeg_path: Path) -> str:
    b64 = base64.b64encode(jpeg_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def safe_country_for_filename(country: str) -> str:
    # Normalize: uppercase, spaces -> underscores, strip weird chars
    country = country.strip().upper().replace(" ", "_")
    country = re.sub(r"[^A-Z0-9_-]+", "", country)
    return country or "UNKNOWN_COUNTRY"

def file_timestamp(path: Path) -> float:
    """
    Use file creation time when available; fallback to modification time.
    """
    st = path.stat()
    ts = getattr(st, "st_birthtime", None)
    return ts if ts is not None else st.st_mtime

def file_date_for_naming(path: Path) -> str:
    """
    Use file creation date when available; fallback to modification date.
    """
    return datetime.fromtimestamp(file_timestamp(path)).strftime("%Y-%m-%d")

def newest_source_file(source_dir: Path) -> Path:
    """
    Return the newest image file from source_dir.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    files = [
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SOURCE_IMAGE_EXTENSIONS
    ]
    if not files:
        raise FileNotFoundError(f"No image files found in source directory: {source_dir}")
    return max(files, key=file_timestamp)

def extract_output_text(resp) -> str:
    """
    The OpenAI Responses API returns content in output items.
    This helper tries to robustly pull all text from the response.
    """
    texts = []
    for item in getattr(resp, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            if getattr(c, "type", None) in ("output_text", "text"):
                t = getattr(c, "text", None)
                if t:
                    texts.append(t)
    return "\n".join(texts).strip()

def detect_country_with_openai(jpeg_path: Path, model: str = "gpt-5.3") -> dict:
    """
    Sends the envelope photo to the model and asks for destination country only.
    """
    client = OpenAI()

    img = data_url_for_jpeg(jpeg_path)

    instructions = (
        "You are reading a photo of a mailed envelope. "
        "Goal: identify the DESTINATION COUNTRY from the destination address. "
        "If multiple addresses are present, choose the recipient/destination address. "
        "Return ONLY strict JSON with keys: "
        '{"country":"...", "confidence":0-1, "notes":"short"} '
        "If you cannot determine, use country='UNKNOWN'."
    )

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "Extract the destination country from this envelope photo.",
                    },
                    {
                        "type": "input_image",
                        "image_url": img,
                        "detail": "high",
                    },
                ],
            },
        ],
    )

    text = extract_output_text(resp)
    if not text:
        return {"country": "UNKNOWN", "confidence": 0.0, "notes": "No text returned"}

    # Model should return JSON only; still, be defensive
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to salvage JSON if it included extra whitespace/newlines
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {"country": "UNKNOWN", "confidence": 0.0, "notes": f"Unparseable output: {text[:120]}"}

# -------- Main workflow --------

def unique_destination_path(dest_dir: Path, filename: str) -> Path:
    """
    Return a unique destination path by appending _2, _3, ... when needed.
    """
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    i = 2
    while True:
        alt = dest_dir / f"{stem}_{i}{suffix}"
        if not alt.exists():
            return alt
        i += 1


@app.command()
def main(
    input_path: Path | None = typer.Argument(
        None, help="Path to envelope photo (HEIC/PNG/JPG). Defaults to newest file in source dir."
    ),
):
    if input_path is None:
        input_path = newest_source_file(DEFAULT_SOURCE_DIR.expanduser().resolve())
        typer.echo(f"Using newest source file: {input_path}")
    else:
        input_path = input_path.expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: convert to JPEG (temporary name)
    tmp_jpeg = output_dir / (input_path.stem + "__tmp.jpg")
    convert_to_jpeg(input_path, tmp_jpeg)

    # Step 2: ask OpenAI for country
    result = detect_country_with_openai(tmp_jpeg, model=os.getenv("OPENAI_MODEL", "gpt-5"))
    country = safe_country_for_filename(result.get("country", "UNKNOWN"))
    conf = result.get("confidence", 0.0)

    # Step 3: rename final
    date_str = file_date_for_naming(input_path)
    final_name = f"{date_str}_{country}.jpg"
    final_path = unique_destination_path(output_dir, final_name)

    shutil.move(str(tmp_jpeg), str(final_path))

    typer.echo(f"Country: {result.get('country')} (confidence={conf})")
    typer.echo(f"Saved: {final_path}")

    # Step 4: move original HEIC only when confidence is high enough
    if conf > 0.75:
        if input_path.suffix.lower() == ".heic":
            DEFAULT_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            processed_dest = unique_destination_path(DEFAULT_PROCESSED_DIR, input_path.name)
            shutil.move(str(input_path), str(processed_dest))
            typer.echo(f"Moved original: {processed_dest}")
        else:
            typer.secho(
                "Confidence > 0.75, but original file is not HEIC; nothing moved.",
                fg=typer.colors.YELLOW,
            )
    else:
        typer.secho(
            f"Low confidence ({conf:.2f}); original file left in place: {input_path}",
            fg=typer.colors.YELLOW,
        )

if __name__ == "__main__":
    app()
