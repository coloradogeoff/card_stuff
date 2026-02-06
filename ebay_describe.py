#!/usr/bin/env python

import base64
import csv
import os
import re
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from threading import Lock
from openai import OpenAI

import pytesseract
import typer
import yaml
from PIL import Image


def _ensure_typing_extensions_override():
    # Work around older typing_extensions lacking override (used by openai).
    try:
        import typing_extensions
    except Exception:
        return
    if not hasattr(typing_extensions, "override"):
        def override(func):
            return func
        typing_extensions.override = override


_ensure_typing_extensions_override()


def load_instructions(category: str) -> str:
    rules_path = Path(__file__).parent / "instructions.yaml"
    with open(rules_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get(category, "")

def resize_image(input_path: Path, max_size=1024) -> Image.Image:
    img = Image.open(input_path)
    if img.width > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img

def compress_image(img: Image.Image, quality=85) -> bytes:
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()

app = typer.Typer()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Simple global rate limiter for OpenAI calls ---
_rate_lock = Lock()
_last_call_time = 0.0
_MIN_INTERVAL = 0.5  # seconds between calls (tune this as needed)

def chat_with_openai(messages):
    """Thread-safe wrapper that enforces a minimum delay between API calls."""
    global _last_call_time

    with _rate_lock:
        now = time.time()
        wait = _MIN_INTERVAL - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()

    return client.chat.completions.create(
        model="gpt-5.2",
        messages=messages,
        max_completion_tokens=500
    )

def validate_image(path: Path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as e:
        typer.echo(f"Invalid image {path}: {e}")
        return False

def extract_ocr_text(image_path: Path) -> str:
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        typer.echo(f"OCR failed: {e}")
        return ""

def image_to_base64(image_path: Path, compress=True) -> str:
    if compress:
        resized = resize_image(image_path)
        jpeg_bytes = compress_image(resized)
        return base64.b64encode(jpeg_bytes).decode("utf-8")
    else:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
        
def build_messages(
    examples: list[str],
    ocr_text: str,
    postmark: str,
    user_condition: str,
    images: list[dict],
    category: str = "postcards",
    set_override: Optional[str] = None,
) -> list:
    system = load_instructions(category)

    if ocr_text:
        system += f"Text found on the front of the item: {ocr_text}\n"
    if postmark:
        system += f"Postmark info: {postmark}\n"
    if user_condition:
        system += f"Seller notes: {user_condition}\n"
    if category == "sports_cards" and set_override:
        system += (
            "Card set override: Use exactly the following card set string in the title "
            f"(season years + manufacturer + set name): {set_override}\n"
        )

    messages = [{"role": "system", "content": system}]

    for image in images:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": image["desc"]},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image['data']}"}}
            ]
        })

    return messages

def process_pair(
    front: Path,
    back: Path,
    compress: bool,
    category: str = "postcards",
    set_override: Optional[str] = None,
) -> dict:
    typer.echo(f"Processing {front.name} ...")
    if not (validate_image(front) and validate_image(back)):
        return None
    ocr_text = extract_ocr_text(front)
    images = [
        {"desc": "This is the front of the item.", "data": image_to_base64(front, compress=compress)},
        {"desc": "This is the back of the item.", "data": image_to_base64(back, compress=compress)},
    ]
    
    messages = build_messages([], ocr_text, "", "", images, category, set_override)
    response = chat_with_openai(messages)

    content = response.choices[0].message.content.strip()
    # Parse Title from output
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    print(lines)
    title = ""
    # Match lines like '1. Title:', '**Title:**', 'Title:', etc.
    title_pattern = re.compile(r"^\s*(?:\d+\.\s*)?\**\s*title\s*:\s*(.+)$", re.IGNORECASE)
    for line in lines:
        title_match = title_pattern.match(line)
        if title_match:
            title = title_match.group(1).strip()
            title = title.replace("*", "")  # Remove any asterisks
            break
    if not title and lines:
        title = lines[0].replace("*", "")
    return {
        "front": str(front.name),
        "title": title
    }


@app.callback(invoke_without_command=True)
def describe(
    compress: bool = typer.Option(True, help="Compress and resize images before upload"),
    sports: bool = typer.Option(True, help="Use sports card rules"),
    postal: bool = typer.Option(False, help="Use postal history rules"),
    postcards: bool = typer.Option(False, help="Use postcard rules"),
    set_override: Optional[str] = typer.Option(
        None,
        "--set",
        help="Override the card set (season years, manufacturer, set name) for sports cards",
    ),
    viewer: bool = typer.Option(False, "--viewer", "-v", help="Open the viewer after generating the CSV")
):
    """
    Generates titles for image pairs in the current directory and writes results to description.csv.
    """
    directory = Path.cwd()
    # Determine category for instructions
    if postal:
        category = "postal_history"
    elif postcards:
        category = "postcards"
    else:
        category = "sports_cards"

    # Find all card_*.jpg files
    image_files = sorted(directory.glob("*.jpg"))
    # Build list of (front, back) pairs
    pairs = []
    for i in range(0, len(image_files), 2):
        if i+1 >= len(image_files):
            # Skip if there's an unmatched front/back
            continue
        front = image_files[i]
        back = image_files[i+1]
        pairs.append((front, back))
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Pass category to process_pair
        futures = [
            executor.submit(process_pair, front, back, compress, category, set_override)
            for front, back in pairs
        ]
        for future in futures:
            res = future.result()
            if res is not None:
                results.append(res)
    
    csv_path = directory / "description.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["front", "title"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    typer.echo(f"Generated description.csv with {len(results)} entries.")
    if viewer:
        import viewer as viewer_app
        viewer_app.main(str(csv_path))

if __name__ == "__main__":
    app()
