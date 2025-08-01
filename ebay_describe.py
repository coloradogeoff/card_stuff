#!/usr/bin/env python

import base64
import typer
import os
import re
import yaml
from pathlib import Path
from PIL import Image
import pytesseract
from openai import OpenAI
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

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
        
def build_messages(examples: list[str], ocr_text: str, postmark: str, user_condition: str, images: list[dict], category: str = "postcards") -> list:
    system = load_instructions(category)

    if ocr_text:
        system += f"Text found on postcard front: {ocr_text}\n"
    if postmark:
        system += f"Postmark info: {postmark}\n"
    if user_condition:
        system += f"Seller notes: {user_condition}\n"

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

def process_pair(front: Path, back: Path, compress: bool, category: str = "postcards") -> dict:
    typer.echo(f"Processing {front.name} ...")
    if not (validate_image(front) and validate_image(back)):
        return None
    ocr_text = extract_ocr_text(front)
    images = [
        {"desc": "This is the front of the postcard.", "data": image_to_base64(front, compress=compress)},
        {"desc": "This is the back of the postcard.", "data": image_to_base64(back, compress=compress)},
    ]
    messages = build_messages([], ocr_text, "", "", images, category)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=500
    )
    content = response.choices[0].message.content.strip()
    # Parse Title and Condition from output
    lines = content.splitlines()
    print(lines)
    title = ""
    cond = ""
    # Match lines like '1. Title:', '**Title:**', 'Title:', etc.
    title_pattern = re.compile(r"^\s*(?:\d+\.\s*)?\**\s*title\s*:\s*(.+)$", re.IGNORECASE)
    cond_pattern = re.compile(r"^\s*(?:\d+\.\s*)?\**\s*condition\s*:\s*(.+)$", re.IGNORECASE)
    for line in lines:
        title_match = title_pattern.match(line)
        cond_match = cond_pattern.match(line)
        if title_match:
            title = title_match.group(1).strip()
            title = title.replace("*", "")  # Remove any asterisks
        if cond_match:
            cond = cond_match.group(1).strip()
            cond = cond.replace("*", "")  # Remove any asterisks
    return {
        "front": str(front.name),
        "title": title,
        "condition": cond
    }


@app.command()
def describe(
    directory: Path = typer.Argument(..., help="Directory containing image files"),
    compress: bool = typer.Option(True, help="Compress and resize images before upload"),
    sports: bool = typer.Option(False, help="Use sports card rules"),
    postal: bool = typer.Option(False, help="Use postal history rules")
):
    """
    Generates descriptions for image pairs in a directory and writes results to descriptions.csv.
    """
    # Determine category for instructions
    if sports:
        category = "sports_cards"
    elif postal:
        category = "postal_history"
    else:
        category = "postcards"

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
        futures = [executor.submit(process_pair, front, back, compress, category) for front, back in pairs]
        for future in futures:
            res = future.result()
            if res is not None:
                results.append(res)
    # Write to CSV
    import csv
    csv_path = directory / "descriptions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["front", "title", "condition"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    typer.echo(f"Generated descriptions.csv with {len(results)} entries.")

if __name__ == "__main__":
    app()