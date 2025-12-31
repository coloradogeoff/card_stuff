#!/usr/bin/env python3

import base64
import csv
import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

import pytesseract
import typer
from PIL import Image
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


app = typer.Typer(add_completion=False)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Simple global rate limiter for OpenAI calls ---
_rate_lock = Lock()
_last_call_time = 0.0
_MIN_INTERVAL = 0.5


def _rate_limited_chat(messages, model: str):
    global _last_call_time
    with _rate_lock:
        now = time.time()
        wait = _MIN_INTERVAL - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()
    return client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=400,
    )


def _resize_image(input_path: Path, max_size: int) -> Image.Image:
    img = Image.open(input_path)
    if img.width > max_size:
        img.thumbnail((max_size, max_size), Image.LANCZOS)
    return img


def _compress_image(img: Image.Image, quality: int) -> bytes:
    buffer = BytesIO()
    img.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def _image_to_base64(image_path: Path, max_size: int, quality: int) -> str:
    resized = _resize_image(image_path, max_size)
    jpeg_bytes = _compress_image(resized, quality)
    return base64.b64encode(jpeg_bytes).decode("utf-8")


def _validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception as exc:
        typer.echo(f"Invalid image {path.name}: {exc}")
        return False


def _extract_ocr_text(image_path: Path, crop: Optional[str] = None) -> str:
    try:
        img = Image.open(image_path)
        if crop == "bottom":
            width, height = img.size
            top = int(height * 0.75)
            img = img.crop((0, top, width, height))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as exc:
        typer.echo(f"OCR failed for {image_path.name}: {exc}")
        return ""


def _build_messages(
    ocr_front: str,
    ocr_back: str,
    ocr_back_bottom: str,
    front_b64: str,
    back_b64: str,
) -> List[Dict]:
    system = (
        "You are a sports card identification assistant. Using the images and OCR text, "
        "extract: year (4-digit start year of the season), last_name (player last name only), "
        "manufacturer (e.g., Topps, Panini), series (e.g., Chrome, Select, Mosaic), and number "
        "(card number only, no #). Return ONLY a JSON object with keys: "
        "year, last_name, manufacturer, series, number. If unknown, use 'Unknown'."
    )
    if ocr_front:
        system += f"\nOCR front text:\n{ocr_front}\n"
    if ocr_back:
        system += f"\nOCR back text:\n{ocr_back}\n"
    if ocr_back_bottom:
        system += (
            "\nOCR back bottom text (often includes the card year):\n"
            f"{ocr_back_bottom}\n"
        )

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Front of card."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{front_b64}"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Back of card."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{back_b64}"}},
            ],
        },
    ]


def _strip_code_fences(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        value = re.sub(r"^```[a-zA-Z]*\n", "", value)
        value = value.rstrip("`").strip()
    return value


def _parse_response(content: str) -> Dict[str, str]:
    cleaned = _strip_code_fences(content)
    data: Dict[str, str] = {}
    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            for key in ("year", "last_name", "manufacturer", "series", "number"):
                value = payload.get(key)
                if isinstance(value, (str, int, float)):
                    data[key] = str(value).strip()
    except json.JSONDecodeError:
        pass
    if not data:
        for line in cleaned.splitlines():
            match = re.match(r"^\s*([a-z_ ]+)\s*:\s*(.+)\s*$", line, flags=re.IGNORECASE)
            if match:
                key = match.group(1).strip().lower().replace(" ", "_")
                if key in {"year", "last_name", "manufacturer", "series", "number"}:
                    data[key] = match.group(2).strip()
    return data


def _normalize_year(value: Optional[str], fallback_text: str, prefer_last: bool = False) -> str:
    if value:
        match = re.search(r"(?:19|20)\d{2}", value)
        if match:
            return match.group(0)
    years = re.findall(r"(?:19|20)\d{2}", fallback_text)
    if years:
        return years[-1] if prefer_last else years[0]
    return "Unknown"


def _extract_last_name(value: Optional[str]) -> str:
    if not value:
        return "Unknown"
    tokens = [t for t in re.split(r"\s+", value.strip()) if t]
    if not tokens:
        return "Unknown"
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    last = tokens[-1].rstrip(".")
    if last.lower() in suffixes and len(tokens) > 1:
        last = tokens[-2].rstrip(".")
    return last


def _slugify(value: str) -> str:
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "-", value).strip("-")
    return value or "Unknown"


def _clean_variety(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    compact = re.sub(r"\s+", "", value.strip())
    return compact or None


def _build_base_name(details: Dict[str, str], variety: Optional[str]) -> str:
    year = _normalize_year(details.get("year"), " ".join(details.values()))
    last_name = _extract_last_name(details.get("last_name"))
    manufacturer = details.get("manufacturer") or "Unknown"
    series = details.get("series") or "Unknown"
    number = details.get("number") or "Unknown"
    parts = [
        _slugify(year),
        _slugify(last_name),
        _slugify(manufacturer),
        _slugify(series),
    ]
    variety_value = _clean_variety(variety)
    if variety_value:
        parts.append(_slugify(variety_value))
    parts.append(_slugify(number))
    return "-".join(parts)


def _find_available_base(directory: Path, base: str, back_suffix: str) -> str:
    candidate = base
    counter = 2
    while True:
        front_path = directory / f"{candidate}.jpg"
        back_path = directory / f"{candidate}{back_suffix}.jpg"
        if not front_path.exists() and not back_path.exists():
            return candidate
        candidate = f"{base}-{counter}"
        counter += 1


def _pattern_has_extension(pattern: str) -> bool:
    lowered = pattern.lower()
    return ".jpg" in lowered or ".jpeg" in lowered


def _collect_images(directory: Path, pattern: str) -> List[Path]:
    patterns = [part.strip() for part in pattern.split(",") if part.strip()]
    if not patterns:
        patterns = ["card_*"]
    images_set = set()
    for item in patterns:
        if _pattern_has_extension(item):
            for path in directory.glob(item):
                if path.is_file():
                    images_set.add(path)
        else:
            for ext in (".jpg", ".jpeg"):
                for path in directory.glob(f"{item}{ext}"):
                    if path.is_file():
                        images_set.add(path)
    images = list(images_set)

    def sort_key(path: Path) -> Tuple[int, int, str]:
        match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
        if match:
            return (0, int(match.group(1)), path.name)
        return (1, 0, path.name)

    return sorted(images, key=sort_key)


def _describe_pair(
    front: Path,
    back: Path,
    model: str,
    max_size: int,
    quality: int,
) -> Dict[str, str]:
    typer.echo(f"Processing {front.name} + {back.name} ...")
    if not (_validate_image(front) and _validate_image(back)):
        return {}

    ocr_front = _extract_ocr_text(front)
    ocr_back = _extract_ocr_text(back)
    ocr_back_bottom = _extract_ocr_text(back, crop="bottom")
    front_b64 = _image_to_base64(front, max_size=max_size, quality=quality)
    back_b64 = _image_to_base64(back, max_size=max_size, quality=quality)
    messages = _build_messages(ocr_front, ocr_back, ocr_back_bottom, front_b64, back_b64)
    response = _rate_limited_chat(messages, model=model)
    content = response.choices[0].message.content or ""
    details = _parse_response(content)
    if not details:
        details = {
            "year": "Unknown",
            "last_name": "Unknown",
            "manufacturer": "Unknown",
            "series": "Unknown",
            "number": "Unknown",
        }
    bottom_year = _normalize_year(None, ocr_back_bottom, prefer_last=True)
    if bottom_year != "Unknown":
        details["year"] = bottom_year
    else:
        fallback_text = f"{ocr_back} {ocr_front}".strip()
        details["year"] = _normalize_year(details.get("year"), fallback_text, prefer_last=True)
    for key in ("last_name", "manufacturer", "series", "number"):
        details.setdefault(key, "Unknown")
    return details


@app.command()
def main(
    directory: Path = typer.Option(Path.cwd(), "--directory", "-d", help="Directory with card_*.jpg files"),
    rename: bool = typer.Option(False, help="Rename files in place"),
    back_suffix: str = typer.Option("_b", help="Suffix for back images"),
    model: str = typer.Option("gpt-5.2", help="OpenAI model"),
    max_size: int = typer.Option(1024, help="Max image width before upload"),
    quality: int = typer.Option(85, help="JPEG quality for uploads"),
    workers: int = typer.Option(4, help="Number of worker threads"),
    variety: Optional[str] = typer.Option(None, help="Override variety to include in filename"),
    pattern: str = typer.Option("card_*", help="Glob pattern(s) for input images; comma-separated"),
    output_csv: Path = typer.Option("card_names.csv", help="CSV output path"),
):
    """
    Generate card filenames from paired images (front + back).
    """
    directory = directory.expanduser().resolve()
    if not directory.exists():
        raise typer.BadParameter(f"Directory not found: {directory}")

    image_files = _collect_images(directory, pattern)
    if not image_files:
        typer.echo("No card_*.jpg images found.")
        raise typer.Exit(code=1)

    pairs = []
    for i in range(0, len(image_files), 2):
        if i + 1 >= len(image_files):
            typer.echo(f"Skipping unmatched image: {image_files[i].name}")
            continue
        pairs.append((image_files[i], image_files[i + 1]))

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_describe_pair, front, back, model, max_size, quality): (index, front, back)
            for index, (front, back) in enumerate(pairs)
        }
        for future in as_completed(future_map):
            index, front, back = future_map[future]
            details = future.result()
            base = _build_base_name(details, variety)
            available_base = _find_available_base(directory, base, back_suffix)
            new_front = f"{available_base}.jpg"
            new_back = f"{available_base}{back_suffix}.jpg"
            variety_value = _clean_variety(variety) or ""

            row = {
                "index": index,
                "front": front.name,
                "back": back.name,
                "year": details.get("year", ""),
                "last_name": details.get("last_name", ""),
                "manufacturer": details.get("manufacturer", ""),
                "series": details.get("series", ""),
                "variety": variety_value,
                "number": details.get("number", ""),
                "new_front": new_front,
                "new_back": new_back,
            }
            results.append(row)

    results.sort(key=lambda item: item["index"])
    for row in results:
        if rename:
            front_target = directory / row["new_front"]
            back_target = directory / row["new_back"]
            front_path = directory / row["front"]
            back_path = directory / row["back"]
            if front_path.resolve() != front_target.resolve():
                front_path.rename(front_target)
            if back_path.resolve() != back_target.resolve():
                back_path.rename(back_target)

    output_csv = output_csv.expanduser()
    if not output_csv.is_absolute():
        output_csv = directory / output_csv
    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "front",
                "back",
                "year",
                "last_name",
                "manufacturer",
                "series",
                "variety",
                "number",
                "new_front",
                "new_back",
            ],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        for row in results:
            row_copy = dict(row)
            row_copy.pop("index", None)
            writer.writerow(row_copy)

    typer.echo(f"Wrote {output_csv} with {len(results)} entries.")
    if rename:
        typer.echo("Renamed files in place.")
    else:
        typer.echo("Dry run only. Re-run with --rename to apply.")


if __name__ == "__main__":
    app()
