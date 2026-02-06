#!/usr/bin/env python3

import base64
import csv
import json
import os
import re
import time
import subprocess
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import pytesseract
import typer
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed


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

from openai import OpenAI

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


def _extract_season_year(text: str) -> Optional[str]:
    """Extract the start year from a season string like 2024-25, 2024/25, or 24-25.

    Cards may include multiple season references (stats tables, narrative blurbs, etc.).
    We choose the *most recent* start-year found (max), which is usually the current
    product season.
    """
    if not text:
        return None

    candidates: List[int] = []

    # Full season: 2024-25 or 2024/2025
    for m in re.finditer(r"((?:19|20)\d{2})\s*[-/]\s*(\d{2}|(?:19|20)\d{2})", text):
        try:
            candidates.append(int(m.group(1)))
        except Exception:
            pass

    # Short season: 24-25 or 24/25 (very common in stat tables)
    for m in re.finditer(r"\b(\d{2})\s*[-/]\s*(\d{2})\b", text):
        try:
            yy = int(m.group(1))
        except Exception:
            continue
        # Heuristic: treat 00-79 as 2000s (NBA modern era); otherwise 1900s.
        start = 2000 + yy if yy <= 79 else 1900 + yy
        candidates.append(start)

    if not candidates:
        return None
    return str(max(candidates))


# Prefer the set season year when it appears near the manufacturer name.
def _extract_set_season_year(text: str) -> Optional[str]:
    """Prefer the set season year when it appears near the manufacturer name.

    This avoids picking up seasons from stat tables (e.g., `2023-24`) when the card’s
    product line near the bottom says something like `2024-25 PANINI ...`.
    """
    if not text:
        return None

    brands = r"(?:panini|topps|upper\s*deck|donruss|fleer|bowman|leaf|score)"
    # Season then brand (common on card backs)
    pat_after = re.compile(
        rf"((?:19|20)\d{{2}})\s*[-/]\s*(\d{{2}}|(?:19|20)\d{{2}})[^\n]{{0,60}}\b{brands}\b",
        flags=re.IGNORECASE,
    )
    # Brand then season (less common)
    pat_before = re.compile(
        rf"\b{brands}\b[^\n]{{0,60}}((?:19|20)\d{{2}})\s*[-/]\s*(\d{{2}}|(?:19|20)\d{{2}})",
        flags=re.IGNORECASE,
    )

    found: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = pat_after.search(line)
        if m:
            found.append(m.group(1))
            continue
        m = pat_before.search(line)
        if m:
            found.append(m.group(1))

    return found[-1] if found else None


def _extract_copyright_year(text: str) -> Optional[str]:
    if not text:
        return None
    # Many cards use the © symbol rather than the word "copyright".
    trigger = re.compile(r"(?:copyright|\(c\)|©|\(©\))", flags=re.IGNORECASE)
    for line in text.splitlines():
        if trigger.search(line):
            years = re.findall(r"(?:19|20)\d{2}", line)
            if years:
                return years[-1]
    return None


# Helper for Topps: extract end year from short seasons like 24-25 or 24/25.
def _extract_short_season_end_year(text: str) -> Optional[str]:
    """Return the 4-digit *end* year from short seasons like 24-25 or 24/25.

    Example: '24-25' -> '2025'. We return the max end-year found.
    """
    if not text:
        return None

    candidates: List[int] = []
    for m in re.finditer(r"\b(\d{2})\s*[-/]\s*(\d{2})\b", text):
        try:
            yy2 = int(m.group(2))
        except Exception:
            continue
        end = 2000 + yy2 if yy2 <= 79 else 1900 + yy2
        candidates.append(end)

    if not candidates:
        return None
    return str(max(candidates))


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


# --- TCDB URL helpers ---
def _season_string_from_year(year_str: str, manufacturer: str) -> str:
    """Convert a 4-digit year into a TCDB-style season string like 2020-21.

    For Panini (NBA), filenames use the *start* year (2020 -> 2020-21).
    For Topps, some products may be stored as the *end* year in our filenames
    (2025 -> 2024-25), so we shift back one year.
    """
    try:
        year = int(re.search(r"(19|20)\d{2}", year_str or "").group(0))
    except Exception:
        return year_str or "Unknown"

    mfg = (manufacturer or "").strip().lower()
    # Heuristic: if Topps and year looks like an end-year, shift back one.
    start_year = year - 1 if mfg == "topps" else year

    end_yy = (start_year + 1) % 100
    return f"{start_year}-{end_yy:02d}"


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to the macOS clipboard (pbcopy). Returns True on success."""
    if not text:
        return False
    try:
        # macOS
        subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        return True
    except Exception:
        return False


def _tcdb_search_url_from_parts(
    year: str,
    last_name: str,
    manufacturer: str,
    series: str,
    variety: Optional[str],
    number: str,
    sport: str = "Basketball",
) -> str:
    """Build a TCDB Search.cfm URL from card parts."""
    season = _season_string_from_year(year, manufacturer)

    tokens: List[str] = [
        season,
        manufacturer or "",
        series or "",
    ]

    variety_value = _clean_variety(variety)
    if variety_value and variety_value.lower() not in {"base", "unknown", "none"}:
        tokens.append(variety_value)

    if number and number != "Unknown":
        tokens.append(str(number))

    if last_name and last_name != "Unknown":
        tokens.append(str(last_name))

    q = " ".join(t for t in tokens if t).strip()
    return f"https://www.tcdb.com/Search.cfm?SearchCategory={sport}&q={quote_plus(q)}"


def _parse_card_filename_for_tcdb(path_or_name: str, back_suffix: str = "_b") -> Dict[str, str]:
    """Parse a filename like:

    2020-Gordon-Panini-Mosaic-Genesis-22.jpg

    Returns a dict compatible with _tcdb_search_url_from_parts.

    Notes:
    - We treat the pattern as: YEAR - LAST - MFG - SERIES - (VARIETY...)? - NUMBER
    - If there are extra hyphen-separated parts between SERIES and NUMBER, we join
      them into VARIETY.
    """
    name = os.path.basename(path_or_name)
    stem = re.sub(r"\.(jpg|jpeg)$", "", name, flags=re.IGNORECASE)

    # Drop back image suffix if present
    if stem.endswith(back_suffix):
        stem = stem[: -len(back_suffix)]

    parts = [p for p in stem.split("-") if p]

    out = {
        "year": "Unknown",
        "last_name": "Unknown",
        "manufacturer": "Unknown",
        "series": "Unknown",
        "variety": "",
        "number": "Unknown",
    }

    if len(parts) < 5:
        return out

    out["year"] = parts[0]
    out["last_name"] = parts[1]
    out["manufacturer"] = parts[2]
    out["series"] = parts[3]
    out["number"] = parts[-1]

    # Variety is everything between series and number
    if len(parts) > 5:
        out["variety"] = "-".join(parts[4:-1])
    else:
        # exactly 5 parts -> no explicit variety
        out["variety"] = "Base"

    return out


def _tcdb_search_url_from_filename(path_or_name: str, back_suffix: str = "_b") -> str:
    info = _parse_card_filename_for_tcdb(path_or_name, back_suffix=back_suffix)
    return _tcdb_search_url_from_parts(
        year=info.get("year", "Unknown"),
        last_name=info.get("last_name", "Unknown"),
        manufacturer=info.get("manufacturer", "Unknown"),
        series=info.get("series", "Unknown"),
        variety=info.get("variety") or None,
        number=info.get("number", "Unknown"),
    )


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
    combined_text = "\n".join(
        part
        for part in (ocr_front, ocr_back, ocr_back_bottom, details.get("year", ""))
        if part
    )

    # Prefer the product-line season (often near the bottom and near the maker name),
    # rather than a stats season or narrative season that may refer to a different year.
    set_season_year = (
        _extract_set_season_year(ocr_back_bottom)
        or _extract_set_season_year(ocr_back)
        or _extract_set_season_year(ocr_front)
    )
    copyright_year = _extract_copyright_year(combined_text)

    if set_season_year:
        details["year"] = set_season_year
    elif copyright_year:
        # If the set-season isn't explicitly printed near the maker, copyright is the
        # most reliable indicator and avoids grabbing older seasons mentioned in blurbs.
        details["year"] = copyright_year
    else:
        # Topps Chrome (and some other Topps products) often do NOT print an explicit
        # product season like "2024-25 TOPPS". In those cases, the back typically has a
        # short season in the stats row (e.g., 24-25). For Topps, we treat the product
        # year as the *end* year (24-25 -> 2025), which aligns with the copyright year
        # when it is present.
        is_topps = (
            (details.get("manufacturer") or "").strip().lower() == "topps"
            or "topps" in combined_text.lower()
        )
        if is_topps:
            end_year = _extract_short_season_end_year(combined_text)
            if end_year:
                details["year"] = end_year
            else:
                season_year = _extract_season_year(combined_text)
                if season_year:
                    details["year"] = season_year
        else:
            season_year = _extract_season_year(combined_text)
            if season_year:
                details["year"] = season_year
        if details.get("year") in (None, ""):
            details["year"] = "Unknown"
        if not details.get("year") or details.get("year") == "Unknown":
            bottom_year = _normalize_year(None, ocr_back_bottom, prefer_last=True)
            if bottom_year != "Unknown":
                details["year"] = bottom_year
            else:
                fallback_text = f"{ocr_back} {ocr_front}".strip()
                details["year"] = _normalize_year(
                    details.get("year"), fallback_text, prefer_last=True
                )
    for key in ("last_name", "manufacturer", "series", "number"):
        details.setdefault(key, "Unknown")
    return details


@app.command()
def main(
    directory: Path = typer.Option(Path.cwd(), "--directory", "-d", help="Directory with card_*.jpg files"),
    rename: bool = typer.Option(False, help="Rename files in place"),
    usecsv: bool = typer.Option(
        False, help="Rename using the existing card_names.csv and remove it when done"
    ),
    back_suffix: str = typer.Option("_b", help="Suffix for back images"),
    model: str = typer.Option("gpt-5.2", help="OpenAI model"),
    max_size: int = typer.Option(1024, help="Max image width before upload"),
    quality: int = typer.Option(85, help="JPEG quality for uploads"),
    workers: int = typer.Option(4, help="Number of worker threads"),
    variety: Optional[str] = typer.Option(None, help="Override variety to include in filename"),
    pattern: str = typer.Option(
        "card_*", help="Glob pattern(s) for input images; comma-separated (ignored with --images)"
    ),
    images: Tuple[str, str] = typer.Option(
        ("", ""),
        "--images",
        help="Process a single card by providing front and back image paths (front first, then back).",
    ),
    output_csv: Path = typer.Option("card_names.csv", help="CSV output path"),
    tcdb: Optional[List[str]] = typer.Option(
        None,
        "--tcdb",
        help=(
            "Given one or more existing card filenames, print TCDB search URL(s) "
            "and copy the first to the clipboard. Repeat --tcdb for multiple."
        ),
    ),
):
    """
    Generate card filenames from paired images (front + back).
    Use --images FRONT BACK to process a single card.
    """
    directory = directory.expanduser().resolve()
    if not directory.exists():
        raise typer.BadParameter(f"Directory not found: {directory}")

    output_csv = output_csv.expanduser()
    if not output_csv.is_absolute():
        output_csv = directory / output_csv

    dry_run = not rename and not usecsv

    # TCDB helper mode: build search URL(s) from existing filename(s).
    if tcdb:
        urls: List[str] = []
        for name in tcdb:
            url = _tcdb_search_url_from_filename(name, back_suffix=back_suffix)
            urls.append(url)
            typer.echo(url)

        # Copy the first URL to the clipboard (convenient default).
        first_url = urls[0] if urls else ""
        if first_url and _copy_to_clipboard(first_url):
            if len(urls) == 1:
                typer.echo("(Copied TCDB search URL to clipboard)")
            else:
                typer.echo(f"(Copied first TCDB search URL to clipboard; {len(urls)} total printed)")
        else:
            typer.echo("(Could not copy to clipboard)")

        raise typer.Exit(code=0)

    if usecsv:
        failures = 0
        if not output_csv.exists():
            raise typer.BadParameter(f"CSV not found: {output_csv}")
        with open(output_csv, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                front_name = row.get("front")
                back_name = row.get("back")
                new_front = row.get("new_front")
                new_back = row.get("new_back")
                if not (front_name and back_name and new_front and new_back):
                    failures += 1
                    typer.echo(f"Skipping row with missing names: {row}")
                    continue
                front_path = directory / front_name
                back_path = directory / back_name
                front_target = directory / new_front
                back_target = directory / new_back

                if front_path.exists():
                    if front_path.resolve() != front_target.resolve():
                        front_path.rename(front_target)
                elif not front_target.exists():
                    failures += 1
                    typer.echo(f"Missing front image: {front_name}")

                if back_path.exists():
                    if back_path.resolve() != back_target.resolve():
                        back_path.rename(back_target)
                elif not back_target.exists():
                    failures += 1
                    typer.echo(f"Missing back image: {back_name}")

                # Build a TCDB search URL from the new filename and copy it.
                tcdb_url = _tcdb_search_url_from_filename(new_front, back_suffix=back_suffix)
                typer.echo(tcdb_url)
                _copy_to_clipboard(tcdb_url)

        if failures:
            typer.echo("Renaming completed with errors; CSV was not removed.")
            raise typer.Exit(code=1)
        output_csv.unlink()
        typer.echo("Renamed files from CSV and removed card_names.csv.")
        raise typer.Exit(code=0)

    if images[0] or images[1]:
        if not images[0] or not images[1]:
            raise typer.BadParameter("Provide both front and back images for --images.")
        front_input, back_input = images
        resolved: List[Path] = []
        for raw in (front_input, back_input):
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = directory / path
            path = path.resolve()
            if not path.exists() or not path.is_file():
                raise typer.BadParameter(f"Image not found: {raw}")
            if path.parent != directory:
                raise typer.BadParameter(
                    f"Image {path.name} is not in {directory}. "
                    "When using --images, point --directory at the image folder."
                )
            resolved.append(path)
        image_files = resolved
    else:
        image_files = _collect_images(directory, pattern)
    if not image_files:
        typer.echo("No images found.")
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
            if dry_run:
                typer.echo(f"Dry run: {front.name} -> {new_front}, {back.name} -> {new_back}")
                tcdb_url = _tcdb_search_url_from_filename(new_front, back_suffix=back_suffix)
                typer.echo(f"TCDB: {tcdb_url}")

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
            # After renaming, build a TCDB search URL for the card and copy it.
            tcdb_url = _tcdb_search_url_from_filename(row["new_front"], back_suffix=back_suffix)
            typer.echo(tcdb_url)
            _copy_to_clipboard(tcdb_url)

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
        typer.echo("Dry run only. Re-run with --rename to apply or --usecsv to rename from the CSV.")


if __name__ == "__main__":
    app()
