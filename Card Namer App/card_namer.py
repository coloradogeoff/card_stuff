#!/usr/bin/env python3

import base64
import csv
import errno
import json
import os
import re
import shutil
import sys
import time
import subprocess
import traceback
import webbrowser
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import pytesseract
import typer
from PyQt5.QtCore import QFileSystemWatcher, QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QGuiApplication, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)
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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PREVIEW_HEIGHT = 800
GUI_MODEL = "gpt-5.2"


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


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class NameWorker(QObject):
    finished = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, front: Path, back: Path):
        super().__init__()
        self.front = front
        self.back = back

    def run(self):
        try:
            details = _describe_pair(
                self.front,
                self.back,
                model=GUI_MODEL,
                max_size=1024,
                quality=85,
            )
            proposed = _build_base_name(details, variety=None)
            self.finished.emit(proposed)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")


def _default_cards_directory() -> Path:
    target = (Path.home() / "Desktop" / "incoming cards").expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return target


def _existing_cards_directory() -> Path:
    return (Path.home() / "Desktop" / "Cards").expanduser()


def _natural_sort_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def _list_gui_images(directory: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: _natural_sort_key(path.name),
    )


def _is_back_image(path: Path) -> bool:
    return path.stem.lower().endswith("_b")


def _build_gui_pairs(image_files: List[Path]) -> List[Tuple[Path, Path]]:
    if not image_files:
        return []

    pairs: List[Tuple[Path, Path]] = []
    used: set[Path] = set()

    fronts_by_stem: Dict[str, List[Path]] = {}
    for path in image_files:
        if _is_back_image(path):
            continue
        fronts_by_stem.setdefault(path.stem.lower(), []).append(path)

    for stem in fronts_by_stem:
        fronts_by_stem[stem].sort(key=lambda item: _natural_sort_key(item.name))

    backs = [path for path in image_files if _is_back_image(path)]
    backs.sort(key=lambda item: _natural_sort_key(item.name))

    for back in backs:
        base_stem = back.stem[:-2].lower()
        candidates = fronts_by_stem.get(base_stem, [])
        front = next((item for item in candidates if item not in used), None)
        if front is None:
            continue
        pairs.append((front, back))
        used.add(front)
        used.add(back)

    remaining = [path for path in image_files if path not in used]
    remaining.sort(key=lambda item: _natural_sort_key(item.name))
    for index in range(0, len(remaining), 2):
        if index + 1 >= len(remaining):
            break
        first = remaining[index]
        second = remaining[index + 1]
        if _is_back_image(first) and not _is_back_image(second):
            first, second = second, first
        pairs.append((first, second))

    return sorted(
        pairs,
        key=lambda pair: max(pair[0].stat().st_mtime, pair[1].stat().st_mtime),
        reverse=True,
    )


def _sanitize_base_name(value: str) -> str:
    value = value.strip().strip(".")
    value = re.sub(r"[\\/:\*\?\"<>\|]+", "-", value)
    value = re.sub(r"\s+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    return value


def _pair_base_name(front: Path, back: Path) -> str:
    if _is_back_image(front) and front.stem[:-2].lower() == back.stem.lower():
        return back.stem
    if _is_back_image(back) and back.stem[:-2].lower() == front.stem.lower():
        return front.stem
    if not _is_back_image(front):
        return front.stem
    if not _is_back_image(back):
        return back.stem
    return front.stem[:-2]


def _move_file_overwrite(source: Path, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        source.replace(target)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
    if target.exists():
        target.unlink()
    shutil.move(str(source), str(target))


def _name_to_search_terms(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    chars: List[str] = []
    length = len(value)
    for index, char in enumerate(value):
        if char != "-":
            chars.append(char)
            continue
        prev_is_digit = index > 0 and value[index - 1].isdigit()
        next_is_digit = index + 1 < length and value[index + 1].isdigit()
        if prev_is_digit and next_is_digit:
            chars.append(char)
        else:
            chars.append(" ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()


class CardNamerGui(QWidget):
    def __init__(self):
        super().__init__()
        self.thread: Optional[QThread] = None
        self.worker: Optional[NameWorker] = None
        self.current_front_path: Optional[Path] = None
        self.current_back_path: Optional[Path] = None
        self.current_showing_back = False
        self.image_pairs: List[Tuple[Path, Path]] = []
        self.incoming_cards_directory = _default_cards_directory()
        self.existing_cards_directory = _existing_cards_directory()
        self.watched_directory: Optional[Path] = None
        self.pending_auto_refresh = False
        self.fs_watcher = QFileSystemWatcher(self)
        self.fs_watcher.directoryChanged.connect(self.on_watched_directory_changed)
        self.refresh_debounce = QTimer(self)
        self.refresh_debounce.setSingleShot(True)
        self.refresh_debounce.setInterval(400)
        self.refresh_debounce.timeout.connect(self.run_auto_refresh)

        self.setWindowTitle("Card Namer")
        self.resize(1200, 800)
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            self.move(screen.availableGeometry().topLeft())
        else:
            self.move(0, 0)

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Image Directory"))
        self.directory_edit = QLineEdit(str(self.incoming_cards_directory))
        top.addWidget(self.directory_edit, stretch=1)
        self.incoming_button = QPushButton("Incoming Cards")
        self.incoming_button.clicked.connect(self.use_incoming_directory)
        top.addWidget(self.incoming_button)
        self.existing_button = QPushButton("Existing Card Images")
        self.existing_button.clicked.connect(self.use_existing_directory)
        top.addWidget(self.existing_button)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.choose_directory)
        top.addWidget(self.browse_button)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_images)
        top.addWidget(self.refresh_button)
        root.addLayout(top)

        middle = QHBoxLayout()
        left = QVBoxLayout()
        self.preview_side_label = QLabel("Front (click image to toggle)")
        self.preview_side_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.preview_side_label)
        self.preview_label = ClickableLabel("Select a card pair")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumWidth(680)
        self.preview_label.setFixedHeight(PREVIEW_HEIGHT)
        self.preview_label.setStyleSheet("border: 1px solid #cccccc; background: #f5f5f5;")
        self.preview_label.clicked.connect(self.toggle_preview_side)
        left.addWidget(self.preview_label)
        middle.addLayout(left, stretch=2)

        right = QVBoxLayout()
        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_image_selected)
        right.addWidget(self.image_list, stretch=1)
        move_buttons = QHBoxLayout()
        self.move_selected_button = QPushButton("Move Card")
        self.move_selected_button.clicked.connect(self.move_selected_card)
        move_buttons.addWidget(self.move_selected_button)
        self.move_all_button = QPushButton("Move All Cards")
        self.move_all_button.clicked.connect(self.move_all_cards)
        move_buttons.addWidget(self.move_all_button)
        right.addLayout(move_buttons)
        middle.addLayout(right, stretch=1)
        root.addLayout(middle, stretch=1)

        self.start_button = QPushButton("Start Naming")
        self.start_button.setStyleSheet("background-color: #b9f6ca;")
        self.start_button.clicked.connect(self.start_naming)
        root.addWidget(self.start_button)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Proposed Name"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Generated card name appears here")
        self.name_edit.textChanged.connect(self.update_action_buttons)
        name_row.addWidget(self.name_edit, stretch=1)
        root.addLayout(name_row)

        self.accept_button = QPushButton("Accept Name")
        self.accept_button.setStyleSheet("background-color: #4caf50; color: white;")
        self.accept_button.clicked.connect(self.accept_name)
        root.addWidget(self.accept_button)

        search_row = QHBoxLayout()
        self.tcdb_button = QPushButton("Open TCDB Search")
        self.tcdb_button.clicked.connect(self.open_tcdb_search)
        search_row.addWidget(self.tcdb_button)
        self.ebay_button = QPushButton("Search eBay")
        self.ebay_button.clicked.connect(self.open_ebay_search)
        search_row.addWidget(self.ebay_button)
        root.addLayout(search_row)

        self.feedback = QPlainTextEdit()
        self.feedback.setReadOnly(True)
        root.addWidget(self.feedback, stretch=1)

        self.refresh_images()

    def append_log(self, message: str):
        self.feedback.appendPlainText(message)

    def choose_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Card Directory", self.directory_edit.text()
        )
        if directory:
            self.directory_edit.setText(directory)
            self.refresh_images()

    def use_incoming_directory(self):
        self.directory_edit.setText(str(self.incoming_cards_directory))
        self.refresh_images()

    def use_existing_directory(self):
        self.directory_edit.setText(str(self.existing_cards_directory))
        self.refresh_images()

    def set_watched_directory(self, directory: Optional[Path]):
        current_paths = self.fs_watcher.directories()
        if current_paths:
            self.fs_watcher.removePaths(current_paths)
        self.watched_directory = None
        if directory is None:
            return
        directory_str = str(directory)
        if directory.exists() and directory.is_dir():
            self.fs_watcher.addPath(directory_str)
            self.watched_directory = directory

    def on_watched_directory_changed(self, _path: str):
        if self.thread is not None:
            self.pending_auto_refresh = True
            return
        self.refresh_debounce.start()

    def run_auto_refresh(self):
        if self.thread is not None:
            self.pending_auto_refresh = True
            return
        self.pending_auto_refresh = False
        self.refresh_images(silent=True)

    def clear_preview(self):
        self.preview_side_label.setText("Front (click image to toggle)")
        self.preview_label.setText("Select a card pair")
        self.preview_label.setPixmap(QPixmap())

    def set_preview_pixmap(self, image_path: Path, fallback_text: str):
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.preview_label.setText(fallback_text)
            self.preview_label.setPixmap(QPixmap())
            return
        scaled = pixmap.scaled(
            self.preview_label.width(),
            PREVIEW_HEIGHT,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setText("")
        self.preview_label.setPixmap(scaled)

    def refresh_images(
        self,
        silent: bool = False,
        preferred_pair: Optional[Tuple[Path, Path]] = None,
    ):
        if preferred_pair is None:
            current_item = self.image_list.currentItem()
            if current_item is not None:
                preferred_pair = (
                    Path(current_item.data(Qt.UserRole)),
                    Path(current_item.data(Qt.UserRole + 1)),
                )

        self.image_list.clear()
        self.image_pairs = []
        self.current_front_path = None
        self.current_back_path = None
        self.current_showing_back = False
        self.clear_preview()
        self.name_edit.clear()

        try:
            directory = Path(self.directory_edit.text()).expanduser().resolve()
        except Exception as exc:
            self.append_log(f"Invalid directory: {exc}")
            self.set_watched_directory(None)
            self.update_action_buttons()
            return

        if not directory.exists() or not directory.is_dir():
            self.append_log(f"Directory does not exist: {directory}")
            self.set_watched_directory(None)
            self.update_action_buttons()
            return

        self.set_watched_directory(directory)
        image_files = _list_gui_images(directory)
        self.image_pairs = _build_gui_pairs(image_files)

        for idx, pair in enumerate(self.image_pairs):
            front, back = pair
            item = QListWidgetItem(f"{idx + 1:03d}. {front.name} | {back.name}")
            item.setData(Qt.UserRole, str(front))
            item.setData(Qt.UserRole + 1, str(back))
            self.image_list.addItem(item)

        selected_row: Optional[int] = None
        if preferred_pair is not None:
            preferred_front, preferred_back = preferred_pair
            for row in range(self.image_list.count()):
                item = self.image_list.item(row)
                item_front = Path(item.data(Qt.UserRole))
                item_back = Path(item.data(Qt.UserRole + 1))
                if (
                    item_front == preferred_front and item_back == preferred_back
                ) or (
                    item_front == preferred_back and item_back == preferred_front
                ):
                    selected_row = row
                    break

        if selected_row is not None:
            self.image_list.setCurrentRow(selected_row)
        elif self.image_pairs:
            self.image_list.setCurrentRow(0)

        if not silent:
            if not image_files:
                self.append_log("No image files found.")
            elif len(image_files) % 2 != 0:
                self.append_log(
                    f"Found {len(image_files)} images. The last unmatched image is ignored."
                )
            elif not self.image_pairs:
                self.append_log("No complete front/back pairs found.")
            else:
                self.append_log(
                    f"Loaded {len(self.image_pairs)} card pairs from {directory}. "
                    "Most recent pair selected."
                )

        self.update_action_buttons()

    def on_image_selected(self, current, _previous):
        if current is None:
            self.current_front_path = None
            self.current_back_path = None
            self.current_showing_back = False
            self.clear_preview()
            self.update_action_buttons()
            return
        self.current_front_path = Path(current.data(Qt.UserRole))
        self.current_back_path = Path(current.data(Qt.UserRole + 1))
        self.current_showing_back = False
        self.name_edit.setText(_pair_base_name(self.current_front_path, self.current_back_path))
        self.update_preview_pixmap()
        self.update_action_buttons()

    def update_preview_pixmap(self):
        if self.current_front_path is None or self.current_back_path is None:
            return
        if self.current_showing_back:
            self.preview_side_label.setText("Back (click image to toggle)")
            self.set_preview_pixmap(
                self.current_back_path,
                f"Cannot preview: {self.current_back_path.name}",
            )
        else:
            self.preview_side_label.setText("Front (click image to toggle)")
            self.set_preview_pixmap(
                self.current_front_path,
                f"Cannot preview: {self.current_front_path.name}",
            )

    def toggle_preview_side(self):
        if self.current_front_path is None or self.current_back_path is None:
            return
        self.current_showing_back = not self.current_showing_back
        self.update_preview_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview_pixmap()

    def set_controls_enabled(self, enabled: bool):
        self.directory_edit.setEnabled(enabled)
        self.incoming_button.setEnabled(enabled)
        self.existing_button.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.refresh_button.setEnabled(enabled)
        self.image_list.setEnabled(enabled)
        self.name_edit.setEnabled(enabled)
        if enabled:
            self.update_action_buttons()
        else:
            self.start_button.setEnabled(False)
            self.accept_button.setEnabled(False)
            self.tcdb_button.setEnabled(False)
            self.ebay_button.setEnabled(False)
            self.move_selected_button.setEnabled(False)
            self.move_all_button.setEnabled(False)

    def is_existing_cards_directory_selected(self) -> bool:
        try:
            current = Path(self.directory_edit.text()).expanduser().resolve()
            target = self.existing_cards_directory.expanduser().resolve()
            return current == target
        except Exception:
            return False

    def update_action_buttons(self):
        has_selection = self.image_list.currentItem() is not None
        busy = self.thread is not None
        has_name = bool(self.name_edit.text().strip())
        can_move = (not busy) and (not self.is_existing_cards_directory_selected())
        self.start_button.setEnabled(has_selection and not busy)
        self.accept_button.setEnabled(has_selection and has_name and not busy)
        self.tcdb_button.setEnabled(has_selection and has_name and not busy)
        self.ebay_button.setEnabled(has_selection and has_name and not busy)
        self.move_selected_button.setEnabled(has_selection and can_move)
        self.move_all_button.setEnabled(bool(self.image_pairs) and can_move)

    def open_tcdb_search(self):
        current = self.image_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "No Pair Selected", "Select a card pair first.")
            return

        accepted_name = self.name_edit.text().strip()
        if not accepted_name:
            QMessageBox.warning(
                self,
                "Missing Name",
                "Enter or accept a proposed name before opening TCDB search.",
            )
            return

        sanitized = _sanitize_base_name(accepted_name)
        if not sanitized:
            QMessageBox.warning(self, "Invalid Name", "Enter a valid card name.")
            return
        if sanitized != accepted_name:
            self.name_edit.setText(sanitized)

        url = _tcdb_search_url_from_filename(f"{sanitized}.jpg", back_suffix="_b")
        opened = webbrowser.open(url, new=2)
        if not opened:
            QMessageBox.warning(self, "Browser Error", "Could not open a web browser for TCDB.")
            return

        _copy_to_clipboard(url)
        self.append_log(f"Opened TCDB search: {url}")

    def open_ebay_search(self):
        current = self.image_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "No Pair Selected", "Select a card pair first.")
            return

        accepted_name = self.name_edit.text().strip()
        if not accepted_name:
            QMessageBox.warning(
                self,
                "Missing Name",
                "Enter or accept a proposed name before opening eBay search.",
            )
            return

        search_terms = _name_to_search_terms(accepted_name)
        if not search_terms:
            QMessageBox.warning(self, "Invalid Name", "Enter a valid card name.")
            return

        url = f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(search_terms)}"
        opened = webbrowser.open(url, new=2)
        if not opened:
            QMessageBox.warning(self, "Browser Error", "Could not open a web browser for eBay.")
            return

        _copy_to_clipboard(url)
        self.append_log(f"Opened eBay search: {url}")

    def move_selected_card(self):
        current = self.image_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "No Pair Selected", "Select a card pair first.")
            return
        front = Path(current.data(Qt.UserRole))
        back = Path(current.data(Qt.UserRole + 1))
        self._move_pairs_to_existing([(front, back)])

    def move_all_cards(self):
        if not self.image_pairs:
            QMessageBox.warning(self, "No Cards", "No card pairs are loaded.")
            return
        self._move_pairs_to_existing(list(self.image_pairs))

    def _move_pairs_to_existing(self, pairs: List[Tuple[Path, Path]]):
        target_dir = self.existing_cards_directory
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Move Failed", f"Cannot create target directory:\n{exc}")
            return

        moved = 0
        skipped = 0
        failures: List[str] = []

        for front, back in pairs:
            for source in (front, back):
                if not source.exists():
                    skipped += 1
                    continue
                destination = target_dir / source.name
                try:
                    if source.resolve() == destination.resolve():
                        skipped += 1
                        continue
                    _move_file_overwrite(source, destination)
                    moved += 1
                except Exception as exc:
                    failures.append(f"{source.name}: {exc}")

        if failures:
            preview = "\n".join(failures[:8])
            if len(failures) > 8:
                preview += f"\n...and {len(failures) - 8} more"
            QMessageBox.critical(self, "Move Failed", preview)

        self.append_log(
            f"Move complete to {target_dir}: moved {moved} file(s), skipped {skipped} file(s)."
        )
        self.refresh_images()

    def start_naming(self):
        if self.thread is not None:
            return
        current = self.image_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "No Pair Selected", "Select a card pair first.")
            return

        front = Path(current.data(Qt.UserRole))
        back = Path(current.data(Qt.UserRole + 1))
        if not front.exists() or not back.exists():
            QMessageBox.warning(self, "Missing Files", "Selected image pair is missing on disk.")
            self.refresh_images()
            return

        self.append_log(f"Naming {front.name} + {back.name} ...")
        self.name_edit.clear()
        self.set_controls_enabled(False)

        self.thread = QThread(self)
        self.worker = NameWorker(front, back)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_naming_finished)
        self.worker.failed.connect(self.on_naming_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_worker)
        self.thread.start()

    def on_naming_finished(self, proposed_name: str):
        self.name_edit.setText(proposed_name)
        self.name_edit.setFocus()
        self.name_edit.selectAll()
        self.append_log(f"Suggested name: {proposed_name}")

    def on_naming_failed(self, message: str):
        first_line = message.splitlines()[0] if message else "Unknown error"
        self.append_log(f"Naming failed: {first_line}")
        QMessageBox.critical(self, "Naming Failed", first_line)

    def cleanup_worker(self):
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None
        self.set_controls_enabled(True)
        if self.pending_auto_refresh:
            self.run_auto_refresh()

    def accept_name(self):
        current = self.image_list.currentItem()
        if current is None:
            QMessageBox.warning(self, "No Pair Selected", "Select a card pair first.")
            return

        base_name_input = self.name_edit.text().strip()
        if not base_name_input:
            QMessageBox.warning(self, "Missing Name", "Generate or enter a name first.")
            return

        sanitized = _sanitize_base_name(base_name_input)
        if not sanitized:
            QMessageBox.warning(self, "Invalid Name", "Enter a valid card name.")
            return
        if sanitized != base_name_input:
            self.name_edit.setText(sanitized)

        front = Path(current.data(Qt.UserRole))
        back = Path(current.data(Qt.UserRole + 1))
        directory = front.parent

        front_ext = front.suffix.lower() or ".jpg"
        back_ext = back.suffix.lower() or front_ext

        front_target = directory / f"{sanitized}{front_ext}"
        back_target = directory / f"{sanitized}_b{back_ext}"

        if front_target.exists() and front_target.resolve() != front.resolve():
            QMessageBox.warning(
                self,
                "Name Conflict",
                f"Target already exists: {front_target.name}",
            )
            return
        if back_target.exists() and back_target.resolve() != back.resolve():
            QMessageBox.warning(
                self,
                "Name Conflict",
                f"Target already exists: {back_target.name}",
            )
            return

        renamed_front = False
        try:
            if front.resolve() != front_target.resolve():
                front.rename(front_target)
                renamed_front = True
            if back.resolve() != back_target.resolve():
                back.rename(back_target)
        except Exception as exc:
            if renamed_front:
                try:
                    front_target.rename(front)
                except Exception:
                    pass
            QMessageBox.critical(self, "Rename Failed", str(exc))
            return

        self.append_log(f"Renamed to {front_target.name} and {back_target.name}")
        self.name_edit.clear()
        self.refresh_images(preferred_pair=(front_target, back_target))


def run_gui():
    qt_app = QApplication(sys.argv)
    window = CardNamerGui()
    window.show()
    sys.exit(qt_app.exec_())


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
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        del sys.argv[1]
        app()
    else:
        run_gui()
