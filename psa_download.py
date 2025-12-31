#!/usr/bin/env python3

import json
import re
import shutil
import subprocess
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
import typer
from PIL import Image, ImageOps

app = typer.Typer(add_completion=False)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
)

API_DEFAULT_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "User-Agent": DEFAULT_USER_AGENT,
    "Origin": "https://www.psacard.com",
    "Referer": "https://www.psacard.com/",
}

def _first_match(patterns: List[str], text: str) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def _walk_json(obj, key_names: List[str]) -> List[str]:
    matches: List[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key.lower() in key_names and isinstance(value, (str, int)):
                matches.append(str(value))
            matches.extend(_walk_json(value, key_names))
    elif isinstance(obj, list):
        for item in obj:
            matches.extend(_walk_json(item, key_names))
    return matches


def _normalize_whitespace(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    return re.sub(r"\s+", " ", value).strip()


def _smart_title(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    words = []
    for word in re.split(r"(\s+)", value.strip()):
        if not word or word.isspace():
            words.append(word)
            continue
        if word.isupper():
            if len(word) <= 3:
                words.append(word)
            else:
                words.append(word.title())
        else:
            words.append(word)
    return "".join(words)


def _compact_slug(value: str) -> str:
    compact = _slugify(value).replace("-", "")
    return compact or "Unknown"


def _collect_strings(obj) -> List[str]:
    values: List[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            values.extend(_collect_strings(value))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(_collect_strings(item))
    elif isinstance(obj, str):
        values.append(obj)
    return values


def _load_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("access_token", "token"):
                token = payload.get(key)
                if isinstance(token, str) and token.strip():
                    raw = token.strip()
                    break
    raw = re.sub(r"(?i)^authorization:\s*", "", raw).strip()
    raw = re.sub(r"(?i)^bearer\s+", "", raw).strip()
    return raw or None


def _extract_year_from_json(data: dict) -> Optional[str]:
    candidates = _walk_json(data, ["year", "cardyear", "certyear"])
    for candidate in candidates:
        match = re.search(r"(19|20)\d{2}", str(candidate))
        if match:
            return match.group(0)
    return _first_match([r"\b(19|20)\d{2}\b"], " ".join(_collect_strings(data)))


def _extract_card_details(data: dict) -> Dict[str, Optional[str]]:
    details: Dict[str, Optional[str]] = {
        "year": None,
        "player": None,
        "manufacturer": None,
        "series": None,
        "variety": None,
    }

    details["year"] = _extract_year_from_json(data)
    details["manufacturer"] = _normalize_whitespace(
        next(
            iter(_walk_json(data, ["brand", "manufacturer", "brandname"])),
            None,
        )
    )
    details["series"] = _normalize_whitespace(
        next(
            iter(_walk_json(data, ["setname", "series", "cardset", "set"])),
            None,
        )
    )
    details["player"] = _normalize_whitespace(
        next(
            iter(_walk_json(data, ["player", "subject", "playername"])),
            None,
        )
    )
    details["variety"] = _normalize_whitespace(
        next(iter(_walk_json(data, ["variety", "parallel"])), None)
    )

    return details


def _slugify(value: str) -> str:
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[\s_-]+", "-", value).strip("-")
    return value or "Unknown"


def _extract_lastname(player: Optional[str]) -> Optional[str]:
    if not player:
        return None
    tokens = [t for t in re.split(r"\s+", player.strip()) if t]
    if not tokens:
        return None
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    if tokens[-1].rstrip(".").lower() in suffixes and len(tokens) > 1:
        tokens = tokens[:-1]
    return tokens[-1]


KNOWN_MANUFACTURER_PREFIXES = [
    "Upper Deck",
    "Panini",
    "Topps",
    "Donruss",
    "Fleer",
    "Leaf",
    "Bowman",
    "Score",
]

KNOWN_VARIETY_SUFFIXES = [
    "Campus Legends",
]


def _split_manufacturer_series(manufacturer: Optional[str], series: Optional[str]) -> Tuple[str, str]:
    manufacturer = manufacturer or ""
    series = series or ""
    if series:
        series = re.sub(r"^(?:19|20)\d{2}\s+", "", series).strip()
    if manufacturer and not series and " " in manufacturer:
        for prefix in KNOWN_MANUFACTURER_PREFIXES:
            if manufacturer.lower().startswith(prefix.lower() + " "):
                remainder = manufacturer[len(prefix) :].strip()
                if remainder:
                    manufacturer = prefix
                    series = remainder
                break
    if not manufacturer and series:
        parts = series.split()
        if parts:
            manufacturer = parts[0]
            series = " ".join(parts[1:]) or series
    if manufacturer and series:
        series_lower = series.lower()
        manu_lower = manufacturer.lower()
        if series_lower.startswith(manu_lower):
            series = series[len(manufacturer) :].strip(" -")
    return manufacturer or "Unknown", series or "Unknown"


def _split_series_variety(series: str, variety: Optional[str]) -> Tuple[str, Optional[str]]:
    if not series:
        return series, variety
    if variety:
        return series, variety
    series_title = _smart_title(series) or series
    for suffix in KNOWN_VARIETY_SUFFIXES:
        if series_title.lower().endswith(" " + suffix.lower()):
            trimmed = series_title[: -len(suffix)].rstrip(" -")
            if trimmed:
                return trimmed, suffix
    return series, variety


def _normalize_url(value: str, base_url: str) -> str:
    if value.startswith("//"):
        value = f"https:{value}"
    return urljoin(base_url, value)


def _looks_like_image(value: str, key_hint: Optional[str] = None) -> bool:
    trimmed = value.strip()
    if re.search(r"\.(jpg|jpeg|png|webp|gif|tif|tiff)(\?|#|$)", trimmed, re.IGNORECASE):
        return True
    if not trimmed.startswith(("http://", "https://", "//", "/")):
        return False
    lower = trimmed.lower()
    if "/image" in lower or "/img" in lower:
        return True
    return bool(key_hint and "image" in key_hint.lower())


def _extract_image_urls_from_json(data) -> Tuple[List[str], Optional[str], Optional[str]]:
    base_url = "https://www.psacard.com/"
    urls: List[str] = []
    front: Optional[str] = None
    back: Optional[str] = None

    def add_url(raw: str, key_hint: Optional[str] = None, type_hint: Optional[str] = None) -> None:
        nonlocal front, back
        url = _normalize_url(raw.strip(), base_url)
        urls.append(url)
        hint = " ".join(filter(None, [key_hint, type_hint])).lower()
        if not front and re.search(r"(front|obverse|recto)", hint):
            front = url
        if not back and re.search(r"(back|reverse|verso)", hint):
            back = url
        if not front and key_hint and "front" in key_hint.lower():
            front = url
        if not back and key_hint and "back" in key_hint.lower():
            back = url

    def walk(obj) -> None:
        if isinstance(obj, dict):
            type_hint = None
            image_url_value: Optional[str] = None
            is_front_value: Optional[bool] = None
            for key, value in obj.items():
                if not isinstance(value, str):
                    if key.lower() in {"isfrontimage", "isfront"} and isinstance(value, bool):
                        is_front_value = value
                    continue
                key_lower = key.lower()
                if key_lower in {"imageurl", "url", "image", "imageuri"}:
                    image_url_value = value
                if key_lower in {"imagetype", "view", "side", "position", "phototype"}:
                    type_hint = value
            if image_url_value:
                if is_front_value is True:
                    add_url(image_url_value, key_hint="front", type_hint=type_hint)
                elif is_front_value is False:
                    add_url(image_url_value, key_hint="back", type_hint=type_hint)
                else:
                    add_url(image_url_value, key_hint=None, type_hint=type_hint)
            for key, value in obj.items():
                if not isinstance(value, str):
                    continue
                if _looks_like_image(value, key_hint=key):
                    add_url(value, key_hint=key, type_hint=type_hint)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
        elif isinstance(obj, str):
            if _looks_like_image(obj):
                add_url(obj)

    walk(data)

    seen = set()
    unique_urls: List[str] = []
    for url in urls:
        if url not in seen:
            unique_urls.append(url)
            seen.add(url)
    return unique_urls, front, back


def _has_face_tag(url: str, tags: List[str]) -> bool:
    for tag in tags:
        pattern = rf"(?<![a-z0-9]){re.escape(tag)}(?![a-z0-9])"
        if re.search(pattern, url, flags=re.IGNORECASE):
            return True
    return False


def _pick_front_back(image_urls: List[str]) -> Tuple[str, str]:
    if not image_urls:
        raise ValueError("No image URLs found in the API response.")

    front_candidates: List[str] = []
    back_candidates: List[str] = []
    for url in image_urls:
        if _has_face_tag(url, ["front", "obverse", "recto"]):
            front_candidates.append(url)
        if _has_face_tag(url, ["back", "reverse", "verso"]):
            back_candidates.append(url)

    front = front_candidates[0] if front_candidates else None
    back = back_candidates[0] if back_candidates else None

    if front and not back:
        back_guess = re.sub(r"front", "back", front, flags=re.IGNORECASE)
        if back_guess != front:
            back = back_guess
    if back and not front:
        front_guess = re.sub(r"back", "front", back, flags=re.IGNORECASE)
        if front_guess != back:
            front = front_guess

    if front and back:
        return front, back

    if len(image_urls) >= 2:
        return image_urls[0], image_urls[1]

    raise ValueError("Unable to determine front/back image URLs.")


def _fetch_api_json(
    session: requests.Session, url: str, verbose: bool = False
) -> Tuple[dict, str]:
    try:
        resp = session.get(url, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"Unable to fetch PSA API data: {exc}") from exc
    if verbose:
        typer.echo(f"GET {url} -> {resp.status_code}")
    if not resp.ok:
        preview = resp.text.strip()
        if verbose and preview:
            typer.echo(preview[:500])
        if resp.status_code == 403 and "Security Check" in preview:
            raise RuntimeError(
                "PSA API blocked the request with a Security Check (403). "
                "This usually means the edge WAF is blocking non-browser traffic."
            )
        if resp.status_code == 429:
            raise RuntimeError("PSA API rate limit exceeded (429).")
        raise RuntimeError(f"Unable to fetch PSA API data. Status {resp.status_code} from {url}.")
    try:
        return resp.json(), resp.url
    except ValueError as exc:
        raise RuntimeError("PSA API response was not valid JSON.") from exc


def _fetch_cert_data(
    session: requests.Session, cert: str, verbose: bool = False
) -> Tuple[dict, str]:
    url = f"https://api.psacard.com/publicapi/cert/GetByCertNumber/{cert}"
    return _fetch_api_json(session, url, verbose=verbose)


def _fetch_images_data(
    session: requests.Session, cert: str, verbose: bool = False
) -> Tuple[dict, str]:
    url = f"https://api.psacard.com/publicapi/cert/GetImagesByCertNumber/{cert}"
    return _fetch_api_json(session, url, verbose=verbose)


def _fetch_api_json_via_curl(
    url: str, token: str, verbose: bool = False
) -> Tuple[dict, str]:
    curl_path = shutil.which("curl")
    if not curl_path:
        raise RuntimeError("curl is not available on this system.")
    cmd = [
        curl_path,
        "-sS",
        "-L",
        "-w",
        "\\n__HTTP_STATUS__%{http_code}",
        "-H",
        f"Authorization: bearer {token}",
        "-H",
        "Accept: application/json",
        url,
    ]
    if verbose:
        typer.echo(f"curl {url}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "curl failed"
        raise RuntimeError(f"curl error: {stderr}")
    stdout = result.stdout
    if "__HTTP_STATUS__" not in stdout:
        raise RuntimeError("curl did not return a status code.")
    payload, status_line = stdout.rsplit("__HTTP_STATUS__", 1)
    payload = payload.strip()
    status_line = status_line.strip()
    try:
        status_code = int(status_line)
    except ValueError as exc:
        raise RuntimeError("curl returned an invalid status code.") from exc
    if status_code == 204:
        raise RuntimeError("PSA API returned 204 (empty response).")
    if status_code == 429:
        raise RuntimeError("PSA API rate limit exceeded (429).")
    if not (200 <= status_code < 300):
        if payload:
            preview = payload.strip()
            if "<title>security check</title>" in preview.lower():
                raise RuntimeError("curl received a Security Check page instead of JSON.")
        raise RuntimeError(f"PSA API returned status {status_code}.")
    if not payload:
        raise RuntimeError("curl returned an empty response.")
    if payload.lstrip().lower().startswith("<!doctype html") or "<title>security check</title>" in payload.lower():
        raise RuntimeError("curl received a Security Check page instead of JSON.")
    try:
        return json.loads(payload), url
    except json.JSONDecodeError as exc:
        raise RuntimeError("curl response was not valid JSON.") from exc


def _fetch_cert_data_via_curl(cert: str, token: str, verbose: bool = False) -> Tuple[dict, str]:
    url = f"https://api.psacard.com/publicapi/cert/GetByCertNumber/{cert}"
    return _fetch_api_json_via_curl(url, token, verbose=verbose)


def _fetch_images_data_via_curl(cert: str, token: str, verbose: bool = False) -> Tuple[dict, str]:
    url = f"https://api.psacard.com/publicapi/cert/GetImagesByCertNumber/{cert}"
    return _fetch_api_json_via_curl(url, token, verbose=verbose)


def _download_image(session: requests.Session, url: str) -> bytes:
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def _save_processed_jpg(image_bytes: bytes, path: Path, scale: float, quality: int) -> None:
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if scale <= 0:
        raise ValueError("Scale must be greater than 0.")
    if scale != 1.0:
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.LANCZOS)
    img = img.convert("RGB")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG", quality=quality, optimize=True)


@app.command()
def fetch(
    cert: str = typer.Argument(..., help="PSA certification number"),
    out_dir: Path = typer.Option(Path("."), help="Output directory"),
    scale: float = typer.Option(0.75, help="Scale factor (0.75 = 75%)"),
    quality: int = typer.Option(65, help="JPEG quality (1-95)"),
    token: Optional[str] = typer.Option(None, help="PSA API access token"),
    token_file: Optional[Path] = typer.Option(
        None,
        help=(
            "File containing the PSA API access token. "
            "Defaults: ./psa-token.txt, ~/.config/psa/psa-token.txt, ~/.psa-token.txt"
        ),
    ),
    use_curl: bool = typer.Option(
        True, "--use-curl/--no-use-curl", help="Fetch PSA API JSON using curl"
    ),
    year: Optional[str] = typer.Option(None, help="Override year (YYYY)"),
    lastname: Optional[str] = typer.Option(None, help="Override last name"),
    manufacturer: Optional[str] = typer.Option(None, help="Override manufacturer"),
    series: Optional[str] = typer.Option(None, help="Override series"),
    variety: Optional[str] = typer.Option(None, help="Override variety/parallel"),
    save_json: bool = typer.Option(False, help="Save fetched JSON responses for debugging"),
    verbose: bool = typer.Option(False, help="Print debug details"),
):
    session = requests.Session()
    session.headers.update(API_DEFAULT_HEADERS)

    if not (1 <= quality <= 95):
        raise typer.BadParameter("quality must be between 1 and 95")

    if token is None:
        token_paths: List[Path] = []
        if token_file:
            token_paths.append(token_file)
        script_dir = Path(__file__).resolve().parent
        token_paths.extend(
            [
                script_dir / ".psa-token.txt",
                script_dir / "psa-token.txt",
                Path("psa-token.txt"),
                Path.home() / ".config" / "psa" / "psa-token.txt",
                Path.home() / ".psa-token.txt",
            ]
        )
        for path in token_paths:
            if path.exists():
                token = path.read_text(encoding="utf-8").strip()
                break
    token = _load_token(token)
    if not token:
        raise typer.BadParameter("PSA API token is required (use --token or --token-file).")
    session.headers.update({"Authorization": f"Bearer {token}"})

    def fetch_with_fallback(fetcher, curl_fetcher, label: str) -> Tuple[dict, str]:
        if use_curl:
            return curl_fetcher(cert, token, verbose=verbose)
        try:
            return fetcher(session, cert, verbose=verbose)
        except RuntimeError as exc:
            message = str(exc)
            if "Security Check" in message and shutil.which("curl"):
                typer.echo(f"Falling back to curl for {label} due to Security Check.")
                return curl_fetcher(cert, token, verbose=verbose)
            raise

    data, _final_url = fetch_with_fallback(
        _fetch_cert_data, _fetch_cert_data_via_curl, "cert data"
    )
    images_data, _images_url = fetch_with_fallback(
        _fetch_images_data, _fetch_images_data_via_curl, "image data"
    )

    if save_json:
        cert_json_path = out_dir / f"{cert}_psa.json"
        cert_json_path.write_text(
            json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
        )
        typer.echo(f"Saved JSON to {cert_json_path}")
        images_json_path = out_dir / f"{cert}_psa_images.json"
        images_json_path.write_text(
            json.dumps(images_data, indent=2, sort_keys=True), encoding="utf-8"
        )
        typer.echo(f"Saved JSON to {images_json_path}")

    image_urls, front_url, back_url = _extract_image_urls_from_json(images_data)
    try:
        if not (front_url and back_url):
            front_url, back_url = _pick_front_back(image_urls)
    except ValueError as exc:
        raise RuntimeError(
            "No image URLs found in GetImagesByCertNumber response."
        ) from exc

    if verbose:
        typer.echo(f"Front URL: {front_url}")
        typer.echo(f"Back URL:  {back_url}")

    details = _extract_card_details(data)

    final_year = year or details.get("year") or "Unknown"
    player_name = details.get("player") or ""
    final_lastname = lastname or _extract_lastname(player_name) or "Unknown"
    final_manufacturer, final_series = _split_manufacturer_series(
        manufacturer or details.get("manufacturer"),
        series or details.get("series"),
    )
    final_variety = variety or details.get("variety")

    year_match = re.search(r"(19|20)\d{2}", final_year)
    final_year = year_match.group(0) if year_match else final_year

    final_lastname = _smart_title(final_lastname) or final_lastname
    final_manufacturer = _smart_title(final_manufacturer) or final_manufacturer
    final_series = _smart_title(final_series) or final_series
    final_variety = _smart_title(final_variety) if final_variety else None

    final_series, final_variety = _split_series_variety(final_series, final_variety)

    parts = [
        _slugify(final_year),
        _slugify(final_lastname),
    ]
    manufacturer_slug = _slugify(final_manufacturer)
    series_slug = _slugify(final_series)
    if manufacturer_slug != "Unknown":
        parts.append(manufacturer_slug)
    if series_slug != "Unknown":
        parts.append(series_slug)
    if final_variety:
        variety_slug = _compact_slug(final_variety)
        if variety_slug != "Unknown":
            parts.append(variety_slug)
    base_name = "-".join(parts)

    front_path = out_dir / f"{base_name}.jpg"
    back_path = out_dir / f"{base_name}_b.jpg"

    front_bytes = _download_image(session, front_url)
    back_bytes = _download_image(session, back_url)

    _save_processed_jpg(front_bytes, front_path, scale, quality)
    _save_processed_jpg(back_bytes, back_path, scale, quality)

    typer.echo(f"Saved {front_path}")
    typer.echo(f"Saved {back_path}")


if __name__ == "__main__":
    app()
