#!/usr/bin/env python3
import base64
import datetime
import json
import os
import random
import subprocess
import time
from pathlib import Path

import anthropic
import typer

PROMPT_QUESTIONS = [
    "Name one notable person born on this calendar date in a past year. Use the month and day only, not the current year. Include a 240 chars or less bio. Plain text only, no markdown.",
    "Give one notable historical event that happened on this calendar date in a past year. Use the month and day only, not the current year. Keep it under 240 chars. Plain text only, no markdown.",
    "What is one interesting science fact for today? Keep it under 240 chars. Plain text only, no markdown.",
    "Send me a haiku for today. 240 chars or less. Plain text only, no markdown.",
]

MODEL               = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
SHORTCUT_NAME       = "Send Message"
CARD_SHORTCUT_NAME  = "Send Card Message"
CARD_DIR            = Path("/Volumes/Dutton 2TB/Cards/Mix")
DEFAULT_KEY_FILE    = ".anthropic-api-key.txt"
FACTS_HISTORY_FILE  = Path.home() / ".card_facts_history.json"
CARD_EXCLUDE_FILE   = Path.home() / ".card_facts_exclude.json"
MAX_FACTS_PER_PLAYER = 20

FACT_CATEGORIES = [
    "a surprising fact from their early life or childhood",
    "a lesser-known fact about their rookie season or first year in the league",
    "a specific season's stats — pick one year and team they played for and share one notable stat (e.g., points per game, ERA, batting average, yards per game)",
    "an interesting record or achievement they hold that isn't widely known",
    "something about their personal life, family, or interests outside their sport",
    "a quirky or unusual story from their career",
    "their background before going pro — college, hometown, or international career",
    "a memorable game, play, or moment from their career that's often overlooked",
    "an injury, setback, or adversity they overcame",
    "an interesting fact about a trade, signing, or team change in their career",
]

# cron often runs with a minimal PATH.
os.environ["PATH"] = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def get_api_key() -> str:
    env_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if env_key:
        return env_key

    candidate_paths = [
        Path(__file__).resolve().parent / DEFAULT_KEY_FILE,
        Path.home() / DEFAULT_KEY_FILE,
    ]
    for path in candidate_paths:
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key

    raise RuntimeError(
        "ANTHROPIC_API_KEY not set and no key file found. "
        f"Tried: {', '.join(str(p) for p in candidate_paths)}"
    )


def _load_exclude_set() -> set[str]:
    if CARD_EXCLUDE_FILE.exists():
        try:
            return set(json.loads(CARD_EXCLUDE_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            return set()
    return set()


def _mark_card_excluded(image_path: Path) -> None:
    excluded = _load_exclude_set()
    excluded.add(image_path.name)
    CARD_EXCLUDE_FILE.write_text(json.dumps(sorted(excluded), indent=2), encoding="utf-8")


def pick_random_card() -> Path | None:
    """Return a random front-facing JPG from CARD_DIR, or None if drive unavailable."""
    if not CARD_DIR.exists():
        return None
    excluded = _load_exclude_set()
    fronts = [
        p for p in CARD_DIR.glob("*.jpg")
        if not p.stem.endswith("_b") and p.name not in excluded
    ]
    return random.choice(fronts) if fronts else None


def _encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def _identify_player(image_data: str, client: anthropic.Anthropic) -> str:
    """Return just the player's name from a card image."""
    response = claude_create(
        client,
        model=MODEL,
        max_tokens=50,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                {"type": "text", "text": "What is the name of the player on this sports trading card? Reply with just their name, nothing else."},
            ],
        }],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return " ".join(text.split())


def _get_player_fact(player_name: str, previous_facts: list[str], client: anthropic.Anthropic) -> str:
    """Get a categorized fact about the player, avoiding previously sent facts."""
    category = random.choice(FACT_CATEGORIES)
    avoid = (
        "\n\nDo not repeat any of these facts already sent:\n"
        + "\n".join(f"- {f}" for f in previous_facts)
    ) if previous_facts else ""
    prompt = (
        f"Give me {category} about {player_name}. "
        "Plain text only, no markdown or formatting. "
        f"Keep it under 180 characters.{avoid}"
    )
    response = claude_create(
        client,
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return " ".join(text.split())


def _load_facts_history() -> dict:
    if FACTS_HISTORY_FILE.exists():
        try:
            return json.loads(FACTS_HISTORY_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_facts_history(history: dict) -> None:
    FACTS_HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


def get_card_caption(image_path: Path, client: anthropic.Anthropic) -> str:
    """Two-step: identify player, then fetch a non-repeated categorized fact."""
    image_data = _encode_image(image_path)
    player_name = _identify_player(image_data, client)

    history = _load_facts_history()
    player_key = player_name.lower()
    previous_facts = history.get(player_key, [])

    fact = _get_player_fact(player_name, previous_facts, client)

    history[player_key] = (previous_facts + [fact])[-MAX_FACTS_PER_PLAYER:]
    _save_facts_history(history)

    return f"{player_name}: {fact}"


def claude_create(client: anthropic.Anthropic, **kwargs) -> anthropic.types.Message:
    """Call client.messages.create with up to 3 retries on 529 overloaded errors."""
    for attempt in range(3):
        try:
            return client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            if e.status_code == 529 and attempt < 2:
                time.sleep(30 * (attempt + 1))
                continue
            raise


def build_text_message(client: anthropic.Anthropic) -> str:
    today = datetime.date.today().strftime("%B %d, %Y")
    month_day = datetime.date.today().strftime("%B %d")
    question = random.choice(PROMPT_QUESTIONS)

    if "calendar date" in question:
        prompt = (
            f"Today is {today}. The calendar date is {month_day}. "
            f"{question}"
        )
    else:
        prompt = f"Today is {today}. {question}"

    response = claude_create(
        client,
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text:
        raise RuntimeError("Claude returned no text")
    return " ".join(text.split())


def send_via_shortcut(name: str, message: str) -> None:
    subprocess.run(["shortcuts", "run", name], input=message, text=True, check=True)


def import_to_photos(image_path: Path) -> None:
    """Import image into Photos under a 'cards' album, creating it if needed."""
    script = f'''
    tell application "Photos"
        if not (exists album "cards") then
            make new album named "cards"
        end if
        import POSIX file "{image_path}" into album "cards" skip check duplicates yes
    end tell
    '''
    subprocess.run(["osascript", "-e", script], check=True)
    time.sleep(3)  # give Photos time to finish importing


def send_image_via_imessage(image_path: Path, caption: str) -> bool:
    """Returns True on success, False if Photos timed out."""
    try:
        import_to_photos(image_path)
    except subprocess.CalledProcessError:
        return False
    subprocess.run(["shortcuts", "run", "Text Latest Image"], check=True)
    send_via_shortcut(CARD_SHORTCUT_NAME, caption)
    return True


def main(
    card: bool = typer.Option(False, "--card", help="Force sending a card instead of a daily text"),
    test: bool = typer.Option(False, "--test", help="Print card and fact to stdout without sending anything"),
) -> None:
    client = anthropic.Anthropic(api_key=get_api_key())

    if test:
        card_path = pick_random_card()
        if not card_path:
            typer.echo("No card available (drive not mounted?)")
            raise typer.Exit(1)
        caption = get_card_caption(card_path, client)
        typer.echo(f"{card_path.name}\n{caption}")
        return

    if card:
        card_path = pick_random_card()
        if card_path:
            caption = get_card_caption(card_path, client)
            if send_image_via_imessage(card_path, caption):
                _mark_card_excluded(card_path)
                return
            # Photos timed out — fall through to text message
        # Drive not mounted or send failed — fall through to text message

    message = build_text_message(client)
    send_via_shortcut(SHORTCUT_NAME, message)


if __name__ == "__main__":
    typer.run(main)
