#!/usr/bin/env python3
import base64
import datetime
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import anthropic

PROMPT_QUESTIONS = [
    "Who is famous that was born on this day? 240 chars or less bio.",
    "Give one notable historical event that happened on this day in 240 chars or less.",
    "What is one interesting science fact for today? Keep it under 240 chars.",
    "Send me a hiku for today. 240 chars or less.",
]

MODEL          = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
SHORTCUT_NAME  = "Send Message"
CARD_DIR       = Path("/Volumes/Dutton 2TB/Cards/Mix")
DEFAULT_KEY_FILE = ".anthropic-api-key.txt"

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


def pick_random_card() -> Path | None:
    """Return a random front-facing JPG from CARD_DIR, or None if drive unavailable."""
    if not CARD_DIR.exists():
        return None
    fronts = [p for p in CARD_DIR.glob("*.jpg") if not p.stem.endswith("_b")]
    return random.choice(fronts) if fronts else None


def get_card_caption(image_path: Path, client: anthropic.Anthropic) -> str:
    """Send card image to Claude, get player name + quick fact under 200 chars."""
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    response = client.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data},
                },
                {
                    "type": "text",
                    "text": (
                        "This is a sports trading card. Who is the player? "
                        "Reply with their name and one quick interesting fact. "
                        "Keep the whole reply under 200 characters."
                    ),
                },
            ],
        }],
    )

    text = next((b.text for b in response.content if b.type == "text"), "")
    return " ".join(text.split())


def build_text_message(client: anthropic.Anthropic) -> str:
    today = datetime.date.today().strftime("%B %d, %Y")
    question = random.choice(PROMPT_QUESTIONS)

    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": f"Today is {today}. {question}"}],
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


def send_image_via_imessage(image_path: Path, caption: str) -> None:
    import_to_photos(image_path)
    subprocess.run(["shortcuts", "run", "Text Latest Image"], check=True)
    send_via_shortcut(SHORTCUT_NAME, caption)


def main() -> None:
    force_card = "--card" in sys.argv
    client = anthropic.Anthropic(api_key=get_api_key())

    if force_card or random.random() < 0.5:
        card = pick_random_card()
        if card:
            caption = get_card_caption(card, client)
            send_image_via_imessage(card, caption)
            return
        if force_card:
            raise RuntimeError(f"Card directory not available: {CARD_DIR}")
        # Drive not mounted — fall through to text message

    message = build_text_message(client)
    send_via_shortcut(SHORTCUT_NAME, message)


if __name__ == "__main__":
    main()
