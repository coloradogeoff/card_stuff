#!/usr/bin/env python3
import os
import random
import subprocess
from pathlib import Path

import anthropic

# Pick one question at random each run.
PROMPT_QUESTIONS = [
    "Who is famous that was born on this day? 240 chars or less bio.",
    "Give one notable historical event that happened on this day in 240 chars or less.",
    "What is one interesting science fact for today? Keep it under 240 chars.",
    "Send me a hiku for today. 240 chars or less.",
]

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-6")
SHORTCUT_NAME = "Send Message"
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


def build_message() -> str:
    import datetime
    today = datetime.date.today().strftime("%B %d, %Y")
    question = random.choice(PROMPT_QUESTIONS)
    client = anthropic.Anthropic(api_key=get_api_key())

    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": f"Today is {today}. {question}"}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "")
    if not text:
        raise RuntimeError("Claude returned no text")

    return " ".join(text.split())


def send_via_shortcut(message: str) -> None:
    subprocess.run(
        ["shortcuts", "run", SHORTCUT_NAME],
        input=message,
        text=True,
        check=True,
    )


def main() -> None:
    message = build_message()
    send_via_shortcut(message)


if __name__ == "__main__":
    main()
