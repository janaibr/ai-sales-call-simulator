"""
Shared Gemini helpers for all the games.

Two things every game needs:
  call_json(...)    one request that must come back as JSON matching a schema
  chat_stream(...)  one turn of a role-play conversation, printed as it streams

Set GEMINI_API_KEY (free from aistudio.google.com). Optionally set GEMINI_MODEL.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")  # hide Python 3.9 end-of-life notices from Google's libraries
logging.getLogger("google_genai").setLevel(logging.ERROR)  # hide "non-text parts" notices that clutter the call

from google import genai  # noqa: E402
from google.genai import errors, types  # noqa: E402

MODEL = os.environ.get("GEMINI_MODEL", "gemini-4-hydrogen")
BACKUP_MODEL = "gemini-flash-latest"  # used when the main model is overloaded

_client = None


def _busy(e: Exception) -> bool:
    return isinstance(e, errors.APIError) and e.code in (500, 503)


def _with_retry(request):
    """Run request(model). If Google is overloaded, wait and retry, then try the backup model."""
    for attempt, model in enumerate([MODEL, MODEL, BACKUP_MODEL]):
        try:
            return request(model)
        except errors.APIError as e:
            if not _busy(e) or attempt == 2:
                raise
            print(f"\033[2m(Gemini is busy, retrying...)\033[0m", flush=True)
            time.sleep(2 * (attempt + 1))


def client() -> genai.Client:
    """Created on first use, so a missing key gives a clear message, not a crash at import."""
    global _client
    if _client is None:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            sys.exit("No API key found. Run:  export GEMINI_API_KEY=your-key-here")
        _client = genai.Client()
    return _client


def call_json(system: str, prompt: str, schema: dict) -> dict:
    """One request whose answer is guaranteed to be JSON in the given schema."""
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_json_schema=schema,
    )
    resp = _with_retry(lambda m: client().models.generate_content(model=m, contents=prompt, config=config))
    if not resp.text:
        sys.exit("The model returned nothing (it may have blocked the request). Run again for a new scenario.")
    return json.loads(resp.text)


def chat_stream(system: str, history: list, user_text: str, note: str | None = None) -> str:
    """Send one turn of the conversation, print the reply as it arrives, return it.

    history is a list of Gemini messages and is updated in place.
    note is a private stage direction (e.g. "you're getting nervous"). Gemini has no
    mid-conversation system role, so it rides along in brackets on the user's turn.
    """
    text = user_text
    if note:
        text += f"\n\n[Private stage direction, not said by the other person: {note}]"
    history.append({"role": "user", "parts": [{"text": text}]})

    config = types.GenerateContentConfig(system_instruction=system)

    def stream(model: str) -> str:
        pieces = []
        for chunk in client().models.generate_content_stream(model=model, contents=history, config=config):
            if chunk.text:
                print(chunk.text, end="", flush=True)
                pieces.append(chunk.text)
        return "".join(pieces)

    reply = _with_retry(stream)
    history.append({"role": "model", "parts": [{"text": reply}]})
    return reply


def run(main) -> None:
    """Run a game and turn API errors into plain-English messages."""
    try:
        main()
    except errors.APIError as e:
        if e.code in (400, 401, 403) and "key" in str(e).lower():
            sys.exit("Your Gemini API key isn't valid. Copy it again from aistudio.google.com.")
        if e.code == 429:
            sys.exit("You've hit the free tier's limit. Wait a minute (or until tomorrow) and try again.")
        if e.code == 404:
            names = [m.name for m in client().models.list() if "generateContent" in (m.supported_actions or [])]
            sys.exit(f"Model '{MODEL}' isn't available. Pick one and run e.g.\n"
                     f"  export GEMINI_MODEL={names[0].split('/')[-1] if names else 'gemini-2.5-flash'}\n"
                     "Available:\n  " + "\n  ".join(n.split("/")[-1] for n in names))
        sys.exit(f"Gemini API error {e.code}: {e}")
