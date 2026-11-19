"""ElevenLabs API adapter for the TTS tab."""

import os
import re
from urllib.parse import quote

import httpx


API = "https://api.elevenlabs.io"
MODELS = ("eleven_multilingual_v2", "eleven_flash_v2_5")


def _headers(key):
    key = (key or "").strip() or os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise ValueError("Enter an ElevenLabs API key and click Refresh voices.")
    if not key.isascii() or any(char.isspace() for char in key):
        raise ValueError("The ElevenLabs API key contains invalid characters.")
    return {"xi-api-key": key}


def _check(response):
    if response.status_code in (401, 403):
        response.read()
        try:
            detail = response.json().get("detail", {})
        except (ValueError, AttributeError):
            detail = {}
        if isinstance(detail, dict) and detail.get("status") == "missing_permissions":
            raise RuntimeError("Enable Voices Read to list voices and Text to Speech access to generate audio.")
    if response.status_code == 401:
        raise RuntimeError("ElevenLabs rejected the API key.")
    if response.status_code == 403:
        raise RuntimeError("The ElevenLabs API key lacks access to this voice or operation.")
    if response.status_code == 429:
        raise RuntimeError("ElevenLabs rate limit reached. Try again later.")
    if not response.is_success:
        raise RuntimeError(f"ElevenLabs returned HTTP {response.status_code}. Check your voice, permissions and credits.")


def list_voices(key):
    params = {"page_size": 100}
    catalog = {}
    headers = _headers(key)
    try:
        with httpx.Client(headers=headers, timeout=30) as client:
            while True:
                response = client.get(API + "/v2/voices", params=params)
                _check(response)
                data = response.json()
                for voice in data["voices"]:
                    voice_id = voice["voice_id"]
                    labels = voice.get("labels") or {}
                    catalog[voice_id] = {
                        "ShortName": f"{voice.get('name') or 'Voice'} [{voice_id}]",
                        "Locale": labels.get("language") or "Unspecified",
                        "Gender": (labels.get("gender") or "Unspecified").title(),
                        "VoiceId": voice_id,
                    }
                if not data.get("has_more"):
                    return list(catalog.values())
                token = data.get("next_page_token")
                if not token or token == params.get("next_page_token"):
                    raise RuntimeError("ElevenLabs returned an incomplete voice list.")
                params["next_page_token"] = token
    except httpx.RequestError:
        raise RuntimeError("Unable to reach ElevenLabs. Check your internet connection.") from None
    except (KeyError, TypeError, ValueError):
        raise RuntimeError("ElevenLabs could not return a valid voice list.") from None


def synthesize(text, voice_id, key, model, destination):
    if model not in MODELS:
        raise ValueError("Choose an ElevenLabs speech model.")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", voice_id):
        raise ValueError("Enter a valid ElevenLabs Voice ID.")
    endpoint = API + "/v1/text-to-speech/" + quote(voice_id, safe="")
    try:
        with httpx.stream(
            "POST", endpoint, params={"output_format": "mp3_44100_128"},
            headers={**_headers(key), "Accept": "audio/mpeg"},
            json={"text": text, "model_id": model}, timeout=90,
        ) as response:
            _check(response)
            with destination.open("wb") as audio:
                for chunk in response.iter_bytes():
                    audio.write(chunk)
    except httpx.RequestError:
        raise RuntimeError("Unable to complete ElevenLabs speech generation.") from None
    if not destination.stat().st_size:
        raise RuntimeError("ElevenLabs returned no audio.")
