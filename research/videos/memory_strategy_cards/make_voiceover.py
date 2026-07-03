#!/usr/bin/env python3
"""Generate voiceover segments for the Manim memory strategy video.

Default path uses OpenAI TTS if OPENAI_API_KEY is available. Fallback uses
macOS `say`, which is lower quality but makes the video fully local.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "script.md"
AUDIO_DIR = ROOT / "audio"
MANIFEST = AUDIO_DIR / "manifest.json"


def load_dotenv() -> None:
    env = ROOT.parents[2] / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_sections() -> list[dict[str, str]]:
    text = SCRIPT.read_text()
    parts = re.split(r"^##\s+", text, flags=re.MULTILINE)
    sections = []
    for part in parts[1:]:
        lines = part.strip().splitlines()
        if not lines:
            continue
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        key = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        sections.append({"key": key, "title": title, "text": body})
    return sections


def duration_seconds(path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    return float(out)


def generate_openai(text: str, out: Path) -> None:
    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    voice = os.getenv("OPENAI_TTS_VOICE", "ash")
    instructions = (
        "Read in a clear, calm, technical YouTube narration style. "
        "Do not sound salesy. Keep a steady pace with slight emphasis on key terms."
    )
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions=instructions,
        response_format="mp3",
    ) as response:
        response.stream_to_file(out)


def generate_macos_say(text: str, out: Path) -> None:
    aiff = out.with_suffix(".aiff")
    subprocess.run(["say", "-v", "Samantha", "-r", "168", "-o", str(aiff), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-i", str(aiff), "-codec:a", "libmp3lame", "-q:a", "2", str(out)], check=True)
    aiff.unlink(missing_ok=True)


def main() -> None:
    load_dotenv()
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    sections = parse_sections()
    use_openai = bool(os.getenv("OPENAI_API_KEY"))
    if use_openai:
        try:
            import openai  # noqa: F401
        except Exception:
            use_openai = False

    if not use_openai and not shutil.which("say"):
        raise SystemExit("No OPENAI_API_KEY/openai package and macOS say is unavailable.")

    manifest = []
    for i, section in enumerate(sections):
        out = AUDIO_DIR / f"{i:02d}_{section['key']}.mp3"
        if not out.exists():
            print(f"[voice] {section['title']} -> {out.name}")
            if use_openai:
                generate_openai(section["text"], out)
            else:
                generate_macos_say(section["text"], out)
        dur = duration_seconds(out)
        manifest.append(
            {
                "index": i,
                "key": section["key"],
                "title": section["title"],
                "file": str(out.relative_to(ROOT)),
                "duration": dur,
                "text": section["text"],
            }
        )
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"[voice] wrote {MANIFEST}")


if __name__ == "__main__":
    main()
