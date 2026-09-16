# pipeline/converters/text_passthrough.py
"""TXT/MD -> Markdown: already plain text, just normalize encoding/mojibake
and strip stray null bytes / control characters some exports leave in."""
import ftfy


def convert(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = text.replace("\x00", "")
    return ftfy.fix_text(text).strip()
