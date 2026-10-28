import re


def clean_text(text: str) -> str:
    """Clean extracted document text."""

    if not text:
        return ""

    # Replace non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Normalize line endings
    text = text.replace("\r\n", "\n")

    # Remove unnecessary spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Reduce excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
