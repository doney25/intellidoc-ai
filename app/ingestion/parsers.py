
from pathlib import Path

import fitz  # PyMuPDF
from docx import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def extract_pdf(file_path: str) -> list[dict]:
    """Extract text page by page from a PDF document."""

    pages = []

    with fitz.open(file_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            pages.append({
                "page_number": page_number,
                "text": text
            })

    return pages


def extract_docx(file_path: str) -> list[dict]:
    """Extract text from a Word document."""

    document = Document(file_path)

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [{
        "page_number": None,
        "text": text
    }]


def extract_txt(file_path: str) -> list[dict]:
    """Extract text from a plain-text file."""

    path = Path(file_path)

    text = path.read_text(encoding="utf-8")

    return [{
        "page_number": None,
        "text": text
    }]


def extract_document(file_path: str) -> dict:
    """Extract text from a supported document."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    if extension == ".pdf":
        pages = extract_pdf(str(path))

    elif extension == ".docx":
        pages = extract_docx(str(path))

    else:
        pages = extract_txt(str(path))

    return {
        "document_name": path.name,
        "file_type": extension,
        "pages": pages
    }

