from pathlib import Path

from app.ingestion.parsers import extract_document
from app.ingestion.cleaning import clean_text
from app.ingestion.chunking import split_text


def process_document(file_path: str) -> list[dict]:
    """Extract, clean, and chunk a document."""

    document = extract_document(file_path)

    processed_chunks = []
    chunk_counter = 0

    for page in document["pages"]:
        cleaned_text = clean_text(page["text"])

        if not cleaned_text:
            continue

        chunks = split_text(cleaned_text)

        for chunk in chunks:
            chunk_counter += 1

            processed_chunks.append({
                "chunk_id": (
                    f"{Path(file_path).stem}_{chunk_counter}"
                ),
                "document_name": document["document_name"],
                "page_number": page["page_number"],
                "text": chunk
            })

    return processed_chunks


if __name__ == "__main__":
    file_path = input(
        "Enter the path to a PDF, DOCX, or TXT file: "
    ).strip().strip('"')

    try:
        chunks = process_document(file_path)

        print(f"\nTotal chunks: {len(chunks)}")

        for chunk in chunks[:3]:
            print("\n" + "=" * 60)
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Document: {chunk['document_name']}")
            print(f"Page: {chunk['page_number']}")
            print(f"Text preview: {chunk['text'][:300]}")

    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error processing document: {error}")
