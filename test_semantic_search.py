from test_ingestion import process_document

from app.retrieval.vector_store import (
    add_chunks,
    search_chunks
)


def main():
    file_path = input(
        "Enter the path to your document: "
    ).strip().strip('"')

    print("\nProcessing document...")

    chunks = process_document(file_path)

    print(f"Extracted {len(chunks)} chunks.")

    stored_count = add_chunks(chunks)

    print(f"Stored {stored_count} chunks in ChromaDB.")

    while True:
        query = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if query.lower() == "exit":
            break

        if not query:
            print("Please enter a question.")
            continue

        results = search_chunks(query, top_k=3)

        if not results:
            print("No indexed document chunks found.")
            continue

        print("\nMost relevant passages:")

        for rank, result in enumerate(results, start=1):
            print(f"\n{'=' * 60}")
            print(f"Result: {rank}")
            print(f"Document: {result['document_name']}")
            print(f"Page: {result['page_number']}")
            print(f"Cosine distance: {result['distance']:.4f}")
            print(f"Text: {result['text']}")


if __name__ == "__main__":
    main()

