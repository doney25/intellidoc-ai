from app.retrieval.vector_store import (
    add_chunks,
    search_chunks,
    list_documents,
    delete_document,
    create_document_id
)


def main():
    print("\n=== IntelliDoc AI: Document Management Test ===")

    test_document_name = "intellidoc_test_document.txt"
    test_document_id = create_document_id(test_document_name)

    test_chunks = [
        {
            "chunk_id": "test_chunk_1",
            "document_name": test_document_name,
            "page_number": 1,
            "text": (
                "IntelliDoc AI uses semantic search "
                "to retrieve relevant document passages."
            )
        },
        {
            "chunk_id": "test_chunk_2",
            "document_name": test_document_name,
            "page_number": 2,
            "text": (
                "ChromaDB stores vector embeddings "
                "for similarity-based retrieval."
            )
        }
    ]

    # Index the test document.
    added = add_chunks(test_chunks)
    print(f"\nChunks indexed: {added}")

    # List indexed documents.
    print("\nIndexed documents:")

    for document in list_documents():
        print(
            f"- {document['document_name']} "
            f"({document['chunk_count']} chunks)"
        )

    # Search within the test document.
    print("\nSearching only the test document:")

    results = search_chunks(
        query="How does semantic search work?",
        top_k=3,
        document_id=test_document_id
    )

    for result in results:
        print(f"\nDocument: {result['document_name']}")
        print(f"Page: {result['page_number']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Text: {result['text']}")

    # Clean up test data.
    deleted = delete_document(test_document_id)
    print(f"\nTest chunks deleted: {deleted}")


if __name__ == "__main__":
    main()