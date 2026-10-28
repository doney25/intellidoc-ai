from test_ingestion import process_document

from app.retrieval.vector_store import (
    add_chunks,
    search_chunks
)

from app.llm.generator import generate_answer


def main():
    file_path = input(
        "Enter the path to your document: "
    ).strip().strip('"')

    print("\nProcessing document...")

    chunks = process_document(file_path)

    if not chunks:
        print("No readable text was found in the document.")
        return

    stored_count = add_chunks(chunks)

    print(f"Indexed {stored_count} document chunks.")

    print("\nIntelliDoc AI is ready!")
    print("Ask questions about your document.")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            retrieved_chunks = search_chunks(
                question,
                top_k=3
            )

            print("\n--- RETRIEVED CONTEXT DEBUG ---")

            for i, chunk in enumerate(retrieved_chunks, start=1):
                print(f"\nSource {i}")
                print(f"Document: {chunk['document_name']}")
                print(f"Page: {chunk['page_number']}")
                print(f"Distance: {chunk['distance']:.4f}")
                print(f"Text: {chunk['text'][:800]}")

            print("\n--- END DEBUG ---\n")

            answer = generate_answer(
                question,
                retrieved_chunks
            )

            print("\nIntelliDoc AI:")
            print(answer)

        except Exception as error:
            print(f"\nRequest failed: {error}")


if __name__ == "__main__":
    main()
