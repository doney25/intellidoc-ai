import sys
from pathlib import Path

import streamlit as st

# Add project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ingestion.parsers import extract_document
from app.ingestion.cleaning import clean_text
from app.ingestion.chunking import split_text
from app.retrieval.vector_store import (
    add_chunks,
    search_chunks,
    list_documents,
    delete_document,
)
from app.llm.generator import generate_answer


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="IntelliDoc AI",
    page_icon="📚",
    layout="wide",
)

st.title("📚 IntelliDoc AI")
st.subheader("Your AI-powered document intelligence assistant")

st.caption(
    "Upload documents, explore their contents, "
    "and get answers using Retrieval-Augmented Generation."
)


# --------------------------------------------------
# Sidebar: Document management
# --------------------------------------------------

st.sidebar.header("📄 Document Library")

uploaded_file = st.sidebar.file_uploader(
    "Upload a document",
    type=["pdf", "docx", "txt"],
    help="Supported formats: PDF, DOCX, and TXT.",
)

if st.sidebar.button(
    "Process Document",
    disabled=uploaded_file is None,
):
    try:
        with st.spinner("Processing your document..."):
            upload_dir = PROJECT_ROOT / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)

            # Use a unique filename to avoid overwriting uploads.
            import hashlib

            file_bytes = uploaded_file.getvalue()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            safe_name = Path(uploaded_file.name).name
            saved_path = upload_dir / f"{file_hash[:12]}_{safe_name}"

            saved_path.write_bytes(file_bytes)

            # Extract and clean text.
            extracted = extract_document(str(saved_path))

            chunks = []

            for page in extracted["pages"]:
                cleaned_text = clean_text(page["text"])

                if not cleaned_text:
                    continue

                page_chunks = split_text(cleaned_text)

                for index, chunk_text in enumerate(
                    page_chunks,
                    start=1,
                ):
                    chunks.append({
                        "chunk_id": (
                            f"{file_hash}_{page['page_number']}_{index}"
                        ),
                        "document_name": uploaded_file.name,
                        "page_number": page["page_number"],
                        "text": chunk_text,
                    })

            if not chunks:
                st.warning(
                    "No extractable text was found. "
                    "The document may be scanned or empty."
                )
            else:
                indexed_count = add_chunks(chunks)

                st.sidebar.success(
                    f"Indexed {indexed_count} chunks successfully."
                )

    except Exception as error:
        st.sidebar.error(f"Document processing failed: {error}")


# --------------------------------------------------
# Display indexed documents
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.subheader("Indexed Documents")

try:
    documents = list_documents()

    if not documents:
        st.sidebar.info("No documents indexed yet.")

    for document in documents:
        with st.sidebar.expander(
            f"📄 {document['document_name']}"
        ):
            st.write(
                f"Chunks: {document['chunk_count']}"
            )

            if st.button(
                "Delete document",
                key=f"delete_{document['document_id']}",
            ):
                deleted = delete_document(
                    document["document_id"]
                )

                st.sidebar.success(
                    f"Deleted {deleted} chunks."
                )
                st.rerun()

except Exception as error:
    st.sidebar.error(
        f"Could not load document library: {error}"
    )


# --------------------------------------------------
# Main chat interface
# --------------------------------------------------

st.divider()
st.header("💬 Ask Your Documents")

question = st.text_input(
    "Enter your question",
    placeholder="e.g. What technologies were used in my traffic prediction project?",
)

if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")

    elif not documents:
        st.warning("Upload and process a document first.")

    else:
        try:
            with st.spinner(
                "Searching documents and generating your answer..."
            ):
                # Retrieve relevant passages.
                retrieved_chunks = search_chunks(
                    query=question,
                    top_k=3,
                )

                # Generate a grounded answer.
                answer = generate_answer(
                    question,
                    retrieved_chunks,
                )

            st.subheader("Answer")
            st.markdown(answer)

            # Display retrieved source passages.
            st.subheader("📚 Retrieved Sources")

            if retrieved_chunks:
                for index, chunk in enumerate(
                    retrieved_chunks,
                    start=1,
                ):
                    page = chunk["page_number"]
                    page_label = (
                        f"Page {page}"
                        if page is not None
                        else "Page unavailable"
                    )

                    with st.expander(
                        f"Source {index}: "
                        f"{chunk['document_name']} — {page_label}"
                    ):
                        st.write(chunk["text"])
                        st.caption(
                            f"Cosine distance: "
                            f"{chunk['distance']:.4f}"
                        )
            else:
                st.info("No relevant passages were found.")

        except Exception as error:
            st.error(f"Question answering failed: {error}")