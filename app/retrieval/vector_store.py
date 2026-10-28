from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "intellidoc_documents"


# --------------------------------------------------
# Embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# ChromaDB initialization
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def create_document_id(document_name: str) -> str:
    """Create a stable identifier for a document name."""

    return str(
        uuid5(
            NAMESPACE_URL,
            document_name.strip().lower()
        )
    )


def create_chunk_id(
    document_id: str,
    original_chunk_id: str
) -> str:
    """Create a unique chunk ID scoped to a document."""

    return str(
        uuid5(
            NAMESPACE_URL,
            f"{document_id}:{original_chunk_id}"
        )
    )


# --------------------------------------------------
# Add document chunks
# --------------------------------------------------

def add_chunks(chunks: list[dict]) -> int:
    """Generate embeddings and store document chunks."""

    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []

    for chunk in chunks:
        document_name = chunk["document_name"]

        document_id = create_document_id(document_name)

        original_chunk_id = str(chunk["chunk_id"])

        ids.append(
            create_chunk_id(
                document_id,
                original_chunk_id
            )
        )

        documents.append(chunk["text"])

        metadatas.append({
            "document_id": document_id,
            "document_name": document_name,
            "original_chunk_id": original_chunk_id,
            "page_number": (
                chunk["page_number"]
                if chunk["page_number"] is not None
                else -1
            )
        })

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return len(ids)


# --------------------------------------------------
# Semantic search
# --------------------------------------------------

def search_chunks(
    query: str,
    top_k: int = 3,
    document_id: str | None = None
) -> list[dict]:
    """Retrieve relevant chunks, optionally from one document."""

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    query_embedding = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Restrict retrieval to a specific document when requested.
    where_filter = (
        {"document_id": document_id}
        if document_id
        else None
    )

    query_arguments = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, total_chunks),
        "include": [
            "documents",
            "metadatas",
            "distances"
        ]
    }

    if where_filter:
        query_arguments["where"] = where_filter

    results = collection.query(**query_arguments)

    matches = []

    if not results["documents"] or not results["documents"][0]:
        return []

    for index, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][index]

        matches.append({
            "text": document,
            "document_id": metadata["document_id"],
            "document_name": metadata["document_name"],
            "page_number": (
                None
                if metadata["page_number"] == -1
                else metadata["page_number"]
            ),
            "distance": results["distances"][0][index]
        })

    return matches


# --------------------------------------------------
# List indexed documents
# --------------------------------------------------

def list_documents() -> list[dict]:
    """Return distinct documents, including legacy database records."""

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    stored_data = collection.get(
        include=["metadatas"]
    )

    documents_by_id = {}

    for index, metadata in enumerate(stored_data["metadatas"]):
        metadata = metadata or {}

        document_name = metadata.get(
            "document_name",
            "unknown_document"
        )

        # Support older records that don't have document_id.
        document_id = metadata.get("document_id")

        if not document_id:
            document_id = create_document_id(document_name)

        if document_id not in documents_by_id:
            documents_by_id[document_id] = {
                "document_id": document_id,
                "document_name": document_name,
                "chunk_count": 0
            }

        documents_by_id[document_id]["chunk_count"] += 1

    return list(documents_by_id.values())
# --------------------------------------------------
# Delete a document
# --------------------------------------------------

def delete_document(document_id: str) -> int:
    """Delete all chunks belonging to a specific document."""

    if not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    existing = collection.get(
        where={"document_id": document_id},
        include=["metadatas"]
    )

    ids_to_delete = existing["ids"]

    if not ids_to_delete:
        return 0

    collection.delete(ids=ids_to_delete)

    return len(ids_to_delete)