import json
from pathlib import Path

import chromadb
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "train.jsonl"
CHROMA_PATH = str(ROOT / "chroma_db")
COLLECTION_NAME = "aws_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256


def load_data(file_path=DATA_PATH):
    """Load documents from a JSONL file ({"text": ..., "source": ...})."""
    documents = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                continue
            text = data.get("text")
            source = data.get("source")
            if text:
                metadata = {"source": source} if source else {}
                documents.append(Document(page_content=text, metadata=metadata))
    return documents


def split_documents(docs):
    """Split documents into 512-character chunks (~20% overlap)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=102,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def create_vector_store(docs):
    """Embed the chunks and store them in Chroma (the collection is rebuilt on every call)."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # Starting from an empty collection avoids duplicates / ID conflicts on re-runs
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start:start + BATCH_SIZE]
        texts = [doc.page_content for doc in batch]
        collection.add(
            ids=[str(start + i) for i in range(len(batch))],
            embeddings=model.encode(texts).tolist(),
            documents=texts,
            metadatas=[doc.metadata for doc in batch],
        )
        print(f"  {min(start + BATCH_SIZE, len(docs))}/{len(docs)} chunks indexed", end="\r")
    print()
    return collection


if __name__ == "__main__":
    print("Loading documents...")
    docs = load_data()
    print(f"{len(docs)} documents loaded")
    print("\nSplitting into chunks...")
    chunks = split_documents(docs)
    print(f"{len(chunks)} chunks created")
    print("\nBuilding the vector store...")
    collection = create_vector_store(chunks)
    print(f"Vector store built successfully ({collection.count()} chunks)")
