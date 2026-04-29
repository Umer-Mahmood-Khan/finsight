import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ── Constants ──────────────────────────────────────────────
DATA_DIR        = Path("data")
VECTORSTORE_DIR = Path("vectorstore")
EMBEDDING_MODEL = "text-embedding-3-small"

# Why text-embedding-3-small?
# Cost: $0.00002 per 1000 tokens — processing a 200-page annual
# report costs less than $0.02. For a financial services demo
# this is negligible. The model produces 1536-dim vectors which
# give excellent semantic similarity for financial text.


def load_documents(data_dir: Path) -> list:
    """
    Loads all PDFs from the data/ folder.
    Adds filename and page number to each document's metadata
    so we can cite sources in every answer.
    """
    documents = []
    pdf_files = list(data_dir.glob("**/*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {data_dir}/")
        return []

    for pdf_path in pdf_files:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = pdf_path.name
        documents.extend(docs)
        print(f"    {len(docs)} pages loaded")

    print(f"\nTotal pages loaded: {len(documents)}")
    return documents


def chunk_documents(documents: list) -> list:
    """
    Splits documents into chunks for embedding.

    Why 600 characters for financial documents vs 800 for medical?
    Financial reports have dense numerical tables and short paragraphs.
    Smaller chunks keep tables and their headers together and give
    more precise retrieval for specific figures like revenue or ratios.

    Why RecursiveCharacterTextSplitter?
    It tries paragraph breaks first, then sentence breaks, then word
    breaks — always preferring semantically meaningful split points
    over arbitrary character counts.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list) -> FAISS:
    """
    Embeds all chunks and builds a FAISS index.

    Each chunk becomes a 1536-dimensional vector.
    FAISS builds an index so that at query time, given
    a question vector, it can find the top-k most similar
    chunks in milliseconds — this is semantic search,
    not keyword matching.
    """
    print("Embedding chunks — calling OpenAI API...")

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("FAISS index built successfully")
    return vectorstore


def save_vectorstore(vectorstore: FAISS):
    """
    Saves the FAISS index to disk.
    Two files created:
      vectorstore/index.faiss — the vector index
      vectorstore/index.pkl   — chunk text and metadata
    """
    VECTORSTORE_DIR.mkdir(exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"Vectorstore saved to {VECTORSTORE_DIR}/")


def load_vectorstore() -> FAISS:
    """
    Loads existing FAISS index from disk.
    Called by agents at query time — no re-embedding needed.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    return FAISS.load_local(
        str(VECTORSTORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )


def get_document_hash(data_dir: Path) -> str:
    """
    Hashes all PDF filenames and sizes.
    If nothing changed, we reuse the existing vectorstore
    instead of re-embedding — saves API cost and time.
    """
    files = sorted(data_dir.glob("**/*.pdf"))
    hash_input = "".join(
        f"{f.name}{f.stat().st_size}" for f in files
    )
    return hashlib.md5(hash_input.encode()).hexdigest()


def ingest(force_rebuild: bool = False) -> FAISS:
    """
    Main entry point for the ingestion pipeline.

    Smart rebuild logic:
    - First run: embeds everything, saves to disk (~20 sec per 100 pages)
    - Subsequent runs with same docs: loads from disk instantly
    - New document added: detects change via hash, rebuilds automatically

    This is important for a financial services tool where analysts
    add new reports regularly — they should not wait for re-embedding
    every time they open the app.
    """
    DATA_DIR.mkdir(exist_ok=True)
    hash_file = VECTORSTORE_DIR / "doc_hash.txt"

    if not force_rebuild and VECTORSTORE_DIR.exists() and hash_file.exists():
        current_hash = get_document_hash(DATA_DIR)
        saved_hash   = hash_file.read_text().strip()

        if current_hash == saved_hash:
            print("Documents unchanged — loading existing vectorstore")
            return load_vectorstore()
        else:
            print("New documents detected — rebuilding vectorstore")

    print("Starting ingestion pipeline...")
    documents = load_documents(DATA_DIR)

    if not documents:
        return None

    chunks      = chunk_documents(documents)
    vectorstore = build_vectorstore(chunks)
    save_vectorstore(vectorstore)

    hash_file.write_text(get_document_hash(DATA_DIR))
    print("Ingestion complete.")
    return vectorstore


if __name__ == "__main__":
    ingest(force_rebuild=True)