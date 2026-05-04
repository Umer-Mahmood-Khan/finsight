import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.tools import tool

load_dotenv()

VECTORSTORE_DIR = Path("vectorstore")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL       = "gpt-4o-mini"
TOP_K_CHUNKS    = 6

# Why 6 chunks for financial documents vs 5 for medical?
# Financial reports reference the same figure across multiple
# sections — a revenue number appears in the CEO letter,
# the financial statements, and the MD&A. Retrieving 6 chunks
# gives better coverage across these repeated references.


def get_vectorstore() -> FAISS:
    """
    Loads the FAISS vectorstore from disk.
    Called once per agent session — not on every query.
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


def retrieve_chunks(query: str, vectorstore: FAISS) -> list:
    """
    Core retrieval function.
    Embeds the query and finds the top-k most semantically
    similar chunks from the FAISS index.

    Returns chunks with relevance scores — the score is
    critical for the confidence calculation downstream.
    A high average score means the documents clearly contain
    the answer. A low score means the topic may not be covered.

    Why similarity_search_with_score over similarity_search?
    The score tells us how relevant the retrieved chunks are.
    We use this in the critic agent to assign confidence:
    - avg score > 0.8: HIGH confidence
    - avg score > 0.6: MEDIUM confidence
    - avg score < 0.6: LOW confidence — answer may not be in docs
    """
    results = vectorstore.similarity_search_with_score(
        query, k=TOP_K_CHUNKS
    )

    chunks = []
    for doc, score in results:
        chunks.append({
            "content":  doc.page_content,
            "source":   doc.metadata.get("source", "unknown"),
            "page":     doc.metadata.get("page", 0),
            "score":    round(float(score), 4)
        })

    return chunks


def format_context(chunks: list) -> str:
    """
    Formats retrieved chunks into a context string for the LLM.
    Each chunk is labelled with its source and page number so
    the LLM can include proper citations in its answer.

    Format example:
    [Source 1: rbc_annual_2024.pdf, page 45, relevance: 0.923]
    ... chunk text ...
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk['source']}, "
            f"page {chunk['page']}, "
            f"relevance: {chunk['score']}]\n"
            f"{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)


def calculate_confidence(chunks: list) -> tuple[str, float]:
    """
    Calculates confidence level based on retrieval scores.

    FAISS returns L2 distance scores — lower is MORE similar.
    We invert and normalise to get a 0-1 similarity score.

    Thresholds are calibrated for financial text:
    - HIGH (>0.75):   strong semantic match, answer likely in docs
    - MEDIUM (>0.55): partial match, answer may be incomplete
    - LOW (<0.55):    weak match, topic likely not in documents
    """
    if not chunks:
        return "LOW", 0.0

    # Convert L2 distances to similarity scores
    # FAISS L2 distance of 0 = identical, higher = less similar
    # We use 1/(1+distance) to get a 0-1 similarity score
    scores = [1 / (1 + chunk["score"]) for chunk in chunks]
    avg_score = sum(scores) / len(scores)

    if avg_score >= 0.75:
        level = "HIGH"
    elif avg_score >= 0.55:
        level = "MEDIUM"
    else:
        level = "LOW"

    return level, round(avg_score, 3)


def run_retrieval_agent(
    query: str,
    vectorstore: FAISS
) -> dict:
    """
    Main retrieval agent function.

    Retrieves relevant chunks, formats context, calculates
    confidence, and returns everything needed by downstream
    agents — the analysis agent, critic agent, and aggregator.

    Returns a structured dict so LangGraph can pass it cleanly
    through the StateGraph to the next agent.
    """
    start_time = time.time()

    # Step 1: retrieve relevant chunks
    chunks = retrieve_chunks(query, vectorstore)

    # Step 2: calculate confidence from retrieval scores
    confidence_level, confidence_score = calculate_confidence(chunks)

    # Step 3: format context for LLM consumption
    context = format_context(chunks)

    # Step 4: get unique source documents
    sources = list(set(chunk["source"] for chunk in chunks))

    latency_ms = round((time.time() - start_time) * 1000, 1)

    return {
        "query":            query,
        "chunks":           chunks,
        "context":          context,
        "confidence_level": confidence_level,
        "confidence_score": confidence_score,
        "sources":          sources,
        "latency_ms":       latency_ms,
        "top_score":        chunks[0]["score"] if chunks else None
    }


if __name__ == "__main__":
    import json

    print("Testing retrieval agent...")
    print("Loading vectorstore...")

    vectorstore = get_vectorstore()
    print("Vectorstore loaded.\n")

    test_queries = [
        "What is the total revenue reported in this document?",
        "What are the main risk factors mentioned?",
        "What does the document say about climate risk?",
    ]

    for query in test_queries:
        print(f"Q: {query}")
        print("-" * 55)
        result = run_retrieval_agent(query, vectorstore)
        print(f"Confidence:  {result['confidence_level']} "
              f"({result['confidence_score']})")
        print(f"Sources:     {result['sources']}")
        print(f"Chunks:      {len(result['chunks'])} retrieved")
        print(f"Latency:     {result['latency_ms']}ms")
        print(f"Top chunk preview:")
        print(f"  {result['chunks'][0]['content'][:200]}...")
        print()