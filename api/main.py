import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.ingest import ingest, load_vectorstore
from agents.pii_filter import redact_pii, redact_document_chunks
from agents.audit import log_query, log_ingest, get_recent_queries, get_recent_ingests, get_audit_stats
from agents.retrieval import run_retrieval_agent, get_vectorstore
from agents.risk_extractor import run_risk_extractor
from agents.compliance_summariser import run_compliance_summariser

load_dotenv()

# ── Why FastAPI over Flask? ────────────────────────────────
# 1. Async support — handles concurrent analyst requests
# 2. Automatic /docs endpoint — interactive API documentation
#    generated from type hints, no extra work needed
# 3. Pydantic validation — request/response schemas enforced
#    automatically, bad inputs rejected before hitting agents
# 4. Industry standard for Python AI APIs in 2024/2025
# 5. Type hints make the code self-documenting

app = FastAPI(
    title="FinSight API",
    description="AI-powered financial document intelligence platform. "
                "Built for compliance-aware analysis of financial documents. "
                "Every query is logged for audit purposes.",
    version="1.0.0",
    docs_url="/docs",      # interactive Swagger UI
    redoc_url="/redoc"     # alternative ReDoc UI
)

# CORS middleware — allows Streamlit frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ───────────────────────────────────────────
# Vectorstore loaded once on startup — not on every request
# This is the production pattern — loading FAISS on every
# query would add 2-3 seconds of latency per request
_vectorstore = None


def get_vs():
    """Returns the global vectorstore, loading if needed."""
    global _vectorstore
    if _vectorstore is None:
        try:
            _vectorstore = load_vectorstore()
        except Exception:
            pass  # vectorstore not built yet — ingest first
    return _vectorstore


# ── Request / Response schemas ─────────────────────────────
# Pydantic models define the API contract.
# FastAPI validates every incoming request against these.
# Invalid requests get a 422 error with clear field-level
# error messages — no custom validation code needed.

class QueryRequest(BaseModel):
    question:   str
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "question":   "What is RBC's exposure to credit risk?",
                "session_id": "analyst-session-001"
            }
        }


class QueryResponse(BaseModel):
    session_id:        str
    question:          str
    question_redacted: str
    answer:            str
    confidence_level:  str
    confidence_score:  float
    sources:           list
    pii_found:         int
    audit_log_id:      int
    latency_ms:        float


class RiskRequest(BaseModel):
    session_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {"session_id": "analyst-session-001"}
        }


class IngestResponse(BaseModel):
    filename:           str
    pages_loaded:       int
    chunks_created:     int
    pii_entities_found: int
    audit_log_id:       int
    processing_ms:      float
    message:            str


# ── Health check ───────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Used by load balancers and monitoring systems to verify
    the API is running. Returns vectorstore status so
    ops teams can see if ingestion has been done.
    """
    vs = get_vs()
    return {
        "status":           "healthy",
        "version":          "1.0.0",
        "vectorstore_ready": vs is not None,
        "timestamp":        time.time()
    }


# ── Document ingestion endpoint ────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """
    Uploads and ingests a financial PDF document.

    Pipeline:
    1. Save uploaded file to data/ folder
    2. Run PII detection on extracted text
    3. Embed chunks into FAISS vectorstore
    4. Log ingestion event to audit database
    5. Return processing summary

    The PII redaction happens BEFORE embedding — no raw
    customer data ever enters the vectorstore or the LLM.
    """
    global _vectorstore

    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    start_time = time.time()

    # Save file to data/ folder
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / file.filename

    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    file_size_kb = round(len(contents) / 1024, 1)

    try:
        # Run ingestion pipeline
        from agents.ingest import (
            load_documents, chunk_documents,
            build_vectorstore, save_vectorstore
        )
        from pathlib import Path as P

        documents = load_documents(data_dir)
        chunks    = chunk_documents(documents)

        # PII redaction before embedding
        from agents.pii_filter import redact_document_chunks
        redacted_chunks, pii_summary = redact_document_chunks(chunks)

        total_pii = sum(pii_summary.values())

        # Build and save vectorstore
        vectorstore = build_vectorstore(redacted_chunks)
        save_vectorstore(vectorstore)
        _vectorstore = vectorstore

        processing_ms = round((time.time() - start_time) * 1000, 1)

        # Log to audit database
        audit_id = log_ingest(
            filename           = file.filename,
            file_size_kb       = file_size_kb,
            pages_loaded       = len(documents),
            chunks_created     = len(chunks),
            pii_entities_found = total_pii,
            pii_summary        = pii_summary,
            embedding_model    = "text-embedding-3-small",
            processing_ms      = processing_ms
        )

        return IngestResponse(
            filename           = file.filename,
            pages_loaded       = len(documents),
            chunks_created     = len(chunks),
            pii_entities_found = total_pii,
            audit_log_id       = audit_id,
            processing_ms      = processing_ms,
            message            = f"Successfully ingested {file.filename}. "
                                 f"{total_pii} PII entities detected and redacted."
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


# ── Query endpoint ─────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Answers a question about the ingested financial documents.

    Pipeline:
    1. Generate session ID if not provided
    2. Redact PII from the question itself
    3. Retrieve relevant chunks from FAISS
    4. Run risk extraction on retrieved context
    5. Generate compliance-aware answer
    6. Log everything to audit database
    7. Return structured response

    Every step is logged. Every answer is traceable.
    """
    start_time = time.time()
    session_id = request.session_id or str(uuid.uuid4())

    vs = get_vs()
    if vs is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. "
                   "Please upload a PDF via POST /ingest first."
        )

    # Step 1: redact PII from the question
    question_redacted, pii_entities = redact_pii(request.question)

    # Step 2: retrieve relevant chunks
    retrieval_result = run_retrieval_agent(request.question, vs)

    # Step 3: generate answer using LLM
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    answer_prompt = f"""You are a financial document analyst assistant.
Answer the question based ONLY on the provided document excerpts.
Always cite your sources with document name and page number.
If the answer is not in the documents, say so explicitly.
Never diagnose, recommend investments, or give financial advice.

QUESTION: {request.question}

DOCUMENT EXCERPTS:
{retrieval_result['context']}

Provide a clear, cited answer:"""

    response   = llm.invoke([HumanMessage(content=answer_prompt)])
    answer     = response.content.strip()
    latency_ms = round((time.time() - start_time) * 1000, 1)

    # Step 4: log to audit database
    audit_id = log_query(
        session_id        = session_id,
        question          = request.question,
        question_redacted = question_redacted,
        retrieved_chunks  = retrieval_result["chunks"],
        pii_entities      = pii_entities,
        llm_response      = answer,
        confidence_level  = retrieval_result["confidence_level"],
        confidence_score  = retrieval_result["confidence_score"],
        risk_indicators   = {},
        model_used        = "gpt-4o-mini",
        latency_ms        = latency_ms,
        documents_queried = retrieval_result["sources"]
    )

    return QueryResponse(
        session_id        = session_id,
        question          = request.question,
        question_redacted = question_redacted,
        answer            = answer,
        confidence_level  = retrieval_result["confidence_level"],
        confidence_score  = retrieval_result["confidence_score"],
        sources           = retrieval_result["sources"],
        pii_found         = len(pii_entities),
        audit_log_id      = audit_id,
        latency_ms        = latency_ms
    )


# ── Risk analysis endpoint ─────────────────────────────────
@app.post("/analyze/risk")
async def analyze_risk(request: RiskRequest):
    """
    Runs full risk extraction on ingested documents.
    Returns structured risk indicators across all 7
    Basel III / OSFI risk categories.
    """
    vs = get_vs()
    if vs is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet."
        )

    session_id = request.session_id or str(uuid.uuid4())

    retrieval_result = run_retrieval_agent(
        "risk factors credit market liquidity regulatory climate operational",
        vs
    )

    risk_result = run_risk_extractor(
        context       = retrieval_result["context"],
        document_name = retrieval_result["sources"][0]
                        if retrieval_result["sources"] else "unknown",
        chunks        = retrieval_result["chunks"]
    )

    if not risk_result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Risk extraction failed: {risk_result['error']}"
        )

    return {
        "session_id":  session_id,
        "risk_data":   risk_result["risk_data"],
        "latency_ms":  risk_result["latency_ms"],
        "sources":     retrieval_result["sources"]
    }


# ── Compliance summary endpoint ────────────────────────────
@app.post("/analyze/compliance")
async def analyze_compliance(request: RiskRequest):
    """
    Generates a full audit-ready compliance summary.
    Runs retrieval -> risk extraction -> compliance summary
    in sequence and returns the complete report.
    """
    vs = get_vs()
    if vs is None:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet."
        )

    session_id = request.session_id or str(uuid.uuid4())

    retrieval_result = run_retrieval_agent(
        "risk compliance capital liquidity regulatory audit",
        vs
    )

    risk_result = run_risk_extractor(
        context       = retrieval_result["context"],
        document_name = retrieval_result["sources"][0]
                        if retrieval_result["sources"] else "unknown",
        chunks        = retrieval_result["chunks"]
    )

    summary_result = run_compliance_summariser(
        context       = retrieval_result["context"],
        document_name = retrieval_result["sources"][0]
                        if retrieval_result["sources"] else "unknown",
        risk_data     = risk_result.get("risk_data", {}),
        chunks        = retrieval_result["chunks"]
    )

    if not summary_result["success"]:
        raise HTTPException(
            status_code=500,
            detail=f"Compliance summary failed: {summary_result['error']}"
        )

    return {
        "session_id":  session_id,
        "summary":     summary_result["summary"],
        "risk_data":   risk_result.get("risk_data", {}),
        "latency_ms":  summary_result["latency_ms"],
        "sources":     retrieval_result["sources"]
    }


# ── Audit log endpoints ────────────────────────────────────
@app.get("/audit/queries")
async def get_audit_queries(limit: int = 20):
    """
    Returns recent query audit log entries.
    Used by compliance officers to review AI activity.
    """
    return {
        "queries": get_recent_queries(limit=limit),
        "count":   limit
    }


@app.get("/audit/ingests")
async def get_audit_ingests(limit: int = 10):
    """
    Returns recent document ingestion log entries.
    Shows what documents were processed and when.
    """
    return {
        "ingests": get_recent_ingests(limit=limit),
        "count":   limit
    }


@app.get("/audit/stats")
async def get_stats():
    """
    Returns audit dashboard statistics.
    Total queries, average confidence, PII caught, latency.
    """
    return get_audit_stats()


# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True     # auto-reload on code changes during development
    )