import os
import json
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, Text, DateTime, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# ── Why SQLite + SQLAlchemy? ───────────────────────────────
# SQLite: zero config, file-based, perfect for demo scale.
# In production at RBC this would be PostgreSQL on RDS with
# encrypted storage, row-level security, and 7-year retention
# to meet FINTRAC record-keeping requirements.
#
# SQLAlchemy ORM: industry standard Python database layer.
# Using ORM instead of raw SQL means the same code works
# with SQLite locally and PostgreSQL in production — just
# change the connection string in .env.
#
# Why log every LLM call?
# OSFI E-23 guideline on model risk management requires that
# financial institutions maintain records of AI model inputs,
# outputs, and decisions. The audit log is not optional in
# a regulated environment — it is a compliance requirement.

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

DB_PATH = LOG_DIR / "finsight_audit.db"
engine  = create_engine(f"sqlite:///{DB_PATH}", echo=False)
Base    = declarative_base()
Session = sessionmaker(bind=engine)


# ── Database schema ────────────────────────────────────────

class QueryLog(Base):
    """
    Logs every question asked through the system.
    Includes the question, retrieved context, LLM answer,
    confidence score, PII entities found, and timing.
    """
    __tablename__ = "query_logs"

    id                = Column(Integer, primary_key=True, autoincrement=True)
    timestamp         = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    session_id        = Column(String(64), nullable=False)
    question          = Column(Text, nullable=False)
    question_redacted = Column(Text)           # question after PII redaction
    retrieved_chunks  = Column(JSON)           # list of source chunk metadata
    pii_entities      = Column(JSON)           # PII found in the question
    llm_response      = Column(Text)           # final answer from LLM
    confidence_level  = Column(String(10))     # HIGH / MEDIUM / LOW
    confidence_score  = Column(Float)          # numeric score 0.0-1.0
    risk_indicators   = Column(JSON)           # extracted risk flags
    model_used        = Column(String(50))     # e.g. gpt-4o-mini
    latency_ms        = Column(Float)          # end-to-end response time
    documents_queried = Column(JSON)           # which PDFs were searched


class IngestLog(Base):
    """
    Logs every document ingestion event.
    Tracks what was uploaded, how many chunks were created,
    and how many PII entities were found and redacted.
    """
    __tablename__ = "ingest_logs"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    timestamp        = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    filename         = Column(String(255), nullable=False)
    file_size_kb     = Column(Float)
    pages_loaded     = Column(Integer)
    chunks_created   = Column(Integer)
    pii_entities_found = Column(Integer)
    pii_summary      = Column(JSON)            # breakdown by entity type
    embedding_model  = Column(String(50))
    processing_ms    = Column(Float)


# Create tables on import
Base.metadata.create_all(engine)


# ── Logging functions ──────────────────────────────────────

def log_query(
    session_id:        str,
    question:          str,
    question_redacted: str,
    retrieved_chunks:  list,
    pii_entities:      list,
    llm_response:      str,
    confidence_level:  str,
    confidence_score:  float,
    risk_indicators:   dict,
    model_used:        str,
    latency_ms:        float,
    documents_queried: list
) -> int:
    """
    Logs a complete query interaction to the audit database.
    Returns the log entry ID for reference.

    Every field matters for compliance:
    - question + question_redacted: shows what was asked and
      that PII was handled before processing
    - retrieved_chunks: proves the answer came from documents,
      not hallucination
    - confidence_level: shows the system flagged uncertainty
    - latency_ms: SLA monitoring for production systems
    """
    session = Session()
    try:
        # Serialize chunk metadata for storage
        chunk_metadata = [
            {
                "source": c.get("source", "unknown"),
                "page":   c.get("page", 0),
                "score":  round(c.get("score", 0.0), 4)
            }
            for c in retrieved_chunks
        ]

        entry = QueryLog(
            session_id        = session_id,
            question          = question,
            question_redacted = question_redacted,
            retrieved_chunks  = chunk_metadata,
            pii_entities      = pii_entities,
            llm_response      = llm_response,
            confidence_level  = confidence_level,
            confidence_score  = confidence_score,
            risk_indicators   = risk_indicators,
            model_used        = model_used,
            latency_ms        = latency_ms,
            documents_queried = documents_queried
        )

        session.add(entry)
        session.commit()
        entry_id = entry.id
        return entry_id

    except Exception as e:
        session.rollback()
        print(f"Audit log error: {e}")
        return -1
    finally:
        session.close()


def log_ingest(
    filename:           str,
    file_size_kb:       float,
    pages_loaded:       int,
    chunks_created:     int,
    pii_entities_found: int,
    pii_summary:        dict,
    embedding_model:    str,
    processing_ms:      float
) -> int:
    """
    Logs a document ingestion event.
    Creates a permanent record of what was uploaded,
    when, and what PII was found during processing.
    """
    session = Session()
    try:
        entry = IngestLog(
            filename           = filename,
            file_size_kb       = file_size_kb,
            pages_loaded       = pages_loaded,
            chunks_created     = chunks_created,
            pii_entities_found = pii_entities_found,
            pii_summary        = pii_summary,
            embedding_model    = embedding_model,
            processing_ms      = processing_ms
        )

        session.add(entry)
        session.commit()
        entry_id = entry.id
        return entry_id

    except Exception as e:
        session.rollback()
        print(f"Ingest log error: {e}")
        return -1
    finally:
        session.close()


def get_recent_queries(limit: int = 20) -> list:
    """
    Returns the most recent query log entries.
    Used by the Streamlit audit log viewer in the UI.
    """
    session = Session()
    try:
        entries = (
            session.query(QueryLog)
            .order_by(QueryLog.timestamp.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id":               e.id,
                "timestamp":        e.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "question":         e.question[:100] + "..." if len(e.question) > 100 else e.question,
                "confidence_level": e.confidence_level,
                "confidence_score": e.confidence_score,
                "model_used":       e.model_used,
                "latency_ms":       e.latency_ms,
                "pii_found":        len(e.pii_entities) if e.pii_entities else 0,
                "sources":          [c["source"] for c in (e.retrieved_chunks or [])]
            }
            for e in entries
        ]
    finally:
        session.close()


def get_recent_ingests(limit: int = 10) -> list:
    """
    Returns the most recent ingestion log entries.
    Shows compliance officers what documents were processed
    and what PII was found in each.
    """
    session = Session()
    try:
        entries = (
            session.query(IngestLog)
            .order_by(IngestLog.timestamp.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id":                e.id,
                "timestamp":         e.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "filename":          e.filename,
                "file_size_kb":      e.file_size_kb,
                "pages_loaded":      e.pages_loaded,
                "chunks_created":    e.chunks_created,
                "pii_entities_found": e.pii_entities_found,
                "processing_ms":     e.processing_ms
            }
            for e in entries
        ]
    finally:
        session.close()


def get_audit_stats() -> dict:
    """
    Returns summary statistics for the audit dashboard.
    Shows total queries, average confidence, average latency,
    and total PII entities caught — useful for the UI dashboard.
    """
    session = Session()
    try:
        total_queries = session.query(QueryLog).count()
        total_ingests = session.query(IngestLog).count()

        queries = session.query(QueryLog).all()

        if queries:
            avg_confidence = sum(
                q.confidence_score for q in queries
                if q.confidence_score is not None
            ) / max(len(queries), 1)

            avg_latency = sum(
                q.latency_ms for q in queries
                if q.latency_ms is not None
            ) / max(len(queries), 1)

            total_pii = sum(
                len(q.pii_entities) for q in queries
                if q.pii_entities
            )

            confidence_breakdown = {
                "HIGH":   sum(1 for q in queries if q.confidence_level == "HIGH"),
                "MEDIUM": sum(1 for q in queries if q.confidence_level == "MEDIUM"),
                "LOW":    sum(1 for q in queries if q.confidence_level == "LOW"),
            }
        else:
            avg_confidence       = 0.0
            avg_latency          = 0.0
            total_pii            = 0
            confidence_breakdown = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}

        return {
            "total_queries":        total_queries,
            "total_ingests":        total_ingests,
            "avg_confidence":       round(avg_confidence, 3),
            "avg_latency_ms":       round(avg_latency, 1),
            "total_pii_caught":     total_pii,
            "confidence_breakdown": confidence_breakdown
        }
    finally:
        session.close()


if __name__ == "__main__":
    # Test the audit logger with a sample entry
    print("Testing audit logger...")

    test_id = log_query(
        session_id        = "test-session-001",
        question          = "What is RBC's exposure to credit risk?",
        question_redacted = "What is RBC's exposure to credit risk?",
        retrieved_chunks  = [
            {"source": "rbc_annual_report_2024.pdf", "page": 45, "score": 0.923},
            {"source": "rbc_annual_report_2024.pdf", "page": 46, "score": 0.891}
        ],
        pii_entities      = [],
        llm_response      = "According to the RBC Annual Report 2024 (page 45), credit risk exposure...",
        confidence_level  = "HIGH",
        confidence_score  = 0.92,
        risk_indicators   = {"credit_risk": "MEDIUM", "market_risk": "LOW"},
        model_used        = "gpt-4o-mini",
        latency_ms        = 1842.5,
        documents_queried = ["rbc_annual_report_2024.pdf"]
    )

    print(f"Query logged with ID: {test_id}")

    stats = get_audit_stats()
    print(f"\nAudit stats: {json.dumps(stats, indent=2)}")

    recent = get_recent_queries(limit=5)
    print(f"\nRecent queries: {len(recent)} entries")
    for q in recent:
        print(f"  [{q['timestamp']}] {q['question']} — {q['confidence_level']}")

    print(f"\nAudit database created at: {DB_PATH}")