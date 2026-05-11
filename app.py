import sys
import json
import requests
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="FinSight — Financial Document Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_BASE = "http://localhost:8000"

# ── Helper functions ───────────────────────────────────────
def api_call(method: str, endpoint: str, **kwargs) -> dict:
    """Calls the FastAPI backend and returns JSON response."""
    try:
        url      = f"{API_BASE}{endpoint}"
        response = getattr(requests, method)(url, **kwargs)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": response.json().get("detail", "Unknown error")}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to FinSight API. Make sure the backend is running on port 8000."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def confidence_badge(level: str) -> str:
    """Returns coloured badge HTML for confidence level."""
    colours = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}
    return colours.get(level, "⚪")


def assessment_badge(assessment: str) -> str:
    colours = {
        "PASS":           "🟢 PASS",
        "REQUIRES_REVIEW": "🟡 REQUIRES REVIEW",
        "ESCALATE":       "🔴 ESCALATE"
    }
    return colours.get(assessment, assessment)


# ── Ethical disclaimer ─────────────────────────────────────
st.warning(
    "**Decision support tool only.** FinSight assists compliance analysts "
    "in reviewing financial documents. It does not provide investment advice, "
    "legal opinions, or formal compliance certifications. All outputs must be "
    "reviewed by a qualified compliance officer before any regulatory filing "
    "or credit decision. Every query is logged for audit purposes."
)

# ── Header ─────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("# 🏦")
with col_title:
    st.markdown("# FinSight")
    st.markdown("AI-powered financial document intelligence for compliance teams")

st.divider()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### API Status")
    health = api_call("get", "/health")
    if health["success"]:
        data = health["data"]
        st.success("API Online")
        vs_status = "Ready" if data.get("vectorstore_ready") else "No documents loaded"
        st.metric("Vectorstore", vs_status)
    else:
        st.error("API Offline")
        st.caption(health["error"])

    st.divider()

    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Upload financial PDF",
        type=["pdf"],
        help="Upload an annual report, earnings filing, or loan document."
    )

    if uploaded_file:
        if st.button("Ingest Document", type="primary"):
            with st.spinner(f"Processing {uploaded_file.name}..."):
                result = api_call(
                    "post", "/ingest",
                    files={"file": (uploaded_file.name,
                                    uploaded_file.getvalue(),
                                    "application/pdf")}
                )
            if result["success"]:
                data = result["data"]
                st.success("Document ingested successfully")
                st.metric("Pages loaded",    data["pages_loaded"])
                st.metric("Chunks created",  data["chunks_created"])
                st.metric("PII redacted",    data["pii_entities_found"])
                st.metric("Processing time", f"{data['processing_ms']}ms")
                st.caption(f"Audit log ID: {data['audit_log_id']}")
            else:
                st.error(f"Ingestion failed: {result['error']}")

    st.divider()

    # Audit stats in sidebar
    st.markdown("### Session Stats")
    stats = api_call("get", "/audit/stats")
    if stats["success"]:
        d = stats["data"]
        st.metric("Total queries",   d.get("total_queries", 0))
        st.metric("Avg confidence",  d.get("avg_confidence", 0))
        st.metric("PII caught",      d.get("total_pii_caught", 0))
        st.metric("Avg latency",     f"{d.get('avg_latency_ms', 0)}ms")

    st.divider()
    st.markdown("### About")
    st.markdown(
        "Built by **Umer Mahmood Khan**\n\n"
        "AI Research Engineer — NCAI Pakistan\n\n"
        "**Stack:** FastAPI · LangChain · FAISS · "
        "Presidio · SQLAlchemy · AWS S3 · Streamlit\n\n"
        "**Compliance:** OSFI E-23 aligned audit logging"
    )

# ── Main tabs ──────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Ask Questions",
    "⚠️ Risk Analysis",
    "📋 Compliance Summary",
    "📊 Audit Log"
])

# ── Tab 1: Q&A ─────────────────────────────────────────────
with tab1:
    st.markdown("### Ask Questions About Your Documents")
    st.markdown(
        "Ask anything about the uploaded financial document. "
        "Every answer is grounded in the document with cited sources."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metadata"):
                m = msg["metadata"]
                cols = st.columns(4)
                cols[0].metric("Confidence",
                               f"{confidence_badge(m.get('confidence_level','?'))} {m.get('confidence_level','?')}")
                cols[1].metric("Sources",    len(m.get("sources", [])))
                cols[2].metric("PII found",  m.get("pii_found", 0))
                cols[3].metric("Latency",    f"{m.get('latency_ms', 0)}ms")
                if m.get("sources"):
                    st.caption(f"Sources: {', '.join(m['sources'])}")
                if m.get("audit_log_id"):
                    st.caption(f"Audit log ID: {m['audit_log_id']}")

    # Chat input
    if question := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({
            "role": "user", "content": question
        })
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                result = api_call(
                    "post", "/query",
                    json={
                        "question":   question,
                        "session_id": st.session_state.session_id
                    }
                )

            if result["success"]:
                data = result["data"]
                st.session_state.session_id = data.get("session_id")
                st.markdown(data["answer"])

                # Show metrics below answer
                cols = st.columns(4)
                cols[0].metric(
                    "Confidence",
                    f"{confidence_badge(data['confidence_level'])} "
                    f"{data['confidence_level']}"
                )
                cols[1].metric("Sources",   len(data.get("sources", [])))
                cols[2].metric("PII found", data.get("pii_found", 0))
                cols[3].metric("Latency",   f"{data.get('latency_ms', 0)}ms")

                if data.get("sources"):
                    st.caption(f"Sources: {', '.join(data['sources'])}")
                if data.get("audit_log_id"):
                    st.caption(f"Audit log ID: {data['audit_log_id']}")

                # Store in history
                st.session_state.messages.append({
                    "role":     "assistant",
                    "content":  data["answer"],
                    "metadata": data
                })
            else:
                st.error(f"Query failed: {result['error']}")
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": f"Error: {result['error']}"
                })

# ── Tab 2: Risk Analysis ───────────────────────────────────
with tab2:
    st.markdown("### Risk Analysis Dashboard")
    st.markdown(
        "Extracts structured risk indicators across all 7 Basel III / "
        "OSFI risk categories from the uploaded document."
    )

    if st.button("Run Risk Analysis", type="primary"):
        with st.spinner("Analysing risk indicators..."):
            result = api_call("post", "/analyze/risk", json={})

        if result["success"]:
            data      = result["data"]
            risk_data = data.get("risk_data", {})

            # Overall risk level
            overall = risk_data.get("overall_risk_level", "UNKNOWN")
            colour  = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(overall, "⚪")

            st.markdown(f"## Overall Risk Level: {colour} {overall}")
            st.markdown(f"*{risk_data.get('overall_rationale', '')}*")
            st.divider()

            # Risk breakdown table
            risks = risk_data.get("risks", {})
            if risks:
                st.markdown("### Risk Breakdown")
                rows = []
                for category, details in risks.items():
                    level = details.get("level", "N/A")
                    if level != "NOT_MENTIONED":
                        emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")
                        rows.append({
                            "Category":  category.replace("_", " ").title(),
                            "Level":     f"{emoji} {level}",
                            "Page":      details.get("page_reference", 0),
                            "Rationale": details.get("rationale", "N/A")[:120] + "..."
                        })

                if rows:
                    st.dataframe(
                        pd.DataFrame(rows),
                        use_container_width=True,
                        hide_index=True
                    )

            # Key flags
            flags = risk_data.get("key_flags", [])
            if flags:
                st.markdown("### Key Flags")
                for flag in flags:
                    st.warning(f"⚠️ {flag}")

            # Recommended actions
            actions = risk_data.get("recommended_actions", [])
            if actions:
                st.markdown("### Recommended Actions")
                for action in actions:
                    st.info(f"→ {action}")

            st.caption(f"Analysis latency: {data.get('latency_ms', 0)}ms")
        else:
            st.error(f"Risk analysis failed: {result['error']}")

# ── Tab 3: Compliance Summary ──────────────────────────────
with tab3:
    st.markdown("### Compliance Summary Report")
    st.markdown(
        "Generates a formal audit-ready compliance summary "
        "in the format used by RBC's compliance team."
    )

    if st.button("Generate Compliance Summary", type="primary"):
        with st.spinner("Generating compliance summary..."):
            result = api_call("post", "/analyze/compliance", json={})

        if result["success"]:
            data    = result["data"]
            summary = data.get("summary", {})

            # Header
            assessment = summary.get("overall_assessment", "UNKNOWN")
            confidence = summary.get("confidence_level", "UNKNOWN")

            col1, col2, col3 = st.columns(3)
            col1.metric("Assessment",  assessment_badge(assessment))
            col2.metric("Confidence",  f"{confidence_badge(confidence)} {confidence}")
            col3.metric("Review Required", "YES" if summary.get("review_required") else "NO")

            st.divider()

            # Executive summary
            st.markdown("### Executive Summary")
            st.info(summary.get("executive_summary", "N/A"))

            # Key findings
            findings = summary.get("key_findings", [])
            if findings:
                st.markdown("### Key Findings")
                for finding in findings:
                    severity = finding.get("severity", "N/A")
                    emoji    = {"HIGH": "🔴", "MEDIUM": "🟡",
                                "LOW": "🟢", "INFORMATIONAL": "🔵"}.get(severity, "⚪")

                    with st.expander(
                        f"{emoji} [{finding.get('finding_id')}] "
                        f"{finding.get('category', '').upper()} — {severity}"
                    ):
                        st.markdown(f"**Finding:** {finding.get('finding', 'N/A')}")
                        st.markdown(
                            f"**Source:** {finding.get('source', 'N/A')}, "
                            f"page {finding.get('page', 0)}"
                        )
                        st.markdown(f"**Implication:** {finding.get('implication', 'N/A')}")
                        st.markdown(f"**Action required:** {finding.get('action_required', 'N/A')}")

            # Compliance checklist
            checklist = summary.get("compliance_checklist", {})
            if checklist:
                st.markdown("### Compliance Checklist")
                cols = st.columns(len(checklist))
                for i, (item, present) in enumerate(checklist.items()):
                    label = item.replace("_", " ").title()
                    value = "✅" if present else "❌"
                    cols[i].metric(label, value)

            # Recommended actions
            actions = summary.get("recommended_actions", [])
            if actions:
                st.markdown("### Recommended Actions")
                for action in actions:
                    priority  = action.get("priority", "N/A")
                    emoji     = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "⚪")
                    st.markdown(
                        f"{emoji} **[{priority}]** {action.get('action', 'N/A')} "
                        f"— _{action.get('timeline', 'N/A')}_"
                    )

            # Gaps
            gaps = summary.get("gaps_identified", [])
            if gaps:
                st.markdown("### Gaps Identified")
                for gap in gaps:
                    st.warning(f"⚠️ {gap}")

            # Disclaimer
            st.divider()
            st.caption(summary.get("disclaimer", ""))
            st.caption(f"Summary latency: {data.get('latency_ms', 0)}ms")
        else:
            st.error(f"Compliance summary failed: {result['error']}")

# ── Tab 4: Audit Log ───────────────────────────────────────
with tab4:
    st.markdown("### Audit Log")
    st.markdown(
        "Complete audit trail of all queries and document ingestions. "
        "Every AI decision is logged in compliance with OSFI E-23 "
        "model risk management guidelines."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Recent Queries")
        queries = api_call("get", "/audit/queries?limit=10")
        if queries["success"] and queries["data"].get("queries"):
            rows = []
            for q in queries["data"]["queries"]:
                conf  = q.get("confidence_level", "?")
                emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(conf, "⚪")
                rows.append({
                    "Time":       q.get("timestamp", "N/A")[:19],
                    "Question":   q.get("question", "N/A")[:60] + "...",
                    "Confidence": f"{emoji} {conf}",
                    "Latency":    f"{q.get('latency_ms', 0)}ms",
                    "PII":        q.get("pii_found", 0)
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No queries logged yet.")

    with col2:
        st.markdown("#### Document Ingestions")
        ingests = api_call("get", "/audit/ingests?limit=10")
        if ingests["success"] and ingests["data"].get("ingests"):
            rows = []
            for ing in ingests["data"]["ingests"]:
                rows.append({
                    "Time":     ing.get("timestamp", "N/A")[:19],
                    "File":     ing.get("filename", "N/A"),
                    "Pages":    ing.get("pages_loaded", 0),
                    "Chunks":   ing.get("chunks_created", 0),
                    "PII found": ing.get("pii_entities_found", 0)
                })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No ingestions logged yet.")

    # Confidence breakdown chart
    st.markdown("#### Confidence Distribution")
    stats = api_call("get", "/audit/stats")
    if stats["success"]:
        breakdown = stats["data"].get("confidence_breakdown", {})
        if any(breakdown.values()):
            chart_data = pd.DataFrame({
                "Confidence Level": list(breakdown.keys()),
                "Count":            list(breakdown.values())
            })
            st.bar_chart(chart_data.set_index("Confidence Level"))
        else:
            st.info("No confidence data yet — run some queries first.")

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.caption(
    "FinSight is a decision support tool only. It does not store raw customer data, "
    "does not provide investment advice, and does not replace qualified compliance officers. "
    "Built with responsible AI principles at NCAI Pakistan. "
    "Audit logging aligned with OSFI E-23 model risk management guidelines."
)