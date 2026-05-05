import os
import json
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

LLM_MODEL = "gpt-4o-mini"

# ── Why a dedicated risk extraction agent? ─────────────────
# A general Q&A agent retrieves and answers. A risk extraction
# agent does something more specific — it reads financial text
# and outputs STRUCTURED risk indicators in a consistent schema.
#
# This matters for RBC because:
# 1. Risk teams need structured data, not prose summaries
# 2. Structured output can feed downstream systems (dashboards,
#    alerts, risk databases) without further parsing
# 3. Consistent schema means outputs are comparable across
#    multiple documents over time
#
# In production this output would feed into RBC's internal
# risk management platform via API — not just display in a UI.

RISK_CATEGORIES = [
    "credit_risk",      # counterparty default, loan losses
    "market_risk",      # interest rate, FX, equity price risk
    "liquidity_risk",   # ability to meet short-term obligations
    "operational_risk", # systems, people, process failures
    "regulatory_risk",  # compliance, legal, regulatory changes
    "climate_risk",     # physical and transition climate risks
    "reputational_risk" # brand, ESG, public perception risks
]

SYSTEM_PROMPT = """You are a financial risk analyst assistant at a major 
Canadian bank. Your job is to extract structured risk indicators from 
financial documents.

STRICT RULES:
- Only extract what is explicitly stated in the provided text
- Never infer or fabricate risk levels not supported by the text
- If a risk category is not mentioned, set it to NOT_MENTIONED
- Always cite the specific text that supports your assessment
- Risk levels must be: HIGH, MEDIUM, LOW, or NOT_MENTIONED
- Return ONLY valid JSON — no explanation, no markdown

This output will be used by compliance officers and risk managers.
Accuracy and traceability are more important than completeness."""

EXTRACTION_PROMPT = """Analyze the following financial document excerpts 
and extract structured risk indicators.

For each risk category provide:
- level: HIGH / MEDIUM / LOW / NOT_MENTIONED  
- rationale: one sentence explaining why, quoting the document
- page_reference: page number where evidence was found

Return this exact JSON structure:
{{
  "document_name": "{document_name}",
  "analysis_timestamp": "{timestamp}",
  "overall_risk_level": "HIGH/MEDIUM/LOW",
  "overall_rationale": "one sentence summary of overall risk profile",
  "risks": {{
    "credit_risk": {{
      "level": "HIGH/MEDIUM/LOW/NOT_MENTIONED",
      "rationale": "quote or paraphrase from document",
      "page_reference": 0
    }},
    "market_risk": {{
      "level": "HIGH/MEDIUM/LOW/NOT_MENTIONED",
      "rationale": "quote or paraphrase from document",
      "page_reference": 0
    }},
    "liquidity_risk": {{
      "level": "HIGH/MEDIUM/LOW/NOT_MENTIONED",
      "rationale": "quote or paraphrase from document",
      "page_reference": 0
    }},
    "operational_risk": {{
      "level": "HIGH/MEDIUM/LOW/NOT_MENTIONED",
      "rationale": "quote or paraphrase from document",
      "page_reference": 0
    }},
    "regulatory_risk": {{
      "level": "HIGH/MEDIUM/LOW/NOT_MENTIONED",
      "rationale": "quote or paraphrase from document",
      "page_reference": 0
    }},
    "climate_risk": {{
      "level": "HIGH/MEDIUM/LOW/NOT_MENTIONED",
      "rationale": "quote or paraphrase from document",
      "page_reference": 0
    }},
    "reputational_risk": {{
      "level": "HIGH/MEDIUM/LOW/NOT_MENTIONED",
      "rationale": "quote or paraphrase from document",
      "page_reference": 0
    }}
  }},
  "key_flags": [
    "specific concern 1",
    "specific concern 2"
  ],
  "recommended_actions": [
    "action 1",
    "action 2"
  ]
}}

DOCUMENT EXCERPTS:
{context}
"""


def run_risk_extractor(
    context: str,
    document_name: str,
    chunks: list
) -> dict:
    """
    Main risk extraction function.

    Takes retrieved chunks from the retrieval agent and
    extracts structured risk indicators using GPT-4o-mini
    with temperature=0 for consistent, deterministic output.

    Why temperature=0?
    Risk assessment must be reproducible. The same document
    should produce the same risk flags every time. Temperature=0
    removes randomness from the LLM response.

    Returns structured dict that feeds into:
    1. The compliance summary agent (Day 7)
    2. The audit log
    3. The Streamlit risk dashboard (Day 9)
    """
    start_time = time.time()

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    prompt = EXTRACTION_PROMPT.format(
        document_name=document_name,
        timestamp=timestamp,
        context=context
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    try:
        response     = llm.invoke(messages)
        raw_response = response.content.strip()

        # Clean markdown if LLM wraps in code fences
        if raw_response.startswith("```"):
            raw_response = raw_response.split("```")[1]
            if raw_response.startswith("json"):
                raw_response = raw_response[4:]

        risk_data = json.loads(raw_response)

        latency_ms = round((time.time() - start_time) * 1000, 1)
        risk_data["latency_ms"] = latency_ms

        return {
            "success":     True,
            "risk_data":   risk_data,
            "latency_ms":  latency_ms,
            "error":       None
        }

    except json.JSONDecodeError as e:
        return {
            "success":    False,
            "risk_data":  None,
            "latency_ms": round((time.time() - start_time) * 1000, 1),
            "error":      f"JSON parse error: {e}"
        }
    except Exception as e:
        return {
            "success":    False,
            "risk_data":  None,
            "latency_ms": round((time.time() - start_time) * 1000, 1),
            "error":      str(e)
        }


def format_risk_summary(risk_data: dict) -> str:
    """
    Formats the structured risk data into a readable summary
    for display in the Streamlit UI and audit log.
    """
    if not risk_data:
        return "Risk extraction failed."

    lines = [
        f"OVERALL RISK LEVEL: {risk_data.get('overall_risk_level', 'UNKNOWN')}",
        f"RATIONALE: {risk_data.get('overall_rationale', 'N/A')}",
        "",
        "RISK BREAKDOWN:"
    ]

    risks = risk_data.get("risks", {})
    for category, details in risks.items():
        level    = details.get("level", "N/A")
        rationale = details.get("rationale", "N/A")
        page     = details.get("page_reference", 0)
        if level != "NOT_MENTIONED":
            lines.append(
                f"  {category.upper():20} {level:8} "
                f"(page {page}) — {rationale[:80]}..."
            )

    flags = risk_data.get("key_flags", [])
    if flags:
        lines.append("\nKEY FLAGS:")
        for flag in flags:
            lines.append(f"  - {flag}")

    actions = risk_data.get("recommended_actions", [])
    if actions:
        lines.append("\nRECOMMENDED ACTIONS:")
        for action in actions:
            lines.append(f"  - {action}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Test using the retrieval agent output
    from agents.retrieval import get_vectorstore, run_retrieval_agent

    print("Testing risk extractor...")
    print("Loading vectorstore...")

    vectorstore = get_vectorstore()

    # Use a broad query to get diverse risk-related chunks
    query  = "risk factors credit market liquidity regulatory climate"
    result = run_retrieval_agent(query, vectorstore)

    print(f"Retrieved {len(result['chunks'])} chunks")
    print("Running risk extraction...\n")

    risk_result = run_risk_extractor(
        context       = result["context"],
        document_name = result["sources"][0] if result["sources"] else "unknown",
        chunks        = result["chunks"]
    )

    if risk_result["success"]:
        print("EXTRACTION SUCCESSFUL")
        print(f"Latency: {risk_result['latency_ms']}ms\n")
        print(format_risk_summary(risk_result["risk_data"]))
        print("\nFull JSON output:")
        print(json.dumps(risk_result["risk_data"], indent=2))
    else:
        print(f"EXTRACTION FAILED: {risk_result['error']}")