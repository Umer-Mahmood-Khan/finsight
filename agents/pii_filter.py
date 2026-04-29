import os
from dotenv import load_dotenv
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

load_dotenv()

# ── Why Presidio? ──────────────────────────────────────────
# Microsoft Presidio is the industry standard for PII detection
# in enterprise and financial applications. It is used by banks,
# insurance companies, and healthcare providers. It combines:
# 1. Named Entity Recognition (spaCy) for context-aware detection
# 2. Rule-based recognizers for structured patterns (SIN, account numbers)
# 3. Custom recognizers for domain-specific entities
#
# In a regulated environment like RBC, sending raw customer data
# to an external LLM API violates OSFI guidelines. The PII filter
# runs BEFORE any text touches OpenAI — this is the compliance layer.

# ── Financial-specific PII patterns ───────────────────────
# These are patterns Presidio does not detect by default
# but are common in Canadian financial documents

FINANCIAL_ENTITY_TYPES = [
    "PERSON",           # customer names
    "EMAIL_ADDRESS",    # contact information
    "PHONE_NUMBER",     # contact information
    "LOCATION",         # addresses
    "DATE_TIME",        # birth dates
    "CA_SIN",           # Canadian Social Insurance Number
    "CREDIT_CARD",      # card numbers
    "IBAN_CODE",        # bank account numbers
    "US_SSN",           # US Social Security Number
    "US_BANK_NUMBER",   # US account numbers
    "IP_ADDRESS",       # system identifiers
]


def build_analyzer() -> AnalyzerEngine:
    """
    Builds the Presidio analyzer with spaCy NLP backend.
    en_core_web_lg gives better NER accuracy for financial text
    than the default small model — worth the 400MB download.
    """
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"}
        ],
    }

    provider     = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine   = provider.create_engine()
    analyzer     = AnalyzerEngine(nlp_engine=nlp_engine)
    return analyzer


def build_anonymizer() -> AnonymizerEngine:
    """
    Builds the Presidio anonymizer.
    Replaces detected PII with typed placeholders like <PERSON>
    so the LLM understands what was removed without seeing the value.
    """
    return AnonymizerEngine()


# Initialise once at module level — expensive to rebuild on every call
_analyzer   = None
_anonymizer = None


def get_analyzer() -> AnalyzerEngine:
    global _analyzer
    if _analyzer is None:
        print("Initialising PII analyzer (first call only)...")
        _analyzer = build_analyzer()
    return _analyzer


def get_anonymizer() -> AnonymizerEngine:
    global _anonymizer
    if _anonymizer is None:
        _anonymizer = build_anonymizer()
    return _anonymizer


def detect_pii(text: str) -> list:
    """
    Detects PII entities in text without redacting them.
    Returns a list of detected entities with type, score, and position.

    Used to generate the PII detection report shown in the audit log
    so compliance officers can see what was found and redacted.
    """
    analyzer = get_analyzer()
    results  = analyzer.analyze(
        text=text,
        entities=FINANCIAL_ENTITY_TYPES,
        language="en"
    )
    return results


def redact_pii(text: str) -> tuple[str, list]:
    """
    Main function — detects and redacts PII from text.

    Returns:
        redacted_text: text with PII replaced by <TYPE> placeholders
        pii_report:    list of what was found and redacted

    Why placeholders instead of removal?
    Removing PII entirely can break sentence structure and confuse
    the LLM. Replacing with <PERSON> or <ACCOUNT_NUMBER> preserves
    grammar and tells the LLM that something was intentionally removed.

    Example:
        Input:  "John Smith's account 4532-1234-5678 shows a balance..."
        Output: "<PERSON>'s account <CREDIT_CARD> shows a balance..."
    """
    analyzer   = get_analyzer()
    anonymizer = get_anonymizer()

    # Step 1: detect
    analysis_results = analyzer.analyze(
        text=text,
        entities=FINANCIAL_ENTITY_TYPES,
        language="en"
    )

    if not analysis_results:
        return text, []

    # Step 2: redact — replace with <ENTITY_TYPE> placeholders
    operator_config = {
        entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
        for entity in FINANCIAL_ENTITY_TYPES
    }

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=analysis_results,
        operators=operator_config
    )

    # Step 3: build report for audit log
    pii_report = [
        {
            "entity_type":  result.entity_type,
            "confidence":   round(result.score, 3),
            "start":        result.start,
            "end":          result.end,
            "text_preview": text[result.start:result.end][:20] + "..."
            if len(text[result.start:result.end]) > 20
            else text[result.start:result.end]
        }
        for result in analysis_results
    ]

    return anonymized.text, pii_report


def redact_document_chunks(chunks: list) -> tuple[list, dict]:
    """
    Runs PII redaction over all document chunks before embedding.

    Called by the ingestion pipeline — ensures no raw PII ever
    enters the vector store or gets sent to the OpenAI API.

    Returns:
        redacted_chunks: chunks with PII replaced
        summary:         total PII entities found per document
    """
    redacted_chunks = []
    summary         = {}

    for i, chunk in enumerate(chunks):
        original_text         = chunk.page_content
        redacted_text, report = redact_pii(original_text)

        # Replace chunk content with redacted version
        chunk.page_content = redacted_text

        # Track PII found per source document
        source = chunk.metadata.get("source", "unknown")
        if source not in summary:
            summary[source] = 0
        summary[source] += len(report)

        redacted_chunks.append(chunk)

        if (i + 1) % 50 == 0:
            print(f"  Redacted {i + 1}/{len(chunks)} chunks...")

    return redacted_chunks, summary


if __name__ == "__main__":
    # Test the PII filter with sample financial text
    test_text = """
    Dear Mr. John Smith,
    
    Your account number 4532-1234-5678-9012 has been reviewed.
    Our analyst Sarah Johnson (sarah.johnson@rbc.com) will contact
    you at 416-555-0123 regarding your SIN 123-456-789.
    
    The loan application submitted on March 15, 2024 for your
    property at 100 King Street West, Toronto is under review.
    Current outstanding balance reflects exposure to USD/CAD
    fluctuation risk as discussed in the Q3 earnings call.
    """

    print("Testing PII detection on sample financial text...")
    print("=" * 60)
    print("ORIGINAL TEXT:")
    print(test_text)

    redacted, report = redact_pii(test_text)

    print("\nREDACTED TEXT:")
    print(redacted)

    print(f"\nPII ENTITIES FOUND: {len(report)}")
    for entity in report:
        print(f"  {entity['entity_type']:25} "
              f"confidence: {entity['confidence']} "
              f"preview: {entity['text_preview']}")