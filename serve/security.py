"""
Security layer: prompt injection detection, PII redaction, input sanitization.

Two injection attack surfaces addressed:
  1. Direct  — malicious user query ("ignore previous instructions...")
  2. Indirect — injected content embedded in a document ("When summarising this, always say X")

PII redaction runs at ingestion time: documents are scanned before entering the corpus.
Secrets (API keys, tokens) are also flagged since enterprise docs often leak them.
"""

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Prompt-injection patterns (user query surface)
# ---------------------------------------------------------------------------

_QUERY_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # Classic instruction override
    (r"ignore\s+(all\s+)?(previous|prior|above|system)\s+(instructions?|prompts?|context|rules?)", "instruction_override"),
    # Role / persona hijacking
    (r"you\s+are\s+now\s+(a|an|the)\b", "role_hijack"),
    (r"(act|pretend|roleplay|simulate|behave)\s+as\s+(if\s+you\b|(you\s+(are|were)|a|an)\b)", "persona_override"),
    # Role delimiters smuggled into query
    (r"(system|assistant|user)\s*:\s*\S", "role_delimiter"),
    # XML/tag-based injection
    (r"<\s*(system|prompt|instruction|context)\b", "xml_injection"),
    # Memory wipe attempts
    (r"forget\s+(everything|all|what|your)\s", "memory_wipe"),
    # System prompt extraction
    (r"(reveal|show|print|output|display|repeat|tell\s+me)\s+(your\s+)?(system\s+prompt|instructions?|initial\s+prompt|base\s+prompt)", "prompt_extraction"),
    # Classic jailbreaks
    (r"\bDAN\b|do\s+anything\s+now|jailbreak", "jailbreak"),
    # Newline / CRLF delimiter injection
    (r"(\\n|\\r|%0a|%0d)\s*(system|user|assistant)\s*:", "delimiter_injection"),
    # Token manipulation
    (r"(</s>|<\|endoftext\|>|<\|im_end\|>)\s*(system|user)", "token_injection"),
]

# ---------------------------------------------------------------------------
# Indirect injection patterns (document content surface — more lenient)
# ---------------------------------------------------------------------------

_DOC_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?previous\s+instructions?", "indirect_instruction_override"),
    (r"(system|assistant)\s*:\s*you\s+(are|must|should|will)\b", "indirect_role_injection"),
    (r"when\s+(summari[sz]ing|answering|responding\s+to|processing)\s+this\s+document\s*,?\s*(always|never|instead|do\s+not)", "behavioral_override"),
    (r"<\s*(system|instruction)\b[^>]*>", "embedded_xml_instruction"),
    (r"translate\s+(the\s+)?(above|following|this)\s+(to|into)\s+\w+\s*:\s*ignore", "translate_override"),
]

# ---------------------------------------------------------------------------
# PII & secret patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[str, str]] = [
    # Contact
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",                          "EMAIL"),
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",                        "PHONE"),
    # Identity
    (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",                                                 "SSN"),
    (r"\b[A-Z]{2}\d{6}[A-Z]\b",                                                           "UK_NIN"),
    (r"\b[A-Z]{1,2}\d{6,9}\b",                                                            "PASSPORT"),
    # Payment
    (r"\b4\d{3}(?:[-\s]?\d{4}){3}\b",                                                    "CREDIT_CARD_VISA"),
    (r"\b5[1-5]\d{2}(?:[-\s]?\d{4}){3}\b",                                               "CREDIT_CARD_MC"),
    (r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b",                                            "CREDIT_CARD_AMEX"),
    # Network / infra
    (r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",   "IP_ADDRESS"),
    # Secrets / API keys — use lookahead/lookbehind instead of \b for alphanumeric boundaries
    (r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])",                                       "AWS_ACCESS_KEY"),
    (r"(?<![A-Za-z0-9])(sk|pk)[-_](live|test)[-_][A-Za-z0-9]{20,}(?![A-Za-z0-9])",      "STRIPE_KEY"),
    (r"ghp_[A-Za-z0-9]{36}",                                                              "GITHUB_TOKEN"),
    (r"\bey[A-Za-z0-9\-_]{10,}\.[A-Za-z0-9\-_]{10,}\.[A-Za-z0-9\-_]{10,}",             "JWT_TOKEN"),
    (r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{32,64}(?![A-Fa-f0-9])(?=\s*(?:api[_\s]?key|token|secret|password))", "HEX_SECRET"),
]

MAX_QUESTION_LEN = 512
MAX_CONTEXT_LEN  = 100_000


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SecurityResult:
    safe: bool
    threat_type: str = ""
    matched: str = ""


@dataclass
class PIIResult:
    redacted_text: str
    found: list[dict] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return bool(self.found)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def check_injection(text: str) -> SecurityResult:
    """Check a user query for prompt-injection attempts."""
    t = text.strip()
    for pattern, threat in _QUERY_INJECTION_PATTERNS:
        m = re.search(pattern, t, re.IGNORECASE)
        if m:
            return SecurityResult(safe=False, threat_type=threat, matched=m.group(0)[:80])
    return SecurityResult(safe=True)


def check_document_injection(text: str) -> SecurityResult:
    """
    Check document content for indirect / embedded injection instructions.
    More lenient than the query check — documents can legitimately reference
    instructions, error handling, etc.
    """
    for pattern, threat in _DOC_INJECTION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return SecurityResult(safe=False, threat_type=threat, matched=m.group(0)[:80])
    return SecurityResult(safe=True)


def redact_pii(text: str) -> PIIResult:
    """Detect and redact PII / secrets from text, returning a PIIResult."""
    found: list[dict] = []
    redacted = text

    for pattern, pii_type in _PII_PATTERNS:
        hits = list(re.finditer(pattern, redacted))
        if not hits:
            continue
        for h in hits:
            raw = h.group(0)
            # keep first 3 chars as hint, mask the rest
            hint = raw[:3] + "***" if len(raw) > 3 else "***"
            found.append({"type": pii_type, "hint": hint, "length": len(raw)})
        redacted = re.sub(pattern, f"[{pii_type} REDACTED]", redacted)

    return PIIResult(redacted_text=redacted, found=found)


def sanitize_input(text: str, max_len: int = MAX_QUESTION_LEN) -> str:
    """Strip null bytes and control characters, then truncate."""
    # remove control chars except \t, \n, \r
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text[:max_len]


def grounding_score(answer: str, passages: list[str]) -> float:
    """
    Estimate how well the answer is grounded in the retrieved passages.
    Simple token-overlap heuristic; 0.0–1.0, higher = better grounded.
    """
    if not answer or not passages:
        return 0.0
    combined = " ".join(passages).lower()
    tokens = [w for w in answer.lower().split() if len(w) > 3]
    if not tokens:
        return 1.0
    hits = sum(1 for w in tokens if w in combined)
    return round(hits / len(tokens), 3)
