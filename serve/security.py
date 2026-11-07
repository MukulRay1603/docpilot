import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field

_OVERRIDE_VERB = r"(ignore|disregard|forget|skip|bypass|override|drop)"
_PRIOR = r"(previous|prior|above|earlier|preceding|original|system|initial)"
_RULES = r"(instructions?|prompts?|context|rules?|directions?|guidelines?|commands?)"

# Patterns run on the raw text and on a normalized copy (see _normalize_variants).
# They aim at phrasing that addresses the model ("your instructions", "you are
# now", line-initial "system:") rather than single keywords, because keyword
# patterns blocked ordinary technician questions ("forget all paired devices",
# "act as an administrator", "show instructions on the HUD").
_QUERY_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (rf"\b{_OVERRIDE_VERB}\s+(all\s+|any\s+|the\s+|your\s+|my\s+|of\s+)*{_PRIOR}\s+{_RULES}", "instruction_override"),
    (r"\b(ignore|disregard)\s+(everything|anything|all)\s+(you\s+were\s+told|(that\s+)?(was\s+)?(said|written)?\s*(above|before))", "instruction_override"),
    (r"\b(ignore|disregard)\s+(the\s+|these\s+|those\s+|all\s+)?(passages?|provided\s+context|sources|documents?)\b", "instruction_override"),
    (r"\b(override|bypass|disable)\s+(your|all|any|the\s+safety)\s+(rules?|instructions?|filters?|restrictions?|guidelines?|safety)", "instruction_override"),
    (r"\bnew\s+instructions?\s*:", "instruction_override"),
    (r"\b(answer|respond)\s+without\s+(using\s+)?(the\s+)?(passages?|context|sources|documents?)\b", "instruction_override"),
    (r"\b(ignora|ignorar)\s+(todas\s+)?(las\s+)?instrucciones|\bignore[zr]?\s+(toutes\s+)?les\s+instructions|\bignoriere\s+(alle\s+)?(vorherigen\s+)?anweisungen", "instruction_override_multilingual"),
    (r"\byou\s+are\s+now\s+(a|an|the|my|in)\b|\bfrom\s+now\s+on\s*,?\s+you\s+(are|will|must)\b", "role_hijack"),
    (r"\b(pretend|imagine)\s+(that\s+)?you\s+(are|were)\b|\broleplay\s+as\b|\b(act|behave)\s+as\s+(if|though)\s+you\b", "persona_override"),
    (r"\b(act|behave|pretend|simulate)\s+(to\s+be\s+|as\s+)(a|an)\s+([\w-]+\s+)?(terminal|shell|ai|assistant|chatbot|bot|model|hacker|evil|unrestricted|unfiltered|jailbroken|different)\b", "persona_override"),
    (r"\b(unrestricted|unfiltered|uncensored|jailbroken)\s+(ai|assistant|model|chatbot|mode)\b", "persona_override"),
    (r"(^|\n)\s*(system|assistant|user)\s*:", "role_delimiter"),
    (r"\b(system|assistant)\s*:\s*(you|ignore|forget|reveal|from\s+now|new\s+instructions?)\b", "role_delimiter"),
    (r"<\s*/?\s*(system|prompt|instructions?)\s*>", "xml_injection"),
    (r"\bforget\s+(everything|all|what)\s+(you|above|before|i\s+(said|told))|\bforget\s+(all\s+)?(your|previous|prior)\s+(instructions|rules|training|context|guidelines)", "memory_wipe"),
    (r"\b(reveal|show|print|output|display|repeat|tell\s+me|leak|dump)\s+(me\s+)?(your|the)\s+(system\s+prompt|(initial|original|hidden|secret|base|system)\s+(prompts?|instructions?))", "prompt_extraction"),
    (r"\bwhat\s+(were|are|was|is)\s+your\s+(initial|original|system|hidden|secret)\s+(instructions?|prompts?|rules)", "prompt_extraction"),
    (r"\b(repeat|print|output|show|dump)\s+(back\s+)?(everything|all\s+text|the\s+text)\s+(above|before)\b", "prompt_extraction"),
    (r"do\s+anything\s+now|\byou\s+are\s+(now\s+)?DAN\b|\bDAN\s+mode\b|\bjailbreak\s+(mode|prompt|yourself|this\s+(ai|chat|model))", "jailbreak"),
    (r"\b(developer|god|debug)\s+mode\b.{0,40}\b(bypass|ignore|disable|without)\b", "jailbreak"),
    (r"(\\n|\\r|%0a|%0d)\s*(system|user|assistant)\s*:", "delimiter_injection"),
    (r"(</s>|<\|endoftext\|>|<\|im_end\|>|<\|im_start\|>)\s*(system|user)?", "token_injection"),
]

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
_B64_TOKEN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")


def _normalize_variants(text: str) -> list[str]:
    """Return extra spellings of `text` to scan: NFKC + collapsed whitespace,
    letter-spaced words joined, leetspeak mapped, and decoded base64 tokens."""
    base = unicodedata.normalize("NFKC", text)
    base = re.sub(r"\s+", " ", base)
    out = [base]
    # "i g n o r e   a l l" -> "ignore all": join runs of single characters,
    # treating 2+ spaces as a word break.
    if re.search(r"(?:\b\w\b ){4,}", base) or re.search(r"(?:\w ){4,}", text):
        words = re.split(r"\s{2,}", unicodedata.normalize("NFKC", text))
        out.append(" ".join(w.replace(" ", "") if re.fullmatch(r"(\w ?)+", w) else w for w in words))
    if re.search(r"[a-z][013457@$]|[013457@$][a-z]", base, re.IGNORECASE):
        out.append(base.translate(_LEET))
    for tok in _B64_TOKEN.findall(text):
        try:
            decoded = base64.b64decode(tok + "=" * (-len(tok) % 4), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        if decoded.isprintable():
            out.append(decoded)
    return out


_DOC_INJECTION_PATTERNS: list[tuple[str, str]] = [
    (r"ignore\s+(all\s+)?previous\s+instructions?", "indirect_instruction_override"),
    (r"(system|assistant)\s*:\s*you\s+(are|must|should|will)\b", "indirect_role_injection"),
    (r"when\s+(summari[sz]ing|answering|responding\s+to|processing)\s+this\s+document\s*,?\s*(always|never|instead|do\s+not)", "behavioral_override"),
    (r"<\s*(system|instruction)\b[^>]*>", "embedded_xml_instruction"),
    (r"translate\s+(the\s+)?(above|following|this)\s+(to|into)\s+\w+\s*:\s*ignore", "translate_override"),
]

_PII_PATTERNS: list[tuple[str, str]] = [
    (r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",                          "EMAIL"),
    (r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",                        "PHONE"),
    (r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",                                                 "SSN"),
    (r"\b[A-Z]{2}\d{6}[A-Z]\b",                                                           "UK_NIN"),
    (r"\b[A-Z]{1,2}\d{6,9}\b",                                                            "PASSPORT"),
    (r"\b4\d{3}(?:[-\s]?\d{4}){3}\b",                                                    "CREDIT_CARD_VISA"),
    (r"\b5[1-5]\d{2}(?:[-\s]?\d{4}){3}\b",                                               "CREDIT_CARD_MC"),
    (r"\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b",                                            "CREDIT_CARD_AMEX"),
    (r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",   "IP_ADDRESS"),
    # Lookahead/lookbehind required: \b doesn't work on all-alphanumeric boundaries
    (r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])",                                       "AWS_ACCESS_KEY"),
    (r"(?<![A-Za-z0-9])(sk|pk)[-_](live|test)[-_][A-Za-z0-9]{20,}(?![A-Za-z0-9])",      "STRIPE_KEY"),
    (r"ghp_[A-Za-z0-9]{36}",                                                              "GITHUB_TOKEN"),
    (r"\bey[A-Za-z0-9\-_]{10,}\.[A-Za-z0-9\-_]{10,}\.[A-Za-z0-9\-_]{10,}",             "JWT_TOKEN"),
    (r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{32,64}(?![A-Fa-f0-9])(?=\s*(?:api[_\s]?key|token|secret|password))", "HEX_SECRET"),
]

MAX_QUESTION_LEN = 512
MAX_CONTEXT_LEN  = 100_000


@dataclass
class SecurityResult:
    safe:        bool
    threat_type: str = ""
    matched:     str = ""


@dataclass
class PIIResult:
    redacted_text: str
    found:         list[dict] = field(default_factory=list)

    @property
    def has_pii(self) -> bool:
        return bool(self.found)


def check_injection(text: str) -> SecurityResult:
    t = text.strip()
    for variant in [t, *_normalize_variants(t)]:
        for pattern, threat in _QUERY_INJECTION_PATTERNS:
            m = re.search(pattern, variant, re.IGNORECASE)
            if m:
                return SecurityResult(safe=False, threat_type=threat, matched=m.group(0)[:80])
    return SecurityResult(safe=True)


def check_document_injection(text: str) -> SecurityResult:
    for pattern, threat in _DOC_INJECTION_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return SecurityResult(safe=False, threat_type=threat, matched=m.group(0)[:80])
    return SecurityResult(safe=True)


def redact_pii(text: str) -> PIIResult:
    found: list[dict] = []
    redacted = text
    for pattern, pii_type in _PII_PATTERNS:
        hits = list(re.finditer(pattern, redacted))
        if not hits:
            continue
        for h in hits:
            raw  = h.group(0)
            hint = raw[:3] + "***" if len(raw) > 3 else "***"
            found.append({"type": pii_type, "hint": hint, "length": len(raw)})
        redacted = re.sub(pattern, f"[{pii_type} REDACTED]", redacted)
    return PIIResult(redacted_text=redacted, found=found)


def screen_document(text: str) -> tuple[PIIResult, SecurityResult]:
    """Shared ingestion gate for every path that adds text to the corpus
    (POST /corpus/ingest and the ingest.py CLI): redact PII first, then run the
    document-surface injection check on the redacted text."""
    pii = redact_pii(text)
    return pii, check_document_injection(pii.redacted_text)


def sanitize_input(text: str, max_len: int = MAX_QUESTION_LEN) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    return text[:max_len]


def grounding_score(answer: str, passages: list[str]) -> float:
    if not answer or not passages:
        return 0.0
    combined = " ".join(passages).lower()
    tokens   = [w for w in answer.lower().split() if len(w) > 3]
    if not tokens:
        return 1.0
    hits = sum(1 for w in tokens if w in combined)
    return round(hits / len(tokens), 3)
