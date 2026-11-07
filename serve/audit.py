import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

from serve.security import redact_pii

# An answered query is flagged for review when its question contained PII, it
# carried a threat type (blocked injection), or the answer's grounding score
# fell below this threshold.
FLAG_GROUNDING_BELOW = float(os.getenv("AUDIT_FLAG_GROUNDING_BELOW", "0.5"))

_DEFAULT_DB = "data/audit.db"


def _db_path() -> Path:
    # Resolved per call so tests can point AUDIT_DB at a temp file.
    return Path(os.getenv("AUDIT_DB", _DEFAULT_DB))


def _conn() -> sqlite3.Connection:
    db = _db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(db))
    c.row_factory = sqlite3.Row
    c.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ts               REAL    NOT NULL,
            question_hash    TEXT,
            question_preview TEXT,
            answer_preview   TEXT,
            answer_type      TEXT,
            model_used       TEXT,
            confidence       REAL,
            grounding        REAL,
            sources          TEXT,
            retrieval_ms     REAL,
            qa_ms            REAL,
            synthesis_ms     REAL,
            total_ms         REAL,
            ip               TEXT,
            flagged          INTEGER DEFAULT 0,
            threat_type      TEXT,
            pii_detected     INTEGER DEFAULT 0,
            pii_types        TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS security_events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         REAL NOT NULL,
            event_type TEXT,
            details    TEXT,
            ip         TEXT
        )
    """)
    c.commit()
    return c


def log_query(
    *,
    question:     str,
    answer:       str,
    answer_type:  str,
    model_used:   str,
    confidence:   float,
    grounding:    float = 1.0,
    sources:      list[str],
    latency:      dict,
    ip:           str = "",
    flagged:      bool = False,
    threat_type:  str = "",
    pii_detected: bool = False,
    pii_types:    list[str] | None = None,
) -> None:
    # Never store raw PII: previews are redacted, and the PII flag/types are
    # derived here so every caller gets them.
    q_pii = redact_pii(question)
    a_pii = redact_pii(answer)
    types = sorted(set(pii_types or []) | {f["type"] for f in q_pii.found + a_pii.found})
    pii_detected = bool(pii_detected or types)
    flagged = bool(
        flagged or threat_type or pii_detected
        or (answer_type != "blocked" and grounding < FLAG_GROUNDING_BELOW)
    )
    try:
        c = _conn()
        c.execute(
            """INSERT INTO audit_log
               (ts, question_hash, question_preview, answer_preview, answer_type,
                model_used, confidence, grounding, sources,
                retrieval_ms, qa_ms, synthesis_ms, total_ms,
                ip, flagged, threat_type, pii_detected, pii_types)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                time.time(),
                hashlib.sha256(question.encode()).hexdigest()[:12],
                q_pii.redacted_text[:140],
                a_pii.redacted_text[:220],
                answer_type,
                model_used,
                round(confidence, 4),
                round(grounding, 3),
                json.dumps(sources),
                latency.get("retrieval_ms", 0),
                latency.get("qa_ms", 0),
                latency.get("synthesis_ms", 0),
                latency.get("total_ms", 0),
                ip,
                int(flagged),
                threat_type,
                int(pii_detected),
                json.dumps(types),
            ),
        )
        c.commit()
        c.close()
    except Exception as exc:
        print(f"[audit] failed to write audit_log row: {exc}", file=sys.stderr)


def log_security_event(event_type: str, details: str, ip: str = "") -> None:
    try:
        c = _conn()
        c.execute(
            "INSERT INTO security_events (ts, event_type, details, ip) VALUES (?,?,?,?)",
            (time.time(), event_type, redact_pii(details).redacted_text[:300], ip),
        )
        c.commit()
        c.close()
    except Exception as exc:
        print(f"[audit] failed to write security event: {exc}", file=sys.stderr)


def get_recent_queries(n: int = 15) -> list[dict]:
    try:
        c    = _conn()
        rows = c.execute(
            """SELECT ts, question_preview, answer_type, model_used,
                      confidence, grounding, total_ms, flagged, threat_type
               FROM audit_log ORDER BY ts DESC LIMIT ?""",
            (n,),
        ).fetchall()
        c.close()
        return [
            {
                "time":       time.strftime("%H:%M:%S", time.localtime(r["ts"])),
                "question":   r["question_preview"],
                "type":       r["answer_type"],
                "model":      r["model_used"],
                "confidence": round(r["confidence"], 2),
                "grounding":  round(r["grounding"], 2),
                "ms":         round(r["total_ms"] or 0, 1),
                "flagged":    "!" if r["flagged"] else "ok",
                "threat":     r["threat_type"] or "",
            }
            for r in rows
        ]
    except Exception:
        return []


def get_security_events(n: int = 20) -> list[dict]:
    try:
        c    = _conn()
        rows = c.execute(
            "SELECT ts, event_type, details, ip FROM security_events ORDER BY ts DESC LIMIT ?",
            (n,),
        ).fetchall()
        c.close()
        return [
            {
                "time":    time.strftime("%H:%M:%S", time.localtime(r["ts"])),
                "type":    r["event_type"],
                "details": r["details"],
                "ip":      r["ip"],
            }
            for r in rows
        ]
    except Exception:
        return []


def stats() -> dict:
    try:
        c       = _conn()
        total   = c.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        flagged = c.execute("SELECT COUNT(*) FROM audit_log WHERE flagged=1").fetchone()[0]
        events  = c.execute("SELECT COUNT(*) FROM security_events").fetchone()[0]
        c.close()
        return {"total_queries": total, "flagged_queries": flagged, "security_events": events}
    except Exception:
        return {}
