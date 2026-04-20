import hashlib
import json
import sqlite3
import time
from pathlib import Path

_DB_PATH = Path("data/audit.db")


def _conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(_DB_PATH))
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
                question[:140],
                answer[:220],
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
                json.dumps(pii_types or []),
            ),
        )
        c.commit()
        c.close()
    except Exception:
        pass


def log_security_event(event_type: str, details: str, ip: str = "") -> None:
    try:
        c = _conn()
        c.execute(
            "INSERT INTO security_events (ts, event_type, details, ip) VALUES (?,?,?,?)",
            (time.time(), event_type, details[:300], ip),
        )
        c.commit()
        c.close()
    except Exception:
        pass


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
