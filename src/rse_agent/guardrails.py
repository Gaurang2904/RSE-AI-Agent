"""Guardrails: read-only SQL, no data changes, no prompt attacks, no export of personal data,
local models only."""
from __future__ import annotations

import re
from urllib.parse import urlparse

WRITE_INTENT = re.compile(
    r"\b(delete|drop|truncate|insert|update|alter|modify|overwrite|erase|wipe)\b"
    r".{0,60}?(rows?|records?|tables?|data(base)?|entries|clients?|columns?)(?![a-z])", re.I)
# Sending data off this machine: a transfer verb followed by a destination ("to ...", "with ...")
# or a cloud service. "What share of clients ..." (noun) must NOT match.
EXFIL_INTENT = re.compile(
    r"\b(email|e-mail|mail|send|upload|forward|whatsapp|post|export|share|sync)\b"
    r"(?:(?!\bof\b).){0,80}?\b(to|with)\b\s+(?!excel\b|a local\b|this computer\b)\w+"
    r"|\b(google (sheets?|drive|docs)|dropbox|onedrive|gmail|whatsapp|slack|teams|cloud)\b", re.I)
INJECTION = re.compile(
    r"(ignore (all |any )?(previous|prior|above) instructions|disregard (the )?(system|previous)|"
    r"you are now|reveal (your|the) (system )?prompt|system prompt)", re.I)
_FORBIDDEN_SQL = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|"
    r"truncate|grant|revoke|begin|commit|rollback|savepoint|load_extension)\b", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")

REFUSAL_WRITE = ("The client database is read-only here, so I can't add, change or delete records. "
                 "I can analyse the data and prepare a report instead.")
REFUSAL_EXFIL = ("Client data stays on this computer, so I can't send, upload or share it. I can "
                 "create a local Excel report that you can review and share yourself if appropriate.")
REFUSAL_ATTACK = "I can't follow instructions that try to change my rules or reveal my configuration."


class GuardrailError(ValueError):
    pass


def check_request(question: str) -> str | None:
    """Return a refusal message if the request must be blocked before the LLM sees it."""
    if INJECTION.search(question):
        return REFUSAL_ATTACK
    if WRITE_INTENT.search(question):
        return REFUSAL_WRITE
    if EXFIL_INTENT.search(question):
        return REFUSAL_EXFIL
    return None


def check_sql(sql: str, max_rows: int = 200) -> str:
    s = sql.strip().rstrip(";").strip()
    if not s:
        raise GuardrailError("empty SQL")
    if "--" in s or "/*" in s:
        raise GuardrailError("SQL comments are not allowed")
    if ";" in s:
        raise GuardrailError("only a single SQL statement is allowed")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise GuardrailError("only SELECT queries are allowed (the database is read-only)")
    if _FORBIDDEN_SQL.search(s):
        raise GuardrailError("write or admin SQL keywords are not allowed (read-only database)")
    if not re.search(r"\blimit\s+\d+\s*$", s, re.I):
        s = f"{s} LIMIT {max_rows}"
    return s


def mask_pii(text: str) -> str:
    text = _EMAIL.sub("[email]", text)
    text = _PHONE.sub("[phone]", text)
    return _PAN.sub("[PAN]", text)


def assert_local(host: str) -> None:
    """Real client data may only go to a model running on this machine."""
    h = urlparse(host).hostname or ""
    if h not in ("localhost", "127.0.0.1", "::1"):
        raise GuardrailError(f"refusing non-local model host '{h}': client data must stay local")
