"""
ISDO Lab C9 — PII Redaction Middleware
Masks PII before any ticket data is sent to Claude.

Detected (each value gets a stable token, e.g. [NAME_1], used for every occurrence):
  NAME         person names - spaCy NER (if the model is installed) PLUS context rules
               that work without spaCy: "User John Smith", "for Michael D'Souza",
               "Dr. Patel", "name: Priya", "assigned to Rahul Verma", ...
               Once a name is found, its parts ("Smith") are masked wherever they recur.
  USERNAME     login names - DOMAIN\\user, "username: jsmith", "login id j.smith01",
               "AD account jsmith", "user j.smith" (dotted / underscored / digit logins)
  EMAIL, IP_ADDRESS, EMPLOYEE_ID (ZEN-9823, EMP-00142, "emp ID 4521"), PHONE (+91 / 10-digit / US)
Ticket numbers (INC0001001, REQ-1002, CHG0000001) are never masked.

Usage:
    from guardrails.pii_redactor import redact, restore

    clean_text, mapping = redact(raw_text)
    # ... send clean_text to Claude ...
    original_text = restore(claude_response, mapping)

spaCy is optional. For the best name detection install the model once:
    python -m spacy download en_core_web_sm
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

# ── spaCy (optional) ──────────────────────────────────────────────────────────

PII_ENGINE = "regex + context rules"
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
        PII_ENGINE = "spaCy NER + regex + context rules"
    except OSError:
        SPACY_AVAILABLE = False
        print("⚠  spaCy is installed but its English model is missing - name detection uses context "
              "rules only.\n   Install the model with:  python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    print("⚠  spaCy not installed - name detection uses context rules only.")

# ── PATTERNS ──────────────────────────────────────────────────────────────────

NAME_WORD = r"(?=[A-Za-z'’]{2})[A-Z][a-z]*(?:['’]?[A-Z][a-z]+)*(?:-[A-Z][a-z]+)?(?![A-Za-z])"  # John, D'Souza, McDonald, Mary-Jane
FULL_NAME = rf"{NAME_WORD}(?:\s+(?:{NAME_WORD}|[A-Z]\.))*?\s+{NAME_WORD}"   # 2+ words
ONE_OR_MORE_NAMES = rf"{NAME_WORD}(?:\s+{NAME_WORD}){{0,2}}"

# (label, compiled pattern, value group). Earlier entries win when matches overlap.
PATTERNS = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), 0),
    ("USERNAME", re.compile(r"\b[A-Za-z][A-Za-z0-9\-]{1,15}\\[A-Za-z][\w.\-]{1,63}\b"), 0),   # ZENSAR\jsmith
    ("IP_ADDRESS", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), 0),
    ("PHONE", re.compile(r"(?<![\w+])(?:\+?91[\s\-]?)?(?:\d{10}|\d{5}[\s\-]\d{5})(?!\d)"
                         r"|(?<![\w+])(?:\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]\d{3}[\s\-.]\d{4}(?!\d)"), 0),
    ("EMPLOYEE_ID", re.compile(r"\b(?:EMP|ZEN)-?\d{3,6}\b", re.IGNORECASE), 0),
    ("EMPLOYEE_ID", re.compile(r"(?i:\b(?:emp(?:loyee)?\.?\s*(?:id|no|number|#))\s*[:#=\-]?\s*)"
                               r"(?P<v>[A-Z]{0,4}-?\d{3,8})\b"), "v"),
    ("USERNAME", re.compile(r"(?i:\b(?:user\s?name|user\s?id|login(?:\s?id|\s?name)?|logon(?:\s?name)?"
                            r"|sam\s?account\s?name|upn|ad\s+(?:account|user(?:\s?name)?|login|id)"
                            r"|account\s?(?:name|id))\b)\s*(?:\bis\b|[:=#])\s*['\"]?"
                            r"(?P<v>[A-Za-z0-9][\w.\-]{1,63})"), "v"),
    ("USERNAME", re.compile(r"(?i:\b(?:user\s?name|user\s?id|login\s?(?:id|name)|logon\s?name"
                            r"|sam\s?account\s?name|upn|ad\s+(?:account|user(?:\s?name)?|login|id)"
                            r"|account\s?(?:name|id))\b)\s+['\"]?(?P<v>[a-z][a-z0-9._\-]{2,63})\b"), "v"),
    ("USERNAME", re.compile(r"(?i:\b(?:user|account|for|login)\b)\s+"
                            r"(?P<v>[a-z][a-z0-9]*(?:[._][a-z0-9]+)+|[a-z]{2,}\d{2,})\b"), "v"),
    ("USERNAME", re.compile(r"(?i:\b(?:user|account)\b)\s+(?P<plain>[a-z]{4,20})\b(?![.\-]\w)"), "plain"),
    ("NAME", re.compile(rf"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+(?P<v>{ONE_OR_MORE_NAMES})"), "v"),
    ("NAME", re.compile(rf"(?i:\b(?:my\s+name\s+is|name\s*[:=]|named|dear|hi|hello|thanks|regards,?|"
                        rf"i\s*am|i'm|this\s+is)\s+)(?P<v>{ONE_OR_MORE_NAMES})"), "v"),
    ("NAME", re.compile(rf"(?i:\b(?:user|employee|contractor|requester|requestor|caller|manager|colleague|"
                        rf"contact|for|by|from|with|cc|assigned\s+to|on\s+behalf\s+of|raised\s+by|"
                        rf"reported\s+by|approved\s+by|joiner|staff|engineer|intern|to|and|or|called|emailed|informed|notified|"
                        rf"contacted|ask|asked|told|per|via|cc:?)\s+)(?P<v>{FULL_NAME})"), "v"),
]

# Ticket references are not PII and must survive redaction untouched.
TICKET_REF = re.compile(r"\b(?:INC|REQ|CHG|RITM|TASK|PRB)-?\d{4,10}\b", re.IGNORECASE)

# Words that look like names to the rules above but are not people.
NOT_A_NAME = {
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january",
    "february", "march", "april", "may", "june", "july", "august", "september", "october",
    "november", "december", "finance", "building", "floor", "room", "team", "teams", "sales",
    "marketing", "legal", "support", "service", "desk", "helpdesk", "admin", "administrator",
    "microsoft", "windows", "outlook", "office", "excel", "word", "exchange", "sharepoint", "azure",
    "google", "chrome", "apple", "mac", "macbook", "iphone", "android", "cisco", "anyconnect",
    "webex", "zoom", "salesforce", "oracle", "sap", "adobe", "acrobat", "python", "network",
    "server", "printer", "laptop", "desktop", "workstation", "user", "users", "error", "password",
    "active", "directory", "project", "phoenix", "hr", "it", "the", "this", "that", "please",
    "access", "request", "issue", "ticket", "manager", "contractor", "employee", "new", "senior",
    "board", "department", "dept", "group", "account", "login", "client", "vpn", "email", "mail",
    "laserjet", "hp", "dell", "lenovo", "linux", "unix", "sonoma", "pro", "team", "corporate",
}
NOT_A_USERNAME = {
    "locked", "disabled", "expired", "is", "was", "has", "not", "the", "a", "an", "for", "after",
    "reset", "password", "unlock", "unlocked", "access", "issue", "request", "and", "or", "of",
}
COMMON_WORDS = {
    "reports", "reported", "reporting", "cannot", "can", "could", "unable", "says", "said", "needs",
    "need", "wants", "requests", "requested", "request", "is", "was", "has", "had", "have", "will",
    "would", "should", "does", "did", "got", "gets", "getting", "tried", "tries", "trying", "sees",
    "saw", "lost", "locked", "lockout", "disabled", "expired", "access", "accounts", "password",
    "passwords", "login", "logins", "logged", "profile", "profiles", "details", "name", "names",
    "data", "group", "groups", "creation", "setup", "settings", "management", "with", "from", "that",
    "this", "there", "their", "they", "also", "still", "again", "only", "just", "when", "after",
    "before", "while", "since", "unable", "complains", "complained", "called", "calls", "mentions",
    "experiencing", "facing", "seeing", "getting", "receiving", "received", "receives", "keeps",
    "error", "errors", "issue", "issues", "problem", "problems", "being", "been", "never", "always",
    "admin", "admins", "guest", "test", "local", "domain", "service", "services", "type", "list",
    # devices / IT nouns that commonly follow "user ..." in ticket text
    "laptop", "laptops", "desktop", "workstation", "machine", "computer", "device", "devices",
    "phone", "mobile", "tablet", "printer", "monitor", "keyboard", "mouse", "headset", "mailbox",
    "inbox", "email", "outlook", "session", "sessions", "screen", "browser", "network", "wifi",
    "connection", "drive", "folder", "files", "file", "license", "licence", "software", "application",
    "app", "apps", "system", "systems", "server", "client", "portal", "token", "badge", "card",
    "credentials", "permissions", "rights", "role", "roles", "home", "office", "team", "manager",
    "experience", "training", "support", "query", "ticket", "tickets", "count", "base", "story",
}
FILE_SUFFIXES = (".md", ".exe", ".msi", ".pdf", ".docx", ".xlsx", ".csv", ".txt", ".log", ".py",
                 ".com", ".net", ".org", ".zip", ".json", ".ps1", ".bat", ".dll")

# ── AUDIT LOGGER (module-level redaction events) ──────────────────────────────

audit_log = []


def _audit(action, detail):
    entry = {"timestamp": datetime.now().isoformat(), "module": "PIIRedactor",
             "action": action, "detail": detail}
    audit_log.append(entry)
    return entry

# ── DETECTION ─────────────────────────────────────────────────────────────────


def _is_name(value: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z'’\-]*", value)
    return bool(words) and not any(w.lower().strip("'’-") in NOT_A_NAME for w in words)


def _is_username(value: str) -> bool:
    v = value.lower().rstrip(".-")
    return (len(v) >= 2 and v not in NOT_A_USERNAME and not v.endswith(FILE_SUFFIXES)
            and not re.fullmatch(r"v?\d+(?:\.\d+)*", v))          # not a version number


def _find_pii(text: str) -> list:
    """All PII spans as (start, end, label, value), non-overlapping, in text order."""
    protected = [m.span() for m in TICKET_REF.finditer(text)]
    found = []

    def free(start, end):
        return all(end <= s or start >= e for s, e in protected + [(f[0], f[1]) for f in found])

    def add(start, end, label):
        value = text[start:end].strip(" .,;:")
        start = text.index(value, start) if value else start
        end = start + len(value)
        if value and free(start, end):
            found.append((start, end, label, value))

    for label, pattern, group in PATTERNS:
        for m in pattern.finditer(text):
            start, end = m.span(group)
            value = m.group(group)
            if label == "NAME" and not _is_name(value):
                continue
            if label == "USERNAME" and group and not _is_username(value):
                continue
            if group == "plain" and value.lower() in COMMON_WORDS:
                continue
            add(start, end, label)

    if SPACY_AVAILABLE:  # spaCy adds names the context rules missed (e.g. "Priya called about...")
        for ent in nlp(text).ents:
            if ent.label_ == "PERSON" and not ent.text.isupper() and _is_name(ent.text):
                add(ent.start_char, ent.end_char, "NAME")

    # A found name's parts ("Smith", "Michael") are masked wherever else they appear.
    for part in {w for f in found if f[2] == "NAME" for w in re.findall(r"[A-Za-z][A-Za-z'’\-]{2,}", f[3])}:
        if part.lower() not in NOT_A_NAME:
            for m in re.finditer(rf"(?<![\w']){re.escape(part)}(?![\w'])", text):
                add(m.start(), m.end(), "NAME")

    # Name parts hidden in emails / dotted logins (john.smith@..., j.smith01) are masked when
    # they appear capitalised elsewhere ("Smith says ...").
    handles = [f[3].split("@")[0].split("\\")[-1] for f in found if f[2] in ("EMAIL", "USERNAME")]
    for part in {p for h in handles if re.search(r"[._\-]", h) for p in re.split(r"[._\-\d]+", h)}:
        if len(part) >= 3 and part.lower() not in NOT_A_NAME | COMMON_WORDS:
            for m in re.finditer(rf"(?<![\w']){re.escape(part.capitalize())}(?![\w'])", text):
                add(m.start(), m.end(), "NAME")

    return sorted(found)

# ── REDACTION FUNCTION ────────────────────────────────────────────────────────


def _redact(text: str):
    mapping, tokens, counters, pieces, last = {}, {}, {}, [], 0
    for start, end, label, value in _find_pii(text):
        # same value -> same token, so Claude sees consistent references
        key = (label, value) if label != "NAME" else ("NAME", value)
        if key not in tokens:
            counters[label] = counters.get(label, 0) + 1
            tokens[key] = f"[{label}_{counters[label]}]"
            mapping[tokens[key]] = value
        pieces += [text[last:start], tokens[key]]
        last = end
    pieces.append(text[last:])
    return "".join(pieces), mapping


def redact(text: str) -> tuple[str, dict]:
    """
    Redact PII from text. Returns:
      - clean_text: text with PII replaced by tokens like [EMAIL_1], [NAME_1], [USERNAME_1]
      - mapping: dict to restore original values later (keep it out of logs and prompts)

    Example:
      clean, m = redact("User John Smith (jsmith) - call 9876543210")
      # clean = "User [NAME_1] ([USERNAME_1]) - call [PHONE_1]"   (depending on context)
    """
    if not text:
        return text, {}
    clean, mapping = _redact(text)
    if mapping:
        labels = sorted({t.strip("[]").rsplit("_", 1)[0] for t in mapping})
        _audit("redact", f"{len(mapping)} PII item(s) masked: {list(mapping.keys())} (types: {labels})")
    else:
        _audit("redact", "No PII detected")
    return clean, mapping


def restore(text: str, mapping: dict) -> str:
    """Restore PII tokens back to original values (for system-of-record logging only)."""
    restored = text
    for token in sorted(mapping, key=len, reverse=True):   # [NAME_10] before [NAME_1]
        restored = restored.replace(token, mapping[token])
    _audit("restore", f"{len(mapping)} PII item(s) restored")
    return restored


def get_audit_log() -> list:
    """Return all PII redaction audit entries."""
    return audit_log

# ── AUDIT TRAIL LOGGER ────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AuditLogger:
    """Logs every agent action with timestamp, agent name, tool, rationale, approval.
    The rationale is PII-redacted before it is stored - the audit trail must not leak PII."""

    def __init__(self, log_file: str = str(PROJECT_ROOT / "logs" / "audit_trail.jsonl")):
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        self.log_file = log_file
        self.entries = []

    def log(self, agent: str, action: str, ticket_number: str = "",
            tool: str = "", rationale: str = "", approval_status: str = "N/A"):
        safe_rationale, masked = _redact(rationale or "")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent,
            "action": action,
            "ticket_number": ticket_number,
            "tool": tool,
            "rationale": safe_rationale[:200],
            "pii_masked": len(masked),
            "approval_status": approval_status,
        }
        self.entries.append(entry)
        with open(self.log_file, "a", encoding="utf-8") as f:   # append-only JSONL
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  [AUDIT] {agent} | {action} | {ticket_number} | {approval_status}")
        return entry

    def print_trail(self):
        print(f"\n{'=' * 55}")
        print(f"FULL AUDIT TRAIL ({len(self.entries)} entries)")
        print(f"{'=' * 55}")
        for e in self.entries:
            print(f"  {e['timestamp'][:19]}  {e['agent']:<22} {e['action']:<20} {e['approval_status']}")

# ── DEMO ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    print("=" * 55)
    print(f"PII REDACTION DEMO  ({PII_ENGINE})")
    print("=" * 55)

    sample_tickets = [
        "User John Smith (emp ID ZEN-9823) reports VPN failure. Contact: john.smith@zensar.com or +91-9876543210.",
        "Contractor sarah.jones@client.com needs access to REQ-1002. IP: 192.168.1.45.",
        "Password reset for Michael D'Souza. Employee EMP-00142. No PII in this part.",
        "VPN not connecting after password change. Error: authentication failed. Ticket INC0001001.",
        # usernames / login names
        "AD account ZENSAR\\jsmith is locked. Username: jsmith, please unlock for John Smith.",
        "User j.smith01 cannot login to SAP since 09:00. Smith says the password was reset yesterday.",
        "Hi, I'm Priya Sharma from Finance. Ticket assigned to Rahul Verma. Dr. Patel approved.",
        "Printer HP LaserJet in Building C is offline for Finance team - Microsoft Teams also slow.",
    ]

    for i, ticket in enumerate(sample_tickets, 1):
        print(f"\n--- Ticket {i} ---")
        print(f"Original : {ticket}")
        clean, mapping = redact(ticket)
        print(f"Redacted : {clean}")
        if mapping:
            print(f"Mapping  : {mapping}")
            assert restore(clean, mapping) == ticket, "restore() must give back the original text"

    print("\n" + "=" * 55)
    print("AUDIT TRAIL DEMO")
    print("=" * 55)

    logger = AuditLogger(str(PROJECT_ROOT / "logs" / "demo_audit.jsonl"))
    logger.log("TriageAgent", "classify_ticket", "INC0001001", "classify_ticket",
               "Network/P2 — VPN failure after password change for user jsmith", "Auto")
    logger.log("ResolutionAgent", "search_kb", "INC0001001", "search_kb",
               "KB article found: vpn_troubleshooting.md (85% confidence)", "Auto")
    logger.log("SLAAgent", "get_sla_status", "INC0001001", "get_sla_status",
               "SLA AT_RISK — 90 min remaining of 240 min total", "Auto")
    logger.log("HITLGate", "approval_request", "INC0001002", "",
               "P1 escalation requires human approval", "PENDING")
    logger.log("HITLGate", "approval_decision", "INC0001002", "",
               "Human operator approved P1 escalation", "APPROVED")
    logger.log("CommunicationAgent", "post_comment", "INC0001001", "post_comment",
               "Resolution sent to John Smith (john.smith@zensar.com) — auto-resolved L1 ticket", "Auto")

    logger.print_trail()
    print(f"\nAudit log saved to: {logger.log_file}")
