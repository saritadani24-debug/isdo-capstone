"""
ISDO Lab C4 - Resolution / KB Agent
Searches the Lab C1 ChromaDB knowledge base and drafts a resolution for a ticket.
HIGH confidence on a non-P1 ticket -> auto-resolve (L1). Anything else -> HITL flag.

Tools
  search_kb         query ChromaDB collection 'isdo_kb', top 2 articles + confidence (1 - distance)
  draft_resolution  ticket_number, resolution_text, auto_resolve, confidence, kb_article_used

Guardrail: confidence (HIGH/MEDIUM/LOW) and auto_resolve are enforced in code from the
real search score and ticket priority - the model cannot talk itself into auto-resolving.

Run from the project folder (KB from Lab C1 in data/chroma_db; built automatically if missing):
  python agents/resolution_agent.py                        # lab test tickets
  python agents/resolution_agent.py "ticket text" [P1-P4]  # your own ticket

Note on temperature=0.0: claude-opus-5 rejects non-default temperature (400) and the
Python SDK v1+ no longer accepts it. Consistency comes from strict tool use, a single
fixed search per ticket, code-enforced thresholds and low effort.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import anthropic
import chromadb
from dotenv import load_dotenv

# -- configuration -------------------------------------------------------------------
PROJECT_ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
                    if (p / "data" / "kb").is_dir())
KB_DIR = PROJECT_ROOT / "data" / "kb"
DB_DIR = PROJECT_ROOT / "data" / "chroma_db"
KB_SETUP = PROJECT_ROOT / "labs" / "C1" / "kb_setup.py"
COLLECTION = "isdo_kb"

load_dotenv(PROJECT_ROOT / ".env")
MODEL = os.getenv("ISDO_MODEL", "claude-opus-5")
MAX_TURNS = 4

HIGH_THRESHOLD = 0.60     # score > 0.60  -> HIGH
MEDIUM_THRESHOLD = 0.35   # score > 0.35  -> MEDIUM, otherwise LOW
AUTO_RESOLVE_PRIORITIES = {"P2", "P3", "P4"}  # never auto-resolve P1 (lab: P1 always needs a human)

try:  # Windows consoles: avoid UnicodeEncodeError on special characters
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


# -- knowledge base (Lab C1) ---------------------------------------------------------
def load_kb():
    """Open the persistent Lab C1 collection; build it with labs/C1/kb_setup.py if missing."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        kb = client.get_collection(COLLECTION)
        if kb.count() > 0:
            print(f"KB loaded: {kb.count()} chunks from {DB_DIR}")
            return kb
    except Exception:
        pass
    print("KB not found - building it with labs/C1/kb_setup.py ...")
    spec = importlib.util.spec_from_file_location("kb_setup", KB_SETUP)
    kb_setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kb_setup)
    return kb_setup.build_collection()


def confidence_level(score):
    if score > HIGH_THRESHOLD:
        return "HIGH"
    return "MEDIUM" if score > MEDIUM_THRESHOLD else "LOW"


# -- tool definitions ------------------------------------------------------------------
tools = [
    {
        "name": "search_kb",
        "description": "Search the IT knowledge base. Returns the top 2 matching KB articles "
                       "(full text) with a confidence score between 0 and 1.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string",
                                     "description": "The ticket summary and details, as given"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "draft_resolution",
        "description": "Record the drafted resolution for the ticket. Call exactly once.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "resolution_text": {"type": "string",
                                    "description": "3-4 numbered steps taken from the KB article, "
                                                   "written for the requester"},
                "auto_resolve": {"type": "boolean",
                                 "description": "True only if the KB article marks this issue as "
                                                "L1 auto-resolvable AND confidence is HIGH"},
                "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"],
                               "description": "Level stated in the search result for the article used"},
                "kb_article_used": {"type": "string",
                                    "description": "Article file name, or 'none' if nothing matches"},
            },
            "required": ["ticket_number", "resolution_text", "auto_resolve", "confidence",
                         "kb_article_used"],
            "additionalProperties": False,
        },
    },
]

SYSTEM_PROMPT = f"""You are the ISDO Resolution Agent for Zensar's IT Service Desk.

For each ticket:
1. Call search_kb ONCE, using the ticket summary and details as the query. Do not rephrase or
   search again - a LOW score is a valid, expected outcome when the KB has no matching article.
2. Call draft_resolution ONCE:
   - confidence: use the confidence_level the search result gives for the article you use
     (HIGH > {HIGH_THRESHOLD:.0%}, MEDIUM > {MEDIUM_THRESHOLD:.0%}, otherwise LOW).
   - resolution_text: 3-4 numbered, specific steps copied from the article's Resolution Steps
     that fit this ticket. For LOW confidence, write triage/escalation steps for L2 instead and
     set kb_article_used to "none".
   - auto_resolve: true only if confidence is HIGH, the article's "Auto-Resolve Eligibility"
     section says this case is L1 auto-resolvable, and the ticket is not P1.
3. Then reply with one short confirmation line."""


# -- tool implementations ----------------------------------------------------------------
KB = None  # opened in main()


def search_kb(query):
    """Best-matching chunk per article -> top 2 articles, each with its full text."""
    raw = KB.query(query_texts=[query], n_results=10, include=["metadatas", "distances"])
    best = {}
    for meta, dist in zip(raw["metadatas"][0], raw["distances"][0]):
        article = meta.get("article", "unknown")
        if article not in best or dist < best[article][0]:
            best[article] = (dist, meta.get("section", ""))
    articles = []
    for article, (dist, section) in sorted(best.items(), key=lambda kv: kv[1][0])[:2]:
        score = round(min(1.0, max(0.0, 1.0 - dist)), 2)  # cosine distance -> 0..1
        path = KB_DIR / article
        articles.append({"article": article, "confidence_score": score,
                         "confidence_level": confidence_level(score), "best_section": section,
                         "content": path.read_text(encoding="utf-8") if path.exists() else ""})
    return {"query": query, "articles": articles}


def apply_guardrail(draft, search_result, priority):
    """Enforce confidence + auto_resolve from the real score and priority. Returns notes."""
    notes = []
    scores = {a["article"]: a["confidence_score"] for a in (search_result or {}).get("articles", [])}
    score = scores.get(draft["kb_article_used"], 0.0)  # unknown / 'none' article -> 0
    level = confidence_level(score)
    if draft["confidence"] != level:
        notes.append(f"confidence {draft['confidence']} -> {level} (score {score:.0%})")
        draft["confidence"] = level
    allowed = level == "HIGH" and priority in AUTO_RESOLVE_PRIORITIES
    if draft["auto_resolve"] and not allowed:
        notes.append(f"auto_resolve True -> False (needs HIGH and priority in "
                     f"{sorted(AUTO_RESOLVE_PRIORITIES)}; got {level}, {priority})")
        draft["auto_resolve"] = False
    draft["kb_score"] = score
    return notes


# -- resolution agent ---------------------------------------------------------------------
def resolve_ticket(client, number, short_description, description, category="", priority="P3"):
    """Run the agentic loop for one ticket. Returns the (guardrailed) resolution dict."""
    print(f"\n{'=' * 55}\nResolving: {number} | Category: {category or '-'} | Priority: {priority}")
    print(f"{'=' * 55}\nIssue: {short_description}")

    messages = [{"role": "user", "content":
                 f"Find a resolution for this ticket:\n\nTicket: {number}\nCategory: {category}\n"
                 f"Priority: {priority}\nSummary: {short_description}\nDetails: {description}"}]
    search_result, draft = None, None

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL, max_tokens=1500, output_config={"effort": "low"},
            system=SYSTEM_PROMPT, tools=tools, messages=messages)

        if response.stop_reason == "end_turn":
            break
        if response.stop_reason != "tool_use":
            print(f"  ! Stopped early: stop_reason={response.stop_reason}")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "search_kb":
                if search_result is None:
                    search_result = search_kb(block.input["query"])
                    result = search_result
                    print(f"  -> KB search: '{block.input['query'][:70]}'")
                    for a in result["articles"]:
                        print(f"     [{a['confidence_score']:.0%}] {a['article']}  ({a['confidence_level']})")
                else:  # enforce the single-search rule
                    result = {"error": "search_kb already called - use the earlier result"}
            elif block.name == "draft_resolution":
                draft = dict(block.input)
                result = {"status": "recorded"}
            else:
                result = {"error": f"Unknown tool: {block.name}"}
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"  ! Stopped after {MAX_TURNS} turns")

    if draft is None:  # model never drafted -> safe default: escalate
        draft = {"ticket_number": number, "resolution_text": "No resolution drafted - escalate to L2.",
                 "auto_resolve": False, "confidence": "LOW", "kb_article_used": "none"}
    for note in apply_guardrail(draft, search_result, priority):
        print(f"  ! Guardrail adjusted: {note}")

    print(f"\n  -> Confidence: {draft['confidence']} ({draft['kb_score']:.0%})"
          f"  |  Auto-resolve: {draft['auto_resolve']}")
    print(f"  -> KB Article: {draft['kb_article_used']}")
    print("\n  RESOLUTION DRAFT:")
    for line in draft["resolution_text"].splitlines():
        print(f"  {line}")
    if not draft["auto_resolve"]:
        reason = "P1 ticket" if priority == "P1" and draft["confidence"] == "HIGH" \
            else f"{draft['confidence']} confidence"
        print(f"\n  WARNING HITL FLAG: {reason} - human review required before sending.")
    return draft


# -- test tickets (Lab Step 4 + Step 5) ------------------------------------------------------
TEST_TICKETS = [  # number, summary, details, category, priority (from incidents.csv / triage)
    ("INC0001001", "VPN not connecting after password change",
     "User reports VPN client fails to connect after AD password was reset. Error: authentication failed.",
     "Network", "P2"),
    ("INC0001006", "Password reset request",
     "User locked out of AD account after 5 failed attempts.", "Access", "P2"),
    ("INC0001002", "Cannot access ERP system - login error",
     "Multiple Finance users unable to login to SAP. Error: DBCON_FAIL.", "Application", "P1"),
    # Lab Step 5 - no KB article covers this: expect LOW confidence + HITL warning
    ("TEST-WEBEX", "Cisco Webex not launching on Mac M2",
     "Cisco Webex not launching on Mac M2", "Software", "P3"),
]


def main():
    global KB
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit(f"ANTHROPIC_API_KEY not set - add it to {PROJECT_ROOT / '.env'}")
    KB = load_kb()
    client = anthropic.Anthropic()
    print(f"ISDO Resolution Agent  |  model: {MODEL}")

    args = sys.argv[1:]
    if args:  # python agents/resolution_agent.py "ticket text" [P1-P4]
        priority = args.pop() if args[-1].upper() in {"P1", "P2", "P3", "P4"} else "P3"
        text = " ".join(args)
        tickets = [("CUSTOM-CLI", text[:60], text, "", priority.upper())]
    else:
        tickets = TEST_TICKETS

    results = []
    for t in tickets:
        try:
            results.append((t[0], t[4], resolve_ticket(client, *t)))
        except anthropic.APIError as e:
            print(f"  ! API error: {e}")

    print(f"\n{'=' * 55}\nRESOLUTION SUMMARY\n{'=' * 55}")
    print(f"  {'Ticket':<12}{'Pri':<5}{'Confidence':<16}{'Auto':<7}{'HITL':<6}KB article")
    for number, pri, d in results:
        conf = f"{d['confidence']} ({d['kb_score']:.0%})"
        print(f"  {number:<12}{pri:<5}{conf:<16}{str(d['auto_resolve']):<7}"
              f"{'yes' if not d['auto_resolve'] else 'no':<6}{d['kb_article_used']}")


if __name__ == "__main__":
    main()
