"""
ISDO Lab C3 - Triage Agent
Reads an IT ticket and assigns category, priority, assignment group and a PII flag,
using the Anthropic SDK with tool calling and an agentic (ReAct) loop.

Tools
  classify_ticket   structured classification (strict schema: 5 required fields)
  get_open_tickets  count of Open tickets by category (ServiceNow shim, CSV fallback)

Run from the project folder:
  python agents/triage_agent.py                      # 6 lab test tickets + open-ticket counts
  python agents/triage_agent.py "ticket text here"   # triage your own ticket (Lab Step 5)

Note on temperature=0.0: claude-opus-5 (like all Opus models from 4.7 on) rejects
non-default temperature/top_p/top_k with a 400 error, and the Python SDK v1+ no longer
accepts them. Consistency comes from strict tool use (grammar-constrained output that
always matches the schema), enum-restricted fields, explicit rules and low effort.
"""
import csv
import json
import os
import sys
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

# -- configuration ---------------------------------------------------------------
PROJECT_ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
                    if (p / "data").is_dir())
INCIDENTS_CSV = PROJECT_ROOT / "data" / "incidents.csv"
SNOW_URL = "http://localhost:5001/api/now/table/incident"

load_dotenv(PROJECT_ROOT / ".env")
MODEL = os.getenv("ISDO_MODEL", "claude-opus-5")
MAX_TURNS = 5  # safety cap on the agentic loop

CATEGORIES = ["Network", "Application", "Hardware", "Access", "Email", "Server", "Software"]
GROUPS = ["Network-Ops", "App-Support", "Desktop-Support", "Email-Support",
          "Service-Desk", "Security-Ops", "Server-Ops", "DBA-Team"]

try:  # Windows consoles: avoid UnicodeEncodeError on special characters
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# -- tool definitions --------------------------------------------------------------
tools = [
    {
        "name": "classify_ticket",
        "description": "Record the triage classification of an IT support ticket. "
                       "Call exactly once per ticket.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES,
                             "description": "The ticket category"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"],
                             "description": "Priority per the rules in the system prompt"},
                "assignment_group": {"type": "string", "enum": GROUPS,
                                     "description": "Team that should receive the ticket"},
                "pii_detected": {"type": "boolean",
                                 "description": "True if the text contains a person's name, email "
                                                "address, employee ID, phone number or IP address"},
                "reasoning": {"type": "string",
                              "description": "One sentence explaining the classification"},
            },
            "required": ["category", "priority", "assignment_group", "pii_detected", "reasoning"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_open_tickets",
        "description": "Get the number of currently Open incidents grouped by category. "
                       "Use it when current workload or a possible wider outage is relevant.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

SYSTEM_PROMPT = """You are the ISDO Triage Agent for Zensar's IT Service Desk.

For each ticket, call the classify_ticket tool exactly once. Do not answer in free text
before classifying. After the tool result, reply with one short confirmation line.

Priority rules (apply the first that matches):
- P1: service down or security risk affecting many users, a whole site/building,
      or a business-critical system (ERP/SAP, Exchange, core network); or a user fully
      unable to log in to any corporate system
- P2: significant impact on one team/department or a single important function,
      or a single user blocked from working with no workaround
- P3: single user impacted, a workaround exists or the impact is limited
- P4: service request (new software, new access, equipment) - nothing is broken

Assignment groups:
- Network-Ops: VPN, switches, Wi-Fi, connectivity     - App-Support: business apps (SAP, CRM, SharePoint)
- Desktop-Support: laptops, printers, workstations     - Email-Support: Outlook, Exchange mailboxes
- Service-Desk: password resets, account unlocks, onboarding accounts
- Security-Ops: MFA, security incidents                - Server-Ops: servers, monitoring alerts
- DBA-Team: databases and backups

PII: flag names, email addresses, employee IDs, phone numbers or IP addresses.
Placeholders such as [REDACTED] are not PII."""


# -- tool implementations -----------------------------------------------------------
def read_incidents_csv():
    """Rows from data/incidents.csv, skipping malformed lines (wrong column count)."""
    with open(INCIDENTS_CSV, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if None not in r and None not in r.values()]


def get_open_tickets():
    """Count Open incidents by category - from the ServiceNow shim, else from the CSV."""
    try:
        resp = requests.get(SNOW_URL, params={"state": "Open"}, timeout=3)
        resp.raise_for_status()
        rows, source = resp.json()["result"], "ServiceNow shim (port 5001)"
    except requests.RequestException:
        rows, source = read_incidents_csv(), "data/incidents.csv (shim not running)"
        rows = [r for r in rows if r.get("state") == "Open"]
    counts = {}
    for r in rows:
        counts[r.get("category", "Unknown")] = counts.get(r.get("category", "Unknown"), 0) + 1
    return {"source": source, "open_by_category": dict(sorted(counts.items())),
            "total_open": len(rows)}


def handle_tool_call(name, tool_input):
    if name == "classify_ticket":
        return {"status": "recorded"}  # the classification is the tool input itself
    if name == "get_open_tickets":
        return get_open_tickets()
    return {"error": f"Unknown tool: {name}"}


# -- triage agent ---------------------------------------------------------------------
def triage_ticket(client, number, short_description, description=""):
    """Run the agentic loop for one ticket. Returns the classification dict (or None)."""
    print(f"\n{'=' * 55}\nTriaging: {number}\n{'=' * 55}")
    print(f"Description: {short_description}")

    messages = [{"role": "user", "content":
                 f"Please triage this ticket:\n\nTicket: {number}\n"
                 f"Summary: {short_description}\nDetails: {description or short_description}"}]
    classification = None

    for _ in range(MAX_TURNS):  # Reason -> Act -> Observe -> Reason ...
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            output_config={"effort": "low"},  # simple, rule-based task: fast and consistent
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            break
        if response.stop_reason != "tool_use":  # max_tokens, refusal, ... -> stop, don't loop
            print(f"  ! Stopped early: stop_reason={response.stop_reason}")
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  -> Tool called: {block.name}")
            result = handle_tool_call(block.name, block.input)
            if block.name == "classify_ticket":
                classification = dict(block.input)
                print(f"  -> Category:    {classification['category']}")
                print(f"  -> Priority:    {classification['priority']}")
                print(f"  -> Assign To:   {classification['assignment_group']}")
                print(f"  -> PII Found:   {classification['pii_detected']}")
                print(f"  -> Reason:      {classification['reasoning']}")
            elif block.name == "get_open_tickets":
                print(f"  -> Open tickets: {result['open_by_category']}")
            tool_results.append({"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result)})
        messages.append({"role": "user", "content": tool_results})
    else:
        print(f"  ! Stopped after {MAX_TURNS} turns without end_turn")

    if classification is None:
        print("  ! No classification was produced for this ticket")
    return classification


# -- test tickets (Lab Step 4 + Step 5) -----------------------------------------------
TEST_TICKETS = [
    ("INC0001001", "VPN not connecting after password change",
     "User reports VPN client fails to connect after AD password was reset. Error: authentication failed."),
    ("INC0001002", "Cannot access ERP system - login error",
     "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. Started 09:00 today."),
    ("INC0001008", "Network switch down - Building C",
     "Network switch in Building C server room unresponsive. 40 users in Building C affected."),
    ("INC0001006", "Password reset request",
     "User locked out of AD account after 5 failed attempts. Needs immediate reset."),
    ("REQ-1002", "VPN access for new contractor joining project Phoenix",
     "New contractor [REDACTED NAME] emp-id ZEN-9823 joining next Monday. Email: contractor@client.com"),
    # Lab Step 5 - your own ticket. Try: "Multiple users in Sales cannot access Salesforce CRM..."
    ("CUSTOM-001", "Cannot access Salesforce CRM",
     "User cannot access Salesforce CRM from company laptop since this morning."),
]


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit(f"ANTHROPIC_API_KEY not set - add it to {PROJECT_ROOT / '.env'}")
    client = anthropic.Anthropic()
    print(f"ISDO Triage Agent  |  model: {MODEL}")

    if len(sys.argv) > 1:  # python agents/triage_agent.py "your ticket text"
        text = " ".join(sys.argv[1:])
        tickets = [("CUSTOM-CLI", text[:60], text)]
    else:
        tickets = TEST_TICKETS

    results = []
    for number, short_desc, desc in tickets:
        try:
            results.append((number, triage_ticket(client, number, short_desc, desc)))
        except anthropic.APIError as e:
            print(f"  ! API error: {e}")
            results.append((number, None))

    print(f"\n{'=' * 55}\nTRIAGE SUMMARY\n{'=' * 55}")
    print(f"  {'Ticket':<12}{'Category':<13}{'Priority':<10}{'Assign To':<17}PII")
    for number, c in results:
        if c:
            print(f"  {number:<12}{c['category']:<13}{c['priority']:<10}"
                  f"{c['assignment_group']:<17}{c['pii_detected']}")
        else:
            print(f"  {number:<12}(no classification)")

    counts = get_open_tickets()  # also demo the get_open_tickets tool directly
    print(f"\n{'=' * 55}\nOPEN TICKET COUNTS BY CATEGORY\n{'=' * 55}")
    print(f"  Source: {counts['source']}")
    for cat, n in counts["open_by_category"].items():
        print(f"  {cat:<20} {n} open")
    print(f"  {'TOTAL':<20} {counts['total_open']} open")


if __name__ == "__main__":
    main()
