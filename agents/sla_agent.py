"""
ISDO Lab C5 - SLA & Escalation Agent
Checks SLA breach risk, escalates CRITICAL/BREACHED P1/P2 tickets, and pauses at a
human-in-the-loop (HITL) gate before ANY escalation of a P1 ticket.

Tools
  get_sla_status  minutes remaining vs SLA target -> BREACHED / CRITICAL / AT_RISK / ON_TRACK
  update_ticket   escalate / add_note / update_state via the ServiceNow shim (PATCH, port 5001)

Guardrails enforced in code (not left to the model):
  * SLA risk and requires_escalation are calculated, never guessed
  * escalation is refused unless get_sla_status said requires_escalation
  * every P1 escalation - including update_state -> "Escalated" - goes through hitl_approve()
  * updates can only target the ticket being monitored
  * every HITL decision is logged to logs/hitl_decisions.jsonl

Run from the project folder (start the shims first: .\\start_shims.bat):
  python agents/sla_agent.py                 # 4 lab test tickets, interactive y/n prompts
  python agents/sla_agent.py --auto-reject   # non-interactive: every HITL request is rejected

Note on temperature=0.0: claude-opus-5 rejects non-default temperature (400) and the
Python SDK v1+ no longer accepts it. Consistency comes from calculated SLA values,
strict tool use, enums and code-enforced rules.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

# -- configuration -------------------------------------------------------------------
PROJECT_ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
                    if (p / "data").is_dir())
load_dotenv(PROJECT_ROOT / ".env")
MODEL = os.getenv("ISDO_MODEL", "claude-opus-5")
SNOW_URL = "http://localhost:5001/api/now/table/incident"
HITL_LOG = PROJECT_ROOT / "logs" / "hitl_decisions.jsonl"
MAX_TURNS = 6

SIMULATED_NOW = datetime(2024, 1, 15, 10, 30)             # fixed 'now' for reproducible demos
SLA_MINUTES = {"P1": 60, "P2": 240, "P3": 480, "P4": 1440}
ESCALATE_RISKS = {"BREACHED", "CRITICAL"}
ESCALATE_PRIORITIES = {"P1", "P2"}                         # P3/P4 are monitored, not escalated
HITL_PRIORITIES = {"P1"}                                   # human must approve these escalations
TEAMS = ["L2-Network-Ops", "L2-App-Support", "L2-Server-Ops", "L2-Security-Ops",
         "L2-Desktop-Support", "L2-Email-Support", "L2-Service-Desk"]

try:  # Windows consoles: avoid UnicodeEncodeError on special characters
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

# -- tool definitions -------------------------------------------------------------------
tools = [
    {
        "name": "get_sla_status",
        "description": "Check a ticket's SLA: minutes remaining, breach risk level and whether "
                       "it requires escalation.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "sla_due": {"type": "string", "description": "YYYY-MM-DD HH:MM:SS"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            },
            "required": ["ticket_number", "sla_due", "priority"],
            "additionalProperties": False,
        },
    },
    {
        "name": "update_ticket",
        "description": "Update the ticket in ServiceNow: escalate it, add a work note, or change "
                       "its state.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_number": {"type": "string"},
                "action": {"type": "string", "enum": ["escalate", "add_note", "update_state"]},
                "escalation_team": {"type": "string", "enum": TEAMS,
                                    "description": "Required for action=escalate"},
                "note": {"type": "string", "description": "Work note (recommended for every action)"},
                "new_state": {"type": "string", "enum": ["In Progress", "On Hold", "Escalated", "Resolved"],
                              "description": "Required for action=update_state"},
            },
            "required": ["ticket_number", "action"],
            "additionalProperties": False,
        },
    },
]

SYSTEM_PROMPT = """You are the ISDO SLA & Escalation Agent for Zensar's IT Service Desk.

For each ticket:
1. Call get_sla_status once with the ticket's number, SLA due time and priority.
2. If requires_escalation is true, call update_ticket with action "escalate", the right
   escalation_team and a short note giving the risk level and minutes remaining.
3. If the ticket is AT_RISK (not escalating), call update_ticket with action "add_note"
   recording the SLA risk. For ON_TRACK tickets, take no action.
4. If an escalation is rejected by the human approver, do not retry it and do not escalate
   another way - add a note that escalation was declined, then stop.
5. Finish with one short summary line.

Escalation teams by category:
  Network -> L2-Network-Ops      Application -> L2-App-Support    Server -> L2-Server-Ops
  Access/Security -> L2-Security-Ops   Hardware/Software -> L2-Desktop-Support
  Email -> L2-Email-Support      anything else -> L2-Service-Desk"""


# -- tool implementations -------------------------------------------------------------------
def get_sla_status(ticket_number, sla_due, priority, now=SIMULATED_NOW):
    try:
        due = datetime.strptime(sla_due, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return {"error": f"Invalid sla_due format: {sla_due} (expected YYYY-MM-DD HH:MM:SS)"}
    target = SLA_MINUTES[priority]
    remaining = int((due - now).total_seconds() // 60)
    pct_left = remaining / target
    if remaining < 0:
        risk, msg = "BREACHED", f"SLA breached by {-remaining} minutes"
    elif pct_left < 0.20:
        risk, msg = "CRITICAL", f"Only {remaining} minutes remaining - breach imminent"
    elif pct_left < 0.50:
        risk, msg = "AT_RISK", f"{remaining} minutes remaining - at risk"
    else:
        risk, msg = "ON_TRACK", f"{remaining} minutes remaining - on track"
    return {"ticket_number": ticket_number, "priority": priority, "sla_due": sla_due,
            "sla_target_minutes": target, "minutes_remaining": remaining,
            "percent_time_left": round(max(pct_left, 0) * 100), "breach_risk": risk,
            "status_message": msg,
            "requires_escalation": risk in ESCALATE_RISKS and priority in ESCALATE_PRIORITIES}


def update_ticket(ticket_number, action, escalation_team=None, note=None, new_state=None):
    """PATCH the ServiceNow shim. Falls back to a clearly-labelled simulation if it is down."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if action == "escalate":
        body = {"state": "Escalated", "assignment_group": escalation_team}
        summary = f"ESCALATED {ticket_number} -> {escalation_team}"
    elif action == "update_state":
        body = {"state": new_state}
        summary = f"STATE CHANGED {ticket_number} -> {new_state}"
    else:
        body = {}
        summary = f"NOTE ADDED to {ticket_number}"
    if note:
        body["work_notes"] = f"[{stamp}] [ISDO SLA Agent] {note}"
    if not body:
        return {"success": False, "error": "Nothing to update (add_note needs a note)"}
    try:
        r = requests.patch(f"{SNOW_URL}/{ticket_number}", json=body, timeout=5)
        if r.status_code != 200:
            return {"success": False, "error": f"ServiceNow returned {r.status_code}: {r.text[:200]}"}
        source = "ServiceNow shim"
    except requests.RequestException:
        source = "SIMULATED - ServiceNow shim not running"
    print(f"  [{source}] {summary}")
    return {"success": True, "ticket_number": ticket_number, "action": action,
            "message": summary, "timestamp": stamp, "source": source}


# -- HITL gate ----------------------------------------------------------------------------
def hitl_approve(ticket_number, action, detail, reason):
    """Ask a human. Anything other than 'y' - including no console (EOF) - is a rejection."""
    print("\n  " + "!!! " * 10)
    print("  HITL APPROVAL REQUIRED")
    print(f"  Ticket:  {ticket_number}\n  Action:  {action}\n  Detail:  {detail}\n  Reason:  {reason}")
    print("  " + "!!! " * 10)
    try:
        approved = input("  Approve escalation? [y/n]: ").strip().lower() == "y"
    except EOFError:
        print("  (no interactive console - treated as rejected)")
        approved = False
    return approved


def auto_reject(*_args):
    print("  HITL: auto-reject mode - escalation not approved")
    return False


def log_hitl(ticket_number, detail, approved):
    HITL_LOG.parent.mkdir(exist_ok=True)
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "ticket": ticket_number,
             "request": detail, "decision": "APPROVED" if approved else "REJECTED"}
    with open(HITL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# -- SLA agent ------------------------------------------------------------------------------
def monitor_ticket(client, number, short_description, category, priority, sla_due,
                   approver=hitl_approve):
    """Run SLA monitoring for one ticket. Returns a summary dict (used by Lab C6)."""
    print(f"\n{'=' * 55}\nSLA Check: {number} | {priority} | Category: {category}\n{'=' * 55}")
    messages = [{"role": "user", "content":
                 f"Monitor SLA for this ticket and escalate if needed:\n\nTicket: {number}\n"
                 f"Description: {short_description}\nCategory: {category}\n"
                 f"Priority: {priority}\nSLA Due: {sla_due}"}]
    outcome = {"ticket_number": number, "priority": priority, "breach_risk": None,
               "minutes_remaining": None, "escalated": False, "escalation_team": None,
               "hitl_decision": None, "actions": []}
    status = None

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL, max_tokens=1024, output_config={"effort": "low"},
            system=SYSTEM_PROMPT, tools=tools, messages=messages)
        if response.stop_reason == "end_turn":
            break
        if response.stop_reason != "tool_use":
            print(f"  ! Stopped early: stop_reason={response.stop_reason}")
            break

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            inp = block.input
            if block.name == "get_sla_status":
                # always check the ticket being monitored, with its real SLA values
                status = get_sla_status(number, sla_due, priority)
                result = status
                outcome.update(breach_risk=status.get("breach_risk"),
                               minutes_remaining=status.get("minutes_remaining"))
                print(f"  -> Risk Level: {status.get('breach_risk')}")
                print(f"  -> Status:     {status.get('status_message')}")
            elif block.name == "update_ticket":
                result = guarded_update(inp, number, priority, category, status, outcome, approver)
            else:
                result = {"error": f"Unknown tool: {block.name}"}
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result)})
        messages.append({"role": "user", "content": results})
    else:
        print(f"  ! Stopped after {MAX_TURNS} turns")

    if outcome["breach_risk"] == "ON_TRACK":
        print("  -> Monitored only - no action needed")
    return outcome


def guarded_update(inp, number, priority, category, status, outcome, approver):
    """All update_ticket calls pass through here: policy checks + HITL gate."""
    action = inp["action"]
    is_escalation = action == "escalate" or inp.get("new_state") == "Escalated"
    if inp["ticket_number"] != number:
        return {"success": False, "error": f"Can only update the monitored ticket {number}"}
    if action == "escalate" and not inp.get("escalation_team"):
        return {"success": False, "error": "escalation_team is required for escalate"}
    if action == "update_state" and not inp.get("new_state"):
        return {"success": False, "error": "new_state is required for update_state"}

    if is_escalation:
        if not (status and status.get("requires_escalation")):
            print("  ! Guardrail: escalation blocked - SLA status does not require it")
            return {"success": False, "error": "Escalation not allowed: call get_sla_status first; "
                                               "only CRITICAL/BREACHED P1/P2 tickets are escalated"}
        if outcome["hitl_decision"] == "REJECTED":
            print("  ! Guardrail: retry blocked - escalation was already rejected by the human")
            return {"success": False, "error": "Escalation was already rejected by the human approver"}
        team = inp.get("escalation_team") or "L2 team"
        if priority in HITL_PRIORITIES:
            detail = f"Escalate to {team}"
            approved = approver(number, "Escalate ticket", detail,
                                f"{status['breach_risk']} - {status['status_message']}")
            log_hitl(number, detail, approved)
            outcome["hitl_decision"] = "APPROVED" if approved else "REJECTED"
            print(f"  Decision: {outcome['hitl_decision']}")
            if not approved:
                print("  Escalation cancelled and logged.")
                return {"success": False, "message": "Escalation rejected by human approver - "
                                                     "do not retry; add a note instead"}
        else:
            print(f"  -> {priority}: no HITL gate - escalating automatically")

    result = update_ticket(number, action, inp.get("escalation_team"), inp.get("note"),
                           inp.get("new_state"))
    if result.get("success"):
        outcome["actions"].append(action if action != "update_state" else f"state:{inp['new_state']}")
        if is_escalation:
            outcome.update(escalated=True, escalation_team=inp.get("escalation_team"))
    return result


# -- test tickets (Lab Step 4 + Step 5) -------------------------------------------------------
# sla_due values are chosen so each ticket lands in the state Step 4 describes under the
# Step 1 rules (the lab's own values give ON_TRACK for INC0001002 and INC0001001).
TEST_TICKETS = [
    # P1, 10 of 60 min left (17%) -> CRITICAL -> HITL prompt: type 'y'
    ("INC0001002", "Cannot access ERP - SAP login failure", "Application", "P1", "2024-01-15 10:40:00"),
    # P1, due 09:30 -> BREACHED -> HITL prompt: type 'n'
    ("INC0001010", "Exchange server high CPU", "Server", "P1", "2024-01-15 09:30:00"),
    # P2, 90 of 240 min left (38%) -> AT_RISK -> note only.
    # Step 5: change to "2024-01-15 10:00:00" -> BREACHED -> auto-escalated to L2-Network-Ops
    ("INC0001001", "VPN not connecting after password change", "Network", "P2", "2024-01-15 12:00:00"),
    # P3, due in 2 days -> ON_TRACK -> monitored only
    ("INC0001003", "Laptop running very slowly", "Hardware", "P3", "2024-01-17 09:00:00"),
]


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit(f"ANTHROPIC_API_KEY not set - add it to {PROJECT_ROOT / '.env'}")
    approver = auto_reject if "--auto-reject" in sys.argv else hitl_approve
    client = anthropic.Anthropic()
    print(f"ISDO SLA Agent  |  model: {MODEL}  |  simulated now: {SIMULATED_NOW:%Y-%m-%d %H:%M}")

    outcomes = []
    for ticket in TEST_TICKETS:
        try:
            outcomes.append(monitor_ticket(client, *ticket, approver=approver))
        except anthropic.APIError as e:
            print(f"  ! API error: {e}")

    print(f"\n{'=' * 55}\nSLA SUMMARY\n{'=' * 55}")
    print(f"  {'Ticket':<12}{'Pri':<5}{'Risk':<10}{'Min left':<10}{'Escalated to':<20}HITL")
    for o in outcomes:
        print(f"  {o['ticket_number']:<12}{o['priority']:<5}{str(o['breach_risk']):<10}"
              f"{str(o['minutes_remaining']):<10}{str(o['escalation_team'] or '-'):<20}"
              f"{o['hitl_decision'] or '-'}")
    print(f"\n  HITL decisions logged to {HITL_LOG}")


if __name__ == "__main__":
    main()
