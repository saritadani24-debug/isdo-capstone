"""
ISDO Lab C6/C7 - LangGraph Orchestrator: wires the Triage, Resolution, SLA and Communication
agents into one StateGraph with a human-in-the-loop (HITL) gate.

  START -> triage -> resolution -> sla --(hitl_required?)--> hitl -> communication -> END
                                        \\------ otherwise ------------^

HITL triggers (Lab C7) - evaluated in sla_node, all matching reasons are recorded in hitl_reason:
  1. P1 ticket whose SLA is CRITICAL or BREACHED                         (from Lab C5/C6)
  2. Access grant: category 'Access' and request_type 'Access Grant'     (security-sensitive)
  3. Resolution Agent confidence is LOW, regardless of priority          (no clear KB fix)
A ticket that goes to HITL is never auto-resolved.

Each node reuses the agent built in its lab, so all guardrails carry over:
  triage_node        agents/triage_agent.py     (C3)  Claude tool call -> structured JSON
  resolution_node    agents/resolution_agent.py (C4)  ChromaDB search + code-enforced confidence
  sla_node           agents/sla_agent.py        (C5)  calculated SLA risk, HITL triggers, P2 auto-escalation
  hitl_node          human y/n approval (input()), logged to logs/hitl_decisions.jsonl
  communication_node Claude drafts the requester message (template fallback)

Run from the project folder (start the shims first: .\\start_shims.bat):
  python orchestrator/supervisor.py               # 3 tickets: P2 VPN, P1 SAP, REQ-1002 access grant
  python orchestrator/supervisor.py --lowconf     # Lab C7 Step 3: VPN ticket replaced by a Webex issue
  python orchestrator/supervisor.py --auto-reject # answer 'n' to every HITL prompt automatically
  python orchestrator/supervisor.py --graph       # print the graph as a Mermaid diagram
"""
import operator
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

PROJECT_ROOT = next(p for p in [Path(__file__).resolve().parent, *Path(__file__).resolve().parents]
                    if (p / "agents").is_dir() and (p / "data").is_dir())
sys.path.insert(0, str(PROJECT_ROOT))

import anthropic  # noqa: E402
import requests  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402

from agents import resolution_agent, sla_agent, triage_agent  # noqa: E402

MODEL = triage_agent.MODEL
JIRA_URL = "http://localhost:5002/rest/api/2/issue"
ESCALATION_TEAMS = {"Network": "L2-Network-Ops", "Application": "L2-App-Support",
                    "Server": "L2-Server-Ops", "Access": "L2-Security-Ops",
                    "Hardware": "L2-Desktop-Support", "Software": "L2-Desktop-Support",
                    "Email": "L2-Email-Support"}
CLIENT = None  # anthropic.Anthropic(), created in main()

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


# -- shared state ------------------------------------------------------------------------
class TicketState(TypedDict, total=False):
    # input ticket
    ticket_number: str
    short_description: str
    description: str
    category: str
    priority: str
    sla_due: str
    request_type: str          # Jira request type, e.g. 'Access Grant' (REQ- tickets)
    # triage agent
    triage_category: str
    triage_priority: str
    triage_assignment_group: str
    pii_detected: bool
    # resolution agent
    kb_article: str
    resolution_text: str
    auto_resolve: bool
    confidence: str
    kb_score: float
    # SLA agent
    sla_breach_risk: str
    minutes_remaining: int
    escalation_required: bool
    escalation_team: str
    escalated: bool
    hitl_required: bool
    hitl_reason: str           # Lab C7: why the HITL gate fired (all matching triggers, " | "-separated)
    hitl_triggers: list        # machine-readable trigger codes: P1_SLA, ACCESS_GRANT, LOW_CONFIDENCE
    # HITL node
    hitl_approved: bool
    # communication agent
    user_message: str
    final_status: str
    # every node appends; operator.add merges the lists instead of overwriting
    audit_log: Annotated[list, operator.add]


def audit(agent, action, detail):
    print(f"  [AUDIT] {agent}: {action} - {detail}")
    return [{"timestamp": datetime.now().isoformat(timespec="seconds"),
             "agent": agent, "action": action, "detail": detail}]


def header(title):
    print(f"\n▶ {title}")


def sla_priority(state):
    """The priority the ticket's sla_due was set for (the ticket's own), else triage's."""
    return state.get("priority") or state.get("triage_priority") or "P3"


def most_severe_priority(state):
    """For safety checks: if either the ticket or triage says P1, treat it as P1."""
    return min(p for p in (state.get("priority"), state.get("triage_priority"), "P4") if p)


def is_jira(number):
    return number.upper().startswith("REQ-")


def lookup_request_type(number):
    """Jira request type for a REQ- ticket from the Jira shim ('' if unavailable)."""
    try:
        r = requests.get(f"{JIRA_URL}/{number}", timeout=3)
        return r.json()["fields"]["issuetype"]["name"] if r.status_code == 200 else ""
    except (requests.RequestException, KeyError, ValueError):
        return ""


def update_record(number, action, team=None, note=None, new_state=None):
    """Update the ticket in the right system: ServiceNow (INC-) or Jira (REQ-)."""
    if not is_jira(number):
        return sla_agent.update_ticket(number, action, team, note, new_state)
    fields = {}
    if action == "escalate":
        fields = {"status": {"name": "Escalated"}, "assignee": {"displayName": team}}
    elif action == "update_state":
        fields = {"status": {"name": new_state}}
    if note:
        fields["work_notes"] = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [ISDO] {note}"
    try:
        r = requests.put(f"{JIRA_URL}/{number}", json={"fields": fields}, timeout=5)
        ok, source = r.status_code == 200, "Jira shim"
    except requests.RequestException:
        ok, source = True, "SIMULATED - Jira shim not running"
    print(f"  [{source}] {action.upper()} {number}" + (f" -> {team or new_state}" if team or new_state else ""))
    return {"success": ok, "source": source}


# -- HITL prompt -------------------------------------------------------------------------------
def hitl_prompt(ticket_number, priority, reason, proposed_action):
    """Ask a human. Anything other than 'y' - including no console (EOF) - is a rejection."""
    print("  " + "WARNING " * 8)
    print(f"  Ticket:  {ticket_number} | Priority: {priority}")
    reasons = reason.split(" | ")
    for i, r in enumerate(reasons, 1):
        print(f"  Reason{' ' + str(i) if len(reasons) > 1 else ''}: {r}")
    print(f"  Proposed action if approved: {proposed_action}")
    print("  " + "WARNING " * 8)
    try:
        return input("  Approve action? [y/n]: ").strip().lower() == "y"
    except EOFError:
        print("  (no interactive console - treated as rejected)")
        return False


def auto_reject(ticket_number, priority, reason, proposed_action):
    print(f"  HITL auto-reject mode - {ticket_number}: {reason}")
    return False


APPROVER = hitl_prompt  # swap for auto_reject, or a test approver


# -- nodes -------------------------------------------------------------------------------
def triage_node(state: TicketState) -> dict:
    header(f"TRIAGE AGENT — {state['ticket_number']}")
    c = triage_agent.triage_ticket(CLIENT, state["ticket_number"], state["short_description"],
                                   state.get("description", ""))
    if not c:  # fall back to the ticket's own fields so the graph can continue safely
        return {"triage_category": state.get("category", ""), "triage_priority": state.get("priority", "P3"),
                "triage_assignment_group": "Service-Desk", "pii_detected": False,
                "audit_log": audit("TriageAgent", "classify_ticket",
                                   "no classification - fell back to ticket fields")}
    if state.get("priority") and c["priority"] != state["priority"]:
        print(f"  Note: triage priority {c['priority']} differs from ticket priority {state['priority']}"
              f" - using triage priority")
    return {"triage_category": c["category"], "triage_priority": c["priority"],
            "triage_assignment_group": c["assignment_group"], "pii_detected": c["pii_detected"],
            "audit_log": audit("TriageAgent", "classify_ticket",
                               f"{c['category']} / {c['priority']} -> {c['assignment_group']}, "
                               f"PII={c['pii_detected']}")}


def resolution_node(state: TicketState) -> dict:
    header("RESOLUTION AGENT — searching KB")
    d = resolution_agent.resolve_ticket(CLIENT, state["ticket_number"], state["short_description"],
                                        state.get("description", ""), state.get("triage_category", ""),
                                        most_severe_priority(state))  # P1 on either side: never auto-resolve
    return {"kb_article": d["kb_article_used"], "resolution_text": d["resolution_text"],
            "auto_resolve": d["auto_resolve"], "confidence": d["confidence"], "kb_score": d["kb_score"],
            "audit_log": audit("ResolutionAgent", "search_kb",
                               f"{d['kb_article_used']} {d['confidence']} ({d['kb_score']:.0%}), "
                               f"auto_resolve={d['auto_resolve']}")}


def hitl_triggers(state, priority, sla_status, request_type):
    """Return [(code, reason)] for every HITL trigger that matches this ticket."""
    triggers = []
    if priority in sla_agent.HITL_PRIORITIES and sla_status.get("requires_escalation"):
        triggers.append(("P1_SLA", f"P1 SLA {sla_status['breach_risk']} - "
                                   f"{sla_status['status_message']} - escalation needs sign-off"))
    categories = {state.get("category", ""), state.get("triage_category", "")}
    if "Access" in categories and request_type.lower() == "access grant":
        pii = " (ticket contains PII)" if state.get("pii_detected") else ""
        triggers.append(("ACCESS_GRANT", f"ACCESS GRANT - '{state['short_description']}' "
                                         f"requires security approval{pii}"))
    if state.get("confidence") == "LOW":
        triggers.append(("LOW_CONFIDENCE", f"LOW KB CONFIDENCE ({state.get('kb_score', 0):.0%}) - "
                                           f"no clear fix in the knowledge base"))
    return triggers


def sla_node(state: TicketState) -> dict:
    header("SLA AGENT — checking deadline and HITL triggers")
    number, priority = state["ticket_number"], most_severe_priority(state)
    request_type = state.get("request_type") or (lookup_request_type(number) if is_jira(number) else "")
    s = sla_agent.get_sla_status(number, state["sla_due"], sla_priority(state))  # SLA clock = ticket priority
    if "error" not in s and priority in sla_agent.ESCALATE_PRIORITIES and \
            s["breach_risk"] in sla_agent.ESCALATE_RISKS:  # triage upgraded the ticket to P1/P2
        s["requires_escalation"] = True
    if "error" in s:  # bad SLA date: can't judge risk, so a human must look
        s = {"breach_risk": "UNKNOWN", "status_message": s["error"], "minutes_remaining": None,
             "requires_escalation": False}
    team = ESCALATION_TEAMS.get(state.get("triage_category") or state.get("category", ""), "L2-Service-Desk")
    escalate = s["requires_escalation"]                        # CRITICAL/BREACHED and P1/P2
    print(f"  SLA Risk: {s['breach_risk']} | Minutes remaining: {s['minutes_remaining']}")
    entries = audit("SLAAgent", "get_sla_status", f"{s['breach_risk']}, {s['status_message']}")

    triggers = hitl_triggers(state, priority, s, request_type)
    if s["breach_risk"] == "UNKNOWN":
        triggers.append(("SLA_UNKNOWN", f"SLA could not be checked - {s['status_message']}"))
    hitl = bool(triggers)
    reason = " | ".join(r for _, r in triggers)
    update = {"sla_breach_risk": s["breach_risk"], "minutes_remaining": s["minutes_remaining"],
              "request_type": request_type, "escalation_required": escalate,
              "escalation_team": team, "hitl_required": hitl, "hitl_reason": reason,
              "hitl_triggers": [code for code, _ in triggers], "escalated": False}

    if hitl:
        print(f"  HITL required: {reason}")
        entries += audit("SLAAgent", "hitl_trigger", reason)
        if state.get("auto_resolve"):  # nothing waiting for a human may auto-resolve
            update["auto_resolve"] = False
            entries += audit("SLAAgent", "auto_resolve_suspended", "pending HITL approval")
    elif escalate:  # P2 CRITICAL/BREACHED with no other trigger: escalate automatically
        r = update_record(number, "escalate", team, f"SLA {s['breach_risk']}: {s['status_message']}")
        update["escalated"] = r.get("success", False)
        entries += audit("SLAAgent", "update_ticket", f"auto-escalated to {team}")
    update["audit_log"] = entries
    return update


def proposed_action(state):
    codes = state.get("hitl_triggers", [])
    team = state.get("escalation_team")
    if "ACCESS_GRANT" in codes:
        return "Approve the access grant and pass it to the provisioning team"
    if "P1_SLA" in codes:
        return f"Escalate to {team}"
    return f"Assign to {team} for L2 investigation"


def hitl_node(state: TicketState) -> dict:
    header("HITL GATE — human approval required")
    number, codes = state["ticket_number"], state.get("hitl_triggers", [])
    action = proposed_action(state)
    approved = APPROVER(number, most_severe_priority(state), state.get("hitl_reason", ""), action)
    decision = "APPROVED" if approved else "REJECTED"
    sla_agent.log_hitl(number, f"{action} | reason: {state.get('hitl_reason', '')}", approved)
    entries = audit("HITLGate", "approval_decision", f"{decision} - {action} (triggers: {', '.join(codes)})")
    print(f"  Decision: {decision}")

    escalated = False
    if approved and "ACCESS_GRANT" in codes:
        update_record(number, "update_state", new_state="Approved",
                      note="Access grant approved by human operator")
        entries += audit("HITLGate", "update_ticket", "access grant approved")
    elif approved:  # P1 escalation, or LOW confidence -> hand to an L2 specialist
        r = update_record(number, "escalate", state["escalation_team"],
                          f"Approved by human operator: {state.get('hitl_reason', '')}")
        escalated = r.get("success", False)
        entries += audit("SLAAgent", "update_ticket", f"escalated to {state['escalation_team']}")
    else:
        update_record(number, "add_note", note=f"HITL rejected - pending approval: {action}")
    return {"hitl_approved": approved, "escalated": escalated, "audit_log": entries}


COMM_PROMPT = """You are the ISDO Communication Agent for Zensar's IT Service Desk.
Write a short, friendly message (max 120 words) to the person who raised the ticket.
Start with "Dear User," (or "Dear Requester," for service requests) and mention the ticket
number. Use only the facts provided - never invent steps, names, teams or times, and never
repeat email addresses or other personal data from the ticket. If resolution steps are
provided, include them as a numbered list. Plain text only, no subject line, sign off as
"ISDO Service Desk"."""


def communication_node(state: TicketState) -> dict:
    header("COMMUNICATION AGENT")
    number, codes = state["ticket_number"], state.get("hitl_triggers", [])
    if state.get("auto_resolve"):
        status, scenario = "RESOLVED", (
            "The issue matched a known fix. Send the self-service resolution steps and say the "
            f"ticket is resolved; they can reply to reopen it.\nSteps:\n{state.get('resolution_text', '')}")
    elif state.get("hitl_required") and not state.get("hitl_approved"):
        status, scenario = "PENDING_APPROVAL", (
            "The request needs approval from a service desk supervisor before we can act. It is "
            "pending approval; they will be updated once a decision is made. No action is needed now.")
    elif "ACCESS_GRANT" in codes:
        status, scenario = "ACCESS_APPROVED", (
            "Their access grant request has been approved by the security approver and passed to "
            "the provisioning team, who will confirm when access is active.")
    elif state.get("escalated"):
        why = ("no standard fix was found, so a specialist will investigate"
               if "LOW_CONFIDENCE" in codes else f"the SLA is {state.get('sla_breach_risk')}")
        status, scenario = "ESCALATED", (
            f"The ticket was escalated to the {state.get('escalation_team')} team because {why}. "
            "Confirm a specialist is on it.")
    else:
        group = state.get("triage_assignment_group", "Service Desk")
        status, scenario = "ASSIGNED", (
            f"The ticket is assigned to the {group} team, who will contact them. "
            "Give a standard assignment notification.")
    try:
        r = CLIENT.messages.create(
            model=MODEL, max_tokens=400, output_config={"effort": "low"}, system=COMM_PROMPT,
            messages=[{"role": "user", "content": f"Ticket: {number}\nIssue: {state['short_description']}\n"
                                                  f"Situation: {scenario}"}])
        message = "".join(b.text for b in r.content if b.type == "text").strip()
    except anthropic.APIError as e:
        print(f"  ! Claude unavailable ({e.__class__.__name__}) - using template message")
        message = ""
    if not message:
        message = f"Dear User,\n\nRegarding {number}: {scenario}\n\nISDO Service Desk"

    if status == "RESOLVED":
        update_record(number, "update_state", new_state="Resolved",
                      note=f"Auto-resolved by ISDO using {state.get('kb_article')}")
    print("  USER MESSAGE:\n" + "\n".join(f"    {line}" for line in message.splitlines()))
    print(f"\n  ✅ FINAL STATUS: {status}")
    return {"user_message": message, "final_status": status,
            "audit_log": audit("CommunicationAgent", "draft_message", f"final_status={status}")}


# -- graph ---------------------------------------------------------------------------------
def route_after_sla(state: TicketState) -> str:
    return "hitl" if state.get("hitl_required") else "communication"


def build_graph():
    g = StateGraph(TicketState)
    g.add_node("triage", triage_node)
    g.add_node("resolution", resolution_node)
    g.add_node("sla", sla_node)
    g.add_node("hitl", hitl_node)
    g.add_node("communication", communication_node)
    g.add_edge(START, "triage")
    g.add_edge("triage", "resolution")
    g.add_edge("resolution", "sla")
    g.add_conditional_edges("sla", route_after_sla, {"hitl": "hitl", "communication": "communication"})
    g.add_edge("hitl", "communication")
    g.add_edge("communication", END)
    return g.compile()


# -- test tickets (Lab C6 Step 4 + Lab C7 Steps 2-4) ------------------------------------------
# sla_due values match Lab C5 so each ticket takes the path the labs describe
# (simulated now = 2024-01-15 10:30; the CSV's 11:00 for INC0001002 would be ON_TRACK -> no HITL).
TEST_TICKETS = [
    {"ticket_number": "INC0001001", "short_description": "VPN not connecting after password change",
     "description": "User reports VPN client fails to connect after AD password was reset. "
                    "Error: authentication failed.",
     "category": "Network", "priority": "P2", "sla_due": "2024-01-15 12:00:00"},
    {"ticket_number": "INC0001002", "short_description": "Cannot access ERP system - login error",
     "description": "Multiple users in Finance unable to login to SAP. Error code: DBCON_FAIL. "
                    "Started 09:00 today.",
     "category": "Application", "priority": "P1", "sla_due": "2024-01-15 10:40:00"},
    # Lab C7 Step 4 - access grant: HITL regardless of priority
    {"ticket_number": "REQ-1002", "short_description": "VPN access for new contractor",
     "description": "Contractor needs VPN access. Email: contractor@client.com",
     "category": "Access", "priority": "P2", "sla_due": "2024-01-15 15:00:00",
     "request_type": "Access Grant"},
]

# Lab C7 Step 3 (--lowconf): the VPN ticket becomes an issue the KB does not cover
LOW_CONFIDENCE_TICKET = {
    "ticket_number": "INC0001001",
    "short_description": "Cisco Webex not launching on MacBook M2 after Sonoma update",
    "description": "Cisco Webex not launching on MacBook M2 after Sonoma update",
    "category": "Software", "priority": "P3", "sla_due": "2024-01-15 17:00:00"}


def main():
    global CLIENT, APPROVER
    graph = build_graph()
    if "--graph" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        return
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit(f"ANTHROPIC_API_KEY not set - add it to {PROJECT_ROOT / '.env'}")
    if "--auto-reject" in sys.argv:
        APPROVER = auto_reject
    tickets = list(TEST_TICKETS)
    if "--lowconf" in sys.argv:
        tickets[0] = LOW_CONFIDENCE_TICKET
    CLIENT = anthropic.Anthropic()
    resolution_agent.KB = resolution_agent.load_kb()
    print(f"ISDO Orchestrator  |  model: {MODEL}")

    finals = []
    for ticket in tickets:
        print(f"\n{'═' * 55}\nPROCESSING TICKET: {ticket['ticket_number']}\n{'═' * 55}")
        finals.append(graph.invoke({**ticket, "audit_log": []}))

    for s in finals:  # Lab C6 Step 5 / C7 Step 5 - the audit trail (persisted to JSONL in Lab C9)
        print(f"\n{'═' * 55}\nAUDIT LOG: {s['ticket_number']}  ->  {s['final_status']}\n{'═' * 55}")
        for e in s["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<19}{e['action']:<23}{e['detail']}")


if __name__ == "__main__":
    main()