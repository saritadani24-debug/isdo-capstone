"""
ISDO Lab C6 - LangGraph Orchestrator: wires the Triage, Resolution, SLA and Communication
agents into one StateGraph with a human-in-the-loop (HITL) gate for P1 tickets.

  START -> triage -> resolution -> sla --(hitl_required?)--> hitl -> communication -> END
                                        \\------ otherwise ------------^

Each node reuses the agent built in its lab, so all guardrails carry over:
  triage_node        agents/triage_agent.py     (C3)  Claude tool call -> structured JSON
  resolution_node    agents/resolution_agent.py (C4)  ChromaDB search + code-enforced confidence
  sla_node           agents/sla_agent.py        (C5)  calculated SLA risk; auto-escalates P2
  hitl_node          agents/sla_agent.py        (C5)  input() y/n approval, logged
  communication_node Claude drafts the requester message (template fallback)

Run from the project folder (start the shims first: .\\start_shims.bat):
  python orchestrator/supervisor.py             # 2 lab tickets: P2 VPN, P1 SAP (type 'y')
  python orchestrator/supervisor.py --graph     # print the graph as a Mermaid diagram
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
from langgraph.graph import END, START, StateGraph  # noqa: E402

from agents import resolution_agent, sla_agent, triage_agent  # noqa: E402

MODEL = triage_agent.MODEL
ESCALATION_TEAMS = {"Network": "L2-Network-Ops", "Application": "L2-App-Support",
                    "Server": "L2-Server-Ops", "Access": "L2-Security-Ops",
                    "Hardware": "L2-Desktop-Support", "Software": "L2-Desktop-Support",
                    "Email": "L2-Email-Support"}
APPROVER = sla_agent.hitl_approve  # swap for sla_agent.auto_reject (or a C7 test approver)
CLIENT = None                      # anthropic.Anthropic(), created in main()

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


def effective_priority(state):
    return state.get("triage_priority") or state.get("priority") or "P3"


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
                                        effective_priority(state))
    return {"kb_article": d["kb_article_used"], "resolution_text": d["resolution_text"],
            "auto_resolve": d["auto_resolve"], "confidence": d["confidence"], "kb_score": d["kb_score"],
            "audit_log": audit("ResolutionAgent", "search_kb",
                               f"{d['kb_article_used']} {d['confidence']} ({d['kb_score']:.0%}), "
                               f"auto_resolve={d['auto_resolve']}")}


def sla_node(state: TicketState) -> dict:
    header("SLA AGENT — checking deadline")
    number, priority = state["ticket_number"], effective_priority(state)
    s = sla_agent.get_sla_status(number, state["sla_due"], priority)
    if "error" in s:
        return {"sla_breach_risk": "UNKNOWN", "escalation_required": False, "hitl_required": False,
                "audit_log": audit("SLAAgent", "get_sla_status", s["error"])}
    team = ESCALATION_TEAMS.get(state.get("triage_category", ""), "L2-Service-Desk")
    escalate = s["requires_escalation"]                     # CRITICAL/BREACHED and P1/P2
    hitl = escalate and priority in sla_agent.HITL_PRIORITIES  # P1 -> human must approve
    print(f"  SLA Risk: {s['breach_risk']} | Minutes remaining: {s['minutes_remaining']}")
    entries = audit("SLAAgent", "get_sla_status", f"{s['breach_risk']}, {s['status_message']}")
    update = {"sla_breach_risk": s["breach_risk"], "minutes_remaining": s["minutes_remaining"],
              "escalation_required": escalate, "escalation_team": team if escalate else "",
              "hitl_required": hitl, "escalated": False}
    if escalate and not hitl:  # P2 CRITICAL/BREACHED: escalate automatically (no HITL gate)
        r = sla_agent.update_ticket(number, "escalate", team,
                                    f"SLA {s['breach_risk']}: {s['status_message']}")
        update["escalated"] = r.get("success", False)
        entries += audit("SLAAgent", "update_ticket", f"auto-escalated to {team}")
    elif hitl:
        print(f"  P1 {s['breach_risk']} -> routing to HITL gate before escalating to {team}")
    update["audit_log"] = entries
    return update


def hitl_node(state: TicketState) -> dict:
    header("HITL GATE — human approval required")
    number, team = state["ticket_number"], state["escalation_team"]
    detail = f"Escalate to {team}"
    approved = APPROVER(number, "Escalate ticket", detail,
                        f"{state['sla_breach_risk']} - {state['minutes_remaining']} min remaining")
    sla_agent.log_hitl(number, detail, approved)
    print(f"  Decision: {'APPROVED' if approved else 'REJECTED'}")
    entries = audit("HITLGate", "hitl_approve", f"{detail}: {'APPROVED' if approved else 'REJECTED'}")
    escalated = False
    if approved:
        r = sla_agent.update_ticket(number, "escalate", team,
                                    f"P1 escalation approved by human operator ({state['sla_breach_risk']})")
        escalated = r.get("success", False)
        entries += audit("SLAAgent", "update_ticket", f"escalated to {team}")
    else:
        sla_agent.update_ticket(number, "add_note", note="P1 escalation declined by human operator")
    return {"hitl_approved": approved, "escalated": escalated, "audit_log": entries}


COMM_PROMPT = """You are the ISDO Communication Agent for Zensar's IT Service Desk.
Write a short, friendly message (max 120 words) to the person who raised the ticket.
Start with "Dear User," and mention the ticket number. Use only the facts provided -
never invent steps, names, teams or times. If resolution steps are provided, include
them as a numbered list. Plain text only, no subject line, sign off as "ISDO Service Desk"."""


def communication_node(state: TicketState) -> dict:
    header("COMMUNICATION AGENT")
    number = state["ticket_number"]
    if state.get("auto_resolve"):
        status, scenario = "RESOLVED", (
            "The issue matched a known fix. Send the self-service resolution steps and say the "
            f"ticket is resolved; they can reply to reopen it.\nSteps:\n{state.get('resolution_text', '')}")
    elif state.get("hitl_approved") or state.get("escalated"):
        status, scenario = "ESCALATED", (
            f"The ticket was escalated to the {state.get('escalation_team')} team because the SLA is "
            f"{state.get('sla_breach_risk')}. Confirm the escalation and that a specialist is on it.")
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
        sla_agent.update_ticket(number, "update_state", new_state="Resolved",
                                note=f"Auto-resolved by ISDO using {state.get('kb_article')}")
    print(f"  USER MESSAGE:\n" + "\n".join(f"    {line}" for line in message.splitlines()))
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


# -- test tickets (Lab Step 4) ----------------------------------------------------------------
# sla_due values match Lab C5 so each ticket takes the path the lab describes
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
]


def main():
    global CLIENT
    graph = build_graph()
    if "--graph" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        return
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit(f"ANTHROPIC_API_KEY not set - add it to {PROJECT_ROOT / '.env'}")
    CLIENT = anthropic.Anthropic()
    resolution_agent.KB = resolution_agent.load_kb()
    print(f"ISDO Orchestrator  |  model: {MODEL}")

    finals = []
    for ticket in TEST_TICKETS:
        print(f"\n{'═' * 55}\nPROCESSING TICKET: {ticket['ticket_number']}\n{'═' * 55}")
        finals.append(graph.invoke({**ticket, "audit_log": []}))

    for s in finals:  # Lab Step 5 - the audit trail (persisted to JSONL in Lab C9)
        print(f"\n{'═' * 55}\nAUDIT LOG: {s['ticket_number']}  ->  {s['final_status']}\n{'═' * 55}")
        for e in s["audit_log"]:
            print(f"  {e['timestamp']}  {e['agent']:<19}{e['action']:<16}{e['detail']}")


if __name__ == "__main__":
    main()