"""
ISDO Lab C2 - automated check of both mock APIs (Lab Steps 3-6 + the PATCH/PUT loop).

Start both shims first, each in its own terminal:
    python mcp_server/snow_shim.py
    python mcp_server/jira_shim.py
Then, in a third terminal:
    python labs/C2/test_apis.py

Any records the test changes are put back to their original values afterwards.
"""
import json
import sys

import requests

SNOW = "http://localhost:5001"
JIRA = "http://localhost:5002"
results = []


def check(name, condition, detail=""):
    results.append(condition)
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}" + (f"  ->  {detail}" if detail else ""))


def get(url, **params):
    r = requests.get(url, params=params, timeout=5)
    return r.status_code, r.json()


def section(title):
    print(f"\n--- {title} ---")


def main():
    for name, base in [("ServiceNow", SNOW), ("Jira", JIRA)]:
        try:
            requests.get(f"{base}/health", timeout=3)
        except requests.ConnectionError:
            sys.exit(f"Cannot reach the {name} shim at {base}. Start it first (see top of this file).")

    section("Step 6 - health endpoints")
    for base in (SNOW, JIRA):
        code, body = get(f"{base}/health")
        check(f"GET {base}/health", code == 200 and body.get("status") == "ok", json.dumps(body))

    section("Step 3 - ServiceNow: all incidents")
    code, body = get(f"{SNOW}/api/now/table/incident")
    check("returns 15 incidents", code == 200 and body["total"] == 15, f"total={body.get('total')}")

    section("Step 4 - ServiceNow: filters and single record")
    code, body = get(f"{SNOW}/api/now/table/incident", priority="P1")
    p1 = [r["number"] for r in body["result"]]
    check("?priority=P1 returns only P1", code == 200 and p1 and all(r["priority"] == "P1" for r in body["result"]),
          f"total={body['total']} {p1}")
    code, body = get(f"{SNOW}/api/now/table/incident", category="Network")
    check("?category=Network", code == 200 and all(r["category"] == "Network" for r in body["result"]),
          f"total={body['total']}")
    code, body = get(f"{SNOW}/api/now/table/incident/INC0001001")
    check("GET /incident/INC0001001", code == 200 and body["result"]["number"] == "INC0001001",
          body["result"].get("short_description", ""))
    code, _ = get(f"{SNOW}/api/now/table/incident/INC9999999")
    check("unknown incident returns 404", code == 404)

    section("ServiceNow: PATCH update and read back")
    url = f"{SNOW}/api/now/table/incident/INC0001002"
    original = get(url)[1]["result"]["state"]
    r = requests.patch(url, json={"state": "Escalated", "work_notes": "Lab C2 test"}, timeout=5)
    check("PATCH state=Escalated", r.status_code == 200)
    check("read back shows Escalated", get(url)[1]["result"]["state"] == "Escalated")
    r = requests.patch(url, data="not json", headers={"Content-Type": "application/json"}, timeout=5)
    check("bad PATCH body returns 400", r.status_code == 400)
    requests.patch(url, json={"state": original, "work_notes": ""}, timeout=5)  # restore

    section("Step 5 - Jira: all requests")
    code, body = get(f"{JIRA}/rest/agile/1.0/board/requests")
    check("returns 10 requests", code == 200 and body["total"] == 10, f"total={body.get('total')}")
    code, body = get(f"{JIRA}/rest/agile/1.0/board/requests", request_type="Access Grant")
    check("?request_type=Access Grant", code == 200 and body["total"] >= 1, f"total={body['total']}")

    section("Jira: single issue (nested 'fields') and PUT")
    url = f"{JIRA}/rest/api/2/issue/REQ-1002"
    code, body = get(url)
    check("GET /issue/REQ-1002 has nested fields",
          code == 200 and body["fields"]["priority"]["name"] == "High", json.dumps(body["fields"])[:90] + "...")
    original = body["fields"]["status"]["name"]
    r = requests.put(url, json={"fields": {"status": {"name": "In Progress"}}}, timeout=5)
    check("PUT status=In Progress", r.status_code == 200)
    check("read back shows In Progress", get(url)[1]["fields"]["status"]["name"] == "In Progress")
    requests.put(url, json={"fields": {"status": {"name": original}}}, timeout=5)  # restore
    code, _ = get(f"{JIRA}/rest/api/2/issue/REQ-9999")
    check("unknown issue returns 404", code == 404)

    passed = sum(results)
    print(f"\n{'=' * 60}\n{passed}/{len(results)} checks passed")
    print("Both mock APIs are ready for Labs C3-C6." if passed == len(results) else "Fix the FAIL items above.")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
