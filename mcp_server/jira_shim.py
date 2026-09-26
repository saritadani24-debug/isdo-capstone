"""
ISDO Lab C2 - Mock Jira Service Management REST API (Flask shim), port 5002.

  GET  /rest/agile/1.0/board/requests   list (filters: request_type, priority, assignee, status)
  GET  /rest/api/2/issue/<key>          one request, Jira-style nested "fields"
  PUT  /rest/api/2/issue/<key>          update a request in memory
  POST /rest/api/2/issue                create a request
  GET  /health                          health check

Run from the project folder:  python mcp_server/jira_shim.py
"""
import csv
import os

from flask import Flask, jsonify, request

app = Flask(__name__)
PORT = 5002
DATA_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "requests.csv"))
FILTERS = ["request_type", "priority", "assignee", "status"]


def load_requests():
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            rows = {}
            for line_no, row in enumerate(csv.DictReader(f), start=2):
                if None in row or None in row.values():  # wrong column count (e.g. unquoted comma)
                    print(f"Warning: skipping malformed line {line_no} in {DATA_FILE}")
                    continue
                rows[row["key"]] = row
            return rows
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
        return {}


REQUESTS = load_requests()  # in-memory store for this session


def json_body():
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) and body else None


def to_jira(req):
    """Flat CSV row -> Jira-style issue with nested 'fields'."""
    return {"key": req["key"], "fields": {
        "summary": req.get("summary"),
        "issuetype": {"name": req.get("request_type")},
        "priority": {"name": req.get("priority")},
        "status": {"name": req.get("status")},
        "assignee": {"displayName": req.get("assignee")},
        "customfield_sla": req.get("sla"),
    }}


def flatten(fields):
    """Jira-style fields ({'status': {'name': 'Done'}}) -> flat CSV-style keys."""
    rename = {"issuetype": "request_type", "customfield_sla": "sla"}
    flat = {}
    for k, v in fields.items():
        if isinstance(v, dict):
            v = v.get("name", v.get("displayName", ""))
        flat[rename.get(k, k)] = v
    flat.pop("key", None)  # the record key cannot be changed
    return flat


@app.get("/rest/agile/1.0/board/requests")
def list_requests():
    results = list(REQUESTS.values())
    for key in FILTERS:
        val = request.args.get(key)
        if val:
            results = [r for r in results if r.get(key, "").lower() == val.lower()]
    return jsonify({"issues": results, "total": len(results)})


@app.get("/rest/api/2/issue/<key>")
def get_request(key):
    req = REQUESTS.get(key)
    if not req:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    return jsonify(to_jira(req))


@app.put("/rest/api/2/issue/<key>")
def update_request(key):
    if key not in REQUESTS:
        return jsonify({"errorMessages": [f"Issue {key} does not exist"]}), 404
    data = json_body()
    if not data:
        return jsonify({"errorMessages": ["Request body must be a non-empty JSON object"]}), 400
    updates = flatten(data.get("fields", data))
    REQUESTS[key].update(updates)
    print(f"[Jira Mock] Updated {key}: {updates}")
    return jsonify({"key": key, "message": "Updated successfully"})


@app.post("/rest/api/2/issue")
def create_request():
    fields = (json_body() or {}).get("fields", {})
    if not fields.get("summary"):
        return jsonify({"errorMessages": ["Missing required field: fields.summary"]}), 400
    next_num = max((int(k.split("-")[1]) for k in REQUESTS), default=1000) + 1
    key = f"REQ-{next_num}"
    REQUESTS[key] = {"key": key, "assignee": "", "sla": "", "status": "Open", "priority": "Medium",
                     **flatten(fields)}
    print(f"[Jira Mock] Created request: {key}")
    return jsonify({"key": key, "message": "Request created"}), 201


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "Jira Mock", "requests_loaded": len(REQUESTS)})


if __name__ == "__main__":
    print(f"Jira Mock API starting on http://localhost:{PORT}")
    print(f"Loaded {len(REQUESTS)} requests from {DATA_FILE}")
    print("Endpoints: GET /rest/agile/1.0/board/requests  |  GET /health")
    app.run(host="127.0.0.1", port=PORT, debug=True)
