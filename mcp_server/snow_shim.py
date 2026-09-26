"""
ISDO Lab C2 - Mock ServiceNow Table API (Flask shim), port 5001.

  GET   /api/now/table/incident            list (filters: category, state, priority, assignment_group)
  GET   /api/now/table/incident/<number>   one incident
  PATCH /api/now/table/incident/<number>   update fields in memory (e.g. state, work_notes)
  POST  /api/now/table/incident            create incident
  GET   /health                            health check

Run from the project folder:  python mcp_server/snow_shim.py
"""
import csv
import os

from flask import Flask, jsonify, request

app = Flask(__name__)
PORT = 5001
DATA_FILE = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data", "incidents.csv"))
FILTERS = ["category", "state", "priority", "assignment_group"]


def load_incidents():
    try:
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            rows = {}
            for line_no, row in enumerate(csv.DictReader(f), start=2):
                if None in row or None in row.values():  # wrong column count (e.g. unquoted comma)
                    print(f"Warning: skipping malformed line {line_no} in {DATA_FILE}")
                    continue
                rows[row["number"]] = row
            return rows
    except FileNotFoundError:
        print(f"Warning: {DATA_FILE} not found. Starting with empty dataset.")
        return {}


INCIDENTS = load_incidents()  # in-memory store for this session


def json_body():
    body = request.get_json(silent=True)
    return body if isinstance(body, dict) and body else None


@app.get("/api/now/table/incident")
def list_incidents():
    results = list(INCIDENTS.values())
    for key in FILTERS:
        val = request.args.get(key)
        if val:
            results = [r for r in results if r.get(key, "").lower() == val.lower()]
    return jsonify({"result": results, "total": len(results)})


@app.get("/api/now/table/incident/<number>")
def get_incident(number):
    incident = INCIDENTS.get(number)
    if not incident:
        return jsonify({"error": f"Incident {number} not found"}), 404
    return jsonify({"result": incident})


@app.patch("/api/now/table/incident/<number>")
def update_incident(number):
    if number not in INCIDENTS:
        return jsonify({"error": f"Incident {number} not found"}), 404
    updates = json_body()
    if not updates:
        return jsonify({"error": "Request body must be a non-empty JSON object"}), 400
    updates.pop("number", None)  # the record key cannot be changed
    INCIDENTS[number].update(updates)
    print(f"[ServiceNow Mock] Updated {number}: {updates}")
    return jsonify({"result": INCIDENTS[number], "message": "Updated successfully"})


@app.post("/api/now/table/incident")
def create_incident():
    data = json_body()
    if not data or not data.get("number"):
        return jsonify({"error": "Missing required field: number"}), 400
    if data["number"] in INCIDENTS:
        return jsonify({"error": f"Incident {data['number']} already exists"}), 409
    data.setdefault("state", "Open")
    INCIDENTS[data["number"]] = data
    print(f"[ServiceNow Mock] Created incident: {data['number']}")
    return jsonify({"result": data, "message": "Incident created"}), 201


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "ServiceNow Mock", "incidents_loaded": len(INCIDENTS)})


if __name__ == "__main__":
    print(f"ServiceNow Mock API starting on http://localhost:{PORT}")
    print(f"Loaded {len(INCIDENTS)} incidents from {DATA_FILE}")
    print("Endpoints: GET /api/now/table/incident  |  GET /health")
    app.run(host="127.0.0.1", port=PORT, debug=True)
