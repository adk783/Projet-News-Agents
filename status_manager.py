import json, os

STATUS_FILE = "pipeline_status.json"

def write_status(pipeline, data):
    status = {}
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r") as f:
            status = json.load(f)
    status[pipeline] = data
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)