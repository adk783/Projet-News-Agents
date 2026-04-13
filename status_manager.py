import json
import os


STATUS_FILE = "pipeline_status.json"


def write_status(pipeline, data):
    status = {}
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, "r", encoding="utf-8") as handle:
            status = json.load(handle)

    status[pipeline] = data

    with open(STATUS_FILE, "w", encoding="utf-8") as handle:
        json.dump(status, handle)
