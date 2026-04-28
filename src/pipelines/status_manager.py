import json
import os
from pathlib import Path

# On place le fichier à la racine du projet
STATUS_FILE = Path(__file__).parent.parent.parent / "pipeline_status.json"


def write_status(pipeline, data):
    status = {}
    if STATUS_FILE.exists():
        try:
            with open(STATUS_FILE) as f:
                status = json.load(f)
        except json.JSONDecodeError:
            pass
    status[pipeline] = data
    with open(STATUS_FILE, "w") as f:
        json.dump(status, f)
