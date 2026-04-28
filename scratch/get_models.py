import os
import requests
from dotenv import load_dotenv

load_dotenv()

providers = {
    "Groq": ("https://api.groq.com/openai/v1/models", os.getenv("GROQ_API_KEY")),
    "Cerebras": ("https://api.cerebras.ai/v1/models", os.getenv("CEREBRAS_API_KEY")),
    "Mistral": ("https://api.mistral.ai/v1/models", os.getenv("MISTRAL_API_KEY")),
    "NVIDIA NIM": ("https://integrate.api.nvidia.com/v1/models", os.getenv("NVIDIA_NIM_API_KEY"))
}

for name, (url, key) in providers.items():
    if not key:
        print(f"Skipping {name}, no API key")
        continue
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {key}"})
        if r.status_code == 200:
            models = [m["id"] for m in r.json().get("data", []) if "id" in m]
            # filter out non-chat models if possible, or just print all
            print(f"\n=== {name} ({len(models)} models) ===")
            print(", ".join(sorted(models)))
        else:
            print(f"Error {name}: {r.status_code} - {r.text[:100]}")
    except Exception as e:
        print(f"Error fetching {name}: {e}")
