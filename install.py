import subprocess
import sys
import urllib.request

PACKAGES = [
    "yfinance",
    "newspaper3k",
    "lxml_html_clean",
    "trafilatura",
    "flask",
    "requests",
    "python-dotenv",
]

# ── 1. Dépendances Python ──────────────────────────────────────────────────────
print("=== 1. Installation des packages Python ===\n")

errors = []
for pkg in PACKAGES:
    print(f"  Installation de {pkg}...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"  ✓ {pkg}")
    else:
        print(f"  ✗ Erreur sur {pkg} : {result.stderr.strip()}")
        errors.append(pkg)

if errors:
    print(f"\n⚠ Échec pour : {', '.join(errors)}")
else:
    print("\n✓ Tous les packages Python sont installés.")

# ── 2. Vérification Ollama ─────────────────────────────────────────────────────
print("\n=== 2. Vérification Ollama (modèle IA) ===\n")

ollama_ok = False
try:
    import requests
    r = requests.get("http://localhost:11434", timeout=3)
    ollama_ok = True
    print("  ✓ Ollama est déjà en cours d'exécution.")
except Exception:
    print("  ✗ Ollama n'est pas lancé ou pas installé.")
    print()
    print("  → Télécharge et installe Ollama depuis : https://ollama.com/download")
    print("  → Lance l'application Ollama, puis relance ce script.")

# ── 3. Vérification du modèle phi4-mini ───────────────────────────────────────
if ollama_ok:
    print("\n=== 3. Vérification du modèle phi4-mini ===\n")
    try:
        import requests
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "phi4-mini", "prompt": "hi", "stream": False, "options": {"num_predict": 1}},
            timeout=30
        )
        if r.status_code == 200:
            print("  ✓ phi4-mini est disponible.")
        else:
            raise Exception()
    except Exception:
        print("  ✗ phi4-mini n'est pas installé. Téléchargement en cours...")
        print("  (cela peut prendre plusieurs minutes selon ta connexion)\n")
        result = subprocess.run(["ollama", "pull", "phi4-mini"], capture_output=False)
        if result.returncode == 0:
            print("\n  ✓ phi4-mini installé avec succès.")
        else:
            print("\n  ✗ Erreur. Lance manuellement dans un terminal : ollama pull phi4-mini")

# ── Résumé final ───────────────────────────────────────────────────────────────
print("\n=== Résumé ===\n")
if not errors and ollama_ok:
    print("✓ Tout est prêt. Lance : python dashboard.py")
else:
    if errors:
        print(f"⚠ Packages Python à régler : {', '.join(errors)}")
    if not ollama_ok:
        print("⚠ Installe et lance Ollama, puis relance ce script.")
