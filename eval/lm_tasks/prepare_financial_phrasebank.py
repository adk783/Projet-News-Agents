"""
Prepare les donnees Financial PhraseBank en local (format JSONL).
Telecharge le fichier ZIP depuis HuggingFace et extrait les donnees
sentences_allagree, puis les convertit au format JSONL pour lm-eval.

Usage:
    python lm_tasks/prepare_financial_phrasebank.py
"""

import json
import re
import sys
import zipfile
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_FILE = OUTPUT_DIR / "financial_phrasebank.jsonl"

# Fichier ZIP officiel heberge sur HuggingFace
ZIP_URL = "https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/data/FinancialPhraseBank-v1.0.zip"

LABEL_MAP = {"negative": "negative", "neutral": "neutral", "positive": "positive"}


def ensure_requests():
    try:
        import requests  # noqa: F401
    except ImportError:
        import subprocess

        print("Installation de requests...")
        subprocess.run([sys.executable, "-m", "pip", "install", "requests", "--quiet"], check=True)


def download_zip(url: str, dest: Path) -> None:
    import requests

    print(f"Telechargement : {url}")
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    size_kb = dest.stat().st_size // 1024
    print(f"  -> Sauvegarde ({size_kb} KB) : {dest}")


def parse_txt_file(content: str) -> list[dict]:
    """
    Parse le format du fichier FinancialPhraseBank.
    Chaque ligne : sentence@label (ex: "ABC acquires XYZ.@positive")
    """
    entries = []
    for line in content.splitlines():
        line = line.strip()
        if not line or "@" not in line:
            continue
        parts = line.rsplit("@", 1)
        if len(parts) != 2:
            continue
        sentence, label = parts[0].strip(), parts[1].strip().lower()
        if label in LABEL_MAP:
            entries.append({"sentence": sentence, "label": LABEL_MAP[label]})
    return entries


def extract_and_convert(zip_path: Path, jsonl_path: Path) -> int:
    print("Extraction et conversion du ZIP...")

    # Chercher le fichier sentences_allagree dans le ZIP
    target_filename = "Sentences_AllAgree.txt"

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        # Chercher le fichier cible (insensible a la casse)
        match = next((n for n in names if target_filename.lower() in n.lower()), None)
        if match is None:
            print("Fichiers disponibles dans le ZIP :")
            for n in names:
                print(f"  - {n}")
            raise FileNotFoundError(f"Fichier '{target_filename}' introuvable dans le ZIP.")

        print(f"  -> Lecture de : {match}")
        raw = zf.read(match)
        # Essayer plusieurs encodages
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                content = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Impossible de decoder le fichier texte.")

    entries = parse_txt_file(content)
    print(f"  -> {len(entries)} exemples analyses")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"  -> Ecrits dans : {jsonl_path}")
    return len(entries)


def main() -> int:
    ensure_requests()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = OUTPUT_DIR / "FinancialPhraseBank-v1.0.zip"

    if not zip_path.exists():
        download_zip(ZIP_URL, zip_path)
    else:
        size_kb = zip_path.stat().st_size // 1024
        print(f"ZIP deja present ({size_kb} KB) : {zip_path}")

    n = extract_and_convert(zip_path, OUTPUT_FILE)

    print(f"\nDataset pret : {n} exemples dans {OUTPUT_FILE}")
    print("Lancez maintenant : .\\evaluate_models.ps1 -Limit 20")
    return 0


if __name__ == "__main__":
    sys.exit(main())
