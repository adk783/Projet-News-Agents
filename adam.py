"""
adam.py — Analyse ABSA + débat multi-agents (Adam / POC-Filtrage-Agents)
------------------------------------------------------------------------
Lit articles.csv, lance l'analyse d'Adam, sauvegarde resultats/adam.csv

Prérequis :
    - Clés API dans un fichier .env à la racine de POC-Filtrage-Agents :
        GROQ_API_KEY=...
        MISTRAL_API_KEY=...   (optionnel, fallback)
    - pip install -r ../POC-Filtrage-Agents/requirements.txt

Usage :
    python adam.py
"""

import sys
import os
import csv
import time
import json
from dotenv import load_dotenv

# Pointer vers le dossier d'Adam
ADAM_PATH = os.path.join(os.path.dirname(__file__), '..', 'POC-Filtrage-Agents')
sys.path.insert(0, os.path.abspath(ADAM_PATH))

# Charger les clés API depuis .env d'Adam
load_dotenv(os.path.join(ADAM_PATH, '.env'))

from src.agents.agent_absa import run_absa

ARTICLES_CSV = os.path.join(os.path.dirname(__file__), 'articles.csv')
OUTPUT_CSV   = os.path.join(os.path.dirname(__file__), 'resultats', 'adam.csv')

# ── Lecture des articles ────────────────────────────────────────────────────────
articles = []
with open(ARTICLES_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    articles = list(reader)

print(f"{len(articles)} articles a analyser (Adam / ABSA)")

# ── Analyse ────────────────────────────────────────────────────────────────────
results = []
for i, row in enumerate(articles):
    url     = row['url']
    ticker  = row['ticker']
    title   = row['title']
    content = row['content'] or ''
    text    = (content if content else title)[:1500]

    print(f"[{i+1}/{len(articles)}] {ticker} | {title[:60]}")

    try:
        absa_result = run_absa(text)
        aspects = absa_result.get('aspects', [])

        # Extraire les scores par aspect
        aspect_scores = {a['aspect']: a['score'] for a in aspects if 'aspect' in a and 'score' in a}

        results.append({
            'url':            url,
            'ticker':         ticker,
            'title':          title,
            'aspects_json':   json.dumps(aspects),           # tous les aspects bruts
            'revenue':        aspect_scores.get('revenue'),
            'earnings':       aspect_scores.get('earnings'),
            'guidance':       aspect_scores.get('guidance'),
            'margin':         aspect_scores.get('margin'),
            'debt':           aspect_scores.get('debt'),
            'litigation':     aspect_scores.get('litigation'),
            'macro_exposure': aspect_scores.get('macro_exposure'),
            'regulatory_risk':aspect_scores.get('regulatory_risk'),
            'competition':    aspect_scores.get('competition'),
            'valuation':      aspect_scores.get('valuation'),
        })
        print(f"  -> {len(aspects)} aspects detectes")

    except Exception as e:
        print(f"  [ERREUR] {e}")
        results.append({
            'url': url, 'ticker': ticker, 'title': title,
            'aspects_json': '[]',
            'revenue': None, 'earnings': None, 'guidance': None,
            'margin': None, 'debt': None, 'litigation': None,
            'macro_exposure': None, 'regulatory_risk': None,
            'competition': None, 'valuation': None,
        })

    time.sleep(0.5)

# ── Export CSV ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
fields = ['url','ticker','title','aspects_json',
          'revenue','earnings','guidance','margin','debt',
          'litigation','macro_exposure','regulatory_risk','competition','valuation']
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResultats sauvegardes -> resultats/adam.csv ({len(results)} lignes)")
