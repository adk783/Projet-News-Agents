"""
antoine.py — Analyse de sentiment via Ollama (phi4-mini)
---------------------------------------------------------
Lit articles.csv, lance l'analyse d'Antoine, sauvegarde resultats/antoine.csv

Prérequis :
    - Ollama installé et lancé (ollama serve)
    - ollama pull phi4-mini
    - pip install requests

Usage :
    python antoine.py
"""

import sys
import os
import csv
import time

# Pointer vers le dossier d'Antoine
ANTOINE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Antoinev2')
sys.path.insert(0, os.path.abspath(ANTOINE_PATH))

from agent_sentiment import analyser_article

ARTICLES_CSV = os.path.join(os.path.dirname(__file__), 'articles.csv')
OUTPUT_CSV   = os.path.join(os.path.dirname(__file__), 'resultats', 'antoine.csv')

# ── Lecture des articles ────────────────────────────────────────────────────────
articles = []
with open(ARTICLES_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    articles = list(reader)

print(f"{len(articles)} articles a analyser (Antoine / Ollama phi4-mini)")

# ── Analyse ────────────────────────────────────────────────────────────────────
results = []
for i, row in enumerate(articles):
    url     = row['url']
    ticker  = row['ticker']
    title   = row['title']
    content = row['content'] or ''

    print(f"[{i+1}/{len(articles)}] {ticker} | {title[:60]}")

    try:
        r = analyser_article(url=url, ticker=ticker, title=title, content=content)
        results.append({
            'url':         url,
            'ticker':      ticker,
            'title':       title,
            'sentiment':   r['sentiment'],      # bullish / bearish / neutral
            'score':       r['score'],          # 0.0-1.0 (None si neutral)
            'reasoning':   r['reasoning'],
            'analyzed_at': r['analyzed_at'],
        })
        print(f"  -> {r['sentiment']} ({r['score']})")
    except Exception as e:
        print(f"  [ERREUR] {e}")
        results.append({
            'url': url, 'ticker': ticker, 'title': title,
            'sentiment': 'error', 'score': None,
            'reasoning': str(e), 'analyzed_at': '',
        })

    time.sleep(0.5)

# ── Export CSV ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['url','ticker','title','sentiment','score','reasoning','analyzed_at'])
    writer.writeheader()
    writer.writerows(results)

print(f"\nResultats sauvegardes -> resultats/antoine.csv ({len(results)} lignes)")
