"""
samuel.py — Analyse via FinBERT + modèles spécialisés (Lorenzo/Samuel)
-----------------------------------------------------------------------
Lit articles.csv, lance les 4 agents de Samuel, sauvegarde resultats/samuel.csv

Prérequis :
    - pip install transformers torch peft scikit-learn
    - Les dossiers de modèles dans ../samuel/ :
        uncertainty_model/, litigious_model/, fundamental_strength_model/

Usage :
    python samuel.py
"""

import sys
import os
import csv
import time

# Pointer vers le dossier de Samuel
SAMUEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'samuel')
sys.path.insert(0, os.path.abspath(SAMUEL_PATH))

from polarity_agent          import PolarityAgent
from uncertainty_agent       import UncertaintyAgent
from litigious_agent         import LitigiousAgent
from fundamental_strength_agent import FundamentalStrengthAgent

ARTICLES_CSV = os.path.join(os.path.dirname(__file__), 'articles.csv')
OUTPUT_CSV   = os.path.join(os.path.dirname(__file__), 'resultats', 'samuel.csv')

MODEL_BASE = os.path.abspath(SAMUEL_PATH)

# ── Chargement des agents (une seule fois) ──────────────────────────────────────
print("Chargement des agents Samuel...")

polarity_agent    = PolarityAgent()
print("  PolarityAgent OK")

uncertainty_agent = None
unc_path = os.path.join(MODEL_BASE, 'uncertainty_model')
if os.path.exists(unc_path):
    uncertainty_agent = UncertaintyAgent(model_path=unc_path)
    print("  UncertaintyAgent OK")
else:
    print("  UncertaintyAgent ABSENT (uncertainty sera null)")

litigious_agent = LitigiousAgent(
    model_path=os.path.join(MODEL_BASE, 'litigious_model'),
    fallback_to_heuristic=True
)
print("  LitigiousAgent OK")

fundamental_agent = FundamentalStrengthAgent(
    model_path=os.path.join(MODEL_BASE, 'fundamental_strength_model'),
    fallback_to_heuristic=True
)
print("  FundamentalStrengthAgent OK")

# ── Lecture des articles ────────────────────────────────────────────────────────
articles = []
with open(ARTICLES_CSV, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    articles = list(reader)

print(f"\n{len(articles)} articles a analyser (Samuel / FinBERT)")

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
        polarity, polarity_conf, polarity_label = polarity_agent.predict(text)

        uncertainty = None
        if uncertainty_agent:
            uncertainty = round(float(uncertainty_agent.predict(text)), 4)

        litigious            = round(float(litigious_agent.predict(text)), 4)
        fundamental_strength = round(float(fundamental_agent.predict(text)), 4)

        results.append({
            'url':                  url,
            'ticker':               ticker,
            'title':                title,
            'polarity':             polarity,             # -1 / 0 / 1
            'polarity_conf':        round(polarity_conf, 4),
            'polarity_label':       polarity_label,       # negative/neutral/positive
            'uncertainty':          uncertainty,          # [0,1] ou null
            'litigious':            litigious,            # [0,1]
            'fundamental_strength': fundamental_strength, # [0,1]
        })
        print(f"  -> pol={polarity} ({polarity_conf:.2f}) | unc={uncertainty} | lit={litigious} | fund={fundamental_strength}")

    except Exception as e:
        print(f"  [ERREUR] {e}")
        results.append({
            'url': url, 'ticker': ticker, 'title': title,
            'polarity': None, 'polarity_conf': None, 'polarity_label': 'error',
            'uncertainty': None, 'litigious': None, 'fundamental_strength': None,
        })

# ── Export CSV ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
fields = ['url','ticker','title','polarity','polarity_conf','polarity_label',
          'uncertainty','litigious','fundamental_strength']
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResultats sauvegardes -> resultats/samuel.csv ({len(results)} lignes)")
