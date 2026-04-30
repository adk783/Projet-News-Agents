"""
antoine_benchmark.py — Benchmark Antoine (Ollama phi4-mini) sur les 25 articles d'Adam
---------------------------------------------------------------------------------------
Lit benchmark_dataset.json, lance analyser_article sur chaque article,
mesure la latence, sauvegarde resultats/benchmark_antoine.csv

Usage :
    python antoine_benchmark.py
    python antoine_benchmark.py --limit 5   (test rapide)

Prérequis :
    - Ollama installé et lancé (ollama serve)
    - ollama pull phi4-mini
"""

import sys
import os
import csv
import json
import time
import argparse

# Pointer vers le dossier d'Antoine
ANTOINE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Antoinev2')
sys.path.insert(0, os.path.abspath(ANTOINE_PATH))

from agent_sentiment import analyser_article

DATASET    = os.path.join(os.path.dirname(__file__), 'benchmark_dataset.json')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'resultats', 'benchmark_antoine.csv')

# Mapping sentiment Antoine → label commun
LABEL_MAP = {
    'bullish': 'Achat',
    'bearish': 'Vente',
    'neutral': 'Neutre',
}

# ── Chargement du dataset ───────────────────────────────────────────────────────
with open(DATASET, encoding='utf-8') as f:
    data = json.load(f)

articles = data['articles']

parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=0, help='Nb articles a traiter (0 = tous)')
args = parser.parse_args()
if args.limit > 0:
    articles = articles[:args.limit]

print(f"\n{'='*65}")
print(f"BENCHMARK ANTOINE — {len(articles)} articles (Ollama phi4-mini)")
print(f"{'='*65}\n")

# ── Analyse ─────────────────────────────────────────────────────────────────────
results      = []
total_time   = 0.0
n_pertinent  = 0
n_correct    = 0

for i, article in enumerate(articles, 1):
    gt      = article['ground_truth']
    url     = f"benchmark://{article['id']}"
    ticker  = article['ticker']
    title   = article['title']
    content = article['content']

    print(f"[{i:02d}/{len(articles):02d}] {article['id']} | {title[:55]}")

    t0 = time.time()
    try:
        r = analyser_article(url=url, ticker=ticker, title=title, content=content)
        raw_sentiment  = r['sentiment']           # bullish / bearish / neutral
        score          = r['score']
        reasoning      = r['reasoning']
        prediction     = LABEL_MAP.get(raw_sentiment, 'Neutre')
    except Exception as e:
        print(f"  [ERREUR] {e}")
        raw_sentiment = 'error'
        score         = None
        reasoning     = str(e)
        prediction    = 'error'

    latency = round(time.time() - t0, 3)
    total_time += latency

    # Comparaison avec la vérité terrain
    gt_signal = gt['signal']  # Achat / Vente / Neutre / null (hors_scope)
    gt_filtrage = gt['filtrage']

    correct = None
    if gt_filtrage == 'pertinent' and gt_signal is not None:
        n_pertinent += 1
        correct = (prediction == gt_signal)
        if correct:
            n_correct += 1

    status = ''
    if gt_filtrage == 'hors_scope':
        status = 'HORS_SCOPE'
    elif correct is True:
        status = 'OK'
    elif correct is False:
        status = f'FAIL (gt={gt_signal})'

    print(f"  -> {raw_sentiment:8s} => {prediction:6s} | {status} | {latency:.2f}s")

    results.append({
        'id':                  article['id'],
        'ticker':              ticker,
        'title':               title,
        'ground_truth_filtrage': gt_filtrage,
        'ground_truth_signal': gt_signal,
        'raw_sentiment':       raw_sentiment,
        'prediction':          prediction,
        'score':               score,
        'reasoning':           reasoning[:200] if reasoning else '',
        'latency_s':           latency,
        'correct':             correct,
    })

    time.sleep(0.3)

# ── Résumé ──────────────────────────────────────────────────────────────────────
acc = n_correct / n_pertinent if n_pertinent > 0 else 0.0
avg_lat = total_time / len(articles)

print(f"\n{'='*65}")
print(f"RÉSULTATS ANTOINE")
print(f"  Articles analysés     : {len(articles)}")
print(f"  Articles pertinents   : {n_pertinent}")
print(f"  Corrects              : {n_correct}/{n_pertinent} = {acc:.0%}")
print(f"  Temps total           : {total_time:.1f}s")
print(f"  Latence moyenne/art.  : {avg_lat:.2f}s")
print(f"{'='*65}\n")

# ── Export CSV ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
fields = ['id','ticker','title','ground_truth_filtrage','ground_truth_signal',
          'raw_sentiment','prediction','score','reasoning','latency_s','correct']
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"Resultats sauvegardes -> resultats/benchmark_antoine.csv ({len(results)} lignes)")
