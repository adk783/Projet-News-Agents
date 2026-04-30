"""
adam_benchmark.py — Benchmark Adam (ABSA + debat multi-agents) sur les 25 articles
------------------------------------------------------------------------------------
Lit benchmark_dataset.json, lance le pipeline complet d'Adam
(filtrage → ABSA → debat → signal), mesure la latence.
Sauvegarde resultats/benchmark_adam.csv

Flow par article :
    Texte → agent_filtrage → agent_absa → agent_debat → signal (Achat/Vente/Neutre)

Usage :
    python adam_benchmark.py
    python adam_benchmark.py --limit 5   (test rapide)

Prerequis :
    - Cle API dans ../POC-Filtrage-Agents/.env :
        GROQ_API_KEY=gsk_...
    - pip install groq langchain langgraph python-dotenv
"""

import sys
import os
import csv
import json
import time
import argparse

# Pointer vers le dossier d'Adam
ADAM_PATH = os.path.join(os.path.dirname(__file__), '..', 'POC-Filtrage-Agents')
sys.path.insert(0, os.path.abspath(ADAM_PATH))

# Charger les cles API
from dotenv import load_dotenv
load_dotenv(os.path.join(ADAM_PATH, '.env'))

# Verifier que la cle est la
if not os.environ.get('GROQ_API_KEY'):
    print("[ERREUR] GROQ_API_KEY manquante.")
    print("  Cree le fichier : POC-Filtrage-Agents/.env")
    print("  Contenu :         GROQ_API_KEY=gsk_...")
    sys.exit(1)

from src.agents.agent_absa import run_absa
from src.agents.agent_debat import workflow_debat_actualite
from src.agents.agent_filtrage import workflow_filtrer_actualite

DATASET    = os.path.join(os.path.dirname(__file__), 'benchmark_dataset.json')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'resultats', 'benchmark_adam.csv')

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
print(f"BENCHMARK ADAM -- {len(articles)} articles (ABSA + debat LLM)")
print(f"{'='*65}\n")

# ── Analyse ─────────────────────────────────────────────────────────────────────
results     = []
total_time  = 0.0
n_pertinent = 0
n_correct   = 0

for i, article in enumerate(articles, 1):
    gt      = article['ground_truth']
    ticker  = article['ticker']
    title   = article['title']
    content = article['content']
    text    = f"{title}\n\n{content}"

    print(f"[{i:02d}/{len(articles):02d}] {article['id']} | {title[:55]}")

    t0 = time.time()
    filtrage_pred = None
    signal        = None
    argument      = None
    parsing_error = False
    pipeline_error = None

    try:
        # Etape 1 — Filtrage
        filtrage = workflow_filtrer_actualite.invoke({
            'texte_article':  text,
            'ticker_symbol':  ticker,
        })
        pertinent = filtrage.get('pertinent', True)
        filtrage_pred = 'pertinent' if pertinent else 'hors_scope'

        if not pertinent:
            signal = None
            print(f"  -> FILTRE (hors scope detecte par l'agent)")
        else:
            # Etape 2 — ABSA
            absa_result = run_absa(text, ticker)

            # Etape 3 — Debat
            decision = workflow_debat_actualite.invoke({
                'texte_article':   text,
                'ticker_symbol':   ticker,
                'contexte_marche': {},
                'absa_result':     absa_result,
            })

            signal   = decision.get('signal', 'Neutre')
            argument = decision.get('argument_dominant', '')

            if argument == 'Parsing impossible':
                parsing_error = True

            print(f"  -> {signal} | {argument[:60] if argument else ''}")

    except Exception as e:
        pipeline_error = f"{type(e).__name__}: {str(e)[:120]}"
        print(f"  [ERREUR] {pipeline_error}")

    latency = round(time.time() - t0, 2)
    total_time += latency

    gt_signal  = gt['signal']
    gt_filtrage = gt['filtrage']

    correct = None
    if gt_filtrage == 'pertinent' and gt_signal is not None and signal is not None and not parsing_error and not pipeline_error:
        n_pertinent += 1
        correct = (signal == gt_signal)
        if correct:
            n_correct += 1

    status = ''
    if gt_filtrage == 'hors_scope':
        status = 'HORS_SCOPE'
    elif pipeline_error:
        status = 'PIPELINE_ERROR'
    elif parsing_error:
        status = 'PARSING_ERROR'
    elif correct is True:
        status = 'OK'
    elif correct is False:
        status = f'FAIL (gt={gt_signal})'

    print(f"  {status} | {latency:.2f}s")

    results.append({
        'id':                    article['id'],
        'ticker':                ticker,
        'title':                 title,
        'ground_truth_filtrage': gt_filtrage,
        'ground_truth_signal':   gt_signal,
        'filtrage_pred':         filtrage_pred,
        'prediction':            signal,
        'argument':              (argument or '')[:200],
        'parsing_error':         parsing_error,
        'pipeline_error':        pipeline_error or '',
        'latency_s':             latency,
        'correct':               correct,
    })

    time.sleep(0.5)   # eviter le rate-limit Groq

# ── Resume ──────────────────────────────────────────────────────────────────────
acc     = n_correct / n_pertinent if n_pertinent > 0 else 0.0
avg_lat = total_time / len(articles)

print(f"\n{'='*65}")
print(f"RESULTATS ADAM")
print(f"  Articles analyses     : {len(articles)}")
print(f"  Articles pertinents   : {n_pertinent}")
print(f"  Corrects              : {n_correct}/{n_pertinent} = {acc:.0%}")
print(f"  Temps total           : {total_time:.1f}s")
print(f"  Latence moyenne/art.  : {avg_lat:.2f}s")
print(f"{'='*65}\n")

# ── Export CSV ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
fields = ['id','ticker','title','ground_truth_filtrage','ground_truth_signal',
          'filtrage_pred','prediction','argument','parsing_error',
          'pipeline_error','latency_s','correct']
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"Resultats sauvegardes -> resultats/benchmark_adam.csv ({len(results)} lignes)")
