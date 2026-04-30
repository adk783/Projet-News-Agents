"""
samuel_benchmark.py — Benchmark Samuel (FinBERT + Market Impact Classifier)
-----------------------------------------------------------------------------
Lit benchmark_dataset.json, lance les 4 agents FinBERT puis le Market Impact
Classifier sur chaque article, mesure la latence.
Sauvegarde resultats/benchmark_samuel.csv

Flow par article :
    Texte → PolarityAgent + UncertaintyAgent + LitigiousAgent + FundamentalStrengthAgent
          → features dérivées + KMeans distances
          → MarketImpactClassifier → Bullish/Neutral/Bearish → Achat/Vente/Neutre

Usage :
    python samuel_benchmark.py
    python samuel_benchmark.py --limit 5   (test rapide)

Prérequis :
    - pip install transformers torch peft scikit-learn pandas joblib numpy
    - Modèles dans ../samuel/ : uncertainty_model/, litigious_model/,
      fundamental_strength_model/, market_impact_model/
"""

import sys
import os
import csv
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')  # Silence sklearn version warnings

import numpy as np
import pandas as pd
from joblib import load

# Pointer vers le dossier de Samuel
SAMUEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'samuel')
sys.path.insert(0, os.path.abspath(SAMUEL_PATH))

from polarity_agent          import PolarityAgent
from uncertainty_agent       import UncertaintyAgent
from litigious_agent         import LitigiousAgent
from fundamental_strength_agent import FundamentalStrengthAgent

DATASET    = os.path.join(os.path.dirname(__file__), 'benchmark_dataset.json')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), 'resultats', 'benchmark_samuel.csv')
MODEL_DIR  = os.path.join(os.path.abspath(SAMUEL_PATH), 'market_impact_model')

# Mapping Market Impact → label commun
LABEL_MAP = {
    'Bullish': 'Achat',
    'Neutral': 'Neutre',
    'Bearish': 'Vente',
}

BASE_FEATURE_COLS = [
    'polarity', 'polarity_conf', 'uncertainty', 'litigious', 'fundamental_strength',
    'risk_adjusted_sentiment', 'headline_conviction', 'fundamental_impact',
    'business_quality_score', 'risk_pressure', 'market_signal_score',
]

# ── Chargement des agents (une seule fois) ──────────────────────────────────────
print("Chargement des agents Samuel...")

polarity_agent = PolarityAgent()
print("  PolarityAgent OK")

uncertainty_agent = None
unc_path = os.path.join(SAMUEL_PATH, 'uncertainty_model')
if os.path.exists(unc_path):
    uncertainty_agent = UncertaintyAgent(model_path=unc_path)
    print("  UncertaintyAgent OK")
else:
    print("  UncertaintyAgent ABSENT (uncertainty = 0.0)")

litigious_agent = LitigiousAgent(
    model_path=os.path.join(SAMUEL_PATH, 'litigious_model'),
    fallback_to_heuristic=True
)
print("  LitigiousAgent OK")

fundamental_agent = FundamentalStrengthAgent(
    model_path=os.path.join(SAMUEL_PATH, 'fundamental_strength_model'),
    fallback_to_heuristic=True
)
print("  FundamentalStrengthAgent OK")

# ── Chargement du Market Impact Classifier ──────────────────────────────────────
classifier        = None
kmeans_obj        = None
cluster_scaler    = None
use_market_impact = False

clf_path = os.path.join(MODEL_DIR, 'classifier.joblib')
km_path  = os.path.join(MODEL_DIR, 'kmeans.joblib')

if os.path.exists(clf_path) and os.path.exists(km_path):
    classifier = load(clf_path)
    # kmeans.joblib est un dict : {'scaler': ..., 'kmeans': ..., ...}
    km_bundle  = load(km_path)
    if isinstance(km_bundle, dict):
        kmeans_obj     = km_bundle['kmeans']
        cluster_scaler = km_bundle['scaler']
    else:
        # fallback au cas où c'est l'objet KMeans directement
        kmeans_obj = km_bundle
    use_market_impact = True
    print("  MarketImpactClassifier OK")
else:
    print("  MarketImpactClassifier ABSENT -> fallback sur polarity directe")


def build_derived_features(polarity, polarity_conf, uncertainty, litigious, fundamental_strength):
    """Calcule les features dérivées identiques à investment_processing_pipeline.py"""
    risk_adj_sent      = polarity * polarity_conf
    headline_conv      = polarity_conf * (1.0 - uncertainty)
    fundamental_impact = fundamental_strength * risk_adj_sent
    business_quality   = 0.70 * fundamental_impact + 0.30 * headline_conv
    risk_pressure      = 0.55 * uncertainty + 0.45 * litigious
    market_signal      = (1.00 * risk_adj_sent
                          + 1.10 * fundamental_impact
                          + 0.20 * headline_conv
                          - 0.85 * uncertainty
                          - 0.35 * litigious)
    return {
        'risk_adjusted_sentiment': risk_adj_sent,
        'headline_conviction':     headline_conv,
        'fundamental_impact':      fundamental_impact,
        'business_quality_score':  business_quality,
        'risk_pressure':           risk_pressure,
        'market_signal_score':     market_signal,
    }


def predict_market_impact(text, ticker, polarity, polarity_conf, uncertainty,
                           litigious, fundamental_strength):
    """
    Applique le Market Impact Classifier et retourne 'Bullish'/'Neutral'/'Bearish'.
    Fallback sur polarity si le classifier est absent.
    """
    if not use_market_impact:
        if polarity > 0:   return 'Bullish'
        if polarity < 0:   return 'Bearish'
        return 'Neutral'

    derived = build_derived_features(polarity, polarity_conf, uncertainty, litigious, fundamental_strength)

    # Distances KMeans — on applique le même scaler qu'à l'entraînement
    base_vec = np.array([[polarity, polarity_conf, uncertainty, litigious, fundamental_strength,
                          derived['risk_adjusted_sentiment'], derived['headline_conviction'],
                          derived['fundamental_impact'], derived['business_quality_score'],
                          derived['risk_pressure'], derived['market_signal_score']]])
    if cluster_scaler is not None:
        base_vec = cluster_scaler.transform(base_vec)
    cluster_dists = kmeans_obj.transform(base_vec)[0]

    row_df = pd.DataFrame([{
        'text':                    text,
        'source_ticker':           ticker,
        'polarity':                polarity,
        'polarity_conf':           polarity_conf,
        'uncertainty':             uncertainty,
        'litigious':               litigious,
        'fundamental_strength':    fundamental_strength,
        **derived,
        'cluster_distance_0':      cluster_dists[0],
        'cluster_distance_1':      cluster_dists[1],
        'cluster_distance_2':      cluster_dists[2],
    }])

    return classifier.predict(row_df)[0]   # 'Bullish' / 'Neutral' / 'Bearish'


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
print(f"BENCHMARK SAMUEL — {len(articles)} articles (FinBERT + Market Impact)")
print(f"  Market Impact Classifier : {'actif' if use_market_impact else 'ABSENT (fallback polarity)'}")
print(f"{'='*65}\n")

# ── Analyse ─────────────────────────────────────────────────────────────────────
results     = []
total_time  = 0.0
n_pertinent = 0
n_correct   = 0

for i, article in enumerate(articles, 1):
    gt      = article['ground_truth']
    text    = f"{article['title']}\n\n{article['content']}"[:1500]
    ticker  = article['ticker']
    title   = article['title']

    print(f"[{i:02d}/{len(articles):02d}] {article['id']} | {title[:55]}")

    t0 = time.time()
    try:
        # 4 agents FinBERT
        polarity, polarity_conf, polarity_label = polarity_agent.predict(text)
        uncertainty          = float(uncertainty_agent.predict(text)) if uncertainty_agent else 0.0
        litigious            = float(litigious_agent.predict(text))
        fundamental_strength = float(fundamental_agent.predict(text))

        # Market Impact Classifier
        market_impact_raw = predict_market_impact(
            text, ticker, polarity, polarity_conf,
            uncertainty, litigious, fundamental_strength
        )
        prediction = LABEL_MAP.get(market_impact_raw, 'Neutre')

    except Exception as e:
        print(f"  [ERREUR] {e}")
        polarity = polarity_conf = uncertainty = litigious = fundamental_strength = None
        polarity_label = 'error'
        market_impact_raw = 'error'
        prediction = 'error'

    latency = round(time.time() - t0, 3)
    total_time += latency

    gt_signal  = gt['signal']
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

    pol_str = f"pol={polarity}({polarity_label[:3]}) unc={uncertainty:.2f} lit={litigious:.2f} fund={fundamental_strength:.2f}" \
              if polarity is not None else "error"
    print(f"  -> {pol_str}")
    print(f"     => {market_impact_raw} -> {prediction} | {status} | {latency:.2f}s")

    results.append({
        'id':                    article['id'],
        'ticker':                ticker,
        'title':                 title,
        'ground_truth_filtrage': gt_filtrage,
        'ground_truth_signal':   gt_signal,
        'polarity':              polarity,
        'polarity_conf':         round(polarity_conf, 4) if polarity_conf else None,
        'polarity_label':        polarity_label,
        'uncertainty':           round(uncertainty, 4) if uncertainty is not None else None,
        'litigious':             round(litigious, 4) if litigious is not None else None,
        'fundamental_strength':  round(fundamental_strength, 4) if fundamental_strength is not None else None,
        'market_impact_raw':     market_impact_raw,
        'prediction':            prediction,
        'latency_s':             latency,
        'correct':               correct,
    })

# ── Résumé ──────────────────────────────────────────────────────────────────────
acc     = n_correct / n_pertinent if n_pertinent > 0 else 0.0
avg_lat = total_time / len(articles)

print(f"\n{'='*65}")
print(f"RÉSULTATS SAMUEL")
print(f"  Articles analysés     : {len(articles)}")
print(f"  Articles pertinents   : {n_pertinent}")
print(f"  Corrects              : {n_correct}/{n_pertinent} = {acc:.0%}")
print(f"  Temps total           : {total_time:.1f}s")
print(f"  Latence moyenne/art.  : {avg_lat:.2f}s")
print(f"{'='*65}\n")

# ── Export CSV ──────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
fields = ['id','ticker','title','ground_truth_filtrage','ground_truth_signal',
          'polarity','polarity_conf','polarity_label','uncertainty','litigious',
          'fundamental_strength','market_impact_raw','prediction','latency_s','correct']
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"Resultats sauvegardes -> resultats/benchmark_samuel.csv ({len(results)} lignes)")
