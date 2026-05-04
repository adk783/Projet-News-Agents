"""
run_antoine.py — Teste Antoine (phi4-mini via Ollama) sur FNSPID
-----------------------------------------------------------------
1. Télécharge des articles depuis benstaf/FNSPID-nasdaq-100-post2019-1newsperrow
2. Calcule les labels Bullish/Neutral/Bearish depuis les prix réels (beachside1234/FNSPID)
3. Lance l'agent sentiment d'Antoine (Ollama phi4-mini)
4. Sauvegarde resultats/antoine_results.csv

Usage :
    python run_antoine.py
    python run_antoine.py --limit 50   (test rapide)

Prérequis :
    - Ollama lancé avec phi4-mini : ollama serve
    - pip install datasets huggingface_hub pandas requests
    - La branche Antoinev2 dans ../Antoinev2/
"""

import sys
import os
import csv
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Pointer vers Antoine
ANTOINE_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'Antoinev2')
sys.path.insert(0, os.path.abspath(ANTOINE_PATH))

from agent_sentiment import analyser_article

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'resultats', 'antoine_results.csv')
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

FORWARD_DAYS   = 2
THRESHOLD      = 0.01   # ±1% → Bullish / Bearish / Neutral

parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=150, help='Nb articles à tester (défaut=150)')
args = parser.parse_args()

# ── 1. Téléchargement des articles ──────────────────────────────────────────
print("Téléchargement des articles FNSPID...")
ds = load_dataset("benstaf/FNSPID-nasdaq-100-post2019-1newsperrow", split="train")
df = ds.to_pandas()
df = df.rename(columns={'Article_title': 'title', 'Stock_symbol': 'ticker', 'Article': 'text', 'Url': 'url'})
df['article_date'] = pd.to_datetime(df['Date'], errors='coerce', utc=True)
df = df.dropna(subset=['article_date', 'ticker', 'text']).reset_index(drop=True)
print(f"  {len(df)} articles disponibles")

# ── 2. Calcul des labels depuis les prix ────────────────────────────────────
_price_cache = {}

def load_price_history(symbol):
    if symbol in _price_cache:
        return _price_cache[symbol]
    try:
        path = hf_hub_download(
            "beachside1234/FNSPID",
            f"Stock_price/full_history/{symbol.upper()}.parquet",
            repo_type="dataset",
        )
        raw = pd.read_parquet(path)
        col = "adj close" if "adj close" in raw.columns else "close"
        raw['date'] = pd.to_datetime(raw['date'], errors='coerce', utc=True)
        price_df = raw.dropna(subset=['date', col]).sort_values('date')[['date', col]].rename(columns={col: 'price'})
        _price_cache[symbol] = price_df
        return price_df
    except Exception:
        _price_cache[symbol] = None
        return None

def compute_forward_return(symbol, article_date):
    price_df = load_price_history(symbol)
    if price_df is None or price_df.empty:
        return np.nan
    dates = price_df['date'].values.astype('datetime64[ns]')
    start_idx = int(np.searchsorted(dates, np.datetime64(article_date.replace(tzinfo=None)), side='left'))
    end_idx = start_idx + FORWARD_DAYS
    if start_idx < 0 or end_idx >= len(price_df):
        return np.nan
    start_price = float(price_df.iloc[start_idx]['price'])
    end_price   = float(price_df.iloc[end_idx]['price'])
    if start_price == 0 or np.isnan(start_price) or np.isnan(end_price):
        return np.nan
    return (end_price / start_price) - 1.0

def label_from_return(r):
    if r > THRESHOLD:  return 'Bullish'
    if r < -THRESHOLD: return 'Bearish'
    return 'Neutral'

# Calcul sur un échantillon équilibré
print("Calcul des labels (retours boursiers à 2 jours)...")
sample = df.sample(min(args.limit * 5, len(df)), random_state=42).copy()

labeled = []
for _, row in sample.iterrows():
    fwd = compute_forward_return(row['ticker'], row['article_date'])
    if not np.isnan(fwd):
        labeled.append({
            'ticker':  row['ticker'],
            'title':   row['title'],
            'text':    row['text'],
            'url':     row['url'],
            'date':    str(row['article_date'].date()),
            'forward_return':      round(fwd, 5),
            'market_impact_label': label_from_return(fwd),
        })
    if len(labeled) >= args.limit:
        break

print(f"  {len(labeled)} articles labellisés")
counts = {}
for a in labeled:
    counts[a['market_impact_label']] = counts.get(a['market_impact_label'], 0) + 1
for lbl, n in sorted(counts.items()):
    print(f"    {lbl:<10}: {n}")

# ── 3. Lancement d'Antoine ───────────────────────────────────────────────────
print(f"\nLancement Antoine (phi4-mini) sur {len(labeled)} articles...")
print(f"{'='*65}")

results = []
n_correct = 0
total_time = 0.0

LABEL_MAP = {'bullish': 'Bullish', 'bearish': 'Bearish', 'neutral': 'Neutral'}

for i, art in enumerate(labeled, 1):
    gt = art['market_impact_label']
    safe_title = art['title'].encode('ascii', errors='replace').decode('ascii')
    print(f"[{i:03d}/{len(labeled):03d}] {art['ticker']} | {safe_title[:50]}")

    t0 = time.time()
    try:
        res = analyser_article(
            url=art['url'],
            ticker=art['ticker'],
            title=art['title'],
            content=art['text'],
        )
        raw_sentiment = res['sentiment']
        prediction    = LABEL_MAP.get(raw_sentiment, raw_sentiment)
        reasoning     = res.get('reasoning', '')
        error         = None
    except Exception as e:
        raw_sentiment = 'error'
        prediction    = 'error'
        reasoning     = ''
        error         = str(e)

    latency = round(time.time() - t0, 3)
    total_time += latency

    correct = (prediction == gt) if prediction != 'error' else None
    if correct:
        n_correct += 1

    status = 'OK' if correct else (f'FAIL (gt={gt})' if correct is False else 'ERROR')
    print(f"  -> {raw_sentiment} => {prediction} | {status} | {latency:.2f}s")

    results.append({
        'ticker':              art['ticker'],
        'title':               art['title'],
        'date':                art['date'],
        'forward_return':      art['forward_return'],
        'ground_truth':        gt,
        'raw_sentiment':       raw_sentiment,
        'prediction':          prediction,
        'reasoning':           reasoning,
        'latency_s':           latency,
        'correct':             correct,
        'error':               error,
    })

# ── 4. Résumé ────────────────────────────────────────────────────────────────
n_valid = sum(1 for r in results if r['correct'] is not None)
acc     = n_correct / n_valid if n_valid > 0 else 0.0
avg_lat = total_time / len(results)

print(f"\n{'='*65}")
print(f"RÉSULTATS ANTOINE sur FNSPID")
print(f"  Articles testés       : {len(results)}")
print(f"  Corrects              : {n_correct}/{n_valid} = {acc:.1%}")
print(f"  Latence moyenne/art.  : {avg_lat:.2f}s")
print(f"  Temps total           : {total_time:.1f}s")
print(f"{'='*65}")

# ── 5. Export CSV ────────────────────────────────────────────────────────────
fields = ['ticker','title','date','forward_return','ground_truth','raw_sentiment',
          'prediction','reasoning','latency_s','correct','error']
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results)

print(f"\nRésultats -> {OUTPUT_FILE} ({len(results)} lignes)")
