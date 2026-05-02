"""
download.py — Télécharge le dataset SentFin v2
-----------------------------------------------
Dataset : yiyanghkust/finbert-tone  (SentFin v2)
Labels  : positive / neutral / negative  (sentiment financier)
Taille  : ~14 000 phrases financières

Sauvegarde :
    resultats/sentfin_v2.csv

Usage :
    python download.py
    python download.py --limit 2000   # limiter à N exemples

Prérequis :
    pip install datasets
"""

import csv
import os
import argparse
from datasets import load_dataset

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), 'resultats', 'sentfin_v2.csv')
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--limit', type=int, default=0, help='Limiter à N exemples (0 = tous)')
args = parser.parse_args()

print("\nTéléchargement SentFin v2 (yiyanghkust/finbert-tone)...")
ds = load_dataset("yiyanghkust/finbert-tone", split="train")

LABEL_MAP = {0: "neutral", 1: "positive", 2: "negative"}

rows = []
for item in ds:
    rows.append({
        'text':  item['text'],
        'label': LABEL_MAP.get(item['label'], str(item['label'])),
    })

if args.limit > 0:
    rows = rows[:args.limit]

with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['text', 'label'])
    writer.writeheader()
    writer.writerows(rows)

counts = {}
for r in rows:
    counts[r['label']] = counts.get(r['label'], 0) + 1

print(f"  {len(rows)} exemples sauvegardés -> {OUTPUT_FILE}")
for lbl, n in sorted(counts.items()):
    print(f"    {lbl:<10} : {n:>5} ({n/len(rows):.1%})")
