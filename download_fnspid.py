"""
download_fnspid.py — Télécharge le dataset FNSPID (benchmark sortie finale)
---------------------------------------------------------------------------
Dataset : benstaf/FNSPID-nasdaq-100-post2019-1newsperrow
Labels  : Bullish / Neutral / Bearish  (retours boursiers réels à 2 jours)
          Bullish  = forward_return > +1%
          Bearish  = forward_return < -1%
          Neutral  = sinon

Sauvegarde :
    resultats/fnspid_validation.json   (2400 articles, split validation)
    resultats/fnspid_train.json        (9600 articles, split train)  [optionnel]

Usage :
    python download_fnspid.py               # validation seulement
    python download_fnspid.py --split train # train seulement
    python download_fnspid.py --split all   # les deux
    python download_fnspid.py --limit 200   # limiter à N articles

Prérequis :
    pip install datasets
"""

import json
import os
import argparse
from datasets import load_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'resultats')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_NAME = "benstaf/FNSPID-nasdaq-100-post2019-1newsperrow"

parser = argparse.ArgumentParser()
parser.add_argument('--split', choices=['validation', 'train', 'all'], default='validation')
parser.add_argument('--limit', type=int, default=0, help='Limiter à N articles (0 = tous)')
args = parser.parse_args()

print(f"\n{'='*60}")
print(f"Téléchargement FNSPID depuis HuggingFace")
print(f"  {DATASET_NAME}")
print(f"{'='*60}\n")

ds = load_dataset(DATASET_NAME)
print(f"Splits disponibles : {list(ds.keys())}")


def save_split(split_name, out_file):
    data = ds[split_name]
    articles = []
    for row in data:
        articles.append({
            'ticker':              row.get('ticker', ''),
            'title':               row.get('title', ''),
            'text':                row.get('article', row.get('summary', row.get('text', ''))),
            'forward_return':      row.get('forward_return', None),
            'market_impact_label': row.get('market_impact_label', None),  # Bullish/Neutral/Bearish
        })

    if args.limit > 0:
        articles = articles[:args.limit]

    out = {
        'source':      DATASET_NAME,
        'split':       split_name,
        'total':       len(articles),
        'label_info':  'Bullish=retour>1%, Bearish=retour<-1%, Neutral=sinon (à 2 jours)',
        'articles':    articles,
    }

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Résumé distribution
    counts = {}
    for a in articles:
        lbl = a['market_impact_label'] or 'N/A'
        counts[lbl] = counts.get(lbl, 0) + 1

    print(f"  [{split_name}] {len(articles)} articles -> {out_file}")
    for lbl, n in sorted(counts.items()):
        print(f"    {lbl:<10} : {n:>5} ({n/len(articles):.1%})")


splits_to_save = []
if args.split == 'all':
    splits_to_save = [
        ('train',      os.path.join(OUTPUT_DIR, 'fnspid_train.json')),
        ('validation', os.path.join(OUTPUT_DIR, 'fnspid_validation.json')),
    ]
elif args.split == 'train':
    splits_to_save = [('train', os.path.join(OUTPUT_DIR, 'fnspid_train.json'))]
else:
    splits_to_save = [('validation', os.path.join(OUTPUT_DIR, 'fnspid_validation.json'))]

for split_name, out_file in splits_to_save:
    save_split(split_name, out_file)

print(f"\nFait. Adam peut maintenant utiliser ces fichiers pour tester son pipeline.")
