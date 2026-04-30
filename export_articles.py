"""
export_articles.py
------------------
Exporte les articles de la DB vers articles.csv (données partagées).
Lance ce script une fois pour mettre à jour le CSV commun.

Usage :
    python export_articles.py
"""

import sqlite3
import csv
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'Projet-News-Agents', 'news_database.db')
OUT_PATH = os.path.join(os.path.dirname(__file__), 'articles.csv')

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT url, ticker, title, content, date_utc, source FROM articles ORDER BY date_utc DESC")
rows = cursor.fetchall()
conn.close()

with open(OUT_PATH, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['url', 'ticker', 'title', 'content', 'date_utc', 'source'])
    writer.writerows(rows)

print(f"{len(rows)} articles exportes -> articles.csv")
