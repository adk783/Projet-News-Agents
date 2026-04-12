from flask import Flask, render_template, jsonify, request
import sqlite3
import subprocess
import sys
import os
import webbrowser
import json

app = Flask(__name__)
DB_PATH = "news_database.db"

# ── helpers ────────────────────────────────────────────────────────────────────
def query(sql, params=()):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows
    except sqlite3.OperationalError:
        return []
# ── routes API ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/api/ticker_scores")
def ticker_scores():
    rows = query("""
        SELECT ticker, score_global, sentiment_global, nb_articles, nb_neutral, confidence, calculated_at
        FROM ticker_scores
        ORDER BY calculated_at DESC
    """)
    # Garder uniquement le score le plus récent par ticker
    seen = {}
    for r in rows:
        if r["ticker"] not in seen:
            seen[r["ticker"]] = r
    return jsonify(list(seen.values()))

@app.route("/api/article_scores")
def article_scores():
    ticker = request.args.get("ticker", None)
    if ticker:
        rows = query("""
            SELECT a.title, s.ticker, s.sentiment, s.score, s.reasoning, s.analyzed_at
            FROM article_scores s
            JOIN articles a ON a.url = s.url
            WHERE s.ticker = ?
            ORDER BY s.analyzed_at DESC
        """, (ticker,))
    else:
        rows = query("""
            SELECT a.title, s.ticker, s.sentiment, s.score, s.reasoning, s.analyzed_at
            FROM article_scores s
            JOIN articles a ON a.url = s.url
            ORDER BY s.analyzed_at DESC
            LIMIT 50
        """)
    return jsonify(rows)

@app.route("/api/ticker_history")
def ticker_history():
    ticker = request.args.get("ticker")
    if not ticker:
        return jsonify([])
    rows = query("""
        SELECT score_global, sentiment_global, nb_articles, calculated_at
        FROM ticker_scores
        WHERE ticker = ?
        ORDER BY calculated_at ASC
    """, (ticker,))
    return jsonify(rows)

@app.route("/api/sentiment_distribution")
def sentiment_distribution():
    ticker = request.args.get("ticker", None)
    if ticker:
        rows = query("""
            SELECT sentiment, COUNT(*) as count
            FROM article_scores
            WHERE ticker = ?
            GROUP BY sentiment
        """, (ticker,))
    else:
        rows = query("""
            SELECT sentiment, COUNT(*) as count
            FROM article_scores
            GROUP BY sentiment
        """)
    return jsonify(rows)

@app.route("/api/tickers_list")
def tickers_list():
    rows = query("SELECT DISTINCT ticker FROM articles ORDER BY ticker")
    return jsonify([r["ticker"] for r in rows])

@app.route("/api/run_pipeline", methods=["POST"])
def run_pipeline():
    data     = request.json
    tickers  = data.get("tickers", [])
    loop     = data.get("loop", None)
    pipeline = data.get("pipeline", "sourcing")  # "sourcing" ou "orchestrateur"

    if not tickers:
        return jsonify({"error": "Aucun ticker sélectionné"}), 400

    if pipeline == "sourcing":
        cmd = [sys.executable, "news_pipeline.py", "--tickers"] + tickers
        if loop:
            cmd += ["--loop", str(loop)]
    else:
        cmd = [sys.executable, "orchestrateur.py"]

    try:
        subprocess.Popen(cmd, cwd=os.getcwd())
        return jsonify({"status": "ok", "message": f"{pipeline} lancé avec {tickers}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats")
def stats():
    # Vérifier quelles tables existent
    tables = [r["name"] for r in query("SELECT name FROM sqlite_master WHERE type='table'")]
    
    if "articles" not in tables:
        return jsonify({"total_articles": 0, "analysed": 0, "pending": 0, "scores": 0})
    
    colonnes = [row["name"] for row in query("PRAGMA table_info(articles)")]
    total    = query("SELECT COUNT(*) as n FROM articles")[0]["n"]
    analysed = query("SELECT COUNT(*) as n FROM articles WHERE is_analyzed=1")[0]["n"] if "is_analyzed" in colonnes else 0
    pending  = total - analysed
    scores   = query("SELECT COUNT(*) as n FROM article_scores")[0]["n"] if "article_scores" in tables else 0

    return jsonify({"total_articles": total, "analysed": analysed, "pending": pending, "scores": scores})

@app.route("/api/pipeline_status")
def pipeline_status():
    if not os.path.exists("pipeline_status.json"):
        return jsonify({"sourcing": {"running": False}, "orchestrateur": {"running": False}})
    with open("pipeline_status.json", "r") as f:
        return jsonify(json.load(f))

if __name__ == "__main__":
    webbrowser.open("http://localhost:5000")
    app.run(debug=True, port=5000, use_reloader=False)