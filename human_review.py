"""
Human-in-the-Loop Review Interface
===================================
Interface web locale pour vérifier manuellement les scores d'incertitude.

Usage :
    python human_review.py
    → Ouvre http://localhost:5555 dans le navigateur

Fonctionnalités :
    - Voir chaque article avec son score d'incertitude
    - Approuver / Modifier / Rejeter le score
    - Naviguer entre articles (suivant/précédent/aléatoire)
    - Dashboard de stats en temps réel
    - Exporter les résultats révisés en CSV
"""

import json
import sqlite3
import os
import random
import csv
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import webbrowser
import threading

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "news_database.db")
PORT = 5555


def get_articles():
    """Charge tous les articles avec leurs scores."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS human_reviews (
            url TEXT PRIMARY KEY,
            human_status TEXT,
            human_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute("""
        SELECT a.url, a.title, a.ticker, a.content, s.polarity, s.polarity_conf, s.uncertainty,
               hr.human_status, hr.human_score
        FROM article_scores s
        JOIN articles a ON s.url = a.url
        LEFT JOIN human_reviews hr ON s.url = hr.url
        ORDER BY s.uncertainty DESC
    """)
    rows = c.fetchall()
    conn.close()
    articles = []
    for r in rows:
        articles.append({
            "url": r[0],
            "title": r[1],
            "ticker": r[2],
            "content": r[3][:2000] if r[3] else "",
            "polarity": r[4],
            "polarity_conf": r[5],
            "uncertainty": r[6],
            "human_status": r[7] if r[7] else "pending",
            "human_score": r[8] if r[7] else None,
        })
    return articles


def get_stats(articles):
    """Calcule les stats pour le dashboard."""
    scores = [a["uncertainty"] for a in articles]
    reviewed = [a for a in articles if a["human_status"] != "pending"]
    approved = [a for a in articles if a["human_status"] == "approved"]
    modified = [a for a in articles if a["human_status"] == "modified"]
    rejected = [a for a in articles if a["human_status"] == "rejected"]

    import statistics
    return {
        "total": len(articles),
        "reviewed": len(reviewed),
        "approved": len(approved),
        "modified": len(modified),
        "rejected": len(rejected),
        "pending": len(articles) - len(reviewed),
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "mean_score": statistics.mean(scores) if scores else 0,
        "stdev_score": statistics.stdev(scores) if len(scores) > 1 else 0,
        "progress_pct": round(100 * len(reviewed) / max(len(articles), 1)),
    }


HTML_PAGE = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HITL Review — Uncertainty Agent</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #f0f0f0;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --bg-hover: #e0e0e0;
            --accent: #222222;
            --accent-light: #555555;
            --green: #10b981;
            --green-dim: rgba(16, 185, 129, 0.1);
            --red: #ef4444;
            --red-dim: rgba(239, 68, 68, 0.1);
            --orange: #f59e0b;
            --orange-dim: rgba(245, 158, 11, 0.1);
            --blue: #3b82f6;
            --text-primary: #000000;
            --text-secondary: #333333;
            --text-dim: #555555;
            --border: rgba(0,0,0,0.15);
            --glow: rgba(0,0,0,0);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
        }

        /* ─── Header ─── */
        .header {
            background: linear-gradient(135deg, var(--bg-secondary), var(--bg-card));
            border-bottom: 1px solid var(--border);
            padding: 16px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            position: sticky;
            top: 0;
            z-index: 100;
            backdrop-filter: blur(12px);
        }
        .header h1 {
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent), var(--green));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header-stats {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        .stat-chip {
            background: rgba(255,255,255,0.05);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
            color: var(--text-secondary);
        }
        .stat-chip .val { color: var(--accent-light); font-weight: 700; }
        .stat-chip.green .val { color: var(--green); }
        .stat-chip.orange .val { color: var(--orange); }
        .stat-chip.red .val { color: var(--red); }

        /* ─── Progress Bar ─── */
        .progress-bar-container {
            width: 200px;
            height: 6px;
            background: rgba(255,255,255,0.08);
            border-radius: 3px;
            overflow: hidden;
        }
        .progress-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--green));
            border-radius: 3px;
            transition: width 0.5s ease;
        }

        /* ─── Main Layout ─── */
        .main {
            display: grid;
            grid-template-columns: 1fr 350px;
            gap: 0;
            height: calc(100vh - 64px);
        }

        /* ─── Article Panel ─── */
        .article-panel {
            padding: 32px;
            overflow-y: auto;
        }
        .nav-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 24px;
        }
        .nav-bar .counter {
            font-size: 14px;
            color: var(--text-secondary);
        }
        .nav-bar .counter strong {
            color: var(--accent-light);
        }
        .nav-btns {
            display: flex;
            gap: 8px;
        }
        .nav-btn {
            background: var(--bg-card);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            padding: 8px 16px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: all 0.2s;
        }
        .nav-btn:hover {
            background: var(--bg-hover);
            color: var(--text-primary);
            border-color: var(--accent);
        }

        /* ─── Article Card ─── */
        .article-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 28px;
            margin-bottom: 24px;
            transition: all 0.3s;
        }
        .article-card:hover {
            border-color: var(--accent);
            box-shadow: 0 4px 30px var(--glow);
        }
        .article-title {
            font-size: 18px;
            font-weight: 700;
            margin-bottom: 8px;
            line-height: 1.4;
        }
        .article-meta {
            display: flex;
            gap: 16px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        .meta-tag {
            font-size: 12px;
            padding: 4px 10px;
            border-radius: 6px;
            font-weight: 600;
        }
        .meta-tag.ticker {
            background: rgba(66,165,245,0.15);
            color: var(--blue);
        }
        .meta-tag.polarity-positive {
            background: var(--green-dim);
            color: var(--green);
        }
        .meta-tag.polarity-negative {
            background: var(--red-dim);
            color: var(--red);
        }
        .meta-tag.polarity-neutral {
            background: rgba(144,144,168,0.15);
            color: var(--text-secondary);
        }
        .article-content {
            font-size: 14px;
            line-height: 1.7;
            color: var(--text-secondary);
            max-height: 300px;
            overflow-y: auto;
            padding: 16px;
            background: rgba(0,0,0,0.2);
            border-radius: 10px;
            margin-bottom: 20px;
        }

        /* ─── Score Display ─── */
        .score-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }
        .score-box {
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
        }
        .score-box .label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-dim);
            margin-bottom: 6px;
        }
        .score-box .value {
            font-size: 28px;
            font-weight: 800;
        }
        .score-box .value.low { color: var(--green); }
        .score-box .value.medium { color: var(--orange); }
        .score-box .value.high { color: var(--red); }

        .uncertainty-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.08);
            border-radius: 4px;
            margin-top: 8px;
            overflow: hidden;
        }
        .uncertainty-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }

        /* ─── Action Buttons ─── */
        .actions {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 12px;
            margin-bottom: 16px;
        }
        .action-btn {
            padding: 14px;
            border-radius: 12px;
            border: 2px solid transparent;
            cursor: pointer;
            font-size: 14px;
            font-weight: 700;
            transition: all 0.3s;
            text-align: center;
        }
        .action-btn.approve {
            background: var(--green-dim);
            color: var(--green);
            border-color: rgba(0,212,170,0.3);
        }
        .action-btn.approve:hover {
            background: rgba(0,212,170,0.25);
            border-color: var(--green);
            box-shadow: 0 0 20px rgba(0,212,170,0.2);
        }
        .action-btn.modify {
            background: var(--orange-dim);
            color: var(--orange);
            border-color: rgba(255,167,38,0.3);
        }
        .action-btn.modify:hover {
            background: rgba(255,167,38,0.25);
            border-color: var(--orange);
        }
        .action-btn.reject {
            background: var(--red-dim);
            color: var(--red);
            border-color: rgba(255,107,107,0.3);
        }
        .action-btn.reject:hover {
            background: rgba(255,107,107,0.25);
            border-color: var(--red);
        }
        .action-btn.active {
            transform: scale(1.03);
        }
        .action-btn.approve.active { border-color: var(--green); box-shadow: 0 0 20px rgba(0,212,170,0.3); }
        .action-btn.modify.active { border-color: var(--orange); box-shadow: 0 0 20px rgba(255,167,38,0.3); }
        .action-btn.reject.active { border-color: var(--red); box-shadow: 0 0 20px rgba(255,107,107,0.3); }

        /* ─── Score Slider ─── */
        .slider-section {
            display: none;
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }
        .slider-section.visible { display: block; }
        .slider-section label {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 8px;
            display: block;
        }
        .slider-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .slider-row input[type="range"] {
            width: 100%;
            accent-color: var(--accent);
            height: 6px;
        }
        .slider-val {
            font-size: 18px;
            font-weight: 800;
            color: var(--accent-light);
            min-width: 50px;
            text-align: right;
        }

        /* ─── Side Panel (Stats Dashboard) ─── */
        .side-panel {
            background: var(--bg-secondary);
            border-left: 1px solid var(--border);
            padding: 24px;
            overflow-y: auto;
        }
        .side-title {
            font-size: 16px;
            font-weight: 700;
            margin-bottom: 20px;
            color: var(--text-primary);
        }
        .stat-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 12px;
        }
        .stat-card .stat-label {
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-dim);
            margin-bottom: 4px;
        }
        .stat-card .stat-value {
            font-size: 24px;
            font-weight: 800;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin-bottom: 16px;
        }
        .mini-stat {
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }
        .mini-stat .ms-val {
            font-size: 18px;
            font-weight: 700;
        }
        .mini-stat .ms-label {
            font-size: 10px;
            color: var(--text-dim);
            margin-top: 2px;
        }

        /* ─── Distribution mini chart ─── */
        .dist-chart {
            display: flex;
            align-items: flex-end;
            gap: 3px;
            height: 80px;
            margin-top: 12px;
        }
        .dist-bar {
            flex: 1;
            border-radius: 3px 3px 0 0;
            transition: height 0.3s;
            position: relative;
        }
        .dist-bar:hover::after {
            content: attr(data-count);
            position: absolute;
            top: -20px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 10px;
            font-weight: 700;
            color: var(--text-primary);
            background: var(--bg-card);
            padding: 2px 6px;
            border-radius: 4px;
        }
        .dist-labels {
            display: flex;
            gap: 3px;
            margin-top: 4px;
        }
        .dist-labels span {
            flex: 1;
            text-align: center;
            font-size: 9px;
            color: var(--text-dim);
        }

        /* ─── Export Button ─── */
        .export-btn {
            width: 100%;
            padding: 14px;
            background: linear-gradient(135deg, var(--accent), var(--accent-light));
            border: none;
            border-radius: 12px;
            color: white;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.3s;
            margin-top: 16px;
        }
        .export-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px var(--glow);
        }

        /* ─── Article List in Side Panel ─── */
        .article-list {
            margin-top: 16px;
        }
        .article-list-item {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 10px;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.2s;
            margin-bottom: 4px;
        }
        .article-list-item:hover { background: var(--bg-hover); }
        .article-list-item.current { background: var(--bg-hover); border: 1px solid var(--border); }
        .article-list-item .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .article-list-item .dot.pending { background: var(--text-dim); }
        .article-list-item .dot.approved { background: var(--green); }
        .article-list-item .dot.modified { background: var(--orange); }
        .article-list-item .dot.rejected { background: var(--red); }
        .article-list-item .ali-title {
            font-size: 12px;
            color: var(--text-secondary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            flex: 1;
        }
        .article-list-item .ali-score {
            font-size: 11px;
            font-weight: 700;
            color: var(--text-dim);
            flex-shrink: 0;
        }

        /* ─── Scrollbar Styling ─── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--accent); }

        /* ─── Animations ─── */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .article-card { animation: fadeIn 0.3s ease; }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
    </style>
</head>
<body>

<div class="header">
    <h1>Human-in-the-Loop Review</h1>
    <div class="header-stats">
        <div class="stat-chip green">
            Approuvés: <span class="val" id="hdr-approved">0</span>
        </div>
        <div class="stat-chip orange">
            Modifiés: <span class="val" id="hdr-modified">0</span>
        </div>
        <div class="stat-chip red">
            Rejetés: <span class="val" id="hdr-rejected">0</span>
        </div>
        <div class="stat-chip">
            <span class="val" id="hdr-progress">0</span>% complété
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar-fill" id="hdr-progress-bar" style="width: 0%"></div>
        </div>
    </div>
</div>

<div class="main">
    <!-- Article Panel -->
    <div class="article-panel">
        <div class="nav-bar">
            <div class="counter">
                Article <strong id="current-idx">1</strong> / <strong id="total-count">0</strong>
                <span id="status-badge" style="margin-left: 10px; font-size:12px;"></span>
            </div>
            <div class="nav-btns">
                <button class="nav-btn" onclick="navigate(-1)">← Précédent</button>
                <button class="nav-btn" onclick="navigateRandom()">Aléatoire</button>
                <button class="nav-btn" onclick="navigate(1)">Suivant →</button>
            </div>
        </div>

        <div class="article-card" id="article-card">
            <div class="article-title" id="article-title">Chargement...</div>
            <div class="article-meta" id="article-meta"></div>
            <div class="article-content" id="article-content"></div>

            <div class="score-section">
                <div class="score-box">
                    <div class="label">Score d'incertitude (modèle)</div>
                    <div class="value" id="model-score">—</div>
                    <div class="uncertainty-bar">
                        <div class="uncertainty-bar-fill" id="unc-bar" style="width: 0%"></div>
                    </div>
                </div>
                <div class="score-box">
                    <div class="label">Confiance polarité</div>
                    <div class="value" id="pol-conf" style="color: var(--blue)">—</div>
                </div>
            </div>

            <div class="actions">
                <button class="action-btn approve" onclick="setAction('approved')">
                    Le modele a raison
                </button>
                <button class="action-btn modify" onclick="setAction('modified')">
                    Je veux corriger
                </button>
                <button class="action-btn reject" onclick="setAction('rejected')">
                    Rejeter l'article
                </button>
            </div>

            <div class="slider-section" id="slider-section">
                <label>Score d'incertitude corrigé :</label>
                <p style="font-size:11px; margin-bottom:12px; font-style:italic; color:#555;">
                   (Sauvegarde auto. Cliquez juste sur Suivant au lieu de valider apres.)
                </p>
                <div class="slider-row">
                    <span style="font-size:12px; color:var(--green)">Factuel</span>
                    <input type="range" min="0" max="100" value="50" id="score-slider"
                           oninput="updateSlider()">
                    <span style="font-size:12px; color:var(--red)">Incertain</span>
                    <span class="slider-val" id="slider-val">0.50</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Side Panel -->
    <div class="side-panel">
        <div class="side-title">Dashboard</div>

        <div class="stat-card">
            <div class="stat-label">Distribution des scores</div>
            <div class="dist-chart" id="dist-chart"></div>
            <div class="dist-labels" id="dist-labels"></div>
        </div>

        <div class="stat-grid">
            <div class="mini-stat">
                <div class="ms-val" style="color:var(--green)" id="stat-min">—</div>
                <div class="ms-label">Min Score</div>
            </div>
            <div class="mini-stat">
                <div class="ms-val" style="color:var(--red)" id="stat-max">—</div>
                <div class="ms-label">Max Score</div>
            </div>
            <div class="mini-stat">
                <div class="ms-val" style="color:var(--accent-light)" id="stat-mean">—</div>
                <div class="ms-label">Moyenne</div>
            </div>
            <div class="mini-stat">
                <div class="ms-val" style="color:var(--orange)" id="stat-stdev">—</div>
                <div class="ms-label">Écart-type</div>
            </div>
        </div>

        <div class="side-title">Liste des articles</div>
        <div class="article-list" id="article-list"></div>

        <button class="export-btn" onclick="exportCSV()">
            Exporter les résultats (CSV)
        </button>
    </div>
</div>

<script>
let articles = [];
let currentIdx = 0;

// ─── Load Data ───
async function loadData() {
    const resp = await fetch('/api/articles');
    articles = await resp.json();
    document.getElementById('total-count').textContent = articles.length;
    renderArticle();
    renderList();
    updateStats();
}

// ─── Render Article ───
function renderArticle() {
    const a = articles[currentIdx];
    document.getElementById('current-idx').textContent = currentIdx + 1;
    document.getElementById('article-title').textContent = a.title;
    document.getElementById('article-content').textContent = a.content;

    // Meta tags
    const metaDiv = document.getElementById('article-meta');
    const polLabel = a.polarity === 1 ? 'Positif' : a.polarity === -1 ? 'Négatif' : 'Neutre';
    const polClass = a.polarity === 1 ? 'polarity-positive' : a.polarity === -1 ? 'polarity-negative' : 'polarity-neutral';
    metaDiv.innerHTML = `
        <span class="meta-tag ticker">${a.ticker}</span>
        <span class="meta-tag ${polClass}">${polLabel} (${(a.polarity_conf * 100).toFixed(0)}%)</span>
    `;

    // Score
    const scoreEl = document.getElementById('model-score');
    scoreEl.textContent = a.uncertainty.toFixed(4);
    scoreEl.className = 'value ' + (a.uncertainty < 0.3 ? 'low' : a.uncertainty < 0.6 ? 'medium' : 'high');

    const barFill = document.getElementById('unc-bar');
    barFill.style.width = (a.uncertainty * 100) + '%';
    barFill.style.background = a.uncertainty < 0.3 ? 'var(--green)' : a.uncertainty < 0.6 ? 'var(--orange)' : 'var(--red)';

    document.getElementById('pol-conf').textContent = (a.polarity_conf * 100).toFixed(1) + '%';

    // Status badge
    const badge = document.getElementById('status-badge');
    if (a.human_status === 'approved') {
        badge.innerHTML = '<span style="color:var(--green)">Approuvé</span>';
    } else if (a.human_status === 'modified') {
        badge.innerHTML = '<span style="color:var(--orange)">Modifié → ' + a.human_score.toFixed(2) + '</span>';
    } else if (a.human_status === 'rejected') {
        badge.innerHTML = '<span style="color:var(--red)">Rejeté</span>';
    } else {
        badge.innerHTML = '<span style="color:var(--text-dim)">En attente</span>';
    }

    // Action button highlights
    document.querySelectorAll('.action-btn').forEach(btn => btn.classList.remove('active'));
    if (a.human_status !== 'pending') {
        const btn = document.querySelector(`.action-btn.${a.human_status === 'approved' ? 'approve' : a.human_status === 'modified' ? 'modify' : 'reject'}`);
        if (btn) btn.classList.add('active');
    }

    // Slider
    const slider = document.getElementById('slider-section');
    if (a.human_status === 'modified') {
        slider.classList.add('visible');
        document.getElementById('score-slider').value = (a.human_score || a.uncertainty) * 100;
        document.getElementById('slider-val').textContent = (a.human_score || a.uncertainty).toFixed(2);
    } else {
        slider.classList.remove('visible');
        document.getElementById('score-slider').value = a.uncertainty * 100;
        document.getElementById('slider-val').textContent = a.uncertainty.toFixed(2);
    }

    // Highlight current in list
    document.querySelectorAll('.article-list-item').forEach((el, i) => {
        el.classList.toggle('current', i === currentIdx);
    });
}

// ─── Navigation ───
function navigate(dir) {
    currentIdx = Math.max(0, Math.min(articles.length - 1, currentIdx + dir));
    renderArticle();
}

function navigateRandom() {
    // Prefer pending articles
    const pending = articles.map((a, i) => a.human_status === 'pending' ? i : -1).filter(i => i >= 0);
    if (pending.length > 0) {
        currentIdx = pending[Math.floor(Math.random() * pending.length)];
    } else {
        currentIdx = Math.floor(Math.random() * articles.length);
    }
    renderArticle();
}

function goToArticle(idx) {
    currentIdx = idx;
    renderArticle();
}

// ─── Actions ───
function setAction(status) {
    const a = articles[currentIdx];
    a.human_status = status;

    if (status === 'approved') {
        a.human_score = a.uncertainty;
        document.getElementById('slider-section').classList.remove('visible');
    } else if (status === 'modified') {
        document.getElementById('slider-section').classList.add('visible');
        const sliderVal = parseFloat(document.getElementById('score-slider').value) / 100;
        a.human_score = sliderVal;
    } else if (status === 'rejected') {
        a.human_score = null;
        document.getElementById('slider-section').classList.remove('visible');
    }

    // Save to server
    fetch('/api/review', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            index: currentIdx,
            status: status,
            human_score: a.human_score
        })
    });

    renderArticle();
    renderList();
    updateStats();
}

function updateSlider() {
    const val = parseFloat(document.getElementById('score-slider').value) / 100;
    document.getElementById('slider-val').textContent = val.toFixed(2);
    articles[currentIdx].human_score = val;

    // Auto-save slider changes
    fetch('/api/review', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            index: currentIdx,
            status: 'modified',
            human_score: val
        })
    });
}

// ─── Stats ───
function updateStats() {
    const scores = articles.map(a => a.uncertainty);
    const reviewed = articles.filter(a => a.human_status !== 'pending');
    const approved = articles.filter(a => a.human_status === 'approved').length;
    const modified = articles.filter(a => a.human_status === 'modified').length;
    const rejected = articles.filter(a => a.human_status === 'rejected').length;
    const progress = Math.round(100 * reviewed.length / articles.length);

    document.getElementById('hdr-approved').textContent = approved;
    document.getElementById('hdr-modified').textContent = modified;
    document.getElementById('hdr-rejected').textContent = rejected;
    document.getElementById('hdr-progress').textContent = progress;
    document.getElementById('hdr-progress-bar').style.width = progress + '%';

    const mean = scores.reduce((a,b) => a+b, 0) / scores.length;
    const stdev = Math.sqrt(scores.map(x => (x-mean)**2).reduce((a,b)=>a+b,0) / scores.length);

    document.getElementById('stat-min').textContent = Math.min(...scores).toFixed(3);
    document.getElementById('stat-max').textContent = Math.max(...scores).toFixed(3);
    document.getElementById('stat-mean').textContent = mean.toFixed(3);
    document.getElementById('stat-stdev').textContent = stdev.toFixed(3);

    // Distribution chart
    const bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
    const counts = new Array(bins.length - 1).fill(0);
    scores.forEach(s => {
        for (let i = 0; i < bins.length - 1; i++) {
            if (s >= bins[i] && s < bins[i+1]) { counts[i]++; break; }
            if (i === bins.length - 2 && s >= bins[i]) { counts[i]++; break; }
        }
    });
    const maxCount = Math.max(...counts, 1);
    const colors = ['#333', '#444', '#555', '#666', '#777', '#888', '#999', '#aaa', '#bbb', '#ccc'];

    const chartDiv = document.getElementById('dist-chart');
    chartDiv.innerHTML = counts.map((c, i) =>
        `<div class="dist-bar" data-count="${c}" style="height:${Math.max(c/maxCount*100, 3)}%; background:${colors[i]}"></div>`
    ).join('');

    const labelsDiv = document.getElementById('dist-labels');
    labelsDiv.innerHTML = bins.slice(0, -1).map(b => `<span>${b.toFixed(1)}</span>`).join('');
}

// ─── Render Article List ───
function renderList() {
    const listDiv = document.getElementById('article-list');
    listDiv.innerHTML = articles.map((a, i) =>
        `<div class="article-list-item ${i === currentIdx ? 'current' : ''}" onclick="goToArticle(${i})">
            <div class="dot ${a.human_status}"></div>
            <div class="ali-title">${a.title}</div>
            <div class="ali-score">${a.uncertainty.toFixed(2)}</div>
        </div>`
    ).join('');
}

// ─── Export ───
async function exportCSV() {
    const resp = await fetch('/api/export');
    const blob = await resp.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'human_review_results.csv';
    a.click();
    URL.revokeObjectURL(url);
}

// ─── Keyboard shortcuts ───
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') navigate(-1);
    if (e.key === 'ArrowRight') navigate(1);
    if (e.key === 'a' || e.key === 'A') setAction('approved');
    if (e.key === 'm' || e.key === 'M') setAction('modified');
    if (e.key === 'r' || e.key === 'R') setAction('rejected');
    if (e.key === '?') navigateRandom();
});

// Init
loadData();
</script>
</body>
</html>"""

# ─── Store human reviews in memory ───
_reviews = {}   # index -> {status, human_score}


class ReviewHandler(BaseHTTPRequestHandler):
    """Handle API requests for the HITL interface."""

    articles_cache = None

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == '/' or parsed.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode('utf-8'))

        elif parsed.path == '/api/articles':
            if ReviewHandler.articles_cache is None:
                ReviewHandler.articles_cache = get_articles()
            # Apply reviews
            articles = ReviewHandler.articles_cache
            for idx, review in _reviews.items():
                if idx < len(articles):
                    articles[idx]['human_status'] = review['status']
                    articles[idx]['human_score'] = review.get('human_score')

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(articles).encode('utf-8'))

        elif parsed.path == '/api/export':
            articles = ReviewHandler.articles_cache or get_articles()
            for idx, review in _reviews.items():
                if idx < len(articles):
                    articles[idx]['human_status'] = review['status']
                    articles[idx]['human_score'] = review.get('human_score')

            # Build CSV
            import io
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(['title', 'ticker', 'polarity', 'polarity_conf',
                           'model_uncertainty', 'human_status', 'human_score'])
            for a in articles:
                writer.writerow([
                    a['title'], a['ticker'], a['polarity'], a['polarity_conf'],
                    a['uncertainty'], a['human_status'], a['human_score']
                ])

            self.send_response(200)
            self.send_header('Content-Type', 'text/csv')
            self.send_header('Content-Disposition', 'attachment; filename=human_review_results.csv')
            self.end_headers()
            self.wfile.write(output.getvalue().encode('utf-8'))

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/review':
            content_length = int(self.headers['Content-Length'])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            idx = data['index']
            _reviews[idx] = {
                'status': data['status'],
                'human_score': data.get('human_score'),
            }

            # Update cache
            if ReviewHandler.articles_cache and idx < len(ReviewHandler.articles_cache):
                article = ReviewHandler.articles_cache[idx]
                article['human_status'] = data['status']
                article['human_score'] = data.get('human_score')
                try:
                    import sqlite3
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute('''
                        INSERT OR REPLACE INTO human_reviews (url, human_status, human_score, timestamp)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ''', (article['url'], data['status'], data.get('human_score')))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    pass

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()


def main():
    server = HTTPServer(('127.0.0.1', PORT), ReviewHandler)
    print(f"")
    print(f"  ╔══════════════════════════════════════════════╗")
    print(f"  ║   Human-in-the-Loop Review Interface       ║")
    print(f"  ║                                              ║")
    print(f"  ║  → http://localhost:{PORT}                    ║")
    print(f"  ║                                              ║")
    print(f"  ║  Raccourcis clavier :                        ║")
    print(f"  ║    ← → : Naviguer entre articles             ║")
    print(f"  ║    A   : Approuver                           ║")
    print(f"  ║    M   : Modifier                            ║")
    print(f"  ║    R   : Rejeter                             ║")
    print(f"  ║    ?   : Article aléatoire                   ║")
    print(f"  ║                                              ║")
    print(f"  ║  Ctrl+C pour arrêter                         ║")
    print(f"  ╚══════════════════════════════════════════════╝")
    print(f"")

    # Open browser
    threading.Timer(1.0, lambda: webbrowser.open(f'http://localhost:{PORT}')).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServeur arrêté.")
        server.server_close()


if __name__ == "__main__":
    main()
