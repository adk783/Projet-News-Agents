from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime

OUTPUT = "rapport_projet_news_agents.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=A4,
    leftMargin=2.5*cm, rightMargin=2.5*cm,
    topMargin=2.5*cm, bottomMargin=2.5*cm
)

W, H = A4
styles = getSampleStyleSheet()

# ── STYLES ─────────────────────────────────────────────────────────────────────
s_title = ParagraphStyle("title",
    fontSize=22, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#ffffff"),
    backColor=colors.HexColor("#0f0f0f"),
    spaceAfter=6, spaceBefore=0,
    leftIndent=0, leading=28, alignment=TA_CENTER
)
s_subtitle = ParagraphStyle("subtitle",
    fontSize=10, fontName="Helvetica",
    textColor=colors.HexColor("#888888"),
    spaceAfter=20, alignment=TA_CENTER
)
s_h1 = ParagraphStyle("h1",
    fontSize=13, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#111111"),
    spaceBefore=18, spaceAfter=6,
    borderPad=4
)
s_h2 = ParagraphStyle("h2",
    fontSize=10, fontName="Helvetica-Bold",
    textColor=colors.HexColor("#333333"),
    spaceBefore=10, spaceAfter=4
)
s_body = ParagraphStyle("body",
    fontSize=9, fontName="Helvetica",
    textColor=colors.HexColor("#222222"),
    leading=14, spaceAfter=6
)
s_bullet = ParagraphStyle("bullet",
    fontSize=9, fontName="Helvetica",
    textColor=colors.HexColor("#222222"),
    leading=14, leftIndent=16, spaceAfter=3,
    bulletIndent=4
)
s_ok = ParagraphStyle("ok",
    fontSize=9, fontName="Helvetica",
    textColor=colors.HexColor("#16a34a"),
    leading=14, leftIndent=16, spaceAfter=3
)
s_warn = ParagraphStyle("warn",
    fontSize=9, fontName="Helvetica",
    textColor=colors.HexColor("#dc2626"),
    leading=14, leftIndent=16, spaceAfter=3
)
s_mono = ParagraphStyle("mono",
    fontSize=8, fontName="Courier",
    textColor=colors.HexColor("#444444"),
    backColor=colors.HexColor("#f5f5f5"),
    leading=12, leftIndent=12, spaceAfter=6,
    borderPad=6
)

def hr():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor("#e0e0e0"), spaceAfter=8, spaceBefore=4)

def h1(txt): return Paragraph(txt, s_h1)
def h2(txt): return Paragraph(txt, s_h2)
def body(txt): return Paragraph(txt, s_body)
def bullet(txt): return Paragraph(f"• {txt}", s_bullet)
def ok(txt): return Paragraph(f"✓  {txt}", s_ok)
def warn(txt): return Paragraph(f"✗  {txt}", s_warn)
def mono(txt): return Paragraph(txt, s_mono)
def sp(n=1): return Spacer(1, n * 0.3 * cm)

# ── TABLE HELPER ───────────────────────────────────────────────────────────────
def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#111111")),
        ("TEXTCOLOR",    (0,0), (-1,0), colors.white),
        ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8.5),
        ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#fafafa"), colors.white]),
        ("GRID",         (0,0), (-1,-1), 0.3, colors.HexColor("#dddddd")),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (-1,-1), 7),
        ("RIGHTPADDING", (0,0), (-1,-1), 7),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t

# ── CONTENU ────────────────────────────────────────────────────────────────────
story = []

# Titre
story.append(Paragraph("SENTIMENT / NEWS AGENTS", s_title))
story.append(Paragraph(f"Rapport technique — {datetime.now().strftime('%d %B %Y')}", s_subtitle))
story.append(hr())

# ── 1. RÉSUMÉ DU PROJET ────────────────────────────────────────────────────────
story.append(h1("1. Résumé du projet"))
story.append(body(
    "Projet de pipeline automatisé de collecte, d'analyse de sentiment et d'agrégation "
    "de news financières par ticker boursier. Le système collecte des articles depuis "
    "plusieurs sources, les analyse via un modèle de langage local (phi4-mini via Ollama), "
    "calcule un score de sentiment global par ticker, et affiche les résultats sur un "
    "dashboard web en temps réel."
))
story.append(sp())

# ── 2. ARCHITECTURE ────────────────────────────────────────────────────────────
story.append(h1("2. Architecture"))
story.append(hr())

arch_table = make_table(
    ["Fichier", "Rôle"],
    [
        ["news_pipeline.py",   "Collecte d'articles (Yahoo Finance + Finnhub), filtrage, stockage SQLite"],
        ["orchestrateur.py",   "Chef d'orchestre : dispatch vers l'agent sentiment, puis l'agrégateur"],
        ["agent_sentiment.py", "Analyse de sentiment via phi4-mini (Ollama local) — retourne JSON"],
        ["agent_agregateur.py","Calcul du score global par ticker sur fenêtre glissante 48h"],
        ["dashboard.py",       "Serveur Flask — API REST + interface web temps réel"],
        ["status_manager.py",  "Gestion du statut d'avancement (pipeline_status.json)"],
    ],
    col_widths=[5*cm, 10.5*cm]
)
story.append(arch_table)
story.append(sp())

story.append(h2("Modèle IA utilisé"))
story.append(bullet("phi4-mini via Ollama (local, http://localhost:11434)"))
story.append(bullet("Temperature = 0.0 — réponses déterministes"))
story.append(bullet("Prompt few-shot avec exemples variés bullish/bearish/neutral"))
story.append(bullet("Format de réponse : JSON strict {sentiment, score, reasoning}"))
story.append(sp())

story.append(h2("Base de données"))
story.append(bullet("SQLite — tables : articles, article_scores, ticker_scores, articles_filtres"))
story.append(bullet("Déduplication par URL (INSERT OR IGNORE)"))
story.append(bullet("Colonne source pour tracer l'origine Yahoo/Finnhub"))

# ── 3. FONCTIONNALITÉS IMPLÉMENTÉES ────────────────────────────────────────────
story.append(sp(2))
story.append(h1("3. Fonctionnalités implémentées"))
story.append(hr())

story.append(h2("Sourcing"))
story.append(ok("Double source : Yahoo Finance + Finnhub API"))
story.append(ok("Filtrage par mots-clés (nom société, symbole, dirigeants, domaine)"))
story.append(ok("Extraction contenu complet via Newspaper3k puis Trafilatura en fallback"))
story.append(ok("Limite : 10 articles Yahoo + 5 articles Finnhub par ticker"))
story.append(ok("Badge source affiché dans le dashboard"))

story.append(sp())
story.append(h2("Analyse de sentiment"))
story.append(ok("Score signé -1 a +1 (bearish negatif, bullish positif)"))
story.append(ok("Neutrals exclus du calcul — score null stocke en base"))
story.append(ok("Fallback 0.70 si le modele retourne null sur bullish/bearish"))
story.append(ok("Prompt enrichi : 3 paliers de score, 5 exemples few-shot varies"))

story.append(sp())
story.append(h2("Aggregation par ticker"))
story.append(ok("Fenetre glissante 48h"))
story.append(ok("Moyenne ponderee par |score| (option A) — articles convaincus pesent plus"))
story.append(ok("Seuils : >= +0.20 bullish / <= -0.20 bearish / entre les deux neutral"))
story.append(ok("Niveau de confiance : insufficient / low / normal / high"))
story.append(ok("nb_neutral affiché séparément dans le dashboard"))

story.append(sp())
story.append(h2("Dashboard"))
story.append(ok("Barre de progression unique, auto-détection du pipeline actif"))
story.append(ok("Scores signés (+/-) avec barre centrée sur 0"))
story.append(ok("Graphe camembert répartition sentiment + courbe d'évolution"))
story.append(ok("Filtres articles par sentiment, badge source Yahoo/Finnhub"))
story.append(ok("Rafraichissement automatique toutes les 10s"))

# ── 4. RÉSULTATS DU TEST ────────────────────────────────────────────────────────
story.append(sp(2))
story.append(h1("4. Résultats du test (26 articles, 3 tickers)"))
story.append(hr())

story.append(make_table(
    ["Ticker", "Sentiment", "Score", "Actifs", "Neutrals", "Confiance"],
    [
        ["TSLA", "bearish",  "-0.83", "6", "4", "high"],
        ["AAPL", "bearish",  "-0.76", "4", "3", "normal"],
        ["MSFT", "bullish",  "+0.50", "6", "3", "high"],
    ],
    col_widths=[2.5*cm, 2.5*cm, 2.5*cm, 2*cm, 2.5*cm, 3*cm]
))
story.append(sp())

story.append(make_table(
    ["Sentiment", "Nb articles", "Score moyen", "Min", "Max"],
    [
        ["bearish", "11 (42%)", "0.784", "0.61", "0.88"],
        ["bullish", "5 (19%)",  "0.730", "0.70", "0.85"],
        ["neutral", "10 (38%)", "null",  "—",    "—"],
    ],
    col_widths=[3*cm, 3*cm, 3*cm, 2.5*cm, 2.5*cm]
))
story.append(sp())

story.append(h2("Qualite des analyses IA"))
story.append(ok("AAPL : 3/3 classifications correctes"))
story.append(ok("TSLA : 4/5 correctes — Tesla Big Rig incohérent entre Yahoo et Finnhub"))
story.append(warn("MSFT : 2 erreurs — article META classé MSFT bullish 0.85 (problème sourcing)"))
story.append(warn("MSFT : article Anthropic classé MSFT bullish 0.70 (extrapolation abusive)"))

# ── 5. PROBLÈMES À RÉSOUDRE ────────────────────────────────────────────────────
story.append(sp(2))
story.append(h1("5. Problèmes à résoudre (par priorité)"))
story.append(hr())

problems = [
    ("P1 — CRITIQUE", "Doublons inter-sources",
     "Un même article collecté par Yahoo ET Finnhub avec des URLs différentes est analysé "
     "deux fois et compte deux fois dans l'agrégation, ce qui biaise le score. "
     "Solution : déduplication par titre normalisé (similarité) en plus de l'URL.",
     "#dc2626"),
    ("P2 — IMPORTANT", "Mauvais articles collectés pour un ticker",
     "Yahoo Finance remonte parfois des articles sur d'autres entreprises (Meta, Anthropic, SpaceX) "
     "sous le ticker MSFT ou TSLA. Le filtrage par mots-clés ne suffit pas. "
     "Solution : ajouter un deuxième filtre IA léger qui vérifie la pertinence de l'article "
     "avant de le passer à l'analyse de sentiment.",
     "#d97706"),
    ("P3 — IMPORTANT", "Incohérence entre sources sur le même article",
     "Tesla Big Rig classé neutral (Yahoo) et bearish 0.61 (Finnhub) pour le même article. "
     "Cela indique que le contenu extrait est différent selon la source (résumé vs article complet). "
     "Solution : toujours prioriser le contenu le plus long des deux.",
     "#d97706"),
    ("P4 — MOYEN", "Granularité des scores bullish encore faible",
     "Les scores bullish se concentrent à 0.70-0.85. Les examples few-shot manquent "
     "d'exemples de signaux très forts (0.95+) pour les bullish. "
     "Solution : enrichir le prompt avec 2 exemples bullish forts supplémentaires.",
     "#2563eb"),
    ("P5 — MOYEN", "Fenêtre 48h fixe pour tous les tickers",
     "TSLA est un titre très volatile, 48h peut capturer des signaux contradictoires. "
     "Un titre stable comme MSFT pourrait mériter une fenêtre plus longue. "
     "Solution : fenêtre adaptative selon la volatilité historique du ticker.",
     "#2563eb"),
    ("P6 — FAIBLE", "Pas de notification de limite API Finnhub",
     "La limite de 60 req/min Finnhub n'est pas surveillée, les erreurs 429 passent "
     "silencieusement dans le log sans alerte dashboard. "
     "Solution : ajouter un compteur de rate-limit et une alerte visuelle.",
     "#6b7280"),
]

for code, titre, desc, color in problems:
    badge_style = ParagraphStyle("badge",
        fontSize=8, fontName="Helvetica-Bold",
        textColor=colors.white,
        backColor=colors.HexColor(color),
        spaceAfter=2, spaceBefore=8,
        leftIndent=0, leading=12
    )
    story.append(Paragraph(f"  {code} — {titre}  ", badge_style))
    story.append(body(desc))

# ── 6. PISTES D'AMÉLIORATION FUTURES ──────────────────────────────────────────
story.append(sp(2))
story.append(h1("6. Pistes d'amélioration futures"))
story.append(hr())

story.append(bullet("Ajouter NewsAPI.org comme troisième source de collecte"))
story.append(bullet("Intégrer Alpha Vantage pour comparer le sentiment IA vs sentiment pré-calculé"))
story.append(bullet("Passer à WebSocket pour un dashboard temps réel sans polling"))
story.append(bullet("Historique des scores sur 7 jours avec graphe d'évolution longue durée"))
story.append(bullet("Export CSV / Excel des scores par ticker"))
story.append(bullet("Alertes par email/webhook quand un ticker dépasse un seuil de sentiment"))

# ── FOOTER ────────────────────────────────────────────────────────────────────
story.append(sp(3))
story.append(hr())
story.append(Paragraph(
    f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')} — Projet News Agents",
    ParagraphStyle("footer", fontSize=7, textColor=colors.HexColor("#aaaaaa"), alignment=TA_CENTER)
))

doc.build(story)
print(f"PDF genere : {OUTPUT}")
