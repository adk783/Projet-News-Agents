import logging
from datetime import datetime, timezone, timedelta

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("AgentAgregateur")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

file_handler = logging.FileHandler("orchestrateur.log", mode="a", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
WINDOW_HOURS = 48   # Fenêtre glissante en heures


# ─── NIVEAU DE CONFIANCE ───────────────────────────────────────────────────────
def get_confidence(nb_articles):
    """
    Retourne le niveau de confiance selon le nombre d'articles analysés.
    """
    if nb_articles == 0:
        return "insufficient"
    elif nb_articles <= 2:
        return "low"
    elif nb_articles <= 5:
        return "normal"
    else:
        return "high"


# ─── SENTIMENT GLOBAL ──────────────────────────────────────────────────────────
def get_sentiment_global(score_global):
    """
    Convertit un score numérique en label sentiment global.
    """
    if score_global >= 0.60:
        return "bullish"
    elif score_global <= 0.40:
        return "bearish"
    else:
        return "neutral"


# ─── CALCUL DU SCORE GLOBAL ────────────────────────────────────────────────────
def calculer_score_ticker(cursor, conn, ticker):
    """
    Calcule le score global d'un ticker sur la fenêtre des WINDOW_HOURS dernières heures.
    Insère une nouvelle ligne dans ticker_scores.
    C'est cette fonction que l'orchestrateur appelle.
    """

    now         = datetime.now(timezone.utc)
    window_end  = now
    window_start = now - timedelta(hours=WINDOW_HOURS)

    window_start_str = window_start.isoformat()
    window_end_str   = window_end.isoformat()

    logger.debug(f"[{ticker}] Fenêtre : {window_start_str} → {window_end_str}")

    # Récupération des scores dans la fenêtre temporelle
    cursor.execute('''
        SELECT score, sentiment
        FROM article_scores
        WHERE ticker = ?
          AND analyzed_at >= ?
          AND analyzed_at <= ?
    ''', (ticker, window_start_str, window_end_str))

    resultats = cursor.fetchall()
    nb_articles = len(resultats)

    logger.info(f"[{ticker}] {nb_articles} article(s) trouvé(s) dans la fenêtre de {WINDOW_HOURS}h")

    # Cas : pas assez de données
    if nb_articles == 0:
        cursor.execute('''
            INSERT INTO ticker_scores
            (ticker, score_global, sentiment_global, nb_articles, confidence, window_start, window_end, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker,
            None,
            None,
            0,
            "insufficient",
            window_start_str,
            window_end_str,
            now.isoformat()
        ))
        conn.commit()
        logger.info(f"[{ticker}] Pas de données → insufficient")
        return

    # Calcul de la moyenne des scores
    scores = [row[0] for row in resultats]
    score_global = round(sum(scores) / len(scores), 4)

    sentiment_global = get_sentiment_global(score_global)
    confidence       = get_confidence(nb_articles)

    # Insertion dans ticker_scores
    cursor.execute('''
        INSERT INTO ticker_scores
        (ticker, score_global, sentiment_global, nb_articles, confidence, window_start, window_end, calculated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        ticker,
        score_global,
        sentiment_global,
        nb_articles,
        confidence,
        window_start_str,
        window_end_str,
        now.isoformat()
    ))
    conn.commit()

    logger.info(f"[{ticker}] Score global → {sentiment_global} ({score_global}) | {nb_articles} articles | confiance : {confidence}")