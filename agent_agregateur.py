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
    Retourne le niveau de confiance selon le nombre d'articles non-neutral analysés.
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
    Convertit un score signé (-1 → +1) en label sentiment global.
    """
    if score_global >= 0.20:
        return "bullish"
    elif score_global <= -0.20:
        return "bearish"
    else:
        return "neutral"


# ─── CALCUL DU SCORE GLOBAL ────────────────────────────────────────────────────
def calculer_score_ticker(cursor, conn, ticker):
    """
    Calcule le score global d'un ticker sur la fenêtre des WINDOW_HOURS dernières heures.
    Score sur échelle signée -1 → +1 (bearish négatif, bullish positif).
    Moyenne pondérée par |score| (option A) — les neutrals sont exclus du calcul.
    Insère une nouvelle ligne dans ticker_scores.
    C'est cette fonction que l'orchestrateur appelle.
    """

    now          = datetime.now(timezone.utc)
    window_end   = now
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

    # Séparation neutrals / articles actifs
    articles_actifs = [(score, sentiment) for score, sentiment in resultats if sentiment != "neutral"]
    nb_neutral      = len(resultats) - len(articles_actifs)
    nb_articles     = len(articles_actifs)

    logger.info(f"[{ticker}] {nb_articles} article(s) actif(s) + {nb_neutral} neutral(s) dans la fenêtre de {WINDOW_HOURS}h")

    # Cas : pas assez de données
    if nb_articles == 0:
        cursor.execute('''
            INSERT INTO ticker_scores
            (ticker, score_global, sentiment_global, nb_articles, nb_neutral, confidence, window_start, window_end, calculated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ticker,
            None,
            None,
            0,
            nb_neutral,
            "insufficient",
            window_start_str,
            window_end_str,
            now.isoformat()
        ))
        conn.commit()
        logger.info(f"[{ticker}] Pas de données actives → insufficient ({nb_neutral} neutral(s) ignoré(s))")
        return

    # Conversion en scores signés + pondération par |score|
    scores_signes = []
    poids         = []
    for score, sentiment in articles_actifs:
        score_signe = -score if sentiment == "bearish" else score
        scores_signes.append(score_signe)
        poids.append(abs(score_signe))

    total_poids  = sum(poids)
    score_global = round(
        sum(s * p for s, p in zip(scores_signes, poids)) / total_poids, 4
    ) if total_poids > 0 else 0.0

    sentiment_global = get_sentiment_global(score_global)
    confidence       = get_confidence(nb_articles)

    # Insertion dans ticker_scores
    cursor.execute('''
        INSERT INTO ticker_scores
        (ticker, score_global, sentiment_global, nb_articles, nb_neutral, confidence, window_start, window_end, calculated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        ticker,
        score_global,
        sentiment_global,
        nb_articles,
        nb_neutral,
        confidence,
        window_start_str,
        window_end_str,
        now.isoformat()
    ))
    conn.commit()

    logger.info(f"[{ticker}] Score global → {sentiment_global} ({score_global:+.4f}) | {nb_articles} actifs + {nb_neutral} neutrals | confiance : {confidence}")