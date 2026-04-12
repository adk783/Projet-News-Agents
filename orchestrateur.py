import sqlite3
import logging
import time
from agent_sentiment import analyser_article
from agent_agregateur import calculer_score_ticker
from status_manager import write_status

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("Orchestrateur")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

file_handler = logging.FileHandler("orchestrateur.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DB_PATH = "news_database.db"


# ─── INITIALISATION DE LA BASE ─────────────────────────────────────────────────
def init_db(cursor):
    """
    Prépare la base de données :
    - Ajoute la colonne is_analyzed à articles si elle n'existe pas
    - Crée la table article_scores si elle n'existe pas
    - Crée la table ticker_scores si elle n'existe pas
    """

    # Ajout de is_analyzed dans articles si absente
    colonnes = [row[1] for row in cursor.execute("PRAGMA table_info(articles)").fetchall()]
    if "is_analyzed" not in colonnes:
        cursor.execute("ALTER TABLE articles ADD COLUMN is_analyzed INTEGER DEFAULT 0")
        logger.info("Colonne is_analyzed ajoutée à la table articles")

    # Table des scores par article
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS article_scores (
            url          TEXT PRIMARY KEY,
            ticker       TEXT,
            sentiment    TEXT,
            score        REAL,
            reasoning    TEXT,
            analyzed_at  TEXT
        )
    ''')

    # Table des scores globaux par ticker
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ticker_scores (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker           TEXT,
            score_global     REAL,
            sentiment_global TEXT,
            nb_articles      INTEGER,
            nb_neutral       INTEGER DEFAULT 0,
            confidence       TEXT,
            window_start     TEXT,
            window_end       TEXT,
            calculated_at    TEXT
        )
    ''')

    # Migration : ajout de nb_neutral si la table existait déjà sans cette colonne
    colonnes_ts = [row[1] for row in cursor.execute("PRAGMA table_info(ticker_scores)").fetchall()]
    if "nb_neutral" not in colonnes_ts:
        cursor.execute("ALTER TABLE ticker_scores ADD COLUMN nb_neutral INTEGER DEFAULT 0")
        logger.info("Colonne nb_neutral ajoutée à ticker_scores")

    logger.info("Base de données initialisée")


# ─── RÉCUPÉRATION DES ARTICLES NON ANALYSÉS ────────────────────────────────────
def get_articles_non_analyses(cursor):
    """
    Retourne tous les articles avec is_analyzed = 0
    """
    cursor.execute('''
        SELECT url, ticker, title, content, date_utc
        FROM articles
        WHERE is_analyzed = 0
    ''')
    articles = cursor.fetchall()
    logger.info(f"{len(articles)} article(s) non analysé(s) trouvé(s)")
    return articles


# ─── BOUCLE PRINCIPALE ─────────────────────────────────────────────────────────
def run_orchestrateur():
    logger.info("=== Démarrage de l'orchestrateur ===")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Initialisation de la base
    init_db(cursor)
    conn.commit()

    # 2. Récupération des articles à analyser
    articles = get_articles_non_analyses(cursor)

    if not articles:
        logger.info("Aucun article à analyser, arrêt.")
        conn.close()
        return

    # 3. Traitement article par article
    tickers_traites = set()
    write_status("orchestrateur", {"running": True, "article_actuel": "", "done": 0, "total": len(articles)})
    for article in articles:
        url, ticker, title, content, date_utc = article

        write_status("orchestrateur", {"running": True, "article_actuel": title[:60], "done": list(articles).index(article), "total": len(articles)})
        logger.info(f"Analyse en cours : [{ticker}] {title}")

        try:
            # Appel de l'agent sentiment
            resultat = analyser_article(
                url=url,
                ticker=ticker,
                title=title,
                content=content
            )

            # Sauvegarde du score dans article_scores
            cursor.execute('''
                INSERT OR IGNORE INTO article_scores 
                (url, ticker, sentiment, score, reasoning, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                url,
                ticker,
                resultat["sentiment"],
                resultat["score"],
                resultat["reasoning"],
                resultat["analyzed_at"]
            ))

            # Marquage de l'article comme analysé
            cursor.execute('''
                UPDATE articles SET is_analyzed = 1 WHERE url = ?
            ''', (url,))

            conn.commit()
            tickers_traites.add(ticker)

            logger.info(f"✓ [{ticker}] {title[:60]}... → {resultat['sentiment']} ({resultat['score']})")

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de '{title}' : {e}")
            continue

        # Petite pause pour ne pas surcharger Ollama
        time.sleep(1)

    # 4. Calcul du score global pour chaque ticker traité
    logger.info(f"Calcul des scores globaux pour : {tickers_traites}")

    for ticker in tickers_traites:
        try:
            calculer_score_ticker(cursor, conn, ticker)
        except Exception as e:
            logger.error(f"Erreur agrégation pour {ticker} : {e}")

    write_status("orchestrateur", {"running": False, "article_actuel": "Terminé", "done": len(articles), "total": len(articles)})
    conn.close()
    logger.info("=== Orchestrateur terminé ===")


# ─── POINT D'ENTRÉE ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_orchestrateur()