"""
Étage 5 — Orchestrateur.

Boucle principale qui :
  1) initialise / migre les tables (articles, article_scores, ticker_scores) ;
  2) lit les articles is_analyzed=0 ;
  3) pour chaque article : processing multi-features → insertion dans
     article_scores → is_analyzed=1
     (le filtrage de pertinence est déjà fait par news_pipeline au sourcing,
     les articles en DB sont par construction pertinents) ;
  4) agrège par ticker via agent_agregateur.calculer_score_ticker ;
  5) écrit le status temps réel dans pipeline_status.json (status_manager).

Provenance :
- Squelette orchestrateur + suivi status_manager : branche `Antoinev2`.
- Étage processing multi-agents                  : branche `samuel`.
"""

import logging
import sqlite3
import sys
import time

import requests

from agent_agregateur import calculer_score_ticker
from processing_pipeline import (
    build_agents,
    ensure_schema,
    insert_article_scores,
    process_article,
)
from sentiment_agent import (
    OLLAMA_URL as SENTIMENT_OLLAMA_URL,
    MODEL_NAME as SENTIMENT_MODEL,
    OllamaUnavailableError,
)
from status_manager import write_status


# ─── LOGGING ───────────────────────────────────────────────────────────────────
logger = logging.getLogger("Orchestrateur")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    file_handler = logging.FileHandler("orchestrateur.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


DB_PATH = "news_database.db"


# ─── HEALTHCHECK OLLAMA ───────────────────────────────────────────────────────
def check_ollama_ready():
    """
    Vérifie qu'Ollama tourne et que le modèle de sentiment est disponible.
    En cas d'échec : log ERREUR explicite, écrit l'erreur dans pipeline_status.json,
    et retourne False.
    """
    base = SENTIMENT_OLLAMA_URL.rsplit("/api/", 1)[0]
    tags_url = base + "/api/tags"

    try:
        r = requests.get(tags_url, timeout=5)
        r.raise_for_status()
        models = {m["name"] for m in r.json().get("models", [])}
    except Exception as e:
        msg = (
            f"Ollama indisponible sur {base} ({e}). "
            f"Lance `ollama serve` dans un autre terminal."
        )
        logger.error(f"[OLLAMA] {msg}")
        write_status(
            "orchestrateur",
            {"running": False, "article_actuel": "ERREUR", "done": 0, "total": 0, "error": msg},
        )
        return False

    if not any(m.startswith(SENTIMENT_MODEL) for m in models):
        msg = (
            f"Modèle '{SENTIMENT_MODEL}' non trouvé dans Ollama "
            f"(modèles présents : {sorted(models)}). "
            f"Lance `ollama pull {SENTIMENT_MODEL}`."
        )
        logger.error(f"[OLLAMA] {msg}")
        write_status(
            "orchestrateur",
            {"running": False, "article_actuel": "ERREUR", "done": 0, "total": 0, "error": msg},
        )
        return False

    logger.info(f"[OLLAMA] OK — {len(models)} modèle(s) ; '{SENTIMENT_MODEL}' trouvé.")
    return True


# ─── INIT BASE ────────────────────────────────────────────────────────────────
def init_db(cursor):
    # Colonne is_analyzed sur articles
    colonnes = [row[1] for row in cursor.execute("PRAGMA table_info(articles)")]
    if "is_analyzed" not in colonnes:
        cursor.execute("ALTER TABLE articles ADD COLUMN is_analyzed INTEGER DEFAULT 0")
        logger.info("Colonne is_analyzed ajoutée à la table articles")

    # article_scores (schéma étendu)
    ensure_schema(cursor)

    # ticker_scores
    cursor.execute(
        """
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
        """
    )
    colonnes_ts = [row[1] for row in cursor.execute("PRAGMA table_info(ticker_scores)")]
    if "nb_neutral" not in colonnes_ts:
        cursor.execute("ALTER TABLE ticker_scores ADD COLUMN nb_neutral INTEGER DEFAULT 0")
        logger.info("Colonne nb_neutral ajoutée à ticker_scores")

    logger.info("Base de données initialisée")


def get_articles_non_analyses(cursor):
    cursor.execute(
        """
        SELECT url, ticker, title, content, date_utc
        FROM articles
        WHERE is_analyzed = 0
        """
    )
    rows = cursor.fetchall()
    logger.info(f"{len(rows)} article(s) non analysé(s)")
    return rows


# ─── BOUCLE PRINCIPALE ────────────────────────────────────────────────────────
def run_orchestrateur():
    logger.info("=== Démarrage de l'orchestrateur ===")

    # --- Healthcheck Ollama AVANT tout chargement coûteux -----------------
    write_status(
        "orchestrateur",
        {"running": True, "article_actuel": "Vérification Ollama…", "done": 0, "total": 0},
    )
    if not check_ollama_ready():
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    init_db(cursor)
    conn.commit()

    articles = get_articles_non_analyses(cursor)
    if not articles:
        logger.info("Aucun article à analyser, arrêt.")
        write_status(
            "orchestrateur",
            {"running": False, "article_actuel": "Aucun article à analyser", "done": 0, "total": 0},
        )
        conn.close()
        return

    # --- Chargement des agents (lourd : FinBERT ~400 Mo, lexiques) -------
    write_status(
        "orchestrateur",
        {
            "running": True,
            "article_actuel": "Chargement des modèles (FinBERT, lexiques)…",
            "done": 0,
            "total": len(articles),
        },
    )
    logger.info("Chargement des agents de processing…")
    agents = build_agents()
    logger.info("Agents prêts.")

    tickers_traites = set()
    total = len(articles)
    write_status(
        "orchestrateur",
        {"running": True, "article_actuel": "", "done": 0, "total": total},
    )

    for idx, article in enumerate(articles):
        url, ticker, title, content, _date_utc = article
        write_status(
            "orchestrateur",
            {"running": True, "article_actuel": (title or "")[:60], "done": idx, "total": total},
        )
        logger.info(f"[{ticker}] {title}")

        try:
            # Étage 3 — processing multi-features
            # (le filtrage de pertinence est déjà fait par news_pipeline)
            row = process_article((url, ticker, title, content), agents)
            insert_article_scores(cursor, row)
            cursor.execute("UPDATE articles SET is_analyzed = 1 WHERE url = ?", (url,))
            conn.commit()
            tickers_traites.add(ticker)

            score_disp = f"{row['score']:.2f}" if row["score"] is not None else "null"
            logger.info(
                f"  ✓ {row['sentiment']} ({score_disp}) | "
                f"polarity={row['polarity']} unc={row['uncertainty']:.2f}"
            )
        except OllamaUnavailableError as e:
            logger.error(f"[OLLAMA] Indisponible en cours de pipeline : {e}")
            write_status(
                "orchestrateur",
                {
                    "running": False,
                    "article_actuel": "ERREUR Ollama",
                    "done": idx,
                    "total": total,
                    "error": str(e),
                },
            )
            conn.close()
            sys.exit(1)
        except Exception as e:
            logger.error(f"Erreur sur '{title}' : {e}")
            continue

        time.sleep(0.3)

    # --- Étage 4 : agrégation par ticker ------------------------------------
    logger.info(f"Agrégation pour : {tickers_traites}")
    for ticker in tickers_traites:
        try:
            calculer_score_ticker(cursor, conn, ticker)
        except Exception as e:
            logger.error(f"Erreur agrégation pour {ticker} : {e}")

    write_status(
        "orchestrateur",
        {"running": False, "article_actuel": "Terminé", "done": total, "total": total},
    )
    conn.close()
    logger.info("=== Orchestrateur terminé ===")


if __name__ == "__main__":
    run_orchestrateur()
