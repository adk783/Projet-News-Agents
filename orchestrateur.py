"""
Étage 5 — Orchestrateur.

Boucle principale qui :
  1) initialise / migre les tables (articles, article_scores, ticker_scores) ;
  2) lit les articles is_analyzed=0 ;
  3) pour chaque article : filtrage cascade (keywords + LLM) → processing
     multi-features → insertion dans article_scores → is_analyzed=1 ;
  4) agrège par ticker via agent_agregateur.calculer_score_ticker ;
  5) écrit le status temps réel dans pipeline_status.json (status_manager).

Provenance :
- Squelette orchestrateur + suivi status_manager : branche `Antoinev2`.
- Insertion d'un étage de filtrage cascade            : `filtrage-keywords` + `Antoinev2`.
- Insertion d'un étage processing multi-agents        : `samuel`.
"""

import logging
import sqlite3
import time

import yfinance as yf

from agent_filtrage import build_keywords, est_pertinent
from agent_agregateur import calculer_score_ticker
from processing_pipeline import (
    build_agents,
    ensure_schema,
    insert_article_scores,
    process_article,
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


# ─── INFOS PAR TICKER (keywords + company_name) ───────────────────────────────
def load_ticker_context(tickers):
    """Pré-charge les keywords et le nom de société pour chaque ticker."""
    ctx = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
            ctx[t] = {
                "company_name": info.get("longName") or info.get("shortName") or t,
                "keywords": build_keywords(info),
            }
            logger.info(f"[{t}] keywords={len(ctx[t]['keywords'])} ; name='{ctx[t]['company_name']}'")
        except Exception as e:
            logger.warning(f"[{t}] yfinance.info indisponible ({e}) — fallback minimal")
            ctx[t] = {"company_name": t, "keywords": {t.lower()}}
    return ctx


# ─── BOUCLE PRINCIPALE ────────────────────────────────────────────────────────
def run_orchestrateur():
    logger.info("=== Démarrage de l'orchestrateur ===")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    init_db(cursor)
    conn.commit()

    articles = get_articles_non_analyses(cursor)
    if not articles:
        logger.info("Aucun article à analyser, arrêt.")
        conn.close()
        return

    # Contexte par ticker (keywords + nom)
    tickers_uniques = sorted({row[1] for row in articles})
    ctx = load_ticker_context(tickers_uniques)

    # Agents de processing (chargement coûteux : on le fait une seule fois)
    agents = build_agents()

    tickers_traites = set()
    total = len(articles)
    write_status("orchestrateur", {"running": True, "article_actuel": "", "done": 0, "total": total})

    for idx, article in enumerate(articles):
        url, ticker, title, content, _date_utc = article
        write_status(
            "orchestrateur",
            {"running": True, "article_actuel": (title or "")[:60], "done": idx, "total": total},
        )
        logger.info(f"[{ticker}] {title}")

        try:
            # --- Étage 2 : filtrage cascade ---------------------------------
            tinfo = ctx.get(ticker, {"company_name": ticker, "keywords": {ticker.lower()}})
            garde, motif = est_pertinent(
                ticker=ticker,
                company_name=tinfo["company_name"],
                title=title or "",
                content=content or "",
                keywords=tinfo["keywords"],
            )

            if not garde:
                cursor.execute("UPDATE articles SET is_analyzed = 1 WHERE url = ?", (url,))
                conn.commit()
                logger.info(f"  → rejeté ({motif})")
                continue

            # --- Étage 3 : processing multi-features ------------------------
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
        except Exception as e:
            logger.error(f"Erreur sur '{title}' : {e}")
            continue

        time.sleep(0.5)

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
