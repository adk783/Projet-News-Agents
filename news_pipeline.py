import os
import json
import logging
import sqlite3
import argparse
from datetime import timezone
from email.utils import parsedate_to_datetime

import yfinance as yf
from newspaper import Article
from agent_filtrage import workflow_filtrer_actualite
from agent_analyste import workflow_analyser_actualite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger(__name__)

os.makedirs(
    os.path.join(os.environ.get("TEMP", "/tmp"), ".newspaper_scraper", "article_resources"),
    exist_ok=True
)

def _normaliser_date(date_str: str) -> str:
    """Normalise une date pubDate vers ISO 8601 UTC pour le stockage et la couche UI."""
    if not date_str:
        return ""
    try:
        dt = parsedate_to_datetime(date_str)
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception:
        return date_str 

def run_news_pipeline(tickers: list[str]) -> None:
    """
    Orchestre le pipeline complet : collecte Yahoo Finance → scraping →
    filtrage DistilRoBERTa → analyse AD-FCoT DeepSeek → persistance SQLite.
    """
    with sqlite3.connect('news_database.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles (
                url             TEXT PRIMARY KEY,
                ticker          TEXT,
                date_utc        TEXT,
                title           TEXT,
                content         TEXT,
                json_brut       TEXT,
                signal_filtrage TEXT,
                score_filtrage  REAL,
                signal_analyste TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS articles_rejetes (
                url      TEXT PRIMARY KEY,
                ticker   TEXT,
                date_utc TEXT,
                title    TEXT,
                raison   TEXT
            )
        ''')
        conn.commit()

        for alter_sql in [
            "ALTER TABLE articles ADD COLUMN signal_filtrage TEXT",
            "ALTER TABLE articles ADD COLUMN score_filtrage REAL",
            "ALTER TABLE articles ADD COLUMN signal_analyste TEXT",
        ]:
            try:
                cursor.execute(alter_sql)
            except Exception:
                pass
        conn.commit()

        for ticker_symbol in tickers:
            logger.info("DÉTECTION : Scan des news pour %s", ticker_symbol)
            stock = yf.Ticker(ticker_symbol)
            news_list = stock.news

            for news_item in news_list:
                content_data = news_item.get('content', {})

                title = content_data.get('title')
                date_utc = _normaliser_date(content_data.get('pubDate', ""))

                url = None
                if content_data.get('clickThroughUrl'):
                    url = content_data['clickThroughUrl'].get('url')
                elif content_data.get('canonicalUrl'):
                    url = content_data['canonicalUrl'].get('url')

                if not url:
                    continue

                cursor.execute("SELECT 1 FROM articles WHERE url = ?", (url,))
                if cursor.fetchone():
                    logger.info("  -> Déjà en base (accepté), ignoré.")
                    continue

                cursor.execute("SELECT 1 FROM articles_rejetes WHERE url = ?", (url,))
                if cursor.fetchone():
                    logger.info("  -> Déjà en base (rejeté), ignoré.")
                    continue

                logger.info("Traitement de l'article : %s", title)
                try:
                    article = Article(url)
                    article.download()
                    article.parse()
                    content = article.text
                    logger.info("  -> Extraction réussie.")
                except Exception as e:
                    logger.warning("  -> Erreur d'extraction : %s", e)
                    cursor.execute(
                        "INSERT OR IGNORE INTO articles_rejetes (url, ticker, date_utc, title, raison) VALUES (?, ?, ?, ?, ?)",
                        (url, ticker_symbol, date_utc, title, "extraction_echec")
                    )
                    conn.commit()
                    continue

                data_dict = {
                    "ticker": ticker_symbol,
                    "title": title,
                    "date_utc": date_utc,
                    "url": url,
                    "content": content
                }

                signal = workflow_filtrer_actualite.invoke(content)
                if signal is None:
                    logger.info("  -> Article rejeté par l'agent de filtrage.")
                    cursor.execute(
                        "INSERT OR IGNORE INTO articles_rejetes (url, ticker, date_utc, title, raison) VALUES (?, ?, ?, ?, ?)",
                        (url, ticker_symbol, date_utc, title, "filtre_modele")
                    )
                    conn.commit()
                    continue

                label_signal, score_signal = signal
                logger.info("  -> Filtre : %s (%.2f%%). Transmission à l'Analyste...", label_signal, score_signal * 100)

                try:
                    decision_dict = workflow_analyser_actualite.invoke({
                        "texte_article": content,
                        "ticker_symbol": ticker_symbol
                    })
                    signal_analyste = decision_dict.get('signal')
                    logger.info("  -> Signal analyste : %s", signal_analyste)
                    json_final = json.dumps({
                        "donnees_brutes": data_dict,
                        "analyse_ad_fcot": decision_dict
                    }, ensure_ascii=False)
                except Exception as e:
                    logger.error("  -> Erreur Ollama (%s) : article mis en file de rejet pour ré-analyse.", type(e).__name__)
                    cursor.execute(
                        "INSERT OR IGNORE INTO articles_rejetes (url, ticker, date_utc, title, raison) VALUES (?, ?, ?, ?, ?)",
                        (url, ticker_symbol, date_utc, title, f"ollama_error:{type(e).__name__}")
                    )
                    conn.commit()
                    continue

                try:
                    cursor.execute('''
                        INSERT INTO articles
                        (url, ticker, date_utc, title, content, json_brut, signal_filtrage, score_filtrage, signal_analyste)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(url) DO UPDATE SET
                            signal_analyste = excluded.signal_analyste,
                            json_brut       = excluded.json_brut
                        WHERE articles.signal_analyste IS NULL
                    ''', (url, ticker_symbol, date_utc, title, content, json_final,
                          label_signal, round(score_signal, 4), signal_analyste))

                    if cursor.rowcount > 0:
                        logger.info("  -> Nouvel article sauvegardé en base de données.")
                    else:
                        logger.info("  -> Article déjà présent en base (ignoré).")

                    conn.commit()
                except Exception as e:
                    logger.error("  -> Erreur lors de la sauvegarde : %s", e)

    logger.info("Fin du Pipeline")

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de news financières")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Liste de tickers à scanner (ex: AAPL MSFT TSLA)"
    )
    args = parser.parse_args()
    run_news_pipeline(args.tickers)