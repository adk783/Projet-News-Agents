import yfinance as yf
from newspaper import Article
import sqlite3
import json
from datetime import datetime, timezone
import argparse
import logging
import time
import trafilatura


logger = logging.getLogger("NewsPipeline")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

file_handler = logging.FileHandler("pipeline.log", mode="w", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def run_news_pipeline(tickers):
    logger.debug(f"Tickers à scanner : {tickers}")
    conn = sqlite3.connect('news_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY, 
            ticker TEXT,
            sector TEXT,
            industry TEXT,
            date_utc TEXT,
            title TEXT,
            content TEXT,
            json_brut TEXT
        )
    ''') 
    conn.commit()
    for ticker_symbol in tickers:
        logger.info(f"DÉTECTION : Scan des news pour {ticker_symbol}")
        stock = yf.Ticker(ticker_symbol)
        sector = stock.info.get('sector', 'Inconnu')
        industry = stock.info.get('industry', 'Inconnu')
        logger.debug(f"{ticker_symbol} : secteur={sector}, industrie={industry}")
        news_list = stock.news
        logger.debug(f"{len(news_list)} articles trouvés pour {ticker_symbol}") 

        for news_item in news_list:
            content_data = news_item.get('content', {})
        
            title = content_data.get('title')
            date_utc = content_data.get('pubDate', "")
        
            url = None
            if content_data.get('clickThroughUrl'):
                url = content_data['clickThroughUrl'].get('url')
            elif content_data.get('canonicalUrl'):
                url = content_data['canonicalUrl'].get('url')
            
            if not url:
                logger.debug(f"Article ignoré (pas d'URL) : {title}")
                continue
            logger.info(f"Traitement : {title}")
            content = ""
            try:
                article = Article(url, request_timeout=10)
                article.download()
                article.parse()
                content = article.text
                logger.debug(f"Newspaper : {len(content)} caractères extraits")
            except Exception as e:
                logger.warning(f"Newspaper a échoué pour '{title}' : {e}")

            if len(content) < 100:
                logger.debug(f"Contenu trop court ({len(content)} car.), tentative avec trafilatura...")
                try:
                    downloaded = trafilatura.fetch_url(url)
                    if downloaded:
                        content = trafilatura.extract(downloaded) or ""
                        logger.debug(f"Trafilatura : {len(content)} caractères extraits")
                except Exception as e:
                    logger.warning(f"Trafilatura a échoué pour '{title}' : {e}")

            if len(content) < 100:
                logger.warning(f"Article ignoré (contenu trop court) : {title}")
                continue

            data_dict = {
                "ticker": ticker_symbol,
                "sector": sector,
                "industry": industry,
                "title": title,
                "date_utc": date_utc,
                "url": url,
                "content": content
            }
            json_standard = json.dumps(data_dict, ensure_ascii=False)

            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO articles (url, ticker, sector, industry, date_utc, title, content, json_brut)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (url, ticker_symbol, sector, industry, date_utc, title, content, json_standard))
            
                if cursor.rowcount > 0:
                    logger.info("LIVRAISON : Nouvel article sauvegardé")
                else:
                    logger.info("LIVRAISON : Article déjà présent (ignoré)")
            
                conn.commit()
                logger.debug(f"URL : {url}")
            except Exception as e:
                logger.error(f"Erreur sauvegarde : {e}")

    conn.close()
    logger.info("Fin du pipeline")

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de news financières")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Liste de tickers à scanner (ex: AAPL MSFT TSLA)"
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=None,
        help="Intervalle en minutes entre chaque scan (minimum 20, ex: --loop 30)"
    )
    args = parser.parse_args()

    MAX_TICKERS = 5

    if len(args.tickers) > MAX_TICKERS:
        print(f"Erreur : maximum {MAX_TICKERS} tickers autorisés, tu en as mis {len(args.tickers)}.")
        exit(1)

    if args.loop is not None and args.loop < 20:
        print(f"Erreur : l'intervalle minimum est de 20 minutes (tu as mis {args.loop}).")
        exit(1)

    if args.loop:
        logger.info(f"Mode continu : scan toutes les {args.loop} minutes")
        while True:
            run_news_pipeline(args.tickers)
            logger.info(f"Prochain scan dans {args.loop} minutes...")
            time.sleep(args.loop * 60)
    else:
        run_news_pipeline(args.tickers)