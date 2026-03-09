import yfinance as yf
from newspaper import Article
import sqlite3
import json
from datetime import datetime, timezone
import argparse

def run_news_pipeline(tickers):
    conn = sqlite3.connect('news_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY, 
            ticker TEXT,
            date_utc TEXT,
            title TEXT,
            content TEXT,
            json_brut TEXT
        )
    ''') 
    conn.commit()
    for ticker_symbol in tickers:
        print(f"1. DÉTECTION : Scan des news pour {ticker_symbol}")
        stock = yf.Ticker(ticker_symbol)
        news_list = stock.news 

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
                continue
            print(f"\nTraitement de l'article : {title}")
            content = ""
            try:
                article = Article(url)
                article.download()
                article.parse()
                content = article.text
                print("  -> Extraction réussie.")
            except Exception as e:
                print(f"  -> Erreur d'extraction (blocage ou format) : {e}")
                continue 

            data_dict = {
                "ticker": ticker_symbol,
                "title": title,
                "date_utc": date_utc,
                "url": url,
                "content": content
            }
            json_standard = json.dumps(data_dict, ensure_ascii=False)

            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO articles (url, ticker, date_utc, title, content, json_brut)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (url, ticker_symbol, date_utc, title, content, json_standard))
            
                if cursor.rowcount > 0:
                    print("LIVRAISON : Nouvel article sauvegardé en base de données")
                else:
                    print("LIVRAISON : Article déjà présent en base (ignoré).")
            
                conn.commit()
            except Exception as e:
                print(f"Erreur lors de la sauvegarde : {e}")

    conn.close()
    print("\nFin du Pipeline")

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