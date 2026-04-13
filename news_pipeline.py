import argparse
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone

import requests
import trafilatura
import yfinance as yf
from newspaper import Article

from status_manager import write_status


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


def load_finnhub_key():
    """Load Finnhub key from .env if available, else from environment variables."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line.startswith("FINNHUB_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    if key and key != "your_finnhub_api_key_here":
                        return key
    return os.environ.get("FINNHUB_API_KEY")


def build_keywords(stock_info):
    keywords = set()

    for field in ["shortName", "longName"]:
        name = stock_info.get(field, "")
        if name:
            keywords.add(name.lower())
            cleaned = re.sub(r"[^a-zA-Z\s-]", " ", name)
            for word in cleaned.split():
                if len(word) > 3:
                    keywords.add(word.lower())

    symbol = stock_info.get("symbol", "")
    if symbol:
        keywords.add(symbol.lower())

    for officer in stock_info.get("companyOfficers", []):
        name = officer.get("name", "")
        if name:
            keywords.add(name.lower())
            parts = name.split()
            if len(parts) >= 2:
                keywords.add(parts[-1].lower())

    website = stock_info.get("website", "")
    if website:
        domain = (
            website.replace("https://", "")
            .replace("http://", "")
            .replace("www.", "")
            .split("/")[0]
        )
        keywords.add(domain.lower())
        if "." in domain:
            pure_domain = domain.rsplit(".", 1)[0]
            if len(pure_domain) > 1:
                keywords.add(pure_domain.lower())

    return keywords


def is_relevant(title, keywords):
    title_lower = (title or "").lower()
    for keyword in keywords:
        if re.search(r"\b" + re.escape(keyword) + r"\b", title_lower):
            return True
    return False


def fetch_content(url, title):
    """Extract full article content using Newspaper first, then Trafilatura fallback."""
    content = ""
    try:
        article = Article(url, request_timeout=10)
        article.download()
        article.parse()
        content = article.text
        logger.debug("Newspaper: %s characters extracted", len(content))
    except Exception as exc:
        logger.warning("Newspaper failed for '%s': %s", title, exc)

    if len(content) < 100:
        logger.debug("Content too short (%s chars), trying trafilatura...", len(content))
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                content = trafilatura.extract(downloaded) or ""
                logger.debug("Trafilatura: %s characters extracted", len(content))
        except Exception as exc:
            logger.warning("Trafilatura failed for '%s': %s", title, exc)

    return content


def ensure_tables(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY,
            ticker TEXT,
            sector TEXT,
            industry TEXT,
            date_utc TEXT,
            title TEXT,
            content TEXT,
            source TEXT DEFAULT 'yahoo',
            json_brut TEXT
        )
        """
    )

    columns = [row[1] for row in cursor.execute("PRAGMA table_info(articles)").fetchall()]
    if "source" not in columns:
        cursor.execute("ALTER TABLE articles ADD COLUMN source TEXT DEFAULT 'yahoo'")
        logger.info("Column 'source' added to articles table")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS articles_filtres (
            url TEXT PRIMARY KEY,
            ticker TEXT,
            title TEXT,
            date_utc TEXT,
            content TEXT,
            motif TEXT,
            match_count INTEGER
        )
        """
    )


def save_filtered_article(cursor, conn, url, ticker_symbol, title, date_utc, content, match_count):
    try:
        cursor.execute(
            """
            INSERT OR IGNORE INTO articles_filtres (
                url, ticker, title, date_utc, content, motif, match_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (url, ticker_symbol, title, date_utc, content, "contenu_hors_sujet", match_count),
        )
        conn.commit()
    except Exception as exc:
        logger.error("Error while saving filtered article: %s", exc)


def save_article(cursor, conn, url, ticker_symbol, sector, industry, date_utc, title, content, source_name):
    payload = json.dumps(
        {"ticker": ticker_symbol, "title": title, "url": url},
        ensure_ascii=False,
    )

    try:
        cursor.execute(
            """
            INSERT OR IGNORE INTO articles (
                url, ticker, sector, industry, date_utc, title, content, source, json_brut
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                url,
                ticker_symbol,
                sector,
                industry,
                date_utc,
                title,
                content,
                source_name,
                payload,
            ),
        )
        inserted = cursor.rowcount > 0
        conn.commit()
        return inserted
    except Exception as exc:
        logger.error("[%s] Save error: %s", source_name, exc)
        return False


def fetch_yahoo(ticker_symbol, stock, keywords, cursor, conn, sector, industry):
    saved = 0
    news_list = stock.news
    logger.debug("Yahoo: %s articles found for %s", len(news_list), ticker_symbol)

    for news_item in news_list:
        content_data = news_item.get("content", {})
        title = content_data.get("title", "")
        date_utc = content_data.get("pubDate", "")

        url = None
        if content_data.get("clickThroughUrl"):
            url = content_data["clickThroughUrl"].get("url")
        elif content_data.get("canonicalUrl"):
            url = content_data["canonicalUrl"].get("url")
        elif content_data.get("previewUrl"):
            url = content_data.get("previewUrl")

        if not url or not title:
            continue

        title_match = is_relevant(title, keywords)
        logger.info("[Yahoo] Processing: %s", title)

        content = fetch_content(url, title)
        if len(content) < 100:
            logger.warning("[Yahoo] Skipped (content too short): %s", title)
            continue

        if not title_match:
            content_lower = content.lower()
            match_count = sum(
                len(re.findall(r"\b" + re.escape(keyword) + r"\b", content_lower))
                for keyword in keywords
            )
            if match_count < 3:
                logger.info("[Yahoo] FILTERED (%s match(es)): %s", match_count, title)
                save_filtered_article(
                    cursor,
                    conn,
                    url,
                    ticker_symbol,
                    title,
                    date_utc,
                    content,
                    match_count,
                )
                continue

        inserted = save_article(
            cursor,
            conn,
            url,
            ticker_symbol,
            sector,
            industry,
            date_utc,
            title,
            content,
            "yahoo",
        )
        if inserted:
            saved += 1
            logger.info("[Yahoo] SAVED: %s", title)
        else:
            logger.info("[Yahoo] Already present: %s", title)

        time.sleep(2)

    return saved


def fetch_finnhub(ticker_symbol, api_key, keywords, cursor, conn, sector, industry):
    saved = 0
    from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    try:
        response = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": ticker_symbol,
                "from": from_date,
                "to": to_date,
                "token": api_key,
            },
            timeout=10,
        )
        response.raise_for_status()
        news_list = response.json()
    except Exception as exc:
        logger.error("[Finnhub] API error for %s: %s", ticker_symbol, exc)
        return 0

    news_list = sorted(
        news_list,
        key=lambda item: item.get("datetime", 0),
        reverse=True,
    )[:5]
    logger.debug("[Finnhub] %s recent articles kept for %s", len(news_list), ticker_symbol)

    for item in news_list:
        title = item.get("headline", "")
        url = item.get("url", "")
        summary = item.get("summary", "")
        timestamp = item.get("datetime", 0)
        date_utc = (
            datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
            if timestamp
            else ""
        )

        if not url or not title:
            continue

        title_match = is_relevant(title, keywords)
        logger.info("[Finnhub] Processing: %s", title)

        content = fetch_content(url, title)
        if len(content) < 100 and len(summary) >= 100:
            content = summary
            logger.debug("[Finnhub] Using summary fallback (%s chars)", len(summary))

        if len(content) < 100:
            logger.warning("[Finnhub] Skipped (content too short): %s", title)
            continue

        if not title_match:
            content_lower = content.lower()
            match_count = sum(
                len(re.findall(r"\b" + re.escape(keyword) + r"\b", content_lower))
                for keyword in keywords
            )
            if match_count < 3:
                logger.info("[Finnhub] FILTERED (%s match(es)): %s", match_count, title)
                save_filtered_article(
                    cursor,
                    conn,
                    url,
                    ticker_symbol,
                    title,
                    date_utc,
                    content,
                    match_count,
                )
                continue

        inserted = save_article(
            cursor,
            conn,
            url,
            ticker_symbol,
            sector,
            industry,
            date_utc,
            title,
            content,
            "finnhub",
        )
        if inserted:
            saved += 1
            logger.info("[Finnhub] SAVED: %s", title)
        else:
            logger.info("[Finnhub] Already present: %s", title)

        time.sleep(1)

    return saved


def run_news_pipeline(tickers):
    finnhub_key = load_finnhub_key()
    if finnhub_key:
        logger.info("Finnhub key loaded - double sourcing enabled (Yahoo + Finnhub)")
    else:
        logger.info("No Finnhub key found - Yahoo Finance sourcing only")

    write_status(
        "sourcing",
        {
            "running": True,
            "ticker_actuel": "",
            "tickers_done": 0,
            "tickers_total": len(tickers),
            "articles_saved": 0,
        },
    )

    logger.debug("Tickers to scan: %s", tickers)
    conn = sqlite3.connect("news_database.db")
    cursor = conn.cursor()
    ensure_tables(cursor)
    conn.commit()

    for index, ticker_symbol in enumerate(tickers):
        logger.info("=== Scan: %s ===", ticker_symbol)
        write_status(
            "sourcing",
            {
                "running": True,
                "ticker_actuel": ticker_symbol,
                "tickers_done": index,
                "tickers_total": len(tickers),
                "articles_saved": cursor.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
            },
        )

        stock = yf.Ticker(ticker_symbol)
        stock_info = stock.info
        sector = stock_info.get("sector", "Inconnu")
        industry = stock_info.get("industry", "Inconnu")
        keywords = build_keywords(stock_info)
        logger.debug("%s keywords: %s", ticker_symbol, keywords)

        saved_yahoo = fetch_yahoo(
            ticker_symbol,
            stock,
            keywords,
            cursor,
            conn,
            sector,
            industry,
        )
        logger.info("[Yahoo] %s new article(s) for %s", saved_yahoo, ticker_symbol)

        if finnhub_key:
            saved_finnhub = fetch_finnhub(
                ticker_symbol,
                finnhub_key,
                keywords,
                cursor,
                conn,
                sector,
                industry,
            )
            logger.info("[Finnhub] %s new article(s) for %s", saved_finnhub, ticker_symbol)

    total = cursor.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    write_status(
        "sourcing",
        {
            "running": False,
            "ticker_actuel": "Termine",
            "tickers_done": len(tickers),
            "tickers_total": len(tickers),
            "articles_saved": total,
        },
    )
    conn.close()
    logger.info("End of news pipeline")


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de news financieres")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--loop", type=int, default=None)
    args = parser.parse_args()

    max_tickers = 5
    if len(args.tickers) > max_tickers:
        print(f"Erreur: maximum {max_tickers} tickers autorises.")
        raise SystemExit(1)

    if args.loop is not None and args.loop < 20:
        print("Erreur: intervalle minimum 20 minutes.")
        raise SystemExit(1)

    if args.loop:
        logger.info("Loop mode: scan every %s minute(s)", args.loop)
        while True:
            run_news_pipeline(args.tickers)
            logger.info("Next scan in %s minute(s)...", args.loop)
            time.sleep(args.loop * 60)
    else:
        run_news_pipeline(args.tickers)
