import yfinance as yf
from newspaper import Article
import sqlite3
import json
import requests
import os
from datetime import datetime, timezone, timedelta
import argparse
import logging
import time
import trafilatura
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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


# ─── CHARGEMENT CLÉ FINNHUB ────────────────────────────────────────────────────
def load_finnhub_key():
    """Lit la clé Finnhub depuis .env si elle existe."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("FINNHUB_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    if key and key != "your_finnhub_api_key_here":
                        return key
    return os.environ.get("FINNHUB_API_KEY", None)


# ─── FILTRAGE PAR MOTS-CLÉS ────────────────────────────────────────────────────
def build_keywords(stock_info):
    keywords = set()

    for field in ['shortName', 'longName']:
        name = stock_info.get(field, '')
        if name:
            keywords.add(name.lower())
            cleaned = re.sub(r'[^a-zA-Z\s-]', ' ', name)
            for word in cleaned.split():
                if len(word) > 3:
                    keywords.add(word.lower())

    symbol = stock_info.get('symbol', '')
    if symbol:
        keywords.add(symbol.lower())

    for officer in stock_info.get('companyOfficers', []):
        name = officer.get('name', '')
        if name:
            keywords.add(name.lower())
            parts = name.split()
            if len(parts) >= 2:
                keywords.add(parts[-1].lower())

    website = stock_info.get('website', '')
    if website:
        domain = website.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
        keywords.add(domain.lower())
        if '.' in domain:
            pure = domain.rsplit('.', 1)[0]
            if len(pure) > 1:
                keywords.add(pure.lower())

    return keywords


def is_relevant(title, keywords):
    title_lower = title.lower()
    for kw in keywords:
        if re.search(r'\b' + re.escape(kw) + r'\b', title_lower):
            return True
    return False


# ─── DÉDUPLICATION INTER-SOURCES ──────────────────────────────────────────────
def normalize_title(title):
    """Normalise un titre pour comparaison : minuscules, sans ponctuation."""
    return re.sub(r'[^a-z0-9\s]', '', title.lower()).strip()

def insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, content, motif, match_count):
    """Insère dans articles_filtres seulement si le titre n'existe pas déjà pour ce ticker."""
    existing = cursor.execute(
        "SELECT 1 FROM articles_filtres WHERE ticker = ? AND title = ?", (ticker_symbol, title)
    ).fetchone()
    if existing:
        return
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO articles_filtres (url, ticker, title, date_utc, content, motif, match_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (url, ticker_symbol, title, date_utc, content, motif, match_count))
        conn.commit()
    except Exception as e:
        logger.error(f"Erreur sauvegarde filtre : {e}")


def is_duplicate_title(title, cursor, ticker_symbol):
    """Retourne True si un article avec exactement le même titre existe déjà en base pour ce ticker."""
    norm = normalize_title(title)
    rows = cursor.execute(
        "SELECT title FROM articles WHERE ticker = ?", (ticker_symbol,)
    ).fetchall()
    for (existing_title,) in rows:
        if normalize_title(existing_title) == norm:
            return True, existing_title
    return False, None


# ─── EXTRACTION DU CONTENU COMPLET ────────────────────────────────────────────
FETCH_TIMEOUT = 15  # timeout dur par article (secondes)

def _fetch_newspaper(url):
    article = Article(url, request_timeout=10)
    article.download()
    article.parse()
    return article.text

def _fetch_trafilatura(url):
    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    if resp.ok:
        return trafilatura.extract(resp.text) or ""
    return ""

def fetch_content(url, title):
    """Tente d'extraire le contenu complet d'un article via Newspaper puis Trafilatura.
    Chaque tentative est limitée à FETCH_TIMEOUT secondes via un thread dédié."""
    content = ""

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_fetch_newspaper, url)
        try:
            content = future.result(timeout=FETCH_TIMEOUT) or ""
            logger.debug(f"Newspaper : {len(content)} caractères extraits")
        except FuturesTimeoutError:
            logger.warning(f"Newspaper timeout ({FETCH_TIMEOUT}s) pour '{title}'")
            future.cancel()
            return None  # signal timeout
        except Exception as e:
            logger.warning(f"Newspaper a échoué pour '{title}' : {e}")

    if len(content) < 100:
        logger.debug(f"Contenu trop court ({len(content)} car.), tentative avec trafilatura...")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch_trafilatura, url)
            try:
                content = future.result(timeout=FETCH_TIMEOUT) or ""
                logger.debug(f"Trafilatura : {len(content)} caractères extraits")
            except FuturesTimeoutError:
                logger.warning(f"Trafilatura timeout ({FETCH_TIMEOUT}s) pour '{title}'")
                future.cancel()
                return None  # signal timeout
            except Exception as e:
                logger.warning(f"Trafilatura a échoué pour '{title}' : {e}")

    return content


# ─── SOURCE YAHOO FINANCE ─────────────────────────────────────────────────────
def fetch_yahoo(ticker_symbol, stock, keywords, cursor, conn, sector, industry):
    """Récupère et insère les articles Yahoo Finance pour un ticker."""
    saved = 0
    news_list = stock.news
    logger.debug(f"Yahoo : {len(news_list)} articles trouvés pour {ticker_symbol}")

    for news_item in news_list:
        content_data = news_item.get('content', {})
        title = content_data.get('title', '')
        date_utc = content_data.get('pubDate', "")

        url = None
        if content_data.get('clickThroughUrl'):
            url = content_data['clickThroughUrl'].get('url')
        elif content_data.get('canonicalUrl'):
            url = content_data['canonicalUrl'].get('url')
        elif content_data.get('previewUrl'):
            url = content_data['previewUrl']

        if not url or not title:
            continue

        title_match = is_relevant(title, keywords)
        logger.info(f"[Yahoo] Traitement : {title}")

        content = fetch_content(url, title)

        if content is None:
            logger.warning(f"[Yahoo] Article ignoré (timeout) : {title}")
            insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, "", "timeout", 0)
            continue

        if len(content) < 100:
            logger.warning(f"[Yahoo] Article ignoré (contenu trop court) : {title}")
            continue

        if not title_match:
            content_lower = content.lower()
            count = sum(len(re.findall(r'\b' + re.escape(kw) + r'\b', content_lower)) for kw in keywords)
            if count < 3:
                logger.info(f"[Yahoo] FILTRÉ ({count} match(s)) : {title}")
                insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, content, "contenu_hors_sujet", count)
                continue

        try:
            cursor.execute('''
                INSERT OR IGNORE INTO articles (url, ticker, sector, industry, date_utc, title, content, source, json_brut)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (url, ticker_symbol, sector, industry, date_utc, title, content, "yahoo",
                  json.dumps({"ticker": ticker_symbol, "title": title, "url": url}, ensure_ascii=False)))
            if cursor.rowcount > 0:
                saved += 1
                logger.info(f"[Yahoo] SAUVEGARDÉ : {title}")
            else:
                logger.info(f"[Yahoo] Déjà présent (ignoré) : {title}")
            conn.commit()
        except Exception as e:
            logger.error(f"[Yahoo] Erreur sauvegarde : {e}")

        time.sleep(2)

    return saved


# ─── SOURCE FINNHUB ───────────────────────────────────────────────────────────
def fetch_finnhub(ticker_symbol, api_key, keywords, cursor, conn, sector, industry):
    """Récupère et insère les articles Finnhub pour un ticker (3 derniers jours)."""
    saved = 0
    from_date = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    to_date   = datetime.now().strftime('%Y-%m-%d')

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker_symbol, "from": from_date, "to": to_date, "token": api_key},
            timeout=10
        )
        resp.raise_for_status()
        news_list = resp.json()
    except Exception as e:
        logger.error(f"[Finnhub] Erreur API pour {ticker_symbol} : {e}")
        return 0

    # Garder uniquement les 5 plus récents (triés par datetime desc)
    news_list = sorted(news_list, key=lambda x: x.get('datetime', 0), reverse=True)[:5]
    logger.debug(f"[Finnhub] {len(news_list)} articles retenus pour {ticker_symbol}")

    for item in news_list:
        title   = item.get('headline', '')
        url     = item.get('url', '')
        summary = item.get('summary', '')
        ts      = item.get('datetime', 0)
        date_utc = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else ""

        if not url or not title:
            continue

        title_match = is_relevant(title, keywords)
        logger.info(f"[Finnhub] Traitement : {title}")

        # Tentative de récupération du contenu complet, fallback sur le summary
        content = fetch_content(url, title)
        if content is None:
            if len(summary) >= 100:
                content = summary
                logger.debug(f"[Finnhub] Timeout — fallback sur summary ({len(summary)} car.)")
            else:
                logger.warning(f"[Finnhub] Article ignoré (timeout + summary insuffisant) : {title}")
                insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, "", "timeout", 0)
                continue
        elif len(content) < 100 and len(summary) >= 100:
            content = summary
            logger.debug(f"[Finnhub] Fallback sur summary ({len(summary)} car.)")

        if len(content) < 100:
            logger.warning(f"[Finnhub] Article ignoré (contenu trop court) : {title}")
            continue

        if not title_match:
            content_lower = content.lower()
            count = sum(len(re.findall(r'\b' + re.escape(kw) + r'\b', content_lower)) for kw in keywords)
            if count < 3:
                logger.info(f"[Finnhub] FILTRÉ ({count} match(s)) : {title}")
                insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, content, "contenu_hors_sujet", count)
                continue

        dup, dup_title = is_duplicate_title(title, cursor, ticker_symbol)
        if dup:
            logger.info(f"[Finnhub] DOUBLON avec Yahoo ignoré : '{title}' ≈ '{dup_title}'")
            continue

        try:
            cursor.execute('''
                INSERT OR IGNORE INTO articles (url, ticker, sector, industry, date_utc, title, content, source, json_brut)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (url, ticker_symbol, sector, industry, date_utc, title, content, "finnhub",
                  json.dumps({"ticker": ticker_symbol, "title": title, "url": url}, ensure_ascii=False)))
            if cursor.rowcount > 0:
                saved += 1
                logger.info(f"[Finnhub] SAUVEGARDÉ : {title}")
            else:
                logger.info(f"[Finnhub] Déjà présent (ignoré) : {title}")
            conn.commit()
        except Exception as e:
            logger.error(f"[Finnhub] Erreur sauvegarde : {e}")

        time.sleep(1)

    return saved


# ─── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────
def run_news_pipeline(tickers):
    finnhub_key = load_finnhub_key()
    if finnhub_key:
        logger.info("Clé Finnhub chargée — double sourcing activé (Yahoo + Finnhub)")
    else:
        logger.info("Pas de clé Finnhub — sourcing Yahoo Finance uniquement")

    write_status("sourcing", {"running": True, "ticker_actuel": "", "tickers_done": 0, "tickers_total": len(tickers), "articles_saved": 0})
    logger.debug(f"Tickers à scanner : {tickers}")

    conn   = sqlite3.connect('news_database.db')
    cursor = conn.cursor()

    # Table articles avec colonne source
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            url       TEXT PRIMARY KEY,
            ticker    TEXT,
            sector    TEXT,
            industry  TEXT,
            date_utc  TEXT,
            title     TEXT,
            content   TEXT,
            source    TEXT DEFAULT 'yahoo',
            json_brut TEXT
        )
    ''')

    # Migration : ajouter source si absente
    colonnes = [row[1] for row in cursor.execute("PRAGMA table_info(articles)").fetchall()]
    if "source" not in colonnes:
        cursor.execute("ALTER TABLE articles ADD COLUMN source TEXT DEFAULT 'yahoo'")
        logger.info("Colonne source ajoutée à la table articles")

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles_filtres (
            url         TEXT PRIMARY KEY,
            ticker      TEXT,
            title       TEXT,
            date_utc    TEXT,
            content     TEXT,
            motif       TEXT,
            match_count INTEGER
        )
    ''')
    conn.commit()

    for i, ticker_symbol in enumerate(tickers):
        logger.info(f"=== Scan : {ticker_symbol} ===")
        write_status("sourcing", {
            "running": True,
            "ticker_actuel": ticker_symbol,
            "tickers_done": i,
            "tickers_total": len(tickers),
            "articles_saved": cursor.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        })

        stock    = yf.Ticker(ticker_symbol)
        sector   = stock.info.get('sector', 'Inconnu')
        industry = stock.info.get('industry', 'Inconnu')
        keywords = build_keywords(stock.info)
        logger.debug(f"{ticker_symbol} : mots-clés = {keywords}")

        # Yahoo Finance
        saved_yahoo = fetch_yahoo(ticker_symbol, stock, keywords, cursor, conn, sector, industry)
        logger.info(f"[Yahoo] {saved_yahoo} nouveaux articles pour {ticker_symbol}")

        # Finnhub (si clé disponible)
        if finnhub_key:
            saved_finnhub = fetch_finnhub(ticker_symbol, finnhub_key, keywords, cursor, conn, sector, industry)
            logger.info(f"[Finnhub] {saved_finnhub} nouveaux articles pour {ticker_symbol}")

    total = cursor.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    write_status("sourcing", {
        "running": False,
        "ticker_actuel": "Terminé",
        "tickers_done": len(tickers),
        "tickers_total": len(tickers),
        "articles_saved": total
    })
    conn.close()
    logger.info("Fin du pipeline")


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de news financières")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--loop", type=int, default=None)
    args = parser.parse_args()

    MAX_TICKERS = 5
    if len(args.tickers) > MAX_TICKERS:
        print(f"Erreur : maximum {MAX_TICKERS} tickers autorisés.")
        exit(1)

    if args.loop is not None and args.loop < 20:
        print(f"Erreur : intervalle minimum 20 minutes.")
        exit(1)

    if args.loop:
        logger.info(f"Mode continu : scan toutes les {args.loop} minutes")
        while True:
            run_news_pipeline(args.tickers)
            logger.info(f"Prochain scan dans {args.loop} minutes...")
            time.sleep(args.loop * 60)
    else:
        run_news_pipeline(args.tickers)
