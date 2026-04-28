from src.utils.logger import get_logger

logger = get_logger(__name__)
import argparse
import json
import logging
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
import trafilatura
import yfinance as yf
from newspaper import Article

from src.pipelines.agent_filtrage_api import est_pertinent

# Imports absolus (mode editable `pip install -e .` requis, cf. ADR-008).
# Plus de sys.path.insert : le projet expose ses modules via `src.x` partout.
from src.pipelines.status_manager import write_status

# --- Couche d'enrichissement (BATCH 1) -------------------------------------
# Import defensif : si l'un des modules manque (deps optionnelles non installees),
# le pipeline tombe en mode legacy sans enrichissement.
try:
    from src.utils.article_enrichment import (
        HistoricalArticle,
        enrich_article,
        ensure_enrichment_schema,
    )

    ENRICHMENT_AVAILABLE = True
except Exception as _enrich_err:  # pragma: no cover
    ENRICHMENT_AVAILABLE = False
    _ENRICH_ERR = str(_enrich_err)

# Création des dossiers nécessaires
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logger = logging.getLogger("NewsPipeline")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

# Rotation auto : pipeline.log, pipeline.log.1 … pipeline.log.5 (10 MB × 5 = 60 MB).
from logging.handlers import RotatingFileHandler

file_handler = RotatingFileHandler(
    "logs/pipeline.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8",
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ─── CHARGEMENT CLÉ FINNHUB ────────────────────────────────────────────────────
def load_finnhub_key():
    """Lit la clé Finnhub depuis .env à la racine du projet."""
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
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
        domain = website.replace("https://", "").replace("http://", "").replace("www.", "").split("/")[0]
        keywords.add(domain.lower())
        if "." in domain:
            pure = domain.rsplit(".", 1)[0]
            if len(pure) > 1:
                keywords.add(pure.lower())

    return keywords


def is_relevant(title, keywords):
    title_lower = title.lower()
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", title_lower):
            return True
    return False


# ─── DÉDUPLICATION INTER-SOURCES ──────────────────────────────────────────────
def normalize_title(title):
    """Normalise un titre pour comparaison : minuscules, sans ponctuation."""
    return re.sub(r"[^a-z0-9\s]", "", title.lower()).strip()


def insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, content, motif, match_count):
    """Insère dans articles_filtres seulement si le titre n'existe pas déjà pour ce ticker."""
    existing = cursor.execute(
        "SELECT 1 FROM articles_filtres WHERE ticker = ? AND title = ?", (ticker_symbol, title)
    ).fetchone()
    if existing:
        return
    try:
        cursor.execute(
            """
            INSERT OR IGNORE INTO articles_filtres (url, ticker, title, date_utc, content, motif, match_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (url, ticker_symbol, title, date_utc, content, motif, match_count),
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Erreur sauvegarde filtre : {e}")


def is_duplicate_title(title, cursor, ticker_symbol):
    """Retourne True si un article avec exactement le même titre existe déjà en base pour ce ticker."""
    norm = normalize_title(title)
    rows = cursor.execute("SELECT title FROM articles WHERE ticker = ?", (ticker_symbol,)).fetchall()
    for (existing_title,) in rows:
        if normalize_title(existing_title) == norm:
            return True, existing_title
    return False, None


# ─── HELPERS ENRICHISSEMENT ───────────────────────────────────────────────────
def _parse_date_utc(date_str):
    """Tolere ISO ou RFC 2822 ; renvoie datetime UTC ou None."""
    if not date_str:
        return None
    try:
        if isinstance(date_str, datetime):
            dt = date_str
        else:
            s = str(date_str).replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _load_recent_history(cursor, ticker_symbol, hours=48, limit=200):
    """Charge l'historique des articles recents pour scoring de nouveaute."""
    if not ENRICHMENT_AVAILABLE:
        return []
    since = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()
    rows = cursor.execute(
        "SELECT url AS article_id, date_utc AS published_at, title, content AS body "
        "FROM articles WHERE ticker = ? AND date_utc >= ? ORDER BY date_utc DESC LIMIT ?",
        (ticker_symbol, since, limit),
    ).fetchall()
    history = []
    for aid, pub, title, body in rows:
        pub_dt = _parse_date_utc(pub)
        if pub_dt is None:
            continue
        history.append(
            HistoricalArticle(
                article_id=aid or "",
                published_at=pub_dt,
                title=title or "",
                body=body or "",
            )
        )
    return history


def _load_recent_titles_bodies(cursor, ticker_symbol, hours=48, limit=200):
    """Charge les articles recents pour check near-duplicate final."""
    if not ENRICHMENT_AVAILABLE:
        return []
    since = (datetime.now(tz=timezone.utc) - timedelta(hours=hours)).isoformat()
    rows = cursor.execute(
        "SELECT title, content FROM articles WHERE ticker = ? AND date_utc >= ? ORDER BY date_utc DESC LIMIT ?",
        (ticker_symbol, since, limit),
    ).fetchall()
    return [{"title": t or "", "content": c or ""} for (t, c) in rows]


def _insert_article_enriched(
    cursor, conn, *, url, ticker, sector, industry, date_utc, title, content, source, enriched
):
    """
    Insert qui cohabite avec le schema legacy : si `enriched` est None ou
    enrichissement indisponible, on fait l'insert classique. Sinon, on ajoute
    les colonnes d'enrichissement en sus.
    """
    base_cols = ["url", "ticker", "sector", "industry", "date_utc", "title", "content", "source", "json_brut"]
    base_vals = [
        url,
        ticker,
        sector,
        industry,
        date_utc,
        title,
        content,
        source,
        json.dumps({"ticker": ticker, "title": title, "url": url}, ensure_ascii=False),
    ]

    if enriched is None:
        sql = f"INSERT OR IGNORE INTO articles ({', '.join(base_cols)}) VALUES ({', '.join(['?'] * len(base_cols))})"
        cursor.execute(sql, base_vals)
        conn.commit()
        return

    enrich_cols = [
        "lang",
        "source_tier",
        "source_name",
        "source_weight",
        "is_press_release",
        "pr_confidence",
        "event_type",
        "event_confidence",
        "entities_json",
        "novelty_score",
        "novelty_decision",
    ]
    enrich_vals = [
        enriched.lang,
        int(enriched.source_tier),
        enriched.source_name,
        float(enriched.source_weight),
        1 if enriched.is_press_release else 0,
        float(enriched.pr_confidence),
        enriched.event_type,
        float(enriched.event_confidence),
        json.dumps(enriched.entities, ensure_ascii=False),
        float(enriched.novelty_score),
        enriched.novelty_decision,
    ]
    all_cols = base_cols + enrich_cols
    all_vals = base_vals + enrich_vals
    sql = f"INSERT OR IGNORE INTO articles ({', '.join(all_cols)}) VALUES ({', '.join(['?'] * len(all_cols))})"
    cursor.execute(sql, all_vals)
    conn.commit()


def _apply_enrichment(cursor, url, ticker, title, body, date_str, company_name, logger_prefix):
    """
    Lance l'enrichissement et retourne (accept, reject_reason, enriched).
    Sur accept=False, l'appelant doit skipper l'article.
    Sur accept=True, enriched contient les metadonnees a persister.
    """
    if not ENRICHMENT_AVAILABLE:
        return True, "", None

    try:
        history = _load_recent_history(cursor, ticker)
        dups_against = _load_recent_titles_bodies(cursor, ticker)
        enriched = enrich_article(
            url=url,
            ticker=ticker,
            title=title,
            body=body,
            published_at=_parse_date_utc(date_str),
            company_name=company_name,
            history=history,
            check_duplicates_against=dups_against,
        )
    except Exception as e:
        logger.error(f"{logger_prefix} enrichissement erreur : {e}")
        return True, "", None  # en cas d'erreur, on laisse passer

    if not enriched.accept:
        logger.info(f"{logger_prefix} REJET enrichissement ({enriched.reject_reason}) : {title[:60]}")
        return False, enriched.reject_reason, enriched

    logger.debug(
        f"{logger_prefix} enrich: lang={enriched.lang} tier={enriched.source_tier} "
        f"event={enriched.event_type} pr={enriched.is_press_release} "
        f"novelty={enriched.novelty_score:.2f}"
    )
    return True, "", enriched


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
def fetch_yahoo(ticker_symbol, company_name, stock, keywords, cursor, conn, sector, industry):
    """Récupère et insère les articles Yahoo Finance pour un ticker."""
    saved = 0
    news_list = stock.news
    logger.debug(f"Yahoo : {len(news_list)} articles trouvés pour {ticker_symbol}")

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
            url = content_data["previewUrl"]

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

        # Enrichissement BATCH 1 : langue, source, event, PR, novelty, dedup
        ok_enrich, reason_enrich, enriched = _apply_enrichment(
            cursor, url, ticker_symbol, title, content, date_utc, company_name, "[Yahoo]"
        )
        if not ok_enrich:
            insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, content, f"enrich:{reason_enrich}", 0)
            continue

        pertinent, motif_ia = est_pertinent(ticker_symbol, company_name, title, content)
        if not pertinent:
            insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, content, motif_ia, 0)
            continue

        if not title_match:
            content_lower = content.lower()
            count = sum(len(re.findall(r"\b" + re.escape(kw) + r"\b", content_lower)) for kw in keywords)
            if count < 3:
                logger.info(f"[Yahoo] FILTRÉ ({count} match(s)) : {title}")
                insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, content, "contenu_hors_sujet", count)
                continue

        try:
            _insert_article_enriched(
                cursor,
                conn,
                url=url,
                ticker=ticker_symbol,
                sector=sector,
                industry=industry,
                date_utc=date_utc,
                title=title,
                content=content,
                source="yahoo",
                enriched=enriched,
            )
            if cursor.rowcount > 0:
                saved += 1
                logger.info(f"[Yahoo] SAUVEGARDÉ : {title}")
            else:
                logger.info(f"[Yahoo] Déjà présent (ignoré) : {title}")
        except Exception as e:
            logger.error(f"[Yahoo] Erreur sauvegarde : {e}")

        time.sleep(2)

    return saved


# ─── SOURCE FINNHUB ───────────────────────────────────────────────────────────
def fetch_finnhub(ticker_symbol, company_name, api_key, keywords, cursor, conn, sector, industry):
    """Récupère et insère les articles Finnhub pour un ticker (3 derniers jours)."""
    saved = 0
    from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    to_date = datetime.now().strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={"symbol": ticker_symbol, "from": from_date, "to": to_date, "token": api_key},
            timeout=10,
        )
        resp.raise_for_status()
        news_list = resp.json()
    except Exception as e:
        logger.error(f"[Finnhub] Erreur API pour {ticker_symbol} : {e}")
        return 0

    # Garder uniquement les 5 plus récents (triés par datetime desc)
    news_list = sorted(news_list, key=lambda x: x.get("datetime", 0), reverse=True)[:5]
    logger.debug(f"[Finnhub] {len(news_list)} articles retenus pour {ticker_symbol}")

    for item in news_list:
        title = item.get("headline", "")
        url = item.get("url", "")
        summary = item.get("summary", "")
        ts = item.get("datetime", 0)
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

        # Enrichissement BATCH 1 (inclut le near-duplicate qui couvre l'ancien
        # is_duplicate_title, mais on le garde en filet de securite).
        ok_enrich, reason_enrich, enriched = _apply_enrichment(
            cursor, url, ticker_symbol, title, content, date_utc, company_name, "[Finnhub]"
        )
        if not ok_enrich:
            insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, content, f"enrich:{reason_enrich}", 0)
            continue

        pertinent, motif_ia = est_pertinent(ticker_symbol, company_name, title, content)
        if not pertinent:
            insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, content, motif_ia, 0)
            continue

        if not title_match:
            content_lower = content.lower()
            count = sum(len(re.findall(r"\b" + re.escape(kw) + r"\b", content_lower)) for kw in keywords)
            if count < 3:
                logger.info(f"[Finnhub] FILTRÉ ({count} match(s)) : {title}")
                insert_filtre(cursor, conn, url, ticker_symbol, title, date_utc, content, "contenu_hors_sujet", count)
                continue

        dup, dup_title = is_duplicate_title(title, cursor, ticker_symbol)
        if dup:
            logger.info(f"[Finnhub] DOUBLON avec Yahoo ignoré : '{title}' ≈ '{dup_title}'")
            continue

        try:
            _insert_article_enriched(
                cursor,
                conn,
                url=url,
                ticker=ticker_symbol,
                sector=sector,
                industry=industry,
                date_utc=date_utc,
                title=title,
                content=content,
                source="finnhub",
                enriched=enriched,
            )
            if cursor.rowcount > 0:
                saved += 1
                logger.info(f"[Finnhub] SAUVEGARDÉ : {title}")
            else:
                logger.info(f"[Finnhub] Déjà présent (ignoré) : {title}")
        except Exception as e:
            logger.error(f"[Finnhub] Erreur sauvegarde : {e}")

        time.sleep(1)

    return saved


# ─── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────
def run_news_pipeline(tickers, limit=None):
    finnhub_key = load_finnhub_key()
    if finnhub_key:
        logger.info("Clé Finnhub chargée — double sourcing activé (Yahoo + Finnhub)")
    else:
        logger.info("Pas de clé Finnhub — sourcing Yahoo Finance uniquement")

    write_status(
        "sourcing",
        {"running": True, "ticker_actuel": "", "tickers_done": 0, "tickers_total": len(tickers), "articles_saved": 0},
    )
    logger.debug(f"Tickers à scanner : {tickers}")

    conn = sqlite3.connect("data/news_database.db", timeout=15)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    # Table articles complète
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY, 
            date_utc TEXT,
            ticker TEXT,
            sector TEXT,
            industry TEXT,
            title TEXT,
            content TEXT,
            signal_filtrage TEXT,
            score_filtrage REAL,
            absa_json TEXT,
            signal_final TEXT,
            consensus_rate REAL,
            impact_strength REAL,
            argument_dominant TEXT,
            consensus_model TEXT,
            transcription_debat TEXT,
            json_brut TEXT,
            source TEXT DEFAULT 'yahoo'
        )
    """)

    # Migration : ajouter source si absente
    colonnes = [row[1] for row in cursor.execute("PRAGMA table_info(articles)").fetchall()]
    if "source" not in colonnes:
        cursor.execute("ALTER TABLE articles ADD COLUMN source TEXT DEFAULT 'yahoo'")
        logger.info("Colonne source ajoutée à la table articles")

    # Migration : colonnes d'enrichissement BATCH 1
    if ENRICHMENT_AVAILABLE:
        added = ensure_enrichment_schema(cursor)
        if added:
            logger.info(f"Colonnes d'enrichissement ajoutées : {added}")
        conn.commit()
    else:
        logger.warning(f"Enrichissement indisponible, mode legacy : {_ENRICH_ERR}")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles_filtres (
            url         TEXT PRIMARY KEY,
            ticker      TEXT,
            title       TEXT,
            date_utc    TEXT,
            content     TEXT,
            motif       TEXT,
            match_count INTEGER
        )
    """)
    conn.commit()

    for i, ticker_symbol in enumerate(tickers):
        logger.info(f"=== Scan : {ticker_symbol} ===")
        write_status(
            "sourcing",
            {
                "running": True,
                "ticker_actuel": ticker_symbol,
                "tickers_done": i,
                "tickers_total": len(tickers),
                "articles_saved": cursor.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
            },
        )

        stock = yf.Ticker(ticker_symbol)
        sector = stock.info.get("sector", "Inconnu")
        industry = stock.info.get("industry", "Inconnu")
        company_name = stock.info.get("shortName", ticker_symbol)
        keywords = build_keywords(stock.info)
        logger.debug(f"{ticker_symbol} ({company_name}) : mots-clés = {keywords}")

        # Yahoo Finance
        saved_yahoo = fetch_yahoo(ticker_symbol, company_name, stock, keywords, cursor, conn, sector, industry)
        logger.info(f"[Yahoo] {saved_yahoo} nouveaux articles pour {ticker_symbol}")

        # Finnhub (si clé disponible)
        if finnhub_key:
            saved_finnhub = fetch_finnhub(
                ticker_symbol, company_name, finnhub_key, keywords, cursor, conn, sector, industry
            )
            logger.info(f"[Finnhub] {saved_finnhub} nouveaux articles pour {ticker_symbol}")

    total = cursor.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    write_status(
        "sourcing",
        {
            "running": False,
            "ticker_actuel": "Terminé",
            "tickers_done": len(tickers),
            "tickers_total": len(tickers),
            "articles_saved": total,
        },
    )
    conn.close()
    logger.info("Fin du pipeline")


DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de news financières")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--limit", type=int, default=None, help="Limite d'articles à traiter")
    parser.add_argument("--loop", type=int, default=None)
    args = parser.parse_args()

    MAX_TICKERS = 5
    if len(args.tickers) > MAX_TICKERS:
        logger.info(f"Erreur : maximum {MAX_TICKERS} tickers autorisés.")
        exit(1)

    if args.loop is not None and args.loop < 20:
        logger.info("Erreur : intervalle minimum 20 minutes.")
        exit(1)

    if args.loop:
        logger.info(f"Mode continu : scan toutes les {args.loop} minutes")
        while True:
            run_news_pipeline(args.tickers, limit=args.limit)
            logger.info(f"Prochain scan dans {args.loop} minutes...")
            time.sleep(args.loop * 60)
    else:
        run_news_pipeline(args.tickers, limit=args.limit)
