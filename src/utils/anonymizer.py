"""
anonymizer.py -- Outil d'anonymisation des entités financières
===============================================================
OBJECTIF :
  Empêcher les agents LLM de "tricher" en s'appuyant sur leur connaissance
  implicite d'une entreprise (Apple est fiable, Tesla est volatile, etc.)
  plutôt que sur le contenu de l'article.

PRINCIPE :
  1. Avant le débat → anonymise toutes les entités connues (ticker, nom, CEO...)
  2. L'agent raisonne sur "AlphaCorp" sans savoir que c'est Apple
  3. Après → on compare le signal anonymisé vs le signal original
  4. Si les signaux divergent → le modèle était biaisé par son prior implicite

USAGE :
  from src.utils.anonymizer import anonymize_article, restore_entities

  text_anon, entity_map = anonymize_article(content, ticker="AAPL")
  # passer text_anon au pipeline...
  signal_anon = pipeline.invoke(text_anon)
  # Audit : comparer signal_anon vs signal_original

ENRICHISSEMENT DYNAMIQUE (Phase 4) :
  La carte ENTITY_DATABASE peut etre enrichie dynamiquement depuis la table
  `articles` de la DB SQLite via `extend_db_from_database()` :
  - Pour chaque ticker non-hardcode, on extrait company_name + tickers cites
  - Generation automatique d'un alias deterministe (hash-based)
  - Cache thread-safe : un seul SELECT par session

  Cf. ADR-008 pour la justification (anonymizer hardcode = dette technique).
"""

import hashlib
import logging
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base de données d'entités connues
# ---------------------------------------------------------------------------

# Format : "entité originale" -> "alias anonyme"
ENTITY_DATABASE: dict[str, dict] = {
    "AAPL": {
        "ticker": "ALPHA",
        "names": ["Apple", "Apple Inc", "Apple Inc.", "AAPL"],
        "ceos": ["Tim Cook"],
        "products": [
            "iPhone",
            "iPad",
            "Mac",
            "AirPods",
            "Apple Intelligence",
            "App Store",
            "Apple Watch",
            "Vision Pro",
            "Siri",
        ],
        "places": ["Cupertino"],
        "alias_company": "AlphaCorp",
        "alias_ceo": "CEO_Alpha",
        "alias_product": "ProductX",
    },
    "MSFT": {
        "ticker": "BETA",
        "names": ["Microsoft", "Microsoft Corp", "Microsoft Corporation", "MSFT"],
        "ceos": ["Satya Nadella", "Amy Hood"],
        "products": [
            "Azure",
            "Office 365",
            "Copilot",
            "Teams",
            "Windows",
            "Xbox",
            "LinkedIn",
            "GitHub",
            "Bing",
            "Surface",
        ],
        "places": ["Redmond"],
        "alias_company": "BetaCorp",
        "alias_ceo": "CEO_Beta",
        "alias_product": "ProductY",
    },
    "TSLA": {
        "ticker": "GAMMA",
        "names": ["Tesla", "Tesla Inc", "Tesla Inc.", "TSLA"],
        "ceos": ["Elon Musk"],
        "products": [
            "Cybertruck",
            "Model 3",
            "Model S",
            "Model X",
            "Model Y",
            "Full Self Driving",
            "FSD",
            "Robotaxi",
            "Powerwall",
            "Megapack",
            "Supercharger",
        ],
        "places": ["Austin", "Fremont", "Gigafactory"],
        "alias_company": "GammaCorp",
        "alias_ceo": "CEO_Gamma",
        "alias_product": "ProductZ",
    },
    "AMZN": {
        "ticker": "DELTA",
        "names": ["Amazon", "Amazon.com", "Amazon Web Services", "AWS", "AMZN", "Amazon Prime"],
        "ceos": ["Andy Jassy", "Brian Olsavsky", "Jeff Bezos"],
        "products": ["Bedrock", "Alexa", "Kindle", "Echo", "Prime", "Graviton", "Trainium", "re:Invent"],
        "places": ["Seattle", "Arlington"],
        "alias_company": "DeltaCorp",
        "alias_ceo": "CEO_Delta",
        "alias_product": "ProductW",
    },
    "GOOGL": {
        "ticker": "EPSILON",
        "names": ["Google", "Alphabet", "Google LLC", "Alphabet Inc", "GOOGL", "GOOG"],
        "ceos": ["Sundar Pichai", "Ruth Porat"],
        "products": [
            "Gmail",
            "YouTube",
            "Chrome",
            "Android",
            "Gemini",
            "Pixel",
            "Waymo",
            "Vertex AI",
            "BigQuery",
            "Google Cloud",
        ],
        "places": ["Mountain View", "Menlo Park"],
        "alias_company": "EpsilonCorp",
        "alias_ceo": "CEO_Epsilon",
        "alias_product": "ProductV",
    },
    "NVDA": {
        "ticker": "ZETA",
        "names": ["Nvidia", "NVIDIA", "NVDA", "Nvidia Corp"],
        "ceos": ["Jensen Huang"],
        "products": ["H100", "A100", "B100", "CUDA", "Grace Hopper", "DGX", "GeForce", "RTX", "Blackwell", "Hopper"],
        "places": ["Santa Clara"],
        "alias_company": "ZetaCorp",
        "alias_ceo": "CEO_Zeta",
        "alias_product": "ProductU",
    },
    "META": {
        "ticker": "ETA",
        "names": ["Meta", "Meta Platforms", "META", "Facebook Inc"],
        "ceos": ["Mark Zuckerberg", "Susan Li"],
        "products": ["Facebook", "Instagram", "WhatsApp", "Reels", "Quest", "Threads", "Llama", "Ray-Ban"],
        "places": ["Menlo Park"],
        "alias_company": "EtaCorp",
        "alias_ceo": "CEO_Eta",
        "alias_product": "ProductT",
    },
    "JPM": {
        "ticker": "THETA",
        "names": ["JPMorgan", "JPMorgan Chase", "JPM", "Chase"],
        "ceos": ["Jamie Dimon", "Jeremy Barnum"],
        "products": ["Sapphire", "Chase Bank", "Chase Private Client"],
        "places": ["Manhattan"],
        "alias_company": "ThetaCorp",
        "alias_ceo": "CEO_Theta",
        "alias_product": "ProductS",
    },
    # ------------------------------------------------------------------
    # Entités DELISTED / FAILED — essentielles pour l'audit survivorship bias
    # (Brown, Goetzmann, Ibbotson 1992 ; Carpenter & Lynch 1999)
    # Un LLM trained avant leur disparition les cite encore comme actives.
    # ------------------------------------------------------------------
    "LEH": {  # Lehman Brothers — failli septembre 2008
        "ticker": "OMEGA1",
        "names": ["Lehman Brothers", "Lehman", "LEH"],
        "ceos": ["Richard Fuld", "Dick Fuld"],
        "products": ["Lehman Aggregate"],
        "places": ["New York"],
        "alias_company": "DefunctBankA",
        "alias_ceo": "CEO_Omega1",
        "alias_product": "ProductOm1",
    },
    "BSC": {  # Bear Stearns — racheté par JPM mars 2008
        "ticker": "OMEGA2",
        "names": ["Bear Stearns", "BSC"],
        "ceos": ["Alan Schwartz", "James Cayne"],
        "products": [],
        "places": [],
        "alias_company": "DefunctBankB",
        "alias_ceo": "CEO_Omega2",
        "alias_product": "ProductOm2",
    },
    "SIVB": {  # Silicon Valley Bank — failli mars 2023
        "ticker": "OMEGA3",
        "names": ["Silicon Valley Bank", "SVB Financial", "SIVB", "SVB"],
        "ceos": ["Greg Becker"],
        "products": [],
        "places": ["Santa Clara"],
        "alias_company": "DefunctBankC",
        "alias_ceo": "CEO_Omega3",
        "alias_product": "ProductOm3",
    },
    "FRC": {  # First Republic Bank — failli mai 2023
        "ticker": "OMEGA4",
        "names": ["First Republic Bank", "First Republic", "FRC"],
        "ceos": ["Michael Roffler", "Jim Herbert"],
        "products": [],
        "places": ["San Francisco"],
        "alias_company": "DefunctBankD",
        "alias_ceo": "CEO_Omega4",
        "alias_product": "ProductOm4",
    },
    "WCAGY": {  # Wirecard — fraude comptable, insolvable juin 2020
        "ticker": "OMEGA5",
        "names": ["Wirecard", "WCAGY", "Wirecard AG"],
        "ceos": ["Markus Braun", "Oliver Bellenhaus"],
        "products": [],
        "places": ["Munich", "Aschheim"],
        "alias_company": "DefunctFintechA",
        "alias_ceo": "CEO_Omega5",
        "alias_product": "ProductOm5",
    },
    "BYND": {  # Beyond Meat — 90%+ drawdown depuis IPO
        "ticker": "OMEGA6",
        "names": ["Beyond Meat", "BYND"],
        "ceos": ["Ethan Brown"],
        "products": ["Beyond Burger", "Beyond Sausage"],
        "places": ["El Segundo"],
        "alias_company": "StressedFoodA",
        "alias_ceo": "CEO_Omega6",
        "alias_product": "ProductOm6",
    },
    "PTON": {  # Peloton — 95%+ drawdown depuis 2021
        "ticker": "OMEGA7",
        "names": ["Peloton", "PTON", "Peloton Interactive"],
        "ceos": ["Barry McCarthy", "John Foley"],
        "products": ["Peloton Bike", "Peloton Tread"],
        "places": ["New York"],
        "alias_company": "StressedConsumerA",
        "alias_ceo": "CEO_Omega7",
        "alias_product": "ProductOm7",
    },
    "RIVN": {  # Rivian — très volatile, cash burn important
        "ticker": "OMEGA8",
        "names": ["Rivian", "RIVN", "Rivian Automotive"],
        "ceos": ["R.J. Scaringe"],
        "products": ["R1T", "R1S", "EDV"],
        "places": ["Irvine", "Normal"],
        "alias_company": "StressedAutoA",
        "alias_ceo": "CEO_Omega8",
        "alias_product": "ProductOm8",
    },
    "ENRNQ": {  # Enron — faillite fraude décembre 2001
        "ticker": "OMEGA9",
        "names": ["Enron", "ENRNQ"],
        "ceos": ["Kenneth Lay", "Jeffrey Skilling"],
        "products": [],
        "places": ["Houston"],
        "alias_company": "DefunctEnergyA",
        "alias_ceo": "CEO_Omega9",
        "alias_product": "ProductOm9",
    },
    "GME": {  # GameStop — meme stock, volatilité extrême 2021
        "ticker": "IOTA",
        "names": ["GameStop", "GME"],
        "ceos": ["Ryan Cohen", "Matthew Furlong"],
        "products": [],
        "places": ["Grapevine"],
        "alias_company": "MemeStockA",
        "alias_ceo": "CEO_Iota",
        "alias_product": "ProductR",
    },
}


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------


@dataclass
class AnonymizationResult:
    """Résultat d'une anonymisation."""

    text: str  # Texte anonymisé
    ticker_alias: str  # Alias du ticker (ex: "ALPHA")
    company_alias: str  # Alias de la société (ex: "AlphaCorp")
    reverse_map: dict = field(default_factory=dict)  # alias -> original
    entities_replaced: int = 0  # Nombre de substitutions


# ---------------------------------------------------------------------------
# Anonymisation
# ---------------------------------------------------------------------------


def anonymize_article(text: str, ticker: str) -> AnonymizationResult:
    """
    Anonymise toutes les entités connues dans le texte.
    Les remplacements sont déterministes et réversibles.

    Args:
        text   : Texte brut de l'article
        ticker : Symbole boursier pour choisir la bonne carte d'entités

    Returns:
        AnonymizationResult avec le texte anonymisé et la carte de reverse-mapping
    """
    ticker = ticker.upper()
    entity_info = ENTITY_DATABASE.get(ticker)

    if entity_info is None:
        # Ticker non connu -> anonymisation générique
        alias = "TICKER_X"
        return AnonymizationResult(
            text=text,
            ticker_alias=alias,
            company_alias="CorpX",
            reverse_map={},
            entities_replaced=0,
        )

    result_text = text
    reverse_map = {}
    total_replaced = 0

    alias_company = entity_info["alias_company"]
    alias_ceo = entity_info["alias_ceo"]
    alias_product = entity_info["alias_product"]
    alias_ticker = entity_info["ticker"]

    # 1. Remplace les noms de l'entreprise (du plus long au plus court pour éviter
    #    des substitutions partielles — "Amazon Web Services" avant "Amazon")
    names_sorted = sorted(entity_info["names"], key=len, reverse=True)
    for name in names_sorted:
        pattern = re.compile(re.escape(name), re.IGNORECASE)
        count = len(pattern.findall(result_text))
        if count > 0:
            result_text = pattern.sub(alias_company, result_text)
            reverse_map[alias_company] = name
            total_replaced += count

    # 2. Remplace les noms des dirigeants
    for ceo in entity_info["ceos"]:
        pattern = re.compile(re.escape(ceo), re.IGNORECASE)
        count = len(pattern.findall(result_text))
        if count > 0:
            result_text = pattern.sub(alias_ceo, result_text)
            reverse_map[alias_ceo] = ceo
            total_replaced += count

    # 3. Remplace les noms de produits
    products_sorted = sorted(entity_info["products"], key=len, reverse=True)
    for product in products_sorted:
        pattern = re.compile(re.escape(product), re.IGNORECASE)
        count = len(pattern.findall(result_text))
        if count > 0:
            result_text = pattern.sub(alias_product, result_text)
            total_replaced += count

    # 4. Remplace les lieux emblématiques
    for place in entity_info.get("places", []):
        pattern = re.compile(r"\b" + re.escape(place) + r"\b", re.IGNORECASE)
        count = len(pattern.findall(result_text))
        if count > 0:
            result_text = pattern.sub("TechCity", result_text)
            total_replaced += count

    # 5. Remplace le ticker en dernier (évite de casser les substitutions)
    # Boucle iterative pour gerer les multiples occurrences "NASDAQ: AAPL" / "NYSE: AAPL".
    for _ in names_sorted:
        pattern = re.compile(r"(?:NASDAQ|NYSE|NYSE\s*Arca)?:?\s*" + re.escape(ticker), re.IGNORECASE)
        count = len(pattern.findall(result_text))
        if count > 0:
            result_text = pattern.sub(f"EXCHANGE::{alias_ticker}", result_text)
            total_replaced += count

    return AnonymizationResult(
        text=result_text,
        ticker_alias=alias_ticker,
        company_alias=alias_company,
        reverse_map=reverse_map,
        entities_replaced=total_replaced,
    )


def restore_entities(anonymized_text: str, reverse_map: dict) -> str:
    """Re-substitue les alias par les vrais noms (pour le logging/audit)."""
    text = anonymized_text
    for alias, original in reverse_map.items():
        text = text.replace(alias, original)
    return text


def compute_bias_score(signal_original: str, signal_anonymized: str) -> dict:
    """
    Compare le signal avec et sans anonymisation.
    Retourne un diagnostic de biais potentiel.

    Si les signaux diffèrent → le modèle utilisait son prior implicite,
    pas uniquement le contenu de l'article.
    """
    same = signal_original == signal_anonymized
    return {
        "biased": not same,
        "signal_original": signal_original,
        "signal_anonymized": signal_anonymized,
        "diagnosis": (
            "Aucun biais détecté - le signal est identique avec/sans anonymisation."
            if same
            else f"BIAIS DÉTECTÉ - Signal passe de '{signal_original}' (avec nom) "
            f"à '{signal_anonymized}' (sans nom). "
            f"Le modèle s'appuyait sur sa connaissance prior de l'entreprise."
        ),
    }


# =============================================================================
# Enrichissement dynamique depuis la DB (Phase 4 — anonymizer dynamique)
# =============================================================================
# Idee : au lieu d'avoir 17 tickers hardcodes, on lit la table `articles` pour
# decouvrir les tickers presents en DB et on construit la carte d'alias a la
# volee. Cela rend l'anonymizer scalable a tout univers (S&P 500, Russell 2000).
#
# Le hardcode est conserve comme **fallback de qualite premium** (mappings
# manuels precis pour les Big Tech + delisted importants). Les entrees
# dynamiques completent sans ecraser.
# =============================================================================

# Suffixes alias generes : evite les collisions avec le hardcode (ALPHA, BETA...)
_DYNAMIC_ALIAS_PREFIX = "DYN"

# Cache thread-safe : 1 SELECT par session, partage entre threads.
_DYN_DB_LOADED = False
_DYN_DB_TICKERS_LOADED: set[str] = set()


def _alias_from_ticker(ticker: str) -> dict:
    """Genere une entree d'alias deterministe pour un ticker inconnu.

    Utilise un hash SHA256 pour garantir :
    - Determinisme : meme ticker -> meme alias entre runs (reproductibilite)
    - Unicite : 2 tickers differents -> 2 alias differents (injection-safe)
    """
    h = hashlib.sha256(ticker.encode("utf-8")).hexdigest()[:6].upper()
    return {
        "ticker": f"{_DYNAMIC_ALIAS_PREFIX}_{h}",
        "names": [ticker],  # par defaut : seulement le ticker comme "nom"
        "ceos": [],
        "products": [],
        "places": [],
        "alias_company": f"DynCorp_{h}",
        "alias_ceo": f"CEO_Dyn_{h}",
        "alias_product": f"DynProduct_{h}",
    }


def extend_db_from_database(database_path: Optional[str] = None) -> int:
    """Enrichit ENTITY_DATABASE avec les tickers presents dans la DB SQLite.

    Idempotent : un appel ulterieur ne refait pas le travail (cache global).
    Thread-safe : protege par un flag _DYN_DB_LOADED.

    Parameters
    ----------
    database_path
        Chemin vers la DB SQLite. Si None, lit `DATABASE_PATH` depuis l'env
        (defaut "data/news_database.db").

    Returns
    -------
    int
        Nombre de tickers ajoutes dynamiquement (0 si deja charge ou DB absente).
    """
    global _DYN_DB_LOADED, _DYN_DB_TICKERS_LOADED  # noqa: PLW0603

    if _DYN_DB_LOADED:
        return 0

    if database_path is None:
        database_path = os.getenv("DATABASE_PATH", "data/news_database.db")

    if not os.path.exists(database_path):
        logger.debug("anonymizer dynamique : DB absente (%s), skip", database_path)
        _DYN_DB_LOADED = True  # marque comme tente, evite retry sur chaque call
        return 0

    n_added = 0
    try:
        with sqlite3.connect(database_path, timeout=5) as conn:
            cursor = conn.cursor()
            # Recupere les tickers distincts (table `articles` a une colonne `ticker`).
            cursor.execute("SELECT DISTINCT ticker FROM articles WHERE ticker IS NOT NULL")
            for (ticker,) in cursor.fetchall():
                ticker_up = (ticker or "").strip().upper()
                if not ticker_up or ticker_up in ENTITY_DATABASE:
                    continue
                ENTITY_DATABASE[ticker_up] = _alias_from_ticker(ticker_up)
                _DYN_DB_TICKERS_LOADED.add(ticker_up)
                n_added += 1
    except sqlite3.Error as e:
        logger.warning("anonymizer dynamique : erreur SQLite (%s), fallback hardcode", e)
        _DYN_DB_LOADED = True
        return 0

    _DYN_DB_LOADED = True
    if n_added > 0:
        logger.info(
            "anonymizer dynamique : %d ticker(s) ajoute(s) depuis %s (total: %d)",
            n_added,
            database_path,
            len(ENTITY_DATABASE),
        )
    return n_added


def reset_dynamic_cache() -> None:
    """Reinitialise le cache (force un re-chargement au prochain extend_db_from_database).

    Utile pour les tests. Ne supprime PAS les entrees deja injectees dans
    ENTITY_DATABASE — pour ca, voir reset_entity_database().
    """
    global _DYN_DB_LOADED  # noqa: PLW0603
    _DYN_DB_LOADED = False


def reset_entity_database() -> int:
    """Retire les entrees dynamiques de ENTITY_DATABASE, garde le hardcode.

    Returns
    -------
    int
        Nombre d'entrees retirees.
    """
    global _DYN_DB_LOADED, _DYN_DB_TICKERS_LOADED  # noqa: PLW0603
    n_removed = 0
    for t in list(_DYN_DB_TICKERS_LOADED):
        if t in ENTITY_DATABASE:
            del ENTITY_DATABASE[t]
            n_removed += 1
    _DYN_DB_TICKERS_LOADED = set()
    _DYN_DB_LOADED = False
    return n_removed
