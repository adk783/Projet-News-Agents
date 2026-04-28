"""
source_tier.py — Classement hierarchique des sources d'information.

RATIONAL
--------
Tous les articles ne valent pas la meme chose. Un papier Reuters/Bloomberg
publie a 9h32 a une valeur informative radicalement differente d'un
blogspam qui recycle le titre a 10h15. Le pipeline actuel traite tout
pareil, ce qui :
  1. Dilue le signal (le LLM lit 12 reformulations du meme fait)
  2. Amplifie le risque de desinformation (rumeurs d'un tweet repris)
  3. Empeche toute priorisation latency-critique (on lit Reuters en dernier)

STRUCTURE DES TIERS
-------------------
Tier 1 - WIRE          : Reuters, Bloomberg, Dow Jones, AP, AFP
                         Latence minimale, precision editoriale maximale.
Tier 2 - QUALITY       : WSJ, FT, NYT, Economist, Les Echos, Le Monde
                         Analyse profonde, verification humaine.
Tier 3 - FINANCIAL     : CNBC, MarketWatch, Barron's, Seeking Alpha
                         Souvent rapide, parfois sensationnaliste.
Tier 4 - GENERAL       : Yahoo Finance, Investing.com, Benzinga, agrégateurs
                         Echo, reformule, latence elevee.
Tier 5 - SOCIAL/BLOG   : Reddit, Twitter/X, blogs persos, Stocktwits
                         Bruit dominant, parfois leading signal.
Tier 6 - UNKNOWN       : domaine non classe, poids minimal.

Un score numerique est associe (1.0, 0.85, 0.65, 0.40, 0.20, 0.10) pour
moduler : (a) la priorite de lecture dans le pipeline, (b) le poids dans
les ensembles RAG, (c) le seuil de confiance requis avant declenchement
d'un signal.

REFERENCES
----------
- Tetlock, P. C. (2007). "Giving Content to Investor Sentiment: The Role
  of Media in the Stock Market." Journal of Finance 62(3).
- Fang, L., Peress, J. (2009). "Media Coverage and the Cross-Section of
  Stock Returns." Journal of Finance 64(5).
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional
from urllib.parse import urlparse


class Tier(IntEnum):
    WIRE = 1
    QUALITY = 2
    FINANCIAL = 3
    GENERAL = 4
    SOCIAL = 5
    UNKNOWN = 6


TIER_WEIGHT = {
    Tier.WIRE: 1.00,
    Tier.QUALITY: 0.85,
    Tier.FINANCIAL: 0.65,
    Tier.GENERAL: 0.40,
    Tier.SOCIAL: 0.20,
    Tier.UNKNOWN: 0.10,
}


@dataclass(frozen=True)
class SourceInfo:
    tier: Tier
    weight: float
    canonical_name: str
    is_press_release: bool = False  # ex: businesswire, prnewswire, globenewswire


# ---------------------------------------------------------------------------
# Registre de domaines (case-insensitive, matche sur suffixe du hostname)
# ---------------------------------------------------------------------------

_TIER_WIRE = {
    "reuters.com": "Reuters",
    "bloomberg.com": "Bloomberg",
    "dowjones.com": "Dow Jones",
    "apnews.com": "Associated Press",
    "afp.com": "Agence France-Presse",
    "kyodonews.net": "Kyodo News",
}

_TIER_QUALITY = {
    "wsj.com": "Wall Street Journal",
    "ft.com": "Financial Times",
    "nytimes.com": "New York Times",
    "economist.com": "The Economist",
    "lesechos.fr": "Les Echos",
    "lemonde.fr": "Le Monde",
    "handelsblatt.com": "Handelsblatt",
    "nikkei.com": "Nikkei",
}

_TIER_FINANCIAL = {
    "cnbc.com": "CNBC",
    "marketwatch.com": "MarketWatch",
    "barrons.com": "Barron's",
    "seekingalpha.com": "Seeking Alpha",
    "forbes.com": "Forbes",
    "fool.com": "Motley Fool",
    "morningstar.com": "Morningstar",
    "investors.com": "Investor's Business Daily",
    "zonebourse.com": "Zone Bourse",
    "boursorama.com": "Boursorama",
}

_TIER_GENERAL = {
    "yahoo.com": "Yahoo Finance",
    "finance.yahoo.com": "Yahoo Finance",
    "investing.com": "Investing.com",
    "benzinga.com": "Benzinga",
    "thestreet.com": "TheStreet",
    "businessinsider.com": "Business Insider",
    "cnn.com": "CNN",
    "bbc.co.uk": "BBC",
    "bbc.com": "BBC",
    "lefigaro.fr": "Le Figaro",
    "latribune.fr": "La Tribune",
}

_TIER_SOCIAL = {
    "reddit.com": "Reddit",
    "twitter.com": "Twitter",
    "x.com": "X/Twitter",
    "stocktwits.com": "StockTwits",
    "medium.com": "Medium",
    "substack.com": "Substack",
    "discord.com": "Discord",
}

_PRESS_RELEASE_DOMAINS = {
    "businesswire.com": "Business Wire",
    "prnewswire.com": "PR Newswire",
    "globenewswire.com": "GlobeNewswire",
    "accesswire.com": "AccessWire",
    "newswire.com": "Newswire",
}


# Map unique: domaine -> (tier, canonical_name)
_DOMAIN_INDEX: dict[str, tuple[Tier, str]] = {}
for d, name in _TIER_WIRE.items():
    _DOMAIN_INDEX[d] = (Tier.WIRE, name)
for d, name in _TIER_QUALITY.items():
    _DOMAIN_INDEX[d] = (Tier.QUALITY, name)
for d, name in _TIER_FINANCIAL.items():
    _DOMAIN_INDEX[d] = (Tier.FINANCIAL, name)
for d, name in _TIER_GENERAL.items():
    _DOMAIN_INDEX[d] = (Tier.GENERAL, name)
for d, name in _TIER_SOCIAL.items():
    _DOMAIN_INDEX[d] = (Tier.SOCIAL, name)
# PR: classe en FINANCIAL mais marque le flag (sera retraite)
for d, name in _PRESS_RELEASE_DOMAINS.items():
    _DOMAIN_INDEX[d] = (Tier.FINANCIAL, name)


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def _normalize_host(url_or_host: str) -> str:
    if not url_or_host:
        return ""
    s = url_or_host.strip().lower()
    if "://" in s:
        try:
            s = urlparse(s).netloc
        except Exception:
            pass
    # supprime www. eventuel
    if s.startswith("www."):
        s = s[4:]
    # supprime port eventuel
    s = re.sub(r":\d+$", "", s)
    return s


def classify_source(url_or_host: str) -> SourceInfo:
    """
    Retourne le tier + poids d'une source donnee par URL ou hostname.
    Matching : hostname exact, sinon suffixe.
    """
    host = _normalize_host(url_or_host)
    if not host:
        return SourceInfo(
            tier=Tier.UNKNOWN,
            weight=TIER_WEIGHT[Tier.UNKNOWN],
            canonical_name="(unknown)",
        )

    # Match exact
    if host in _DOMAIN_INDEX:
        tier, name = _DOMAIN_INDEX[host]
        is_pr = host in _PRESS_RELEASE_DOMAINS
        return SourceInfo(tier=tier, weight=TIER_WEIGHT[tier], canonical_name=name, is_press_release=is_pr)

    # Match suffixe (pour sous-domaines : news.reuters.com, fr.reuters.com)
    for d, (tier, name) in _DOMAIN_INDEX.items():
        if host.endswith("." + d) or host == d:
            is_pr = d in _PRESS_RELEASE_DOMAINS
            return SourceInfo(tier=tier, weight=TIER_WEIGHT[tier], canonical_name=name, is_press_release=is_pr)

    return SourceInfo(
        tier=Tier.UNKNOWN,
        weight=TIER_WEIGHT[Tier.UNKNOWN],
        canonical_name=host,
    )


def tier_rank(url_or_host: str) -> int:
    """Raccourci : retourne juste le numero de tier (1..6)."""
    return int(classify_source(url_or_host).tier)


def source_weight(url_or_host: str) -> float:
    """Raccourci : retourne juste le poids [0, 1]."""
    return classify_source(url_or_host).weight


def is_wire_or_quality(url_or_host: str) -> bool:
    """True pour Tier 1-2 : sources dont on peut declencher un trade seul."""
    return classify_source(url_or_host).tier <= Tier.QUALITY


def is_press_release(url_or_host: str) -> bool:
    """True pour businesswire/prnewswire/globenewswire."""
    return classify_source(url_or_host).is_press_release


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "https://www.reuters.com/markets/us/fed-holds-rates-2026-04-23/",
        "https://www.bloomberg.com/news/articles/aapl-earnings",
        "https://news.wsj.com/article/foo",  # sous-domaine
        "https://ft.com/content/bar",
        "https://www.cnbc.com/video/nvidia-gtc",
        "https://finance.yahoo.com/news/msft-quarter",
        "https://www.reddit.com/r/wallstreetbets/foo",
        "https://x.com/user/status/123",
        "https://www.prnewswire.com/news/apple-partnership",
        "https://businesswire.com/news/home/xyz",
        "https://some-obscure-blog.fr/article",
        "",
    ]

    print(f"{'Host':40}  {'Tier':8}  {'W':>5}  {'PR':3}  Canonical")
    print("-" * 90)
    for s in samples:
        info = classify_source(s)
        host = _normalize_host(s) or "(empty)"
        pr = "YES" if info.is_press_release else "no"
        logger.info(f"{host[:40]:40}  {info.tier.name:8}  {info.weight:4.2f}  {pr:3}  {info.canonical_name}")
