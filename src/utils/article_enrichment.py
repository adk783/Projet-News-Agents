"""
article_enrichment.py — Orchestrateur de la couche d'enrichissement post-scraping.

ROLE
----
Quand le pipeline a extrait un article (titre + body), il doit lui attacher
toute une serie de metadonnees avant de le transmettre au reste du systeme
(filtrage IA, RAG, debat). Cette logique etait auparavant eparpillee dans
news_pipeline.py (filter keywords, dedup brut). On centralise ici :

  1. Detection de langue (rejet des non en/fr)
  2. Extraction d'entites nommees (NER cascade)
  3. Classification de type d'evenement (earnings, M&A, ...)
  4. Detection communique de presse (PR vs news)
  5. Classement de source (tier 1..6, poids)
  6. Scoring de nouveaute (vs fenetre glissante 48h)
  7. Near-duplicate final check (SimHash+Jaccard)

Le resultat est un `EnrichedArticle` serializable (dataclass -> dict -> JSON
-> SQLite). Le pipeline prend ensuite la decision d'ACCEPT/REJECT en fonction
de flags consolides.

DECISIONS D'ACCEPTATION
-----------------------
- Langue non supportee (!= en/fr)      -> REJECT  'lang_unsupported'
- Source UNKNOWN + pas de ticker match -> REJECT  'source_unknown'
- PR sur micro-cap (si flag)           -> REJECT  'pr_microcap'
- Novelty score STALE                  -> REJECT  'stale'
- Near-duplicate d'un article existant -> REJECT  'duplicate'
- Tous les autres                      -> ACCEPT

Les rejets sont traces dans `articles_filtres` avec le motif consolide.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

from src.knowledge.event_classifier import classify_event
from src.knowledge.pr_classifier import detect_press_release
from src.utils.lang_detect import detect_language, is_supported
from src.utils.ner_gliner import entities_by_label, extract_entities, unique_entity_texts
from src.utils.novelty_scorer import (
    NOVELTY_THRESHOLD_ALPHA,
    NOVELTY_THRESHOLD_STALE,
    HistoricalArticle,
    build_history_from_rows,
    compute_novelty_score,
)

# Imports absolus (cf. ADR-008, mode editable `pip install -e .`).
from src.utils.simhash_dedup import is_near_duplicate_text
from src.utils.source_tier import Tier, classify_source

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class EnrichedArticle:
    url: str
    ticker: str
    title: str
    body: str
    published_at: Optional[datetime]

    # Enrichissements
    lang: str = "unknown"
    lang_supported: bool = False

    source_tier: int = int(Tier.UNKNOWN)
    source_name: str = ""
    source_weight: float = 0.1

    is_press_release: bool = False
    pr_confidence: float = 0.0

    event_type: str = "other"
    event_confidence: float = 0.0
    event_secondary: Tuple[str, ...] = ()

    entities: dict = field(default_factory=dict)  # label -> [texts]

    novelty_score: float = 1.0
    novelty_decision: str = "alpha"
    closest_article_id: Optional[str] = None

    is_duplicate: bool = False
    duplicate_method: str = ""
    duplicate_similarity: float = 0.0

    # Decision finale
    accept: bool = True
    reject_reason: str = ""

    def to_row(self) -> dict:
        """Pour ecriture SQLite. Serialise les entites en JSON."""
        d = asdict(self)
        d["entities_json"] = json.dumps(self.entities, ensure_ascii=False)
        d.pop("entities", None)
        # Datetime -> ISO
        if isinstance(self.published_at, datetime):
            d["published_at"] = self.published_at.isoformat()
        d["event_secondary"] = ",".join(self.event_secondary)
        return d


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------


def enrich_article(
    url: str,
    ticker: str,
    title: str,
    body: str,
    published_at: Optional[datetime],
    *,
    company_name: Optional[str] = None,
    history: Optional[Sequence[HistoricalArticle]] = None,
    check_duplicates_against: Optional[Sequence[dict]] = None,
    allow_non_wire_trade: bool = True,
    reject_stale: bool = True,
    supported_langs: frozenset = frozenset({"en", "fr"}),
) -> EnrichedArticle:
    """
    Enrichit un article et rend la decision d'acceptation.

    Args:
        url, ticker, title, body, published_at : donnees brutes du scraping.
        company_name : utile pour PR classifier (densite de mentions).
        history : HistoricalArticle list pour scoring de nouveaute (48h).
        check_duplicates_against : liste de dicts {title, content} d'articles
            deja ingerés, pour near-duplicate final.
        allow_non_wire_trade : si False, REJECT toute source hors Tier 1-2.
        reject_stale : si True, REJECT les articles novelty_decision=='stale'.
        supported_langs : whitelist des codes ISO.

    Returns:
        EnrichedArticle avec .accept et .reject_reason remplis.
    """
    ea = EnrichedArticle(
        url=url,
        ticker=ticker,
        title=title or "",
        body=body or "",
        published_at=published_at,
    )

    # ---- 1) Langue -------------------------------------------------------
    combined = f"{title} {body[:500]}"
    ea.lang = detect_language(combined)
    ea.lang_supported = is_supported(ea.lang, supported=supported_langs)
    if not ea.lang_supported:
        ea.accept = False
        ea.reject_reason = f"lang_unsupported({ea.lang})"
        return ea

    # ---- 2) Source tier --------------------------------------------------
    src_info = classify_source(url)
    ea.source_tier = int(src_info.tier)
    ea.source_name = src_info.canonical_name
    ea.source_weight = src_info.weight
    if not allow_non_wire_trade and src_info.tier > Tier.QUALITY:
        ea.accept = False
        ea.reject_reason = f"source_tier_too_low({src_info.tier.name})"
        return ea

    # ---- 3) Press release ------------------------------------------------
    pr = detect_press_release(body, url=url, company_name=company_name)
    ea.is_press_release = pr.is_press_release
    ea.pr_confidence = pr.confidence

    # ---- 4) Event type ---------------------------------------------------
    ev = classify_event(title, body)
    ea.event_type = ev.primary
    ea.event_confidence = ev.confidence
    ea.event_secondary = ev.secondary

    # ---- 5) Entities -----------------------------------------------------
    try:
        ents = extract_entities(f"{title}. {body[:2000]}")
        ea.entities = {lbl: unique_entity_texts(ents, lbl) for lbl in {e.label for e in ents}}
    except Exception:
        ea.entities = {}

    # ---- 6) Novelty ------------------------------------------------------
    if history and published_at is not None:
        score = compute_novelty_score(title, body, published_at, history)
        ea.novelty_score = score.score
        ea.novelty_decision = score.decision
        ea.closest_article_id = score.closest_article_id
        if reject_stale and score.decision == "stale":
            ea.accept = False
            ea.reject_reason = "stale"
            return ea

    # ---- 7) Near-duplicate final -----------------------------------------
    if check_duplicates_against:
        for cand in check_duplicates_against:
            c_title = cand.get("title", "")
            c_body = cand.get("content", "") or cand.get("body", "")
            # Check titre
            is_dup_t, sim_t, meth_t = is_near_duplicate_text(title, c_title)
            if is_dup_t:
                ea.is_duplicate = True
                ea.duplicate_method = f"title:{meth_t}"
                ea.duplicate_similarity = sim_t
                break
            # Check body (si body non trivial)
            if body and c_body and len(body) > 200 and len(c_body) > 200:
                is_dup_b, sim_b, meth_b = is_near_duplicate_text(body, c_body)
                if is_dup_b:
                    ea.is_duplicate = True
                    ea.duplicate_method = f"body:{meth_b}"
                    ea.duplicate_similarity = sim_b
                    break

    if ea.is_duplicate:
        ea.accept = False
        ea.reject_reason = f"duplicate({ea.duplicate_method})"
        return ea

    ea.accept = True
    ea.reject_reason = ""
    return ea


# ---------------------------------------------------------------------------
# SQLite migration helper
# ---------------------------------------------------------------------------

ENRICHMENT_COLUMNS = [
    ("lang", "TEXT"),
    ("source_tier", "INTEGER"),
    ("source_name", "TEXT"),
    ("source_weight", "REAL"),
    ("is_press_release", "INTEGER DEFAULT 0"),
    ("pr_confidence", "REAL"),
    ("event_type", "TEXT"),
    ("event_confidence", "REAL"),
    ("entities_json", "TEXT"),
    ("novelty_score", "REAL"),
    ("novelty_decision", "TEXT"),
]


def ensure_enrichment_schema(cursor) -> List[str]:
    """
    Ajoute les colonnes d'enrichissement a `articles` si elles manquent.
    Retourne la liste des colonnes effectivement ajoutees.
    """
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(articles)").fetchall()}
    added: List[str] = []
    for col, ddl in ENRICHMENT_COLUMNS:
        if col not in existing:
            cursor.execute(f"ALTER TABLE articles ADD COLUMN {col} {ddl}")
            added.append(col)
    return added


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    now = datetime.now(tz=timezone.utc)

    history = [
        HistoricalArticle(
            article_id="old1",
            published_at=now,
            title="Apple CEO Tim Cook announces resignation",
            body="Tim Cook announced his resignation from Apple today.",
        ),
    ]

    # Cas 1: article normal en anglais, source Reuters, earnings event
    ea = enrich_article(
        url="https://www.reuters.com/markets/aapl-q3-earnings",
        ticker="AAPL",
        title="Apple beats Q3 earnings estimates by 8%",
        body=(
            "Apple Inc. reported quarterly revenue of $89.5 billion, "
            "above consensus estimates. CEO Tim Cook said the quarter "
            "reflected strength in services and emerging markets. "
            "The company raised its full-year guidance."
        ),
        published_at=now,
        company_name="Apple",
        history=history,
    )
    logger.info("[CAS 1] Reuters earnings:")
    logger.info(f"  accept={ea.accept} reason={ea.reject_reason!r}")
    logger.info(f"  lang={ea.lang} tier={ea.source_tier}({ea.source_name})")
    logger.info(f"  event={ea.event_type}({ea.event_confidence:.2f}) PR={ea.is_press_release}")
    logger.info(f"  novelty={ea.novelty_score:.2f}/{ea.novelty_decision}")
    logger.info(f"  entities={ea.entities}")
    print()

    # Cas 2: chinois -> REJECT
    ea2 = enrich_article(
        url="https://www.sohu.com/news/aapl",
        ticker="AAPL",
        title="苹果公司第三季度业绩超预期",
        body="苹果公司今天报告了季度营收，超过了华尔街的预期。",
        published_at=now,
    )
    logger.info("[CAS 2] article chinois:")
    logger.info(f"  accept={ea2.accept} reason={ea2.reject_reason!r}  lang={ea2.lang}")
    print()

    # Cas 3: press release prnewswire
    pr_body = (
        "CUPERTINO, Calif., April 23, 2026 /PRNewswire/ -- Apple Inc. (NASDAQ: AAPL), "
        "a leading global provider, today announced a partnership with Acme Corp.\n"
        "About Apple\n"
        "Apple revolutionized personal technology.\n"
        "Forward-looking statements\n"
        "This press release contains forward-looking statements. For more information, "
        "contact: Investor Relations."
    )
    ea3 = enrich_article(
        url="https://www.prnewswire.com/news/apple-acme",
        ticker="AAPL",
        title="Apple announces strategic partnership with Acme Corp",
        body=pr_body,
        published_at=now,
        company_name="Apple",
    )
    logger.info("[CAS 3] Press release:")
    logger.info(f"  accept={ea3.accept} is_pr={ea3.is_press_release}(conf={ea3.pr_confidence:.2f})")
    logger.info(f"  tier={ea3.source_tier}({ea3.source_name}) event={ea3.event_type}")
    print()

    # Cas 4: stale (duplicate du historique)
    ea4 = enrich_article(
        url="https://www.cnbc.com/2026/tim-cook-resigns",
        ticker="AAPL",
        title="Apple CEO Tim Cook announces resignation",
        body="Tim Cook announced his resignation from Apple today.",
        published_at=now,
        company_name="Apple",
        history=history,
    )
    logger.info("[CAS 4] stale rehash:")
    logger.info(f"  accept={ea4.accept} reason={ea4.reject_reason!r}")
    logger.info(f"  novelty={ea4.novelty_score:.2f}/{ea4.novelty_decision}")
