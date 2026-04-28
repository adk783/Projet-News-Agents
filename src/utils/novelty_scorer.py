"""
novelty_scorer.py — Scoring de nouveaute des articles vs fenetre glissante.

MOTIVATION
----------
Un article qui rapporte "Apple CEO Tim Cook resigns" a une valeur informative
enorme a t=0 et quasi nulle a t+30min (tout le marche l'a lu). Le pipeline
actuel n'a aucun mecanisme pour distinguer :
  - Un premier rapport (alpha pur)
  - La 12e reformulation de la meme depeche par des blogs secondaires
  - Un vieux sujet recycle ("Apple's declining iPhone sales" publie pour la
    4e fois par un meme outlet)

FORMULATION
-----------
novelty = 1 - max(
    similarity(new_article, a) * time_decay(t_now - a.published_at)
    for a in recent_history
)

Avec :
  - similarity(a, b) in [0, 1] (Jaccard sur titre+body, ou cosine embedding)
  - time_decay(dt) = exp(-dt / tau)  ou tau = 6h par defaut
  - recent_history = fenetre glissante des 48 dernieres heures

Un score >= NOVELTY_THRESHOLD_ALPHA (0.80 par defaut) signifie "nouvelle
information non couverte". Un score < NOVELTY_THRESHOLD_STALE (0.30)
declenche un REJECT "rehash".

BACKEND
-------
On reutilise `simhash_dedup.jaccard_similarity` + `is_near_duplicate_text`
pour coller au meme modele de similarite que le dedup. Quand un vector store
sera branche (batch 3), on aura une surcharge `compute_novelty_embeddings`
qui utilisera les cosines embedding.

REFERENCES
----------
- Chen, Y. et al. (2023). "Measuring Information Novelty in Financial News."
  SSRN 4512934.
- Tetlock, P. C. (2011). "All the News That's Fit to Reprint: Do Investors
  React to Stale Information?" Review of Financial Studies 24(5).
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Sequence

try:
    from .simhash_dedup import SIMHASH_BITS, hamming_distance, jaccard_similarity, simhash
except ImportError:  # exec direct du fichier
    from simhash_dedup import SIMHASH_BITS, hamming_distance, jaccard_similarity, simhash


# ---------------------------------------------------------------------------
# Parametres
# ---------------------------------------------------------------------------

NOVELTY_TAU_SECONDS = 6 * 3600  # demi-vie de 6h
NOVELTY_WINDOW_SECONDS = 48 * 3600  # 48h de fenetre
NOVELTY_THRESHOLD_ALPHA = 0.80  # au-dessus = vraie info neuve
NOVELTY_THRESHOLD_STALE = 0.30  # en dessous = rehash
MAX_HISTORY_SAMPLES = 500  # cap anti-explosion memoire


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoricalArticle:
    """Article deja vu dans la fenetre glissante."""

    article_id: str
    published_at: datetime
    title: str
    body: str = ""
    # cache hash si deja calcule
    title_simhash: Optional[int] = None


@dataclass
class NoveltyScore:
    """Resultat d'un scoring."""

    score: float  # in [0, 1]
    closest_article_id: Optional[str]
    closest_similarity: float  # avant time-decay
    closest_time_delta_sec: Optional[float]
    decision: str  # "alpha" | "moderate" | "stale"
    samples_compared: int

    def is_alpha(self) -> bool:
        return self.decision == "alpha"

    def is_stale(self) -> bool:
        return self.decision == "stale"

    def summary(self) -> str:
        return (
            f"novelty={self.score:.2f} ({self.decision}) "
            f"closest_sim={self.closest_similarity:.2f} "
            f"samples={self.samples_compared}"
        )


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def _time_decay(delta_sec: float, tau_sec: float = NOVELTY_TAU_SECONDS) -> float:
    """Exp decay: un article vieux pese moins dans le max-similarite."""
    if delta_sec <= 0:
        return 1.0
    return math.exp(-delta_sec / tau_sec)


def _similarity(new_title: str, new_body: str, hist: HistoricalArticle, new_simhash: Optional[int] = None) -> float:
    """
    Similarite [0,1] titre-titre + body-body.
    Utilise Jaccard sur les deux, qui donne un signal propre meme sur
    textes courts. (SimHash garde un floor ~0.5 sur textes non-lies a
    cause du Hamming random baseline ~32/64 -> 1-0.5=0.5, inutile pour
    ce scoring fin.)

    On prend max(title, body) pour ne rater aucun axe, et on ajoute un
    petit bonus si les deux convergent (co-occurrence renforce le signal).
    """
    sim_title = jaccard_similarity(new_title, hist.title)
    sim_body = 0.0
    if new_body and hist.body:
        sim_body = jaccard_similarity(new_body, hist.body)

    base = max(sim_title, sim_body)
    # Bonus si titre ET body confirment
    if sim_title > 0.3 and sim_body > 0.3:
        base = min(1.0, base + 0.15)
    return base


def compute_novelty_score(
    new_title: str,
    new_body: str,
    new_published_at: datetime,
    history: Sequence[HistoricalArticle],
    window_seconds: float = NOVELTY_WINDOW_SECONDS,
    tau_seconds: float = NOVELTY_TAU_SECONDS,
    threshold_alpha: float = NOVELTY_THRESHOLD_ALPHA,
    threshold_stale: float = NOVELTY_THRESHOLD_STALE,
) -> NoveltyScore:
    """
    Calcule le score de nouveaute d'un article vs sa fenetre glissante.

    Returns:
        NoveltyScore avec decision "alpha" | "moderate" | "stale".
    """
    if new_published_at.tzinfo is None:
        new_published_at = new_published_at.replace(tzinfo=timezone.utc)

    # Filtrer l'historique a la fenetre
    fresh: List[HistoricalArticle] = []
    for h in history:
        pub = h.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        dt = (new_published_at - pub).total_seconds()
        if 0 <= dt <= window_seconds:
            fresh.append(h)

    if not fresh:
        return NoveltyScore(
            score=1.0,
            closest_article_id=None,
            closest_similarity=0.0,
            closest_time_delta_sec=None,
            decision="alpha",
            samples_compared=0,
        )

    # Cap pour limiter le cout
    if len(fresh) > MAX_HISTORY_SAMPLES:
        # Priorite aux plus recents (plus de chance d'etre semblables de toute facon)
        fresh.sort(key=lambda h: h.published_at, reverse=True)
        fresh = fresh[:MAX_HISTORY_SAMPLES]

    new_sh = simhash(new_body) if new_body else None

    best_weighted = 0.0
    best_raw = 0.0
    best_id: Optional[str] = None
    best_dt: Optional[float] = None

    for h in fresh:
        pub = h.published_at
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        dt = (new_published_at - pub).total_seconds()
        sim_raw = _similarity(new_title, new_body, h, new_simhash=new_sh)
        weighted = sim_raw * _time_decay(dt, tau_seconds)
        if weighted > best_weighted:
            best_weighted = weighted
            best_raw = sim_raw
            best_id = h.article_id
            best_dt = dt

    score = max(0.0, 1.0 - best_weighted)

    if score >= threshold_alpha:
        decision = "alpha"
    elif score < threshold_stale:
        decision = "stale"
    else:
        decision = "moderate"

    return NoveltyScore(
        score=score,
        closest_article_id=best_id,
        closest_similarity=best_raw,
        closest_time_delta_sec=best_dt,
        decision=decision,
        samples_compared=len(fresh),
    )


# ---------------------------------------------------------------------------
# Utilitaires pour construire l'historique depuis la DB
# ---------------------------------------------------------------------------


def build_history_from_rows(rows: Iterable[dict]) -> List[HistoricalArticle]:
    """
    Construit une liste HistoricalArticle depuis des dicts SQLite-like.
    Attendu : keys 'id'|'article_id', 'published_at' (iso str ou datetime),
    'title', 'body'|'snippet'.
    """
    out: List[HistoricalArticle] = []
    for r in rows:
        aid = str(r.get("article_id") or r.get("id") or "")
        if not aid:
            continue
        pub_raw = r.get("published_at")
        if isinstance(pub_raw, datetime):
            pub = pub_raw
        elif isinstance(pub_raw, str):
            try:
                pub = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
        else:
            continue
        title = (r.get("title") or "").strip()
        body = (r.get("body") or r.get("snippet") or "").strip()
        out.append(
            HistoricalArticle(
                article_id=aid,
                published_at=pub,
                title=title,
                body=body,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    now = datetime(2026, 4, 23, 14, 0, 0, tzinfo=timezone.utc)

    history = [
        # Meme sujet, publie 1h avant -> doit reduire la nouveaute
        HistoricalArticle(
            article_id="a1",
            published_at=datetime(2026, 4, 23, 13, 0, 0, tzinfo=timezone.utc),
            title="Apple CEO Tim Cook announces resignation",
            body="Tim Cook, the longtime CEO of Apple Inc., announced today his "
            "resignation from the position he has held since 2011.",
        ),
        # Sujet different, meme fenetre
        HistoricalArticle(
            article_id="a2",
            published_at=datetime(2026, 4, 23, 12, 30, 0, tzinfo=timezone.utc),
            title="Microsoft beats Q3 earnings estimates",
            body="Microsoft reported quarterly revenue of $62 billion, above consensus expectations.",
        ),
        # Sujet proche mais publie il y a 30h -> decay tue la similarite
        HistoricalArticle(
            article_id="a3",
            published_at=datetime(2026, 4, 22, 8, 0, 0, tzinfo=timezone.utc),
            title="Rumors about Apple CEO succession plan",
            body="Unconfirmed reports suggest Apple's board has begun discussions "
            "about a succession plan for CEO Tim Cook.",
        ),
    ]

    cases = [
        # Rehash quasi-literal -> STALE
        (
            "Apple CEO Tim Cook announces his resignation",
            "Tim Cook, the longtime CEO of Apple Inc., announced today his resignation "
            "from the position he has held since 2011.",
            "devrait etre STALE (rehash quasi-literal de a1)",
        ),
        # Sujet inedit -> ALPHA
        (
            "Nvidia unveils new GPU architecture at GTC",
            "Nvidia today unveiled Blackwell Ultra, its next-generation GPU architecture targeting AI data centers.",
            "devrait etre ALPHA",
        ),
        # Follow-up meme sujet avec fait nouveau -> MODERATE
        (
            "Apple CEO Tim Cook steps down effective immediately",
            "After 15 years at the helm, Tim Cook is stepping down as CEO of Apple Inc. "
            "effective immediately, the company announced in a press release.",
            "devrait etre MODERATE (follow-up meme sujet, fait nouveau)",
        ),
    ]

    for title, body, expectation in cases:
        score = compute_novelty_score(title, body, now, history)
        logger.info(f"\n[{score.decision:8}] {expectation}")
        logger.info(f"  title  : {title[:70]}")
        logger.info(f"  score  : {score.score:.3f}")
        print(
            f"  closest: {score.closest_article_id} (sim={score.closest_similarity:.2f}, "
            f"dt={score.closest_time_delta_sec}s)"
        )
        logger.info(f"  summary: {score.summary()}")
