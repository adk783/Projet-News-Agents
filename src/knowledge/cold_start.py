"""
cold_start.py — Resolveur du cold-start pour le RAG ticker-based.

PROBLEME
--------
Quand un ticker apparait pour la premiere fois (IPO, rotation portefeuille,
nouvel univers d'investissement), le RAG est vide :
  - 0 articles historiques
  - 0 transcripts d'earnings calls
  - 0 filings SEC indexes
  - 0 debat passe en memoire

Resultat : l'agent ne dispose d'aucun contexte, le score de confiance
s'effondre, le position-sizer est conservateur, ou pire : le LLM hallucine
des "souvenirs" inexistants.

C'est le "cold start problem", largement etudie dans les recommender
systems (Schein et al. 2002), et particulierement aigu en finance car
chaque ticker a une histoire et un secteur uniques.

STRATEGIE EN 3 ETAGES
---------------------
1. **Detection** : `is_cold(ticker)` — verifie si le RAG a moins de
   `min_documents` documents. Detection bon-marche, idempotente.

2. **Backfill multi-source** : `backfill(ticker)` orchestre :
     a. SEC EDGAR : derniers 10-K, 10-Q, 8-K (factuel, gratuit, gold).
     b. Earnings calls : derniers 4 transcripts (si dispo).
     c. Fundamentals : profile synthetique (sector, market cap, peers).
     d. Historical news : snapshot via WebSearch (best-effort).
   Indexe tout dans le RAG avec dates correctes (decay temporel
   s'applique normalement).

3. **Bootstrap par similarite** : `seed_from_peers(ticker)` — si meme
   apres backfill on est sous le seuil, on emprunte des "patterns
   sectoriels" depuis 1-3 tickers similaires deja chauds. Inspire de
   collaborative filtering : "ce qui marche pour XOM/CVX a tendance a
   marcher pour OXY".

PROFIL SYNTHETIQUE
------------------
A partir des fundamentals (yfinance), on construit un document de
contexte synthetique : sector, market cap, P/E, marge, croissance,
dividende, recommandations analystes. Indexe comme `doc_type=cold_start_profile`
avec une date proche du now mais marquee dans les metadonnees comme
"synthetic" pour traceabilite. Le decay temporel s'applique mais avec
un poids de fiabilite reduit en aval.

REFERENCES
----------
- Schein, A. I. et al. (2002). "Methods and Metrics for Cold-Start
  Recommendations." SIGIR 2002.
- Lika, B., Kolomvatsos, K. & Hadjiefthymiades, S. (2014). "Facing the
  cold start problem in recommender systems." Expert Systems with App.
- Park, S.-T. & Chu, W. (2009). "Pairwise preference regression for
  cold-start recommendation." RecSys 2009.
- Fama, E. F. & French, K. R. (1997). "Industry costs of equity."
  Journal of Financial Economics. → Justification du transfert de
  patterns intra-secteur.
- Hoberg, G. & Phillips, G. (2016). "Text-Based Network Industries and
  Endogenous Product Differentiation." JPE. → Similarite text-based
  intra-secteur pour identifier les peers.

CONFIGURATION
-------------
Tous les seuils sont configurables via env vars (preferable en CI/test) :
  COLD_START_MIN_DOCS         (default 5)
  COLD_START_BACKFILL_LIMIT   (default 30)
  COLD_START_PEER_FALLBACK    (default 1 — on autorise le fallback peer)

USAGE
-----
    from cold_start import ColdStartManager
    csm = ColdStartManager()
    if csm.is_cold("OXY"):
        rep = csm.bootstrap("OXY", company_name="Occidental Petroleum",
                            sector="Energy", peers=["XOM", "CVX"])
        print(rep.summary())
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes / config
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


# Seuil sous lequel on considere le ticker "cold"
MIN_DOCS_DEFAULT = _env_int("COLD_START_MIN_DOCS", 5)
# Plafond de docs a backfill par run (eviter de saturer ChromaDB)
BACKFILL_LIMIT_DEFAULT = _env_int("COLD_START_BACKFILL_LIMIT", 30)
# Activation fallback peer si toujours cold apres backfill
PEER_FALLBACK_DEFAULT = _env_bool("COLD_START_PEER_FALLBACK", True)

# Combien de jours en arriere pour le backfill par defaut
BACKFILL_LOOKBACK_DAYS_DEFAULT = _env_int("COLD_START_LOOKBACK_DAYS", 365)

# Tags doc_type specifiques cold-start (preserves dans le RAG)
DOC_TYPE_COLD_PROFILE = "cold_start_profile"
DOC_TYPE_COLD_PEER = "cold_start_peer_seed"
DOC_TYPE_COLD_NEWS = "cold_start_news"
DOC_TYPE_COLD_SEC = "cold_start_sec"
DOC_TYPE_COLD_CALL = "cold_start_call"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class BackfillSourceResult:
    """Resultat d'un backfill par une source individuelle."""

    name: str
    success: bool = False
    documents_added: int = 0
    error: str = ""
    duration_sec: float = 0.0


@dataclass
class ColdStartReport:
    """Rapport global d'un cycle cold-start."""

    ticker: str
    was_cold: bool
    is_cold_after: bool
    docs_before: int
    docs_after: int
    sources: List[BackfillSourceResult] = field(default_factory=list)
    peers_used: List[str] = field(default_factory=list)
    profile_indexed: bool = False
    duration_total_sec: float = 0.0

    def summary(self) -> str:
        return (
            f"ticker={self.ticker} cold={self.was_cold}->{self.is_cold_after} "
            f"docs={self.docs_before}->{self.docs_after} "
            f"sources_ok={sum(1 for s in self.sources if s.success)}/{len(self.sources)} "
            f"peers={len(self.peers_used)} duration={self.duration_total_sec:.1f}s"
        )

    def to_prompt_block(self) -> str:
        lines = [
            "<cold_start_report>",
            f"  ticker={self.ticker}",
            f"  was_cold={self.was_cold}  is_cold_after={self.is_cold_after}",
            f"  docs_before={self.docs_before}  docs_after={self.docs_after}",
            f"  profile_indexed={self.profile_indexed}",
        ]
        for s in self.sources:
            lines.append(f"  source={s.name} ok={s.success} added={s.documents_added} err={s.error[:80]}")
        if self.peers_used:
            lines.append(f"  peers_used={self.peers_used}")
        lines.append(f"  duration={self.duration_total_sec:.1f}s")
        lines.append("</cold_start_report>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_call(fn: Callable[..., Any], *args, **kwargs) -> Tuple[bool, Any, str]:
    """Execute fn et capture les erreurs en (ok, result, error_msg)."""
    try:
        return True, fn(*args, **kwargs), ""
    except Exception as e:
        return False, None, str(e)[:200]


def _short_str(s: Any, n: int = 200) -> str:
    s2 = str(s) if s is not None else ""
    return s2[:n]


# ---------------------------------------------------------------------------
# ColdStartManager
# ---------------------------------------------------------------------------


class ColdStartManager:
    """
    Orchestrateur du cycle cold-start.

    Tous les fetchers / store sont injectables (Dependency Injection) pour
    permettre des tests sans reseau et la composition.

    Architecture :
      ColdStartManager
        |-- rag_store          : LocusRAGStore (ou compatible)
        |-- edgar_client       : SecEdgarClient (optionnel)
        |-- earnings_fetcher   : Callable (ticker) -> List[(date_iso, text)]
        |-- news_fetcher       : Callable (ticker, days) -> List[Article]
        |-- fundamentals_fn    : Callable (ticker) -> FundamentalsData
        |-- peer_resolver      : Callable (ticker) -> List[str]
    """

    def __init__(
        self,
        rag_store: Optional[Any] = None,
        edgar_client: Optional[Any] = None,
        earnings_fetcher: Optional[Callable[[str], List[Tuple[str, str]]]] = None,
        news_fetcher: Optional[Callable[[str, int], List[Dict[str, Any]]]] = None,
        fundamentals_fn: Optional[Callable[[str], Any]] = None,
        peer_resolver: Optional[Callable[[str], List[str]]] = None,
        min_docs: int = MIN_DOCS_DEFAULT,
        backfill_limit: int = BACKFILL_LIMIT_DEFAULT,
        peer_fallback: bool = PEER_FALLBACK_DEFAULT,
        lookback_days: int = BACKFILL_LOOKBACK_DAYS_DEFAULT,
    ):
        self.rag_store = rag_store
        self.edgar_client = edgar_client
        self.earnings_fetcher = earnings_fetcher
        self.news_fetcher = news_fetcher
        self.fundamentals_fn = fundamentals_fn
        self.peer_resolver = peer_resolver

        self.min_docs = max(1, int(min_docs))
        self.backfill_limit = max(1, int(backfill_limit))
        self.peer_fallback = bool(peer_fallback)
        self.lookback_days = max(30, int(lookback_days))

        # Cache interne pour eviter de re-detecter sur la meme run
        self._cold_cache: Dict[str, Tuple[bool, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)

    # ----- Detection -----

    def count_documents(self, ticker: str) -> int:
        """Compte les documents indexes pour un ticker."""
        if self.rag_store is None:
            return 0
        try:
            return int(self.rag_store.collection_size(ticker))
        except Exception as e:
            logger.debug("[ColdStart] collection_size err %s : %s", ticker, e)
            return 0

    def is_cold(self, ticker: str, *, force_refresh: bool = False) -> bool:
        """Retourne True si le ticker a moins de min_docs documents indexes."""
        ticker = ticker.upper()
        now = datetime.now(timezone.utc)

        if not force_refresh and ticker in self._cold_cache:
            cold, ts = self._cold_cache[ticker]
            if now - ts < self._cache_ttl:
                return cold

        n = self.count_documents(ticker)
        cold = n < self.min_docs
        self._cold_cache[ticker] = (cold, now)
        logger.debug("[ColdStart] %s : %d docs (cold=%s)", ticker, n, cold)
        return cold

    def invalidate_cache(self, ticker: Optional[str] = None) -> None:
        """Invalide le cache de detection (apres un backfill par exemple)."""
        if ticker is None:
            self._cold_cache.clear()
        else:
            self._cold_cache.pop(ticker.upper(), None)

    # ----- Backfill orchestration -----

    def bootstrap(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        sector: Optional[str] = None,
        peers: Optional[Sequence[str]] = None,
        force: bool = False,
    ) -> ColdStartReport:
        """
        Cycle complet : detection -> profile synthetique -> backfill multi-source
        -> peer seeding si toujours cold -> reverification.

        Args:
            ticker        : ticker a bootstrap (ex: "OXY")
            company_name  : nom complet (utile pour searches news)
            sector        : secteur (utile pour peer fallback)
            peers         : peers explicites (override peer_resolver)
            force         : skip la detection initiale et force le bootstrap

        Returns:
            ColdStartReport detaille.
        """
        ticker = ticker.upper()
        t_start = time.time()

        rep = ColdStartReport(
            ticker=ticker,
            was_cold=False,
            is_cold_after=False,
            docs_before=self.count_documents(ticker),
            docs_after=0,
        )
        rep.was_cold = rep.docs_before < self.min_docs

        if not rep.was_cold and not force:
            rep.is_cold_after = False
            rep.docs_after = rep.docs_before
            rep.duration_total_sec = time.time() - t_start
            return rep

        logger.info(
            "[ColdStart] Bootstrap %s : %d docs < %d (cold)",
            ticker,
            rep.docs_before,
            self.min_docs,
        )

        # 1. Profil synthetique (fundamentals)
        if self.fundamentals_fn is not None:
            ok = self._index_synthetic_profile(ticker, company_name, sector)
            rep.profile_indexed = ok

        # 2. Backfill SEC EDGAR
        if self.edgar_client is not None:
            rep.sources.append(self._backfill_sec(ticker))

        # 3. Backfill earnings calls
        if self.earnings_fetcher is not None:
            rep.sources.append(self._backfill_earnings(ticker))

        # 4. Backfill historical news
        if self.news_fetcher is not None:
            rep.sources.append(self._backfill_news(ticker))

        # 5. Re-evaluation
        self.invalidate_cache(ticker)
        docs_after_main = self.count_documents(ticker)

        # 6. Peer seeding si toujours cold
        if docs_after_main < self.min_docs and self.peer_fallback:
            peer_list = list(peers) if peers else (self.peer_resolver(ticker) if self.peer_resolver else [])
            peer_list = [p.upper() for p in peer_list if p and p.upper() != ticker][:5]
            if peer_list:
                seeded = self._seed_from_peers(ticker, peer_list)
                rep.peers_used = peer_list[: seeded[0]] if seeded else []
                rep.sources.append(
                    BackfillSourceResult(
                        name="peer_seed",
                        success=seeded[1] > 0,
                        documents_added=seeded[1],
                        duration_sec=seeded[2],
                    )
                )

        # 7. Final check
        self.invalidate_cache(ticker)
        rep.docs_after = self.count_documents(ticker)
        rep.is_cold_after = rep.docs_after < self.min_docs
        rep.duration_total_sec = time.time() - t_start

        logger.info("[ColdStart] %s", rep.summary())
        return rep

    # ----- Profile synthetique -----

    def _index_synthetic_profile(
        self,
        ticker: str,
        company_name: Optional[str],
        sector: Optional[str],
    ) -> bool:
        """
        Construit un document de profile synthetique a partir des fundamentals
        et l'indexe dans le RAG. Date_iso = now (frais), mais
        metadata.synthetic=True pour pondre la fiabilite en aval.
        """
        try:
            fund = self.fundamentals_fn(ticker)  # type: ignore[misc]
        except Exception as e:
            logger.warning("[ColdStart] fundamentals %s echec : %s", ticker, e)
            return False

        if fund is None:
            return False

        # Fabriquer un texte structure facile a embedder
        text = self._render_profile_text(fund, ticker, company_name, sector)
        if not text:
            return False

        # Tenter l'indexation dans le RAG
        try:
            from src.knowledge.rag_store import RAGDocument  # type: ignore
        except ImportError:
            try:
                from .rag_store import RAGDocument  # type: ignore
            except ImportError:
                logger.warning("[ColdStart] RAGDocument inaccessible — skip profile")
                return False

        doc = RAGDocument(
            doc_id=f"cold-profile-{ticker}-{int(time.time())}",
            ticker=ticker,
            text=text,
            doc_type=DOC_TYPE_COLD_PROFILE,
            date_iso=_now_iso(),
            metadata={
                "synthetic": "true",
                "source": "cold_start_profile",
                "company_name": _short_str(company_name or getattr(fund, "company_name", ""), 100),
                "sector": _short_str(sector or getattr(fund, "sector", ""), 60),
                "industry": _short_str(getattr(fund, "industry", ""), 60),
            },
        )
        ok = bool(self.rag_store.add_document(doc))
        if ok:
            logger.info("[ColdStart] Profil synthetique indexe pour %s", ticker)
        return ok

    @staticmethod
    def _render_profile_text(
        fund: Any,
        ticker: str,
        company_name: Optional[str],
        sector: Optional[str],
    ) -> str:
        """Genere un texte structure depuis FundamentalsData (best-effort)."""
        name = company_name or getattr(fund, "company_name", "") or ticker
        sec = sector or getattr(fund, "sector", "") or "Unknown"
        ind = getattr(fund, "industry", "") or "Unknown"

        lines = [
            f"Synthetic profile for {name} ({ticker}).",
            f"Sector: {sec}. Industry: {ind}.",
        ]

        # Champs optionnels — affiches uniquement si non-None
        def line_if(label: str, val: Any, fmt: str = "{}") -> Optional[str]:
            if val is None:
                return None
            try:
                return f"{label}: " + fmt.format(val)
            except Exception:
                return f"{label}: {val}"

        spec_lines = [
            line_if("Market cap (USD)", getattr(fund, "market_cap", None), "{:,.0f}"),
            line_if("Enterprise value (USD)", getattr(fund, "enterprise_value", None), "{:,.0f}"),
            line_if("P/E (trailing)", getattr(fund, "pe_trailing", None), "{:.2f}"),
            line_if("P/E (forward)", getattr(fund, "pe_forward", None), "{:.2f}"),
            line_if("PEG ratio", getattr(fund, "peg_ratio", None), "{:.2f}"),
            line_if("Price/Book", getattr(fund, "price_to_book", None), "{:.2f}"),
            line_if("EV/EBITDA", getattr(fund, "ev_to_ebitda", None), "{:.2f}"),
            line_if("Revenue TTM (USD)", getattr(fund, "revenue_ttm", None), "{:,.0f}"),
            line_if("Revenue growth YoY", getattr(fund, "revenue_growth_yoy", None), "{:.1%}"),
            line_if("Gross margin", getattr(fund, "gross_margin", None), "{:.1%}"),
            line_if("Operating margin", getattr(fund, "operating_margin", None), "{:.1%}"),
            line_if("Net margin", getattr(fund, "net_margin", None), "{:.1%}"),
            line_if("ROE", getattr(fund, "return_on_equity", None), "{:.1%}"),
            line_if("EPS trailing", getattr(fund, "eps_trailing", None), "{:.2f}"),
            line_if("EPS forward", getattr(fund, "eps_forward", None), "{:.2f}"),
            line_if("Last EPS surprise", getattr(fund, "eps_surprise_last", None), "{:.1f}%"),
            line_if("Debt/Equity", getattr(fund, "debt_to_equity", None), "{:.2f}"),
            line_if("Current ratio", getattr(fund, "current_ratio", None), "{:.2f}"),
            line_if("Free cash flow (USD)", getattr(fund, "free_cash_flow", None), "{:,.0f}"),
            line_if("Dividend yield", getattr(fund, "dividend_yield", None), "{:.2%}"),
            line_if("Analyst consensus", getattr(fund, "analyst_consensus", None)),
            line_if("Analyst target", getattr(fund, "analyst_mean_target", None), "{:.2f}"),
            line_if("Analyst upside", getattr(fund, "analyst_upside", None), "{:.1%}"),
            line_if("Number of analysts", getattr(fund, "n_analysts", None)),
            line_if("Next earnings date", getattr(fund, "next_earnings_date", None)),
            line_if("Days to earnings", getattr(fund, "days_to_earnings", None)),
        ]
        spec_lines = [s for s in spec_lines if s]
        if spec_lines:
            lines.append("Key fundamentals:")
            lines.extend("  - " + s for s in spec_lines)

        lines.append(
            "This is a synthetic snapshot built at cold-start time from public "
            "fundamentals data; treat it as background context, not as a recent event."
        )
        return "\n".join(lines)

    # ----- SEC backfill -----

    def _backfill_sec(self, ticker: str) -> BackfillSourceResult:
        res = BackfillSourceResult(name="sec_edgar")
        t0 = time.time()
        try:
            # On demande les 10 derniers depots multi-form
            ok, filings, err = _safe_call(
                self.edgar_client.find_recent_8k,
                ticker,
                lookback_days=self.lookback_days,
            )
            if not ok:
                res.error = err
                res.duration_sec = time.time() - t0
                return res

            filings = filings or []
            # find_recent_8k retourne soit un EdgarFilingResult soit une liste
            if not isinstance(filings, list):
                filings = [filings] if getattr(filings, "found", False) else []

            n_added = self._index_filings(ticker, filings)
            res.success = True
            res.documents_added = n_added
        except Exception as e:
            res.error = str(e)[:200]
        finally:
            res.duration_sec = time.time() - t0
        return res

    def _index_filings(self, ticker: str, filings: List[Any]) -> int:
        """Indexe une liste d'EdgarFilingResult dans le RAG."""
        try:
            from src.knowledge.rag_store import RAGDocument  # type: ignore
        except ImportError:
            try:
                from .rag_store import RAGDocument  # type: ignore
            except ImportError:
                return 0

        n = 0
        for f in filings[: self.backfill_limit]:
            if not getattr(f, "found", False):
                continue
            text = (
                f"SEC {getattr(f, 'form_type', '?')} filed by {getattr(f, 'company_name', ticker)} "
                f"on {getattr(f, 'filing_date', '?')}. "
                f"{getattr(f, 'description', '')}"
            )
            doc_id = f"cold-sec-{ticker}-{getattr(f, 'accession_number', n)}"
            doc = RAGDocument(
                doc_id=doc_id,
                ticker=ticker,
                text=text,
                doc_type=DOC_TYPE_COLD_SEC,
                date_iso=getattr(f, "filing_date", _now_iso()),
                metadata={
                    "form_type": _short_str(getattr(f, "form_type", ""), 30),
                    "accession": _short_str(getattr(f, "accession_number", ""), 50),
                    "source": "sec_edgar",
                    "synthetic": "false",
                },
            )
            if self.rag_store.add_document(doc):
                n += 1
        return n

    # ----- Earnings calls backfill -----

    def _backfill_earnings(self, ticker: str) -> BackfillSourceResult:
        res = BackfillSourceResult(name="earnings_calls")
        t0 = time.time()
        try:
            ok, transcripts, err = _safe_call(self.earnings_fetcher, ticker)
            if not ok:
                res.error = err
                res.duration_sec = time.time() - t0
                return res
            transcripts = transcripts or []
            n = self._index_transcripts(ticker, transcripts)
            res.success = True
            res.documents_added = n
        except Exception as e:
            res.error = str(e)[:200]
        finally:
            res.duration_sec = time.time() - t0
        return res

    def _index_transcripts(
        self,
        ticker: str,
        transcripts: List[Tuple[str, str]],
    ) -> int:
        try:
            from src.knowledge.rag_store import RAGDocument  # type: ignore
        except ImportError:
            try:
                from .rag_store import RAGDocument  # type: ignore
            except ImportError:
                return 0

        n = 0
        for date_iso, text in transcripts[: self.backfill_limit]:
            if not text or len(text) < 100:
                continue
            doc_id = f"cold-call-{ticker}-{date_iso[:10]}"
            doc = RAGDocument(
                doc_id=doc_id,
                ticker=ticker,
                text=text[:5000],
                doc_type=DOC_TYPE_COLD_CALL,
                date_iso=date_iso,
                metadata={"source": "earnings_call", "synthetic": "false"},
            )
            if self.rag_store.add_document(doc):
                n += 1
        return n

    # ----- News backfill -----

    def _backfill_news(self, ticker: str) -> BackfillSourceResult:
        res = BackfillSourceResult(name="historical_news")
        t0 = time.time()
        try:
            ok, articles, err = _safe_call(
                self.news_fetcher,
                ticker,
                self.lookback_days,
            )
            if not ok:
                res.error = err
                res.duration_sec = time.time() - t0
                return res
            articles = articles or []
            n = self._index_news(ticker, articles)
            res.success = True
            res.documents_added = n
        except Exception as e:
            res.error = str(e)[:200]
        finally:
            res.duration_sec = time.time() - t0
        return res

    def _index_news(self, ticker: str, articles: List[Dict[str, Any]]) -> int:
        try:
            from src.knowledge.rag_store import RAGDocument  # type: ignore
        except ImportError:
            try:
                from .rag_store import RAGDocument  # type: ignore
            except ImportError:
                return 0

        n = 0
        for art in articles[: self.backfill_limit]:
            text = " ".join(
                filter(
                    None,
                    [
                        str(art.get("title", "")),
                        str(art.get("summary", "") or art.get("content", "")),
                    ],
                )
            )[:4000]
            if len(text) < 50:
                continue
            url = str(art.get("url", "")) or f"hist-{ticker}-{n}"
            doc_id = f"cold-news-{ticker}-{abs(hash(url)) % 10**12}"
            date_iso = str(art.get("date_iso") or art.get("published_at") or _now_iso())
            doc = RAGDocument(
                doc_id=doc_id,
                ticker=ticker,
                text=text,
                doc_type=DOC_TYPE_COLD_NEWS,
                date_iso=date_iso,
                metadata={
                    "source": _short_str(art.get("source", "historical"), 60),
                    "url": _short_str(url, 200),
                    "synthetic": "false",
                },
            )
            if self.rag_store.add_document(doc):
                n += 1
        return n

    # ----- Peer seeding -----

    def _seed_from_peers(
        self,
        ticker: str,
        peers: Sequence[str],
    ) -> Tuple[int, int, float]:
        """
        Recupere quelques documents recents d'un peer et les copie dans la
        collection du ticker cold avec doc_type=COLD_PEER.

        Returns:
            (peers_used_count, documents_added, duration_sec)
        """
        t0 = time.time()
        if not self.rag_store:
            return (0, 0, time.time() - t0)

        try:
            from src.knowledge.rag_store import RAGDocument  # type: ignore
        except ImportError:
            try:
                from .rag_store import RAGDocument  # type: ignore
            except ImportError:
                return (0, 0, time.time() - t0)

        peers_used = 0
        total_added = 0
        per_peer_quota = max(1, self.backfill_limit // max(1, len(peers)))

        for peer in peers:
            peer_size = self.count_documents(peer)
            if peer_size <= 0:
                continue

            # On utilise une query large pour recuperer un sample varie
            sample_query = f"{peer} business operations performance"
            try:
                results = self.rag_store.query(
                    ticker=peer,
                    query_text=sample_query,
                    k=per_peer_quota,
                )
            except Exception as e:
                logger.debug("[ColdStart] peer %s query echec : %s", peer, e)
                continue

            if not results:
                continue

            n_local = 0
            for r in results:
                # On retient uniquement les types fondamentaux (pas memory peer-of-peer)
                src_doc_type = getattr(r.doc, "doc_type", "")
                if src_doc_type.startswith("cold_start_peer"):
                    continue  # eviter les chaines transitives

                seed_text = f"[Peer-seed from {peer}] " + getattr(r.doc, "text", "")[:3500]
                doc_id = f"cold-peer-{ticker}-from-{peer}-{abs(hash(r.doc.doc_id)) % 10**10}"
                seed_doc = RAGDocument(
                    doc_id=doc_id,
                    ticker=ticker,
                    text=seed_text,
                    doc_type=DOC_TYPE_COLD_PEER,
                    date_iso=getattr(r.doc, "date_iso", _now_iso()),
                    metadata={
                        "source": "peer_seed",
                        "peer_ticker": peer,
                        "original_doc_type": _short_str(src_doc_type, 40),
                        "synthetic": "true",
                    },
                )
                if self.rag_store.add_document(seed_doc):
                    n_local += 1

            if n_local > 0:
                peers_used += 1
                total_added += n_local
                logger.info(
                    "[ColdStart] Seeded %d docs from peer %s -> %s",
                    n_local,
                    peer,
                    ticker,
                )

        return (peers_used, total_added, time.time() - t0)


# ---------------------------------------------------------------------------
# Resolveur de peers built-in (heuristique sectorielle)
# ---------------------------------------------------------------------------

# Mini-table sectorielle pour fallback offline (compatible smoke test).
# En prod, remplacer par fundamentals_fn + scan industry-match.
_DEFAULT_PEER_TABLE: Dict[str, List[str]] = {
    # Tech mega-caps
    "AAPL": ["MSFT", "GOOGL", "META"],
    "MSFT": ["AAPL", "GOOGL", "ORCL"],
    "GOOGL": ["META", "MSFT", "AAPL"],
    "META": ["GOOGL", "SNAP", "PINS"],
    "NVDA": ["AMD", "AVGO", "TSM"],
    "AMD": ["NVDA", "INTC", "AVGO"],
    # Energy
    "XOM": ["CVX", "OXY", "COP"],
    "CVX": ["XOM", "OXY", "COP"],
    "OXY": ["XOM", "CVX", "COP"],
    "COP": ["XOM", "CVX", "OXY"],
    # Financials
    "JPM": ["BAC", "WFC", "C"],
    "BAC": ["JPM", "WFC", "C"],
    "GS": ["MS", "JPM", "BAC"],
    # Consumer
    "AMZN": ["WMT", "COST", "TGT"],
    "TSLA": ["F", "GM", "RIVN"],
    # Health
    "JNJ": ["PFE", "MRK", "ABBV"],
    "PFE": ["JNJ", "MRK", "ABBV"],
}


def default_peer_resolver(ticker: str) -> List[str]:
    """Resolveur de peers offline (table heuristique). Override pour la prod."""
    return list(_DEFAULT_PEER_TABLE.get(ticker.upper(), []))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ----- Mock RAG store -----
    class _MockDoc:
        def __init__(self, doc_id, ticker, text, doc_type, date_iso, metadata=None):
            self.doc_id = doc_id
            self.ticker = ticker
            self.text = text
            self.doc_type = doc_type
            self.date_iso = date_iso
            self.metadata = metadata or {}

    class _MockResult:
        def __init__(self, doc):
            self.doc = doc
            self.cosine_score = 0.5
            self.temporal_score = 0.5
            self.days_old = 1.0

    class _MockRAGStore:
        def __init__(self):
            self.docs: Dict[str, List[_MockDoc]] = {}

        def collection_size(self, ticker):
            return len(self.docs.get(ticker.upper(), []))

        def add_document(self, doc):
            t = doc.ticker.upper()
            self.docs.setdefault(t, []).append(
                _MockDoc(
                    doc_id=doc.doc_id,
                    ticker=t,
                    text=doc.text,
                    doc_type=doc.doc_type,
                    date_iso=doc.date_iso,
                    metadata=doc.metadata,
                )
            )
            return True

        def query(self, ticker, query_text, k=5):
            return [_MockResult(d) for d in self.docs.get(ticker.upper(), [])[:k]]

    # On force l'import RAGDocument depuis src.knowledge.rag_store si dispo,
    # sinon on construit un dataclass equivalent en mock pour le smoke test
    try:
        from src.knowledge.rag_store import RAGDocument  # noqa: F401

        rag_doc_available = True
    except ImportError:
        rag_doc_available = False

    if not rag_doc_available:
        logger.info("WARN: src.knowledge.rag_store indisponible — smoke en mode degrade")

    rag = _MockRAGStore()

    # ----- Mock fundamentals -----
    class _Fund:
        company_name = "Occidental Petroleum"
        sector = "Energy"
        industry = "Oil & Gas E&P"
        market_cap = 55_000_000_000
        enterprise_value = 80_000_000_000
        pe_trailing = 11.5
        pe_forward = 13.2
        peg_ratio = 1.4
        price_to_book = 2.1
        ev_to_ebitda = 5.8
        revenue_ttm = 28_000_000_000
        revenue_growth_yoy = 0.05
        gross_margin = 0.62
        operating_margin = 0.32
        net_margin = 0.18
        return_on_equity = 0.21
        eps_trailing = 4.5
        eps_forward = 4.7
        eps_surprise_last = 6.0
        debt_to_equity = 0.85
        current_ratio = 1.05
        free_cash_flow = 6_500_000_000
        dividend_yield = 0.018
        analyst_consensus = "Buy"
        analyst_mean_target = 70.0
        analyst_upside = 0.12
        n_analysts = 22
        next_earnings_date = "2026-05-08"
        days_to_earnings = 13

    def fund_fn(ticker):
        return _Fund() if ticker.upper() == "OXY" else None

    # ----- Mock SEC EDGAR -----
    class _MockFiling:
        def __init__(self, form, date, desc, acc):
            self.found = True
            self.form_type = form
            self.filing_date = date
            self.company_name = "Occidental Petroleum"
            self.cik = "0000797468"
            self.accession_number = acc
            self.description = desc

    class _MockEdgar:
        def find_recent_8k(self, ticker, lookback_days=365):
            if ticker.upper() != "OXY":
                return []
            return [
                _MockFiling("10-Q", "2026-02-12", "Q4 2025 Results", "0001-26-001"),
                _MockFiling("8-K", "2026-01-15", "Item 2.02 Earnings preliminary", "0001-26-002"),
                _MockFiling("10-K", "2025-11-10", "Annual report 2025", "0001-25-099"),
            ]

    # ----- Mock earnings fetcher -----
    def earnings_fn(ticker):
        if ticker.upper() != "OXY":
            return []
        return [
            (
                "2026-02-12T16:00:00+00:00",
                "Q4 2025 earnings call. Management discussed permian growth and capex discipline. " * 10,
            ),
            (
                "2025-11-10T16:00:00+00:00",
                "Q3 2025 earnings call. Strong cash flow, dividend increase considered. " * 10,
            ),
        ]

    # ----- Mock news fetcher -----
    def news_fn(ticker, days):
        if ticker.upper() != "OXY":
            return []
        return [
            {
                "title": "Occidental beats Q4 estimates on Permian production",
                "summary": "Occidental Petroleum reported Q4 EPS of $1.23 vs $1.05 expected.",
                "url": "https://example.com/oxy1",
                "date_iso": "2026-02-13T09:00:00+00:00",
                "source": "Reuters",
            },
            {
                "title": "Berkshire Hathaway increases OXY stake",
                "summary": "Berkshire Hathaway disclosed an increase to 28% in its OXY stake.",
                "url": "https://example.com/oxy2",
                "date_iso": "2026-01-20T10:00:00+00:00",
                "source": "Bloomberg",
            },
        ]

    # ----- Build manager -----
    csm = ColdStartManager(
        rag_store=rag,
        edgar_client=_MockEdgar(),
        earnings_fetcher=earnings_fn,
        news_fetcher=news_fn,
        fundamentals_fn=fund_fn,
        peer_resolver=default_peer_resolver,
        min_docs=5,
        backfill_limit=10,
    )

    logger.info("=== SC1 OXY (cold) ===")
    assert csm.is_cold("OXY"), "OXY doit etre cold initialement"
    rep = csm.bootstrap("OXY", company_name="Occidental Petroleum", sector="Energy")
    print(rep.summary())
    print(rep.to_prompt_block())
    assert rep.was_cold
    assert rep.docs_after >= 5, f"Apres backfill on devrait avoir >=5 docs, got {rep.docs_after}"
    assert rep.profile_indexed
    assert any(s.name == "sec_edgar" and s.documents_added > 0 for s in rep.sources)
    assert any(s.name == "earnings_calls" and s.documents_added > 0 for s in rep.sources)
    assert any(s.name == "historical_news" and s.documents_added > 0 for s in rep.sources)

    logger.info("\n=== SC2 OXY (deja chaud) ===")
    rep2 = csm.bootstrap("OXY", company_name="Occidental Petroleum")
    print(rep2.summary())
    assert not rep2.was_cold
    assert rep2.docs_before == rep.docs_after

    logger.info("\n=== SC3 Ticker inconnu (peer fallback) ===")
    # Ticker FAKE avec aucune source : doit tomber sur peer_seed via OXY (peer fictif)
    csm2 = ColdStartManager(
        rag_store=rag,  # rag deja peuple par OXY ci-dessus
        edgar_client=None,  # pas de SEC
        earnings_fetcher=None,  # pas de calls
        news_fetcher=None,  # pas de news
        fundamentals_fn=None,  # pas de fundamentals
        peer_resolver=lambda t: ["OXY"],  # OXY comme peer artificiel
        min_docs=3,
        backfill_limit=5,
        peer_fallback=True,
    )
    rep3 = csm2.bootstrap("FAKE", company_name="Fake Co")
    print(rep3.summary())
    assert rep3.was_cold
    assert "OXY" in rep3.peers_used
    assert rep3.docs_after >= 1, f"Peer seeding devrait copier des docs, got {rep3.docs_after}"

    logger.info("\n=== Default peer resolver ===")
    print(f"  AAPL -> {default_peer_resolver('AAPL')}")
    print(f"  XOM  -> {default_peer_resolver('XOM')}")
    print(f"  ZZZZ -> {default_peer_resolver('ZZZZ')}")
    assert "MSFT" in default_peer_resolver("AAPL")
    assert "CVX" in default_peer_resolver("XOM")
    assert default_peer_resolver("ZZZZ") == []

    logger.info("\nOK - tous les flux cold-start fonctionnent")
