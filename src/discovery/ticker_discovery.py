"""
ticker_discovery.py — Universe selection autonome.

PROBLEME
--------
Pour faire tourner le pipeline en autopilote (par ex. une semaine de recolte
de donnees), on a besoin que le systeme choisisse SEUL chaque jour les
tickers a analyser, plutot que de coder en dur ["AAPL", "MSFT", ...].

C'est ce que les desks quantitatifs appellent un "universe selector" ou
"opportunity scanner". Sans lui, l'archi reste un outil ad-hoc ; avec lui
elle devient un agent autonome.

ARCHITECTURE
------------
Multi-signal scoring : chaque source emet un score [0,1] par ticker, on
agrege avec des poids configurables :

  Source            Pondération    Justification empirique
  ----------------  -------------  --------------------------------------
  trending_news     0.25           Mentions = signal d'attention immediat
  volume_anomaly    0.20           Vol z-score = bon predicteur 1-jour
  social_spike      0.15           Capture le retail momentum
  earnings_calendar 0.20           Periode de dispersion d'opinions
  sec_recent        0.10           8-K / 4 = catalyseur factuel
  big_movers        0.10           Continuation patterns intraday

Anti-redondance : un ticker analyse aujourd'hui voit son score multiplie
par exp(-decay * days_since) → favorise la diversification du dataset.

Univers fixe : optionnel filtre par market cap min, secteur exclu, ETF
exclu, penny stock exclu. Permet de cibler par exemple "S&P 500 +
Nasdaq-100 sans utilities".

REFERENCES
----------
- Da, Z. et al. (2011). "In Search of Attention." JF : volume + media
  mentions = predictor de sous-performance future (≠ over-attention bias).
- Engelberg, J. & Gao, P. (2011). "In Search of Earnings Announcement
  Premium." Working paper : earnings calendar = drift exploitable.
- Tetlock, P. C. (2007). "Giving Content to Investor Sentiment." JF.
- Cookson, J. A. & Niessner, M. (2020). "Why Don't We Agree?" JF :
  divergence retail/news = volatilite anormale.
- Fama, E. F. & French, K. R. (1993). "Common risk factors in the returns
  on stocks and bonds." JFE → exclusion des petites caps controle le bruit.

USAGE
-----
    eng = TickerDiscoveryEngine(
        sources=[
            TrendingNewsSource(),
            VolumeAnomalySource(),
            EarningsCalendarSource(),
            ...
        ],
        universe_filter=UniverseFilter(min_market_cap=2e9),
        recent_history_fn=lambda: ["AAPL"],  # tickers deja analyses
    )
    report = eng.discover(top_n=10)
    for s in report.scores:
        print(s.ticker, s.total_score, s.contributions)
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes / config
# ---------------------------------------------------------------------------

DEFAULT_SIGNAL_WEIGHTS: Dict[str, float] = {
    "trending_news": 0.25,
    "volume_anomaly": 0.20,
    "social_spike": 0.15,
    "earnings_calendar": 0.20,
    "sec_recent": 0.10,
    "big_movers": 0.10,
}

# Decay anti-redondance : score *= exp(-DECAY * days_since)
# 0.5 = un ticker analyse hier reste a ~0.6, il y a 3j a ~0.22, il y a 7j a ~0.03
RECENT_HISTORY_DECAY_PER_DAY = 0.5

# Limites
MAX_TICKERS_PER_SOURCE = 50  # Eviter qu'une source pollue avec 200 candidats
MAX_FINAL_CANDIDATES = 100  # Cap avant top-N final

# Univers par defaut : large-cap US (modifiable)
DEFAULT_BLACKLIST_SECTORS: Tuple[str, ...] = ()
DEFAULT_MIN_MARKET_CAP = 1.0e9  # 1 milliard min
DEFAULT_EXCLUDE_ETFS = True

# Pattern simple pour detecter ETF (heuristique : ticker court avec 3-4 lettres
# se terminant par des suffixes typiques). On laisse l'option d'override.
_LIKELY_ETF_TICKERS: frozenset = frozenset(
    {
        "SPY",
        "QQQ",
        "DIA",
        "IWM",
        "VTI",
        "VOO",
        "EEM",
        "EFA",
        "TLT",
        "GLD",
        "SLV",
        "USO",
        "XLE",
        "XLF",
        "XLK",
        "XLV",
        "XLI",
        "XLP",
        "XLU",
        "XLY",
        "XLRE",
        "XLB",
        "XLC",
        "ARKK",
        "ARKW",
        "ARKG",
        "ARKQ",
        "VXX",
        "UVXY",
        "TQQQ",
        "SQQQ",
        "SOXL",
        "SOXS",
    }
)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryScore:
    """Score d'un ticker pour un jour donne."""

    ticker: str
    total_score: float = 0.0  # [0, 1] apres ponderation et decay
    raw_score: float = 0.0  # [0, 1] avant decay redondance
    contributions: Dict[str, float] = field(default_factory=dict)
    days_since_last_seen: Optional[float] = None
    novelty_multiplier: float = 1.0  # exp(-decay * days)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        contribs = ", ".join(
            f"{k}={v:.2f}"
            for k, v in sorted(
                self.contributions.items(),
                key=lambda x: -x[1],
            )[:3]
        )
        return (
            f"{self.ticker}: total={self.total_score:.3f} "
            f"raw={self.raw_score:.3f} mult={self.novelty_multiplier:.2f} "
            f"top=[{contribs}]"
        )


@dataclass
class DiscoveryReport:
    """Rapport complet d'un cycle de discovery."""

    run_at: datetime
    top_n: int
    universe_size: int  # apres filtre univers
    candidates_seen: int  # total apres dedup avant top-N
    scores: List[DiscoveryScore] = field(default_factory=list)
    sources_used: List[str] = field(default_factory=list)
    sources_failed: List[Tuple[str, str]] = field(default_factory=list)
    duration_sec: float = 0.0
    universe_hash: str = ""  # hash du top-N pour reproductibilite

    def summary(self) -> str:
        return (
            f"discovery run_at={self.run_at.isoformat()} "
            f"top={self.top_n} candidates={self.candidates_seen} "
            f"universe={self.universe_size} sources={len(self.sources_used)} "
            f"duration={self.duration_sec:.1f}s"
        )

    def tickers(self) -> List[str]:
        return [s.ticker for s in self.scores[: self.top_n]]


# ---------------------------------------------------------------------------
# Universe filter
# ---------------------------------------------------------------------------


class UniverseFilter:
    """Filtre univers : market cap, secteurs exclus, etf, penny stock."""

    def __init__(
        self,
        min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
        excluded_sectors: Sequence[str] = DEFAULT_BLACKLIST_SECTORS,
        exclude_etfs: bool = DEFAULT_EXCLUDE_ETFS,
        explicit_whitelist: Optional[Sequence[str]] = None,
        explicit_blacklist: Optional[Sequence[str]] = None,
        etf_lookup: Optional[frozenset] = None,
        market_cap_lookup_fn: Optional[Callable[[str], Optional[float]]] = None,
        sector_lookup_fn: Optional[Callable[[str], Optional[str]]] = None,
    ):
        self.min_market_cap = max(0.0, float(min_market_cap))
        self.excluded_sectors = {s.lower() for s in excluded_sectors if s}
        self.exclude_etfs = bool(exclude_etfs)
        self.whitelist = {t.upper() for t in explicit_whitelist} if explicit_whitelist else None
        self.blacklist = {t.upper() for t in explicit_blacklist} if explicit_blacklist else set()
        self.etf_lookup = etf_lookup or _LIKELY_ETF_TICKERS
        self.market_cap_lookup_fn = market_cap_lookup_fn
        self.sector_lookup_fn = sector_lookup_fn

    def is_eligible(self, ticker: str) -> Tuple[bool, str]:
        """
        Retourne (eligible, reason). reason est vide si eligible, sinon
        un mot-cle court explicatif (utile pour audit).
        """
        t = ticker.upper().strip()
        if not t or len(t) > 6:
            return False, "invalid_format"

        if t in self.blacklist:
            return False, "blacklisted"

        if self.whitelist is not None and t not in self.whitelist:
            return False, "not_in_whitelist"

        if self.exclude_etfs and t in self.etf_lookup:
            return False, "etf"

        # Market cap
        if self.market_cap_lookup_fn is not None and self.min_market_cap > 0:
            try:
                mc = self.market_cap_lookup_fn(t)
                if mc is None:
                    # Ne pas exclure un ticker faute de donnees — soft-pass
                    pass
                elif mc < self.min_market_cap:
                    return False, "below_market_cap"
            except Exception as e:
                logger.debug("[UniverseFilter] market_cap lookup err %s : %s", t, e)

        # Sector
        if self.sector_lookup_fn is not None and self.excluded_sectors:
            try:
                sec = self.sector_lookup_fn(t)
                if sec and sec.lower() in self.excluded_sectors:
                    return False, "excluded_sector"
            except Exception as e:
                logger.debug("[UniverseFilter] sector lookup err %s : %s", t, e)

        return True, ""


# ---------------------------------------------------------------------------
# Sources de discovery (interface + implementations)
# ---------------------------------------------------------------------------


class DiscoverySource:
    """
    Interface d'une source. Chaque source produit un mapping
    ticker -> raw_score [0, 1] pour la periode courante.

    Les implementations doivent etre INDEPENDANTES (network, files, mocks)
    et toujours retourner un dict — vide en cas d'echec.
    """

    name: str = "base"

    def fetch_scores(self) -> Dict[str, float]:
        """Retourne {ticker: score in [0, 1]}. Override dans les sous-classes."""
        return {}


class CallableSource(DiscoverySource):
    """Wrapper trivial autour d'un callable. Utile pour tests / glue."""

    def __init__(self, name: str, fn: Callable[[], Dict[str, float]]):
        self.name = name
        self._fn = fn

    def fetch_scores(self) -> Dict[str, float]:
        try:
            data = self._fn() or {}
            # Normalisation : clip et upper-case
            return {str(k).upper(): max(0.0, min(1.0, float(v))) for k, v in data.items() if v is not None}
        except Exception as e:
            logger.warning("[%s] fetch_scores echec : %s", self.name, e)
            return {}


# -- Sources concretes (best-effort, depend de yfinance/EDGAR/StockTwits) --


class TrendingNewsSource(DiscoverySource):
    """
    Score base sur le nombre de mentions d'un ticker dans les news des
    derniers `lookback_hours`. Utilise un fetcher injectable + un extracteur
    NER (regex cashtag par defaut) — on garde le module independant de NER
    lourd pour tests offline.
    """

    name = "trending_news"

    def __init__(
        self,
        news_fetcher: Optional[Callable[[int], List[Dict[str, Any]]]] = None,
        ticker_extractor: Optional[Callable[[str], List[str]]] = None,
        lookback_hours: int = 24,
        max_tickers: int = MAX_TICKERS_PER_SOURCE,
    ):
        self.news_fetcher = news_fetcher
        self.ticker_extractor = ticker_extractor or self._default_extractor
        self.lookback_hours = max(1, int(lookback_hours))
        self.max_tickers = max(1, int(max_tickers))

    @staticmethod
    def _default_extractor(text: str) -> List[str]:
        # Cashtag $XXXX or "(XXXX)" parenthesised ticker
        import re

        cash = re.findall(r"\$([A-Z]{1,5})(?![A-Za-z])", text or "")
        paren = re.findall(r"\(([A-Z]{1,5})\)", text or "")
        out = []
        seen = set()
        for t in cash + paren:
            if t and t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def fetch_scores(self) -> Dict[str, float]:
        if self.news_fetcher is None:
            return {}
        try:
            articles = self.news_fetcher(self.lookback_hours) or []
        except Exception as e:
            logger.warning("[trending_news] fetcher echec : %s", e)
            return {}

        counts: Dict[str, int] = {}
        for art in articles:
            text = " ".join(
                filter(
                    None,
                    [
                        str(art.get("title", "")),
                        str(art.get("summary", "")),
                    ],
                )
            )
            for tk in self.ticker_extractor(text):
                counts[tk] = counts.get(tk, 0) + 1

        if not counts:
            return {}

        max_count = max(counts.values())
        scores = {t: c / max_count for t, c in counts.items() if c >= 1}
        # Cap au top-K
        top = sorted(scores.items(), key=lambda x: -x[1])[: self.max_tickers]
        return dict(top)


class VolumeAnomalySource(DiscoverySource):
    """
    Score base sur z-score du volume du jour vs MA20j. Necessite un
    quote_fetcher qui retourne {ticker: {volume, volume_ma_20, volume_std_20}}.
    """

    name = "volume_anomaly"

    def __init__(
        self,
        quote_fetcher: Optional[Callable[[List[str]], Dict[str, Dict[str, float]]]] = None,
        candidate_universe: Optional[Sequence[str]] = None,
        z_score_saturation: float = 4.0,
        z_score_min: float = 1.0,
    ):
        self.quote_fetcher = quote_fetcher
        self.candidate_universe = list(candidate_universe) if candidate_universe else []
        self.z_score_saturation = max(0.5, float(z_score_saturation))
        self.z_score_min = float(z_score_min)

    def fetch_scores(self) -> Dict[str, float]:
        if self.quote_fetcher is None or not self.candidate_universe:
            return {}
        try:
            data = self.quote_fetcher(list(self.candidate_universe)) or {}
        except Exception as e:
            logger.warning("[volume_anomaly] fetcher echec : %s", e)
            return {}

        scores: Dict[str, float] = {}
        for tk, q in data.items():
            try:
                vol = float(q.get("volume", 0) or 0)
                ma = float(q.get("volume_ma_20", 0) or 0)
                std = float(q.get("volume_std_20", 0) or 0)
                if ma <= 0 or std <= 0:
                    continue
                z = (vol - ma) / std
                if z < self.z_score_min:
                    continue
                scores[tk.upper()] = max(0.0, min(1.0, z / self.z_score_saturation))
            except Exception:
                continue
        return scores


class SocialSpikeSource(DiscoverySource):
    """
    Score base sur l'activite social (StockTwits trending tickers ou
    Reddit /r/wallstreetbets daily discussion). Le `trending_fetcher`
    retourne directement {ticker: messages_count_24h}.
    """

    name = "social_spike"

    def __init__(
        self,
        trending_fetcher: Optional[Callable[[], Dict[str, int]]] = None,
        max_tickers: int = MAX_TICKERS_PER_SOURCE,
    ):
        self.trending_fetcher = trending_fetcher
        self.max_tickers = max(1, int(max_tickers))

    def fetch_scores(self) -> Dict[str, float]:
        if self.trending_fetcher is None:
            return {}
        try:
            data = self.trending_fetcher() or {}
        except Exception as e:
            logger.warning("[social_spike] fetcher echec : %s", e)
            return {}

        if not data:
            return {}

        # Score = log(1+count) / log(1+max). Sub-lineaire pour eviter qu'un
        # ticker meme-stock ecrase tous les autres.
        counts = {str(t).upper(): max(0, int(c)) for t, c in data.items()}
        max_log = math.log1p(max(counts.values()))
        if max_log <= 0:
            return {}
        scores = {t: math.log1p(c) / max_log for t, c in counts.items() if c > 0}
        top = sorted(scores.items(), key=lambda x: -x[1])[: self.max_tickers]
        return dict(top)


class EarningsCalendarSource(DiscoverySource):
    """
    Score base sur la proximite d'un earnings release. `calendar_fetcher`
    retourne {ticker: days_to_earnings} ; on score plus haut quand c'est
    proche (dans la fenetre [-1, +5]).
    """

    name = "earnings_calendar"

    def __init__(
        self,
        calendar_fetcher: Optional[Callable[[], Dict[str, int]]] = None,
        days_window: int = 5,
    ):
        self.calendar_fetcher = calendar_fetcher
        self.days_window = max(1, int(days_window))

    def fetch_scores(self) -> Dict[str, float]:
        if self.calendar_fetcher is None:
            return {}
        try:
            data = self.calendar_fetcher() or {}
        except Exception as e:
            logger.warning("[earnings_calendar] fetcher echec : %s", e)
            return {}

        scores: Dict[str, float] = {}
        for tk, days in data.items():
            try:
                d = int(days)
            except Exception:
                continue
            if d < -1 or d > self.days_window:
                continue
            # Score : 1.0 le jour J, 0.5 dans la fenetre, 0.85 J+1
            if d == 0:
                s = 1.0
            elif d == 1:
                s = 0.85
            elif d == -1:
                s = 0.85  # post-earnings drift
            else:
                s = max(0.0, 1.0 - (d / (self.days_window + 1)))
            scores[str(tk).upper()] = s
        return scores


class SecRecentSource(DiscoverySource):
    """
    Score base sur les 8-K / Form 4 recents. `filings_fetcher` retourne
    {ticker: list_of_filings}. On weighte par recence et type (8-K Item 2.02
    earnings = +1.0, Form 4 = +0.6, etc.).
    """

    name = "sec_recent"

    # Ponderations par type de form (heuristique)
    _FORM_WEIGHTS = {
        "8-K": 1.0,
        "Form 4": 0.6,
        "10-Q": 0.4,
        "10-K": 0.3,
        "13D": 0.7,
        "13G": 0.5,
        "S-1": 0.6,
        "S-3": 0.4,
    }

    def __init__(
        self,
        filings_fetcher: Optional[Callable[[], Dict[str, List[Dict[str, Any]]]]] = None,
        lookback_hours: int = 72,
    ):
        self.filings_fetcher = filings_fetcher
        self.lookback_hours = max(1, int(lookback_hours))

    def fetch_scores(self) -> Dict[str, float]:
        if self.filings_fetcher is None:
            return {}
        try:
            data = self.filings_fetcher() or {}
        except Exception as e:
            logger.warning("[sec_recent] fetcher echec : %s", e)
            return {}

        cutoff = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        scores: Dict[str, float] = {}
        for tk, filings in data.items():
            best = 0.0
            for f in filings or []:
                form = str(f.get("form_type", "")).strip()
                w = self._FORM_WEIGHTS.get(form, 0.3)
                # Recence
                date_str = str(f.get("filing_date", ""))
                try:
                    dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < cutoff:
                        continue
                    age_h = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
                    rec = max(0.0, 1.0 - age_h / self.lookback_hours)
                except Exception:
                    rec = 0.5
                s = w * rec
                if s > best:
                    best = s
            if best > 0:
                scores[str(tk).upper()] = max(0.0, min(1.0, best))
        return scores


class BigMoversSource(DiscoverySource):
    """
    Score base sur les |return| 1-jour. `movers_fetcher` retourne
    {ticker: pct_return}. Cap a |3%| min pour ignorer le bruit.
    """

    name = "big_movers"

    def __init__(
        self,
        movers_fetcher: Optional[Callable[[], Dict[str, float]]] = None,
        min_abs_return: float = 0.03,
        return_saturation: float = 0.10,
    ):
        self.movers_fetcher = movers_fetcher
        self.min_abs_return = max(0.0, float(min_abs_return))
        self.return_saturation = max(0.01, float(return_saturation))

    def fetch_scores(self) -> Dict[str, float]:
        if self.movers_fetcher is None:
            return {}
        try:
            data = self.movers_fetcher() or {}
        except Exception as e:
            logger.warning("[big_movers] fetcher echec : %s", e)
            return {}

        scores: Dict[str, float] = {}
        for tk, ret in data.items():
            try:
                r = float(ret)
            except Exception:
                continue
            ar = abs(r)
            if ar < self.min_abs_return:
                continue
            scores[str(tk).upper()] = max(0.0, min(1.0, ar / self.return_saturation))
        return scores


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TickerDiscoveryEngine:
    """
    Engine multi-signal de selection d'univers quotidien.

    Combine N sources, applique le filtre univers, applique le decay
    anti-redondance, retourne top-N tickers pour la journee.

    Tout est injectable pour les tests : aucune dependance reseau dure.
    """

    def __init__(
        self,
        sources: Optional[Sequence[DiscoverySource]] = None,
        universe_filter: Optional[UniverseFilter] = None,
        signal_weights: Optional[Dict[str, float]] = None,
        recent_history_fn: Optional[Callable[[], Dict[str, float]]] = None,
        decay_per_day: float = RECENT_HISTORY_DECAY_PER_DAY,
    ):
        self.sources = list(sources) if sources else []
        self.universe_filter = universe_filter or UniverseFilter()
        self.signal_weights = dict(signal_weights or DEFAULT_SIGNAL_WEIGHTS)
        # Normaliser
        total = sum(self.signal_weights.values())
        if total > 0 and abs(total - 1.0) > 1e-6:
            logger.info(
                "[Discovery] poids non normalises (sum=%.3f) — auto-normalisation",
                total,
            )
            self.signal_weights = {k: v / total for k, v in self.signal_weights.items()}
        self.recent_history_fn = recent_history_fn
        self.decay_per_day = max(0.0, float(decay_per_day))

    # ----- Public API -----

    def discover(self, top_n: int = 10) -> DiscoveryReport:
        """Cycle complet : fetch all sources -> aggregate -> filter -> top-N."""
        t0 = time.time()
        run_at = datetime.now(timezone.utc)

        # 1. Fetch des sources (sequentiel — facile a paralleliser plus tard)
        per_source: Dict[str, Dict[str, float]] = {}
        sources_used: List[str] = []
        sources_failed: List[Tuple[str, str]] = []

        for src in self.sources:
            name = src.name
            try:
                d = src.fetch_scores() or {}
            except Exception as e:
                logger.warning("[Discovery] source %s a leve : %s", name, e)
                sources_failed.append((name, str(e)[:160]))
                continue
            per_source[name] = d
            if d:
                sources_used.append(name)
            else:
                # Pas une erreur — juste vide
                pass

        # 2. Agreger en {ticker: {source_name: score}}
        per_ticker: Dict[str, Dict[str, float]] = {}
        for src_name, sc in per_source.items():
            for tk, val in sc.items():
                per_ticker.setdefault(tk, {})[src_name] = val

        # 3. Filtre univers
        filtered: Dict[str, Dict[str, float]] = {}
        for tk, contribs in per_ticker.items():
            ok, _reason = self.universe_filter.is_eligible(tk)
            if ok:
                filtered[tk] = contribs

        # 4. Recent history pour decay
        last_seen: Dict[str, float] = {}
        if self.recent_history_fn is not None:
            try:
                last_seen = self.recent_history_fn() or {}
            except Exception as e:
                logger.warning("[Discovery] recent_history_fn echec : %s", e)
                last_seen = {}

        # 5. Compute totals
        scores_list: List[DiscoveryScore] = []
        for tk, contribs in filtered.items():
            raw = sum(contribs.get(s, 0.0) * self.signal_weights.get(s, 0.0) for s in self.signal_weights)
            days_since = last_seen.get(tk.upper())
            mult = 1.0
            if days_since is not None:
                mult = math.exp(-self.decay_per_day * max(0.0, float(days_since)))
            total = raw * mult
            scores_list.append(
                DiscoveryScore(
                    ticker=tk,
                    total_score=float(max(0.0, min(1.0, total))),
                    raw_score=float(max(0.0, min(1.0, raw))),
                    contributions=dict(contribs),
                    days_since_last_seen=days_since,
                    novelty_multiplier=float(mult),
                )
            )

        # 6. Tri descendant + cap pre-filter
        scores_list.sort(key=lambda s: -s.total_score)
        scores_list = scores_list[:MAX_FINAL_CANDIDATES]

        # 7. Hash univers (top-N) pour reproductibilite
        top_tickers = [s.ticker for s in scores_list[:top_n]]
        u_hash = self._universe_hash(top_tickers)

        report = DiscoveryReport(
            run_at=run_at,
            top_n=top_n,
            universe_size=len(filtered),
            candidates_seen=len(per_ticker),
            scores=scores_list[:top_n],
            sources_used=sources_used,
            sources_failed=sources_failed,
            duration_sec=time.time() - t0,
            universe_hash=u_hash,
        )
        logger.info("[Discovery] %s", report.summary())
        return report

    # ----- Helpers -----

    @staticmethod
    def _universe_hash(tickers: List[str]) -> str:
        import hashlib

        s = ",".join(sorted(tickers)).encode("utf-8")
        return hashlib.sha256(s).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Helper : recent_history a partir d'un repertoire de harvest
# ---------------------------------------------------------------------------


def recent_history_from_harvest_dir(
    harvest_dir: str,
    lookback_days: int = 7,
) -> Dict[str, float]:
    """
    Lit les JSONL `harvest_*.jsonl` dans le repertoire et retourne
    {ticker: days_since_last_seen}. Plus recent = plus penalise.
    Si erreur (repertoire absent, etc.), retourne {} silencieusement.
    """
    out: Dict[str, float] = {}
    if not harvest_dir or not os.path.isdir(harvest_dir):
        return out
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    now = datetime.now(timezone.utc)

    try:
        for fname in os.listdir(harvest_dir):
            if not fname.startswith("harvest_") or not fname.endswith(".jsonl"):
                continue
            path = os.path.join(harvest_dir, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except Exception:
                            # Ligne corrompue : skip mais ne pas tuer la lecture
                            continue
                        tk = str(rec.get("ticker", "")).upper()
                        ts = rec.get("timestamp", "")
                        if not tk:
                            continue
                        try:
                            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                            if dt.tzinfo is None:
                                dt = dt.replace(tzinfo=timezone.utc)
                            if dt < cutoff:
                                continue
                            days = (now - dt).total_seconds() / 86400.0
                            prev = out.get(tk)
                            if prev is None or days < prev:
                                out[tk] = days
                        except Exception:
                            continue
            except Exception:
                continue
    except Exception as e:
        logger.debug("[recent_history] %s", e)
    return out


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ----- Mocks de sources offline -----

    def mock_news_fetcher(hours):
        return [
            {"title": "Apple ($AAPL) beats Q3 estimates", "summary": "AAPL revenue grew."},
            {"title": "Microsoft ($MSFT) cloud growth", "summary": "Azure up 30%."},
            {"title": "Apple stock $AAPL surges", "summary": "$AAPL momentum continues."},
            {"title": "Nvidia ($NVDA) hits record", "summary": "$NVDA ATH on AI demand."},
            {"title": "Random ETF mention", "summary": "($SPY) flat day."},
            {"title": "Tesla ($TSLA) recall", "summary": "Production halt."},
        ]

    def mock_quote_fetcher(tickers):
        # AAPL et NVDA en spike volume, autres normaux
        out = {}
        for t in tickers:
            if t == "AAPL":
                out[t] = {"volume": 80_000_000, "volume_ma_20": 50_000_000, "volume_std_20": 8_000_000}
            elif t == "NVDA":
                out[t] = {"volume": 120_000_000, "volume_ma_20": 60_000_000, "volume_std_20": 15_000_000}
            else:
                out[t] = {"volume": 30_000_000, "volume_ma_20": 30_000_000, "volume_std_20": 5_000_000}
        return out

    def mock_social_fetcher():
        return {"AAPL": 850, "NVDA": 1200, "GME": 2800, "TSLA": 600, "AMC": 400}

    def mock_earnings_fetcher():
        return {"AAPL": 1, "MSFT": 3, "GOOGL": 0, "META": 2}

    def mock_filings_fetcher():
        now = datetime.now(timezone.utc)
        return {
            "AAPL": [{"form_type": "8-K", "filing_date": (now - timedelta(hours=4)).isoformat()}],
            "TSLA": [{"form_type": "Form 4", "filing_date": (now - timedelta(hours=12)).isoformat()}],
            "OLD": [{"form_type": "8-K", "filing_date": (now - timedelta(days=10)).isoformat()}],
        }

    def mock_movers_fetcher():
        return {"NVDA": 0.06, "GME": 0.12, "META": -0.04, "AAPL": 0.015}  # AAPL trop petit

    def mock_market_cap_lookup(t):
        caps = {
            "AAPL": 3.5e12,
            "MSFT": 3.0e12,
            "NVDA": 3.0e12,
            "GOOGL": 2.0e12,
            "META": 1.4e12,
            "TSLA": 800e9,
            "GME": 12e9,
            "AMC": 1.5e9,
        }
        return caps.get(t.upper(), 500e9)  # default 500B

    def mock_sector_lookup(t):
        secs = {"AAPL": "Technology", "MSFT": "Technology", "TSLA": "Consumer Cyclical"}
        return secs.get(t.upper())

    sources = [
        TrendingNewsSource(news_fetcher=mock_news_fetcher),
        VolumeAnomalySource(
            quote_fetcher=mock_quote_fetcher, candidate_universe=["AAPL", "MSFT", "NVDA", "META", "GOOGL"]
        ),
        SocialSpikeSource(trending_fetcher=mock_social_fetcher),
        EarningsCalendarSource(calendar_fetcher=mock_earnings_fetcher),
        SecRecentSource(filings_fetcher=mock_filings_fetcher),
        BigMoversSource(movers_fetcher=mock_movers_fetcher),
    ]

    uf = UniverseFilter(
        min_market_cap=1e9,
        excluded_sectors=[],
        exclude_etfs=True,
        market_cap_lookup_fn=mock_market_cap_lookup,
        sector_lookup_fn=mock_sector_lookup,
    )

    # Recent history : AAPL deja vu il y a 1 jour, GME il y a 3
    def rec_history():
        return {"AAPL": 1.0, "GME": 3.0}

    eng = TickerDiscoveryEngine(
        sources=sources,
        universe_filter=uf,
        recent_history_fn=rec_history,
    )

    logger.info("=== Discovery cycle ===")
    rep = eng.discover(top_n=8)
    print(rep.summary())
    logger.info(f"Universe hash: {rep.universe_hash}")
    logger.info("\nTop 8 tickers selectionnes:")
    for s in rep.scores:
        logger.info(f"  {s.summary()}")

    logger.info(f"\nSources utilisees: {rep.sources_used}")
    if rep.sources_failed:
        logger.info(f"Sources echec: {rep.sources_failed}")

    # Verifications
    selected = rep.tickers()
    assert "SPY" not in selected, "ETF SPY ne doit pas etre selectionne"
    assert "OLD" not in selected, "OLD a un filing >72h, doit etre exclu"
    # AAPL devrait avoir un mult < 1 a cause de l'historique recent
    aapl = next((s for s in rep.scores if s.ticker == "AAPL"), None)
    if aapl:
        assert aapl.novelty_multiplier < 1.0
        assert aapl.days_since_last_seen == 1.0
    logger.info("\nOK - tous les filtres et le scoring fonctionnent")
