"""
social_sentiment.py — Flux de sentiment social (Reddit + StockTwits).

OBJECTIF
--------
Capter la psychologie retail (petits porteurs) en temps reel via :
  - Reddit r/wallstreetbets, r/stocks, r/investing (via API publique ou
    snapshot Pushshift)
  - StockTwits cashtags ($AAPL, $TSLA) via API publique /api/2/streams/symbol
  - Twitter/X : hook optionnel (API v2 payante — pas active par defaut)

POURQUOI
--------
Le sentiment social est orthogonal au sentiment news :
  1. Les retail investors agissent sur des signaux SOCIAUX pas sur les
     10-K. Leur activite cree du momentum a court terme (meme-stock effect).
  2. Une divergence news=positif / social=negatif (ou inverse) est un
     signal FORT de desalignement entre informe (smart money) et retail.
  3. Le VOLUME de mentions sociales est un proxy d'attention retail —
     Da et al. (2011), "In Search of Attention", Journal of Finance.

SOURCES
-------
- Reddit : via PRAW (si installe) ou via l'API JSON publique de Reddit
  (https://www.reddit.com/r/wallstreetbets/search.json?q=$AAPL&restrict_sr=1
  — rate-limited mais sans cle). Fallback local : snapshots JSON.
- StockTwits : API publique https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json
  Pas de cle requise. ~200 messages par endpoint call, cache possible.
- Twitter/X : via tweepy si TWITTER_BEARER_TOKEN dispo. Sinon ignore.

SCORING
-------
Chaque message recoit un score de sentiment via :
  1. StockTwits "entities.sentiment" si fourni (Bullish/Bearish) — gold standard.
  2. VADER (Hutto & Gilbert 2014) si le package est dispo — robuste sur texte informel.
  3. Loughran-McDonald financial dictionary comme fallback pur Python.

On produit :
  - sentiment_score [-1, +1] : moyenne ponderee par source_weight
  - mention_volume : nombre total de messages (dernieres 24h par defaut)
  - volume_zscore : z-score du volume vs baseline 7j (detection spike)
  - bot_ratio : fraction de messages a faible unique-content-ratio
  - bullish_ratio / bearish_ratio : distribution des labels

REFERENCES
----------
- Chen, H. et al. (2014). "Wisdom of Crowds: The Value of Stock Opinions
  Transmitted Through Social Media." Review of Financial Studies.
- Bollen, J. et al. (2011). "Twitter mood predicts the stock market."
  Journal of Computational Science.
- Renault, T. (2017). "Intraday online investor sentiment and return patterns
  in the U.S. stock market." Journal of Banking & Finance.
- Da, Z. et al. (2011). "In Search of Attention." Journal of Finance.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import json
import logging
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})(?![A-Za-z])")

STOCKTWITS_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
REDDIT_SEARCH_URL = "https://www.reddit.com/r/{subreddit}/search.json?q={query}&restrict_sr=1&sort=new&limit=100&t=week"
REDDIT_USER_AGENT = "Locus-MultiAgent-Research/1.0 (financial news research)"

# Subreddits de reference
DEFAULT_SUBREDDITS = ["wallstreetbets", "stocks", "investing", "StockMarket"]

# Ponderation par source (moins fiable = poids plus faible)
SOURCE_WEIGHTS = {
    "stocktwits": 1.00,  # label fourni par l'auteur (gold)
    "reddit": 0.60,  # plus bruite
    "twitter": 0.70,  # intermediaire
}

# Seuils de confiance / alerte
SPIKE_ZSCORE_THRESHOLD = 2.0  # z >= 2 = spike significatif
HIGH_VOLUME_ABSOLUTE = 100  # >= 100 messages 24h = attention elevee

# Lexiques (reutilise la logique de transcript_analyzer + ajoute argot WSB)
_BULLISH_WORDS = {
    "bull",
    "bullish",
    "long",
    "moon",
    "rocket",
    "tendie",
    "tendies",
    "diamond",
    "hands",
    "hodl",
    "hold",
    "buy",
    "calls",
    "call",
    "yolo",
    "breakout",
    "rally",
    "surge",
    "pump",
    "gains",
    "winner",
    "winning",
    "strong",
    "beat",
    "beats",
    "beating",
    "record",
    "up",
    "upward",
    "positive",
    "bullrun",
    "squeeze",
    "explode",
    "parabolic",
}

_BEARISH_WORDS = {
    "bear",
    "bearish",
    "short",
    "puts",
    "put",
    "dump",
    "crash",
    "tank",
    "sell",
    "drop",
    "plunge",
    "collapse",
    "disaster",
    "loss",
    "losses",
    "red",
    "bleeding",
    "falling",
    "decline",
    "miss",
    "missed",
    "weak",
    "bagholder",
    "rugpull",
    "pump_and_dump",
    "fade",
    "dead",
    "overvalued",
    "bubble",
    "correction",
    "downtrend",
    "rekt",
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class SocialMessage:
    """Un message social normalise."""

    source: str  # "stocktwits" | "reddit" | "twitter"
    message_id: str
    author: str
    text: str
    posted_at: datetime  # UTC
    cashtags: List[str] = field(default_factory=list)
    url: str = ""
    author_age_days: Optional[float] = None  # age compte auteur si dispo
    explicit_label: Optional[str] = None  # "Bullish"/"Bearish" si fourni
    likes: int = 0
    replies: int = 0

    def key(self) -> str:
        return f"{self.source}:{self.message_id}"


@dataclass
class SocialSentimentReport:
    """Rapport agrege de sentiment social pour un ticker."""

    ticker: str
    window_start: datetime
    window_end: datetime

    messages_count: int = 0
    sources_used: List[str] = field(default_factory=list)

    # Distribution
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    # Scores
    sentiment_score: float = 0.0  # [-1, +1] pondere
    bullish_ratio: float = 0.0  # bullish / (bullish + bearish)

    # Volume
    baseline_avg_24h: float = 0.0
    volume_zscore: float = 0.0  # z-score vs baseline
    is_spike: bool = False

    # Qualite du signal
    bot_ratio: float = 0.0  # fraction de messages suspects
    unique_authors: int = 0
    top_keywords: List[str] = field(default_factory=list)
    top_messages: List[SocialMessage] = field(default_factory=list)

    # Meta
    error: str = ""

    def summary(self) -> str:
        return (
            f"ticker={self.ticker} n={self.messages_count} "
            f"sentiment={self.sentiment_score:+.2f} bull_ratio={self.bullish_ratio:.2f} "
            f"spike={self.is_spike} z={self.volume_zscore:+.2f} bot={self.bot_ratio:.2f}"
        )

    def to_prompt_block(self) -> str:
        lines = [
            "<social_sentiment>",
            f"  ticker={self.ticker}",
            f"  window_hours={(self.window_end - self.window_start).total_seconds() / 3600:.0f}",
            f"  messages={self.messages_count}",
            f"  sentiment_score={self.sentiment_score:+.3f}  (range [-1,+1])",
            f"  bullish_ratio={self.bullish_ratio:.2f}",
            f"  bullish_count={self.bullish_count} bearish_count={self.bearish_count}",
            f"  volume_zscore={self.volume_zscore:+.2f}  (spike={self.is_spike})",
            f"  unique_authors={self.unique_authors}",
            f"  bot_ratio={self.bot_ratio:.2f}",
            f"  sources={','.join(self.sources_used) or 'none'}",
        ]
        if self.top_keywords:
            lines.append(f"  top_keywords={self.top_keywords[:8]}")
        lines.append("</social_sentiment>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class SocialSentimentScorer:
    """Attribue un score [-1, +1] a un message social."""

    def __init__(self, use_vader: bool = True):
        self.use_vader = use_vader
        self._vader = None
        self._vader_attempted = False

    def _try_load_vader(self):
        if self._vader_attempted:
            return
        self._vader_attempted = True
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

            self._vader = SentimentIntensityAnalyzer()
            logger.info("[SocialSentiment] VADER charge")
        except ImportError:
            logger.debug("[SocialSentiment] VADER absent, fallback lexicon")

    def score(self, msg: SocialMessage) -> float:
        """Retourne un score [-1, +1] pour un message."""
        # 1. Label explicite (StockTwits Bullish/Bearish) — gold
        if msg.explicit_label:
            lab = msg.explicit_label.lower()
            if "bull" in lab:
                return 0.8
            if "bear" in lab:
                return -0.8

        # 2. VADER si dispo
        if self.use_vader:
            self._try_load_vader()
        if self._vader is not None:
            try:
                s = self._vader.polarity_scores(msg.text)
                base = float(s.get("compound", 0.0))
                # Ajuster avec lexique bull/bear financier
                fin = self._financial_lexicon_score(msg.text)
                return max(-1.0, min(1.0, 0.7 * base + 0.3 * fin))
            except Exception:
                pass

        return self._financial_lexicon_score(msg.text)

    def _financial_lexicon_score(self, text: str) -> float:
        """Fallback pur Python : lexique bull/bear."""
        if not text:
            return 0.0
        tokens = re.findall(r"[a-z_]+", text.lower())
        if not tokens:
            return 0.0
        n_bull = sum(1 for t in tokens if t in _BULLISH_WORDS)
        n_bear = sum(1 for t in tokens if t in _BEARISH_WORDS)
        if n_bull + n_bear == 0:
            return 0.0
        raw = (n_bull - n_bear) / (n_bull + n_bear)
        # Saturate a [-1, 1] et multiplie par intensite
        intensity = min(1.0, (n_bull + n_bear) / max(1, len(tokens)) * 10.0)
        return raw * intensity


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------


class StockTwitsFetcher:
    """Fetcher StockTwits (API publique, pas de cle)."""

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def fetch(self, ticker: str, max_messages: int = 200) -> List[SocialMessage]:
        try:
            import requests
        except ImportError:
            logger.warning("[StockTwits] requests absent, skip")
            return []

        url = STOCKTWITS_URL.format(ticker=ticker.upper())
        try:
            resp = requests.get(
                url,
                timeout=self.timeout,
                headers={
                    "User-Agent": "Locus-MultiAgent-Research/1.0",
                },
            )
            if resp.status_code != 200:
                logger.debug("[StockTwits] HTTP %d pour %s", resp.status_code, ticker)
                return []
            data = resp.json()
        except Exception as e:
            logger.warning("[StockTwits] fetch echec : %s", e)
            return []

        messages: List[SocialMessage] = []
        for item in data.get("messages", [])[:max_messages]:
            try:
                body = str(item.get("body", ""))
                cashtags = [s["symbol"] for s in item.get("symbols", []) if isinstance(s, dict) and "symbol" in s]
                created_at = item.get("created_at", "")
                try:
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                except Exception:
                    ts = datetime.now(timezone.utc)
                user = item.get("user", {}) or {}
                entities = item.get("entities", {}) or {}
                sentiment = entities.get("sentiment") or {}
                label = sentiment.get("basic") if isinstance(sentiment, dict) else None

                msg = SocialMessage(
                    source="stocktwits",
                    message_id=str(item.get("id", "")),
                    author=str(user.get("username", "unknown")),
                    text=body,
                    posted_at=ts,
                    cashtags=cashtags,
                    url=f"https://stocktwits.com/message/{item.get('id', '')}",
                    explicit_label=label,
                    likes=int(item.get("likes", {}).get("total", 0)) if isinstance(item.get("likes"), dict) else 0,
                )
                messages.append(msg)
            except Exception:
                continue
        logger.info("[StockTwits] %d messages pour %s", len(messages), ticker)
        return messages


class RedditFetcher:
    """Fetcher Reddit via endpoint public JSON (pas de cle)."""

    def __init__(
        self,
        subreddits: Sequence[str] = DEFAULT_SUBREDDITS,
        timeout: float = 10.0,
    ):
        self.subreddits = list(subreddits)
        self.timeout = timeout

    def fetch(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        max_messages: int = 100,
    ) -> List[SocialMessage]:
        try:
            import requests
        except ImportError:
            logger.warning("[Reddit] requests absent, skip")
            return []

        # On cherche le cashtag et le nom de l'entreprise
        queries = [f"${ticker.upper()}"]
        if company_name:
            queries.append(f'"{company_name}"')

        messages: List[SocialMessage] = []
        seen_ids = set()

        for sub in self.subreddits:
            for q in queries:
                if len(messages) >= max_messages:
                    break
                url = REDDIT_SEARCH_URL.format(
                    subreddit=sub,
                    query=q.replace(" ", "+").replace('"', "%22"),
                )
                try:
                    resp = requests.get(
                        url,
                        timeout=self.timeout,
                        headers={
                            "User-Agent": REDDIT_USER_AGENT,
                        },
                    )
                    if resp.status_code != 200:
                        logger.debug("[Reddit] HTTP %d sub=%s q=%s", resp.status_code, sub, q)
                        continue
                    data = resp.json()
                except Exception as e:
                    logger.debug("[Reddit] fetch echec sub=%s : %s", sub, e)
                    continue

                children = data.get("data", {}).get("children", [])
                for c in children:
                    d = c.get("data", {})
                    mid = d.get("id")
                    if not mid or mid in seen_ids:
                        continue
                    seen_ids.add(mid)

                    text = " ".join(
                        filter(
                            None,
                            [
                                d.get("title", ""),
                                d.get("selftext", ""),
                            ],
                        )
                    )[:3000]
                    cashtags = list(set(CASHTAG_RE.findall(text)))
                    try:
                        ts = datetime.fromtimestamp(d.get("created_utc", 0), tz=timezone.utc)
                    except Exception:
                        ts = datetime.now(timezone.utc)

                    # Filtre soft : le texte doit mentionner le ticker
                    if ticker.upper() not in text.upper():
                        if not company_name or company_name.lower() not in text.lower():
                            continue

                    msg = SocialMessage(
                        source="reddit",
                        message_id=str(mid),
                        author=str(d.get("author", "unknown")),
                        text=text,
                        posted_at=ts,
                        cashtags=cashtags,
                        url="https://reddit.com" + d.get("permalink", ""),
                        likes=int(d.get("score", 0)),
                        replies=int(d.get("num_comments", 0)),
                    )
                    messages.append(msg)
                    if len(messages) >= max_messages:
                        break

                # Fair-use Reddit : petite pause
                time.sleep(1.0)

            if len(messages) >= max_messages:
                break

        logger.info("[Reddit] %d messages pour %s", len(messages), ticker)
        return messages


# ---------------------------------------------------------------------------
# Bot / spam detector
# ---------------------------------------------------------------------------


def estimate_bot_ratio(messages: Sequence[SocialMessage]) -> float:
    """
    Estime la fraction de messages suspects. Heuristiques :
      - Duplicate content : memes textes par des auteurs differents
      - Tres court (<3 mots)
      - Full caps + spam keywords
      - Mentions trop nombreuses de cashtags (> 8)

    Pas une vraie detection bot mais un floor protection.
    """
    if not messages:
        return 0.0

    n = len(messages)
    suspect = 0
    text_counts: Dict[str, int] = {}

    for m in messages:
        t = m.text.strip()
        tl = t.lower()
        text_counts[tl] = text_counts.get(tl, 0) + 1

    for m in messages:
        t = m.text.strip()
        if len(t.split()) < 3:
            suspect += 1
            continue
        if len(m.cashtags) > 8:
            suspect += 1
            continue
        if text_counts.get(t.lower(), 0) > 1:
            suspect += 1
            continue
        # Full caps + ponctuation excessive
        if t.upper() == t and len(t) > 20 and t.count("!") > 2:
            suspect += 1
            continue

    return suspect / n


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


class SocialSentimentAggregator:
    """
    Agrege plusieurs sources en un rapport unique.

    Usage :
        agg = SocialSentimentAggregator()
        report = agg.analyze(
            ticker="AAPL",
            company_name="Apple",
            baseline_24h_avg=50.0,  # calcule ailleurs a partir du historique
        )
    """

    def __init__(
        self,
        stocktwits: Optional[StockTwitsFetcher] = None,
        reddit: Optional[RedditFetcher] = None,
        scorer: Optional[SocialSentimentScorer] = None,
        window_hours: int = 24,
    ):
        self.stocktwits = stocktwits if stocktwits is not None else StockTwitsFetcher()
        self.reddit = reddit if reddit is not None else RedditFetcher()
        self.scorer = scorer if scorer is not None else SocialSentimentScorer()
        self.window_hours = window_hours

    def analyze(
        self,
        ticker: str,
        company_name: Optional[str] = None,
        baseline_24h_avg: float = 0.0,
        sources: Sequence[str] = ("stocktwits", "reddit"),
        messages: Optional[Sequence[SocialMessage]] = None,
    ) -> SocialSentimentReport:
        """Pipeline complet. `messages` override le fetch (utile pour tests)."""
        window_end = datetime.now(timezone.utc)
        window_start = window_end - timedelta(hours=self.window_hours)

        report = SocialSentimentReport(
            ticker=ticker.upper(),
            window_start=window_start,
            window_end=window_end,
        )

        all_msgs: List[SocialMessage] = []
        if messages is not None:
            all_msgs = list(messages)
            report.sources_used = list({m.source for m in messages})
        else:
            if "stocktwits" in sources:
                try:
                    st = self.stocktwits.fetch(ticker)
                    all_msgs.extend(st)
                    if st:
                        report.sources_used.append("stocktwits")
                except Exception as e:
                    logger.warning("[SocialSentiment] StockTwits echec : %s", e)
            if "reddit" in sources:
                try:
                    rd = self.reddit.fetch(ticker, company_name=company_name)
                    all_msgs.extend(rd)
                    if rd:
                        report.sources_used.append("reddit")
                except Exception as e:
                    logger.warning("[SocialSentiment] Reddit echec : %s", e)

        # Filtre fenetre temporelle
        in_window = [m for m in all_msgs if m.posted_at >= window_start]
        report.messages_count = len(in_window)
        report.unique_authors = len({m.author for m in in_window})

        if not in_window:
            return report

        # -- Scoring --
        weighted_sum = 0.0
        weight_total = 0.0
        bullish = bearish = neutral = 0
        keyword_counts: Dict[str, int] = {}

        for m in in_window:
            s = self.scorer.score(m)
            w = SOURCE_WEIGHTS.get(m.source, 0.5)
            # Poids additionnel : engagement (likes/replies) cap a 2x
            engagement_mult = 1.0 + min(1.0, (m.likes + m.replies) / 50.0)
            effective_w = w * engagement_mult
            weighted_sum += s * effective_w
            weight_total += effective_w

            if s > 0.15:
                bullish += 1
            elif s < -0.15:
                bearish += 1
            else:
                neutral += 1

            # Keyword tracking
            for tok in re.findall(r"[a-z_\$]+", m.text.lower()):
                if tok in _BULLISH_WORDS or tok in _BEARISH_WORDS:
                    keyword_counts[tok] = keyword_counts.get(tok, 0) + 1

        report.bullish_count = bullish
        report.bearish_count = bearish
        report.neutral_count = neutral
        report.sentiment_score = weighted_sum / weight_total if weight_total > 0 else 0.0
        report.bullish_ratio = bullish / (bullish + bearish) if (bullish + bearish) > 0 else 0.5
        report.bot_ratio = estimate_bot_ratio(in_window)

        # -- Volume z-score --
        if baseline_24h_avg > 0:
            # Approximation : sigma estime a sqrt(baseline) (Poisson-like)
            sigma = max(math.sqrt(baseline_24h_avg), 1.0)
            report.baseline_avg_24h = baseline_24h_avg
            report.volume_zscore = (report.messages_count - baseline_24h_avg) / sigma
            report.is_spike = (
                report.volume_zscore >= SPIKE_ZSCORE_THRESHOLD or report.messages_count >= HIGH_VOLUME_ABSOLUTE
            )
        else:
            report.is_spike = report.messages_count >= HIGH_VOLUME_ABSOLUTE

        # -- Top keywords --
        report.top_keywords = [w for w, _ in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]]

        # -- Top messages (par likes) --
        report.top_messages = sorted(
            in_window,
            key=lambda m: m.likes + m.replies,
            reverse=True,
        )[:5]

        return report


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Scenario offline : on fabrique des messages
    now = datetime.now(timezone.utc)

    mock_msgs = [
        SocialMessage(
            source="stocktwits",
            message_id="1",
            author="a1",
            text="$AAPL to the moon! Diamond hands bulls win again, record quarter incoming",
            posted_at=now - timedelta(hours=2),
            cashtags=["AAPL"],
            explicit_label="Bullish",
            likes=42,
        ),
        SocialMessage(
            source="stocktwits",
            message_id="2",
            author="a2",
            text="$AAPL overvalued bubble, puts are free money",
            posted_at=now - timedelta(hours=3),
            cashtags=["AAPL"],
            explicit_label="Bearish",
            likes=5,
        ),
        SocialMessage(
            source="reddit",
            message_id="3",
            author="a3",
            text="Why $AAPL is a strong buy: Q3 revenue beat, services margin record, bullish",
            posted_at=now - timedelta(hours=4),
            cashtags=["AAPL"],
            likes=120,
            replies=35,
        ),
        SocialMessage(
            source="reddit",
            message_id="4",
            author="a4",
            text="$AAPL bagholders in shambles, this stock is dead, puts to $150",
            posted_at=now - timedelta(hours=6),
            cashtags=["AAPL"],
            likes=15,
        ),
        SocialMessage(
            source="stocktwits",
            message_id="5",
            author="bot1",
            text="$AAPL $AAPL $AAPL $AAPL $AAPL $AAPL $AAPL $AAPL $AAPL $AAPL",
            posted_at=now - timedelta(hours=1),
            cashtags=["AAPL"] * 10,
        ),
    ]

    agg = SocialSentimentAggregator()
    report = agg.analyze(
        ticker="AAPL",
        company_name="Apple",
        baseline_24h_avg=2.0,  # baseline tres basse => spike
        messages=mock_msgs,
    )

    logger.info("=== Social Sentiment Report ===")
    print(report.summary())
    logger.info(f"\nBullish: {report.bullish_count}, Bearish: {report.bearish_count}, Neutral: {report.neutral_count}")
    logger.info(f"Bot ratio: {report.bot_ratio:.2f}")
    logger.info(f"Top keywords: {report.top_keywords}")
    logger.info(f"Unique authors: {report.unique_authors}")
    logger.info(f"\n{report.to_prompt_block()}")

    # Validation
    assert report.messages_count == 5
    assert report.bullish_count >= 2
    assert report.bearish_count >= 1
    assert report.is_spike, "Baseline 2 avg vs 5 messages should trigger spike"
    assert report.bot_ratio > 0.0, "Bot message should be detected"
    logger.info("\nOK - tous les signaux agreges correctement")

    # Test cashtag extraction
    logger.info("\n=== Cashtag Extraction ===")
    samples = [
        "Buy $AAPL and $MSFT today",
        "I love TSLA no cashtag",
        "$SPY $QQQ $DIA",
    ]
    for s in samples:
        tags = CASHTAG_RE.findall(s)
        print(f"  '{s[:40]}' -> {tags}")

    # Scorer tests
    logger.info("\n=== Scorer Tests ===")
    scorer = SocialSentimentScorer(use_vader=False)
    tests = [
        ("AAPL moon rocket diamond hands", ">0"),
        ("AAPL crash dump puts bearish", "<0"),
        ("AAPL stock analysis", "~0"),
    ]
    for text, expected in tests:
        m = SocialMessage(
            source="reddit",
            message_id="x",
            author="x",
            text=text,
            posted_at=now,
            cashtags=[],
        )
        s = scorer.score(m)
        print(f"  '{text}' -> {s:+.3f} (expected {expected})")
