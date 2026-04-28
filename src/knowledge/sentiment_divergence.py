"""
sentiment_divergence.py — Detecteur de divergence news vs social.

PROBLEME
--------
Le sentiment news (analystes pro, journalistes) et le sentiment social (retail
WSB, StockTwits) sont deux SIGNAUX DIFFERENTS :
  - News  : consomme par institutionnels, integre rapidement aux prix.
  - Social: consomme par retail, cree du momentum non-fondamental.

Quand les DEUX s'accordent : signal robuste mais deja price-in.
Quand ils DIVERGENT : signal asymetrique, souvent rentable.

Cas d'usage classiques :
  1. News POSITIF + Social NEGATIF → "smart money buying, retail capitulating".
     Souvent un buy signal contrarien a court terme.
  2. News NEGATIF + Social TRES POSITIF → "meme-stock pump deconnecte des
     fondamentaux". Souvent un short signal a 1-3 semaines.
  3. News NEUTRAL + Social SPIKE → "attention retail anormale sans fondement".
     Signal de volatilite a venir.
  4. Volume social >> historique + sentiment uniforme → "echo chamber",
     risque de retournement brutal.

REFERENCES
----------
- Tetlock, P. C. (2007). "Giving Content to Investor Sentiment: The Role of
  Media in the Stock Market." Journal of Finance.
- Antweiler, W. & Frank, M. Z. (2004). "Is All That Talk Just Noise? The
  Information Content of Internet Stock Message Boards." J. of Finance.
- Da, Z., Engelberg, J., & Gao, P. (2015). "The sum of all FEARS: Investor
  sentiment and asset prices." Review of Financial Studies.
- Bartov, E., Faurel, L., & Mohanram, P. S. (2018). "Can Twitter Help Predict
  Firm-Level Earnings and Stock Returns?" The Accounting Review.
- Cookson, J. A. & Niessner, M. (2020). "Why Don't We Agree? Evidence from a
  Social Network of Investors." Journal of Finance.

MODELE FORMEL
-------------
On note :
  N  : sentiment news    in [-1, +1]
  S  : sentiment social  in [-1, +1]
  Vn : volume news      (article count)
  Vs : volume social    (message count)
  Z  : z-score volume social vs baseline
  Bot: bot ratio social  in [0, 1]

Divergence brute :
  D = N - S                       in [-2, +2]

Divergence ponderee par confiance :
  conf_news   = clip(log(1 + Vn) / log(1 + 50), 0, 1)
  conf_social = (1 - Bot) * clip(log(1 + Vs) / log(1 + 200), 0, 1)
  D_weighted  = D * (conf_news * conf_social) ** 0.5

Regimes (pattern detection) :
  - "smart_buy"        : N >= +0.3 et S <= -0.2  → contrarien long
  - "smart_sell"       : N <= -0.3 et S >= +0.2  → contrarien short
  - "retail_pump"      : S >= +0.5 et Z >= 2 et Bot >= 0.3
                         et N <= 0                → meme-stock alarme
  - "echo_chamber"     : S in [+0.6, +1.0] et Z >= 2
                         et bullish_ratio >= 0.85 → upside fragile
  - "panic_capitulation": S <= -0.5 et Z >= 2     → bottom fishing
  - "consensus_strong" : |N - S| <= 0.2 et abs((N+S)/2) >= 0.4
                                                  → tendance robuste
  - "neutral_quiet"    : tous proches de 0       → no signal

USAGE
-----
    from sentiment_divergence import DivergenceAnalyzer
    da = DivergenceAnalyzer()
    report = da.analyze(news_sentiment=+0.4, news_volume=12,
                        social_report=social_report)
    print(report.regime, report.confidence, report.suggested_bias)
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Seuils de classification de regime
SMART_MONEY_NEWS_THRESHOLD = 0.30  # |N| pour qualifier smart-money tilt
SMART_MONEY_SOCIAL_OPPOSITE = 0.20  # |S| dans direction opposee
RETAIL_PUMP_SOCIAL_THRESHOLD = 0.50  # S >= ce seuil pour pump
ECHO_CHAMBER_BULLISH_RATIO = 0.85  # bullish_ratio extreme
ECHO_CHAMBER_SOCIAL_MIN = 0.60  # S minimum
PANIC_SOCIAL_THRESHOLD = -0.50  # S <= ce seuil pour panic
CONSENSUS_DELTA = 0.20  # |N - S| max pour consensus
CONSENSUS_AVG_MIN = 0.40  # |moyenne| minimum pour consensus
NEUTRAL_QUIET_MAX = 0.15  # tout sous ce seuil = pas de signal

# Z-score volume social pour spike
SPIKE_Z_THRESHOLD = 2.0

# Bot ratio au-dela duquel on suspecte manipulation
HIGH_BOT_RATIO = 0.30

# Regimes (label canonique)
REGIME_SMART_BUY = "smart_buy"
REGIME_SMART_SELL = "smart_sell"
REGIME_RETAIL_PUMP = "retail_pump"
REGIME_ECHO_CHAMBER = "echo_chamber"
REGIME_PANIC = "panic_capitulation"
REGIME_CONSENSUS_BULL = "consensus_bullish"
REGIME_CONSENSUS_BEAR = "consensus_bearish"
REGIME_QUIET = "neutral_quiet"
REGIME_MIXED = "mixed_signals"

# Mapping regime -> bias suggere {-1,-0.5,0,+0.5,+1}
REGIME_BIAS: Dict[str, float] = {
    REGIME_SMART_BUY: +0.7,  # contrarien long
    REGIME_SMART_SELL: -0.7,  # contrarien short
    REGIME_RETAIL_PUMP: -0.5,  # short le pump
    REGIME_ECHO_CHAMBER: -0.3,  # tilt short modere
    REGIME_PANIC: +0.4,  # bottom fishing prudent
    REGIME_CONSENSUS_BULL: +0.5,  # tendance robuste mais price-in
    REGIME_CONSENSUS_BEAR: -0.5,
    REGIME_QUIET: 0.0,
    REGIME_MIXED: 0.0,
}

# Description des regimes (pour prompt LLM / dashboard)
REGIME_DESCRIPTIONS: Dict[str, str] = {
    REGIME_SMART_BUY: (
        "Smart money positif, retail negatif. Contrarien LONG : les "
        "institutionnels achetent pendant que le retail capitule."
    ),
    REGIME_SMART_SELL: (
        "Smart money negatif, retail positif. Contrarien SHORT : les "
        "institutionnels vendent pendant que le retail euphorique."
    ),
    REGIME_RETAIL_PUMP: ("Pump retail social avec spike de volume + bot detecte. Risque eleve de retournement brutal."),
    REGIME_ECHO_CHAMBER: (
        "Sentiment social tres uniforme (>85% bullish) avec spike volume. "
        "Echo chamber, fragile a la moindre nouvelle negative."
    ),
    REGIME_PANIC: (
        "Capitulation retail (sentiment tres negatif + spike de volume). "
        "Souvent un point bas a court terme — bottom fishing prudent."
    ),
    REGIME_CONSENSUS_BULL: (
        "News et social alignes positifs. Tendance forte mais probablement "
        "deja price-in. Suivre uniquement si momentum confirme."
    ),
    REGIME_CONSENSUS_BEAR: ("News et social alignes negatifs. Confirmation de tendance baissiere."),
    REGIME_QUIET: ("Aucun signal franc. Eviter de prendre position sur ce ticker."),
    REGIME_MIXED: ("Signaux contradictoires sans pattern dominant. Attendre clarification."),
}


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class DivergenceReport:
    """Rapport de divergence news vs social."""

    ticker: str
    timestamp: datetime

    # Signaux bruts
    news_sentiment: float
    news_volume: int
    social_sentiment: float
    social_volume: int
    social_bullish_ratio: float
    social_zscore: float
    social_bot_ratio: float
    social_is_spike: bool

    # Calculs derives
    divergence: float  # N - S, [-2, +2]
    divergence_weighted: float  # divergence * sqrt(conf_n * conf_s)
    confidence_news: float  # [0, 1]
    confidence_social: float  # [0, 1]
    confidence_overall: float  # [0, 1]

    # Classification
    regime: str = REGIME_QUIET
    regime_score: float = 0.0  # force de la signature dans [0, 1]
    suggested_bias: float = 0.0  # [-1, +1] suggestion long/short
    description: str = ""

    # Detail / debug
    triggered_rules: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"ticker={self.ticker} regime={self.regime} "
            f"D={self.divergence:+.2f} D_w={self.divergence_weighted:+.2f} "
            f"bias={self.suggested_bias:+.2f} conf={self.confidence_overall:.2f}"
        )

    def to_prompt_block(self) -> str:
        lines = [
            "<sentiment_divergence>",
            f"  ticker={self.ticker}",
            f"  news_sentiment={self.news_sentiment:+.3f} (n={self.news_volume})",
            f"  social_sentiment={self.social_sentiment:+.3f} (n={self.social_volume})",
            f"  social_bullish_ratio={self.social_bullish_ratio:.2f}",
            f"  social_volume_zscore={self.social_zscore:+.2f}  spike={self.social_is_spike}",
            f"  social_bot_ratio={self.social_bot_ratio:.2f}",
            f"  divergence={self.divergence:+.3f}  weighted={self.divergence_weighted:+.3f}",
            f"  confidence={self.confidence_overall:.2f}  (news={self.confidence_news:.2f}, social={self.confidence_social:.2f})",
            f"  regime={self.regime}",
            f"  suggested_bias={self.suggested_bias:+.2f}",
            f"  description={self.description}",
        ]
        if self.triggered_rules:
            lines.append(f"  rules={self.triggered_rules}")
        lines.append("</sentiment_divergence>")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyseur
# ---------------------------------------------------------------------------


class DivergenceAnalyzer:
    """
    Analyseur de divergence news / social.

    L'analyseur est PUR : il prend des scores deja calcules et produit un
    rapport. Ne fait pas de fetch lui-meme.

    Usage avec social_sentiment.SocialSentimentReport :

        from src.knowledge.social_sentiment import SocialSentimentAggregator
        from src.knowledge.sentiment_divergence import DivergenceAnalyzer

        social_rep = SocialSentimentAggregator().analyze("AAPL")
        # news_sentiment vient du pipeline ABSA agrege
        div = DivergenceAnalyzer().analyze(
            ticker="AAPL",
            news_sentiment=+0.4,
            news_volume=15,
            social_report=social_rep,
        )
    """

    def __init__(
        self,
        news_volume_saturation: int = 50,
        social_volume_saturation: int = 200,
        smart_money_news_thresh: float = SMART_MONEY_NEWS_THRESHOLD,
        retail_pump_social_thresh: float = RETAIL_PUMP_SOCIAL_THRESHOLD,
        consensus_delta: float = CONSENSUS_DELTA,
    ):
        self.news_volume_saturation = news_volume_saturation
        self.social_volume_saturation = social_volume_saturation
        self.smart_money_news_thresh = smart_money_news_thresh
        self.retail_pump_social_thresh = retail_pump_social_thresh
        self.consensus_delta = consensus_delta

    # ----- API principale -----

    def analyze(
        self,
        ticker: str,
        news_sentiment: float,
        news_volume: int,
        # Soit on passe un SocialSentimentReport, soit les valeurs brutes
        social_report: Optional[Any] = None,
        social_sentiment: Optional[float] = None,
        social_volume: Optional[int] = None,
        social_bullish_ratio: Optional[float] = None,
        social_zscore: Optional[float] = None,
        social_bot_ratio: Optional[float] = None,
        social_is_spike: Optional[bool] = None,
    ) -> DivergenceReport:
        # Normalisation des inputs
        N = float(max(-1.0, min(1.0, news_sentiment)))
        Vn = int(max(0, news_volume))

        if social_report is not None:
            S = float(max(-1.0, min(1.0, getattr(social_report, "sentiment_score", 0.0))))
            Vs = int(max(0, getattr(social_report, "messages_count", 0)))
            BR = float(max(0.0, min(1.0, getattr(social_report, "bullish_ratio", 0.5))))
            Z = float(getattr(social_report, "volume_zscore", 0.0))
            Bot = float(max(0.0, min(1.0, getattr(social_report, "bot_ratio", 0.0))))
            Spike = bool(getattr(social_report, "is_spike", False))
        else:
            if any(
                v is None
                for v in [
                    social_sentiment,
                    social_volume,
                    social_bullish_ratio,
                    social_zscore,
                    social_bot_ratio,
                    social_is_spike,
                ]
            ):
                raise ValueError("Soit social_report soit toutes les valeurs explicites doivent etre fournies")
            S = float(max(-1.0, min(1.0, social_sentiment)))  # type: ignore[arg-type]
            Vs = int(max(0, social_volume))  # type: ignore[arg-type]
            BR = float(max(0.0, min(1.0, social_bullish_ratio)))  # type: ignore[arg-type]
            Z = float(social_zscore)  # type: ignore[arg-type]
            Bot = float(max(0.0, min(1.0, social_bot_ratio)))  # type: ignore[arg-type]
            Spike = bool(social_is_spike)  # type: ignore[arg-type]

        # Confiance par source
        conf_n = self._confidence_from_volume(Vn, self.news_volume_saturation)
        conf_s_raw = self._confidence_from_volume(Vs, self.social_volume_saturation)
        # Penalite bot : (1 - bot)^2 — penalisation forte
        conf_s = conf_s_raw * (1.0 - Bot) ** 2
        conf_overall = math.sqrt(max(0.0, conf_n) * max(0.0, conf_s))

        # Divergence
        D = N - S
        D_w = D * conf_overall

        report = DivergenceReport(
            ticker=ticker.upper(),
            timestamp=datetime.now(timezone.utc),
            news_sentiment=N,
            news_volume=Vn,
            social_sentiment=S,
            social_volume=Vs,
            social_bullish_ratio=BR,
            social_zscore=Z,
            social_bot_ratio=Bot,
            social_is_spike=Spike,
            divergence=D,
            divergence_weighted=D_w,
            confidence_news=conf_n,
            confidence_social=conf_s,
            confidence_overall=conf_overall,
        )

        self._classify(report)
        return report

    # ----- Helpers internes -----

    @staticmethod
    def _confidence_from_volume(v: int, sat: int) -> float:
        """Confiance log-saturee : 0 a v=0, ~0.5 a v=sat/3, 1.0 a v>=sat."""
        if v <= 0:
            return 0.0
        if sat <= 1:
            return 1.0
        c = math.log1p(v) / math.log1p(sat)
        return float(max(0.0, min(1.0, c)))

    def _classify(self, r: DivergenceReport) -> None:
        """
        Pattern matching ordonne par specificite :
          1. retail_pump (haute prio — alerte la plus actionnable)
          2. smart_buy / smart_sell (contrarien)
          3. echo_chamber (long fragile)
          4. panic_capitulation (bottom fishing)
          5. consensus_bull / consensus_bear
          6. quiet
          7. mixed (defaut)

        On accumule "triggered_rules" pour audit, mais le regime final est
        determine par le premier pattern qui matche.
        """
        N, S = r.news_sentiment, r.social_sentiment
        rules = r.triggered_rules

        # 1. Retail pump (priorite max)
        if (
            S >= self.retail_pump_social_thresh
            and r.social_is_spike
            and r.social_bot_ratio >= HIGH_BOT_RATIO
            and N <= 0.0
        ):
            rules.append("retail_pump_pattern")
            r.regime = REGIME_RETAIL_PUMP
            # Force = combinaison de la magnitude et du signal bot/spike
            r.regime_score = min(1.0, 0.5 * S + 0.3 * (r.social_bot_ratio) + 0.2 * (min(r.social_zscore, 5.0) / 5.0))
            r.suggested_bias = REGIME_BIAS[REGIME_RETAIL_PUMP]
            r.description = REGIME_DESCRIPTIONS[REGIME_RETAIL_PUMP]
            return

        # 2. Smart-money divergence (contrarien)
        if N >= self.smart_money_news_thresh and S <= -SMART_MONEY_SOCIAL_OPPOSITE:
            rules.append("smart_buy_pattern")
            r.regime = REGIME_SMART_BUY
            r.regime_score = min(1.0, ((N - (-S)) / 2.0))
            r.suggested_bias = REGIME_BIAS[REGIME_SMART_BUY] * r.confidence_overall
            r.description = REGIME_DESCRIPTIONS[REGIME_SMART_BUY]
            return

        if N <= -self.smart_money_news_thresh and S >= SMART_MONEY_SOCIAL_OPPOSITE:
            rules.append("smart_sell_pattern")
            r.regime = REGIME_SMART_SELL
            r.regime_score = min(1.0, ((-N + S) / 2.0))
            r.suggested_bias = REGIME_BIAS[REGIME_SMART_SELL] * r.confidence_overall
            r.description = REGIME_DESCRIPTIONS[REGIME_SMART_SELL]
            return

        # 3. Echo chamber (extreme bullish ratio + spike)
        if r.social_bullish_ratio >= ECHO_CHAMBER_BULLISH_RATIO and S >= ECHO_CHAMBER_SOCIAL_MIN and r.social_is_spike:
            rules.append("echo_chamber_pattern")
            r.regime = REGIME_ECHO_CHAMBER
            r.regime_score = min(1.0, 0.5 * r.social_bullish_ratio + 0.3 * S + 0.2 * (min(r.social_zscore, 5.0) / 5.0))
            r.suggested_bias = REGIME_BIAS[REGIME_ECHO_CHAMBER]
            r.description = REGIME_DESCRIPTIONS[REGIME_ECHO_CHAMBER]
            return

        # 4. Panic (capitulation)
        if S <= PANIC_SOCIAL_THRESHOLD and r.social_is_spike:
            rules.append("panic_capitulation_pattern")
            r.regime = REGIME_PANIC
            r.regime_score = min(1.0, 0.6 * abs(S) + 0.4 * (min(r.social_zscore, 5.0) / 5.0))
            # Bias positif uniquement si news pas franchement negatif
            bias = REGIME_BIAS[REGIME_PANIC]
            if N <= -0.3:
                bias *= 0.4  # news baissier confirme la capitulation, ne pas acheter
                r.notes.append("news_negatif_confirme_capitulation")
            r.suggested_bias = bias * r.confidence_overall
            r.description = REGIME_DESCRIPTIONS[REGIME_PANIC]
            return

        # 5. Consensus
        avg = (N + S) / 2.0
        if abs(N - S) <= self.consensus_delta and abs(avg) >= CONSENSUS_AVG_MIN:
            if avg > 0:
                rules.append("consensus_bullish_pattern")
                r.regime = REGIME_CONSENSUS_BULL
                r.suggested_bias = REGIME_BIAS[REGIME_CONSENSUS_BULL] * r.confidence_overall
                r.description = REGIME_DESCRIPTIONS[REGIME_CONSENSUS_BULL]
            else:
                rules.append("consensus_bearish_pattern")
                r.regime = REGIME_CONSENSUS_BEAR
                r.suggested_bias = REGIME_BIAS[REGIME_CONSENSUS_BEAR] * r.confidence_overall
                r.description = REGIME_DESCRIPTIONS[REGIME_CONSENSUS_BEAR]
            r.regime_score = min(1.0, abs(avg))
            return

        # 6. Quiet
        if abs(N) <= NEUTRAL_QUIET_MAX and abs(S) <= NEUTRAL_QUIET_MAX:
            rules.append("quiet_pattern")
            r.regime = REGIME_QUIET
            r.regime_score = 1.0 - max(abs(N), abs(S)) / max(NEUTRAL_QUIET_MAX, 1e-6)
            r.suggested_bias = 0.0
            r.description = REGIME_DESCRIPTIONS[REGIME_QUIET]
            return

        # 7. Defaut : mixed
        r.regime = REGIME_MIXED
        r.regime_score = 1.0 - r.confidence_overall  # plus confiance basse, plus mixed
        # Bias = une fraction de la divergence ponderee
        r.suggested_bias = max(-0.3, min(0.3, r.divergence_weighted * 0.3))
        r.description = REGIME_DESCRIPTIONS[REGIME_MIXED]
        rules.append("mixed_default")


# ---------------------------------------------------------------------------
# Helper : conversion pour PositionSizer
# ---------------------------------------------------------------------------


def divergence_to_confidence_multiplier(report: DivergenceReport) -> float:
    """
    Convertit un DivergenceReport en multiplicateur de confiance pour le
    position sizing.

    Heuristique :
      - smart_buy / smart_sell  → boost 1.10-1.20 (signal contrarien fort)
      - retail_pump             → reduction 0.6 (court le pump = haute incert.)
      - echo_chamber            → reduction 0.8
      - panic_capitulation      → reduction 0.85 (volatil)
      - consensus_*             → leger boost 1.05 (tendance robuste)
      - quiet / mixed           → 1.0 (neutre)

    Le multiplicateur reste dans [0.5, 1.25] pour eviter de saturer Kelly.
    """
    base = 1.0
    if report.regime in (REGIME_SMART_BUY, REGIME_SMART_SELL):
        base = 1.10 + 0.10 * report.regime_score
    elif report.regime == REGIME_RETAIL_PUMP:
        base = 0.60
    elif report.regime == REGIME_ECHO_CHAMBER:
        base = 0.80
    elif report.regime == REGIME_PANIC:
        base = 0.85
    elif report.regime in (REGIME_CONSENSUS_BULL, REGIME_CONSENSUS_BEAR):
        base = 1.05
    # Pondere par la confiance globale (donnees fiables = effet plus marque)
    delta = (base - 1.0) * report.confidence_overall
    return float(max(0.5, min(1.25, 1.0 + delta)))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pas d'import de SocialSentimentReport pour rester independant en smoke
    da = DivergenceAnalyzer()

    # Scenario 1 : smart money buying
    r1 = da.analyze(
        ticker="AAPL",
        news_sentiment=+0.55,
        news_volume=18,
        social_sentiment=-0.35,
        social_volume=80,
        social_bullish_ratio=0.30,
        social_zscore=+0.5,
        social_bot_ratio=0.10,
        social_is_spike=False,
    )
    logger.info("--- SC1 smart_buy ---")
    print(r1.summary())
    print(r1.to_prompt_block())

    # Scenario 2 : retail pump (meme stock)
    r2 = da.analyze(
        ticker="GME",
        news_sentiment=-0.10,
        news_volume=8,
        social_sentiment=+0.85,
        social_volume=350,
        social_bullish_ratio=0.92,
        social_zscore=+4.5,
        social_bot_ratio=0.45,
        social_is_spike=True,
    )
    logger.info("\n--- SC2 retail_pump ---")
    print(r2.summary())
    print(r2.to_prompt_block())
    logger.info(f"  confidence_multiplier = {divergence_to_confidence_multiplier(r2):.3f}")

    # Scenario 3 : echo chamber
    r3 = da.analyze(
        ticker="NVDA",
        news_sentiment=+0.15,
        news_volume=20,
        social_sentiment=+0.78,
        social_volume=400,
        social_bullish_ratio=0.91,
        social_zscore=+3.2,
        social_bot_ratio=0.15,
        social_is_spike=True,
    )
    logger.info("\n--- SC3 echo_chamber ---")
    print(r3.summary())

    # Scenario 4 : panic capitulation
    r4 = da.analyze(
        ticker="TSLA",
        news_sentiment=-0.10,
        news_volume=12,
        social_sentiment=-0.65,
        social_volume=300,
        social_bullish_ratio=0.18,
        social_zscore=+3.0,
        social_bot_ratio=0.10,
        social_is_spike=True,
    )
    logger.info("\n--- SC4 panic_capitulation ---")
    print(r4.summary())

    # Scenario 5 : consensus bullish
    r5 = da.analyze(
        ticker="MSFT",
        news_sentiment=+0.50,
        news_volume=25,
        social_sentiment=+0.45,
        social_volume=150,
        social_bullish_ratio=0.70,
        social_zscore=+0.8,
        social_bot_ratio=0.05,
        social_is_spike=False,
    )
    logger.info("\n--- SC5 consensus_bull ---")
    print(r5.summary())

    # Scenario 6 : quiet
    r6 = da.analyze(
        ticker="JNJ",
        news_sentiment=+0.05,
        news_volume=3,
        social_sentiment=-0.02,
        social_volume=12,
        social_bullish_ratio=0.50,
        social_zscore=-0.2,
        social_bot_ratio=0.05,
        social_is_spike=False,
    )
    logger.info("\n--- SC6 quiet ---")
    print(r6.summary())

    # Validation
    assert r1.regime == REGIME_SMART_BUY, r1.regime
    assert r2.regime == REGIME_RETAIL_PUMP, r2.regime
    assert r3.regime == REGIME_ECHO_CHAMBER, r3.regime
    assert r4.regime == REGIME_PANIC, r4.regime
    assert r5.regime == REGIME_CONSENSUS_BULL, r5.regime
    assert r6.regime == REGIME_QUIET, r6.regime
    assert r1.suggested_bias > 0
    assert r2.suggested_bias < 0
    assert divergence_to_confidence_multiplier(r2) < 1.0
    assert divergence_to_confidence_multiplier(r5) >= 1.0
    logger.info("\nOK - tous les regimes detectes correctement")
