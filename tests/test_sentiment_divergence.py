"""
test_sentiment_divergence.py — Tests pour le detecteur de divergence.

Couvre :
  - confiance par volume (saturation log)
  - bot ratio penalise la confiance social
  - 6 regimes : smart_buy/sell, retail_pump, echo_chamber, panic, consensus, quiet
  - regime mixed par defaut
  - bias borne et signed coherent
  - mapping confidence multiplier
  - to_prompt_block
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Confiance volume
# ---------------------------------------------------------------------------


def test_confidence_zero_for_zero_volume():
    from src.knowledge.sentiment_divergence import DivergenceAnalyzer

    da = DivergenceAnalyzer()
    rep = da.analyze(
        ticker="X",
        news_sentiment=0.0,
        news_volume=0,
        social_sentiment=0.0,
        social_volume=0,
        social_bullish_ratio=0.5,
        social_zscore=0.0,
        social_bot_ratio=0.0,
        social_is_spike=False,
    )
    assert rep.confidence_news == 0.0
    assert rep.confidence_social == 0.0
    assert rep.confidence_overall == 0.0


def test_confidence_increases_with_volume():
    from src.knowledge.sentiment_divergence import DivergenceAnalyzer

    da = DivergenceAnalyzer()
    low = da.analyze(
        ticker="X",
        news_sentiment=0.4,
        news_volume=2,
        social_sentiment=0.4,
        social_volume=2,
        social_bullish_ratio=0.5,
        social_zscore=0.0,
        social_bot_ratio=0.0,
        social_is_spike=False,
    )
    high = da.analyze(
        ticker="X",
        news_sentiment=0.4,
        news_volume=50,
        social_sentiment=0.4,
        social_volume=200,
        social_bullish_ratio=0.5,
        social_zscore=0.0,
        social_bot_ratio=0.0,
        social_is_spike=False,
    )
    assert high.confidence_overall > low.confidence_overall


def test_bot_ratio_penalizes_social_confidence():
    from src.knowledge.sentiment_divergence import DivergenceAnalyzer

    da = DivergenceAnalyzer()
    no_bot = da.analyze(
        ticker="X",
        news_sentiment=0.0,
        news_volume=20,
        social_sentiment=0.0,
        social_volume=200,
        social_bullish_ratio=0.5,
        social_zscore=0.0,
        social_bot_ratio=0.0,
        social_is_spike=False,
    )
    full_bot = da.analyze(
        ticker="X",
        news_sentiment=0.0,
        news_volume=20,
        social_sentiment=0.0,
        social_volume=200,
        social_bullish_ratio=0.5,
        social_zscore=0.0,
        social_bot_ratio=1.0,
        social_is_spike=False,
    )
    assert no_bot.confidence_social > full_bot.confidence_social
    assert full_bot.confidence_social == 0.0


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------


def test_regime_smart_buy_news_pos_social_neg():
    from src.knowledge.sentiment_divergence import (
        REGIME_SMART_BUY,
        DivergenceAnalyzer,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="AAPL",
        news_sentiment=+0.5,
        news_volume=20,
        social_sentiment=-0.4,
        social_volume=80,
        social_bullish_ratio=0.20,
        social_zscore=0.0,
        social_bot_ratio=0.0,
        social_is_spike=False,
    )
    assert rep.regime == REGIME_SMART_BUY
    assert rep.suggested_bias > 0


def test_regime_smart_sell_news_neg_social_pos():
    from src.knowledge.sentiment_divergence import (
        REGIME_SMART_SELL,
        DivergenceAnalyzer,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="X",
        news_sentiment=-0.5,
        news_volume=15,
        social_sentiment=+0.4,
        social_volume=80,
        social_bullish_ratio=0.85,
        social_zscore=0.5,
        social_bot_ratio=0.0,
        social_is_spike=False,
    )
    assert rep.regime == REGIME_SMART_SELL
    assert rep.suggested_bias < 0


def test_regime_retail_pump_priority_over_others():
    """retail_pump doit etre detecte meme si echo_chamber matcherait aussi."""
    from src.knowledge.sentiment_divergence import (
        REGIME_RETAIL_PUMP,
        DivergenceAnalyzer,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="GME",
        news_sentiment=-0.05,
        news_volume=8,
        social_sentiment=+0.85,
        social_volume=400,
        social_bullish_ratio=0.92,
        social_zscore=4.5,
        social_bot_ratio=0.45,
        social_is_spike=True,
    )
    assert rep.regime == REGIME_RETAIL_PUMP
    assert rep.suggested_bias < 0


def test_regime_echo_chamber_no_bots():
    from src.knowledge.sentiment_divergence import (
        REGIME_ECHO_CHAMBER,
        DivergenceAnalyzer,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="NVDA",
        news_sentiment=+0.20,
        news_volume=20,
        social_sentiment=+0.78,
        social_volume=300,
        social_bullish_ratio=0.90,
        social_zscore=3.5,
        social_bot_ratio=0.10,
        social_is_spike=True,
    )
    assert rep.regime == REGIME_ECHO_CHAMBER


def test_regime_panic_capitulation():
    from src.knowledge.sentiment_divergence import (
        REGIME_PANIC,
        DivergenceAnalyzer,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="X",
        news_sentiment=-0.10,
        news_volume=10,
        social_sentiment=-0.65,
        social_volume=300,
        social_bullish_ratio=0.18,
        social_zscore=3.0,
        social_bot_ratio=0.10,
        social_is_spike=True,
    )
    assert rep.regime == REGIME_PANIC


def test_regime_consensus_bull():
    from src.knowledge.sentiment_divergence import (
        REGIME_CONSENSUS_BULL,
        DivergenceAnalyzer,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="MSFT",
        news_sentiment=+0.50,
        news_volume=25,
        social_sentiment=+0.45,
        social_volume=150,
        social_bullish_ratio=0.70,
        social_zscore=0.8,
        social_bot_ratio=0.05,
        social_is_spike=False,
    )
    assert rep.regime == REGIME_CONSENSUS_BULL
    assert rep.suggested_bias > 0


def test_regime_consensus_bear():
    from src.knowledge.sentiment_divergence import (
        REGIME_CONSENSUS_BEAR,
        DivergenceAnalyzer,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="X",
        news_sentiment=-0.50,
        news_volume=25,
        social_sentiment=-0.55,
        social_volume=150,
        social_bullish_ratio=0.20,
        social_zscore=0.5,
        social_bot_ratio=0.05,
        social_is_spike=False,
    )
    assert rep.regime == REGIME_CONSENSUS_BEAR
    assert rep.suggested_bias < 0


def test_regime_neutral_quiet():
    from src.knowledge.sentiment_divergence import (
        REGIME_QUIET,
        DivergenceAnalyzer,
    )

    rep = DivergenceAnalyzer().analyze(
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
    assert rep.regime == REGIME_QUIET
    assert rep.suggested_bias == 0.0


def test_regime_mixed_default():
    """Cas qui ne matche aucun pattern specifique -> mixed."""
    from src.knowledge.sentiment_divergence import (
        REGIME_MIXED,
        DivergenceAnalyzer,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="X",
        news_sentiment=+0.20,
        news_volume=8,
        social_sentiment=+0.18,
        social_volume=40,
        social_bullish_ratio=0.55,
        social_zscore=0.3,
        social_bot_ratio=0.05,
        social_is_spike=False,
    )
    assert rep.regime == REGIME_MIXED


# ---------------------------------------------------------------------------
# Divergence math
# ---------------------------------------------------------------------------


def test_divergence_equals_news_minus_social():
    from src.knowledge.sentiment_divergence import DivergenceAnalyzer

    rep = DivergenceAnalyzer().analyze(
        ticker="X",
        news_sentiment=+0.6,
        news_volume=10,
        social_sentiment=-0.3,
        social_volume=50,
        social_bullish_ratio=0.30,
        social_zscore=0.0,
        social_bot_ratio=0.0,
        social_is_spike=False,
    )
    assert rep.divergence == pytest.approx(0.9, abs=0.001)


def test_inputs_are_clipped_to_unit_interval():
    from src.knowledge.sentiment_divergence import DivergenceAnalyzer

    rep = DivergenceAnalyzer().analyze(
        ticker="X",
        news_sentiment=+5.0,
        news_volume=10,
        social_sentiment=-99.0,
        social_volume=10,
        social_bullish_ratio=2.0,
        social_zscore=0.0,
        social_bot_ratio=99.0,
        social_is_spike=False,
    )
    assert rep.news_sentiment == 1.0
    assert rep.social_sentiment == -1.0
    assert rep.social_bullish_ratio == 1.0
    assert rep.social_bot_ratio == 1.0


# ---------------------------------------------------------------------------
# SocialSentimentReport adapter
# ---------------------------------------------------------------------------


def test_accepts_social_report_object():
    from src.knowledge.sentiment_divergence import (
        REGIME_SMART_BUY,
        DivergenceAnalyzer,
    )

    class _FakeReport:
        sentiment_score = -0.4
        messages_count = 80
        bullish_ratio = 0.20
        volume_zscore = 0.0
        bot_ratio = 0.0
        is_spike = False

    rep = DivergenceAnalyzer().analyze(
        ticker="X",
        news_sentiment=+0.5,
        news_volume=20,
        social_report=_FakeReport(),
    )
    assert rep.regime == REGIME_SMART_BUY


def test_missing_social_inputs_raises():
    from src.knowledge.sentiment_divergence import DivergenceAnalyzer

    with pytest.raises(ValueError):
        DivergenceAnalyzer().analyze(
            ticker="X",
            news_sentiment=+0.5,
            news_volume=10,
            # ni report ni valeurs explicites
        )


# ---------------------------------------------------------------------------
# Confidence multiplier
# ---------------------------------------------------------------------------


def test_confidence_multiplier_smart_buy_boosts():
    from src.knowledge.sentiment_divergence import (
        DivergenceAnalyzer,
        divergence_to_confidence_multiplier,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="AAPL",
        news_sentiment=+0.6,
        news_volume=30,
        social_sentiment=-0.4,
        social_volume=100,
        social_bullish_ratio=0.20,
        social_zscore=0.0,
        social_bot_ratio=0.0,
        social_is_spike=False,
    )
    m = divergence_to_confidence_multiplier(rep)
    assert m >= 1.0
    assert m <= 1.25


def test_confidence_multiplier_retail_pump_reduces():
    from src.knowledge.sentiment_divergence import (
        DivergenceAnalyzer,
        divergence_to_confidence_multiplier,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="GME",
        news_sentiment=-0.05,
        news_volume=8,
        social_sentiment=+0.85,
        social_volume=400,
        social_bullish_ratio=0.92,
        social_zscore=4.5,
        social_bot_ratio=0.45,
        social_is_spike=True,
    )
    m = divergence_to_confidence_multiplier(rep)
    assert m < 1.0
    assert m >= 0.5


def test_confidence_multiplier_quiet_neutral():
    from src.knowledge.sentiment_divergence import (
        DivergenceAnalyzer,
        divergence_to_confidence_multiplier,
    )

    rep = DivergenceAnalyzer().analyze(
        ticker="JNJ",
        news_sentiment=+0.02,
        news_volume=3,
        social_sentiment=-0.01,
        social_volume=10,
        social_bullish_ratio=0.50,
        social_zscore=-0.2,
        social_bot_ratio=0.05,
        social_is_spike=False,
    )
    assert divergence_to_confidence_multiplier(rep) == pytest.approx(1.0, abs=0.001)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------


def test_to_prompt_block_well_formed_and_ascii_safe():
    from src.knowledge.sentiment_divergence import DivergenceAnalyzer

    rep = DivergenceAnalyzer().analyze(
        ticker="AAPL",
        news_sentiment=+0.5,
        news_volume=20,
        social_sentiment=-0.3,
        social_volume=80,
        social_bullish_ratio=0.30,
        social_zscore=0.0,
        social_bot_ratio=0.0,
        social_is_spike=False,
    )
    block = rep.to_prompt_block()
    block.encode("cp1252")
    assert "<sentiment_divergence>" in block
    assert "</sentiment_divergence>" in block
    assert "regime=" in block
    assert "suggested_bias=" in block
