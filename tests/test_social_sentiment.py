"""
test_social_sentiment.py — Tests pour le module social_sentiment.

Couvre :
  - regex cashtag
  - scorer (lexicon, VADER fallback, label explicite)
  - estimate_bot_ratio
  - aggregator : volume, sentiment, spike, bot_ratio, top messages
  - to_prompt_block ASCII-safe
  - source weights et engagement weighting
  - filtrage temporel
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Cashtag regex
# ---------------------------------------------------------------------------


def test_cashtag_extracts_uppercase_tickers():
    from src.knowledge.social_sentiment import CASHTAG_RE

    assert CASHTAG_RE.findall("Buy $AAPL and $MSFT today") == ["AAPL", "MSFT"]
    assert CASHTAG_RE.findall("$SPY $QQQ $DIA") == ["SPY", "QQQ", "DIA"]


def test_cashtag_ignores_lowercase_or_word_continuations():
    from src.knowledge.social_sentiment import CASHTAG_RE

    # Pas de cashtag sans $
    assert CASHTAG_RE.findall("TSLA stock is great") == []
    # $abc en minuscule pas detecte
    assert CASHTAG_RE.findall("$abc $XyZ") == []
    # >5 caracteres rejet (le lookahead negatif voit la 6e lettre)
    assert CASHTAG_RE.findall("$ABCDEF stuff") == []
    # 5 caracteres max accepte (NASDAQ tickers comme GOOGL = 5)
    assert "GOOGL" in CASHTAG_RE.findall("$GOOGL is fine")


def test_cashtag_handles_punctuation_neighbors():
    from src.knowledge.social_sentiment import CASHTAG_RE

    assert CASHTAG_RE.findall("buy ($AAPL),hold $MSFT.") == ["AAPL", "MSFT"]


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


def test_scorer_explicit_bullish_label_overrides_text():
    from src.knowledge.social_sentiment import (
        SocialMessage,
        SocialSentimentScorer,
    )

    msg = SocialMessage(
        source="stocktwits",
        message_id="x",
        author="x",
        text="actually this looks weak",
        posted_at=datetime.now(timezone.utc),
        explicit_label="Bullish",
    )
    s = SocialSentimentScorer(use_vader=False).score(msg)
    assert s > 0.5  # le label gold doit dominer


def test_scorer_lexicon_bullish_text():
    from src.knowledge.social_sentiment import (
        SocialMessage,
        SocialSentimentScorer,
    )

    msg = SocialMessage(
        source="reddit",
        message_id="x",
        author="x",
        text="moon rocket diamond hands tendies bullish",
        posted_at=datetime.now(timezone.utc),
    )
    s = SocialSentimentScorer(use_vader=False).score(msg)
    assert s > 0.0


def test_scorer_lexicon_bearish_text():
    from src.knowledge.social_sentiment import (
        SocialMessage,
        SocialSentimentScorer,
    )

    msg = SocialMessage(
        source="reddit",
        message_id="x",
        author="x",
        text="crash dump bearish puts disaster bagholder",
        posted_at=datetime.now(timezone.utc),
    )
    s = SocialSentimentScorer(use_vader=False).score(msg)
    assert s < 0.0


def test_scorer_neutral_text_close_to_zero():
    from src.knowledge.social_sentiment import (
        SocialMessage,
        SocialSentimentScorer,
    )

    msg = SocialMessage(
        source="reddit",
        message_id="x",
        author="x",
        text="I am holding the position right now",
        posted_at=datetime.now(timezone.utc),
    )
    s = SocialSentimentScorer(use_vader=False).score(msg)
    assert -0.5 <= s <= 0.5


# ---------------------------------------------------------------------------
# Bot detection
# ---------------------------------------------------------------------------


def test_estimate_bot_ratio_flags_too_many_cashtags():
    from src.knowledge.social_sentiment import (
        SocialMessage,
        estimate_bot_ratio,
    )

    now = datetime.now(timezone.utc)
    msgs = [
        SocialMessage(
            source="stocktwits",
            message_id="1",
            author="a",
            text="legit message about apple earnings yesterday",
            posted_at=now,
            cashtags=["AAPL"],
        ),
        SocialMessage(
            source="stocktwits",
            message_id="2",
            author="b",
            text="$AAPL $MSFT $GOOGL $AMZN $NVDA $META $TSLA $JPM $BAC",
            posted_at=now,
            cashtags=["A", "B", "C", "D", "E", "F", "G", "H", "I"],
        ),
    ]
    ratio = estimate_bot_ratio(msgs)
    assert 0.0 < ratio <= 1.0


def test_estimate_bot_ratio_zero_for_clean_messages():
    from src.knowledge.social_sentiment import (
        SocialMessage,
        estimate_bot_ratio,
    )

    now = datetime.now(timezone.utc)
    msgs = [
        SocialMessage(
            source="stocktwits",
            message_id=str(i),
            author=f"u{i}",
            text=f"This is a substantive analysis of position number {i} today",
            posted_at=now,
            cashtags=["AAPL"],
        )
        for i in range(5)
    ]
    ratio = estimate_bot_ratio(msgs)
    assert ratio == 0.0


def test_estimate_bot_ratio_empty_input():
    from src.knowledge.social_sentiment import estimate_bot_ratio

    assert estimate_bot_ratio([]) == 0.0


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _bull_msg(i, source="stocktwits", likes=0):
    from src.knowledge.social_sentiment import SocialMessage

    return SocialMessage(
        source=source,
        message_id=str(i),
        author=f"u{i}",
        text="moon rocket diamond hands tendies bullish $AAPL",
        posted_at=datetime.now(timezone.utc) - timedelta(hours=2),
        cashtags=["AAPL"],
        likes=likes,
        explicit_label="Bullish",
    )


def _bear_msg(i, source="stocktwits", likes=0):
    from src.knowledge.social_sentiment import SocialMessage

    return SocialMessage(
        source=source,
        message_id=str(i),
        author=f"u{i}",
        text="crash dump bearish puts disaster $AAPL",
        posted_at=datetime.now(timezone.utc) - timedelta(hours=2),
        cashtags=["AAPL"],
        likes=likes,
        explicit_label="Bearish",
    )


def test_aggregator_basic_sentiment_distribution():
    from src.knowledge.social_sentiment import SocialSentimentAggregator

    msgs = [_bull_msg(i) for i in range(3)] + [_bear_msg(i + 10) for i in range(2)]
    agg = SocialSentimentAggregator()
    rep = agg.analyze(ticker="AAPL", baseline_24h_avg=10.0, messages=msgs)
    assert rep.messages_count == 5
    assert rep.bullish_count == 3
    assert rep.bearish_count == 2
    assert rep.sentiment_score > 0.0


def test_aggregator_spike_detection_via_zscore():
    from src.knowledge.social_sentiment import SocialSentimentAggregator

    # baseline tres bas + nb messages eleve -> z >> 2
    msgs = [_bull_msg(i) for i in range(20)]
    rep = SocialSentimentAggregator().analyze(
        ticker="AAPL",
        baseline_24h_avg=2.0,
        messages=msgs,
    )
    assert rep.is_spike is True
    assert rep.volume_zscore > 2.0


def test_aggregator_no_spike_when_volume_normal():
    from src.knowledge.social_sentiment import SocialSentimentAggregator

    msgs = [_bull_msg(i) for i in range(5)]
    rep = SocialSentimentAggregator().analyze(
        ticker="AAPL",
        baseline_24h_avg=20.0,
        messages=msgs,
    )
    assert rep.is_spike is False


def test_aggregator_filters_messages_outside_window():
    from src.knowledge.social_sentiment import (
        SocialMessage,
        SocialSentimentAggregator,
    )

    now = datetime.now(timezone.utc)
    in_window = SocialMessage(
        source="reddit",
        message_id="recent",
        author="u",
        text="moon bullish",
        posted_at=now - timedelta(hours=2),
        explicit_label="Bullish",
    )
    too_old = SocialMessage(
        source="reddit",
        message_id="old",
        author="u",
        text="moon bullish",
        posted_at=now - timedelta(hours=48),
        explicit_label="Bullish",
    )
    rep = SocialSentimentAggregator(window_hours=24).analyze(
        ticker="AAPL",
        messages=[in_window, too_old],
    )
    assert rep.messages_count == 1


def test_aggregator_engagement_weighting():
    """Un message tres engaging doit peser plus."""
    from src.knowledge.social_sentiment import SocialSentimentAggregator

    big_bull = _bull_msg(1, likes=200)  # tres engaging
    small_bear = _bear_msg(2, likes=0)
    rep = SocialSentimentAggregator().analyze(
        ticker="AAPL",
        messages=[big_bull, small_bear],
    )
    # le bull devrait dominer le score
    assert rep.sentiment_score > 0.0


def test_aggregator_to_prompt_block_ascii_safe():
    from src.knowledge.social_sentiment import SocialSentimentAggregator

    msgs = [_bull_msg(i) for i in range(2)] + [_bear_msg(i + 10) for i in range(1)]
    rep = SocialSentimentAggregator().analyze(
        ticker="AAPL",
        baseline_24h_avg=5.0,
        messages=msgs,
    )
    block = rep.to_prompt_block()
    # Doit pouvoir s'encoder sur un terminal Windows cp1252 sans crash
    block.encode("cp1252")
    assert "<social_sentiment>" in block
    assert "</social_sentiment>" in block


def test_aggregator_unique_authors_count():
    from src.knowledge.social_sentiment import SocialSentimentAggregator

    msgs = [_bull_msg(0), _bull_msg(0), _bear_msg(1)]  # 2 auteurs uniques
    rep = SocialSentimentAggregator().analyze(
        ticker="AAPL",
        messages=msgs,
    )
    assert rep.unique_authors == 2


def test_aggregator_empty_messages_gives_zero():
    from src.knowledge.social_sentiment import SocialSentimentAggregator

    rep = SocialSentimentAggregator().analyze(ticker="ZZZ", messages=[])
    assert rep.messages_count == 0
    assert rep.sentiment_score == 0.0
    assert rep.is_spike is False
