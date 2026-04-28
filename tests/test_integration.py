"""
test_pipeline_integration.py
Test d'intégration des 3 nouveaux patterns sur un article simulé.
Lance uniquement le débat (sans news_pipeline ni SQLite).
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add root project dir to path
# Charge .env avant tout
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Article test — taille modérée (~1800 chars) → niveau MICRO attendu
ARTICLE_TEST = """
Apple Inc. reported first-quarter fiscal 2026 results that exceeded Wall Street expectations
on virtually every metric. Revenue came in at $124.3 billion, up 4% year-over-year and
beating analyst consensus of $117.5 billion by 6%. Earnings per share reached $2.18,
a 16% jump from the same period last year, well above the $1.89 estimate.

Gross margin expanded to 45.9%, driven by the continued strength of the Services segment,
which posted $26.3 billion in revenue — a record quarter. iPhone revenue was $69.1 billion,
slightly below expectations but offset by Mac and iPad outperformance.

However, the company disclosed that CFO Luca Maestri will step down effective next quarter,
citing personal reasons. His successor has not been named. Analysts reacted cautiously to
the leadership change, noting Maestri's instrumental role in Apple's capital allocation
strategy including the $90 billion share buyback program announced last year.

Management raised full-year guidance above consensus estimates, projecting 5-7% revenue
growth for fiscal 2026. The company also warned of potential FX headwinds due to
US dollar strengthening against major currencies, particularly the euro and yen.
Apple shares were up 3.2% in after-hours trading following the earnings release.
"""

ABSA_TEST = {
    "aspects": [
        {
            "aspect": "earnings",
            "sentiment": "positive",
            "evidence": "Earnings per share reached $2.18, a 16% jump",
            "reason": "EPS beat drives valuation re-rating",
        },
        {
            "aspect": "revenue",
            "sentiment": "positive",
            "evidence": "Revenue came in at $124.3 billion, beating analyst consensus by 6%",
            "reason": "Top-line beat reduces growth concern premium",
        },
        {
            "aspect": "guidance",
            "sentiment": "positive",
            "evidence": "Management raised full-year guidance above consensus estimates",
            "reason": "Forward guidance revision signals management confidence",
        },
        {
            "aspect": "leadership",
            "sentiment": "negative",
            "evidence": "CFO Luca Maestri will step down effective next quarter",
            "reason": "CFO departure introduces execution and strategy uncertainty",
        },
        {
            "aspect": "macro_exposure",
            "sentiment": "negative",
            "evidence": "potential FX headwinds due to US dollar strengthening",
            "reason": "Currency exposure reduces international revenue when converted to USD",
        },
    ]
}

CONTEXTE_MARCHE_TEST = {"current_price": 189.5, "volume": 55_000_000, "variation_5d": 2.3}


def test_integration_pipeline():
    print("\n" + "=" * 70)
    print("TEST D'INTÉGRATION — Architecture LangGraph + AutoGen")
    print("Patterns : Shared Scratchpad + AutoDream Memory + Context Compression")
    print("=" * 70)

    # 1. Test du compresseur de contexte en isolation
    print("\n[1/3] Test — Context Compressor (autoCompact)")
    print("-" * 50)
    from src.utils.context_compressor import compress_article_if_needed, estimate_tokens

    result_compression = compress_article_if_needed(ARTICLE_TEST, "AAPL")
    assert result_compression.level.value in ["micro", "auto", "full"]

    # 2. Test du débat complet avec Shared Scratchpad
    print("\n[2/3] Test — Débat multi-agent + Shared Scratchpad")
    print("-" * 50)
    from src.agents.agent_debat import workflow_debat_actualite
    from src.pipelines.agent_pipeline import _calculer_metrics_objectives

    decision = workflow_debat_actualite.invoke(
        {
            "texte_article": ARTICLE_TEST,
            "ticker_symbol": "AAPL",
            "contexte_marche": CONTEXTE_MARCHE_TEST,
            "absa_result": ABSA_TEST,
        }
    )

    fake_finbert_score = 0.95
    consensus_rate, impact_strength = _calculer_metrics_objectives(
        decision.get("signal", "Neutre"), fake_finbert_score, ABSA_TEST
    )

    assert "signal" in decision
    assert "scratchpad_xml" in decision

    # 3. Test de l'Agent Mémoire
    print("\n[3/3] Test — Agent Mémoire AutoDream (lecture index)")
    print("-" * 50)
    from src.agents.agent_memoire import load_context_for_session

    memory = load_context_for_session(["AAPL"])
    # Pas de plantage = OK


if __name__ == "__main__":
    test_integration_pipeline()
