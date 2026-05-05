"""
Étage 3 — Processing multi-features.

Pour chaque article filtré :
  - PolarityAgent       FinBERT          → polarity, polarity_conf
  - UncertaintyAgent    heuristique L&M  → uncertainty
  - LitigiousAgent      lexique L&M      → legal_risk
  - FundamentalAgent    lexique fund.    → fundamental_strength
  - SentimentAgent LLM  phi4-mini JSON   → sentiment, score, reasoning

Features dérivées (samuel) :
    risk_adjusted_sentiment = polarity × polarity_conf
    headline_conviction     = polarity_conf × (1 − uncertainty)
    fundamental_impact      = fundamental_strength × risk_adjusted_sentiment

Provenance :
- Architecture multi-agents + features dérivées : branche `samuel`.
- SentimentAgent LLM JSON                      : branche `Antoinev2`.
- Séparation news/processing                   : branche `poc-processing-lorenzo`.
"""

import logging
import sqlite3
from datetime import datetime, timezone

from polarity_agent import PolarityAgent
from uncertainty_agent import UncertaintyAgent
from litigious_agent import LitigiousAgent
from fundamental_strength_agent import FundamentalStrengthAgent
from sentiment_agent import analyser_article, OllamaUnavailableError


logger = logging.getLogger("ProcessingPipeline")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)


DB_PATH = "news_database.db"
MAX_TEXT = 1500


# ─── SCHÉMA article_scores ────────────────────────────────────────────────────
ARTICLE_SCORES_COLUMNS = [
    ("ticker", "TEXT"),
    ("polarity", "INTEGER"),
    ("polarity_conf", "REAL"),
    ("uncertainty", "REAL"),
    ("legal_risk", "REAL"),
    ("fundamental_strength", "REAL"),
    ("sentiment", "TEXT"),
    ("score", "REAL"),
    ("reasoning", "TEXT"),
    ("risk_adjusted_sentiment", "REAL"),
    ("headline_conviction", "REAL"),
    ("fundamental_impact", "REAL"),
    ("analyzed_at", "TEXT"),
]


def ensure_schema(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS article_scores (
            url TEXT PRIMARY KEY
        )
        """
    )
    existing = {row[1] for row in cursor.execute("PRAGMA table_info(article_scores)")}
    for name, dtype in ARTICLE_SCORES_COLUMNS:
        if name not in existing:
            cursor.execute(f"ALTER TABLE article_scores ADD COLUMN {name} {dtype}")


# ─── FEATURES DÉRIVÉES (samuel) ───────────────────────────────────────────────
def compute_derived_features(polarity, polarity_conf, uncertainty, fundamental_strength):
    risk_adjusted_sentiment = polarity * polarity_conf
    headline_conviction = polarity_conf * (1.0 - uncertainty)
    fundamental_impact = fundamental_strength * risk_adjusted_sentiment
    return (
        round(risk_adjusted_sentiment, 4),
        round(headline_conviction, 4),
        round(fundamental_impact, 4),
    )


# ─── ANALYSE D'UN ARTICLE ─────────────────────────────────────────────────────
def process_article(article, agents):
    """
    article = (url, ticker, title, content)
    agents  = dict avec polarity, uncertainty, litigious, fundamental
    """
    url, ticker, title, content = article
    text = (content if content else title) or ""
    text = text[:MAX_TEXT]

    # 1. Agents déterministes amont (parallèle dans la sémantique, séquentiel ici)
    polarity, polarity_conf, _label = agents["polarity"].predict(text)
    uncertainty = agents["uncertainty"].predict(text)
    legal_risk = agents["litigious"].predict(text)
    fundamental_strength = agents["fundamental"].predict(text)

    # 2. Agrégation des features amont → dict transmis à phi4 comme contexte
    #    Permet à phi4 de jouer le rôle d'arbitre final (synthèse) au lieu
    #    d'analyser l'article en aveugle, en parallèle des autres agents.
    upstream_features = {
        "polarity": int(polarity),
        "polarity_conf": float(polarity_conf),
        "uncertainty": float(uncertainty),
        "legal_risk": float(legal_risk),
        "fundamental_strength": float(fundamental_strength),
    }

    # 3. SentimentAgent LLM (phi4-mini) en synthèse, informé par l'archi
    # - OllamaUnavailableError → on remonte, l'orchestrateur arrête tout
    #   (sinon : tous les articles seraient marqués 'neutral' silencieusement)
    # - Autres erreurs (JSON parse, score invalide…) → on tag 'neutral' et on continue
    try:
        result = analyser_article(url, ticker, title, content, features=upstream_features)
        sentiment = result["sentiment"]
        score = result["score"]
        reasoning = result["reasoning"]
    except OllamaUnavailableError:
        raise
    except Exception as e:
        logger.warning(f"[SentimentLLM] Erreur parsing ({e}) pour '{title}' → fallback neutral")
        sentiment, score, reasoning = "neutral", None, f"sentiment_llm_error: {e}"

    risk_adj, conviction, fund_impact = compute_derived_features(
        polarity, polarity_conf, uncertainty, fundamental_strength
    )

    return {
        "url": url,
        "ticker": ticker,
        "polarity": int(polarity),
        "polarity_conf": float(polarity_conf),
        "uncertainty": float(uncertainty),
        "legal_risk": float(legal_risk),
        "fundamental_strength": float(fundamental_strength),
        "sentiment": sentiment,
        "score": score,
        "reasoning": reasoning,
        "risk_adjusted_sentiment": risk_adj,
        "headline_conviction": conviction,
        "fundamental_impact": fund_impact,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


def insert_article_scores(cursor, row):
    cursor.execute(
        """
        INSERT OR REPLACE INTO article_scores (
            url, ticker, polarity, polarity_conf, uncertainty,
            legal_risk, fundamental_strength,
            sentiment, score, reasoning,
            risk_adjusted_sentiment, headline_conviction, fundamental_impact,
            analyzed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            row["url"], row["ticker"], row["polarity"], row["polarity_conf"],
            row["uncertainty"], row["legal_risk"], row["fundamental_strength"],
            row["sentiment"], row["score"], row["reasoning"],
            row["risk_adjusted_sentiment"], row["headline_conviction"],
            row["fundamental_impact"], row["analyzed_at"],
        ),
    )


# ─── ENTRÉE PUBLIQUE ──────────────────────────────────────────────────────────
def build_agents():
    logger.info("Chargement des agents de processing...")
    polarity = PolarityAgent()
    uncertainty = UncertaintyAgent()
    litigious = LitigiousAgent(model_path="./litigious_model", fallback_to_heuristic=True)
    fundamental = FundamentalStrengthAgent(
        model_path="./fundamental_strength_model", fallback_to_heuristic=True
    )
    return {
        "polarity": polarity,
        "uncertainty": uncertainty,
        "litigious": litigious,
        "fundamental": fundamental,
    }


def run_processing_pipeline():
    """Mode standalone : analyse tous les articles présents en base."""
    agents = build_agents()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    ensure_schema(cursor)
    conn.commit()

    cursor.execute("SELECT url, ticker, title, content FROM articles")
    articles = cursor.fetchall()
    logger.info(f"{len(articles)} article(s) à traiter")

    for article in articles:
        result = process_article(article, agents)
        insert_article_scores(cursor, result)
        conn.commit()
        logger.info(
            f"[{result['ticker']}] {result['sentiment']} "
            f"(score={result['score']}, polarity={result['polarity']}, "
            f"unc={result['uncertainty']:.2f}, legal={result['legal_risk']:.2f}, "
            f"fund={result['fundamental_strength']:.2f})"
        )

    conn.close()
    logger.info("Fin du processing pipeline.")


if __name__ == "__main__":
    run_processing_pipeline()
