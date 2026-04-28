"""
agent_absa.py

Aspect-Based Sentiment Analysis (ABSA) avec taxonomie économique fermée.
Utilisé comme couche intermédiaire entre DistilRoBERTa et le débat multi-agent.

Taxonomie fermée (15 aspects) :
  revenue, earnings, guidance, margin, debt, liquidity,
  litigation, macro_exposure, leadership, product_launch,
  regulatory_risk, competition, valuation, supply_chain, esg_risk

Améliorations v2 :
  - 5 nouveaux aspects pour mieux couvrir les articles macro-économiques
  - Prompt enrichi avec instructions spécifiques pour les articles vagues
  - Fallback contextuel : si l'article est macro, forcer l'aspect macro_exposure

Usage direct :
    from agent_absa import run_absa
    result = run_absa(content)  # → {"aspects": [...]}
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)
import hashlib
import json
import logging
import os
import re
from collections import OrderedDict
from typing import Optional

from src.utils.llm_client import AllProvidersFailedError, LLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Client LLM unifie : remplace l'ancien _get_absa_client + _absa_llm_call.
# ---------------------------------------------------------------------------
# Le LLMClient gere :
#   - Selection automatique du provider disponible (Groq -> Mistral -> Cerebras)
#   - Retry exponentiel borne (max 3 tentatives, cap 8s)
#   - Fallback inter-provider si tout un fournisseur tombe
#   - Logging structure de chaque tentative
# Voir src/utils/llm_client.py et ADR-003 dans ARCHITECTURE_DECISIONS.md.
_ABSA_CLIENT: LLMClient | None = None


def _get_absa_client() -> LLMClient:
    """Lazy-init du LLMClient partage pour ABSA."""
    global _ABSA_CLIENT  # noqa: PLW0603
    if _ABSA_CLIENT is None:
        _ABSA_CLIENT = LLMClient.from_env()
    return _ABSA_CLIENT


# ---------------------------------------------------------------------------
# Cache LRU en mémoire (anti-recomputation ABSA)
# ---------------------------------------------------------------------------
# Deux articles identiques (même URL re-scrapée, même contenu republié) ne
# doivent pas déclencher deux appels LLM. On mémoïse sur un hash du contenu
# tronqué à 500 chars (suffisant pour caractériser l'article).
#
# Taille : 256 entrées max — chaque entrée ~500 octets → ~128 KB en mémoire.
# Rationale : un batch quotidien traite ~100-200 articles, un cache de 256
# couvre un jour entier de pipeline.

_ABSA_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_ABSA_CACHE_MAX = 256


def _absa_cache_key(text: str) -> str:
    """Hash stable du contenu tronqué (500 chars) pour mémoïsation."""
    h = hashlib.sha256(text[:500].strip().lower().encode("utf-8")).hexdigest()
    return h[:16]


def _absa_cache_get(key: str) -> Optional[dict]:
    """LRU get : déplace la clé en fin de queue si hit."""
    if key in _ABSA_CACHE:
        _ABSA_CACHE.move_to_end(key)
        return _ABSA_CACHE[key]
    return None


def _absa_cache_put(key: str, result: dict) -> None:
    """LRU put : éviction de la plus ancienne entrée si plein."""
    _ABSA_CACHE[key] = result
    _ABSA_CACHE.move_to_end(key)
    while len(_ABSA_CACHE) > _ABSA_CACHE_MAX:
        _ABSA_CACHE.popitem(last=False)


# ---------------------------------------------------------------------------
# Taxonomie
# ---------------------------------------------------------------------------

ASPECTS = [
    # Aspects fondamentaux (taxonomie originale)
    "revenue",
    "earnings",
    "guidance",
    "margin",
    "debt",
    "liquidity",
    "litigation",
    "macro_exposure",
    "leadership",
    "product_launch",
    # Nouveaux aspects v2 — couvrent mieux les articles macro et sectoriels
    "regulatory_risk",  # Risques réglementaires, lois, amendes potentielles
    "competition",  # Pression concurrentielle, parts de marché
    "valuation",  # Valorisation boursière, P/E ratio, rachats d'actions
    "supply_chain",  # Chaîne d'approvisionnement, pénuries, fournisseurs
    "esg_risk",  # Risques ESG : environnement, social, gouvernance
]

# ---------------------------------------------------------------------------
# Prompt ABSA
# ---------------------------------------------------------------------------

PROMPT_ABSA_SYSTEM = """You are a financial text analyst specialized in Aspect-Based Sentiment Analysis (ABSA).

CLOSED TAXONOMY — You MUST only use these exact aspect names, nothing else:
  revenue, earnings, guidance, margin, debt, liquidity, litigation,
  macro_exposure, leadership, product_launch,
  regulatory_risk, competition, valuation, supply_chain, esg_risk

Aspect definitions (use these to choose the best match):
  revenue         : Sales, top-line growth, market share in revenue terms
  earnings        : Net income, EPS, profit, EBITDA
  guidance        : Forward outlook, forecast revision, management estimates
  margin          : Gross/operating/net margin, cost pressure, pricing power
  debt            : Leverage, interest rates, credit rating, refinancing
  liquidity       : Cash position, free cash flow, capital allocation
  litigation      : Lawsuits, fines, legal proceedings
  macro_exposure  : Broad macro factors (interest rates, FX, inflation, recession, tariffs, geopolitics)
  leadership      : CEO/CFO changes, management quality, board decisions
  product_launch  : New products, services, partnerships, R&D milestones
  regulatory_risk : Government regulations, compliance costs, antitrust, data privacy laws
  competition     : Competitive dynamics, market share loss, new entrants, pricing war
  valuation       : Stock buybacks, P/E ratio, analyst target price, market cap discussions
  supply_chain    : Component shortages, supplier issues, logistics, inventory
  esg_risk        : Environmental, social, or governance risks and controversies

STRICT RULES:
1. Only include aspects that are EXPLICITLY stated or clearly implied in the text.
2. Do NOT invent, extrapolate, or hallucinate aspects.
3. Do NOT use any aspect outside the taxonomy above.
4. For macro-economic or vague articles, prefer macro_exposure, regulatory_risk, or competition
   over generic terms. If the article discusses broad economic conditions affecting the company,
   use macro_exposure even if the link is indirect.
5. For each detected aspect provide:
   - "aspect"    : exact name from the taxonomy
   - "sentiment" : "positive" | "negative" | "neutral"
   - "evidence"  : verbatim quote from the text (max 2 sentences, no paraphrase)
   - "reason"    : one-sentence economic justification
6. If NO aspect is detected, return {"aspects": []}.
7. Return ONLY valid JSON. No preamble, no explanation, no markdown fences.

OUTPUT FORMAT (strict):
{
    "aspects": [
        {
            "aspect": "<taxonomy aspect>",
            "sentiment": "positive|negative|neutral",
            "evidence": "<exact verbatim quote>",
            "reason": "<brief economic justification>"
        }
    ]
}"""


# ---------------------------------------------------------------------------
# Validation et nettoyage du résultat ABSA
# ---------------------------------------------------------------------------


def _validate_absa(raw_result: dict) -> dict:
    """
    Filtre les aspects hors taxonomie et normalise les sentiments.
    Garantit un JSON propre et exploitable par les agents du débat.
    """
    valid_sentiments = {"positive", "negative", "neutral"}
    cleaned = []

    for item in raw_result.get("aspects", []):
        aspect = item.get("aspect", "").strip().lower()
        sentiment = item.get("sentiment", "").strip().lower()

        if aspect not in ASPECTS:
            logger.warning("ABSA: aspect hors taxonomie ignoré → '%s'", aspect)
            continue
        if sentiment not in valid_sentiments:
            logger.warning("ABSA: sentiment invalide pour '%s' → '%s'", aspect, sentiment)
            sentiment = "neutral"

        cleaned.append(
            {
                "aspect": aspect,
                "sentiment": sentiment,
                "evidence": item.get("evidence", "").strip(),
                "reason": item.get("reason", "").strip(),
            }
        )

    return {"aspects": cleaned}


# ---------------------------------------------------------------------------
# Fonction principale
# ---------------------------------------------------------------------------


def run_absa(texte_article: str, max_chars: int = 3000) -> dict:
    """
    Analyse les aspects économiques d'un article financier.

    Args:
        texte_article : contenu de l'article (str)
        max_chars     : limite de caractères soumis au LLM (défaut: 3000)

    Returns:
        dict {"aspects": [...]} — liste vide si aucun aspect détecté.
        Tous les aspects sont garantis dans la taxonomie fermée.
    """
    if not texte_article or len(texte_article) < 50:
        logger.debug("ABSA: article trop court, skip.")
        return {"aspects": []}

    # Cache hit ? Évite un appel LLM sur contenu déjà analysé (duplication DB,
    # feed multi-source, re-scrape). Clé = hash(content[:500]).
    cache_key = _absa_cache_key(texte_article)
    cached = _absa_cache_get(cache_key)
    if cached is not None:
        logger.info("ABSA: cache HIT (%s) — skip LLM call.", cache_key)
        return cached

    if len(texte_article) <= max_chars:
        text_input = texte_article
    else:
        # Prevent cutting mid-sentence by finding the last period within the limit
        cut_index = texte_article.rfind(".", 0, max_chars)
        if cut_index == -1:
            text_input = texte_article[:max_chars]
        else:
            text_input = texte_article[: cut_index + 1]

    try:
        # Migration vers LLMClient (ADR-003) : fallback automatique entre
        # Groq -> Mistral -> Cerebras + retry exponentiel borne intégré.
        client = _get_absa_client()
        try:
            response, provider, model = client.complete_raw(
                messages=[
                    {"role": "system", "content": PROMPT_ABSA_SYSTEM},
                    {"role": "user", "content": f"Analyze this financial text:\n\n{text_input}"},
                ],
                model_preference=["groq", "mistral", "cerebras"],
                temperature=0.0,
                max_tokens=1200,
            )
        except AllProvidersFailedError as e:
            logger.error("ABSA: tous les providers LLM ont echoue : %s", e)
            return {"aspects": []}
        logger.info(
            "ABSA: appel LLM OK (provider=%s, model=%s) sur %d caracteres",
            provider,
            model,
            len(text_input),
        )

        # Tracking coût LLM (lève BudgetExceededError si budget quotidien dépassé)
        try:
            from src.utils.llm_cost_tracker import track_from_openai_usage

            track_from_openai_usage(model, getattr(response, "usage", None))
        except Exception as _cost_exc:
            logger.debug("cost tracker : %s", _cost_exc)

        raw_text = (response.choices[0].message.content or "").strip()
        logger.debug("ABSA réponse brute: %s", raw_text[:400])

        # Extraction robuste du JSON (tolère le texte parasite autour)
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not json_match:
            logger.warning("ABSA: aucun JSON dans la réponse LLM.")
            return {"aspects": []}

        raw_result = json.loads(json_match.group())
        result = _validate_absa(raw_result)

        logger.info(
            "ABSA: %d aspect(s) validé(s) → %s", len(result["aspects"]), [a["aspect"] for a in result["aspects"]]
        )
        # Mémoïsation — on cache même les résultats vides pour éviter de retenter
        # un LLM sur un texte qui n'a pas retourné d'aspects.
        _absa_cache_put(cache_key, result)
        return result

    except Exception as e:
        logger.error("ABSA: erreur → %s", e)
        return {"aspects": []}


# ---------------------------------------------------------------------------
# Utilitaire : formatage lisible pour injection dans les prompts de débat
# ---------------------------------------------------------------------------


def format_absa_for_prompt(absa_result: dict) -> str:
    """
    Produit une représentation textuelle structurée de l'ABSA
    pour injection dans les prompts des agents de débat.
    """
    aspects = absa_result.get("aspects", [])
    if not aspects:
        return "No specific economic aspects detected in this article."

    lines = []
    for a in aspects:
        sentiment_icon = {"positive": "▲", "negative": "▼", "neutral": "◆"}.get(a["sentiment"], "?")
        lines.append(
            f"  {sentiment_icon} {a['aspect'].upper()} [{a['sentiment']}]\n"
            f'     Evidence : "{a["evidence"]}"\n'
            f"     Reason   : {a['reason']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    test_text = """
    Apple reported quarterly revenue of $124.3 billion, beating analyst expectations by 6%.
    EPS came in at $2.18, up 16% year-over-year. However, the CFO announced his resignation
    effective next quarter, citing personal reasons. Gross margin expanded to 45.9%, driven
    by strong services growth. The company raised full-year guidance above consensus estimates.
    Management warned of potential FX headwinds due to dollar strengthening.
    """

    result = run_absa(test_text)
    logger.info("\n=== ABSA Result ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    logger.info("\n=== Formatted for prompt ===")
    print(format_absa_for_prompt(result))
