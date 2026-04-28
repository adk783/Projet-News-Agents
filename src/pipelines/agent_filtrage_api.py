"""Agent de filtrage par API LLM (pertinence financiere binaire YES/NO).

Migration ADR-003 : utilise LLMClient avec fallback Groq -> Mistral -> Cerebras
au lieu de l'appel `requests.post()` direct vers Groq uniquement. Si Groq tombe,
le filtre bascule sur Mistral puis Cerebras pour rester operationnel.
"""

from __future__ import annotations

import logging

from src.utils.llm_client import AllProvidersFailedError, LLMClient

logger = logging.getLogger("AgentFiltrageAPI")
logger.setLevel(logging.DEBUG)

MAX_CHARS = 1000

# Client LLM partage (lazy-init).
_FILTER_CLIENT: LLMClient | None = None


def _get_filter_client() -> LLMClient:
    """Lazy-init du LLMClient partage pour le filtrage de pertinence."""
    global _FILTER_CLIENT  # noqa: PLW0603
    if _FILTER_CLIENT is None:
        _FILTER_CLIENT = LLMClient.from_env()
    return _FILTER_CLIENT


def build_prompt(ticker: str, company_name: str, title: str, content: str) -> str:
    """Genere le prompt YES/NO pour le filtre de pertinence financiere."""
    return f"""You are a financial news filter. Decide if a news article is relevant to a specific company's stock.

Answer only YES or NO. Nothing else.

Rules:
- YES if the company name or ticker appears in the TITLE — this is the strongest signal, always prioritize it
- YES if the company is the main subject OR one of the main subjects of the article
- YES if the article discusses the company's products, strategy, earnings, stock, or executives
- YES if the article compares the company directly to a competitor
- NO only if the company is briefly mentioned in passing in an article about something else entirely
- NO if the article is about general market indexes, unrelated sectors, or other companies with no link to this one
- Do NOT be confused by broad topics like AI, markets, or macro trends in the content — if the company name is in the title, answer YES

Now decide:
Ticker: {ticker} ({company_name})
Title: {title}
Content preview: {content[:MAX_CHARS]}

Answer (YES or NO):"""


def est_pertinent(ticker: str, company_name: str, title: str, content: str) -> tuple[bool, str]:
    """Filtre la pertinence financiere d'un article via LLM.

    Returns
    -------
    (is_relevant, reason_code)
        is_relevant : True si l'article concerne le ticker
        reason_code : code court pour audit (ia_pertinent, ia_hors_sujet, ia_erreur, ...)
    """
    try:
        prompt = build_prompt(ticker, company_name, title, content)
        client = _get_filter_client()
        try:
            response = client.complete(
                messages=[{"role": "user", "content": prompt}],
                model_preference=["groq", "mistral", "cerebras"],
                temperature=0.1,
                max_tokens=10,  # YES ou NO suffit
            )
        except AllProvidersFailedError as e:
            logger.warning("[Filtrage IA API] Tous providers KO (%s) -> laisse passer : %s", e, title)
            return True, "ia_erreur"

        reponse = response.content.strip().upper()
        if reponse.startswith("YES"):
            logger.debug("[Filtrage IA API] PERTINENT : %s", title)
            return True, "ia_pertinent"
        if reponse.startswith("NO"):
            logger.info("[Filtrage IA API] HORS SUJET : %s", title)
            return False, "ia_hors_sujet"
        logger.warning(
            "[Filtrage IA API] Reponse inattendue '%s' pour : %s -> laisse passer",
            reponse,
            title,
        )
        return True, "ia_reponse_inattendue"

    except Exception as e:  # noqa: BLE001 - filet de securite : on ne fait JAMAIS crasher la pipeline
        logger.warning("[Filtrage IA API] Erreur (%s) -> laisse passer : %s", e, title)
        return True, "ia_erreur"
