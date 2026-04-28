"""
context_compressor.py — Compression de contexte à 3 niveaux
Inspiré de autoCompact.ts (Claude Code leak — Anthropic)

PROBLÈME RÉSOLU :
  Quand le texte envoyé aux agents débatteurs dépasse un seuil critique
  (article long + mémoire AutoDream + ABSA), le modèle souffre de "context rot" :
  il commence à négliger les informations au milieu de la fenêtre de contexte
  et ses réponses se dégradent.

SOLUTION — 3 niveaux de compression (fidèle à autoCompact.ts) :
  MICRO (< SEUIL_MICRO tokens) : pass-through, aucune modification
  AUTO  (SEUIL_MICRO-SEUIL_FULL) : compression de l'article uniquement, ABSA/mémoire préservés
  FULL  (> SEUIL_FULL tokens) : compression totale de l'entrée en bloc XML <summary>

MÉCANISME :
  Un sous-modèle léger et rapide (Cerebras llama3.1-8b, gratuit et quasi-instantané)
  génère un résumé encadré par des balises XML <summary>.
  Le texte d'origine est supprimé de la mémoire vive et remplacé par ce bloc —
  l'agent principal raisonne sur le résumé, pas sur le texte brut.

  C'est le "prompt système invisible" mentionné dans la fuite :
  l'agent de débat ne sait pas qu'il reçoit un résumé plutôt que l'original.
"""

import logging
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from src.utils.llm_client import AllProvidersFailedError, LLMClient

logger = logging.getLogger(__name__)

# Client LLM unifie (cf. ADR-003). Fallback Cerebras -> Groq -> Mistral pour la
# compression : on prefere les modeles rapides, l'ordre est inverse de l'ABSA.
_COMPRESSION_CLIENT: LLMClient | None = None


def _get_compression_llm_client() -> LLMClient:
    """Lazy-init du LLMClient partage pour la compression."""
    global _COMPRESSION_CLIENT  # noqa: PLW0603
    if _COMPRESSION_CLIENT is None:
        _COMPRESSION_CLIENT = LLMClient.from_env()
    return _COMPRESSION_CLIENT


# ---------------------------------------------------------------------------
# Seuils de compression (en tokens estimés)
# Estimation rapide : 1 token ≈ 4 caractères (heuristique conservatrice)
# ---------------------------------------------------------------------------

CHARS_PER_TOKEN = 4  # conservateur pour textes FR/EN mixtes

SEUIL_MICRO_TOKENS = 1_500  # < 1500 tokens → pas de compression (~6 000 chars)
SEUIL_FULL_TOKENS = 4_000  # > 4000 tokens → compression totale (~16 000 chars)

SEUIL_MICRO_CHARS = SEUIL_MICRO_TOKENS * CHARS_PER_TOKEN  # 6 000
SEUIL_FULL_CHARS = SEUIL_FULL_TOKENS * CHARS_PER_TOKEN  # 16 000


class CompressionLevel(Enum):
    MICRO = "micro"  # Pas de compression — texte en dessous du seuil
    AUTO = "auto"  # Compression partielle — seul l'article est compressé
    FULL = "full"  # Compression totale — tout le contexte résumé


@dataclass
class CompressionResult:
    """Résultat retourné par le compresseur."""

    text: str  # Texte compressé (ou original si MICRO)
    level: CompressionLevel  # Niveau appliqué
    original_chars: int  # Taille originale
    compressed_chars: int  # Taille après compression
    compression_ratio: float  # original / compressed
    model_used: Optional[str] = None  # Modèle utilisé pour la compression


# ---------------------------------------------------------------------------
# Estimation de tokens (pas besoin de tiktoken — heuristique suffisante)
# ---------------------------------------------------------------------------


def estimate_tokens(text: str) -> int:
    """Estimation rapide du nombre de tokens (1 token ≈ 4 chars)."""
    return len(text) // CHARS_PER_TOKEN


def _decide_level(text: str) -> CompressionLevel:
    """Détermine le niveau de compression selon la taille du texte."""
    n = len(text)
    if n <= SEUIL_MICRO_CHARS:
        return CompressionLevel.MICRO
    elif n <= SEUIL_FULL_CHARS:
        return CompressionLevel.AUTO
    else:
        return CompressionLevel.FULL


# ---------------------------------------------------------------------------
# Client LLM de compression — toujours le plus léger disponible
# La compression doit être rapide et peu coûteuse.
# Priorité : Cerebras (sub-seconde) → Groq → Mistral
# ---------------------------------------------------------------------------

# `_get_compression_client()` retire — remplace par `_get_compression_llm_client()`
# en haut du module. Ordre de fallback : Cerebras (sub-seconde) -> Groq -> Mistral.
COMPRESSION_PREFERENCE = ["cerebras", "groq", "mistral"]


# ---------------------------------------------------------------------------
# Prompts de compression — invisibles pour les agents principaux
# ---------------------------------------------------------------------------

# Niveau AUTO : compression de l'article uniquement
PROMPT_COMPRESS_AUTO = """Tu es un compresseur de texte financier.
Tu reçois un article de presse financier. Tu dois produire un résumé FACTUEL
encadré par des balises XML <summary>.

RÈGLES STRICTES :
1. Conserve TOUS les chiffres, dates, noms de personnes, montants en dollars/euros
2. Conserve les citations importantes (entre guillemets dans le texte original)
3. Supprime : phrases de style journalistique, redondances, contexte général
4. Longueur cible : 30-50% du texte original
5. Réponds UNIQUEMENT avec le bloc XML — pas de texte avant ou après

FORMAT OBLIGATOIRE :
<summary compression="auto" type="financial_article">
  <key_facts>
    [liste de faits factuels avec chiffres — 1 fait par ligne]
  </key_facts>
  <financial_figures>
    [tous les chiffres financiers mentionnés : revenus, EPS, guidance, etc.]
  </financial_figures>
  <critical_quote>[citation la plus importante ou N/A]</critical_quote>
  <temporal_context>[date(s) mentionnée(s) ou contexte temporel]</temporal_context>
</summary>"""


# Niveau FULL : compression totale de tout le contexte
PROMPT_COMPRESS_FULL = """Tu es un compresseur de contexte multi-source pour analyse financière.
Tu reçois un bloc de contexte complet : mémoire historique + article + données marché.
Tu dois tout compresser en un seul bloc XML <summary> très compact.

RÈGLES STRICTES :
1. Préserve : tous les signaux financiers passés (Achat/Vente/Neutre + dates ISO)
2. Préserve : tous les chiffres et données marché
3. Préserve : l'argument dominant de chaque décision passée (1 phrase max)
4. Supprime : tout texte narratif, explications, contexte général
5. Longueur cible : 20-30% du contexte original
6. Réponds UNIQUEMENT avec le bloc XML

FORMAT OBLIGATOIRE :
<summary compression="full" type="full_context">
  <historical_signals>
    [DATE ISO] [TICKER]: [Signal] | Force:[0.0-1.0] | [Argument 1 phrase]
  </historical_signals>
  <article_facts>
    [Faits clés de l'article — 3-5 points avec chiffres]
  </article_facts>
  <market_data>
    [Prix, volume, variation — données numériques uniquement]
  </market_data>
  <absa_summary>
    [Aspects ABSA détectés — format: aspect(sentiment)]
  </absa_summary>
</summary>"""


# ---------------------------------------------------------------------------
# Fonctions de compression par niveau
# ---------------------------------------------------------------------------


def _compress_auto(article_text: str, client: LLMClient) -> str:
    """Niveau AUTO : compresse uniquement l'article (memoire et ABSA inchanges)."""
    try:
        response = client.complete(
            messages=[
                {"role": "system", "content": PROMPT_COMPRESS_AUTO},
                {"role": "user", "content": f"Compresse cet article financier :\n\n{article_text}"},
            ],
            model_preference=COMPRESSION_PREFERENCE,
            temperature=0.0,
            max_tokens=800,
        )
        compressed = response.content.strip()
        if "<summary" not in compressed:
            logger.warning("[Compresseur] Niveau AUTO : XML absent, fallback sur texte brut.")
            return article_text
        return compressed
    except (AllProvidersFailedError, Exception) as e:
        logger.error("[Compresseur] Erreur niveau AUTO : %s", e)
        return article_text


def _compress_full(full_context: str, client: LLMClient) -> str:
    """Niveau FULL : compresse l'integralite du contexte (memoire + article + marche)."""
    try:
        response = client.complete(
            messages=[
                {"role": "system", "content": PROMPT_COMPRESS_FULL},
                {"role": "user", "content": f"Compresse ce contexte complet :\n\n{full_context}"},
            ],
            model_preference=COMPRESSION_PREFERENCE,
            temperature=0.0,
            max_tokens=1000,
        )
        compressed = response.content.strip()
        if "<summary" not in compressed:
            logger.warning("[Compresseur] Niveau FULL : XML absent, fallback sur texte brut.")
            return full_context
        return compressed
    except (AllProvidersFailedError, Exception) as e:
        logger.error("[Compresseur] Erreur niveau FULL : %s", e)
        return full_context


# ---------------------------------------------------------------------------
# Point d'entrée principal — compress_article_if_needed()
# Utilisé dans agent_debat.py avant le lancement du débat
# ---------------------------------------------------------------------------


def compress_article_if_needed(
    texte_article: str,
    ticker: str,
) -> CompressionResult:
    """
    Compresse l'article (qui peut contenir la mémoire AutoDream préfixée)
    si sa taille dépasse les seuils définis.

    C'est le point d'entrée unique pour la compression dans agent_debat.py.
    La partie ABSA et contexte marché sont gérées séparément dans le pipeline
    et ne transitent PAS par cette fonction.

    Args:
        texte_article : le texte envoyé au débat (peut inclure mémoire + article)
        ticker        : pour les logs

    Returns:
        CompressionResult avec le texte compressé (ou original si MICRO)
    """
    original_chars = len(texte_article)
    level = _decide_level(texte_article)

    logger.info(
        "[Compresseur] %s -- %d chars (~%d tokens) -> Niveau : %s",
        ticker,
        original_chars,
        estimate_tokens(texte_article),
        level.value.upper(),
    )

    if level == CompressionLevel.MICRO:
        # Pas de compression — texte sous le seuil
        return CompressionResult(
            text=texte_article,
            level=level,
            original_chars=original_chars,
            compressed_chars=original_chars,
            compression_ratio=1.0,
            model_used=None,
        )

    # On a besoin d'un LLM de compression
    client = _get_compression_llm_client()
    if not client.available_providers():
        logger.warning("[Compresseur] Aucun provider LLM dispo, pas de compression.")
        return CompressionResult(
            text=texte_article,
            level=CompressionLevel.MICRO,
            original_chars=original_chars,
            compressed_chars=original_chars,
            compression_ratio=1.0,
        )

    if level == CompressionLevel.AUTO:
        # Compression de l'article uniquement
        # On cherche à séparer la partie "mémoire" (préfixée par AutoDream) de l'article
        if "=== MÉMOIRE HISTORIQUE" in texte_article:
            # Séparer mémoire (déjà structurée, on la garde) et article brut
            split_marker = "===========================================\n"
            parts = texte_article.split(split_marker, 1)
            memory_part = parts[0] + split_marker if len(parts) > 1 else ""
            article_part = parts[1] if len(parts) > 1 else texte_article
        else:
            memory_part = ""
            article_part = texte_article

        compressed_article = _compress_auto(article_part, client)
        final_text = memory_part + compressed_article

    else:  # FULL
        final_text = _compress_full(texte_article, client)

    compressed_chars = len(final_text)
    ratio = original_chars / max(compressed_chars, 1)

    # `model_used` : le LLMClient choisit dynamiquement (cf. logs llm_complete_ok).
    model_label = f"LLMClient(chain={COMPRESSION_PREFERENCE})"
    logger.info(
        "[Compresseur] %s — %s : %d chars → %d chars (ratio %.1f×) | %s",
        ticker,
        level.value.upper(),
        original_chars,
        compressed_chars,
        ratio,
        model_label,
    )

    return CompressionResult(
        text=final_text,
        level=level,
        original_chars=original_chars,
        compressed_chars=compressed_chars,
        compression_ratio=ratio,
        model_used=model_label,
    )


# ---------------------------------------------------------------------------
# Utilitaire : extraction du texte depuis un bloc <summary> existant
# Utilisé pour inspecter ou logger le contenu compressé
# ---------------------------------------------------------------------------


def extract_summary_text(compressed_text: str) -> str:
    """
    Extrait le contenu textuel brut d'un bloc <summary> XML.
    Utile pour les logs ou le dashboard.
    """
    match = re.search(r"<summary[^>]*>(.*?)</summary>", compressed_text, re.DOTALL)
    if match:
        # Nettoyage des balises XML internes
        inner = match.group(1)
        inner = re.sub(r"<[^>]+>", "", inner)
        return inner.strip()
    return compressed_text
