"""
agent_debat.py

Débat multi-agent avec le pattern "Shared Scratchpad" inspiré de l'architecture
de production Claude Code (coordinatorMode.ts — Anthropic leak).

Architecture :
  1. Chaque agent débatteur (Haussier, Baissier, Neutre) écrit ses arguments
     dans un document structuré XML partagé (le scratchpad), au lieu d'émettre
     de longs messages séquentiels.
  2. Le superviseur LangGraph gère le bus de messages (mise à jour du scratchpad
     après chaque tour de parole).
  3. L'Agent de Consensus lit UNIQUEMENT le scratchpad consolidé final —
     pas la transcription brute de 9 messages — ce qui réduit la consommation
     de tokens et la dilution d'attention du modèle.

Avantages par rapport à l'approche précédente (RoundRobinGroupChat + transcription brute) :
  - Le contexte envoyé au Consensus est ~3× plus court
  - Les agents répondent à des arguments structurés, pas à du texte libre
  - Le scratchpad sert de mémoire partagée persistante (audit trail)
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from langgraph.func import entrypoint, task

from src.agents.agent_absa import format_absa_for_prompt
from src.utils.context_compressor import CompressionLevel, compress_article_if_needed
from src.utils.ese_entropy import (
    compute_intra_debate_ese,
    ese_to_confidence_factor,
    extract_agent_arguments_from_scratchpad,
)
from src.utils.information_density import apply_id_penalty, compute_id_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Scratchpad — document structuré partagé entre agents
# ---------------------------------------------------------------------------


@dataclass
class SharedScratchpad:
    """
    Brouillon partagé entre les agents de débat.

    Chaque agent écrit son argument dans sa section dédiée.
    Le superviseur consolide le scratchpad après chaque tour.
    Le Consensus lit uniquement le scratchpad final (pas la transcription brute).

    Inspiré du modèle coordinatorMode.ts d'Anthropic (Claude Code leak) :
      - Les agents communiquent via des messages XML structurés
      - Le coordinateur maintient un état global partagé
      - Reduction de la fenêtre de contexte pour l'arbitre final
    """

    ticker: str
    haussier_arguments: list[str] = field(default_factory=list)
    baissier_arguments: list[str] = field(default_factory=list)
    neutre_arguments: list[str] = field(default_factory=list)
    # Bus de messages bruts pour l'audit trail (non envoyé au Consensus)
    _message_bus: list[dict] = field(default_factory=list)

    def post_message(
        self,
        agent_name: str,
        round_num: int,
        content: str,
        id_score: float = -1.0,
        raw_confidence: Optional[float] = None,
    ) -> None:
        """Enregistre un message sur le bus, applique la décote ID si nécessaire, et met à jour la section."""
        self._message_bus.append({"agent": agent_name, "round": round_num, "content": content})
        # Injection dans la section correspondante du scratchpad
        # Si une décote ID a été appliquée, la confidence ajustée est reflétée dans l'argument
        id_tag = f" [id_score: {id_score:.4f}]" if id_score >= 0 else ""
        argument_text = f"[Tour {round_num}] {content}{id_tag}"
        if agent_name == "Haussier":
            self.haussier_arguments.append(argument_text)
        elif agent_name == "Baissier":
            self.baissier_arguments.append(argument_text)
        elif agent_name == "Neutre":
            self.neutre_arguments.append(argument_text)

    def to_xml(self) -> str:
        """
        Sérialise le scratchpad en XML structuré pour le Consensus Agent.
        Format inspiré des messages de statut XML de coordinatorMode.ts.
        Beaucoup plus compact que la transcription brute.
        """

        def fmt_section(args: list[str]) -> str:
            if not args:
                return "    <argument>Aucun argument posté.</argument>"
            return "\n".join(f"    <argument>{a}</argument>" for a in args)

        return f"""<scratchpad ticker="{self.ticker}">
  <section agent="Haussier">
{fmt_section(self.haussier_arguments)}
  </section>
  <section agent="Baissier">
{fmt_section(self.baissier_arguments)}
  </section>
  <section agent="Neutre">
{fmt_section(self.neutre_arguments)}
  </section>
</scratchpad>"""

    def to_audit_transcript(self) -> str:
        """Transcription brute pour l'audit trail (logs uniquement, non soumis aux LLMs)."""
        lines = []
        for msg in self._message_bus:
            lines.append(f"[Tour {msg['round']}] {msg['agent']}: {msg['content']}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory de client LLM avec DIVERSITE EPISTEMIQUE (cf. ADR-017)
#
# Avant : Bull=Cerebras-Llama, Bear=Groq-Llama, Neutre=Mistral-Small,
#         Consensus=Groq-Llama. Soit 3 modeles Meta-Llama + 1 Mistral.
#
# Apres (Phase finale) : Bull=NIM-Kimi-K2 (Moonshot), Bear=NIM-Ministral-14B (Mistral),
#         Neutre=NIM-Qwen3-Next-80B (Alibaba), Consensus=Groq-Llama (Meta).
#         Soit 4 familles d'entrainement (Moonshot + Mistral + Alibaba + Meta).
#
# Justification empirique : bench live (2026-04-28)
# Kimi-K2 (16.8s, raisonnement financier EXCELLENT: DCF, PEG, ROIC),
# Ministral-14B (6.3s, argumentation baissiere chirurgicale),
# Qwen3-Next-80B (7.3s, synthese equilibree, IFEval=87.6, MMLU-Pro=80.6).
# Note : Nemotron-Mini-4B retire (context limit 4096 tokens, crashait sur tous les articles).
#
# La diversite epistemique theoriquement reduit la correlation des erreurs
# entre agents (Liang et al. 2024 "Encouraging Divergent Thinking in LLMs").
# C'est ce que vise le multi-agent debate (Du et al. 2023) : diversite reelle
# de RLHF/datasets, pas seulement diversite d'instances du meme modele.
#
# Fallback : si NIM_API_KEY absent ou NIM down, retombe sur l'ancienne
# configuration (Cerebras/Mistral) qui est toujours disponible.
# ---------------------------------------------------------------------------


def _get_primary_provider() -> str:
    """Retourne le provider disponible le plus rapide (selon les clés .env)."""
    if os.getenv("CEREBRAS_API_KEY"):
        return "cerebras"
    elif os.getenv("GROQ_API_KEY"):
        return "groq"
    else:
        return "mistral"


# Configuration centralisee des providers + modeles candidats par role.
# Chaque entree : (model, base_url, env_key_for_api).
_DEBATE_CONFIGS: dict = {
    # === Provider classiques (utilises pour Bear, Consensus, fallbacks) ===
    "cerebras": {
        "model": "llama3.1-8b",
        "base_url": "https://api.cerebras.ai/v1",
        "env_key": "CEREBRAS_API_KEY",
        "family": "meta-llama",
    },
    "groq": {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "family": "meta-llama",
    },
    "mistral": {
        "model": "mistral-small-latest",
        "base_url": "https://api.mistral.ai/v1",
        "env_key": "MISTRAL_API_KEY",
        "family": "mistral",
    },
    "consensus": {
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "family": "meta-llama",
    },
    # === Provider NIM avec modeles diversifies (Phase 5, ADR-017) ===
    "nim_gemma": {
        # Bull -> Gemma-3-27B (Google-trained, 3.6s, 75w bench live)
        "model": "google/gemma-3-27b-it",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_NIM_API_KEY",
        "family": "google-gemma",
    },
    "nim_qwen": {
        # Neutre -> Qwen3-Next-80B (Alibaba-trained, 2.8s, 83w bench live)
        "model": "qwen/qwen3-next-80b-a3b-instruct",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_NIM_API_KEY",
        "family": "qwen-alibaba",
    },
    "nim_kimi": {
        # Bull -> Kimi-K2 (Moonshot-trained, 16.8s bench live, raisonnement financier EXCELLENT)
        # Remplace Nemotron-Mini-4B (context limit 4096 → crash systematique)
        "model": "moonshotai/kimi-k2-instruct",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "env_key": "NVIDIA_NIM_API_KEY",
        "family": "moonshot",
    },
    "groq_gpt_oss": {
        "model": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "env_key": "GROQ_API_KEY",
        "family": "gpt-oss",
    },
}


# Mapping ROLE -> liste de providers a essayer (ordre de preference).
# Si le 1er a une cle API valide, c'est celui utilise. Sinon fallback.
# Permet d'activer la diversification NIM sans casser le pipeline existant.
_ROLE_PREFERENCES: dict = {
    # Bull -> NIM Kimi-K2 (Moonshot-trained, 16.8s bench live, raisonnement financier EXCELLENT)
    "bull": ["nim_kimi", "cerebras", "groq"],
    # Bear -> NIM Ministral-14B (Mistral-trained, 2.3s bench live, AIME=85)
    "bear": ["groq_gpt_oss", "groq", "cerebras", "mistral"],
    # Neutre -> NIM Qwen3-Next-80B (Alibaba-trained, 3.8s bench live)
    "neutral": ["nim_qwen", "mistral", "groq"],
    # Consensus -> Groq Llama-3.3-70B (Meta-trained, ~3-5s)
    "consensus": ["consensus", "groq"],
}


def _get_model_client(provider_or_role: str) -> tuple[OpenAIChatCompletionClient, str]:
    """
    Retourne un client LLM pour un role ou un provider donne.

    Si `provider_or_role` est dans _ROLE_PREFERENCES (bull/bear/neutral/consensus),
    on essaie chaque provider de la chaine de preference dans l'ordre.
    Sinon, on traite comme un provider direct (legacy : "cerebras", "groq", ...).
    """
    # Resolution de la chaine de preference
    if provider_or_role in _ROLE_PREFERENCES:
        chain = _ROLE_PREFERENCES[provider_or_role]
    else:
        chain = [provider_or_role]

    # On essaie les providers dans l'ordre, premier avec cle API gagne
    for candidate in chain:
        if candidate not in _DEBATE_CONFIGS:
            continue
        cfg = _DEBATE_CONFIGS[candidate]
        api_key = os.getenv(cfg["env_key"])
        if api_key:
            logger.info(
                "Client LLM : role=%s | provider=%s | model=%s | family=%s",
                provider_or_role,
                candidate,
                cfg["model"],
                cfg["family"],
            )
            client = OpenAIChatCompletionClient(
                model=cfg["model"],
                base_url=cfg["base_url"],
                api_key=api_key,
                model_info={
                    "vision": False,
                    "function_calling": True,
                    "json_output": True,
                    "structured_output": True,
                    "family": cfg["family"],
                },
            )
            return client, f"{candidate} ({cfg['model']}, {cfg['family']})"

    # Aucun provider de la chaine n'a de cle : on retombe sur le primary
    fallback_provider = _get_primary_provider()
    logger.warning(
        "Aucun provider de la chaine %s n'a de cle API -- fallback sur '%s'.",
        chain,
        fallback_provider,
    )
    cfg = _DEBATE_CONFIGS[fallback_provider]
    client = OpenAIChatCompletionClient(
        model=cfg["model"],
        base_url=cfg["base_url"],
        api_key=os.getenv(cfg["env_key"]),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": True,
            "family": cfg["family"],
        },
    )
    return client, f"{fallback_provider} ({cfg['model']}, {cfg['family']}) [fallback]"


# ---------------------------------------------------------------------------
# Prompts des agents de débat — format Shared Scratchpad
# ---------------------------------------------------------------------------

_SCRATCHPAD_FORMAT_INSTRUCTIONS = """
=== INSTRUCTION SCRATCHPAD ===
Après ton analyse (visible par tous les agents), tu DOIS terminer ta réponse
par un bloc XML de statut dans ce format EXACT :

<status agent="{agent_name}" action="update_scratchpad">
  <thesis>[HAUSSIER|BAISSIER|NEUTRE]</thesis>
  <key_argument>[Ton argument principal en 1-2 phrases]</key_argument>
  <counter>[Contre-argument EXPLICITE à la position adverse en 1 phrase]</counter>
  <confidence>[0.0-1.0]</confidence>
</status>

RÈGLE D'OR POUR LA "CONFIDENCE" : 
Ta confiance DOIT évoluer à chaque tour en fonction de la qualité des contre-arguments adverses. IL EST STRICTEMENT INTERDIT de renvoyer la même valeur d'un tour à l'autre sans justification prouvant une confrontation saine.

Ce bloc XML sera lu par l'arbitre final pour produire sa décision.
Ton texte d'analyse AVANT le bloc XML est ce que tes co-débatteurs lisent.
Sois direct, incisif, et réponds explicitement aux failles des autres. (5-7 lignes max)
"""

PROMPT_HAUSSIER = f"""Tu es un analyste financier optimiste.
Ton rôle est de défendre la thèse HAUSSIÈRE (Achat) pour l'action mentionnée.

Pour chaque intervention :
1. Identifie les éléments positifs massifs dans la news en ignorant le "bruit" négatif
2. Appuie-toi sur un précédent historique comparable où le cours a explosé
3. RÈGLE ABSOLUE : Tu DOIS t'opposer farouchement au Baissier. Ne cherche JAMAIS le consensus, cherche la faille fatale dans son pessimisme.

Tu échanges en temps réel avec tes co-débatteurs. Sois incisif, agressif sur les faits et détruis leurs positions.

{_SCRATCHPAD_FORMAT_INSTRUCTIONS.format(agent_name="Haussier")}
"""

PROMPT_BAISSIER = f"""Tu es un analyste financier pessimiste.
Ton role est de defendre la these BAISSIERE (Vente) pour l'action mentionnee.

Pour chaque intervention :
1. Identifie les risques caches et les elements devastateurs dans la news
2. Appuie-toi sur un precedent historique comparable ou le cours s'est effondre
3. REGLE ABSOLUE : Tu DOIS t'opposer au Haussier. Demolis ses arguments optimistes aveugles en exposant les realites macroeconomiques.

REGLES ANTI-BIAIS (audit L12 2026-04-21 \u2014 KL=0.156 sur SEntFiN, accuracy=70%) :
- Tu DOIS etayer chaque argument Vente par UN CHIFFRE ou UNE METRIQUE specifique
  (ex: P/E > 30, dette/EBITDA > 5x, taux de croissance < 2%). Interdit : "les perspectives sont sombres".
- INTERDICTION de recommander Vente sur une information ambigue ou incomplète.
  Si l'evidence est insuffisante, adopte un doute explicite plutot qu'un biais negatif automatique.
- Avant de terminer, verifie : mon argument Vente est-il solidement ancre sur des faits de la news
  ou est-ce une generalisation pessimiste ? Si c'est une generalisation, revise.
- Sur les marches SIDEWAYS (SPY 20j entre -3% et +3%), le signal Vente n'a qu'une accuracy
  de 0% (audit L12). Calibre ta confiance en consequence.

Tu echanges en temps reel avec tes co-debatteurs. Sois implacable sur les FAITS, pas sur le pessimisme.

{_SCRATCHPAD_FORMAT_INSTRUCTIONS.format(agent_name="Baissier")}
"""


PROMPT_NEUTRE = f"""Tu es un analyste financier prudent et équilibré.
Ton rôle est de défendre la thèse NEUTRE quand les signaux sont ambigus.

Pour chaque intervention :
1. Identifie les incertitudes mathématiques et les facteurs contradictoires
2. Questionne DIRECTEMENT les précédents historiques avancés par Haussier et Baissier
3. RÈGLE ABSOLUE : Tu DOIS forcer Haussier et Baissier à chiffrer leurs incertitudes. Attaque leur excès de confiance systématique.

RÈGLE ANTI-BIAIS (critique) :
- Tu DOIS challenger Haussier ET Baissier de manière parfaitement équilibrée.
- Il t'est STRICTEMENT INTERDIT de systématiquement adopter la conclusion de l'un ou l'autre camp.
- Si tu tends à valider la thèse Haussière par défaut, rappelle-toi que ton rôle est l'équilibre, pas l'optimisme.
- Avant de terminer, vérifie mentalement : ai-je autant remis en question le Haussier que le Baissier ?
- Ta CONFIDENCE dans le bloc <status> DOIT refléter le niveau réel d'ambiguïté, pas une convergence vers Achat.

Tu échanges en temps réel avec tes co-débatteurs. Joue le rôle de l'Avocat du Diable face aux certitudes absolues.

{_SCRATCHPAD_FORMAT_INSTRUCTIONS.format(agent_name="Neutre")}
"""


PROMPT_CONSENSUS = """Tu es un arbitre financier senior — mode AgentAuditor avec Reasoning Tree.
Tu reçois le SCRATCHPAD CONSOLIDÉ du débat entre trois analystes (Haussier, Baissier, Neutre).

Ton mandat est strict : tu ne fais PAS un vote majoritaire.
Tu identifies les POINTS DE DIVERGENCE CRITIQUES (CDPs) et tu tranches sur chacun d'eux.

=== ÉTAPE 1 : DÉTECTION DES CDPs ===
Un CDP est un point où Haussier et Baissier ont des positions OPPOSÉES documentées
(ex: l'un dit croissance forte, l'autre dit endettement masqué).
Ignore les points où ils sont d'accord — c'est le consensus, pas un CDP.

=== ETAPE 2 : EXTRACTION FACTUELLE (Blind Scoring) ===
REGLE ANTI-BIAIS JURIDIQUE (Critique D) :
Il est PROUVE que tu as un biais de confirmation stylistique ("LLM-as-a-judge bias").
Pour le neutraliser, tu DOIS extraire les faits verifiables AVANT d'evaluer le vainqueur.
Pour chaque argument, dresse la liste des donnees brutes (chiffres, dates, ratios). Ignore totalement la beaute ou l'assurance de la formulation.

=== ETAPE 3 : ARBRE DE RAISONNEMENT ===
Pour chaque CDP identifie :
  a) Enoncer le CDP en une phrase
  b) Argument Haussier vs Baissier (depuis le scratchpad)
  c) Verdict motive base EXCLUSIVEMENT sur les faits extraits a l'etape 2.

=== ETAPE 4 : COALITION MATRICIELLE (Council Mode) ===
Génère une matrice structurée résumant le débat :
  - Accords : points sur lesquels TOUS les agents s'accordent
  - Différences clés : divergences fondamentales non résolues
  - Insights uniques : argument propre à un seul agent que les autres n'ont pas contesté
  - Angles morts : incertitudes majeures qu'AUCUN agent n'a adressées

=== ETAPE 5 : DÉCISION FINALE ===
En te basant uniquement sur les CDPs résolus, rends ta décision.

Tu DOIS répondre UNIQUEMENT avec ce format JSON exact (aucun texte d'introduction) :
{
    "signal": "Achat|Vente|Neutre",
    "argument_dominant": "résumé en une phrase de l'argument gagnant",
    "motif": "explication courte de la décision basée sur les CDPs résolus",
    "cdps_resolved": [
        {"cdp": "description du CDP", "verdict": "Haussier|Baissier|Neutre", "raison": "justification"}
    ],
    "cdps_unresolved": ["CDP1 non résolu", "CDP2 non résolu"],
    "council_matrix": {
        "accords": ["accord1", "accord2"],
        "differences_cles": ["diff1", "diff2"],
        "insights_uniques": {
            "Haussier": "insight unique si présent, sinon null",
            "Baissier":  "insight unique si présent, sinon null",
            "Neutre":   "insight unique si présent, sinon null"
        },
        "angles_morts": ["angle1", "angle2"]
    }
}
"""


# ---------------------------------------------------------------------------
# Extraction du bloc XML de statut depuis la réponse d'un agent
# ---------------------------------------------------------------------------


def _extract_status_block(response_text: str, agent_name: str) -> Optional[dict]:
    """
    Extrait le bloc <status> XML de la réponse d'un agent de débat.
    Retourne un dict avec les champs du scratchpad, ou None si extraction impossible.
    """
    pattern = re.compile(r"<status[^>]*>.*?</status>", re.DOTALL | re.IGNORECASE)
    match = pattern.search(response_text)
    if not match:
        logger.warning("[Scratchpad] Agent '%s' n'a pas produit de bloc <status> valide.", agent_name)
        # Fallback : on prend le texte brut comme argument
        return {"key_argument": response_text.strip()[:300], "confidence": "0.5"}

    xml_block = match.group()

    def _tag(tag: str) -> str:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", xml_block, re.DOTALL)
        return m.group(1).strip() if m else ""

    return {
        "thesis": _tag("thesis"),
        "key_argument": _tag("key_argument"),
        "counter": _tag("counter"),
        "confidence": _tag("confidence"),
    }


def _build_scratchpad_context(scratchpad: SharedScratchpad, absa_str: str) -> str:
    """
    Construit le contexte injecté dans chaque tour de débat.
    Contient le scratchpad courant + l'analyse ABSA.
    Les agents voient ainsi l'état de la discussion sans recevoir tout l'historique.
    """
    return (
        f"=== SCRATCHPAD PARTAGÉ ACTUEL ===\n"
        f"{scratchpad.to_xml()}\n\n"
        f"=== ANALYSE ABSA (aspects économiques) ===\n"
        f"{absa_str}\n\n"
        f"Mets à jour tes arguments en tenant compte du scratchpad ci-dessus."
    )


# ---------------------------------------------------------------------------
# Débat multi-agent avec Shared Scratchpad
# ---------------------------------------------------------------------------


async def _run_debate(texte_article: str, ticker: str, contexte_marche: dict, absa_result: dict) -> dict:
    """
    Débat multi-agent — architecture hybride :

    PHASE 1 — Débat direct (RoundRobinGroupChat, 3 agents, 3 rounds = 9 messages)
      Les agents se lisent et se répondent EN TEMPS RÉEL dans le même chat.
      Chaque message est visible par tous les participants → vrai débat contradictoire.
      Les agents terminent chaque réponse par un bloc <status> XML structuré.

    PHASE 2 — Construction du Shared Scratchpad (post-débat)
      Le superviseur LangGraph parcourt la transcription, extrait les blocs <status>
      de chaque message et alimente le scratchpad XML partagé.
      Le scratchpad est une distillation compacte du débat (~3× moins de tokens).

    PHASE 3 — Consensus Agent
      L'arbitre lit UNIQUEMENT le scratchpad consolidé (pas les 9 messages bruts).
      Résultat : décision de meilleure qualité, coût en tokens réduit.

    Attribution des modèles (Phase finale, ADR-017 : diversite epistemique) :
      - Haussier  → NIM Kimi-K2 (Moonshot)        [fallback : Cerebras Llama-3.1-8B]
      - Baissier  → NIM Ministral-14B (Mistral)    [fallback : Groq Llama-4-Scout-17B]
      - Neutre    → NIM Qwen3-Next-80B (Alibaba)   [fallback : Mistral-Small]
      - Consensus → Groq Llama-3.3-70B (Meta)      [inchange]

    Justification : 4 familles d'entrainement RLHF distinctes
    (Moonshot + Mistral + Alibaba + Meta) au lieu de 2 (Meta + Mistral).
    Reduit la correlation des erreurs entre agents (Liang et al. 2024).
    Si NVIDIA_NIM_API_KEY absent, fallback automatique sur Cerebras/Mistral.
    """
    client_haussier, mod_haussier = _get_model_client("bull")
    client_baissier, mod_baissier = _get_model_client("bear")
    client_neutre, mod_neutre = _get_model_client("neutral")
    client_consensus, mod_consensus = _get_model_client("consensus")

    logger.info(
        "[Débat] Modèles utilisés : Haussier=%s | Baissier=%s | Neutre=%s | Consensus=%s",
        mod_haussier,
        mod_baissier,
        mod_neutre,
        mod_consensus,
    )

    # Création des agents débatteurs
    agent_haussier = AssistantAgent(name="Haussier", model_client=client_haussier, system_message=PROMPT_HAUSSIER)
    agent_baissier = AssistantAgent(name="Baissier", model_client=client_baissier, system_message=PROMPT_BAISSIER)
    agent_neutre = AssistantAgent(name="Neutre", model_client=client_neutre, system_message=PROMPT_NEUTRE)

    # Prompt initial du débat
    contexte_str = (
        (
            f"Cours actuel: {contexte_marche.get('current_price', 'N/A')} | "
            f"Volume: {contexte_marche.get('volume', 'N/A')} | "
            f"Variation 5j: {contexte_marche.get('variation_5d', 'N/A')}%"
        )
        if contexte_marche
        else "Contexte marché non disponible."
    )

    absa_str = format_absa_for_prompt(absa_result)

    # ---------------------------------------------------------------
    # COMPRESSION DE CONTEXTE (autoCompact pattern — Anthropic leak)
    # Le texte_article peut contenir : mémoire AutoDream + article brut.
    # On compresse AVANT le débat si la taille dépasse les seuils :
    #   MICRO (< 6 000 chars)  : pass-through, aucune modification
    #   AUTO  (6k-16k chars)   : compression de l'article, mémoire préservée
    #   FULL  (> 16 000 chars) : compression totale en XML <summary>
    # Le sous-modèle léger (Cerebras) génère le résumé — invisible pour les agents.
    # ---------------------------------------------------------------
    compression_result = compress_article_if_needed(texte_article, ticker)
    texte_article_final = compression_result.text

    task_prompt = (
        f"Ticker: {ticker}\n"
        f"Contexte marché: {contexte_str}\n\n"
        f"=== ANALYSE ABSA (aspects économiques détectés) ===\n"
        f"{absa_str}\n\n"
        f"=== ARTICLE (contexte de discussion) ===\n"
        f"{texte_article_final}\n\n"
        f"Débattez de l'impact de cette actualité sur le cours de {ticker}. "
        f"Appuyez-vous sur les aspects ABSA détectés. "
        f"Répondez directement aux arguments des autres agents. "
        f"Terminez TOUJOURS votre réponse par votre bloc <status>."
    )

    # ---------------------------------------------------------------
    # PHASE 1 — Débat direct : les 3 agents se parlent en RoundRobin
    # 1 message initial (task_prompt) + 3 tours × 3 agents = 10 messages max
    # ---------------------------------------------------------------
    termination = MaxMessageTermination(max_messages=10)
    debat_team = RoundRobinGroupChat(
        participants=[agent_haussier, agent_baissier, agent_neutre], termination_condition=termination
    )

    logger.info("[Débat] Lancement du débat direct multi-agent pour %s...", ticker)
    debate_result = await debat_team.run(task=task_prompt)
    logger.info("[Débat] Fin du débat (%d messages échangés).", len(debate_result.messages))

    # ---------------------------------------------------------------
    # PHASE 2 — Construction du Shared Scratchpad à partir de la transcription
    # Le superviseur extrait les blocs <status> de chaque message
    # et alimente le scratchpad XML partagé
    # ---------------------------------------------------------------
    scratchpad = SharedScratchpad(ticker=ticker)
    round_counter = {"Haussier": 0, "Baissier": 0, "Neutre": 0}

    # Transcription brute pour l'audit trail
    transcription_raw = []

    for msg in debate_result.messages:
        if not hasattr(msg, "content") or not msg.content:
            continue

        agent_name = getattr(msg, "source", "Unknown")
        transcription_raw.append(f"{agent_name}: {msg.content}")

        if agent_name not in round_counter:
            continue  # on ignore le message initial (task)

        round_counter[agent_name] += 1
        current_round = round_counter[agent_name]

        # Extraction du bloc <status> pour le scratchpad
        status = _extract_status_block(msg.content, agent_name)
        if status:
            raw_confidence_str = status.get("confidence", "0.5")
            try:
                raw_confidence = float(raw_confidence_str)
            except ValueError:
                raw_confidence = 0.5

            # -----------------------------------------------
            # ID Score — Anti-verbosité (#2)
            # Calcule la densité d'information de l'argument
            # et applique une décote si le texte est trop creux
            # -----------------------------------------------
            key_argument = status.get("key_argument", msg.content[:200])
            id_result = compute_id_score(key_argument)
            adjusted_confidence = apply_id_penalty(raw_confidence, id_result)

            if id_result.is_penalized:
                logger.debug(
                    "[ID Score] Agent %s Tour %d : confiance %.2f → %.2f (ID=%.4f)",
                    agent_name,
                    current_round,
                    raw_confidence,
                    adjusted_confidence,
                    id_result.id_score,
                )

            summary = f"{key_argument} [confiance: {adjusted_confidence}]"
        else:
            raw_confidence = 0.5
            id_result = compute_id_score(msg.content[:300])
            summary = msg.content[:300]

        scratchpad.post_message(
            agent_name,
            current_round,
            summary,
            id_score=id_result.id_score if status else -1.0,
            raw_confidence=raw_confidence if status else None,
        )
        logger.info("[Scratchpad] Mise à jour — %s (Tour %d) → %s", agent_name, current_round, summary[:100])

    logger.info("[Scratchpad] État final :\n%s", scratchpad.to_xml())

    # ------------------------------------------------------------------
    # AUGMENTATION — Actor-Critic + Chain-of-Verification (BATCH 2)
    # On audite chaque argument du scratchpad avec CriticAgent et on verifie
    # les claims factuels avec VerifierAgent. Les signaux aggreges sont
    # injectes dans le prompt du Consensus pour qu'il module sa confiance.
    # ------------------------------------------------------------------
    augmentation_block = ""
    augmentation_multiplier = 1.0
    try:
        from src.utils.debate_augmentation import (
            run_augmentation,
            wrap_autogen_client,
        )

        critic_llm_callable = wrap_autogen_client(client_neutre)  # critique impartial
        # Pour la verification on peut utiliser le meme client : l'extractor
        # et le verifier ne sont pas les memes prompts, donc pas de collusion.
        verifier_extractor = wrap_autogen_client(client_haussier)
        verifier_checker = wrap_autogen_client(client_baissier)

        aug_report = run_augmentation(
            scratchpad=scratchpad,
            article_text=texte_article_final,
            critic_call_llm=critic_llm_callable,
            verifier_extractor_llm=verifier_extractor,
            verifier_checker_llm=verifier_checker,
            context={"absa_summary": absa_str[:400], "ticker": ticker},
            max_arguments=6,
        )
        augmentation_block = aug_report.to_prompt_block()
        augmentation_multiplier = aug_report.confidence_multiplier
        logger.info(
            "[Augmentation] sev_avg=%.2f max_sev=%d ver_ratio=%.2f mult=%.2f",
            aug_report.avg_severity,
            aug_report.max_severity,
            aug_report.verification_ratio,
            aug_report.confidence_multiplier,
        )
    except Exception as aug_err:  # pragma: no cover
        logger.warning("[Augmentation] skipped (%s)", aug_err)

    # ------------------------------------------------------------------
    # Consensus Agent — lit le scratchpad XML + le bloc d'augmentation
    # ------------------------------------------------------------------

    consensus_agent = AssistantAgent(name="Consensus", model_client=client_consensus, system_message=PROMPT_CONSENSUS)

    # Le prompt Consensus est ~3× plus court qu'avant (scratchpad vs transcription brute)
    consensus_task = (
        f"Ticker analysé: {ticker}\n\n"
        f"=== ANALYSE ABSA ===\n{absa_str}\n\n"
        f"=== SCRATCHPAD CONSOLIDÉ DU DÉBAT ===\n"
        f"{scratchpad.to_xml()}\n\n"
        + (
            f"=== AUGMENTATION CRITIC + VERIFIER ===\n{augmentation_block}\n\n"
            f"(Le confidence_multiplier={augmentation_multiplier:.2f} "
            f"t'indique dans quelle mesure moduler ta confiance finale. "
            f"Sev HIGH ou contradictions => baisse ta confiance.)\n\n"
            if augmentation_block
            else ""
        )
        + "En te basant sur le scratchpad consolidé et les aspects ABSA, "
        "rends ta décision finale en JSON."
    )

    consensus_team = RoundRobinGroupChat(
        participants=[consensus_agent], termination_condition=MaxMessageTermination(max_messages=2)
    )

    consensus_result = await consensus_team.run(task=consensus_task)
    consensus_text = consensus_result.messages[-1].content if consensus_result.messages else "{}"

    logger.info("[Consensus] Réponse brute: %s", consensus_text[:200])

    # ------------------------------------------------------------------
    # Tracking coût LLM — chaque agent est tracké séparément pour estimer
    # le coût réel si les APIs étaient payantes. Autogen n'expose pas toujours
    # usage.prompt_tokens ; on estime à 4 chars/token (règle de pouce GPT).
    # ------------------------------------------------------------------
    try:
        from src.utils.llm_cost_tracker import track_llm_call

        # Résolution des noms de modèles depuis les configs actives
        def _resolve_model(role: str) -> str:
            for p in _ROLE_PREFERENCES.get(role, []):
                cfg = _DEBATE_CONFIGS.get(p)
                if cfg and os.getenv(cfg["env_key"]):
                    return cfg["model"]
            return "unknown"

        debate_models = [_resolve_model(r) for r in ("bull", "bear", "neutral")]

        # Estimation tokens par agent depuis le scratchpad
        scratchpad_xml = scratchpad.to_xml()
        consensus_chars = sum(len(getattr(m, "content", "") or "") for m in (consensus_result.messages or []))

        # Chaque agent de débat : prompt (~2000 chars) + réponse (~1500 chars) × 3 tours
        for agent_model in debate_models:
            agent_est_tokens = max(1, (2000 + 1500) * 3 // 4)  # ~2625 tokens/agent
            track_llm_call(
                model=agent_model,
                prompt_tokens=int(agent_est_tokens * 0.6),
                completion_tokens=int(agent_est_tokens * 0.4),
            )

        # Consensus : reçoit tout le scratchpad + produit la décision
        consensus_est_tokens = max(1, (len(scratchpad_xml) + consensus_chars) // 4)
        track_llm_call(
            model="llama-3.3-70b-versatile",
            prompt_tokens=int(consensus_est_tokens * 0.6),
            completion_tokens=int(consensus_est_tokens * 0.4),
        )
    except Exception as _exc:
        logger.debug("[CostTracker] debate estimate failed: %s", _exc)

    # Parse JSON — supporte le nouveau format étendu AgentAuditor + Council Mode
    try:
        clean_text = consensus_text.strip()
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0]
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0]

        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group().strip())
        else:
            raise ValueError("Pas de JSON trouvé dans la réponse du Consensus")
    except Exception as e:
        logger.error("[Consensus] Erreur parsing JSON: %s", e)
        logger.error("=== TEXTE COMPLET DE L'AGENT CONSENSUS ===")
        logger.error(consensus_text)
        logger.error("==========================================")
        decision = {
            "signal": "Neutre",
            "argument_dominant": "Parsing impossible",
            "motif": str(e),
            "cdps_resolved": [],
            "cdps_unresolved": [],
            "council_matrix": None,
        }

    # Valeurs par défaut si le LLM a omis les nouveaux champs
    decision.setdefault("cdps_resolved", [])
    decision.setdefault("cdps_unresolved", [])
    decision.setdefault("council_matrix", None)

    # Log des CDPs pour traçabilité
    n_resolved = len(decision.get("cdps_resolved", []))
    n_unresolved = len(decision.get("cdps_unresolved", []))
    logger.info("[AgentAuditor] CDPs résolus: %d | non résolus: %d", n_resolved, n_unresolved)
    if n_unresolved > 0:
        logger.debug("[AgentAuditor] CDPs non résolus: %s", decision.get("cdps_unresolved"))

    # En-tête informatif pour l'affichage (non envoyé aux modèles)
    header = (
        f"=== CASTING EXÉCUTION ===\n"
        f"• Haussier  : {mod_haussier}\n"
        f"• Baissier  : {mod_baissier}\n"
        f"• Neutre    : {mod_neutre}\n"
        f"• Consensus : {mod_consensus}\n"
        f"=========================\n\n"
    )

    decision["ticker"] = ticker
    # On stoppe l'audit trail (transcription lisible) dans le champ transcription
    decision["transcription"] = header + scratchpad.to_audit_transcript()
    # On expose aussi le scratchpad XML pour débogage / dashboarding futur
    decision["scratchpad_xml"] = scratchpad.to_xml()
    decision["consensus_model"] = mod_consensus
    # Métadonnées de compression (traçabilité — non envoyé aux LLMs)
    decision["compression_level"] = compression_result.level.value
    decision["compression_ratio"] = round(compression_result.compression_ratio, 2)
    decision["compression_model"] = compression_result.model_used or "none"

    # ----------------------------------------------------------------
    # ESE Intra-débat (#3) — Entropie Sémantique d'Ensemble
    # Mesure le désaccord épistémique réel entre les 3 agents
    # sur la base de leurs arguments (trigrammes)
    # ----------------------------------------------------------------
    agent_arguments = extract_agent_arguments_from_scratchpad(scratchpad.to_xml())
    ese_result = compute_intra_debate_ese(agent_arguments)
    ese_confidence_factor = ese_to_confidence_factor(ese_result)

    decision["ese_score"] = ese_result.ese_score
    decision["ese_high_divergence"] = ese_result.is_high_divergence
    decision["ese_confidence_factor"] = ese_confidence_factor

    if ese_result.is_high_divergence:
        logger.warning(
            "[ESE] Désaccord épistémique élevé (ESE=%.3f > seuil) pour %s — facteur confiance réduit à %.2f",
            ese_result.ese_score,
            ticker,
            ese_confidence_factor,
        )
    else:
        logger.info(
            "[ESE] Divergence sémantique intra-débat : %.3f (facteur=%.2f)", ese_result.ese_score, ese_confidence_factor
        )

    return decision


@task
def lancer_debat(texte_article: str, ticker_symbol: str, contexte_marche: dict, absa_result: dict) -> dict:
    """Tâche LangGraph — lance le débat async via asyncio.run()."""
    return asyncio.run(_run_debate(texte_article, ticker_symbol, contexte_marche, absa_result))


@entrypoint()
def workflow_debat_actualite(inputs: dict) -> dict:
    """
    Point d'entrée LangGraph.
    Attend : {
        "texte_article"  : str,
        "ticker_symbol"  : str,
        "contexte_marche": dict,
        "absa_result"    : dict
    }
    """
    res = lancer_debat(
        inputs["texte_article"],
        inputs["ticker_symbol"],
        inputs.get("contexte_marche", {}),
        inputs.get("absa_result", {"aspects": []}),
    )
    return res.result()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test = {
        "texte_article": "Apple annonce des résultats trimestriels record, dépassant les attentes des analystes avec un bénéfice par action de 2.18$.",
        "ticker_symbol": "AAPL",
        "contexte_marche": {"current_price": 189.5, "volume": 55000000, "variation_5d": 2.3},
    }
    result = workflow_debat_actualite.invoke(test)
    logger.info("\nRésultat du débat:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
