"""
evaluate_debate_dynamics.py -- Dynamique de débat multi-agents
===============================================================
PROBLÈME QU'ON CHERCHE À DÉTECTER :
  L'écho de chambre multi-agent (Groupthink) :
  Tous les agents expriment le même signal dès le Tour 1 avec une confiance
  élevée, les tours suivants ne font que se confirmer mutuellement.
  → Le débat ne sert à rien, il consomme des tokens pour rien.

  Un débat UTILE :
  - Les agents commencent en désaccord (entropie élevée, Tour 1)
  - Ils échangent des arguments DIFFÉRENTS (diversité élevée)
  - Ils convergent progressivement (entropie décroissante, Tour 3)
  → La convergence est le fruit d'un raisonnement, pas d'un biais groupthink.

MÉTRIQUES IMPLÉMENTÉES :

  1. AUC AGREEMENT
     Courbe d'accord = std des confiances à chaque tour.
     AUC sous cette courbe (normalisée) mesure à quel point l'accord était
     PRÉCOCE vs TARDIF.
     → AUC faible = désaccord puis convergence = BON DÉBAT
     → AUC élevée = accord dès le départ = ÉCHO DE CHAMBRE

  2. CONFIDENCE EVOLUTION (Confidence Score par agent/tour)
     Trace comment chaque agent CHANGE de confiance entre les tours.
     Un agent qui passe de 0.45 → 0.65 → 0.88 est en train de se convaincre.
     Un agent à 0.90 → 0.91 → 0.90 n'a jamais remis en question sa position.
     → Delta de confiance moyen = signal de plasticité intellectuelle des agents.

  3. ARGUMENT DIVERSITY SCORE
     Similarity lexicale (Jaccard) entre les arguments des différents agents.
     Si tous les agents utilisent les mêmes mots → écho de chambre.
     Si les vocabulaires divergent → ils explorent vraiment des facettes différentes.
     → Similarity moyenne > 0.6 = arguments quasi-identiques (mauvais signe).
     → Similarity < 0.3 = bonne diversité argumentative.

  4. ENTROPIE SÉMANTIQUE
     Shannon entropy sur la distribution des confiances par tour.
     H = -Σ p_i * log(p_i) où p_i = confiance normalisée de l'agent i.
     → H(Tour 1) élevée = désaccord initial fort (bon)
     → H(Tour Final) faible = convergence (bon)
     → Delta H = H(T1) - H(TF) = mesure de la qualité de la convergence

  5. DEBATE QUALITY SCORE (composite, 0-1)
     Combinaison pondérée des 4 métriques.
     > 0.7 = Débat de qualité (désaccord utile → convergence)
     < 0.4 = Écho de chambre probable

SOURCE DES DONNÉES :
  - Scratchpad XML généré par agent_debat.py (in-memory ou SQLite)
  - Colonne `transcription_debat` de la table articles

Lancé via : python eval/run_eval.py --layer 5 --sub debate
"""

import logging
import math
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("DebateDynamics")
logging.basicConfig(level=logging.WARNING)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------


@dataclass
class AgentRound:
    """Un tour de parole d'un agent."""

    agent: str  # "Haussier" | "Baissier" | "Neutre"
    round_num: int
    text: str  # Texte de l'argument (nettoyé)
    confidence: float  # Score extrait du [confiance: X.XX]
    words: set  # Vocabulaire utilisé (pour Jaccard)


@dataclass
class DebateAnalysis:
    """Résultat complet de l'analyse d'un débat."""

    article_id: str
    ticker: str
    n_rounds: int
    agents_found: list[str]
    rounds: list[AgentRound]

    # Métriques
    auc_agreement: Optional[float] = None  # 0-1, bas = bon débat
    confidence_delta: Optional[float] = None  # Variation moyenne de confiance
    argument_diversity: Optional[float] = None  # 0-1, haut = bien diversifié
    semantic_entropy_t1: Optional[float] = None  # Entropie au Tour 1
    semantic_entropy_tf: Optional[float] = None  # Entropie au Tour Final
    entropy_delta: Optional[float] = None  # T1 - TF, positif = convergence
    debate_quality_score: Optional[float] = None  # Score composite 0-1
    verdict: str = ""


# ---------------------------------------------------------------------------
# Parsing du scratchpad XML
# ---------------------------------------------------------------------------

# Patterns de parsing robustes
_RE_SECTION = re.compile(r'<section agent="([^"]+)">(.*?)</section>', re.DOTALL)
_RE_ARGUMENT = re.compile(r"<argument>(.*?)</argument>", re.DOTALL)
_RE_TOUR = re.compile(r"\[Tour\s*(\d+)\]")
_RE_CONFIDENCE = re.compile(r"\[confiance\s*:\s*([0-9.]+)\]", re.IGNORECASE)

# Mots vides à exclure du vocabulaire
_STOPWORDS = {
    "le",
    "la",
    "les",
    "de",
    "du",
    "des",
    "un",
    "une",
    "et",
    "est",
    "en",
    "à",
    "au",
    "que",
    "qui",
    "pour",
    "par",
    "sur",
    "dans",
    "avec",
    "pas",
    "ne",
    "il",
    "elle",
    "on",
    "nous",
    "vous",
    "ils",
    "elles",
    "je",
    "tu",
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "of",
    "in",
    "to",
    "and",
    "or",
    "but",
    "not",
    "with",
    "for",
    "at",
    "this",
    "that",
    "it",
    "has",
    "have",
    "had",
    "will",
    "would",
    "should",
    "could",
    "can",
}


def _tokenize(text: str) -> set[str]:
    """Extrait un ensemble de mots significatifs (>3 chars, hors stopwords)."""
    words = re.findall(r"\b[a-zA-ZÀ-ÿ]{4,}\b", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def parse_scratchpad_xml(xml: str, article_id: str = "", ticker: str = "") -> Optional[DebateAnalysis]:
    """
    Parse un scratchpad XML et extrait la structure du débat.

    Args:
        xml        : Contenu XML du scratchpad (balise <scratchpad> ou <status>)
        article_id : ID de l'article pour l'identification
        ticker     : Symbole boursier

    Returns:
        DebateAnalysis ou None si le parsing échoue.
    """
    # Extraire le ticker depuis le XML si non fourni
    if not ticker:
        ticker_match = re.search(r'ticker="([^"]+)"', xml)
        ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"

    sections = _RE_SECTION.findall(xml)
    if not sections:
        logger.warning("[DebateDyn] Aucune section trouvée dans le XML de %s", article_id)
        return None

    rounds: list[AgentRound] = []
    agents_found = []

    for agent_name, section_content in sections:
        agents_found.append(agent_name)
        arguments = _RE_ARGUMENT.findall(section_content)

        for arg_text in arguments:
            # Extraire le numéro de tour
            tour_match = _RE_TOUR.search(arg_text)
            round_num = int(tour_match.group(1)) if tour_match else 1

            # Extraire la confiance
            conf_match = _RE_CONFIDENCE.search(arg_text)
            confidence = float(conf_match.group(1)) if conf_match else 0.5

            # Nettoyer le texte (retire les balises de tour et confiance)
            clean_text = _RE_TOUR.sub("", arg_text)
            clean_text = _RE_CONFIDENCE.sub("", clean_text).strip()

            words = _tokenize(clean_text)

            rounds.append(
                AgentRound(
                    agent=agent_name,
                    round_num=round_num,
                    text=clean_text,
                    confidence=confidence,
                    words=words,
                )
            )

    if not rounds:
        return None

    n_rounds = max(r.round_num for r in rounds)
    return DebateAnalysis(
        article_id=article_id,
        ticker=ticker,
        n_rounds=n_rounds,
        agents_found=list(set(agents_found)),
        rounds=rounds,
    )


def parse_raw_transcript(transcript: str, article_id: str = "", ticker: str = "") -> Optional[DebateAnalysis]:
    """
    Parse la transcription brute. Supporte :
    1. L'ancien format avec [Tour N]
    2. Format plus recent avec <scratchpad>
    3. L'ancien format complet direct sans tours (fallback)
    """
    if "<scratchpad" in transcript or "<section" in transcript:
        return parse_scratchpad_xml(transcript, article_id, ticker)

    # Pattern : "[Tour 1] Haussier: texte [confiance: 0.85]"
    pattern_with_tours = re.compile(
        r"\[Tour\s*(\d+)\]\s*(Haussier|Baissier|Neutre)\s*:?\s*(.*?)(?=\[Tour|\Z)", re.DOTALL | re.IGNORECASE
    )
    matches_tours = pattern_with_tours.findall(transcript)

    rounds = []
    if matches_tours:
        for round_str, agent, text in matches_tours:
            conf_match = _RE_CONFIDENCE.search(text)
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            clean_text = _RE_CONFIDENCE.sub("", text).strip()
            rounds.append(
                AgentRound(
                    agent=agent.capitalize(),
                    round_num=int(round_str),
                    text=clean_text,
                    confidence=confidence,
                    words=_tokenize(clean_text),
                )
            )
    else:
        # Fallback pour les vieilles transcripts sans tour: "Haussier: bla bla"
        pattern_fallback = re.compile(
            r"^(Haussier|Baissier|Neutre)\s*:(.*?)(?=^(?:Haussier|Baissier|Neutre)\s*:|\Z)",
            re.DOTALL | re.IGNORECASE | re.MULTILINE,
        )
        matches_fallback = pattern_fallback.findall(transcript)

        # S'il y a un header cast, on le vire
        trans = transcript.split("=========================\n\n")[-1]
        matches_fallback = pattern_fallback.findall(trans)

        agent_counts = {}
        for agent, text in matches_fallback:
            agent_cap = agent.capitalize()
            agent_counts[agent_cap] = agent_counts.get(agent_cap, 0) + 1
            round_num = agent_counts[agent_cap]

            conf_match = _RE_CONFIDENCE.search(text)
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            clean_text = _RE_CONFIDENCE.sub("", text).strip()
            rounds.append(
                AgentRound(
                    agent=agent_cap,
                    round_num=round_num,
                    text=clean_text,
                    confidence=confidence,
                    words=_tokenize(clean_text),
                )
            )

    if not rounds:
        return None

    return DebateAnalysis(
        article_id=article_id,
        ticker=ticker,
        n_rounds=max(r.round_num for r in rounds),
        agents_found=list({r.agent for r in rounds}),
        rounds=rounds,
    )


# ---------------------------------------------------------------------------
# Métriques
# ---------------------------------------------------------------------------


def compute_auc_agreement(debate: DebateAnalysis) -> Optional[float]:
    """
    1. AUC Agreement

    Pour chaque tour, calcule l'écart-type des confiances des agents.
    Écart-type élevé = désaccord élevé.

    Ensuite, normalise la courbe [0, 1] et calcule l'AUC :
    - AUC proche de 0 → tout le désaccord était au début (bon débat)
    - AUC proche de 1 → les agents étaient d'accord depuis le début (écho)

    On inverse : AUC retourné = 1 - AUC_brut
    → Valeur haute = bon débat (désaccord puis convergence)
    → Valeur basse = écho de chambre
    """
    n_rounds = debate.n_rounds
    if n_rounds < 2:
        return None  # Pas de dynamique temporelle avec 1 seul tour

    # Confidence par tour (moyenne de tous les agents)
    stds_by_round = []
    for r in range(1, n_rounds + 1):
        confs = [ag.confidence for ag in debate.rounds if ag.round_num == r]
        if len(confs) >= 2:
            mean = sum(confs) / len(confs)
            variance = sum((c - mean) ** 2 for c in confs) / len(confs)
            std = math.sqrt(variance)
            stds_by_round.append(std)
        else:
            stds_by_round.append(0.0)

    if not stds_by_round or max(stds_by_round) == 0:
        return 0.0

    # Normaliser entre 0 et 1
    max_std = max(stds_by_round)
    norm = [s / max_std for s in stds_by_round]

    # AUC via la règle du trapèze
    auc_raw = sum((norm[i] + norm[i - 1]) / 2 for i in range(1, len(norm))) / (len(norm) - 1)

    # Inverser : on veut 1 = désaccord en fin de processus (convergence active)
    return round(1.0 - auc_raw, 3)


def compute_confidence_delta(debate: DebateAnalysis) -> Optional[float]:
    """
    2. Confidence Evolution — Plasticité intellectuelle des agents.

    Pour chaque agent, calcule Δconfiance = confiance(Tour_final) - confiance(Tour_1).
    Retourne la valeur absolue moyenne des deltas.

    → Delta élevé = les agents ont CHANGÉ d'opinion au cours du débat (bon)
    → Delta proche de 0 = les agents n'ont pas bougé (écho de chambre)
    """
    deltas = []
    for agent in set(r.agent for r in debate.rounds):
        agent_rounds = sorted([r for r in debate.rounds if r.agent == agent], key=lambda x: x.round_num)
        if len(agent_rounds) >= 2:
            delta = abs(agent_rounds[-1].confidence - agent_rounds[0].confidence)
            deltas.append(delta)

    if not deltas:
        return None
    return round(sum(deltas) / len(deltas), 3)


def compute_argument_diversity(debate: DebateAnalysis) -> Optional[float]:
    """
    3. Argument Diversity Score — Similarité de Jaccard inter-agents.

    Calcule la similarité lexicale (vocabulaire en commun) entre les
    arguments de paires d'agents différents.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Retourne 1 - Jaccard_moyen (donc 1 = très diversifié, 0 = identiques).
    """
    # Agrège tous les mots par agent
    agent_words: dict[str, set] = {}
    for r in debate.rounds:
        agent_words[r.agent] = agent_words.get(r.agent, set()) | r.words

    agents = list(agent_words.keys())
    if len(agents) < 2:
        return None

    jaccards = []
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            a_words = agent_words[agents[i]]
            b_words = agent_words[agents[j]]
            intersection = a_words & b_words
            union = a_words | b_words
            if union:
                jaccards.append(len(intersection) / len(union))

    if not jaccards:
        return None

    mean_jaccard = sum(jaccards) / len(jaccards)
    # Inverser : 1 = très différents (bon), 0 = identiques (écho)
    return round(1.0 - mean_jaccard, 3)


def compute_semantic_entropy(debate: DebateAnalysis) -> tuple[Optional[float], Optional[float]]:
    """
    4. Entropie Sémantique de Shannon sur les confiances par tour.

    H(tour) = -Σ p_i * log2(p_i) où p_i = confiance normalisée de l'agent i.

    Retourne (H_tour1, H_tour_final).
    Un bon débat : H(T1) élevée → H(TF) faible (convergence réelle).
    Un écho de chambre : H(T1) faible = accord dès le départ.
    """

    def _entropy(confs: list[float]) -> Optional[float]:
        if not confs:
            return None
        total = sum(confs)
        if total == 0:
            return None
        probs = [c / total for c in confs]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    rounds_t1 = [r.confidence for r in debate.rounds if r.round_num == 1]
    rounds_tf = [r.confidence for r in debate.rounds if r.round_num == debate.n_rounds]

    h1 = _entropy(rounds_t1)
    hf = _entropy(rounds_tf)
    return h1, hf


def compute_debate_quality_score(
    auc: Optional[float],
    conf_delta: Optional[float],
    diversity: Optional[float],
    h1: Optional[float],
    hf: Optional[float],
) -> float:
    """
    5. Score composite de qualité du débat (0 à 1).

    Pondération :
      50% — AUC Agreement (convergence tardive = bon débat)
      25% — Argument Diversity (vocabulaires différents)
      25% — Confidence Delta (agents ont changé d'opinion)

    L'entropie est utilisée comme information contextuelle, pas dans la pondération.
    """
    components = []
    weights = []

    if auc is not None:
        components.append(auc)
        weights.append(0.50)

    if diversity is not None:
        components.append(diversity)
        weights.append(0.25)

    if conf_delta is not None:
        # Normaliser le delta (0.3+ = très bon, 0 = statique)
        norm_delta = min(conf_delta / 0.30, 1.0)
        components.append(norm_delta)
        weights.append(0.25)

    if not components:
        return 0.5  # Valeur par défaut

    total_weight = sum(weights)
    if total_weight == 0:
        return 0.5

    score = sum(c * w for c, w in zip(components, weights)) / total_weight
    return round(score, 3)


def analyse_debate(debate: DebateAnalysis) -> DebateAnalysis:
    """
    Calcule toutes les métriques sur un DebateAnalysis parsé.
    Modifie debate in-place et retourne-le.
    """
    debate.auc_agreement = compute_auc_agreement(debate)
    debate.confidence_delta = compute_confidence_delta(debate)
    debate.argument_diversity = compute_argument_diversity(debate)

    h1, hf = compute_semantic_entropy(debate)
    debate.semantic_entropy_t1 = round(h1, 3) if h1 is not None else None
    debate.semantic_entropy_tf = round(hf, 3) if hf is not None else None
    debate.entropy_delta = round(h1 - hf, 3) if (h1 is not None and hf is not None) else None

    debate.debate_quality_score = compute_debate_quality_score(
        debate.auc_agreement,
        debate.confidence_delta,
        debate.argument_diversity,
        h1,
        hf,
    )

    # Verdict
    score = debate.debate_quality_score
    if score >= 0.70:
        debate.verdict = "DEBAT PRODUCTIF — Desaccord utile puis convergence argumentee."
    elif score >= 0.50:
        debate.verdict = "DEBAT MODERE — Quelques desaccords mais convergence partielle."
    elif score >= 0.30:
        debate.verdict = "ECO DE CHAMBRE PARTIEL — Les agents s'accordent trop vite."
    else:
        debate.verdict = "ECHO DE CHAMBRE FRANC — Le debat ne produit aucune valeur ajoutee."

    return debate


# ---------------------------------------------------------------------------
# Rapport
# ---------------------------------------------------------------------------


def _print_debate_report(analysis: DebateAnalysis) -> None:
    """Affiche le rapport d'analyse d'un débat."""
    print(f"\n  Article : {analysis.article_id} | Ticker: {analysis.ticker}")
    print(f"  Agents  : {', '.join(analysis.agents_found)} | {analysis.n_rounds} tours")
    print(f"  {'-' * 58}")

    # Tableau des confiances par tour
    print(f"  {'Tour':<6}", end="")
    for agent in sorted(set(r.agent for r in analysis.rounds)):
        print(f"  {agent[:8]:>10}", end="")
    print()

    for rnd in range(1, analysis.n_rounds + 1):
        print(f"  {rnd:<6}", end="")
        for agent in sorted(set(r.agent for r in analysis.rounds)):
            matches = [r.confidence for r in analysis.rounds if r.round_num == rnd and r.agent == agent]
            val = f"{matches[0]:.2f}" if matches else "--"
            print(f"  {val:>10}", end="")
        print()

    # Métriques
    print(
        f"\n  AUC Agreement      : {analysis.auc_agreement} "
        f"({'bon debat' if (analysis.auc_agreement or 0) > 0.5 else 'echo de chambre'})"
    )
    print(
        f"  Argument Diversity : {analysis.argument_diversity} "
        f"({'diversifie' if (analysis.argument_diversity or 0) > 0.5 else 'vocabulaire similaire'})"
    )
    print(
        f"  Confidence Delta   : {analysis.confidence_delta} "
        f"({'agents flexibles' if (analysis.confidence_delta or 0) > 0.15 else 'agents rigides'})"
    )
    print(
        f"  Entropie Tour 1    : {analysis.semantic_entropy_t1} "
        f"({'desaccord initial' if (analysis.semantic_entropy_t1 or 0) > 1.0 else 'accord precoce'})"
    )
    print(f"  Entropie Tour F    : {analysis.semantic_entropy_tf}")
    print(
        f"  Delta Entropie     : {analysis.entropy_delta} "
        f"({'convergence reelle' if (analysis.entropy_delta or 0) > 0.1 else 'pas de convergence'})"
    )
    print(f"\n  Score qualite      : {analysis.debate_quality_score:.2f}/1.00")
    print(f"  Verdict            : {analysis.verdict}")


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------

DATABASE_PATH = "data/news_database.db"


def run_debate_dynamics_analysis(limit: int = 10) -> dict:
    """
    Analyse la dynamique de tous les débats stockés en base de données.
    Fonctionne sur `transcription_debat` (SQLite) et les scratchpad XML associés.
    """
    print(f"\n{'=' * 70}")
    print("COUCHE 5d : Dynamique de Debat Multi-Agents (AUC Agreement, Diversite, Entropie)")
    print(f"{'=' * 70}")

    # Charger les transcriptions depuis SQLite
    try:
        conn = sqlite3.connect(DATABASE_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT url, ticker, title, signal_final, transcription_debat, argument_dominant
            FROM articles
            WHERE transcription_debat IS NOT NULL
              AND transcription_debat != ''
              AND transcription_debat != 'Parsing impossible'
            ORDER BY date_utc DESC
            LIMIT ?
        """,
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()
    except Exception as e:
        print(f"\n  [ERREUR] Lecture SQLite: {e}")
        return {}

    print(f"\n  {len(rows)} debates trouves en base.\n")

    if not rows:
        print("  Aucun debat disponible. Lancez quelques articles via le pipeline.")
        return {}

    analyses = []
    scores = []
    echo_count = 0
    productive_count = 0

    for row in rows:
        transcript = row["transcription_debat"]
        article_id = row["url"][-30:] if row["url"] else "?"
        ticker = row["ticker"]

        # Essayer d'abord le format XML (nouveau), puis transcript brut
        debate = None
        if "<scratchpad" in transcript or "<section" in transcript:
            debate = parse_scratchpad_xml(transcript, article_id, ticker)
        if debate is None:
            debate = parse_raw_transcript(transcript, article_id, ticker)

        if debate is None or not debate.rounds:
            print(f"  [SKIP] {article_id} — format non reconnu")
            continue

        # Analyse complète
        analyse_debate(debate)
        _print_debate_report(debate)

        analyses.append(debate)
        scores.append(debate.debate_quality_score)
        if debate.debate_quality_score >= 0.60:
            productive_count += 1
        else:
            echo_count += 1

    # Résumé agrégé
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"\n{'=' * 70}")
        print(f"RESUME ({len(analyses)} debates analyses)")
        print(f"{'=' * 70}")
        print(f"  Score qualite moyen    : {avg_score:.2f}/1.00")
        print(f"  Debats productifs      : {productive_count}/{len(analyses)} ({productive_count / len(analyses):.0%})")
        print(f"  Debats echo de chambre : {echo_count}/{len(analyses)} ({echo_count / len(analyses):.0%})")

        avg_diversity = sum(a.argument_diversity for a in analyses if a.argument_diversity is not None)
        n_div = sum(1 for a in analyses if a.argument_diversity is not None)
        if n_div:
            print(
                f"  Diversite arg. moyenne : {avg_diversity / n_div:.2f}/1.00 "
                f"({'bon' if avg_diversity / n_div > 0.5 else 'a ameliorer'})"
            )

        # Recommandations automatiques
        print("\n  RECOMMANDATIONS :")
        if avg_score < 0.50:
            print("  [!] Echo de chambre detecte. Suggestions :")
            print("      -> Augmenter la temperature des agents debatteurs")
            print("      -> Renforcer le prompt des agents (forcer la contradiction)")
            print("      -> Ajouter un agent Devil's Advocate explicite")
        elif echo_count > productive_count:
            print("  [~] Debat instable. Suggestions :")
            print("      -> Ajouter un prompt de recapitulation inter-tours")
            print("      -> Fixer un minimum d'arguments contraires par tour")
        else:
            print("  [OK] La dynamique de debat est saine.")

        print("=" * 70)

    return {
        "sub": "debate",
        "n_debates": len(analyses),
        "avg_quality": round(sum(scores) / len(scores), 3) if scores else None,
        "productive_count": productive_count,
        "echo_count": echo_count,
    }


# ---------------------------------------------------------------------------
# Analyse rapide in-memory (sans base de données)
# ---------------------------------------------------------------------------


def analyse_scratchpad_live(
    scratchpad_xml: str, article_id: str = "live", ticker: str = "?"
) -> Optional[DebateAnalysis]:
    """
    Analyse un scratchpad XML directement (pour intégration dans le pipeline).
    Peut être appelé depuis agent_pipeline.py après chaque débat.

    Usage :
        from eval.evaluate_debate_dynamics import analyse_scratchpad_live
        analysis = analyse_scratchpad_live(scratchpad_xml, article_id, ticker)
        if analysis:
            print(f"Qualite debat : {analysis.debate_quality_score}")
            if analysis.debate_quality_score < 0.40:
                logger.warning("Echo de chambre detecte pour %s", article_id)
    """
    debate = parse_scratchpad_xml(scratchpad_xml, article_id, ticker)
    if debate is None:
        return None
    return analyse_debate(debate)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=10, help="Nombre de debates a analyser depuis SQLite")
    a = p.parse_args()
    run_debate_dynamics_analysis(limit=a.limit)
