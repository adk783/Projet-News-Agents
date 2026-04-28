"""
stress_generator.py -- Générateur de variants de stress pour articles financiers
==================================================================================
OBJECTIF :
  Mesurer la robustesse du pipeline face à des inputs dégradés.
  Un système robuste doit produire des signaux cohérents même si l'article
  est tronqué, bruité, ou partiellement contradictoire.

TROIS NIVEAUX DE STRESS :

  NIVEAU 1 — TRONCATURE (Incomplétude)
    L'information financière est souvent incomplète en temps réel.
    On tronque l'article à 50% puis 25% de sa taille.
    → Le signal devrait rester cohérent pour L1 (50%), plus incertain pour L2 (25%).

  NIVEAU 2 — BRUIT CONTRADICTOIRE (Ambiguïté)
    Les marchés reçoivent souvent des news contradictoires simultanément.
    On injecte une ou deux phrases qui contredisent la thèse principale.
    → Le signal doit devenir "Neutre" ou conserver sa direction mais avec
      un risk_level YOLO plus élevé.

  NIVEAU 3 — INVERSION SÉMANTIQUE (Test de compréhension)
    On inverse les termes clés financiers (beat/miss, hausse/baisse).
    → Le signal DOIT s'inverser — sinon l'agent ne lit pas vraiment les chiffres.

USAGE :
  from src.utils.stress_generator import generate_stress_variants, StressVariant

  variants = generate_stress_variants(original_text, ticker)
  for v in variants:
      print(v.stress_type, v.severity, len(v.text))
      # Lancer v.text dans le pipeline
"""

import random
import re
from dataclasses import dataclass
from typing import Optional

# Phrases de bruit contradictoires par type de signal
NOISE_PHRASES_BEARISH = [
    " However, analysts noted significant concerns about the sustainability of these results.",
    " Multiple sources indicate that key executives are privately pessimistic about Q2 outlook.",
    " Credit rating agencies have placed the company on negative watch this week.",
    " Short interest has surged 40% in the past month, suggesting institutional skepticism.",
    " Supply chain disruptions are expected to materially impact next quarter's gross margins.",
]

NOISE_PHRASES_BULLISH = [
    " Despite near-term headwinds, the long-term structural thesis remains intact.",
    " Several analysts upgraded the stock following the announcement.",
    " Institutional investors reportedly accumulated shares in after-hours trading.",
    " The company's balance sheet remains strong with significant liquidity.",
    " Management reaffirmed confidence in full-year targets on a private investor call.",
]

# Termes financiers à inverser (beat ↔ miss, beat ↔ miss, etc.)
SEMANTIC_INVERSIONS = [
    (r"\bbeat\b", "missed"),
    (r"\bmissed\b", "beat"),
    (r"\bexceeded\b", "fell short of"),
    (r"\bbelow\b", "above"),
    (r"\babove\b", "below"),
    (r"\bgrowth\b", "decline"),
    (r"\bdecline\b", "growth"),
    (r"\bincreased\b", "decreased"),
    (r"\bdecreased\b", "increased"),
    (r"\bimproved\b", "deteriorated"),
    (r"\bexpanded\b", "compressed"),
    (r"\bcompressed\b", "expanded"),
    (r"\braised\b", "cut"),
    (r"\bcut\b", "raised"),
    (r"\bpositive\b", "negative"),
    (r"\bnegative\b", "positive"),
    (r"\bstrong\b", "weak"),
    (r"\bweak\b", "strong"),
    (r"\bprofitable\b", "unprofitable"),
    (r"\bbullish\b", "bearish"),
    (r"\bbearish\b", "bullish"),
]


@dataclass
class StressVariant:
    """Un article stressé avec ses métadonnées."""

    original_id: str  # ID de l'article original
    stress_type: str  # "truncation" | "noise" | "inversion"
    severity: str  # "light" | "medium" | "heavy"
    text: str  # Texte stressé
    description: str  # Description lisible de la transformation
    expected_shift: Optional[str] = None  # Signal attendu après stress


def truncate(text: str, ratio: float) -> str:
    """Tronque le texte à `ratio` de sa taille originale, en coupant proprement à la phrase."""
    target_len = int(len(text) * ratio)
    truncated = text[:target_len]
    # Coupe à la dernière phrase complète
    last_period = max(truncated.rfind(". "), truncated.rfind(".\n"))
    if last_period > target_len * 0.5:
        truncated = truncated[: last_period + 1]
    return truncated.strip()


def inject_noise(text: str, signal_direction: str = "bearish") -> str:
    """
    Injecte une phrase contradictoire au milieu de l'article.
    signal_direction: le signal ATTENDU de l'article original
    → on injecte une phrase qui contredit ce signal
    """
    phrases = NOISE_PHRASES_BEARISH if signal_direction == "bullish" else NOISE_PHRASES_BULLISH

    noise = random.choice(phrases)

    # Injection au milieu du texte (après le premier paragraphe)
    paragraphs = text.split("\n\n")
    if len(paragraphs) >= 2:
        paragraphs.insert(1, noise.strip())
        return "\n\n".join(paragraphs)
    else:
        mid = len(text) // 2
        last_period = text.rfind(". ", 0, mid)
        if last_period > 0:
            return text[: last_period + 1] + noise + text[last_period + 1 :]
        return text + noise


def invert_semantics(text: str) -> tuple[str, int]:
    """
    Inverse les termes financiers clés dans le texte.
    Retourne (texte inversion, nombre d'inversions effectuées).
    """
    result = text
    n_inversions = 0
    for pattern, replacement in SEMANTIC_INVERSIONS:
        new_text, count = re.subn(pattern, replacement, result, flags=re.IGNORECASE)
        result = new_text
        n_inversions += count
    return result, n_inversions


def generate_stress_variants(
    text: str,
    article_id: str,
    original_signal: Optional[str] = None,
    seed: int = 42,
) -> list[StressVariant]:
    """
    Génère tous les variants de stress pour un article.

    Args:
        text           : Texte original de l'article
        article_id     : Identifiant de l'article original
        original_signal: Signal prédit par le pipeline sans stress (Achat/Vente/Neutre)
        seed           : Seed pour reproduce les variants de bruit

    Returns:
        Liste de 5 StressVariant : L1, L2, Bruit1, Bruit2, Inversion
    """
    random.seed(seed)

    variants = []

    # --- NIVEAU 1 : Troncature légère (50%) ---
    text_50 = truncate(text, 0.50)
    variants.append(
        StressVariant(
            original_id=article_id,
            stress_type="truncation",
            severity="light",
            text=text_50,
            description="Article tronqué à 50% (simuler info partielle en temps réel)",
            expected_shift=original_signal,  # Le signal devrait être stable
        )
    )

    # --- NIVEAU 2 : Troncature sévère (25%) ---
    text_25 = truncate(text, 0.25)
    variants.append(
        StressVariant(
            original_id=article_id,
            stress_type="truncation",
            severity="heavy",
            text=text_25,
            description="Article tronqué à 25% (simuler info très fragmentée)",
            expected_shift="Neutre",  # Sans assez d'info, l'agent devrait être plus neutre
        )
    )

    # --- NIVEAU 3 : Bruit contradictoire faible (1 phrase contradictoire) ---
    direction = "bullish" if original_signal == "Achat" else "bearish"
    text_noise_1 = inject_noise(text, direction)
    variants.append(
        StressVariant(
            original_id=article_id,
            stress_type="noise",
            severity="light",
            text=text_noise_1,
            description="Injection d'une phrase contradictoire (simuler info mixte)",
            expected_shift=original_signal,  # Signal devrait tenir face à 1 phrase
        )
    )

    # --- NIVEAU 4 : Bruit contradictoire fort (2 phrases contradictoires) ---
    text_noise_2 = inject_noise(text_noise_1, direction)  # Deuxième injection
    variants.append(
        StressVariant(
            original_id=article_id,
            stress_type="noise",
            severity="heavy",
            text=text_noise_2,
            description="Injection de 2 phrases contradictoires (news très contradictoire)",
            expected_shift="Neutre",  # Avec 2 contradictions, l'agent devrait hésiter
        )
    )

    # --- NIVEAU 5 : Inversion sémantique ---
    text_inverted, n_inv = invert_semantics(text)
    inverted_signal = "Vente" if original_signal == "Achat" else "Achat" if original_signal == "Vente" else "Neutre"
    variants.append(
        StressVariant(
            original_id=article_id,
            stress_type="inversion",
            severity="heavy",
            text=text_inverted,
            description=f"Inversion sémantique ({n_inv} termes financiers inversés)",
            expected_shift=inverted_signal,  # Le signal DOIT s'inverser
        )
    )

    return variants


def compute_stability_score(
    original_signal: str,
    variant_signals: list[tuple[str, str, Optional[str]]],
) -> dict:
    """
    Calcule le score de stabilité du pipeline face aux variants de stress.

    Args:
        original_signal  : Signal sans stress
        variant_signals  : [(stress_type, severity, signal_prédit), ...]

    Returns:
        Dict avec stability_score, degradation_map, verdict
    """
    results = []
    for stress_type, severity, predicted in variant_signals:
        expected = {
            "truncation_light": original_signal,
            "truncation_heavy": "Neutre",
            "noise_light": original_signal,
            "noise_heavy": "Neutre",
            "inversion_heavy": (
                "Vente" if original_signal == "Achat" else "Achat" if original_signal == "Vente" else "Neutre"
            ),
        }.get(f"{stress_type}_{severity}")

        correct = (predicted == expected) if expected and predicted else None
        results.append(
            {
                "test": f"{stress_type}_{severity}",
                "expected": expected,
                "got": predicted,
                "pass": correct,
            }
        )

    n_pass = sum(1 for r in results if r["pass"] is True)
    n_total = sum(1 for r in results if r["pass"] is not None)

    stability_score = n_pass / n_total if n_total else 0.0

    if stability_score >= 0.8:
        verdict = "ROBUSTE - Le système résiste bien au bruit et aux informations incomplètes."
    elif stability_score >= 0.6:
        verdict = "FRAGILE - Le système diverge sous stress modéré. Revoir les prompts agents."
    else:
        verdict = "INSTABLE - Le système est très sensible aux perturbations. Architecture à consolider."

    return {
        "original_signal": original_signal,
        "stability_score": round(stability_score, 3),
        "tests": results,
        "verdict": verdict,
    }
