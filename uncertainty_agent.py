"""
UncertaintyAgent — version heuristique légère (sans fine-tuning).

Score continu [0, 1] basé sur :
  1) Ratio de termes du lexique Loughran-McDonald « Uncertainty » (poids 50%)
  2) Phrases conditionnelles / modales (poids 25%)
  3) Hedging / expressions de couverture (poids 15%)
  4) Ancrage factuel : présence de chiffres / unités → réduit l'incertitude (poids 10%)

Provenance :
- compute_uncertainty_score + patterns : branche `samuel` (uncertainty_agent.py).
- Variante allégée : pas de LoRA / fine-tuning ; lexique hardcodé (extrait L&M
  Uncertainty 2018) pour rester sans dépendance HuggingFace au runtime.
"""

import re


# ─── LEXIQUE Loughran-McDonald — catégorie Uncertainty (extrait stable) ──────
LM_UNCERTAINTY_LEXICON = {
    "approximate", "approximated", "approximately", "approximation",
    "approximations", "assume", "assumed", "assumes", "assuming",
    "assumption", "assumptions", "believe", "believed", "believes",
    "believing", "cautious", "cautiously", "conditional", "conditionally",
    "confidence", "confident", "contingencies", "contingency",
    "contingent", "could", "depend", "depended", "dependence", "dependencies",
    "dependency", "dependent", "depending", "depends", "destabilizing",
    "deviate", "deviated", "deviates", "deviating", "deviation", "deviations",
    "differ", "differed", "differing", "differs", "doubt", "doubted", "doubtful",
    "doubts", "exposure", "exposures", "fluctuate", "fluctuated", "fluctuates",
    "fluctuating", "fluctuation", "fluctuations", "hidden", "imprecise",
    "imprecision", "imprecisions", "improbability", "improbable", "incomplete",
    "incompleteness", "indefinite", "indefinitely", "indefiniteness",
    "indeterminable", "indeterminate", "inexact", "inexactness",
    "instabilities", "instability", "intangible", "intangibles", "likelihood",
    "may", "maybe", "might", "nearly", "nonassessable", "occasionally",
    "ordinarily", "pending", "perhaps", "possibilities", "possibility",
    "possible", "possibly", "precaution", "precautionary", "precautions",
    "predict", "predictability", "predicted", "predicting", "prediction",
    "predictions", "predictive", "predictor", "predictors", "predicts",
    "preliminarily", "preliminary", "presumably", "presume", "presumed",
    "presumes", "presuming", "presumption", "presumptions", "probabilistic",
    "probabilities", "probability", "probable", "probably", "reassess",
    "reassessed", "reassesses", "reassessing", "reassessment",
    "reassessments", "recalculate", "recalculated", "recalculates",
    "recalculating", "recalculation", "recalculations", "reconsider",
    "reconsidered", "reconsidering", "reconsiders", "revise", "revised",
    "revises", "revising", "risk", "risked", "riskier", "riskiest", "riskiness",
    "risking", "risks", "risky", "roughly", "rumors", "seems", "seldom",
    "seldomly", "sometime", "sometimes", "somewhat", "speculate", "speculated",
    "speculates", "speculating", "speculation", "speculations", "speculative",
    "speculatively", "sudden", "suddenly", "suggest", "suggested",
    "suggesting", "suggests", "susceptibility", "tentative", "tentatively",
    "turbulence", "uncertain", "uncertainly", "uncertainties", "uncertainty",
    "unclear", "unconfirmed", "undecided", "undefined", "undeterminable",
    "undetermined", "undocumented", "unexpected", "unexpectedly", "unforeseen",
    "unforeseeable", "unguaranteed", "unhedged", "unidentifiable",
    "unidentified", "unknown", "unknowns", "unobservable", "unplanned",
    "unpredictability", "unpredictable", "unpredictably", "unpredicted",
    "unproved", "unproven", "unquantifiable", "unquantified", "unreconciled",
    "unsettled", "unspecific", "unspecified", "untested", "unusual",
    "unusually", "unwritten", "vagaries", "vague", "vaguely", "vagueness",
    "vaguenesses", "vaguer", "vaguest", "variability", "variable", "variables",
    "variably", "variance", "variances", "variant", "variants", "variation",
    "variations", "varied", "varies", "vary", "varying", "volatile",
    "volatilities", "volatility",
}

CONDITIONAL_PATTERNS = [
    r"\bif\b", r"\bwhether\b", r"\bcould\b", r"\bmight\b", r"\bwould\b",
    r"\bshould\b", r"\bperhaps\b", r"\bpossibly\b", r"\bpotentially\b",
    r"\bit remains to be seen\b", r"\bremains unclear\b", r"\bhard to say\b",
    r"\bdifficult to predict\b", r"\btime will tell\b",
    r"\bon the other hand\b", r"\bhowever\b", r"\balthough\b",
    r"\bdespite\b", r"\bnevertheless\b", r"\byet\b",
    r"\bwe believe\b", r"\bwe expect\b", r"\bwe anticipate\b",
    r"\banalysts expect\b", r"\banalysts predict\b",
    r"\bgoing forward\b", r"\bin the near term\b", r"\bin the long run\b",
]

HEDGING_PATTERNS = [
    r"\bto some extent\b", r"\bmore or less\b", r"\bbroadly speaking\b",
    r"\bgenerally\b", r"\btypically\b", r"\btends to\b",
    r"\bmay or may not\b", r"\bnot necessarily\b", r"\bnot always\b",
    r"\bsubject to\b", r"\bcontingent upon\b", r"\bdepending on\b",
    r"\bassuming that\b", r"\bprovided that\b", r"\bunless\b",
    r"\bbarring\b", r"\babsent\b",
]


def _clip(score: float) -> float:
    return round(max(0.0, min(float(score), 1.0)), 4)


def compute_uncertainty_score(text: str) -> float:
    if not text:
        return 0.0

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0.0

    text_lower = text.lower()
    n_words = len(words)

    # Signal 1 — ratio lexique L&M (saturation à 2.5%)
    unc_count = sum(1 for w in words if w in LM_UNCERTAINTY_LEXICON)
    lexicon_score = min((unc_count / n_words) / 0.025, 1.0)

    # Signal 2 — conditionnelles / modales
    cond_hits = sum(1 for pat in CONDITIONAL_PATTERNS if re.search(pat, text_lower))
    conditional_score = min(cond_hits / 6.0, 1.0)

    # Signal 3 — hedging
    hedge_hits = sum(1 for pat in HEDGING_PATTERNS if re.search(pat, text_lower))
    hedging_score = min(hedge_hits / 4.0, 1.0)

    # Signal 4 — ancrage factuel (chiffres + symboles $/%) réduit l'incertitude
    digits_hits = len(re.findall(r"\d", text_lower))
    pct_hits = len(re.findall(r"%|\$", text_lower))
    factual_density = (digits_hits + 2 * pct_hits) / max(n_words, 1)
    factual_score = min(factual_density / 0.05, 1.0)

    score = (
        0.50 * lexicon_score
        + 0.25 * conditional_score
        + 0.15 * hedging_score
        - 0.10 * factual_score
    )

    return _clip(score)


class UncertaintyAgent:
    """Agent d'incertitude heuristique (pas de modèle entraîné requis)."""

    def __init__(self):
        pass

    def predict(self, text: str) -> float:
        return compute_uncertainty_score(text or "")

    def predict_batch(self, texts):
        return [self.predict(t) for t in (texts or [])]


if __name__ == "__main__":
    samples = [
        "Apple reports record revenue and 15% iPhone sales growth.",
        "The outlook remains uncertain and analysts are unsure whether margins could expand.",
    ]
    for s in samples:
        print(f"{compute_uncertainty_score(s):.3f}  {s[:60]}")
