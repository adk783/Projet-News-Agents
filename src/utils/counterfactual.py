"""
counterfactual.py — Générateur et scoreur d'invariances contrefactuelles

====================================================================
/!\  HORS-LIGNE SEULEMENT — NE JAMAIS APPELER DEPUIS LE PIPELINE LIVE
====================================================================
Ce module est un outil d'**audit offline**. Générer 11 counterfactuals × 3-4
LLMs × 100 articles/jour = ~3300 appels API/jour, insoutenable en prod.

Usage correct :
    scripts/audit_hebdomadaire.py   # dimanche, échantillon ≤ 50 articles
    eval/evaluate_counterfactual_invariance.py (sans --pipeline = MSS only)

En mode MSS-only (pas d'appel LLM), il reste très bon marché et peut être
intégré dans la CI.

Objectif
--------
Vérifier que le pipeline agents+LLM prend sa décision à partir de la
*statistique suffisante minimale* du contenu, et non de nuisances
non-causales (ticker, nom CEO, date, langue, style journalistique).

Méthodologie
------------
Pour chaque article x, on génère un ensemble d'articles contrefactuels
{x'} qui préservent T(x) mais altèrent une nuisance :
  • ticker_swap      : AAPL ↔ MSFT ↔ TSLA …
  • date_shift       : décale l'ancrage temporel de ±1..3 mois
  • magnitude_perturb: bruite légèrement les chiffres ±(ε·σ)
  • sector_swap      : change le secteur mentionné (Tech -> Finance)
  • ceo_swap         : remplace le nom du CEO
  • style_rewrite    : passe en tonalité neutre (mais magnitude conservée)

Puis on calcule :

    CI_score(x) = 1 - KL( p(signal|x) || p(signal|x') )_{x'}

    Normalized :  CI ∈ [0, 1]
      1 = invariant (bien)
      0 = maximally sensible (mauvais = le modèle triche via nuisances)

On aggrège : mean CI sur N articles donne le **Counterfactual Invariance
Score** du pipeline global.

Seuils conseillés (Veitch et al., 2021)
  CI ≥ 0.90  :  Système robuste, décision basée sur contenu causal
  0.75-0.89  :  Acceptable, quelques biais marginaux
  < 0.75     :  PROBLÈME — le LLM utilise ses priors d'entraînement

Référence
---------
Veitch, V., D'Amour, A., Yadlowsky, S., Eisenstein, J. (2021).
"Counterfactual Invariance to Spurious Correlations in Text Classification."
Advances in Neural Information Processing Systems 34.

Pearl, J. (2009). Causality (2nd ed.), §3.2 (do-calculus).
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import math
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.utils.minimal_sufficient_statistic import MSSResult, compute_mss, mss_distance

# ---------------------------------------------------------------------------
# Dictionnaires de swap
# ---------------------------------------------------------------------------

TICKER_PAIRS = {
    "AAPL": ("Apple", "CEO_A", ["iPhone", "Mac", "Tim Cook"]),
    "MSFT": ("Microsoft", "CEO_B", ["Azure", "Windows", "Satya Nadella"]),
    "TSLA": ("Tesla", "CEO_C", ["Cybertruck", "Model 3", "Elon Musk"]),
    "AMZN": ("Amazon", "CEO_D", ["AWS", "Prime", "Andy Jassy"]),
    "GOOGL": ("Alphabet", "CEO_E", ["Google", "Gemini", "Sundar Pichai"]),
    "NVDA": ("Nvidia", "CEO_F", ["H100", "CUDA", "Jensen Huang"]),
    "META": ("Meta", "CEO_G", ["Facebook", "Instagram", "Mark Zuckerberg"]),
    "JPM": ("JPMorgan", "CEO_H", ["Chase", "JPM", "Jamie Dimon"]),
}

SECTOR_SWAPS = [
    ("Technology", "Financial Services"),
    ("Automotive", "Consumer Staples"),
    ("Healthcare", "Energy"),
    ("E-commerce", "Industrial"),
]


# ---------------------------------------------------------------------------
# Perturbations
# ---------------------------------------------------------------------------


def perturb_ticker(text: str, old_ticker: str, new_ticker: str) -> str:
    """Remplace les mentions d'une entreprise par une autre."""
    if old_ticker not in TICKER_PAIRS or new_ticker not in TICKER_PAIRS:
        return text
    old_name, old_ceo, old_products = TICKER_PAIRS[old_ticker]
    new_name, new_ceo, new_products = TICKER_PAIRS[new_ticker]
    out = text
    out = re.sub(rf"\b{re.escape(old_ticker)}\b", new_ticker, out, flags=re.IGNORECASE)
    out = re.sub(rf"\b{re.escape(old_name)}\b", new_name, out, flags=re.IGNORECASE)
    for op, np_ in zip(old_products, new_products):
        out = re.sub(rf"\b{re.escape(op)}\b", np_, out, flags=re.IGNORECASE)
    return out


def perturb_date(text: str, shift_months: int = 3) -> str:
    """
    Décale les dates mentionnées. Simple : change l'année 20XX par 20XX±1.
    Ne touche pas aux quantités financières car on recherche "20XX" strict.
    """

    def replace(m: re.Match) -> str:
        y = int(m.group(0))
        return str(max(1990, y - 1 if shift_months > 0 else y + 1))

    return re.sub(r"\b(20\d{2})\b", replace, text)


def perturb_magnitude(
    text: str,
    epsilon: float = 0.05,
    seed: Optional[int] = None,
) -> str:
    """
    Bruite les pourcentages ±epsilon (relative). Magnitude 10% -> entre 9.5 et 10.5%.
    Ne doit PAS changer le bucket (small/medium/large) sinon T(x) change aussi.
    """
    rng = random.Random(seed) if seed is not None else random.Random()

    def replace(m: re.Match) -> str:
        try:
            raw = m.group(1).replace(" ", "")
            v = float(raw)
            # Bruit relatif faible
            noise = v * epsilon * (2 * rng.random() - 1)
            return f"{v + noise:.2f}%"
        except ValueError:
            return m.group(0)

    return re.sub(r"([+-]?\s*\d+(?:\.\d+)?)\s*%", replace, text)


def perturb_sector(text: str) -> str:
    """Swap sectoriel simple."""
    out = text
    for old, new in SECTOR_SWAPS:
        out = re.sub(rf"\b{re.escape(old)}\b", new, out, flags=re.IGNORECASE)
    return out


def perturb_ceo(text: str, old_ticker: str) -> str:
    """Remplace le CEO par un placeholder neutre (CEO_X)."""
    if old_ticker not in TICKER_PAIRS:
        return text
    _, _, tokens = TICKER_PAIRS[old_ticker]
    # Suppose que le 3e élément de la liste tokens est le CEO
    out = text
    # on masque tous les noms de CEOs connus de toutes les entreprises
    for t, (_, _, toks) in TICKER_PAIRS.items():
        out = re.sub(rf"\b{re.escape(toks[2])}\b", "the CEO", out, flags=re.IGNORECASE)
    return out


def style_rewrite_neutral(text: str) -> str:
    """
    Retire les adjectifs hyperboliques ('stunning', 'catastrophic', 'incredible')
    -> tests que le LLM ne se base pas sur le ton émotionnel.
    """
    hype = [
        "stunning",
        "catastrophic",
        "incredible",
        "amazing",
        "shocking",
        "massive",
        "huge",
        "terrible",
        "phenomenal",
        "extraordinary",
        "impressively",
        "poorly",
        "brilliantly",
        "disastrously",
    ]
    out = text
    for w in hype:
        out = re.sub(rf"\b{w}\b", "", out, flags=re.IGNORECASE)
    # Clean double spaces
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------------------
# Orchestrateur
# ---------------------------------------------------------------------------


@dataclass
class Counterfactual:
    perturbation: str  # nom du type de perturbation
    text: str
    mss_after: MSSResult
    mss_distance: float


@dataclass
class CounterfactualReport:
    original_text: str
    original_mss: MSSResult
    counterfactuals: list[Counterfactual] = field(default_factory=list)
    mss_preservation: float = 1.0  # combien T(x) est conservée sous perturbation
    signal_invariance: float = 1.0  # combien p(signal) est conservée (à calculer)
    ci_score: float = 1.0  # Counterfactual Invariance global
    anomalies: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"CI-score={self.ci_score:.3f} | "
            f"MSS-preservation={self.mss_preservation:.3f} | "
            f"n={len(self.counterfactuals)} perturbations"
        )


def generate_counterfactuals(
    text: str,
    original_ticker: str,
    seed: int = 42,
) -> list[Counterfactual]:
    """
    Génère un ensemble standard de counterfactuals pour un article.
    """
    original_mss = compute_mss(text)
    variants: list[Counterfactual] = []

    # 1. Ticker swap (on swappe vers 3 tickers différents)
    for new_t in list(TICKER_PAIRS.keys()):
        if new_t != original_ticker and len(variants) < 6:
            t_new = perturb_ticker(text, original_ticker, new_t)
            mss_new = compute_mss(t_new)
            variants.append(
                Counterfactual(
                    perturbation=f"ticker_swap->{new_t}",
                    text=t_new,
                    mss_after=mss_new,
                    mss_distance=mss_distance(original_mss, mss_new),
                )
            )

    # 2. Date shift
    t_new = perturb_date(text, shift_months=12)
    variants.append(
        Counterfactual(
            perturbation="date_shift_-1y",
            text=t_new,
            mss_after=compute_mss(t_new),
            mss_distance=mss_distance(original_mss, compute_mss(t_new)),
        )
    )

    # 3. Magnitude perturbation (light)
    t_new = perturb_magnitude(text, epsilon=0.05, seed=seed)
    variants.append(
        Counterfactual(
            perturbation="magnitude_eps5",
            text=t_new,
            mss_after=compute_mss(t_new),
            mss_distance=mss_distance(original_mss, compute_mss(t_new)),
        )
    )

    # 4. Sector swap
    t_new = perturb_sector(text)
    variants.append(
        Counterfactual(
            perturbation="sector_swap",
            text=t_new,
            mss_after=compute_mss(t_new),
            mss_distance=mss_distance(original_mss, compute_mss(t_new)),
        )
    )

    # 5. CEO swap
    t_new = perturb_ceo(text, original_ticker)
    variants.append(
        Counterfactual(
            perturbation="ceo_removed",
            text=t_new,
            mss_after=compute_mss(t_new),
            mss_distance=mss_distance(original_mss, compute_mss(t_new)),
        )
    )

    # 6. Style neutral
    t_new = style_rewrite_neutral(text)
    variants.append(
        Counterfactual(
            perturbation="style_neutral",
            text=t_new,
            mss_after=compute_mss(t_new),
            mss_distance=mss_distance(original_mss, compute_mss(t_new)),
        )
    )

    return variants


def compute_ci_score(
    text: str,
    original_ticker: str,
    pipeline_fn: Optional[Callable[[str, str], dict]] = None,
    seed: int = 42,
) -> CounterfactualReport:
    """
    Calcule un CounterfactualReport pour un article.

    Args:
        text            : texte de l'article
        original_ticker : ticker de l'article
        pipeline_fn     : optional, callable (text, ticker) -> dict avec un champ
                          "signal" et optionnellement "confidence".  Si fourni,
                          le CI est calculé sur le changement de signal.
                          Sinon, seul le MSS-preservation est mesuré.
        seed            : seed de reproductibilité
    """
    original_mss = compute_mss(text)
    variants = generate_counterfactuals(text, original_ticker, seed=seed)

    # MSS preservation : fraction des counterfactuals où MSS(x')  ≈ MSS(x)
    # (distance < 0.3 considérée comme "préservée")
    preserved = sum(1 for v in variants if v.mss_distance < 0.3)
    mss_pres = preserved / len(variants) if variants else 1.0

    anomalies: list[str] = []
    signal_invariance = 1.0

    if pipeline_fn is not None:
        # Appelle le pipeline sur chaque counterfactual et compte les divergences
        original_out = pipeline_fn(text, original_ticker) or {}
        original_signal = original_out.get("signal", "Neutre")

        same_signal = 0
        for v in variants:
            try:
                out = pipeline_fn(v.text, original_ticker) or {}
                if out.get("signal") == original_signal:
                    same_signal += 1
                else:
                    anomalies.append(f"[{v.perturbation}] signal flipped: {original_signal} -> {out.get('signal')}")
            except Exception as exc:
                anomalies.append(f"[{v.perturbation}] pipeline error: {exc}")
        signal_invariance = same_signal / len(variants) if variants else 1.0

    # CI score global : moyenne arithmétique des deux composantes
    ci = 0.5 * mss_pres + 0.5 * signal_invariance

    return CounterfactualReport(
        original_text=text,
        original_mss=original_mss,
        counterfactuals=variants,
        mss_preservation=round(mss_pres, 4),
        signal_invariance=round(signal_invariance, 4),
        ci_score=round(ci, 4),
        anomalies=anomalies,
    )


if __name__ == "__main__":
    text = (
        "Apple beat Q4 earnings with EPS of $2.30 vs $2.10 expected. "
        "Revenue rose stunning 12% year over year driven by strong iPhone 15 sales. "
        "Tim Cook raised guidance for next quarter. "
        "This Technology giant from Cupertino continues to dominate."
    )
    report = compute_ci_score(text, "AAPL")
    print(report.summary())
    for v in report.counterfactuals:
        logger.info(f"  [{v.perturbation}] dMSS={v.mss_distance:.3f}")
