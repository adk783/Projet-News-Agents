"""
calibration.py — Calibration probabiliste des confidences LLM

OBJECTIF
--------
La confiance auto-rapportée par un LLM (`[confiance: 0.85]`) n'est pas
calibrée : une étude Open AI (2023) et Kadavath et al. (2022) montrent
qu'un LLM qui dit 0.85 a souvent raison seulement 0.55 à 0.70 du temps.

On apprend donc une fonction de calibration g: p_raw → p_calibrated à
partir d'un historique (p_raw, outcome ∈ {0, 1}).

MÉTHODES IMPLÉMENTÉES
---------------------
1. **Platt scaling** (Platt, 1999) — régression logistique 1-D sur (p, y)
   Rapide, assume que la miscalibration est sigmoïdale.

2. **Isotonic regression** (Zadrozny & Elkan, 2002) via PAVA
   (Pool-Adjacent-Violators Algorithm). Non-paramétrique, plus flexible,
   mais peut sur-ajuster si peu de données.

3. **ECE** (Expected Calibration Error) — Guo et al. (2017)
   Partitionne [0,1] en M bins, mesure |accuracy - confidence| par bin,
   pondère par nombre d'échantillons.

4. **Brier score** (Brier, 1950) — mean squared error ( p - y )²

API
---
    cal = PlattCalibrator().fit(p_raw, y_true)
    p_cal = cal.transform(p_new)
    ece = expected_calibration_error(p, y, n_bins=10)
    brier = brier_score(p, y)

Sans dépendances externes (pas de scikit-learn requis).
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

# ---------------------------------------------------------------------------
# Platt scaling (régression logistique 1-D par descente de gradient)
# ---------------------------------------------------------------------------


@dataclass
class PlattCalibrator:
    a: float = 0.0
    b: float = 0.0
    fitted: bool = False

    def fit(
        self,
        probs: Sequence[float],
        labels: Sequence[int],
        lr: float = 0.1,
        n_iter: int = 500,
        tol: float = 1e-6,
    ) -> PlattCalibrator:
        """
        MLE logistique : p_cal = sigmoid(a · p_raw + b)
        Minimise la cross-entropy négative par gradient descent.
        """
        assert len(probs) == len(labels), "Length mismatch"
        if len(probs) == 0:
            return self
        # Initialisation identité
        self.a, self.b = 1.0, 0.0
        prev_loss = float("inf")
        n = len(probs)
        for _ in range(n_iter):
            grad_a = 0.0
            grad_b = 0.0
            loss = 0.0
            for p, y in zip(probs, labels):
                logit = self.a * p + self.b
                # sigmoid numeriquement stable
                if logit >= 0:
                    z = math.exp(-logit)
                    pred = 1.0 / (1.0 + z)
                else:
                    z = math.exp(logit)
                    pred = z / (1.0 + z)
                eps = 1e-12
                loss -= y * math.log(pred + eps) + (1 - y) * math.log(1 - pred + eps)
                err = pred - y
                grad_a += err * p
                grad_b += err
            loss /= n
            self.a -= lr * grad_a / n
            self.b -= lr * grad_b / n
            if abs(prev_loss - loss) < tol:
                break
            prev_loss = loss
        self.fitted = True
        return self

    def transform(self, probs: Sequence[float]) -> list[float]:
        if not self.fitted:
            return list(probs)
        out = []
        for p in probs:
            logit = self.a * p + self.b
            if logit >= 0:
                z = math.exp(-logit)
                pred = 1.0 / (1.0 + z)
            else:
                z = math.exp(logit)
                pred = z / (1.0 + z)
            out.append(pred)
        return out


# ---------------------------------------------------------------------------
# Isotonic regression via PAVA (Pool-Adjacent-Violators)
# ---------------------------------------------------------------------------


@dataclass
class IsotonicCalibrator:
    """
    Isotonic regression sans scikit-learn.
    Stocke les points (x, y) triés par x et les y ajustés monotones croissants.
    Inference par interpolation linéaire.
    """

    xs: list[float] = field(default_factory=list)
    ys: list[float] = field(default_factory=list)
    fitted: bool = False

    def fit(self, probs: Sequence[float], labels: Sequence[int]) -> IsotonicCalibrator:
        assert len(probs) == len(labels)
        if len(probs) == 0:
            return self
        paired = sorted(zip(probs, labels), key=lambda t: t[0])
        xs = [p for p, _ in paired]
        ys = [float(y) for _, y in paired]
        weights = [1.0] * len(xs)

        # PAVA algorithm — enforce monotone non-decreasing
        i = 0
        while i < len(ys) - 1:
            if ys[i] > ys[i + 1]:
                # Merge i and i+1
                w_sum = weights[i] + weights[i + 1]
                y_merged = (weights[i] * ys[i] + weights[i + 1] * ys[i + 1]) / w_sum
                ys[i] = y_merged
                weights[i] = w_sum
                del ys[i + 1]
                del weights[i + 1]
                del xs[i + 1]
                # Step back to check previous constraint
                if i > 0:
                    i -= 1
            else:
                i += 1

        self.xs = xs
        self.ys = ys
        self.fitted = True
        return self

    def transform(self, probs: Sequence[float]) -> list[float]:
        if not self.fitted or not self.xs:
            return list(probs)
        out = []
        for p in probs:
            # interpolation linéaire entre les points connus
            if p <= self.xs[0]:
                out.append(self.ys[0])
            elif p >= self.xs[-1]:
                out.append(self.ys[-1])
            else:
                # recherche dichotomique
                lo, hi = 0, len(self.xs) - 1
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    if self.xs[mid] <= p:
                        lo = mid
                    else:
                        hi = mid
                x0, x1 = self.xs[lo], self.xs[hi]
                y0, y1 = self.ys[lo], self.ys[hi]
                if x1 == x0:
                    out.append(y0)
                else:
                    alpha = (p - x0) / (x1 - x0)
                    out.append(y0 + alpha * (y1 - y0))
        return out


# ---------------------------------------------------------------------------
# Métriques de calibration
# ---------------------------------------------------------------------------


def brier_score(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Brier score (Brier, 1950) : mean((p - y)²). 0 = parfait, 1 = pire."""
    n = len(probs)
    if n == 0:
        return 0.0
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / n


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> float:
    """
    ECE (Guo et al., 2017) :
        ECE = Σ_b (n_b / N) · | accuracy(b) - confidence(b) |
    Partitionne [0, 1] en n_bins.
    """
    n = len(probs)
    if n == 0 or n_bins <= 0:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        bin_idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[bin_idx].append((p, y))
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(p for p, _ in b) / len(b)
        avg_acc = sum(y for _, y in b) / len(b)
        ece += (len(b) / n) * abs(avg_acc - avg_conf)
    return round(ece, 6)


def reliability_diagram_bins(
    probs: Sequence[float],
    labels: Sequence[int],
    n_bins: int = 10,
) -> list[dict]:
    """
    Retourne les bins d'un reliability diagram (confidence vs accuracy)
    pour tracer ou logger.
    """
    bins = [[] for _ in range(n_bins)]
    for p, y in zip(probs, labels):
        bin_idx = min(n_bins - 1, max(0, int(p * n_bins)))
        bins[bin_idx].append((p, y))
    out = []
    for i, b in enumerate(bins):
        if not b:
            out.append({"bin": i, "n": 0, "avg_conf": None, "avg_acc": None})
        else:
            out.append(
                {
                    "bin": i,
                    "n": len(b),
                    "avg_conf": round(sum(p for p, _ in b) / len(b), 4),
                    "avg_acc": round(sum(y for _, y in b) / len(b), 4),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Helper de haut-niveau : fit + score comparatif
# ---------------------------------------------------------------------------


def fit_best_calibrator(
    probs: Sequence[float],
    labels: Sequence[int],
    prefer_isotonic_if_n_above: int = 50,
):
    """
    Fit Platt et Isotonic ; retourne le meilleur selon Brier sur les données
    d'entraînement. Si n < threshold, privilégie Platt (moins de sur-ajustement).
    """
    platt = PlattCalibrator().fit(probs, labels)
    iso = IsotonicCalibrator().fit(probs, labels)

    if len(probs) < prefer_isotonic_if_n_above:
        return platt, {"chosen": "platt", "reason": f"n={len(probs)} below threshold"}

    p_platt = platt.transform(probs)
    p_iso = iso.transform(probs)
    b_platt = brier_score(p_platt, labels)
    b_iso = brier_score(p_iso, labels)
    if b_platt < b_iso:
        return platt, {"chosen": "platt", "brier_platt": b_platt, "brier_iso": b_iso}
    return iso, {"chosen": "isotonic", "brier_platt": b_platt, "brier_iso": b_iso}


if __name__ == "__main__":
    # Smoke test : probas miscalibrées (over-confident LLM)
    import random

    random.seed(42)
    raw = [random.uniform(0.5, 1.0) for _ in range(200)]
    # Truth : LLM a raison 60% du temps à 0.85 de confiance
    y = [1 if random.random() < (0.4 + 0.3 * p) else 0 for p in raw]

    logger.info("Before calibration:")
    logger.info(f"  Brier = {brier_score(raw, y):.4f}")
    logger.info(f"  ECE   = {expected_calibration_error(raw, y):.4f}")

    best, info = fit_best_calibrator(raw, y)
    p_cal = best.transform(raw)
    print(f"Chose : {info['chosen']}")
    logger.info("After calibration:")
    logger.info(f"  Brier = {brier_score(p_cal, y):.4f}")
    logger.info(f"  ECE   = {expected_calibration_error(p_cal, y):.4f}")
