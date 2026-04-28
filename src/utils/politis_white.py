"""
politis_white.py — Sélection automatique de la longueur de bloc pour
le bootstrap bloc-stationnaire (Politis & Romano 1994).

CONTEXTE
--------
Le bootstrap bloc-stationnaire utilise un paramètre `b` (longueur moyenne
du bloc ; les blocs ont une distribution géométrique de paramètre 1/b).
`b` doit être :
  - assez grand pour capturer l'autocorrélation présente dans la série
  - assez petit pour garder suffisamment d'indépendance entre blocs

Un `b` hardcodé à 5 (comme `evaluate_event_study.py` le faisait) est :
  - trop petit si la série est très auto-corrélée (retours post-event study
    des small-caps, vol clustering sur haute fréquence)
  - trop grand si la série est presque i.i.d. -> on gaspille de la variance

ALGORITHME Politis-White (2004)
-------------------------------
"Automatic Block-Length Selection for the Dependent Bootstrap",
Econometric Reviews, 23(1):53-70.

Étapes :
  1. Estimer ρ(k) (autocorrélation) pour k = 1, ..., K_N
     où K_N = max(5, ⌈log10(n)⌉).
  2. Trouver m̂ = plus petit lag tel que K_N lags consécutifs après m̂
     aient |ρ(k)| < c·√(log10(n)/n)  (seuil Bartlett, c=2 par défaut).
     Puis M = 2·m̂.
  3. Calculer, avec la fenêtre plate (flat-top) λ :
         Ĝ  = Σ_{|k|≤M, k≠0} |k| · λ(k/M) · γ̂(k)
         σ̂²(0) = γ̂(0) + 2·Σ_{k=1..M} λ(k/M) · γ̂(k)
         D_SB = 2 · σ̂²(0)²
  4. b_opt = (2 · Ĝ² / D_SB)^(1/3) · n^(1/3)

Fenêtre flat-top (Politis-Romano 1995) :
     λ(t) = 1              si |t| ≤ 0.5
     λ(t) = 2(1 − |t|)     si 0.5 < |t| ≤ 1
     λ(t) = 0              sinon

RÉFÉRENCES
----------
- Politis, D. N., & White, H. (2004).
  "Automatic Block-Length Selection for the Dependent Bootstrap."
  Econometric Reviews, 23(1), 53-70.
- Patton, A., Politis, D. N., & White, H. (2009).
  "Correction to 'Automatic Block-Length Selection for the Dependent
  Bootstrap'."  (Fix un bug dans la formule originale de D_SB.)
- Politis, D. N., & Romano, J. P. (1994).
  "The Stationary Bootstrap." J. Amer. Statist. Assoc., 89(428), 1303-1313.
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import math
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Fenêtre plate (flat-top) Politis-Romano 1995
# ---------------------------------------------------------------------------


def _flat_top_kernel(t: float) -> float:
    """
    λ(t) =
        1            si |t| ≤ 0.5
        2(1 − |t|)   si 0.5 < |t| ≤ 1
        0            sinon
    """
    a = abs(t)
    if a <= 0.5:
        return 1.0
    if a <= 1.0:
        return 2.0 * (1.0 - a)
    return 0.0


# ---------------------------------------------------------------------------
# Autocovariance / autocorrélation échantillonnales
# ---------------------------------------------------------------------------


def _autocov(x: np.ndarray, k: int) -> float:
    """γ̂(k) = (1/n) Σ_{t=1..n-k} (x_t − x̄)(x_{t+k} − x̄). k ≥ 0."""
    n = len(x)
    if k >= n:
        return 0.0
    mu = x.mean()
    xc = x - mu
    return float((xc[: n - k] * xc[k:]).sum() / n)


def _autocorr(x: np.ndarray, k: int) -> float:
    """ρ̂(k) = γ̂(k)/γ̂(0). k ≥ 0."""
    g0 = _autocov(x, 0)
    if g0 <= 0:
        return 0.0
    return _autocov(x, k) / g0


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def politis_white_block_length(
    data: Sequence[float] | np.ndarray,
    bootstrap_type: str = "stationary",
    c_bartlett: float = 2.0,
    b_min: int = 2,
    b_max: int | None = None,
) -> int:
    """
    Sélectionne la longueur de bloc optimale par la méthode Politis-White (2004).

    Parameters
    ----------
    data : array-like
        Série temporelle 1-D (typiquement des retours ou des CARs).
    bootstrap_type : {"stationary", "circular"}
        Variante du bootstrap. La formule D_SB change légèrement entre les
        deux (cf Patton-Politis-White 2009). Par défaut "stationary" (celui
        qu'on utilise dans `evaluate_event_study.py`).
    c_bartlett : float
        Constante pour le seuil Bartlett |ρ| < c·√(log10(n)/n). Défaut 2.
    b_min : int
        Longueur minimale retournée (sécurité).
    b_max : int ou None
        Longueur maximale. Si None, vaut max(2, n // 4).

    Returns
    -------
    int
        Longueur de bloc ≥ b_min, ≤ b_max. Par défaut 5 si données insuffisantes
        (n < 20).
    """
    x = np.asarray(data, dtype=float).ravel()
    n = len(x)

    # Garde-fou : séries courtes -> valeur par défaut
    if n < 20:
        return max(b_min, 5)

    if b_max is None:
        b_max = max(b_min, n // 4)

    log10_n = math.log10(n)
    K_N = max(5, int(math.ceil(log10_n)))
    bartlett_thresh = c_bartlett * math.sqrt(log10_n / n)

    # --- Étape 1-2 : trouver m̂ par flat-top lag-window criterion ---------
    # On parcourt k = 1, 2, ... et on cherche le plus petit m tel que
    # les K_N valeurs ρ(m+1), ..., ρ(m+K_N) soient toutes sous le seuil.
    max_lag = min(n - 1, max(K_N * 4, int(2 * math.sqrt(n))))
    rhos = [_autocorr(x, k) for k in range(0, max_lag + 1)]

    m_hat = 1
    for m in range(0, max_lag - K_N + 1):
        window = rhos[m + 1 : m + 1 + K_N]
        if all(abs(r) < bartlett_thresh for r in window):
            m_hat = max(1, m)
            break
    else:
        # Aucune fenêtre plate trouvée -> autocorrélation persistante,
        # on prend le lag maximum raisonnable
        m_hat = max(1, max_lag // 2)

    M = min(max_lag, max(1, 2 * m_hat))

    # --- Étape 3 : Ĝ, σ̂²(0), D_SB ----------------------------------------
    gammas = [_autocov(x, k) for k in range(0, M + 1)]

    # Ĝ = Σ_{k=-M..M, k≠0} |k| λ(k/M) γ(k)
    #   = 2 Σ_{k=1..M} k λ(k/M) γ(k)   (γ est paire)
    G_hat = 0.0
    for k in range(1, M + 1):
        G_hat += 2.0 * k * _flat_top_kernel(k / M) * gammas[k]

    # σ̂²(0) = γ(0) + 2 Σ_{k=1..M} λ(k/M) γ(k)
    sigma2_0 = gammas[0]
    for k in range(1, M + 1):
        sigma2_0 += 2.0 * _flat_top_kernel(k / M) * gammas[k]

    # D_SB = 2 σ̂²(0)²  (Politis-White 2004, Eq. 9 — stationary bootstrap)
    if bootstrap_type == "circular":
        # Circular bootstrap : D_CB = (4/3) σ̂²(0)²
        D = (4.0 / 3.0) * (sigma2_0**2)
    else:
        D = 2.0 * (sigma2_0**2)

    # --- Étape 4 : b_opt = (2 G² / D)^(1/3) · n^(1/3) ---------------------
    if D <= 0 or G_hat == 0:
        # Série quasi-i.i.d. -> bloc minimal
        return max(b_min, 1)

    b_opt_f = (2.0 * (G_hat**2) / D) ** (1.0 / 3.0) * (n ** (1.0 / 3.0))
    b_opt = int(round(b_opt_f))

    # Clamp
    b_opt = max(b_min, min(b_max, b_opt))
    return b_opt


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as _np

    rng = _np.random.default_rng(42)

    # (1) Bruit blanc : b_opt devrait être petit (1-3)
    white = rng.standard_normal(500)
    b_wn = politis_white_block_length(white)
    logger.info(f"[Politis-White] White noise     n=500 -> b_opt = {b_wn}")

    # (2) AR(1) persistant (phi=0.8) : b_opt devrait être grand (~10+)
    ar1 = _np.zeros(500)
    for t in range(1, 500):
        ar1[t] = 0.8 * ar1[t - 1] + rng.standard_normal()
    b_ar = politis_white_block_length(ar1)
    logger.info(f"[Politis-White] AR(1) phi=0.8     n=500 -> b_opt = {b_ar}")

    # (3) AR(1) modéré (phi=0.3)
    ar3 = _np.zeros(500)
    for t in range(1, 500):
        ar3[t] = 0.3 * ar3[t - 1] + rng.standard_normal()
    b_ar3 = politis_white_block_length(ar3)
    logger.info(f"[Politis-White] AR(1) phi=0.3     n=500 -> b_opt = {b_ar3}")

    # (4) Série courte : fallback
    short = rng.standard_normal(10)
    b_short = politis_white_block_length(short)
    logger.info(f"[Politis-White] Série courte    n=10  -> b_opt = {b_short}")
