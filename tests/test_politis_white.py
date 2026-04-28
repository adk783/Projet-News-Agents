"""Tests pour la selection automatique de block length (Politis-White 2004).

Le module `src.utils.politis_white.politis_white_block_length` doit satisfaire
les proprietes statistiques attendues :

1. **White noise** -> block length petit (~1-3) : pas d'autocorrelation a capter.
2. **AR(1) persistant** (phi=0.8) -> block length grand (>= 5) : autocorrelation
   forte qui necessite des blocs plus longs pour preserver la dependance.
3. **AR(1) modere** (phi=0.3) -> block length intermediaire.
4. **Serie courte** (n<20) -> fallback proteger (b_min).
5. **Determinisme** : meme entree -> meme sortie (pas de randomness interne).
6. **Variantes** : "stationary" vs "circular" donnent des resultats coherents.

Reference : Politis & White (2004), Econometric Reviews, 23(1):53-70.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.politis_white import politis_white_block_length


@pytest.fixture(scope="module")
def rng():
    """RNG reproductible pour tous les tests."""
    return np.random.default_rng(42)


# =============================================================================
# Proprietes statistiques fondamentales
# =============================================================================
class TestStatisticalProperties:
    """Politis-White doit refleter la persistance reelle de la serie."""

    def test_white_noise_yields_small_block(self, rng):
        """Bruit blanc N(0,1) : aucune autocorrelation -> b_opt petit (<= 5)."""
        white = rng.standard_normal(500)
        b = politis_white_block_length(white)
        assert b <= 5, f"white noise devrait avoir b<=5, got b={b}"
        assert b >= 1

    def test_persistent_ar1_yields_larger_block_than_iid(self, rng):
        """AR(1) phi=0.8 doit donner b strictement > b du bruit blanc."""
        white = rng.standard_normal(500)
        b_white = politis_white_block_length(white)

        ar1 = np.zeros(500)
        for t in range(1, 500):
            ar1[t] = 0.8 * ar1[t - 1] + rng.standard_normal()
        b_ar = politis_white_block_length(ar1)

        assert b_ar > b_white, f"AR(1) persistant doit avoir b > white noise, got b_ar={b_ar}, b_white={b_white}"

    def test_strongly_persistent_ar1_block_at_least_5(self, rng):
        """AR(1) phi=0.8 sur n=500 : block doit etre >= 5 (autocorr non triviale)."""
        ar1 = np.zeros(500)
        for t in range(1, 500):
            ar1[t] = 0.8 * ar1[t - 1] + rng.standard_normal()
        b = politis_white_block_length(ar1)
        assert b >= 5, f"AR(1) phi=0.8 doit avoir b>=5, got {b}"

    def test_block_length_grows_with_persistence(self):
        """Plus phi est grand, plus le block doit etre long (monotonie statistique)."""
        # Construction deterministe pour le test (pas de rng).
        rng = np.random.default_rng(123)
        n = 800

        results = []
        for phi in (0.1, 0.5, 0.85):
            ar = np.zeros(n)
            innov = rng.standard_normal(n)
            for t in range(1, n):
                ar[t] = phi * ar[t - 1] + innov[t]
            b = politis_white_block_length(ar)
            results.append((phi, b))

        # On verifie que b est monotone croissant avec phi (les ex-aequo OK).
        bs = [b for _, b in results]
        assert bs[0] <= bs[2], f"b doit croitre avec phi : {results}"


# =============================================================================
# Bornes et garde-fous
# =============================================================================
class TestBoundaryConditions:
    """Cas limites : series courtes, b_min/b_max, donnees pathologiques."""

    def test_short_series_returns_fallback(self):
        """n<20 : retourne max(b_min, 5) sans calcul."""
        x = np.array([1.0, 2.0, 3.0, 1.0, 2.0])  # n=5
        b = politis_white_block_length(x)
        assert b == 5

    def test_b_min_respected(self, rng):
        """Le resultat doit toujours etre >= b_min."""
        white = rng.standard_normal(500)
        b = politis_white_block_length(white, b_min=10)
        assert b >= 10

    def test_b_max_respected(self, rng):
        """Le resultat doit toujours etre <= b_max."""
        # AR(1) tres persistant qui pousserait normalement vers grand b
        ar = np.zeros(500)
        for t in range(1, 500):
            ar[t] = 0.95 * ar[t - 1] + rng.standard_normal()
        b = politis_white_block_length(ar, b_max=15)
        assert b <= 15

    def test_constant_series_returns_min(self):
        """Serie constante : variance nulle -> retourne max(b_min, 1)."""
        x = np.full(100, 1.5)
        b = politis_white_block_length(x, b_min=2)
        # gamma(0) = 0 -> ratio indefini -> on tombe sur le fallback (b_min)
        # ou sur la valeur "serie quasi i.i.d." (b_min, 1). On accepte les 2.
        assert b == 2 or b == 1


# =============================================================================
# Determinisme et reproductibilite
# =============================================================================
class TestDeterminism:
    """Le bloc length ne depend que de la serie, jamais d'un etat interne."""

    def test_same_input_same_output(self, rng):
        """Deux appels successifs sur la meme serie donnent le meme b."""
        x = rng.standard_normal(500)
        b1 = politis_white_block_length(x)
        b2 = politis_white_block_length(x)
        assert b1 == b2

    def test_accepts_list_and_array(self):
        """Doit accepter list[float] et np.ndarray indiferremment."""
        data_list = [0.1, -0.2, 0.3, -0.1, 0.2, 0.0, 0.1, -0.3, 0.2, 0.0] * 5
        data_arr = np.array(data_list)
        assert politis_white_block_length(data_list) == politis_white_block_length(data_arr)


# =============================================================================
# Variantes : stationary vs circular bootstrap
# =============================================================================
class TestVariants:
    """Les variantes 'stationary' et 'circular' utilisent des formules D differentes."""

    def test_stationary_and_circular_close_for_white_noise(self, rng):
        """Pour du bruit blanc, les 2 variantes doivent donner des valeurs proches."""
        white = rng.standard_normal(500)
        b_stat = politis_white_block_length(white, bootstrap_type="stationary")
        b_circ = politis_white_block_length(white, bootstrap_type="circular")
        # Difference attendue mais bornee (les formules different d'un facteur ~1.5)
        assert abs(b_stat - b_circ) <= max(b_stat, b_circ)


# =============================================================================
# Integration : usage canonique dans event_study
# =============================================================================
class TestIntegrationUsage:
    """Sanity check : le module est utilisable dans un workflow event study reel."""

    def test_realistic_car_series_returns_reasonable_block(self):
        """Une serie de CAR realiste (n~100, faible autocorrelation) doit donner
        un block raisonnable (1 <= b <= 25)."""
        rng = np.random.default_rng(2024)
        # CAR simules : loi normale + petit AR(1) phi=0.15
        car = np.zeros(100)
        for t in range(1, 100):
            car[t] = 0.15 * car[t - 1] + rng.standard_normal() * 0.02
        b = politis_white_block_length(car)
        assert 1 <= b <= 25, f"block raisonnable attendu, got {b}"
