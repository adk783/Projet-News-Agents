"""
yolo_classifier.py — Classificateur de risque de trading
Inspiré du "YOLO classifier" de Claude Code (Anthropic leak — autoApprove logic)

PRINCIPE :
  Ce n'est PAS un appel LLM supplémentaire.
  C'est un scorer multi-facteurs déterministe qui extrait des features quantifiées
  du scratchpad XML et des métriques déjà calculées dans le pipeline.
  Il tourne en microsecondes — zéro latence, zéro coût API.

ARCHITECTURE :
  Le YOLO Classifier s'intercale entre l'Agent Consensus (étape 6) et
  une hypothétique étape d'exécution d'ordre boursier.

  FAIBLE (risk_score < 0.35) → auto-exécution autorisée
  MOYEN  (0.35 ≤ score < 0.65) → exécution avec warning obligatoire
  ELEVE  (score ≥ 0.65) → interruption, approbation humaine requise

FEATURES EXTRAITES :
  1. HTC Score (Holistic Trajectory Calibration)
       Analyse 3D de la trajectoire complète de confiance du débat :
       micro-stabilité + macro-dynamique + position finale.
       Remplace les anciennes features confidence_variance + debate_convergence.
  2. finbert_align         : FinBERT et le signal LLM sont-ils d'accord?
                             (désaccord = signal peu fiable)
  3. absa_ambiguity        : Le ratio ABSA est-il proche de 0.5?
                             (plus proche de 0.5 = plus ambigu)
  4. market_volatility     : Variation 5j du cours (marché agité = plus risqué)
  5. signal_strength       : impact_strength + consensus_rate (force du signal)
  6. TrustTrade            : Pondération dynamique croisée numérique/sémantique
                             + décote sélective sur signaux faiblement justifiés
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Optional

from src.utils.htc_calibrator import HTCResult, compute_htc_score, htc_to_risk_contribution
from src.utils.information_density import IDScoreResult, compute_id_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes régimes de marché (calibrées sur L12 — audit 2026-04-21)
# Ref : Hamilton (1989) Regime-Switching, Lo (2004) Adaptive Markets
# Seuils empiriques validés : 0% accuracy en SIDEWAYS sur 19 trades
# ---------------------------------------------------------------------------

REGIME_BULL = "BULL"
REGIME_BEAR = "BEAR"
REGIME_SIDEWAYS = "SIDEWAYS"
REGIME_HIGH_VOL = "HIGH_VOL"
REGIME_UNKNOWN = "UNKNOWN"

# Seuils alignés avec evaluate_historical_backtest.py (definition "vol-adjusted")
# SIDEWAYS = rendement 20j < 0.5 · volatilité historique annualisée (bruit statistique)
# Cette définition est stochastiquement cohérente (pas un seuil fixe arbitraire).
# Ref : Ang & Bekaert (2002), "International Asset Allocation with Regime Shifts"
SPY_BULL_THRESHOLD = 3.0  # fallback si spy_20d_vol absente
SPY_BEAR_THRESHOLD = -3.0  # fallback si spy_20d_vol absente
VOL_ADJ_RET_THRESHOLD = 0.5  # k · σ (k=0.5, Ang-Bekaert)
VIX_LOW_THRESHOLD = 25.0  # VIX < 25 => regime calme
VIX_HIGH_THRESHOLD = 30.0  # VIX > 30 => HIGH_VOL (prioritaire)
BEAR_VIX_FLOOR = 20.0  # VIX > 20 requis pour confirmer BEAR

# Penalite de risque appliquee en SIDEWAYS (escalade forcee vers ELEVE)
SIDEWAYS_RISK_PENALTY = 0.40


# ---------------------------------------------------------------------------
# Niveaux de risque
# ---------------------------------------------------------------------------

RISK_FAIBLE = "FAIBLE"  # Auto-execution possible
RISK_MOYEN = "MOYEN"  # Execution avec avertissement
RISK_ELEVE = "ELEVE"  # Approbation humaine obligatoire

SEUIL_FAIBLE = 0.35
SEUIL_ELEVE = 0.60


# ---------------------------------------------------------------------------
# Résultat du classificateur
# ---------------------------------------------------------------------------


@dataclass
class YOLODecision:
    """Décision du classificateur de risque."""

    risk_level: str  # FAIBLE | MOYEN | ELEVE
    risk_score: float  # 0.0 (sûr) → 1.0 (très risqué)
    auto_execute: bool  # True si FAIBLE
    requires_human: bool  # True si ELEVE
    reasons: list[str]  # Explications humainement lisibles
    features: dict = field(default_factory=dict)  # Features brutes (audit)
    htc_result: Optional[HTCResult] = None  # Résultat HTC détaillé
    trust_trade_score: float = 0.0  # Score TrustTrade [0, 1]
    market_regime: str = REGIME_UNKNOWN  # Regime detecte au moment de la decision
    regime_veto: bool = False  # True si signal override par filtre regime

    def log_summary(self, ticker: str) -> None:
        """Log la décision de manière lisible."""
        icon = {"FAIBLE": "[OK]", "MOYEN": "[!!]", "ELEVE": "[XX]"}[self.risk_level]
        logger.info(
            "%s [YOLO] %s | Risque : %s (score=%.2f) | Auto-exec : %s",
            icon,
            ticker,
            self.risk_level,
            self.risk_score,
            "OUI" if self.auto_execute else "NON",
        )
        for reason in self.reasons:
            logger.info("       %s", reason)


# ---------------------------------------------------------------------------
# Feature 1 : Extraction des scores de confiance depuis le scratchpad XML
# ---------------------------------------------------------------------------


def _extract_confidence_scores(scratchpad_xml: str) -> list[float]:
    """
    Parse le scratchpad XML et extrait tous les scores <confidence>.
    Exemple : [0.9, 0.85, 0.9, 0.8, 0.88, 0.85, 0.6, 0.7, 0.75]
    (3 agents × 3 tours = jusqu'à 9 valeurs)
    """
    pattern = re.compile(r"\[(?:Tour \d+)\][^\[]*\[confiance:\s*([\d.]+)\]")
    matches = pattern.findall(scratchpad_xml)
    scores = []
    for m in matches:
        try:
            scores.append(float(m))
        except ValueError:
            pass
    return scores


def _extract_confidence_by_round(scratchpad_xml: str) -> dict[int, list[float]]:
    """
    Extrait les scores de confiance groupés par tour.
    Retourne {1: [0.9, 0.8, 0.7], 2: [0.95, 0.85, 0.65], 3: [0.9, 0.88, 0.75]}
    """
    pattern = re.compile(r"\[Tour (\d+)\][^\[]*\[confiance:\s*([\d.]+)\]")
    by_round: dict[int, list[float]] = {}
    for m in pattern.finditer(scratchpad_xml):
        tour = int(m.group(1))
        score = float(m.group(2))
        by_round.setdefault(tour, []).append(score)
    return by_round


# ---------------------------------------------------------------------------
# Feature 2 : Variance des confiances (mesure du conflit inter-agents)
# ---------------------------------------------------------------------------


def _compute_confidence_variance(scores: list[float]) -> float:
    """
    Variance des scores de confiance extraits du scratchpad.
    Haute variance = agents très en désaccord = débat conflictuel = plus risqué.
    Retourne 0.0 si pas assez de scores.
    """
    if len(scores) < 2:
        return 0.0
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return round(math.sqrt(variance), 4)  # On retourne l'écart-type


# ---------------------------------------------------------------------------
# Feature 3 : Convergence du débat
# La confiance des agents augmente-t-elle au fil des tours ?
# Si oui → convergence → moins risqué
# Si non → divergence → plus risqué
# ---------------------------------------------------------------------------


def _compute_debate_convergence(by_round: dict[int, list[float]]) -> float:
    """
    Calcule si la confiance a tendance à augmenter ou diminuer au fil des tours.
    Retourne un score de convergence : positif = convergence, négatif = divergence.
    Normalisé entre -1.0 et +1.0.
    """
    rounds = sorted(by_round.keys())
    if len(rounds) < 2:
        return 0.0  # Pas assez de tours pour mesurer

    # Moyenne de confiance par tour
    avg_by_round = {r: sum(by_round[r]) / len(by_round[r]) for r in rounds}

    # Tendance linéaire (pente = convergence si positive)
    first_round_avg = avg_by_round[rounds[0]]
    last_round_avg = avg_by_round[rounds[-1]]
    delta = last_round_avg - first_round_avg  # Positif = convergence

    return round(max(-1.0, min(1.0, delta * 5)), 4)  # Amplify et clip


# ---------------------------------------------------------------------------
# Feature 4 : Alignement FinBERT / Signal LLM
# ---------------------------------------------------------------------------


def _compute_finbert_alignment(signal_final: str, score_finbert: float) -> float:
    """
    Mesure si FinBERT et l'agent Consensus sont d'accord sur la direction.
    FinBERT > 0.5 = sentiment positif (favorable à Achat).
    FinBERT < 0.5 = sentiment négatif (favorable à Vente).

    Retourne 1.0 si alignement parfait, 0.0 si désaccord total.
    """
    finbert_direction = "positif" if score_finbert >= 0.5 else "negatif"
    if signal_final == "Achat" and finbert_direction == "positif":
        return 1.0
    elif signal_final == "Vente" and finbert_direction == "negatif":
        return 1.0
    elif signal_final == "Neutre":
        return 0.5  # Neutre est toujours ambigu
    else:
        # Désaccord : FinBERT dit positif mais LLM dit Vente (ou vice-versa)
        disagreement = abs(score_finbert - 0.5)  # Distance de la neutralité
        return round(0.5 - disagreement, 4)  # Plus le désaccord est fort, plus on s'approche de 0.0


# ---------------------------------------------------------------------------
# Feature 5 : Ambiguïté du ratio ABSA
# ---------------------------------------------------------------------------


def _compute_absa_ambiguity(absa_result: dict) -> float:
    """
    Mesure l'ambiguïté de la news selon l'ABSA.
    Un ratio de 0.5 = autant de positif que de négatif = très ambigu.
    Un ratio de 0.9 ou 0.1 = signal très clair = peu ambigu.

    Retourne 0.0 = clair / 1.0 = très ambigu.
    """
    aspects = absa_result.get("aspects", [])
    nb_pos = len([a for a in aspects if a.get("sentiment") == "positive"])
    nb_neg = len([a for a in aspects if a.get("sentiment") == "negative"])
    total = nb_pos + nb_neg

    if total == 0:
        return 0.5  # Aucune info ABSA = ambigu par défaut

    ratio = nb_pos / total
    # Distance à 0.5 → plus on est proche de 0.5, plus c'est ambigu
    ambiguity = 1.0 - abs(ratio - 0.5) * 2  # 0 si ratio=0 ou 1, 1.0 si ratio=0.5
    return round(ambiguity, 4)


# ---------------------------------------------------------------------------
# Feature 6 : Volatilité du marché
# ---------------------------------------------------------------------------


def _compute_market_volatility(contexte_marche: dict) -> float:
    """
    Utilise la variation 5j comme proxy de la volatilité court terme.
    Une forte variation (quel que soit le sens) = marché plus risqué.
    Normalisé entre 0.0 et 1.0 (plafond à ±15% de variation).
    """
    variation = abs(contexte_marche.get("variation_5d", 0.0))
    # Normalise : 0% = 0.0, 15%+ = 1.0
    return round(min(1.0, variation / 15.0), 4)


# ---------------------------------------------------------------------------
# Calcul du score de risque global
# ---------------------------------------------------------------------------


def _compute_trust_trade_score(
    signal_final: str,
    score_finbert: float,
    absa_result: dict,
    argument_dominant: str,
    contexte_marche: dict,
) -> tuple[float, list[str]]:
    """
    TrustTrade — Pondération dynamique croisée numérique et sémantique.

    Principe :
      1. Accord numérique   : FinBERT et ABSA pointent dans la même direction ?
      2. Accord sémantique  : l'argument dominant cite des faits cohérents avec l'ABSA ?
      3. Décote sélective   : ID_score < seuil ET contexte marché faible → pénalité 30%

    Retourne (trust_score [0, 1], reasons) :
      trust_score = 1.0 → signal très fiable (accord croisé complet)
      trust_score = 0.0 → signal non fiable (désaccords multiples)
    """
    reasons: list[str] = []
    trust_score = 1.0

    # --- 1. Accord numérique FinBERT ↔ ABSA ---
    aspects = absa_result.get("aspects", [])
    nb_pos = len([a for a in aspects if a.get("sentiment") == "positive"])
    nb_neg = len([a for a in aspects if a.get("sentiment") == "negative"])
    total = nb_pos + nb_neg
    absa_ratio = (nb_pos / total) if total > 0 else 0.5

    finbert_direction = "positif" if score_finbert >= 0.5 else "negatif"
    absa_direction = "positif" if absa_ratio > 0.5 else ("negatif" if absa_ratio < 0.5 else "neutre")

    numerical_agreement = finbert_direction == absa_direction
    if not numerical_agreement and absa_direction != "neutre":
        trust_score -= 0.20
        reasons.append(
            f"Désaccord numérique FinBERT({finbert_direction}) ↔ ABSA({absa_direction}) "
            f"[FinBERT={score_finbert:.2f}, ABSA_ratio={absa_ratio:.2f}]"
        )

    # --- 2. Accord sémantique : l'argument cite-t-il des faits chiffrés ? ---
    id_result = compute_id_score(argument_dominant)
    if id_result.is_penalized:
        trust_score -= 0.15
        reasons.append(
            f"Argument peu dense en faits (ID={id_result.id_score:.4f} < seuil) → signal insuffisamment documenté"
        )

    # --- 3. Décote sélective : contexte marché faible + ID faible ---
    # Si le marché est peu volatile (signal peu informatif) ET l'argument est creux
    market_vol = abs(contexte_marche.get("variation_5d", 0.0))
    if id_result.is_penalized and market_vol < 0.5:
        trust_score -= 0.15
        reasons.append(
            f"Décote TrustTrade combinée : contexte faible (variation_5j={market_vol:.1f}%) "
            f"+ argument peu factuel (ID={id_result.id_score:.4f})"
        )

    # Cohérence directionnelle signal final ↔ FinBERT (bonus si accord)
    if signal_final == "Achat" and finbert_direction == "positif":
        trust_score = min(1.0, trust_score + 0.10)
    elif signal_final == "Vente" and finbert_direction == "negatif":
        trust_score = min(1.0, trust_score + 0.10)

    trust_score = round(max(0.0, min(1.0, trust_score)), 4)

    if not reasons:
        reasons.append("TrustTrade : accord numérique et sémantique croisé validé")

    return trust_score, reasons


def _compute_risk_score(
    htc_risk: float,
    finbert_align: float,
    absa_ambiguity: float,
    market_volatility: float,
    impact_strength: float,
    consensus_rate: float,
    trust_trade_score: float,
) -> tuple[float, list[str]]:
    """
    Calcule le score de risque global (0.0 à 1.0).
    Retourne (risk_score, liste de raisons) pour l'audit.

    Pondération mise à jour avec HTC et TrustTrade :
    - Force du signal    : 25% (impact_strength + consensus_rate)
    - TrustTrade         : 20% (accord croisé numérique + sémantique)
    - Alignement LLM/ML  : 20% (finbert_align)
    - Ambiguité news     : 15% (absa_ambiguity)
    - HTC (trajectoire)  : 15% (micro-stabilité + macro-dynamique + position finale)
    - Volatilité marché  :  5% (market_volatility)
    """
    reasons = []
    components = {}

    # 1. Score de force du signal (inversé : plus fort = moins risqué)
    signal_strength_avg = (impact_strength + consensus_rate) / 2
    signal_risk = 1.0 - signal_strength_avg
    components["signal_risk"] = signal_risk * 0.25
    if signal_strength_avg < 0.50:
        reasons.append(
            f"Signal faible (Force={impact_strength:.2f}, Consensus={consensus_rate:.2f}) — signal peu actionnable"
        )

    # 2. TrustTrade — accord croisé numérique/sémantique (inversé : plus fiable = moins risqué)
    trust_risk = 1.0 - trust_trade_score
    components["trust_risk"] = trust_risk * 0.20
    if trust_trade_score < 0.65:
        reasons.append(f"TrustTrade faible (score={trust_trade_score:.2f}) — accord croisé insuffisant")

    # 3. Désalignement FinBERT/LLM (inversé : plus aligné = moins risqué)
    align_risk = 1.0 - finbert_align
    components["align_risk"] = align_risk * 0.20
    if finbert_align < 0.6:
        reasons.append(f"Désaccord FinBERT/LLM (align={finbert_align:.2f}) — ML et LLM contredits")

    # 4. Ambiguité ABSA (direct : plus ambigu = plus risqué)
    components["absa_risk"] = absa_ambiguity * 0.15
    if absa_ambiguity > 0.65:
        reasons.append(f"News ambiguë (ABSA ambiguity={absa_ambiguity:.2f}) — aspects équilibrés")

    # 5. HTC — Trajectoire du débat (remplace confidence_std + convergence)
    components["htc_risk"] = htc_risk * 0.15
    if htc_risk > 0.55:
        reasons.append(
            f"Trajectoire de débat instable (HTC_risk={htc_risk:.2f}) — agents divergents ou confiance faible"
        )

    # 6. Volatilité marché (réduit à 5% car couvert partiellement par TrustTrade)
    components["market_risk"] = market_volatility * 0.05
    if market_volatility > 0.5:
        reasons.append(f"Marché volatile (variation 5j={market_volatility * 15:.1f}%)")

    total_risk = round(sum(components.values()), 4)

    # Hard-rule : Si contradiction forte entre quantitatif et sémantique, risque ÉLEVÉ
    if finbert_align < 0.3:
        reasons.append("CRITIQUE : Contradiction forte entre le modèle quantitatif (FinBERT) et le LLM. Risque majoré.")
        total_risk += 0.35

    if not reasons:
        reasons.append("Tous les indicateurs sont dans les seuils normaux")

    return total_risk, reasons


# ---------------------------------------------------------------------------
# Filtre de régime de marché — basé sur L12 audit (21 avril 2026)
# ---------------------------------------------------------------------------


def classify_market_regime(
    spy_20d_return: Optional[float] = None,
    vix: Optional[float] = None,
    spy_20d_vol: Optional[float] = None,
) -> str:
    """
    Classifie le regime de marche courant selon une définition vol-adjusted
    cohérente avec evaluate_historical_backtest.py (critique de l'audit Quant).

    Priorite de classification (ordre strict) :
      1. HIGH_VOL  : VIX > 30 (risque systémique — toujours prioritaire)
      2. BULL      : SPY_20d > 0.5·σ_20d ET VIX < 25
      3. BEAR      : SPY_20d < -0.5·σ_20d ET VIX > 20
      4. SIDEWAYS  : rendement 20j < 0.5·σ_20d (bruit, non-directionnel)
      5. UNKNOWN   : données non disponibles

    Si spy_20d_vol n'est pas fournie, on retombe sur des seuils fixes
    ±3% (legacy) avec un warning implicite (moins rigoureux).

    Ref : Ang & Bekaert (2002), "International Asset Allocation with Regime
          Shifts." Review of Financial Studies, 15(4).
          Hamilton (1989) Regime-Switching, Lo (2004) Adaptive Markets.

    Args:
        spy_20d_return : rendement SPY sur 20 jours glissants (en %)
        vix            : indice VIX courant
        spy_20d_vol    : volatilité annualisée 20j du SPY (en %)

    Returns:
        str : BULL | BEAR | HIGH_VOL | SIDEWAYS | UNKNOWN
    """
    if spy_20d_return is None and vix is None:
        return REGIME_UNKNOWN

    # HIGH_VOL — prioritaire : VIX > 30 indique un risque systemique
    if vix is not None and vix > VIX_HIGH_THRESHOLD:
        return REGIME_HIGH_VOL

    # Définition vol-adjusted si σ disponible
    if spy_20d_return is not None and spy_20d_vol is not None and spy_20d_vol > 0:
        vol_threshold = VOL_ADJ_RET_THRESHOLD * spy_20d_vol

        if spy_20d_return > vol_threshold and (vix is None or vix < VIX_LOW_THRESHOLD):
            return REGIME_BULL
        if spy_20d_return < -vol_threshold and (vix is None or vix > BEAR_VIX_FLOOR):
            return REGIME_BEAR
        return REGIME_SIDEWAYS

    # Fallback hard-threshold (legacy) si spy_20d_vol manquante
    if spy_20d_return is not None and spy_20d_return > SPY_BULL_THRESHOLD and (vix is None or vix < VIX_LOW_THRESHOLD):
        return REGIME_BULL
    if spy_20d_return is not None and spy_20d_return < SPY_BEAR_THRESHOLD and (vix is None or vix > BEAR_VIX_FLOOR):
        return REGIME_BEAR
    if spy_20d_return is not None:
        return REGIME_SIDEWAYS

    return REGIME_UNKNOWN


# ---------------------------------------------------------------------------
# Point d'entrée principal — classify_risk()
# Intercalé entre l'Agent Consensus et l'exécution
# ---------------------------------------------------------------------------


def classify_risk(
    signal_final: str,
    consensus_rate: float,
    impact_strength: float,
    scratchpad_xml: str,
    absa_result: dict,
    score_finbert: float,
    contexte_marche: dict,
    argument_dominant: str = "",
    seuil_faible: float | None = None,
    seuil_eleve: float | None = None,
    spy_20d_return: Optional[float] = None,
    vix: Optional[float] = None,
    spy_20d_vol: Optional[float] = None,
    processing_time_ms: Optional[float] = None,
) -> YOLODecision:
    """
    Classificateur de risque YOLO — version améliorée avec HTC, TrustTrade,
    seuils dynamiques Arrow-Pratt et filtre de régime de marche.

    A appeler après l'Agent Consensus, avant toute exécution d'ordre.

    Args:
        signal_final      : "Achat" | "Vente" | "Neutre"
        consensus_rate    : taux d'alignement Data (0.0-1.0)
        impact_strength   : force du signal (0.0-1.0)
        scratchpad_xml    : XML du débat multi-agent (source des features débat)
        absa_result       : {"aspects": [...]} issu de l'agent ABSA
        score_finbert     : probabilité FinBERT (0.0-1.0)
        contexte_marche   : {"current_price", "volume", "variation_5d"}
        argument_dominant : argument gagnant du Consensus (pour TrustTrade ID Score)
        seuil_faible      : seuil FAIBLE (défaut: SEUIL_FAIBLE=0.35). Overridé par
                            profil Arrow-Pratt si fourni depuis InvestorProfile.
        seuil_eleve       : seuil ELEVE (défaut: SEUIL_ELEVE=0.60). Idem.
        spy_20d_return    : rendement SPY sur 20 jours glissants (%), pour filtre regime.
        vix               : indice VIX courant, pour filtre regime.

    Returns:
        YOLODecision avec risk_level, risk_score, auto_execute, reasons,
        market_regime et regime_veto.
    """
    # ---------------------------------------------------------------------------
    # FILTRE REGIME DE MARCHE — Priorite absolue (audit L12)
    # Si le marche est SIDEWAYS, aucun signal directionnel n'est fiable.
    # Accuracy SIDEWAYS = 0% sur 19 trades (audit 21 avril 2026).
    # Ref : Lo (2004) Adaptive Markets, Hamilton (1989) Regime-Switching
    # ---------------------------------------------------------------------------
    market_regime = classify_market_regime(
        spy_20d_return=spy_20d_return,
        vix=vix,
        spy_20d_vol=spy_20d_vol,
    )
    regime_veto = False

    if market_regime == REGIME_SIDEWAYS and signal_final in ("Achat", "Vente"):
        logger.warning(
            "[YOLO-Regime] Marche SIDEWAYS detecte (SPY_20d=%.1f%%, VIX=%.1f) — "
            "application d'une penalite Bayesienne (+%.0f%% risque) au lieu d'un veto dur.",
            spy_20d_return or 0.0,
            vix or 0.0,
            SIDEWAYS_RISK_PENALTY * 100,
        )
        # On ne force PLUS le signal a Neutre (Critique B : Bayesian Prior)
        regime_veto = True

    if market_regime == REGIME_HIGH_VOL:
        logger.warning(
            "[YOLO-Regime] HIGH_VOL detecte (VIX=%.1f > %.0f) — escalade du risque.", vix or 0.0, VIX_HIGH_THRESHOLD
        )

    # ---------------------------------------------------------------------------
    # Feature 1 : HTC — Holistic Trajectory Calibration
    # Remplace l'ancienne paire (confidence_std, debate_convergence)
    # ---------------------------------------------------------------------------
    htc_result = compute_htc_score(scratchpad_xml)
    htc_risk = htc_to_risk_contribution(htc_result)

    logger.debug(
        "[YOLO] HTC Score=%.3f (micro=%.3f, macro=%.3f, final=%.3f) -> HTC_risk=%.3f",
        htc_result.htc_score,
        htc_result.micro_stability,
        htc_result.macro_dynamic,
        htc_result.final_position,
        htc_risk,
    )

    # ---------------------------------------------------------------------------
    # Features classiques
    # ---------------------------------------------------------------------------
    finbert_align = _compute_finbert_alignment(signal_final, score_finbert)
    absa_ambiguity = _compute_absa_ambiguity(absa_result)
    market_volatility = _compute_market_volatility(contexte_marche)

    # ---------------------------------------------------------------------------
    # Feature 5 : TrustTrade — pondération dynamique croisée
    # ---------------------------------------------------------------------------
    trust_trade_score, trust_reasons = _compute_trust_trade_score(
        signal_final=signal_final,
        score_finbert=score_finbert,
        absa_result=absa_result,
        argument_dominant=argument_dominant,
        contexte_marche=contexte_marche,
    )

    logger.debug("[YOLO] TrustTrade score=%.3f", trust_trade_score)

    # ---------------------------------------------------------------------------
    # Score de risque global
    # ---------------------------------------------------------------------------
    risk_score, reasons = _compute_risk_score(
        htc_risk=htc_risk,
        finbert_align=finbert_align,
        absa_ambiguity=absa_ambiguity,
        market_volatility=market_volatility,
        impact_strength=impact_strength,
        consensus_rate=consensus_rate,
        trust_trade_score=trust_trade_score,
    )

    # Ajoute les raisons TrustTrade si elles signalent un problème
    for r in trust_reasons:
        if "validé" not in r:
            reasons.append(r)

    # Penalite de risque supplementaire pour les regimes HIGH_VOL et SIDEWAYS
    if market_regime == REGIME_SIDEWAYS and regime_veto:
        risk_score = min(1.0, risk_score + SIDEWAYS_RISK_PENALTY)
        reasons.append(
            f"[Regime SIDEWAYS] Application d'un Prior Bayesien de risque "
            f"(SPY_20d={spy_20d_return:.1f}%, VIX={vix:.1f}). "
            f"Penalite +{SIDEWAYS_RISK_PENALTY:.0%} sur les signaux directionnels."
            if spy_20d_return is not None and vix is not None
            else "[Regime SIDEWAYS] Prior Bayesien : penalite de risque appliquee."
        )
    elif market_regime == REGIME_HIGH_VOL:
        risk_score = min(1.0, risk_score + 0.20)
        reasons.append(
            f"[Regime HIGH_VOL] VIX={vix:.1f} > {VIX_HIGH_THRESHOLD:.0f} — "
            f"volatilite systemique detectee. Escalade de risque +20%%."
            if vix is not None
            else "[Regime HIGH_VOL] Volatilite systemique detectee. Escalade +20%%."
        )

    # ---------------------------------------------------------------------------
    # Critique E : Latency Penalty (Alpha Decay)
    # Le seuil de decroissance depend de la liquidite de l'actif (Volume).
    # Actif tres liquide (AAPL) -> decroissance rapide. Illiquide -> lente.
    # ---------------------------------------------------------------------------
    if processing_time_ms is not None:
        # Recuperation du volume, par defaut 10M si absent
        vol = float(contexte_marche.get("volume", 10_000_000)) if isinstance(contexte_marche, dict) else 10_000_000
        vol = max(100_000, vol)  # Eviter log(0) ou valeurs trop faibles
        import math

        liquidity_factor = math.log10(vol)  # ex: 10M -> 7.0

        # Seuil de latence dynamique: Plus c'est liquide, plus le seuil est bas
        # Base: 5000ms pour une liquidite moyenne (log10=7)
        latency_threshold_ms = 5000.0 * (7.0 / liquidity_factor)

        if processing_time_ms > latency_threshold_ms:
            # Penalite max 30%, echelle 10000ms
            latency_penalty = min(0.30, (processing_time_ms - latency_threshold_ms) / 10000.0 * 0.15)
            risk_score = min(1.0, risk_score + latency_penalty)
            reasons.append(
                f"[Latence] Actif très liquide (Vol {vol / 1e6:.1f}M) : Traitement {processing_time_ms / 1000:.1f}s "
                f"depasse seuil microstructure ({latency_threshold_ms / 1000:.1f}s). Penalite +{latency_penalty * 100:.1f}%."
            )

    # Signal Neutre -> toujours MOYEN minimum (on ne peut pas auto-executer 'Neutre')
    _seuil_faible = seuil_faible if seuil_faible is not None else SEUIL_FAIBLE
    _seuil_eleve = seuil_eleve if seuil_eleve is not None else SEUIL_ELEVE

    if signal_final == "Neutre" and risk_score < _seuil_eleve:
        risk_score = max(risk_score, _seuil_faible)
        reasons.append("Signal 'Neutre' : approbation humaine recommandee avant toute action")

    # Classification avec seuils dynamiques (Arrow-Pratt si profil fourni)
    if risk_score < _seuil_faible:
        risk_level = RISK_FAIBLE
    elif risk_score < _seuil_eleve:
        risk_level = RISK_MOYEN
    else:
        risk_level = RISK_ELEVE

    return YOLODecision(
        risk_level=risk_level,
        risk_score=risk_score,
        auto_execute=(risk_level == RISK_FAIBLE),
        requires_human=(risk_level == RISK_ELEVE),
        reasons=reasons,
        htc_result=htc_result,
        trust_trade_score=trust_trade_score,
        market_regime=market_regime,
        regime_veto=regime_veto,
        features={
            "htc_score": htc_result.htc_score,
            "htc_micro": htc_result.micro_stability,
            "htc_macro": htc_result.macro_dynamic,
            "htc_final_pos": htc_result.final_position,
            "htc_risk": htc_risk,
            "trust_trade_score": trust_trade_score,
            "finbert_align": finbert_align,
            "absa_ambiguity": absa_ambiguity,
            "market_volatility": market_volatility,
            "impact_strength": impact_strength,
            "consensus_rate": consensus_rate,
            "n_htc_samples": htc_result.n_samples,
            "market_regime": market_regime,
            "regime_veto": regime_veto,
            "spy_20d_return": spy_20d_return,
            "vix_at_decision": vix,
            "processing_time_ms": processing_time_ms,
        },
    )
