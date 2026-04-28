"""
temporal_fence.py -- Garde temporelle pour l'intégrité du backtesting
=======================================================================
OBJECTIF :
  Quand on évalue le pipeline sur des articles historiques (ex: 2024-02-01),
  s'assurer que les agents n'ont pas accès à des informations post-publication.

PROBLÈMES ÉVITÉS :
  1. Biais de visibilité future : le `contexte_marche` (prix actuel, variation 5j)
     transmis à l'agent contient des données d'AUJOURD'HUI, pas de 2024-02-01.
     Un agent voyant le cours Apple à 195$ en 2024 sait déjà que l'action a monté.

  2. Biais de survie : on ne teste que sur des entreprises qui existent encore.
     Un ticker délité (faillite, acquisition, retrait de cote) entre la date de
     l'article et aujourd'hui n'est plus testable --- ce qui crée une sur-représentation
     des entreprises "survivantes" et surestime les performances du système.
     Implémenté via : check_survivor_bias(ticker, article_date_str)

  3. Contexte contamination : la mémoire AutoDream peut contenir des décisions
     d'articles plus récents qui influencent l'évaluation d'articles anciens.

SOLUTION :
  - Pour le contexte marché : reconstruire le contexte à la date exacte via yfinance
  - Pour la mémoire AutoDream : désactiver l'injection de mémoire en mode eval
  - Pour le biais de survie : détecter les tickers délités et ajouter un warning
  - Ajouter un rapport d'intégrité temporelle à chaque run d'évaluation

USAGE :
  from src.utils.temporal_fence import build_historical_context, TemporalIntegrityReport

  ctx, report = get_clean_eval_context("AAPL", "2024-02-01")
  # ctx est sûr à passer aux agents --- il reflète vraiment 2024-02-01
  # report.survivor_bias contient l'analyse de biais de survie
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Biais de survie
# ---------------------------------------------------------------------------


@dataclass
class SurvivorBiasReport:
    """
    Rapport d'analyse du biais de survie pour un ticker donné.

    Un ticker est considéré comme potentiellement délité si yfinance
    ne retourne aucune donnée historique entre la date de l'article
    et aujourd'hui, alors qu'il en avait à la date de publication.
    """

    ticker: str
    article_date: str
    is_potentially_delisted: bool  # True = ticker probablement délité
    had_data_at_article_date: bool  # True = données présentes à la date de l'article
    has_data_today: bool  # True = toujours coté aujourd'hui
    price_at_article: Optional[float] = None  # Cours à la date de l'article
    price_today: Optional[float] = None  # Cours actuel (None si délité)
    days_since_article: int = 0
    bias_risk: str = "FAIBLE"  # FAIBLE | MODERE | ELEVE
    notes: list[str] = field(default_factory=list)


def check_survivor_bias(ticker: str, article_date_str: str) -> SurvivorBiasReport:
    """
    Détecte si un ticker est potentiellement affecté par le biais de survie.

    Stratégie détection (sans base de tickers délistés externe) :
      1. Vérifie qu'il y avait des données à la date de l'article (préchauffé)
      2. Vérifie qu'il y a des données récentes (aujourd'hui ± 7j)
      3. Si (1) OK et (2) vide → ticker probablement délisté entre les deux dates
      4. Si aucune donnée du tout → ticker invalide (ticker invalide / testé hors-marché)

    Catbais de risque :
      - FAIBLE  : toujours coté , données disponibles aux deux dates
      - MODERE  : délisté détecté ou volume très faible à l'article
      - ELEVE   : aucune donnée du tout (ticker introuvable)

    Args:
        ticker           : Symbole boursier (ex: "AAPL", "SVB", "BYND")
        article_date_str : Date ISO de publication de l'article

    Returns:
        SurvivorBiasReport avec l'analyse complète.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("[SurvivorBias] yfinance non disponible")
        return SurvivorBiasReport(
            ticker=ticker,
            article_date=article_date_str,
            is_potentially_delisted=False,
            had_data_at_article_date=False,
            has_data_today=False,
            bias_risk="INCONNU",
            notes=["yfinance non installé"],
        )

    notes = []
    today = datetime.now()
    try:
        art_date = datetime.strptime(article_date_str[:10], "%Y-%m-%d")
    except ValueError:
        return SurvivorBiasReport(
            ticker=ticker,
            article_date=article_date_str,
            is_potentially_delisted=False,
            had_data_at_article_date=False,
            has_data_today=False,
            bias_risk="INCONNU",
            notes=[f"Date invalide : {article_date_str}"],
        )

    days_since = (today - art_date).days

    # --- 1. Données à la date de l'article ---
    art_start = (art_date - timedelta(days=7)).strftime("%Y-%m-%d")
    art_end = (art_date + timedelta(days=2)).strftime("%Y-%m-%d")
    try:
        hist_art = yf.Ticker(ticker).history(start=art_start, end=art_end)
        had_data_art = not hist_art.empty
        price_at_art = float(hist_art["Close"].iloc[-1]) if had_data_art else None
        if had_data_art and hist_art["Volume"].iloc[-1] == 0:
            notes.append(f"Volume nul à la date article ({article_date_str}) — ticker potentiellement illiquide")
    except Exception as ex:
        had_data_art = False
        price_at_art = None
        notes.append(f"Erreur yfinance (article) : {ex}")

    # --- 2. Données récentes (7 derniers jours) ---
    today_start = (today - timedelta(days=10)).strftime("%Y-%m-%d")
    today_end = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        hist_today = yf.Ticker(ticker).history(start=today_start, end=today_end)
        has_data_today = not hist_today.empty
        price_today = float(hist_today["Close"].iloc[-1]) if has_data_today else None
    except Exception as ex:
        has_data_today = False
        price_today = None
        notes.append(f"Erreur yfinance (aujourd'hui) : {ex}")

    # --- 3. Analyse du biais de survie ---
    is_potentially_delisted = had_data_art and not has_data_today

    if is_potentially_delisted:
        notes.append(
            f"BIAIS DE SURVIE DÉTECTÉ : '{ticker}' avait des données le {article_date_str} "
            f"mais n'est plus disponible aujourd'hui ({today.strftime('%Y-%m-%d')}). "
            f"Probable décotation, faillite ou rachat."
        )
        bias_risk = "ELEVE"
    elif not had_data_art and not has_data_today:
        notes.append(f"Ticker '{ticker}' introuvable aux deux dates — symbole invalide ?")
        bias_risk = "ELEVE"
    elif not had_data_art and has_data_today:
        notes.append(
            f"Ticker '{ticker}' : pas de données à la date de l'article ({article_date_str}) "
            f"mais présent aujourd'hui. Possible IPO postérieure."
        )
        bias_risk = "MODERE"
    elif days_since < 30:
        notes.append("Article récent (< 30j) — biais de survie non applicable.")
        bias_risk = "FAIBLE"
    else:
        notes.append(f"Ticker '{ticker}' toujours coté. Biais de survie non détecté.")
        bias_risk = "FAIBLE"

    logger.debug(
        "[SurvivorBias] %s @ %s — had_data=%s, has_today=%s, delisted=%s, risk=%s",
        ticker,
        article_date_str,
        had_data_art,
        has_data_today,
        is_potentially_delisted,
        bias_risk,
    )

    return SurvivorBiasReport(
        ticker=ticker,
        article_date=article_date_str,
        is_potentially_delisted=is_potentially_delisted,
        had_data_at_article_date=had_data_art,
        has_data_today=has_data_today,
        price_at_article=round(price_at_art, 2) if price_at_art else None,
        price_today=round(price_today, 2) if price_today else None,
        days_since_article=days_since,
        bias_risk=bias_risk,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Structure du rapport d'intégrité
# ---------------------------------------------------------------------------


@dataclass
class TemporalIntegrityReport:
    """Rapport d'intégrité temporelle pour un article évalué."""

    article_date: str
    context_date_used: str  # Date réelle du contexte marché
    is_temporally_valid: bool  # True si contexte <= article date
    days_contaminated: int  # 0 si valide, N si futur utilisé
    memory_disabled: bool  # True si mémoire AutoDream désactivée
    warnings: list[str] = field(default_factory=list)
    survivor_bias: Optional[SurvivorBiasReport] = None  # Rapport biais de survie


# ---------------------------------------------------------------------------
# Reconstruction du contexte marché historique
# ---------------------------------------------------------------------------


def build_historical_context(ticker: str, article_date_str: str) -> dict:
    """
    Reconstruit le contexte marché tel qu'il était à la date de l'article.
    Utilise yfinance pour récupérer les données historiques.

    Args:
        ticker          : Symbole boursier (ex: "AAPL")
        article_date_str: Date ISO de l'article (ex: "2024-02-01")

    Returns:
        Dict contexte_marche cohérent avec la date de l'article.
        Vide si données non disponibles (article trop ancien / weekend...).
    """
    try:
        from datetime import datetime, timedelta

        import yfinance as yf

        # Parse la date de l'article
        pub_date = datetime.strptime(article_date_str[:10], "%Y-%m-%d")

        # On récupère les 7 jours AVANT la date de l'article
        # pour obtenir le cours au moment de la publication
        start = (pub_date - timedelta(days=7)).strftime("%Y-%m-%d")
        end = (pub_date + timedelta(days=1)).strftime("%Y-%m-%d")

        hist = yf.Ticker(ticker).history(start=start, end=end)
        if hist.empty:
            logger.warning("[TemporalFence] Données historiques non disponibles pour %s à %s", ticker, article_date_str)
            return {}

        # Prix de clôture au jour de la publication (ou dernier jour ouvré avant)
        current_price = float(hist["Close"].iloc[-1])
        volume = int(hist["Volume"].iloc[-1])

        # Variation 5j: uniquement à base des 5j AVANT la publication
        if len(hist) >= 5:
            price_5d_ago = float(hist["Close"].iloc[-5])
            variation_5d = round((current_price - price_5d_ago) / price_5d_ago * 100, 2)
        else:
            variation_5d = 0.0

        logger.debug(
            "[TemporalFence] Contexte historique %s@%s: prix=%.2f, vol=%d, var5d=%+.2f%%",
            ticker,
            article_date_str,
            current_price,
            volume,
            variation_5d,
        )

        return {
            "current_price": round(current_price, 2),
            "volume": volume,
            "variation_5d": variation_5d,
            "_temporal_note": f"Données au {article_date_str} (backtesting)",
        }

    except Exception as e:
        logger.warning("[TemporalFence] Erreur construction contexte historique: %s", e)
        return {}


def validate_context_temporality(
    context: dict,
    article_date_str: str,
    memory_injected: bool = False,
) -> TemporalIntegrityReport:
    """
    Valide qu'un contexte marché n'est pas "du futur" par rapport à l'article.

    Cas problématique : on évalue un article de 2024-02-01 mais le pipeline
    a appelé yfinance aujourd'hui (2026-04-13) → le cours transmis aux agents
    est le cours actuel, pas celui de 2024.

    Args:
        context          : Dict contexte_marche transmis aux agents
        article_date_str : Date ISO de publication de l'article
        memory_injected  : True si la mémoire AutoDream a été injectée

    Returns:
        TemporalIntegrityReport avec un diagnostic clair.
    """
    warnings = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    article_date = datetime.strptime(article_date_str[:10], "%Y-%m-%d")

    # Si le contexte contient la clé _temporal_note → construit via build_historical_context
    # → temporellement valide
    if "_temporal_note" in context:
        context_date_str = article_date_str
        days_contaminated = 0
        is_valid = True
    elif not context:
        # Pas de contexte → valide (conservateur)
        context_date_str = "N/A (aucun contexte)"
        days_contaminated = 0
        is_valid = True
    else:
        # Contexte généré en live (aujourd'hui) → potentiellement du futur
        context_date_str = today_str
        today = datetime.strptime(today_str, "%Y-%m-%d")
        days_contaminated = (today - article_date).days
        is_valid = days_contaminated == 0

        if days_contaminated > 0:
            warnings.append(
                f"CONTAMINATION : contexte marché d'aujourd'hui ({today_str}) "
                f"transmis pour un article du {article_date_str} "
                f"({days_contaminated} jours dans le futur de l'article)"
            )

    if memory_injected:
        warnings.append(
            "MÉMOIRE AutoDream injectée — peut contenir des décisions prises "
            "sur des articles plus récents que celui évalué."
        )

    return TemporalIntegrityReport(
        article_date=article_date_str,
        context_date_used=context_date_str,
        is_temporally_valid=is_valid,
        days_contaminated=days_contaminated,
        memory_disabled=not memory_injected,
        warnings=warnings,
    )


def get_clean_eval_context(
    ticker: str,
    article_date_str: str,
    check_bias: bool = True,
) -> tuple[dict, TemporalIntegrityReport]:
    """
    Point d'entrée principal pour l'évaluation.
    Retourne un contexte marché temporellement propre + son rapport d'intégrité.

    Inclut désormais la détection du biais de survie.

    Args:
        ticker           : Symbole boursier
        article_date_str : Date ISO de l'article
        check_bias       : Si True, lance le check_survivor_bias() (1 appel yfinance suppl.)

    Usage dans evaluate_pipeline.py :
        ctx, integrity = get_clean_eval_context("AAPL", "2024-02-01")
        if integrity.survivor_bias and integrity.survivor_bias.is_potentially_delisted:
            logger.warning("Biais de survie détecté pour %s", ticker)
    """
    hist_context = build_historical_context(ticker, article_date_str)
    report = validate_context_temporality(
        context=hist_context,
        article_date_str=article_date_str,
        memory_injected=False,  # En mode eval, on désactive la mémoire
    )

    # Analyse du biais de survie
    if check_bias:
        bias_report = check_survivor_bias(ticker, article_date_str)
        report.survivor_bias = bias_report

        # Propager les warnings critiques dans le rapport principal
        if bias_report.is_potentially_delisted:
            report.warnings.append(
                f"[BIAIS DE SURVIE] {ticker} potentiellement décoté depuis {article_date_str}. "
                f"Les résultats de ce ticker sont EXCLUS de l'évaluation globale."
            )
        elif bias_report.bias_risk == "MODERE":
            report.warnings.append(
                f"[BIAIS MODERE] {ticker} : {bias_report.notes[-1] if bias_report.notes else 'anomalie détectée'}"
            )

        logger.debug(
            "[TemporalFence] %s bias_risk=%s | delisted=%s",
            ticker,
            bias_report.bias_risk,
            bias_report.is_potentially_delisted,
        )

    return hist_context, report
