"""
agent_pipeline.py

Lit les articles collectés par news_pipeline.py depuis SQLite,
applique le filtre DistilRoBERTa, puis l'ABSA taxonomique,
puis le débat multi-agent AutoGen, et écrit les résultats dans la même base.

Pipeline cible :
  news brute → DistilRoBERTa filtre → ABSA taxonomique → débat → consensus

Usage :
    python agent_pipeline.py
    python agent_pipeline.py --tickers AAPL MSFT
    python agent_pipeline.py --loop 30
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)
import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()  # Charge les clés API depuis .env
import os as _os_for_flags

from src.agents.agent_absa import run_absa
from src.agents.agent_debat import workflow_debat_actualite
from src.agents.agent_filtrage import workflow_filtrer_actualite
from src.agents.agent_memoire import load_context_for_session, run_nightly_consolidation
from src.config import (
    DRY_RUN,
    LLM_DAILY_BUDGET_USD,
    MAX_PAIRWISE_CORRELATION,
    MAX_SECTOR_EXPOSURE_PCT,
    VIX_KILL_SWITCH_THRESHOLD,
)
from src.knowledge import (
    DOC_TYPE_ARTICLE,
    DOC_TYPE_DEBATE,
    RAGDocument,
    format_fundamentals_for_prompt,
    format_macro_for_prompt,
    get_edgar_client,
    get_fundamentals,
    get_macro_context,
    get_rag_store,
)
from src.knowledge.liquidity import get_liquidity_profile
from src.strategy import (
    build_strategy_context,
    calculate_position_size,
    check_pairwise_correlation,
    check_sector_concentration,
    load_investor_profile,
    load_portfolio_state,
    refresh_portfolio_prices,
    save_portfolio_state,
)
from src.utils.anonymizer import anonymize_article, compute_bias_score
from src.utils.bayesian_aggregator import consensus_from_scratchpad
from src.utils.llm_cost_tracker import BudgetExceededError
from src.utils.llm_cost_tracker import flush_snapshot as flush_llm_cost
from src.utils.security_sanitizer import check_prompt_injection
from src.utils.temporal_fence import check_survivor_bias
from src.utils.yolo_classifier import RISK_ELEVE, RISK_FAIBLE, classify_risk

try:
    from src.knowledge.earnings_calls import (
        fetch_and_index_earnings,
        format_earnings_for_prompt,
    )

    EARNINGS_CALLS_AVAILABLE = True
except ImportError:
    EARNINGS_CALLS_AVAILABLE = False
    logger_import_warn = logging.getLogger("AgentPipeline")
    logger_import_warn.info("[EarningsCall] Module non disponible — ignoré.")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("AgentPipeline")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

# Rotation auto : agent_pipeline.log, agent_pipeline.log.1, ..., agent_pipeline.log.5.
# 10 MB par fichier × 5 backups = 60 MB max sur disque. Évite la saturation en prod
# sans perdre d'historique d'audit (on conserve jusqu'à ~5-10 jours de logs selon l'activité).
from logging.handlers import RotatingFileHandler

Path("logs").mkdir(exist_ok=True)
file_handler = RotatingFileHandler(
    "logs/agent_pipeline.log",
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,
    encoding="utf-8",
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(console_handler)
logger.addHandler(file_handler)

DATABASE_PATH = "data/news_database.db"


# ---------------------------------------------------------------------------
# Contexte marché
# ---------------------------------------------------------------------------

_contexte_cache: dict[str, dict] = {}


def _get_contexte_marche(ticker: str) -> dict:
    """
    Récupère cours, volume et variation 5j via yfinance.
    Cache en mémoire pour éviter les appels répétés dans la même session.
    """
    if ticker in _contexte_cache:
        return _contexte_cache[ticker]

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            return {}
        # Enrichissement avec secteur / industrie pour la couche stratégique
        info = {}
        try:
            info = stock.info
        except Exception:
            pass
        contexte = {
            "current_price": round(float(hist["Close"].iloc[-1]), 2),
            "volume": int(hist["Volume"].iloc[-1]),
            "variation_5d": round((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100, 2),
            "sector": info.get("sector", "Inconnu"),
            "industry": info.get("industry", "Inconnu"),
        }
        logger.debug(
            "Contexte marché %s — Prix: %s | Secteur: %s | Variation 5j: %s%%",
            ticker,
            contexte["current_price"],
            contexte["sector"],
            contexte["variation_5d"],
        )
        _contexte_cache[ticker] = contexte
        return contexte
    except Exception as e:
        logger.warning("Impossible de récupérer le contexte marché pour %s : %s", ticker, e)
        return {}


# ---------------------------------------------------------------------------
# Calcul de l'Impact Strength
# ---------------------------------------------------------------------------


def _calculer_metrics_objectives(signal_final: str, score_finbert: float, absa_result: dict) -> tuple[float, float]:
    """
    Calcule mathématiquement la force de la décision sans demander à un LLM.

    1. Score ABSA (Factuel) : basé sur le comptage des aspects extraits par le LLM.
    2. Score FinBERT (ML) : basé sur la probabilité pure du modèle de Deep Learning.

    Retourne :
      - consensus_rate : L'alignement entre l'analyse FinBERT et l'analyse ABSA (0.0 à 1.0)
      - impact_strength : La force absolue du signal final (Achat/Vente) (0.0 à 1.0)
    """
    # 1. Calcul de la polarité ABSA
    aspects = absa_result.get("aspects", [])
    nb_pos = len([a for a in aspects if a.get("sentiment") == "positive"])
    nb_neg = len([a for a in aspects if a.get("sentiment") == "negative"])
    total = nb_pos + nb_neg

    absa_ratio = (nb_pos / total) if total > 0 else 0.5

    # 2. Polarisation de FinBERT (score directionnel)
    # Si le signal est Achat, on veut que finbert soit proche de 1.0.
    # S'il était négatif (score < 0.5), c'est une divergence.
    finbert_pos_ratio = score_finbert if score_finbert > 0.5 else (1.0 - score_finbert)

    # 3. Calcul de la force selon le signal choisi par l'agent
    if signal_final == "Achat":
        absa_force = absa_ratio
        finbert_force = finbert_pos_ratio
    elif signal_final == "Vente":
        absa_force = 1.0 - absa_ratio
        # Inverse la force FinBERT si c'était classé "positif"
        finbert_force = score_finbert if score_finbert < 0.5 else (1.0 - score_finbert)
    else:
        absa_force = 0.5
        finbert_force = 0.5

    # Impact = Moyenne pondérée entre la NLP classique (FinBERT) et l'extraction de faits (ABSA)
    impact_strength = round((finbert_force * 0.4) + (absa_force * 0.6), 4)

    # Consensus = À quel point FinBERT et l'agent ABSA étaient d'accord sur la direction
    # 1.0 = Accord parfait. 0.0 = Désaccord total.
    consensus_rate = round(1.0 - abs(finbert_force - absa_force), 4)

    return consensus_rate, impact_strength


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------


def run_agent_pipeline(
    tickers: list[str] | None = None, limit: int | None = None, consolidate_memory: bool = True
) -> None:
    """
    Lit les articles non traités depuis la table `articles` (produite par news_pipeline.py),
    applique FinBERT puis le débat multi-agent, et écrit les résultats dans la même table.

    Si tickers est fourni, filtre uniquement ces tickers.

    En fin de run, si consolidate_memory=True, déclenche l'Agent de Nuit (AutoDream)
    pour consolider les décisions du jour dans data/memory/.

    Intègre la couche stratégique (profil investisseur + portefeuille + sizing Kelly)
    pour ancrer chaque décision dans la réalité financière de l'investisseur.
    """
    # -----------------------------------------------------------------------
    # Couche Stratégique — Chargement du profil investisseur et du portefeuille
    # Ref: MiFID II (ESMA 2018), Markowitz (1952), Kelly (1956)
    # -----------------------------------------------------------------------
    investor_profile = load_investor_profile()
    portfolio_state = load_portfolio_state()
    active_tickers = tickers or []
    if active_tickers:
        refresh_portfolio_prices(portfolio_state)
        portfolio_state.log_summary()

    # -----------------------------------------------------------------------
    # Couche Connaissance — Init RAG + Macro au démarrage de la session
    # -----------------------------------------------------------------------
    rag_store = get_rag_store()
    edgar_client = get_edgar_client()

    # Cache Earnings Calls par ticker (1 fetch par session pour éviter spam EDGAR)
    # Ref: Loughran & McDonald (2011) — la tonalité du management est stable
    # à l'échelle d'une session de trading (le 8-K ne change pas en cours de journée)
    _earnings_cache: dict = {}  # ticker → EarningsCallResult

    # Macro context : 1 seul appel par session (valeurs stables à l'échelle de la journée)
    import os

    fred_key = os.getenv("FRED_API_KEY")  # Optionnel — fallback yfinance si absent
    macro_snap = get_macro_context(fred_api_key=fred_key)
    macro_ctx_str = format_macro_for_prompt(macro_snap)

    # -----------------------------------------------------------------------
    # KILL-SWITCH systémique VIX (Whaley 2009)
    # -----------------------------------------------------------------------
    # Si VIX > VIX_KILL_SWITCH_THRESHOLD (défaut 45 — zone krach COVID/2008),
    # on refuse TOUTE nouvelle prise de position. On ne loggue qu'un WARN et
    # on skip le reste du run. Les positions ouvertes continuent d'être
    # gérées en dehors du pipeline (stop-loss broker-side).
    _kill_switch_active = False
    if macro_snap.vix is not None and macro_snap.vix >= VIX_KILL_SWITCH_THRESHOLD:
        _kill_switch_active = True
        logger.error("=" * 70)
        logger.error(
            "[!!! KILL-SWITCH !!!] VIX=%.1f >= seuil systémique %.1f",
            macro_snap.vix,
            VIX_KILL_SWITCH_THRESHOLD,
        )
        logger.error(
            "Mode de krach détecté — toute nouvelle position est BLOQUÉE. "
            "Signal forcé à HOLD_SYSTEMIC pour tous les articles traités."
        )
        logger.error("=" * 70)

    # -----------------------------------------------------------------------
    # DRY_RUN / Paper trading banner
    # -----------------------------------------------------------------------
    if DRY_RUN:
        logger.warning("=" * 70)
        logger.warning("[DRY_RUN ACTIVE] Aucun ordre ne sera persisté sur le portefeuille réel.")
        logger.warning("Les ordres hypothétiques sont loggés dans logs/dry_run_trades.jsonl")
        logger.warning("=" * 70)

    if LLM_DAILY_BUDGET_USD > 0:
        logger.info("[LLMCost] Budget quotidien = $%.2f (hard-stop si dépassé)", LLM_DAILY_BUDGET_USD)

    # -----------------------------------------------------------------------
    # Phase 1 AutoDream — Orientation : chargement de la mémoire historique
    # Le contexte mémoire est injecté dans chaque débat pour que les agents
    # bénéficient des décisions passées sans surcharger la fenêtre de contexte.
    # -----------------------------------------------------------------------
    memory_context = load_context_for_session(active_tickers)
    if memory_context:
        logger.info("[Mémoire] Contexte historique chargé pour %s (AutoDream Phase 1).", active_tickers)
    else:
        logger.info("[Mémoire] Aucun historique disponible — premier run ou mémoire vide.")
    with sqlite3.connect(DATABASE_PATH, timeout=15) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")

        # ---------------------------------------------------------------------------
        # Migrations rétrocompatibles — ajout des nouvelles colonnes si absentes
        # ALTER TABLE ne fait rien si la colonne existe déjà (OperationalError ignorée)
        # ---------------------------------------------------------------------------
        _new_columns = [
            "ALTER TABLE articles ADD COLUMN consensus_model    TEXT;",
            "ALTER TABLE articles ADD COLUMN cdps_resolved      TEXT;",  # JSON — AgentAuditor
            "ALTER TABLE articles ADD COLUMN cdps_unresolved    TEXT;",  # JSON — CDPs non résolus
            "ALTER TABLE articles ADD COLUMN council_matrix     TEXT;",  # JSON — Council Mode
            "ALTER TABLE articles ADD COLUMN ese_score          REAL;",  # Entropie Sémantique ESE
            "ALTER TABLE articles ADD COLUMN htc_score          REAL;",  # Holistic Trajectory Calibration
            "ALTER TABLE articles ADD COLUMN trust_trade_score  REAL;",  # TrustTrade
            # --- Couche Stratégique (Locus v2) ---
            "ALTER TABLE articles ADD COLUMN position_size_euros           REAL;",
            "ALTER TABLE articles ADD COLUMN position_nb_actions            REAL;",
            "ALTER TABLE articles ADD COLUMN sizing_method                  TEXT;",
            "ALTER TABLE articles ADD COLUMN sizing_kelly_f_star            REAL;",
            "ALTER TABLE articles ADD COLUMN sizing_kelly_fraction          REAL;",
            "ALTER TABLE articles ADD COLUMN sizing_win_prob                REAL;",
            "ALTER TABLE articles ADD COLUMN action_type                    TEXT;",
            "ALTER TABLE articles ADD COLUMN portfolio_drawdown_at_decision REAL;",
            "ALTER TABLE articles ADD COLUMN secteur_exposition_at_decision REAL;",
            # --- Couche Connaissance (Locus v3) ---
            "ALTER TABLE articles ADD COLUMN rag_context_used     TEXT;",  # passages RAG injectés
            "ALTER TABLE articles ADD COLUMN sec_8k_found         INTEGER;",  # 0/1
            "ALTER TABLE articles ADD COLUMN sec_insider_signal   TEXT;",  # BULLISH/BEARISH/NEUTRE
            "ALTER TABLE articles ADD COLUMN analyst_consensus    TEXT;",  # consensus analystes
            "ALTER TABLE articles ADD COLUMN forward_pe           REAL;",  # Forward P/E
            "ALTER TABLE articles ADD COLUMN days_to_earnings     INTEGER;",  # Earnings dans N jours
            "ALTER TABLE articles ADD COLUMN vix_at_decision      REAL;",  # VIX
            "ALTER TABLE articles ADD COLUMN yield_curve_at_decision REAL;",  # Yield curve spread
            # --- Filtre de regime de marche (audit L12 — 2026-04-21) ---
            "ALTER TABLE articles ADD COLUMN market_regime          TEXT;",  # BULL|BEAR|SIDEWAYS|HIGH_VOL|UNKNOWN
            "ALTER TABLE articles ADD COLUMN spy_20d_return         REAL;",  # SPY 20j glissants (%) au moment de la decision
            "ALTER TABLE articles ADD COLUMN regime_veto            INTEGER;",  # 1 si signal override par filtre regime
            # --- Production hygiene (Phase 3 — 2026-04-22) ---
            "ALTER TABLE articles ADD COLUMN dry_run                INTEGER;",  # 1 si trade non persisté sur le portefeuille réel
            "ALTER TABLE articles ADD COLUMN kill_switch_active     INTEGER;",  # 1 si VIX > seuil systémique
        ]
        for _migration_sql in _new_columns:
            try:
                cursor.execute(_migration_sql)
                conn.commit()
            except sqlite3.OperationalError:
                pass  # Colonne déjà présente — migration ignorée

        if tickers:
            placeholders = ",".join("?" * len(tickers))
            cursor.execute(
                f"""
                SELECT url, ticker, title, content
                FROM articles
                WHERE signal_final IS NULL
                  AND ticker IN ({placeholders})
                  AND content IS NOT NULL
                  AND length(content) > 100
                ORDER BY date_utc DESC
                """,
                tickers,
            )
        else:
            cursor.execute(
                """
                SELECT url, ticker, title, content
                FROM articles
                WHERE signal_final IS NULL
                  AND content IS NOT NULL
                  AND length(content) > 100
                ORDER BY date_utc DESC
                """
            )

        articles_raw = cursor.fetchall()

        # Application de la limite par ticker en Python
        if limit is not None:
            ticker_counts = {}
            articles = []
            for row in articles_raw:
                t = row["ticker"]
                if ticker_counts.get(t, 0) < limit:
                    articles.append(row)
                    ticker_counts[t] = ticker_counts.get(t, 0) + 1
        else:
            articles = articles_raw
        logger.info("%d article(s) à traiter.", len(articles))

        if not articles:
            logger.info("Aucun article en attente de traitement.")
            return

        for row in articles:
            url = row["url"]
            ticker = row["ticker"]
            title = row["title"]
            content = row["content"]

            logger.info("=" * 60)
            logger.info("Traitement : %s [%s]", title, ticker)

            # ----------------------------------------------------------------
            # Étape 0 : Filtrage de Sécurité (Zero-Trust Prompt Injection)
            # ----------------------------------------------------------------
            if check_prompt_injection(content):
                logger.warning("  [!] ALERTE SÉCURITÉ : Tentative d'injection de prompt détectée. Article rejeté.")
                cursor.execute(
                    "UPDATE articles SET signal_final = ?, argument_dominant = ? WHERE url = ?",
                    ("Rejeté (Sécurité)", "Tentative d'injection de prompt", url),
                )
                conn.commit()
                continue

            # ----------------------------------------------------------------
            # Étape 0.5 : Temporal Fence — anti survivorship-bias
            # Vérifie qu'on ne traite pas un article d'un ticker delisted après
            # sa faillite (Brown, Goetzmann, Ibbotson 1992 ; Carpenter & Lynch 1999).
            # ----------------------------------------------------------------
            article_date_str = str(row["date"]) if "date" in row.keys() else ""
            sb_report = check_survivor_bias(ticker, article_date_str)
            if sb_report.is_failed_ticker and sb_report.needs_warning:
                logger.warning(
                    "[TemporalFence] Ticker %s marque comme 'failed' : %s", ticker, sb_report.warning_message
                )
                # On continue mais on marque l'article pour audit
            # ----------------------------------------------------------------
            # Étape 0.7 : Anonymisation (optionnelle, mode eval)
            # Si EVAL_ANONYMIZE=1, on remplace les entités connues pour mesurer
            # l'effet du prior implicite du LLM (Veitch 2021, module Counterfactual).
            # ----------------------------------------------------------------
            anonymization_used = False
            if _os_for_flags.environ.get("EVAL_ANONYMIZE", "0") == "1":
                anon_result = anonymize_article(content, ticker=ticker)
                if anon_result.entities_replaced > 0:
                    logger.info(
                        "[Anonymizer] %d entites remplacees (%s -> %s)",
                        anon_result.entities_replaced,
                        ticker,
                        anon_result.ticker_alias,
                    )
                    content = anon_result.text
                    anonymization_used = True

            # ----------------------------------------------------------------
            # Étape 1 : Filtre DistilRoBERTa
            # ----------------------------------------------------------------
            signal_filtre = workflow_filtrer_actualite.invoke(content)

            if signal_filtre is None:
                logger.info("  -> Rejeté par DistilRoBERTa (neutre).")
                cursor.execute(
                    "UPDATE articles SET signal_filtrage = ?, score_filtrage = ?, signal_final = ? WHERE url = ?",
                    ("neutral", 0.0, "Rejeté (Filtre)", url),
                )
                conn.commit()
                continue

            label_finbert, score_finbert = signal_filtre
            logger.info("  -> DistilRoBERTa : %s (%.2f%%). Lancement ABSA...", label_finbert, score_finbert * 100)

            cursor.execute(
                "UPDATE articles SET signal_filtrage = ?, score_filtrage = ? WHERE url = ?",
                (label_finbert, round(score_finbert, 4), url),
            )
            conn.commit()

            # ----------------------------------------------------------------
            # Étape 2 : ABSA taxonomique
            # ----------------------------------------------------------------
            absa_result = run_absa(content)
            absa_json_str = json.dumps(absa_result, ensure_ascii=False)

            logger.info(
                "  -> ABSA : %d aspect(s) détecté(s) — %s",
                len(absa_result.get("aspects", [])),
                [a["aspect"] for a in absa_result.get("aspects", [])],
            )

            cursor.execute("UPDATE articles SET absa_json = ? WHERE url = ?", (absa_json_str, url))
            conn.commit()

            # ----------------------------------------------------------------
            # Étape 3 : Débat multi-agent AutoGen
            # ----------------------------------------------------------------
            contexte_marche = _get_contexte_marche(ticker)

            # ----------------------------------------------------------------
            # Contexte Stratégique — Ancrage dans la réalité investisseur
            # ----------------------------------------------------------------
            secteur_ticker = contexte_marche.get("sector", "Inconnu")
            industrie_ticker = contexte_marche.get("industry", "Inconnu")
            strategy_ctx = build_strategy_context(
                ticker=ticker,
                signal_candidat=label_finbert,
                profile=investor_profile,
                portfolio=portfolio_state,
                prix_actuel=contexte_marche.get("current_price", 0.0),
                secteur=secteur_ticker,
            )
            logger.debug("[Strategy] Contexte injecté pour %s :\n%s", ticker, strategy_ctx)

            # ----------------------------------------------------------------
            # Couche Connaissance — Fundamentals, Macro, RAG
            # Ref: Fama (1970), Lewis et al. (2020), Kanhabua & Nørvåg (2010)
            # ----------------------------------------------------------------
            # 1. Données fondamentales (yfinance, gratuit)
            fund_data = get_fundamentals(ticker, current_price=contexte_marche.get("current_price", 0.0))
            fund_str = format_fundamentals_for_prompt(fund_data)

            # 2. Contexte macro (déjà calculé au démarrage de session)
            macro_str = format_macro_for_prompt(macro_snap, ticker_sector=secteur_ticker)

            # 3. RAG — passages les plus pertinents temporellement pondérés
            rag_results = rag_store.query(ticker, query_text=f"{title} {content[:500]}", k=4)
            rag_str = rag_store.format_for_prompt(rag_results, max_chars=2500)
            rag_ids_str = json.dumps([r.doc.doc_id[:80] for r in rag_results])

            # 4. Earnings Call — 1 fetch par ticker par session (cache)
            # Ref: Loughran & McDonald (2011) — tone score prédit +1.8% / -2.3% à J+5
            earnings_str = ""
            if EARNINGS_CALLS_AVAILABLE:
                if ticker not in _earnings_cache:
                    logger.info("  [EarningsCall] Fetch SEC EDGAR pour %s...", ticker)
                    ec_result = fetch_and_index_earnings(ticker, rag_store)
                    _earnings_cache[ticker] = ec_result
                else:
                    ec_result = _earnings_cache[ticker]
                earnings_str = format_earnings_for_prompt(ec_result)
                if ec_result.found:
                    logger.info(
                        "  [EarningsCall] %s — %s | LM=%s (%.2f)",
                        ticker,
                        ec_result.quarter,
                        ec_result.lm_label,
                        ec_result.lm_score,
                    )

            logger.info(
                "  [Knowledge] Fundamentals: %s | RAG: %d passages | Macro: %s | VIX: %s | Earnings: %s",
                fund_data.data_quality,
                len(rag_results),
                macro_snap.data_quality,
                f"{macro_snap.vix:.1f} ({macro_snap.vix_regime})" if macro_snap.vix else "N/A",
                _earnings_cache.get(ticker, type("", (), {"found": False})).found
                if EARNINGS_CALLS_AVAILABLE
                else "N/A",
            )

            try:
                # La mémoire historique est injectée dans le débat.
                # Les agents découvrent les précédents de façon structurée
                # sans que ça coûte la pleine transcription des débats passés.
                # ----------------------------------------------------------------
                # Fact-Checking (Validation SEC / Rumeurs)
                # ----------------------------------------------------------------
                is_verified, fc_reason, confidence_penalty = edgar_client.fact_check_article(
                    ticker=ticker,
                    article_text=content,
                    article_url=url,
                    days_back=10,
                )
                fc_context = (
                    f"\n\n[FACT-CHECKING SEC EDGAR]\n"
                    f"Statut: {'Validé (Filing officiel EDGAR)' if is_verified else 'Non officiel / Rumeur potentielle'}\n"
                    f"Raison: {fc_reason}\n"
                    f"Impact Confiance: {confidence_penalty}x"
                )

                # Assemblage complet : mémoire + article + fact-check + RAG + fundamentals + macro + earnings + stratégie
                base_content = (
                    f"{memory_context}\n{content}{fc_context}" if memory_context else f"{content}{fc_context}"
                )
                knowledge_block = "\n\n".join(filter(None, [rag_str, fund_str, macro_str, earnings_str]))
                enriched_content = f"{base_content}\n\n{knowledge_block}\n\n{strategy_ctx}"

                import time

                t0 = time.time()
                decision = workflow_debat_actualite.invoke(
                    {
                        "texte_article": enriched_content,
                        "ticker_symbol": ticker,
                        "contexte_marche": contexte_marche,
                        "absa_result": absa_result,
                    }
                )
                processing_time_ms = (time.time() - t0) * 1000.0

                signal_final = decision.get("signal", "Neutre")
                argument_dominant = decision.get("argument_dominant", "")
                transcription = decision.get("transcription", "")
                consensus_model = decision.get("consensus_model", "Inconnu")

                # Le calcul devient strictement mathématique (pas d'hallucinations LLM)
                consensus_rate, impact_strength = _calculer_metrics_objectives(signal_final, score_finbert, absa_result)

                # Application de la pénalité de Fact-Checking (Rumeurs)
                impact_strength = round(impact_strength * confidence_penalty, 4)

                logger.info(
                    "  -> Signal : %s | Consensus (Data) : %.0f%% | Impact Strength : %.2f | Modèle : %s",
                    signal_final,
                    consensus_rate * 100,
                    impact_strength,
                    consensus_model,
                )

            except BudgetExceededError as bexc:
                # Budget LLM quotidien atteint : on arrête PROPREMENT la session
                # sans marquer l'article en erreur (il sera repris demain).
                logger.error("=" * 70)
                logger.error("[BUDGET GUARD] %s", bexc)
                logger.error(
                    "Interruption de la session. Le pipeline reprendra "
                    "automatiquement après minuit UTC (rotation quotidienne)."
                )
                logger.error("=" * 70)
                break
            except Exception as e:
                logger.error("  -> Erreur débat (%s) : %s", type(e).__name__, str(e))
                # On marque l'article avec un signal d'erreur pour ne pas le retraiter indéfiniment
                cursor.execute(
                    """UPDATE articles
                       SET signal_final = ?, argument_dominant = ?
                       WHERE url = ?""",
                    ("Erreur", str(e)[:200], url),
                )
                conn.commit()
                continue

            # ----------------------------------------------------------------
            # Garde de securite YOLO (Risk Classifier + Filtre Regime)
            # Intercale entre le Consensus (LLM) et l'execution d'ordre.
            # Aucun appel LLM supplementaire — scoring deterministe.
            # Audit L12 : SIDEWAYS accuracy = 0% -> veto sur signaux directionnels
            # ----------------------------------------------------------------
            scratchpad_xml = decision.get("scratchpad_xml", "")

            # ----------------------------------------------------------------
            # Agrégation Bayésienne (Module ②) — Beta-Binomial hierarchical
            # Source de vérité pour p_mean + variance épistémique + kelly_scale
            # Ref: Raftery et al. 2005 BMA, Lakshminarayanan et al. 2017 Deep Ensembles
            # ----------------------------------------------------------------
            try:
                bayes_consensus = consensus_from_scratchpad(scratchpad_xml)
                logger.info(
                    "  [Bayes] E[p]=%.3f | Var_total=%.4f (epi=%.4f, ale=%.4f) | "
                    "CI95=[%.2f, %.2f] | Kelly_scale=%.3f | signal=%s",
                    bayes_consensus.p_mean,
                    bayes_consensus.p_var_total,
                    bayes_consensus.p_var_epistemic,
                    bayes_consensus.p_var_aleatoric,
                    bayes_consensus.ci95_lower,
                    bayes_consensus.ci95_upper,
                    bayes_consensus.kelly_scale,
                    bayes_consensus.signal,
                )
                for r in bayes_consensus.reasons[:2]:
                    logger.info("       %s", r)
            except Exception as exc:
                logger.debug("[Bayes] consensus aggregator disabled: %s", exc)
                bayes_consensus = None

            # Extraction des donnees de regime depuis macro_snap (deja calcule au debut de session)
            spy_20d_return = getattr(macro_snap, "spy_20d_return", None)
            spy_20d_vol = getattr(macro_snap, "spy_20d_vol", None)
            vix_current = macro_snap.vix if macro_snap.vix else None

            # Fallback : calcul yfinance si macro_snap n'a pas spy_20d_return
            if spy_20d_return is None or spy_20d_vol is None:
                try:
                    import numpy as _np
                    import yfinance as _yf

                    spy_hist = _yf.Ticker("SPY").history(period="45d")["Close"]
                    if len(spy_hist) >= 20:
                        if spy_20d_return is None:
                            spy_20d_return = round((spy_hist.iloc[-1] / spy_hist.iloc[-20] - 1) * 100, 2)
                        if spy_20d_vol is None:
                            daily_ret = spy_hist.pct_change().dropna().tail(20)
                            spy_20d_vol = round(float(daily_ret.std() * _np.sqrt(252) * 100), 2)
                except Exception:
                    pass

            # Seuils YOLO dynamiques selon le profil Arrow-Pratt (ESMA 2018)
            yolo_thresholds = investor_profile.get_yolo_thresholds()
            yolo = classify_risk(
                signal_final=signal_final,
                consensus_rate=consensus_rate,
                impact_strength=impact_strength,
                scratchpad_xml=scratchpad_xml,
                absa_result=absa_result,
                score_finbert=score_finbert,
                contexte_marche=contexte_marche,
                argument_dominant=argument_dominant,
                seuil_faible=yolo_thresholds["faible"],
                seuil_eleve=yolo_thresholds["eleve"],
                spy_20d_return=spy_20d_return,
                spy_20d_vol=spy_20d_vol,
                vix=vix_current,
                processing_time_ms=processing_time_ms,
            )
            yolo.log_summary(ticker)

            # Log du regime detecte
            if yolo.regime_veto:
                logger.warning(
                    "  [Regime] %s SIDEWAYS veto: signal original '%s' -> 'Neutre'", ticker, decision.get("signal", "?")
                )
            else:
                logger.info(
                    "  [Regime] %s | Regime=%s | SPY_20d=%s | VIX=%s",
                    ticker,
                    yolo.market_regime,
                    f"{spy_20d_return:.1f}%" if spy_20d_return is not None else "N/A",
                    f"{vix_current:.1f}" if vix_current is not None else "N/A",
                )

            # Mise a jour du signal_final si veto regime a eu lieu
            signal_final = yolo.features.get("signal_after_regime", signal_final)
            # Note : le signal peut avoir ete modifie en interne par classify_risk()
            # On utilise directement le regime_veto flag pour la DB

            # ----------------------------------------------------------------
            # Kill-switch VIX — force HOLD_SYSTEMIC avant tout sizing
            # ----------------------------------------------------------------
            if _kill_switch_active:
                signal_final = "HOLD_SYSTEMIC"
                logger.warning(
                    "  [KILL-SWITCH] VIX=%.1f — signal forcé à HOLD_SYSTEMIC pour %s",
                    macro_snap.vix or 0.0,
                    ticker,
                )

            # ----------------------------------------------------------------
            # ADV dynamique (yfinance) — remplace le default 5 M actions
            # Biais majeur pour small-caps si laissé au défaut (cf audit Phase 3)
            # ----------------------------------------------------------------
            liq_profile = get_liquidity_profile(ticker)
            adv_volume_dyn = liq_profile.adv_volume
            sigma_daily_dyn = liq_profile.sigma_daily

            # ----------------------------------------------------------------
            # Position Sizing — Kelly / Half-Kelly (Thorp 1956, 2008)
            # Calcul du montant optimal à investir selon profil + état portif
            # ----------------------------------------------------------------
            sizing = calculate_position_size(
                ticker=ticker,
                prix_actuel=contexte_marche.get("current_price", 0.0),
                signal_final=signal_final,
                impact_strength=impact_strength,
                trust_trade_score=yolo.trust_trade_score,
                profile=investor_profile,
                portfolio=portfolio_state,
                secteur=secteur_ticker,
                bayesian_consensus=bayes_consensus,
                adv_volume=adv_volume_dyn,
                sigma_daily=sigma_daily_dyn,
                delay_ms=float(processing_time_ms or 0.0),
            )

            # ----------------------------------------------------------------
            # Portfolio constraints — refuse si dépassement sectoriel > 30%
            # ----------------------------------------------------------------
            if signal_final in ("Achat", "Vente") and sizing.montant_euros > 0:
                constraint = check_sector_concentration(
                    portfolio_state=portfolio_state,
                    secteur=secteur_ticker,
                    montant_propose_euros=sizing.montant_euros,
                )
                if not constraint.allowed:
                    logger.warning("  [SECTOR CAP] %s", constraint.summary())
                    signal_final = "HOLD_SECTOR_CAP"
                    # Neutralise la position proposée (sizing méta préservée pour audit)
                    sizing.montant_euros = 0.0
                    sizing.nb_actions = 0.0
                    sizing.action_type = "REJECT_SECTOR"
                else:
                    logger.debug("  [SECTOR CAP] %s", constraint.summary())

            # ----------------------------------------------------------------
            # Portfolio constraints — cap de corrélation cross-sectionnelle
            #   Rankin-Jegadeesh 1993, Ang-Chen 2002 : ρ>0.8 = risque dupliqué
            # ----------------------------------------------------------------
            if signal_final == "Achat" and sizing.montant_euros > 0:
                try:
                    corr_constraint = check_pairwise_correlation(
                        portfolio_state=portfolio_state,
                        ticker_propose=ticker,
                        cap_rho=MAX_PAIRWISE_CORRELATION,
                    )
                    if not corr_constraint.allowed:
                        logger.warning("  [CORR CAP] %s", corr_constraint.summary())
                        signal_final = "HOLD_CORR_CAP"
                        sizing.montant_euros = 0.0
                        sizing.nb_actions = 0.0
                        sizing.action_type = "REJECT_CORR"
                    else:
                        logger.debug("  [CORR CAP] %s", corr_constraint.summary())
                except Exception as e:
                    # Ne pas bloquer la décision si yfinance/cache KO : on logge et on laisse passer
                    logger.debug("  [CORR CAP] check skipped (%s)", e)
            logger.info(
                "  [Sizing] %s | %.2f€ (%s actions) | Méthode: %s | Kelly p=%.0f%%",
                sizing.action_type,
                sizing.montant_euros,
                sizing.nb_actions,
                sizing.methode_finale,
                sizing.kelly_win_prob * 100,
            )

            from eval.evaluate_debate_dynamics import analyse_scratchpad_live

            debate_analysis = analyse_scratchpad_live(scratchpad_xml, str(url), ticker)
            if debate_analysis:
                if debate_analysis.debate_quality_score < 0.40:
                    logger.warning(
                        "  [!] Echo de chambre detecte (score: %.2f) pour %s",
                        debate_analysis.debate_quality_score,
                        ticker,
                    )
                else:
                    logger.info(
                        "  -> Qualite debat : %.2f/1.00 (%s)",
                        debate_analysis.debate_quality_score,
                        debate_analysis.verdict,
                    )

            if yolo.requires_human:
                logger.warning(
                    "  [YOLO xx] RISQUE ELEVE — Approbation humaine requise avant exécution de l'ordre %s sur %s",
                    signal_final,
                    ticker,
                )
            elif yolo.auto_execute:
                logger.info("  [YOLO ok] RISQUE FAIBLE — Exécution autonome autorisée (%s %s)", signal_final, ticker)
            else:
                logger.info("  [YOLO !!] RISQUE MOYEN — Exécuter avec précaution (%s %s)", signal_final, ticker)

            # ----------------------------------------------------------------
            # DRY_RUN : loggue l'ordre hypothétique ET met à jour le portefeuille virtuel
            # ----------------------------------------------------------------
            if DRY_RUN and signal_final in ("Achat", "Vente") and sizing.montant_euros > 0:
                from src.utils.dry_run_logger import log_dry_run_order

                log_dry_run_order(
                    ticker=ticker,
                    signal=signal_final,
                    prix=contexte_marche.get("current_price", 0.0),
                    quantite=sizing.nb_actions,
                    montant_eur=sizing.montant_euros,
                    sizing_method=sizing.methode_finale,
                    win_prob=sizing.kelly_win_prob,
                    risk_level=yolo.risk_level,
                    market_regime=yolo.market_regime,
                    vix=macro_snap.vix,
                    yield_curve_spread=macro_snap.yield_curve_spread,
                    notes=f"dry_run p_var={getattr(bayes_consensus, 'p_var_total', 0.0):.4f}",
                    extras={
                        "adv_volume": adv_volume_dyn,
                        "sigma_daily": sigma_daily_dyn,
                        "liq_source": liq_profile.source,
                    },
                )

                # Mise à jour du portefeuille virtuel pour simuler le budget !
                prix_actuel = contexte_marche.get("current_price", 0.0)
                if prix_actuel > 0:
                    if signal_final == "Achat":
                        portfolio_state.enregistrer_achat(
                            ticker=ticker,
                            nb_actions=sizing.nb_actions,
                            prix=prix_actuel,
                            secteur=secteur_ticker,
                            industrie=industrie_ticker,
                        )
                    elif signal_final == "Vente":
                        portfolio_state.enregistrer_vente(ticker=ticker, prix=prix_actuel)

                    save_portfolio_state(portfolio_state)

            # ----------------------------------------------------------------
            # Persistance des résultats — champs étendus (CDPs, Council, ESE, HTC)
            # ----------------------------------------------------------------
            try:
                cdps_resolved_json = json.dumps(decision.get("cdps_resolved", []), ensure_ascii=False)
                cdps_unresolved_json = json.dumps(decision.get("cdps_unresolved", []), ensure_ascii=False)
                council_matrix_json = json.dumps(decision.get("council_matrix", None), ensure_ascii=False)

                # Métriques portefeuille au moment de la décision
                drawdown_at_decision = portfolio_state.drawdown_actuel()
                expo_sect_at_decision = portfolio_state.exposition_sectorielle().get(secteur_ticker, 0.0)

                # Métriques Knowledge Layer (Locus v3)
                insider_act = edgar_client.get_insider_activity(ticker, days_back=30)
                sec_8k_found = (
                    1
                    if (edgar_client.find_recent_8k(ticker, days_back=7) or type("", (), {"found": False})).found
                    else 0
                )
                sec_8k_obj = edgar_client.find_recent_8k(ticker, days_back=7)
                sec_8k_found = 1 if (sec_8k_obj and sec_8k_obj.found) else 0

                cursor.execute(
                    """UPDATE articles
                       SET signal_final        = ?,
                           consensus_rate      = ?,
                           impact_strength     = ?,
                           argument_dominant   = ?,
                           consensus_model     = ?,
                           transcription_debat = ?,
                           risk_level          = ?,
                           cdps_resolved       = ?,
                           cdps_unresolved     = ?,
                           council_matrix      = ?,
                           ese_score           = ?,
                           htc_score           = ?,
                           trust_trade_score   = ?,
                           position_size_euros           = ?,
                           position_nb_actions           = ?,
                           sizing_method                 = ?,
                           sizing_kelly_f_star           = ?,
                           sizing_kelly_fraction         = ?,
                           sizing_win_prob               = ?,
                           action_type                   = ?,
                           portfolio_drawdown_at_decision = ?,
                           secteur_exposition_at_decision = ?,
                           rag_context_used              = ?,
                           sec_8k_found                  = ?,
                           sec_insider_signal            = ?,
                           analyst_consensus             = ?,
                           forward_pe                    = ?,
                           days_to_earnings              = ?,
                           vix_at_decision               = ?,
                           yield_curve_at_decision       = ?,
                           market_regime                 = ?,
                           spy_20d_return                = ?,
                           regime_veto                   = ?,
                           dry_run                       = ?,
                           kill_switch_active            = ?
                       WHERE url = ?""",
                    (
                        signal_final,
                        round(consensus_rate, 4),
                        impact_strength,
                        argument_dominant,
                        consensus_model,
                        transcription,
                        yolo.risk_level,
                        cdps_resolved_json,
                        cdps_unresolved_json,
                        council_matrix_json,
                        decision.get("ese_score", None),
                        yolo.features.get("htc_score", None),
                        yolo.trust_trade_score,
                        sizing.montant_euros,
                        sizing.nb_actions,
                        sizing.methode_finale,
                        sizing.kelly_f_star,
                        sizing.kelly_fraction_applied,
                        sizing.kelly_win_prob,
                        sizing.action_type,
                        drawdown_at_decision,
                        expo_sect_at_decision,
                        rag_ids_str,
                        sec_8k_found,
                        insider_act.signal,
                        fund_data.analyst_consensus,
                        fund_data.pe_forward,
                        fund_data.days_to_earnings,
                        macro_snap.vix,
                        macro_snap.yield_curve_spread,
                        yolo.market_regime,
                        spy_20d_return,
                        1 if yolo.regime_veto else 0,
                        1 if DRY_RUN else 0,
                        1 if _kill_switch_active else 0,
                        url,
                    ),
                )
                conn.commit()
                logger.info(
                    "  -> Résultats sauvegardés (risque=%s | regime=%s | veto=%s | HTC=%.3f | TrustTrade=%.3f | ESE=%.3f | SEC_8K=%s).",
                    yolo.risk_level,
                    yolo.market_regime,
                    yolo.regime_veto,
                    yolo.features.get("htc_score", 0.0),
                    yolo.trust_trade_score,
                    decision.get("ese_score", 0.0),
                    sec_8k_found,
                )

                # ----------------------------------------------------------------
                # Indexation RAG post-débat
                # Ref: Lewis et al. (2020) — l'article ET le résumé du débat
                # sont indexés pour enrichir les futurs débats sur le même ticker
                # ----------------------------------------------------------------
                from datetime import datetime, timezone

                date_now = datetime.now(timezone.utc).isoformat()

                # Document article
                rag_store.add_document(
                    RAGDocument(
                        doc_id=f"article_{url[:100]}",
                        ticker=ticker,
                        text=f"[TITRE] {title}\n{content[:3000]}",
                        doc_type=DOC_TYPE_ARTICLE,
                        date_iso=date_now,
                        metadata={
                            "source": url[:200],
                            "signal": signal_final,
                            "impact": str(impact_strength),
                            "absa_aspects": ", ".join(a["aspect"] for a in absa_result.get("aspects", [])),
                        },
                    )
                )

                # Document résumé du débat
                if argument_dominant:
                    rag_store.add_document(
                        RAGDocument(
                            doc_id=f"debate_{url[:100]}",
                            ticker=ticker,
                            text=(
                                f"[DÉBAT {ticker}] Signal: {signal_final} | "
                                f"Force: {impact_strength:.3f}\n"
                                f"Argument: {argument_dominant}\n"
                                f"Transcription: {transcription[:1500]}"
                            ),
                            doc_type=DOC_TYPE_DEBATE,
                            date_iso=date_now,
                            metadata={
                                "signal": signal_final,
                                "consensus_rate": str(round(consensus_rate, 4)),
                                "impact_strength": str(impact_strength),
                                "risk_level": yolo.risk_level,
                            },
                        )
                    )
                    logger.info(
                        "  [RAG] Article + débat indexés | Collection %s : %d documents.",
                        ticker,
                        rag_store.collection_size(ticker),
                    )
            except Exception as e:
                logger.error("  -> Erreur sauvegarde : %s", e)

    logger.info("Agent pipeline terminé.")

    # -----------------------------------------------------------------------
    # Phase AutoDream — Consolidation de fin de session
    # Lance l'Agent de Nuit pour distiller les décisions du jour
    # dans data/memory/<TICKER>.md et mettre à jour MEMORY.md
    # -----------------------------------------------------------------------
    if consolidate_memory:
        logger.info("[Mémoire] Lancement de la consolidation AutoDream...")
        try:
            run_nightly_consolidation()
        except Exception as e:
            logger.warning("[Mémoire] Consolidation ignorée (non bloquante) : %s", e)

    # -----------------------------------------------------------------------
    # Flush du cost tracker LLM (reports/llm_cost_daily/YYYY-MM-DD.json)
    # -----------------------------------------------------------------------
    try:
        path = flush_llm_cost()
        if path is not None:
            logger.info("[LLMCost] Snapshot journalier -> %s", path)
    except Exception as exc:
        logger.debug("[LLMCost] flush échoué : %s", exc)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agent pipeline — POC 2")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Filtrer par tickers (ex: AAPL MSFT). Sans argument : traite tous les articles en attente.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limite d'articles à traiter au total")
    parser.add_argument(
        "--loop", type=int, default=None, help="Intervalle en minutes entre chaque run (minimum 5, ex: --loop 20)"
    )
    args = parser.parse_args()

    if args.loop is not None and args.loop < 5:
        logger.info(f"Erreur : intervalle minimum 5 minutes (tu as mis {args.loop}).")
        exit(1)

    if args.loop:
        logger.info("Mode continu : run toutes les %d minutes", args.loop)
        while True:
            run_agent_pipeline(args.tickers, limit=args.limit)
            logger.info("Prochain run dans %d minutes...", args.loop)
            time.sleep(args.loop * 60)
    else:
        run_agent_pipeline(args.tickers, limit=args.limit)
