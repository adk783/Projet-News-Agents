"""
orchestrator.py — Boucle quotidienne d'autopilote pour la recolte de donnees.

OBJECTIF
--------
Ce module fait le pont entre :
  - TickerDiscoveryEngine (qui choisit qui analyser)
  - ColdStartManager      (qui amorce les tickers inconnus)
  - le pipeline d'analyse (injectable, callable -> dict)
  - DataHarvester         (qui persiste tout en JSONL)

Resultat : une fonction `run_daily_harvest()` qui represente une journee
de campagne autonome. Lancee chaque jour (cron / Task Scheduler), elle
selectionne, analyse, ecrit. Au bout d'une semaine, on a un dataset
exploitable pour benchmarks, fine-tuning, ou simplement audit.

DESIGN
------
- Tout est injectable. Aucune dependance reseau dure dans ce module : les
  composants reseau sont passes en parametres (discovery_engine, ColdStart,
  pipeline_fn). Les tests construisent des mocks.

- Resilient. Une exception sur un ticker ne tue pas la boucle ; elle est
  captee, loggee dans le HarvestRecord (decision="ERROR"), et la boucle
  continue.

- Anti-redondance. La feedback loop avec le harvester se fait via
  `recent_history_from_harvest_dir(...)` cote discovery_engine — pas
  besoin de coupler ici.

- Reproductible. On stocke pipeline_version, prompts_hash, git_sha,
  model_versions dans chaque record — quand on relit le dataset 3 mois
  plus tard, on sait quelle config a produit chaque ligne.

- Optionnellement parallele. `max_workers > 1` lance les analyses dans un
  ThreadPoolExecutor. Par defaut sequentiel (plus facile a debug).

- Timeout per-ticker. Un pipeline qui pendrait sur un appel reseau ne
  bloque pas la journee : Future.result(timeout=...) le tue proprement.

USAGE
-----
    eng = TickerDiscoveryEngine(sources=[...], universe_filter=...)
    cold = ColdStartManager(rag_store=store, ...)
    harvester = DataHarvester(output_dir="data/harvest")

    def my_pipeline(ticker, context):
        # context contient discovery_score, cold_start_report, etc.
        decision = run_full_analysis_pipeline(ticker)
        return {
            "decision": decision.action,
            "confidence": decision.confidence,
            "position_size": decision.size,
            "rationale": decision.rationale,
            "signals": decision.signals,
            "errors": [],
        }

    orch = DailyHarvestOrchestrator(
        discovery_engine=eng,
        harvester=harvester,
        analysis_pipeline_fn=my_pipeline,
        cold_start_manager=cold,
        top_n=10,
        pipeline_version="v3.4.1",
    )
    rep = orch.run_daily_harvest()
    print(rep.summary())
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence

from .data_harvester import DataHarvester, HarvestRecord
from .ticker_discovery import (
    DiscoveryReport,
    DiscoveryScore,
    TickerDiscoveryEngine,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

DEFAULT_TOP_N = 10
DEFAULT_MAX_WORKERS = 1  # sequentiel par defaut
DEFAULT_PER_TICKER_TIMEOUT_SEC = 180.0  # 3 min max par ticker
DEFAULT_COLD_START_BUDGET_SEC = 120.0  # cap dur sur le bootstrap


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TickerRunResult:
    """Resultat de l'analyse d'un ticker dans la boucle quotidienne."""

    ticker: str
    rank: int
    success: bool = False
    decision: Optional[str] = None
    confidence: Optional[float] = None
    position_size: Optional[float] = None
    rationale: Optional[str] = None
    signals: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    cold_start_triggered: bool = False
    cold_start_succeeded: bool = False
    duration_sec: float = 0.0
    harvest_id: Optional[str] = None

    def short(self) -> str:
        flag = "OK" if self.success else "ERR"
        cs = " cold_start" if self.cold_start_triggered else ""
        conf = f" conf={self.confidence:.2f}" if self.confidence is not None else ""
        return f"[{flag}] #{self.rank} {self.ticker} -> {self.decision or '-'}{conf} ({self.duration_sec:.1f}s){cs}"


@dataclass
class OrchestratorReport:
    """Rapport global d'une journee de harvest."""

    run_at: datetime
    date_label: str  # YYYY-MM-DD
    discovery_report: Optional[DiscoveryReport] = None
    ticker_results: List[TickerRunResult] = field(default_factory=list)
    total_duration_sec: float = 0.0
    cold_starts_triggered: int = 0
    cold_starts_succeeded: int = 0
    pipeline_failures: int = 0

    def summary(self) -> str:
        n = len(self.ticker_results)
        succ = sum(1 for r in self.ticker_results if r.success)
        decisions: Dict[str, int] = {}
        for r in self.ticker_results:
            d = r.decision or "?"
            decisions[d] = decisions.get(d, 0) + 1
        dec_str = ", ".join(f"{k}={v}" for k, v in sorted(decisions.items()))
        u_hash = self.discovery_report.universe_hash if self.discovery_report else "-"
        return (
            f"orchestrator date={self.date_label} "
            f"tickers={n} ok={succ} fail={self.pipeline_failures} "
            f"cold_start={self.cold_starts_succeeded}/{self.cold_starts_triggered} "
            f"decisions=[{dec_str}] "
            f"duration={self.total_duration_sec:.1f}s "
            f"universe_hash={u_hash}"
        )

    def detail_lines(self) -> List[str]:
        return [r.short() for r in self.ticker_results]


# ---------------------------------------------------------------------------
# DailyHarvestOrchestrator
# ---------------------------------------------------------------------------


class DailyHarvestOrchestrator:
    """
    Orchestre une journee complete : discovery -> (cold_start) -> pipeline -> harvest.

    `analysis_pipeline_fn` est la cle : c'est la fonction qui execute le
    pipeline complet pour un ticker donne. Elle prend `(ticker, context)`
    et doit retourner un dict avec :
        {
            "decision":      str,       # "BUY" | "SELL" | "HOLD" | "PASS"
            "confidence":    float,     # [0, 1]
            "position_size": float,     # frac portfolio
            "rationale":     str,
            "signals":       Dict[str, Any],
            "errors":        List[str], # optionnel
        }

    Le `context` passe au pipeline contient :
        {
            "discovery_score":    DiscoveryScore,
            "discovery_report":   DiscoveryReport,
            "cold_start_report":  Optional[ColdStartReport],
            "rank":               int,
            "date_label":         str,
        }
    """

    def __init__(
        self,
        discovery_engine: TickerDiscoveryEngine,
        harvester: DataHarvester,
        analysis_pipeline_fn: Callable[[str, Dict[str, Any]], Dict[str, Any]],
        cold_start_manager: Optional[Any] = None,  # ColdStartManager (duck-typed)
        top_n: int = DEFAULT_TOP_N,
        max_workers: int = DEFAULT_MAX_WORKERS,
        per_ticker_timeout_sec: float = DEFAULT_PER_TICKER_TIMEOUT_SEC,
        cold_start_budget_sec: float = DEFAULT_COLD_START_BUDGET_SEC,
        pipeline_version: Optional[str] = None,
        prompts_hash: Optional[str] = None,
        git_sha: Optional[str] = None,
        model_versions: Optional[Dict[str, str]] = None,
        on_ticker_done: Optional[Callable[[TickerRunResult], None]] = None,
        skip_already_harvested_today: bool = False,
    ):
        self.discovery_engine = discovery_engine
        self.harvester = harvester
        self.analysis_pipeline_fn = analysis_pipeline_fn
        self.cold_start_manager = cold_start_manager
        self.top_n = max(1, int(top_n))
        self.max_workers = max(1, int(max_workers))
        self.per_ticker_timeout_sec = max(1.0, float(per_ticker_timeout_sec))
        self.cold_start_budget_sec = max(1.0, float(cold_start_budget_sec))
        self.pipeline_version = pipeline_version
        self.prompts_hash = prompts_hash
        self.git_sha = git_sha
        self.model_versions = dict(model_versions or {})
        self.on_ticker_done = on_ticker_done
        self.skip_already_harvested_today = bool(skip_already_harvested_today)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_daily_harvest(
        self,
        *,
        date_label: Optional[str] = None,
        company_name_lookup: Optional[Callable[[str], Optional[str]]] = None,
        sector_lookup: Optional[Callable[[str], Optional[str]]] = None,
    ) -> OrchestratorReport:
        """
        Execute une journee complete.

        Args:
            date_label : YYYY-MM-DD a forcer (sinon today UTC). Utile pour
                         backfill ou tests deterministes.
            company_name_lookup : optionnel, fournit le company_name au
                         ColdStartManager si dispo (ameliore les news
                         backfill par recherche textuelle).
            sector_lookup : idem pour le secteur (utilise dans peer fallback).
        """
        t0 = time.time()
        run_at = datetime.now(timezone.utc)
        if not date_label:
            date_label = run_at.strftime("%Y-%m-%d")

        report = OrchestratorReport(run_at=run_at, date_label=date_label)

        # 1. Discovery
        try:
            disc = self.discovery_engine.discover(top_n=self.top_n)
            report.discovery_report = disc
        except Exception as e:
            logger.exception("[Orchestrator] discovery a leve : %s", e)
            report.total_duration_sec = time.time() - t0
            return report

        if not disc.scores:
            logger.warning("[Orchestrator] discovery a renvoye 0 ticker — fin")
            report.total_duration_sec = time.time() - t0
            return report

        # 2. Filtrage des tickers deja harvested aujourd'hui (idempotence soft)
        scores = list(disc.scores)
        if self.skip_already_harvested_today:
            already = self._tickers_already_harvested(date_label)
            if already:
                before = len(scores)
                scores = [s for s in scores if s.ticker.upper() not in already]
                skipped = before - len(scores)
                if skipped:
                    logger.info(
                        "[Orchestrator] %d tickers deja harvested aujourd'hui — skip",
                        skipped,
                    )

        if not scores:
            report.total_duration_sec = time.time() - t0
            return report

        # 3. Pipeline per ticker
        if self.max_workers <= 1:
            results = self._run_sequential(
                scores=scores,
                discovery_report=disc,
                date_label=date_label,
                company_name_lookup=company_name_lookup,
                sector_lookup=sector_lookup,
            )
        else:
            results = self._run_parallel(
                scores=scores,
                discovery_report=disc,
                date_label=date_label,
                company_name_lookup=company_name_lookup,
                sector_lookup=sector_lookup,
            )

        report.ticker_results = results
        report.cold_starts_triggered = sum(1 for r in results if r.cold_start_triggered)
        report.cold_starts_succeeded = sum(1 for r in results if r.cold_start_succeeded)
        report.pipeline_failures = sum(1 for r in results if not r.success)
        report.total_duration_sec = time.time() - t0

        logger.info("[Orchestrator] %s", report.summary())
        return report

    # ------------------------------------------------------------------
    # Internal — sequential / parallel
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        scores: Sequence[DiscoveryScore],
        discovery_report: DiscoveryReport,
        date_label: str,
        company_name_lookup: Optional[Callable[[str], Optional[str]]],
        sector_lookup: Optional[Callable[[str], Optional[str]]],
    ) -> List[TickerRunResult]:
        out: List[TickerRunResult] = []
        for idx, score in enumerate(scores):
            res = self._handle_single_ticker(
                score=score,
                rank=idx + 1,
                discovery_report=discovery_report,
                date_label=date_label,
                company_name_lookup=company_name_lookup,
                sector_lookup=sector_lookup,
            )
            out.append(res)
            if self.on_ticker_done is not None:
                try:
                    self.on_ticker_done(res)
                except Exception as e:
                    logger.debug("[Orchestrator] on_ticker_done callback err : %s", e)
        return out

    def _run_parallel(
        self,
        scores: Sequence[DiscoveryScore],
        discovery_report: DiscoveryReport,
        date_label: str,
        company_name_lookup: Optional[Callable[[str], Optional[str]]],
        sector_lookup: Optional[Callable[[str], Optional[str]]],
    ) -> List[TickerRunResult]:
        out: List[TickerRunResult] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(
                    self._handle_single_ticker,
                    score=score,
                    rank=idx + 1,
                    discovery_report=discovery_report,
                    date_label=date_label,
                    company_name_lookup=company_name_lookup,
                    sector_lookup=sector_lookup,
                ): (idx, score)
                for idx, score in enumerate(scores)
            }
            for fut in as_completed(futures):
                idx, score = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    logger.exception(
                        "[Orchestrator] thread crash sur %s : %s",
                        score.ticker,
                        e,
                    )
                    res = TickerRunResult(
                        ticker=score.ticker,
                        rank=idx + 1,
                        success=False,
                        decision="ERROR",
                        errors=[f"thread_crash: {e}"],
                    )
                out.append(res)
                if self.on_ticker_done is not None:
                    try:
                        self.on_ticker_done(res)
                    except Exception as e2:
                        logger.debug("[Orchestrator] callback err : %s", e2)
        # Garder l'ordre par rank pour lisibilite
        out.sort(key=lambda r: r.rank)
        return out

    # ------------------------------------------------------------------
    # Internal — single ticker pipeline
    # ------------------------------------------------------------------

    def _handle_single_ticker(
        self,
        score: DiscoveryScore,
        rank: int,
        discovery_report: DiscoveryReport,
        date_label: str,
        company_name_lookup: Optional[Callable[[str], Optional[str]]],
        sector_lookup: Optional[Callable[[str], Optional[str]]],
    ) -> TickerRunResult:
        t0 = time.time()
        ticker = score.ticker.upper()
        result = TickerRunResult(ticker=ticker, rank=rank)

        cold_report = None

        # 1. Cold start (optionnel)
        if self.cold_start_manager is not None:
            try:
                if self.cold_start_manager.is_cold(ticker):
                    result.cold_start_triggered = True
                    company = company_name_lookup(ticker) if company_name_lookup else None
                    sector = sector_lookup(ticker) if sector_lookup else None

                    cold_report = self._run_with_timeout(
                        fn=lambda: self.cold_start_manager.bootstrap(
                            ticker=ticker,
                            company_name=company,
                            sector=sector,
                        ),
                        timeout=self.cold_start_budget_sec,
                        what="cold_start",
                    )
                    if cold_report is not None and not cold_report.is_cold_after:
                        result.cold_start_succeeded = True
            except Exception as e:
                logger.warning(
                    "[Orchestrator] cold_start exception sur %s : %s",
                    ticker,
                    e,
                )
                result.errors.append(f"cold_start: {e}")

        # 2. Pipeline d'analyse
        context: Dict[str, Any] = {
            "discovery_score": score,
            "discovery_report": discovery_report,
            "cold_start_report": cold_report,
            "rank": rank,
            "date_label": date_label,
        }

        try:
            pipeline_out = self._run_with_timeout(
                fn=lambda: self.analysis_pipeline_fn(ticker, context),
                timeout=self.per_ticker_timeout_sec,
                what="pipeline",
            )
        except Exception as e:
            logger.exception("[Orchestrator] pipeline exception %s : %s", ticker, e)
            pipeline_out = None
            result.errors.append(f"pipeline: {e}")
            result.errors.append(traceback.format_exc(limit=3))

        if not isinstance(pipeline_out, dict):
            result.success = False
            result.decision = "ERROR"
            result.duration_sec = time.time() - t0
            self._persist(result, score, discovery_report, cold_report)
            return result

        # 3. Extraction structuree
        try:
            result.decision = str(pipeline_out.get("decision", "")).upper() or None
            conf = pipeline_out.get("confidence")
            result.confidence = float(conf) if conf is not None else None
            ps = pipeline_out.get("position_size")
            result.position_size = float(ps) if ps is not None else None
            result.rationale = str(pipeline_out.get("rationale", ""))[:2000] if pipeline_out.get("rationale") else None
            sigs = pipeline_out.get("signals") or {}
            if isinstance(sigs, dict):
                result.signals = dict(sigs)
            errs = pipeline_out.get("errors") or []
            if isinstance(errs, list):
                result.errors.extend(str(e) for e in errs)
            # success = on a au moins une decision et pas de seuil d'erreur fatal
            result.success = bool(result.decision and result.decision != "ERROR")
        except Exception as e:
            logger.warning(
                "[Orchestrator] extraction pipeline_out echec %s : %s",
                ticker,
                e,
            )
            result.errors.append(f"extract: {e}")
            result.success = False
            result.decision = result.decision or "ERROR"

        result.duration_sec = time.time() - t0

        # 4. Persist via DataHarvester
        self._persist(result, score, discovery_report, cold_report)
        return result

    # ------------------------------------------------------------------
    # Internal — persistence
    # ------------------------------------------------------------------

    def _persist(
        self,
        result: TickerRunResult,
        score: DiscoveryScore,
        discovery_report: DiscoveryReport,
        cold_report: Optional[Any],
    ) -> None:
        signals = dict(result.signals)
        # Enrichissement automatique avec quelques meta cold_start
        if cold_report is not None:
            try:
                signals["cold_start_was_cold"] = bool(getattr(cold_report, "was_cold", False))
                signals["cold_start_is_cold_after"] = bool(getattr(cold_report, "is_cold_after", False))
                signals["cold_start_docs_added"] = int(getattr(cold_report, "docs_after", 0)) - int(
                    getattr(cold_report, "docs_before", 0)
                )
                peers = list(getattr(cold_report, "peers_used", []) or [])
                if peers:
                    signals["cold_start_peers_used"] = peers
            except Exception:
                pass

        rec = HarvestRecord(
            ticker=result.ticker,
            rank=result.rank,
            discovery_total_score=score.total_score,
            discovery_raw_score=score.raw_score,
            discovery_contributions=dict(score.contributions),
            discovery_novelty_multiplier=score.novelty_multiplier,
            discovery_days_since_last_seen=score.days_since_last_seen,
            universe_hash=discovery_report.universe_hash,
            decision=result.decision,
            confidence=result.confidence,
            position_size=result.position_size,
            rationale=result.rationale,
            pipeline_duration_sec=result.duration_sec,
            pipeline_errors=list(result.errors),
            signals=signals,
            pipeline_version=self.pipeline_version,
            prompts_hash=self.prompts_hash,
            git_sha=self.git_sha,
            model_versions=dict(self.model_versions),
        )
        try:
            result.harvest_id = self.harvester.write(rec)
        except Exception as e:
            logger.exception(
                "[Orchestrator] harvester.write echec %s : %s",
                result.ticker,
                e,
            )
            result.errors.append(f"persist: {e}")

    # ------------------------------------------------------------------
    # Internal — timeout helper
    # ------------------------------------------------------------------

    @staticmethod
    def _run_with_timeout(fn: Callable[[], Any], timeout: float, what: str) -> Any:
        """
        Execute fn() avec un timeout dur. Implemente avec un mini-pool a 1 worker
        (sans creer de thread pour les timeouts triviaux).

        Note : pour les usages legers et tests, on n'instancie pas systematiquement
        un pool ; on l'utilise seulement quand timeout > 0 et que c'est utile.
        """
        if timeout <= 0:
            return fn()
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(fn)
            try:
                return fut.result(timeout=timeout)
            except FuturesTimeout as e:
                logger.warning("[Orchestrator] %s timeout (%.1fs)", what, timeout)
                raise TimeoutError(f"{what} timeout after {timeout:.1f}s") from e

    # ------------------------------------------------------------------
    # Internal — idempotence helper
    # ------------------------------------------------------------------

    def _tickers_already_harvested(self, date_label: str) -> set:
        """Liste les tickers deja persistes pour la date donnee."""
        seen: set = set()
        try:
            for r in self.harvester.iter_records(start_date=date_label, end_date=date_label):
                tk = str(r.get("ticker", "")).upper()
                if tk:
                    seen.add(tk)
        except Exception as e:
            logger.debug("[Orchestrator] idempotence lookup err : %s", e)
        return seen


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    from .ticker_discovery import (
        CallableSource,
        UniverseFilter,
    )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # ----- Mock discovery -----
    src_news = CallableSource(
        "trending_news",
        lambda: {"AAPL": 0.9, "NVDA": 0.85, "TSLA": 0.7, "GME": 0.5},
    )
    src_volume = CallableSource(
        "volume_anomaly",
        lambda: {"NVDA": 0.95, "AAPL": 0.4},
    )
    src_earnings = CallableSource(
        "earnings_calendar",
        lambda: {"MSFT": 1.0, "AAPL": 0.85},
    )
    eng = TickerDiscoveryEngine(
        sources=[src_news, src_volume, src_earnings],
        universe_filter=UniverseFilter(
            min_market_cap=0,  # off pour smoke test
            exclude_etfs=True,
        ),
    )

    # ----- Mock cold_start manager (duck-typed) -----
    class _MockColdStart:
        def __init__(self):
            self._known = {"AAPL", "NVDA"}  # AAPL et NVDA sont chauds, TSLA/GME cold
            self.calls = []

        def is_cold(self, ticker):
            return ticker.upper() not in self._known

        def bootstrap(self, ticker, company_name=None, sector=None):
            self.calls.append(ticker)
            self._known.add(ticker.upper())

            class _Rep:
                was_cold = True
                is_cold_after = False
                docs_before = 0
                docs_after = 8
                peers_used = ["MSFT"]

            return _Rep()

    cold = _MockColdStart()

    # ----- Mock pipeline -----
    def my_pipeline(ticker, context):
        # Decision triviale basee sur le rank
        rank = context["rank"]
        if rank <= 2:
            decision = "BUY"
            conf = 0.82
            size = 0.10
        elif rank <= 4:
            decision = "HOLD"
            conf = 0.55
            size = 0.0
        else:
            decision = "PASS"
            conf = 0.30
            size = 0.0
        return {
            "decision": decision,
            "confidence": conf,
            "position_size": size,
            "rationale": f"smoke test decision for {ticker} at rank {rank}",
            "signals": {
                "discovery_total_score": context["discovery_score"].total_score,
                "transcript_red_flag": 0.18,
                "social_sentiment": 0.45,
            },
            "errors": [],
        }

    with tempfile.TemporaryDirectory() as tmp:
        harvester = DataHarvester(output_dir=tmp)
        orch = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=harvester,
            analysis_pipeline_fn=my_pipeline,
            cold_start_manager=cold,
            top_n=5,
            pipeline_version="smoke-1.0",
            prompts_hash="abcdef",
            model_versions={"absa": "llama-3.3-70b"},
        )

        rep = orch.run_daily_harvest()
        logger.info("=== Daily harvest report ===")
        print(rep.summary())
        for line in rep.detail_lines():
            print(" ", line)

        logger.info(f"\nCold start calls: {cold.calls}")

        # Verifications
        assert rep.discovery_report is not None
        assert len(rep.ticker_results) <= 5
        assert all(r.harvest_id for r in rep.ticker_results), "tous doivent etre persistes"
        assert rep.cold_starts_triggered >= 1, "au moins TSLA/GME devrait avoir trigger cold_start"
        assert rep.cold_starts_succeeded == rep.cold_starts_triggered, "tous les bootstrap doivent reussir"

        # Verif persistence
        records = harvester.load_records()
        assert len(records) == len(rep.ticker_results)
        for r in records:
            assert r["pipeline"]["decision"] in {"BUY", "HOLD", "PASS"}
            assert r["meta"]["pipeline_version"] == "smoke-1.0"
            assert r["discovery"]["universe_hash"]

        # Re-run avec skip_already_harvested -> rien ne doit s'ajouter
        orch2 = DailyHarvestOrchestrator(
            discovery_engine=eng,
            harvester=harvester,
            analysis_pipeline_fn=my_pipeline,
            cold_start_manager=cold,
            top_n=5,
            pipeline_version="smoke-1.0",
            skip_already_harvested_today=True,
        )
        rep2 = orch2.run_daily_harvest()
        # Tickers analyses la 1re fois ne doivent pas etre re-traites
        assert len(rep2.ticker_results) == 0, "tickers deja harvested doivent etre skip"

        logger.info(
            "\nOK - DailyHarvestOrchestrator fonctionne (discovery + cold_start + pipeline + harvest + idempotence)"
        )
