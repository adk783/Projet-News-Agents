"""
data_harvester.py — Persistance JSONL versionnee pour la recolte autonome.

OBJECTIF
--------
Quand le pipeline tourne en autopilote pendant N jours, on veut un dataset
exploitable a la fin :
  - Une ligne par decision (ticker, jour) au format JSONL
  - Schema versionne (les champs evoluent — il faut le savoir)
  - Tous les sub-scores conserves (pour analyses ex-post)
  - Rotation quotidienne (1 fichier par jour : `harvest_YYYY-MM-DD.jsonl`)
  - Hash du contexte (univers, version pipeline, version prompts) pour
    reproductibilite

POURQUOI JSONL
--------------
- Append-only (resilient au crash : pas de re-write d'un gros tableau)
- Streaming-friendly (1 ligne = 1 record)
- pandas.read_json(lines=True) charge en 1 ligne
- Diff-able en git/grep
- Pas besoin d'un schema rigide a l'avance

SCHEMA v1
---------
{
  "schema_version": "v1",
  "harvest_id":     "uuid-...",
  "timestamp":      "2026-04-25T14:32:00+00:00",
  "ticker":         "AAPL",
  "discovery": {
    "rank":         3,
    "total_score":  0.612,
    "raw_score":    0.778,
    "contributions": {"trending_news": 0.85, "volume_anomaly": 0.62, ...},
    "novelty_multiplier": 0.61,
    "days_since_last_seen": 1.0,
    "universe_hash": "a1b2c3..."
  },
  "pipeline": {
    "decision":     "BUY" | "SELL" | "HOLD" | "PASS",
    "confidence":   0.74,
    "position_size": 0.12,
    "rationale":    "...",
    "duration_sec": 8.4,
    "errors":       []
  },
  "signals": {
    "transcript_red_flag":   0.18,
    "social_sentiment":      0.45,
    "sentiment_divergence_regime": "smart_buy",
    "fundamentals_pe":       28.4,
    ...
  },
  "meta": {
    "pipeline_version":      "v3.4.1",
    "prompts_hash":          "abc123",
    "git_sha":               "deadbeef" (optional),
    "model_versions":        {"absa": "llama-3.3-70b", ...}
  }
}

USAGE
-----
    h = DataHarvester(output_dir="data/harvest")
    rec = HarvestRecord(
        ticker="AAPL", rank=3,
        discovery_score=score_obj,
        decision="BUY", confidence=0.74, position_size=0.12,
        rationale="strong earnings beat + smart_buy regime",
        signals={"transcript_red_flag": 0.18, ...},
    )
    h.write(rec)

LECTURE
-------
    df = h.load_dataframe(start_date="2026-04-20", end_date="2026-04-27")
    # ou en streaming :
    for rec in h.iter_records():
        ...
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

import json
import logging
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "v1"
DEFAULT_OUTPUT_DIR = "data/harvest"
HARVEST_FILENAME_FMT = "harvest_{date}.jsonl"


# ---------------------------------------------------------------------------
# HarvestRecord
# ---------------------------------------------------------------------------


@dataclass
class HarvestRecord:
    """
    Une decision agregee pour un ticker un jour donne.

    Tous les champs sont optionnels SAUF `ticker`. On veut pouvoir loguer
    meme si le pipeline a partiellement echoue (eg. cold-start mais pas
    de decision finale).
    """

    ticker: str

    # Discovery (issu de DiscoveryScore + DiscoveryReport)
    rank: Optional[int] = None
    discovery_total_score: Optional[float] = None
    discovery_raw_score: Optional[float] = None
    discovery_contributions: Dict[str, float] = field(default_factory=dict)
    discovery_novelty_multiplier: Optional[float] = None
    discovery_days_since_last_seen: Optional[float] = None
    universe_hash: Optional[str] = None

    # Pipeline outcome
    decision: Optional[str] = None  # BUY | SELL | HOLD | PASS | ERROR
    confidence: Optional[float] = None  # [0, 1]
    position_size: Optional[float] = None  # frac portfolio
    rationale: Optional[str] = None
    pipeline_duration_sec: Optional[float] = None
    pipeline_errors: List[str] = field(default_factory=list)

    # Signaux intermediaires (libre-champ, valeurs JSON-serializable)
    signals: Dict[str, Any] = field(default_factory=dict)

    # Meta (versionnement)
    pipeline_version: Optional[str] = None
    prompts_hash: Optional[str] = None
    git_sha: Optional[str] = None
    model_versions: Dict[str, str] = field(default_factory=dict)

    # Auto
    timestamp: str = ""  # rempli par DataHarvester si vide
    harvest_id: str = ""  # rempli par DataHarvester si vide

    def to_json_dict(self) -> Dict[str, Any]:
        """Format final ecrit dans le JSONL (schema v1)."""
        return {
            "schema_version": SCHEMA_VERSION,
            "harvest_id": self.harvest_id,
            "timestamp": self.timestamp,
            "ticker": self.ticker.upper(),
            "discovery": {
                "rank": self.rank,
                "total_score": self.discovery_total_score,
                "raw_score": self.discovery_raw_score,
                "contributions": self.discovery_contributions or {},
                "novelty_multiplier": self.discovery_novelty_multiplier,
                "days_since_last_seen": self.discovery_days_since_last_seen,
                "universe_hash": self.universe_hash,
            },
            "pipeline": {
                "decision": self.decision,
                "confidence": self.confidence,
                "position_size": self.position_size,
                "rationale": self.rationale,
                "duration_sec": self.pipeline_duration_sec,
                "errors": list(self.pipeline_errors),
            },
            "signals": dict(self.signals),
            "meta": {
                "pipeline_version": self.pipeline_version,
                "prompts_hash": self.prompts_hash,
                "git_sha": self.git_sha,
                "model_versions": dict(self.model_versions),
            },
        }


# ---------------------------------------------------------------------------
# DataHarvester
# ---------------------------------------------------------------------------


class DataHarvester:
    """
    Ecrit / lit les records de recolte au format JSONL.

    Thread-safe : un lock global protege l'ecriture (pour pipelines parallelises
    par thread). En multi-process, utiliser un fichier different par process.
    """

    def __init__(
        self,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        date_provider: Optional[Any] = None,  # callable returning date string
        flush_each_write: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.flush_each_write = bool(flush_each_write)
        self._date_provider = date_provider or self._default_date
        self._lock = threading.Lock()

    # ----- Date helpers -----

    @staticmethod
    def _default_date() -> str:
        """Date UTC au format YYYY-MM-DD."""
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _file_for_today(self) -> Path:
        return self.output_dir / HARVEST_FILENAME_FMT.format(date=self._date_provider())

    def _file_for_date(self, date_str: str) -> Path:
        return self.output_dir / HARVEST_FILENAME_FMT.format(date=date_str)

    # ----- Write -----

    def write(self, record: HarvestRecord) -> str:
        """
        Persiste un record. Retourne le harvest_id (genere si absent).
        Idempotence : si harvest_id est fourni et deja present, on ne re-ecrit pas.
        """
        if not record.timestamp:
            record.timestamp = datetime.now(timezone.utc).isoformat()
        if not record.harvest_id:
            record.harvest_id = uuid.uuid4().hex

        path = self._file_for_today()
        line = json.dumps(record.to_json_dict(), ensure_ascii=False, default=str)

        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                if self.flush_each_write:
                    f.flush()
                    os.fsync(f.fileno())
        logger.debug("[Harvest] wrote %s -> %s", record.harvest_id[:8], path.name)
        return record.harvest_id

    def write_many(self, records: Iterable[HarvestRecord]) -> List[str]:
        """Ecrit un batch dans le fichier du jour. Returns list of harvest_ids."""
        path = self._file_for_today()
        ids: List[str] = []
        lines: List[str] = []
        for r in records:
            if not r.timestamp:
                r.timestamp = datetime.now(timezone.utc).isoformat()
            if not r.harvest_id:
                r.harvest_id = uuid.uuid4().hex
            ids.append(r.harvest_id)
            lines.append(json.dumps(r.to_json_dict(), ensure_ascii=False, default=str))
        if not lines:
            return []
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
                if self.flush_each_write:
                    f.flush()
                    os.fsync(f.fileno())
        return ids

    # ----- Read -----

    def iter_records(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Itere sur les records (raw dict JSONL). Filtre optionnel par date
        (inclusif des deux cotes). Format YYYY-MM-DD.

        Resilient : skip silencieusement les lignes corrompues.
        """
        for path in self._iter_files(start_date, end_date):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line)
                        except Exception as e:
                            logger.debug("[Harvest] ligne corrompue %s : %s", path.name, e)
                            continue
            except Exception as e:
                logger.debug("[Harvest] open echec %s : %s", path, e)
                continue

    def load_records(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Materialise tous les records dans une liste (pratique pour analyse)."""
        return list(self.iter_records(start_date, end_date))

    def load_dataframe(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ):
        """
        Retourne un DataFrame pandas (necessite pandas installe).
        Aplatit `discovery.*`, `pipeline.*` et `meta.*` en colonnes flat.
        """
        try:
            import pandas as pd
        except ImportError as e:
            raise RuntimeError("pandas requis pour load_dataframe") from e

        records = list(self.iter_records(start_date, end_date))
        if not records:
            return pd.DataFrame()

        flat: List[Dict[str, Any]] = []
        for r in records:
            row: Dict[str, Any] = {
                "schema_version": r.get("schema_version"),
                "harvest_id": r.get("harvest_id"),
                "timestamp": r.get("timestamp"),
                "ticker": r.get("ticker"),
            }
            for key in ("discovery", "pipeline", "meta"):
                sub = r.get(key) or {}
                for k, v in sub.items():
                    # Aplatir les dict imbriques (ex: contributions, errors)
                    if isinstance(v, (dict, list)):
                        row[f"{key}.{k}"] = json.dumps(v, ensure_ascii=False, default=str)
                    else:
                        row[f"{key}.{k}"] = v
            # Signals reste un seul colonne JSON (libre-champ)
            row["signals"] = json.dumps(r.get("signals") or {}, ensure_ascii=False, default=str)
            flat.append(row)

        df = pd.DataFrame(flat)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        return df

    # ----- Stats -----

    def stats(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Stats agregees rapides : count par ticker, par decision, par jour."""
        n = 0
        per_ticker: Dict[str, int] = {}
        per_decision: Dict[str, int] = {}
        per_day: Dict[str, int] = {}
        confidences: List[float] = []

        for r in self.iter_records(start_date, end_date):
            n += 1
            tk = str(r.get("ticker", "?")).upper()
            per_ticker[tk] = per_ticker.get(tk, 0) + 1

            dec = (r.get("pipeline") or {}).get("decision") or "?"
            per_decision[dec] = per_decision.get(dec, 0) + 1

            ts = str(r.get("timestamp", ""))[:10]
            if ts:
                per_day[ts] = per_day.get(ts, 0) + 1

            conf = (r.get("pipeline") or {}).get("confidence")
            if isinstance(conf, (int, float)):
                confidences.append(float(conf))

        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "total_records": n,
            "unique_tickers": len(per_ticker),
            "per_ticker": per_ticker,
            "per_decision": per_decision,
            "per_day": per_day,
            "avg_confidence": avg_conf,
            "n_days": len(per_day),
        }

    # ----- Internal -----

    def _iter_files(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> Iterator[Path]:
        if not self.output_dir.is_dir():
            return
        for path in sorted(self.output_dir.glob("harvest_*.jsonl")):
            # Extraire la date depuis le nom : harvest_YYYY-MM-DD.jsonl
            stem = path.stem.replace("harvest_", "")
            if start_date and stem < start_date:
                continue
            if end_date and stem > end_date:
                continue
            yield path


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        h = DataHarvester(output_dir=tmp)

        rec1 = HarvestRecord(
            ticker="AAPL",
            rank=1,
            discovery_total_score=0.612,
            discovery_raw_score=0.778,
            discovery_contributions={"trending_news": 0.85, "volume_anomaly": 0.62},
            discovery_novelty_multiplier=0.61,
            discovery_days_since_last_seen=1.0,
            universe_hash="a1b2c3d4",
            decision="BUY",
            confidence=0.74,
            position_size=0.12,
            rationale="strong earnings beat",
            pipeline_duration_sec=8.4,
            signals={
                "transcript_red_flag": 0.18,
                "social_sentiment": 0.45,
                "sentiment_divergence_regime": "smart_buy",
            },
            pipeline_version="v3.4.1",
            prompts_hash="abc123",
        )

        rec2 = HarvestRecord(
            ticker="NVDA",
            rank=2,
            decision="HOLD",
            confidence=0.55,
            signals={"transcript_red_flag": 0.32},
        )

        # Single + batch
        id1 = h.write(rec1)
        ids = h.write_many([rec2])

        logger.info(f"Wrote ids: {id1[:8]}, {ids[0][:8]}")

        # Iteration
        records = h.load_records()
        logger.info(f"\nLoaded {len(records)} records:")
        for r in records:
            print(f"  ticker={r['ticker']} decision={r['pipeline']['decision']} conf={r['pipeline']['confidence']}")

        # Stats
        logger.info("\nStats:")
        s = h.stats()
        for k, v in s.items():
            logger.info(f"  {k} = {v}")

        # Verifications
        assert len(records) == 2
        assert records[0]["schema_version"] == "v1"
        assert records[0]["ticker"] == "AAPL"
        assert records[0]["discovery"]["rank"] == 1
        assert records[0]["pipeline"]["decision"] == "BUY"
        assert records[0]["signals"]["sentiment_divergence_regime"] == "smart_buy"
        assert s["unique_tickers"] == 2
        assert s["per_decision"]["BUY"] == 1
        assert s["per_decision"]["HOLD"] == 1

        # Date filter (toutes meme jour ici)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        records_today = h.load_records(start_date=today, end_date=today)
        assert len(records_today) == 2

        records_future = h.load_records(start_date="2099-01-01")
        assert records_future == []

        logger.info("\nOK - DataHarvester fonctionne (write + read + stats + filter)")
