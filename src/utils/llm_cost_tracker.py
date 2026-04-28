"""
llm_cost_tracker.py — Compteur de coût LLM + garde-fou budgétaire

OBJECTIF
--------
1. Compter les tokens entrée/sortie de chaque appel LLM
2. Convertir en USD via `LLM_PRICING_USD_PER_1M_TOKENS` (src/config.py)
3. Bloquer le pipeline si le cumul dépasse `LLM_DAILY_BUDGET_USD`
4. Dumper quotidiennement dans `reports/llm_cost_daily/YYYY-MM-DD.json`

INTÉGRATION
-----------
    from src.utils.llm_cost_tracker import track_llm_call, BudgetExceededError

    resp = client.chat.completions.create(...)
    track_llm_call(
        model="llama-3.1-8b-instant",
        usage=resp.usage,            # OpenAI-style {prompt_tokens, completion_tokens}
    )
    # raise BudgetExceededError si dépassement

RÉFÉRENCES
----------
  Tarifs Groq / Cerebras / Mistral (oct 2025) — cf src/config.py
  Approche token-count inspirée de langchain.callbacks.OpenAICallbackHandler.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import (
    LLM_COST_LOG_DIR,
    LLM_DAILY_BUDGET_USD,
    LLM_PRICING_USD_PER_1M_TOKENS,
)

logger = logging.getLogger("LLMCostTracker")


class BudgetExceededError(RuntimeError):
    """Levée quand LLM_DAILY_BUDGET_USD est franchi."""

    pass


# ---------------------------------------------------------------------------
# Singleton thread-safe
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_state: dict = {
    "date": None,  # YYYY-MM-DD
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_usd": 0.0,
    "calls_by_model": {},  # {model_name: {calls, prompt_tokens, completion_tokens, usd}}
}


def _today_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _rotate_if_new_day() -> None:
    """Au passage d'un jour, flush le snapshot sur disque et reset."""
    today = _today_iso()
    if _state["date"] is None:
        _state["date"] = today
        return
    if _state["date"] != today:
        # Dump du jour écoulé puis reset
        try:
            _dump_snapshot(_state["date"])
        except Exception as exc:
            logger.warning("Échec dump cost snapshot : %s", exc)
        _state["date"] = today
        _state["total_prompt_tokens"] = 0
        _state["total_completion_tokens"] = 0
        _state["total_usd"] = 0.0
        _state["calls_by_model"] = {}


def _dump_snapshot(date_str: str) -> Path:
    """Écrit le snapshot courant dans reports/llm_cost_daily/YYYY-MM-DD.json."""
    out_dir = Path(LLM_COST_LOG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}.json"
    payload = {
        "date": date_str,
        "total_prompt_tokens": _state["total_prompt_tokens"],
        "total_completion_tokens": _state["total_completion_tokens"],
        "total_usd": round(_state["total_usd"], 4),
        "budget_usd": LLM_DAILY_BUDGET_USD,
        "budget_used_pct": round(100 * _state["total_usd"] / LLM_DAILY_BUDGET_USD, 2)
        if LLM_DAILY_BUDGET_USD > 0
        else None,
        "calls_by_model": _state["calls_by_model"],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------


def track_llm_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """
    Enregistre un appel LLM, retourne le coût USD de cet appel.

    Raises:
        BudgetExceededError : si le cumul journalier dépasse LLM_DAILY_BUDGET_USD.
    """
    with _lock:
        _rotate_if_new_day()

        price_per_1m = LLM_PRICING_USD_PER_1M_TOKENS.get(model)
        if price_per_1m is None:
            # Modèle inconnu → tarif fallback conservateur (0.50 $/M)
            price_per_1m = 0.50
            logger.debug("Modèle %s inconnu → fallback $0.50/M tokens", model)

        # Simplification : on applique le prix moyen (in+out combinés) à
        # l'intégralité des tokens. Les providers facturent souvent in/out
        # différemment — pour audit plus fin, passer au schéma à deux prix.
        total_tokens = prompt_tokens + completion_tokens
        call_usd = total_tokens / 1_000_000.0 * price_per_1m

        _state["total_prompt_tokens"] += prompt_tokens
        _state["total_completion_tokens"] += completion_tokens
        _state["total_usd"] += call_usd

        bucket = _state["calls_by_model"].setdefault(
            model,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "usd": 0.0,
            },
        )
        bucket["calls"] += 1
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["usd"] += call_usd

        # Garde-fou : blocage dur si budget dépassé
        if LLM_DAILY_BUDGET_USD > 0 and _state["total_usd"] > LLM_DAILY_BUDGET_USD:
            total = _state["total_usd"]
            raise BudgetExceededError(
                f"Budget LLM quotidien dépassé : ${total:.2f} > ${LLM_DAILY_BUDGET_USD:.2f}. "
                f"Pipeline bloqué. Pour débloquer : augmenter LLM_DAILY_BUDGET_USD "
                f"ou attendre minuit UTC."
            )

        # Warn à 80 % du budget
        if LLM_DAILY_BUDGET_USD > 0 and _state["total_usd"] > 0.80 * LLM_DAILY_BUDGET_USD:
            logger.warning(
                "[LLMCost] %.1f%% du budget quotidien utilisé ($%.2f / $%.2f)",
                100 * _state["total_usd"] / LLM_DAILY_BUDGET_USD,
                _state["total_usd"],
                LLM_DAILY_BUDGET_USD,
            )

        return call_usd


def track_from_openai_usage(model: str, usage) -> float:
    """
    Wrapper quand on a un objet `usage` OpenAI-compatible (Groq, Cerebras,
    Mistral renvoient tous `usage.prompt_tokens` / `usage.completion_tokens`).
    """
    if usage is None:
        return 0.0
    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    return track_llm_call(model, prompt, completion)


def current_snapshot() -> dict:
    """Snapshot lecture seule (pour monitoring / reports)."""
    with _lock:
        _rotate_if_new_day()
        return {
            "date": _state["date"],
            "total_usd": round(_state["total_usd"], 4),
            "budget_usd": LLM_DAILY_BUDGET_USD,
            "calls_by_model": dict(_state["calls_by_model"]),
        }


def flush_snapshot() -> Optional[Path]:
    """Force un dump du snapshot courant sur disque (utile fin de pipeline)."""
    with _lock:
        if _state["date"] is None:
            return None
        return _dump_snapshot(_state["date"])


if __name__ == "__main__":
    # Smoke test
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    track_llm_call("llama-3.1-8b-instant", prompt_tokens=1200, completion_tokens=300)
    track_llm_call("llama-4-scout-17b", prompt_tokens=1800, completion_tokens=450)
    track_llm_call("llama-3.3-70b-versatile", prompt_tokens=2400, completion_tokens=600)
    print(json.dumps(current_snapshot(), indent=2))
    p = flush_snapshot()
    print("Snapshot ->", p)
