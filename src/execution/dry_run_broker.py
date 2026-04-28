"""DryRunBroker : implementation BrokerProtocol qui ne passe AUCUN ordre reel.

Tous les ordres sont :
- Loggues en JSONL append-only dans `DRY_RUN_LOG_PATH` (defaut `logs/dry_run_trades.jsonl`).
- Marques `OrderStatus.SIMULATED` pour audit clair.
- Acceptes systematiquement (pas de rejet sauf qty<=0).

C'est le broker par defaut tant qu'on n'a pas configure `BROKER_BACKEND=alpaca`.
Cf. ADR-011 pour la justification.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

from src.execution.broker_protocol import (
    AccountState,
    BrokerError,
    OrderIntent,
    OrderResult,
    OrderStatus,
    Position,
    utcnow_iso,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class DryRunBroker:
    """Broker fictif : log JSONL + faux remplissage.

    Comportement :
    - place_order : ecrit une ligne JSONL, retourne OrderResult(SIMULATED, qty=qty).
    - get_position : retourne None (le broker fictif ne maintient pas de positions).
    - get_account : retourne 100k$ cash et buying_power constants pour les tests.
    - cancel_order : retourne True (cancel toujours OK en dry-run).

    Thread-safe : les ecritures JSONL sont protegees par un lock global.
    """

    name = "dry_run"
    _file_lock = threading.Lock()

    def __init__(
        self,
        log_path: str | None = None,
        cash: float = 100_000.0,
        buying_power: float = 100_000.0,
    ):
        """Initialise le broker dry-run.

        Parameters
        ----------
        log_path
            Fichier JSONL des ordres simules. Defaut : env DRY_RUN_LOG_PATH ou
            "logs/dry_run_trades.jsonl".
        cash, buying_power
            Etat de compte fictif retourne par get_account(). Utile pour
            tester les position sizers sans broker reel.
        """
        self.log_path = Path(log_path or os.getenv("DRY_RUN_LOG_PATH", "logs/dry_run_trades.jsonl"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._cash = cash
        self._buying_power = buying_power
        log.info("DryRunBroker init", extra={"log_path": str(self.log_path)})

    # ---------------------------------------------------------------------
    # BrokerProtocol API
    # ---------------------------------------------------------------------
    def place_order(self, intent: OrderIntent) -> OrderResult:
        """Simule un ordre : log JSONL + retourne SIMULATED."""
        if intent.qty <= 0:
            raise BrokerError(f"qty doit etre > 0, got {intent.qty}")

        order_id = f"dryrun_{uuid.uuid4().hex[:12]}"
        result = OrderResult(
            intent=intent,
            status=OrderStatus.SIMULATED,
            order_id=order_id,
            filled_qty=intent.qty,  # remplissage fictif total
            filled_avg_price=intent.target_price or 0.0,
            submitted_at=utcnow_iso(),
            broker_metadata={"dry_run": True},
        )
        self._append_jsonl(result)
        log.info(
            "dry_run_order_placed",
            extra={
                "ticker": intent.ticker,
                "side": intent.side.value,
                "qty": intent.qty,
                "order_id": order_id,
                "signal_source": intent.signal_source,
            },
        )
        return result

    def get_position(self, ticker: str) -> Position | None:
        """Le DryRunBroker ne tracke pas les positions (always None).

        Pour un suivi de portefeuille fictif persistant, c'est `portfolio_state.py`
        qui s'en charge en lisant les JSONL produits par ce broker.
        """
        return None

    def get_account(self) -> AccountState:
        """Retourne l'etat fictif du compte."""
        return AccountState(
            cash=self._cash,
            portfolio_value=self._cash,
            buying_power=self._buying_power,
            positions_count=0,
        )

    def cancel_order(self, order_id: str) -> bool:
        """Cancel toujours OK en dry-run (rien a annuler reellement)."""
        log.debug("dry_run_cancel", extra={"order_id": order_id})
        return True

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _append_jsonl(self, result: OrderResult) -> None:
        """Ecrit une ligne JSONL atomiquement (thread-safe)."""
        payload = {
            "ts": result.submitted_at,
            "order_id": result.order_id,
            "status": result.status.value,
            "ticker": result.intent.ticker,
            "side": result.intent.side.value,
            "qty": result.intent.qty,
            "target_price": result.intent.target_price,
            "filled_qty": result.filled_qty,
            "filled_avg_price": result.filled_avg_price,
            "signal_source": result.intent.signal_source,
            "confidence": result.intent.confidence,
        }
        with DryRunBroker._file_lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
