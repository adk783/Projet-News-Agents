"""AlpacaBroker : implementation BrokerProtocol pour Alpaca paper trading.

Utilise le SDK officiel `alpaca-py`. La librairie est une dep optionnelle
(import paresseux dans __init__) — installer via `pip install alpaca-py`
si vous voulez activer ce backend.

Documentation Alpaca :
- API : https://docs.alpaca.markets/reference
- SDK Python : https://alpaca.markets/sdks/python/

Configuration (env vars) :
- ALPACA_API_KEY    : cle API
- ALPACA_SECRET_KEY : cle secrete
- ALPACA_PAPER      : "1" (defaut) pour paper trading, "0" pour live (DANGER)

Mode paper trading recommande pour tous les premiers tests.
"""

from __future__ import annotations

import os
from typing import Any

from src.execution.broker_protocol import (
    AccountState,
    BrokerError,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    Position,
    utcnow_iso,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


# Mapping OrderStatus Alpaca -> notre enum.
# https://docs.alpaca.markets/reference/getallorders
_ALPACA_STATUS_MAP = {
    "new": OrderStatus.PENDING,
    "accepted": OrderStatus.PENDING,
    "pending_new": OrderStatus.PENDING,
    "accepted_for_bidding": OrderStatus.PENDING,
    "filled": OrderStatus.FILLED,
    "partially_filled": OrderStatus.PARTIAL,
    "done_for_day": OrderStatus.PARTIAL,
    "rejected": OrderStatus.REJECTED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.CANCELED,
    "stopped": OrderStatus.CANCELED,
    "suspended": OrderStatus.CANCELED,
}


class AlpacaBroker:
    """Broker Alpaca paper/live (selectionne via `paper_trading`).

    Construction defensive : si `alpaca-py` n'est pas installe, lever
    `BrokerError` avec un message d'install clair.
    """

    name = "alpaca"

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        paper_trading: bool | None = None,
    ):
        """Initialise le client Alpaca.

        Parameters
        ----------
        api_key, secret_key
            Credentials Alpaca. Si None, lus depuis ALPACA_API_KEY / ALPACA_SECRET_KEY.
        paper_trading
            True (defaut) -> sandbox paper trading. False -> compte reel (attention !).
            Si None, lit ALPACA_PAPER (defaut "1" -> True).

        Raises
        ------
        BrokerError
            Si alpaca-py n'est pas installe ou si les credentials manquent.
        """
        # Import paresseux : alpaca-py est dep optionnelle (~5Mo).
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as e:
            raise BrokerError(
                "alpaca-py n'est pas installe. Pour activer le broker Alpaca :\n    pip install alpaca-py"
            ) from e

        self._api_key = api_key or os.getenv("ALPACA_API_KEY", "").strip()
        self._secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY", "").strip()
        if not self._api_key or not self._secret_key:
            raise BrokerError("ALPACA_API_KEY et ALPACA_SECRET_KEY requis. Configurer dans .env ou passer en argument.")

        if paper_trading is None:
            paper_trading = os.getenv("ALPACA_PAPER", "1") == "1"
        self._paper = paper_trading

        self._client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=paper_trading,
        )
        log.info(
            "AlpacaBroker init",
            extra={"paper_trading": paper_trading, "api_key_prefix": self._api_key[:8]},
        )

    # ---------------------------------------------------------------------
    # BrokerProtocol API
    # ---------------------------------------------------------------------
    def place_order(self, intent: OrderIntent) -> OrderResult:
        """Soumet un ordre marche ou limite a Alpaca.

        Strategie :
        - target_price=None -> MarketOrderRequest
        - target_price=X    -> LimitOrderRequest a prix X
        """
        from alpaca.trading.enums import OrderSide as AlpacaOrderSide
        from alpaca.trading.enums import TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

        if intent.qty <= 0:
            raise BrokerError(f"qty doit etre > 0, got {intent.qty}")

        side = AlpacaOrderSide.BUY if intent.side == OrderSide.BUY else AlpacaOrderSide.SELL

        if intent.target_price is None:
            req: Any = MarketOrderRequest(
                symbol=intent.ticker,
                qty=intent.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        else:
            req = LimitOrderRequest(
                symbol=intent.ticker,
                qty=intent.qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                limit_price=intent.target_price,
            )

        try:
            alpaca_order = self._client.submit_order(req)
        except Exception as e:  # noqa: BLE001 - on capture large pour wrap dans BrokerError
            log.error(
                "alpaca_submit_failed",
                extra={
                    "ticker": intent.ticker,
                    "error": str(e)[:200],
                },
            )
            raise BrokerError(f"Alpaca submit_order error: {e}") from e

        status = _ALPACA_STATUS_MAP.get(
            getattr(alpaca_order, "status", "new").lower(),
            OrderStatus.PENDING,
        )
        result = OrderResult(
            intent=intent,
            status=status,
            order_id=str(alpaca_order.id),
            filled_qty=int(alpaca_order.filled_qty or 0),
            filled_avg_price=float(alpaca_order.filled_avg_price or 0.0),
            submitted_at=utcnow_iso(),
            broker_metadata={
                "alpaca_status": str(alpaca_order.status),
                "client_order_id": getattr(alpaca_order, "client_order_id", None),
            },
        )
        log.info(
            "alpaca_order_placed",
            extra={
                "ticker": intent.ticker,
                "side": intent.side.value,
                "qty": intent.qty,
                "order_id": result.order_id,
                "status": result.status.value,
            },
        )
        return result

    def get_position(self, ticker: str) -> Position | None:
        """Retourne la position courante sur ticker, ou None si flat."""
        try:
            pos = self._client.get_open_position(ticker)
        except Exception as e:  # noqa: BLE001 - 404 = flat, autres = vraies erreurs
            err_msg = str(e).lower()
            if "position does not exist" in err_msg or "404" in err_msg:
                return None
            raise BrokerError(f"Alpaca get_position error: {e}") from e

        return Position(
            ticker=ticker,
            qty=int(pos.qty),
            avg_entry_price=float(pos.avg_entry_price),
            market_value=float(pos.market_value or 0.0),
            unrealized_pl=float(pos.unrealized_pl or 0.0),
        )

    def get_account(self) -> AccountState:
        """Retourne l'etat du compte (cash + valeur portefeuille)."""
        try:
            account = self._client.get_account()
        except Exception as e:  # noqa: BLE001
            raise BrokerError(f"Alpaca get_account error: {e}") from e

        return AccountState(
            cash=float(account.cash),
            portfolio_value=float(account.portfolio_value),
            buying_power=float(account.buying_power),
            positions_count=0,  # Alpaca ne fournit pas direct ce compteur
        )

    def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre PENDING. True si succes."""
        try:
            self._client.cancel_order_by_id(order_id)
            log.info("alpaca_cancel_ok", extra={"order_id": order_id})
            return True
        except Exception as e:  # noqa: BLE001
            log.warning(
                "alpaca_cancel_failed",
                extra={
                    "order_id": order_id,
                    "error": str(e)[:200],
                },
            )
            return False
