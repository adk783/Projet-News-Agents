"""Protocol commun pour les brokers (pattern Adapter).

Le `BrokerProtocol` definit le contrat minimal qu'un broker doit satisfaire
pour etre branche dans le pipeline. C'est un Protocol Python (PEP 544) :
duck-typing structurel, pas d'heritage requis.

Le pipeline emet des `OrderIntent` (intentions abstraites) :
    intent = OrderIntent(ticker="AAPL", side=OrderSide.BUY,
                         qty=10, target_price=180.50)
    result = broker.place_order(intent)

Le broker traduit cela en appel API concret (Alpaca, Interactive Brokers, etc.)
ou en log JSONL (DryRunBroker).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol, runtime_checkable


# =============================================================================
# Types & enums
# =============================================================================
class OrderSide(str, Enum):
    """Direction d'un ordre."""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Etat d'un ordre apres soumission."""

    PENDING = "PENDING"  # accepte par le broker, pas encore execute
    FILLED = "FILLED"  # execute integralement
    PARTIAL = "PARTIAL"  # execute partiellement
    REJECTED = "REJECTED"  # refuse par le broker (margin, halt, ...)
    CANCELED = "CANCELED"  # annule (par nous ou par le broker)
    SIMULATED = "SIMULATED"  # mode dry_run : pas reellement passe


@dataclass(frozen=True)
class OrderIntent:
    """Intention d'ordre emise par le pipeline (immutable)."""

    ticker: str
    side: OrderSide
    qty: int  # nombre d'actions (entier positif)
    target_price: float | None = None  # None = market order ; sinon = limit order
    # Metadata pour audit (ne sont PAS envoyes au broker, juste loggues).
    signal_source: str = "unknown"  # ex: "agent_pipeline.AAPL.20260425"
    confidence: float | None = None  # ex: 0.85 (consensus rate)


@dataclass
class OrderResult:
    """Resultat de la tentative de soumission."""

    intent: OrderIntent
    status: OrderStatus
    order_id: str = ""  # ID broker (vide si dry_run)
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    submitted_at: str = ""  # ISO 8601 UTC
    error_message: str = ""
    # Metadata broker-specific (transparent pour le pipeline).
    broker_metadata: dict = field(default_factory=dict)


@dataclass
class Position:
    """Position courante sur un ticker."""

    ticker: str
    qty: int  # negatif = short
    avg_entry_price: float
    market_value: float = 0.0
    unrealized_pl: float = 0.0


@dataclass
class AccountState:
    """Etat du compte (cash + valeur portefeuille)."""

    cash: float
    portfolio_value: float
    buying_power: float
    positions_count: int = 0


# =============================================================================
# Exceptions
# =============================================================================
class BrokerError(Exception):
    """Erreur generique d'un broker."""


# =============================================================================
# Protocol (interface)
# =============================================================================
@runtime_checkable
class BrokerProtocol(Protocol):
    """Contrat minimal qu'un broker doit satisfaire.

    Toutes les methodes peuvent lever `BrokerError` en cas d'echec irrecuperable.
    Les erreurs transitoires (timeout, 5xx) doivent etre retentees en interne.
    """

    name: str  # nom du broker pour logs (ex: "alpaca_paper", "dry_run")

    def place_order(self, intent: OrderIntent) -> OrderResult:
        """Soumet un ordre au broker. Synchrone."""
        ...

    def get_position(self, ticker: str) -> Position | None:
        """Retourne la position courante sur un ticker, ou None si flat."""
        ...

    def get_account(self) -> AccountState:
        """Retourne l'etat du compte (cash, valeur, buying power)."""
        ...

    def cancel_order(self, order_id: str) -> bool:
        """Annule un ordre PENDING. True si succes."""
        ...


# =============================================================================
# Factory : selection du backend selon BROKER_BACKEND
# =============================================================================
def get_broker(backend: str | None = None, **kwargs) -> BrokerProtocol:
    """Retourne une instance du broker selectionne.

    Selection :
    - `backend` argument explicite > `BROKER_BACKEND` env var > "dry_run" (defaut).
    - "dry_run"  -> DryRunBroker (JSONL append-only, pas d'appel reseau).
    - "alpaca"   -> AlpacaBroker (necessite ALPACA_API_KEY + ALPACA_SECRET_KEY).

    Parameters
    ----------
    backend : str, optional
        Nom du backend. Si None, lit `BROKER_BACKEND` depuis l'env.
    **kwargs
        Passes au constructeur du broker (api_key, log_path, ...).

    Returns
    -------
    BrokerProtocol
        Instance prete a l'usage.

    Raises
    ------
    BrokerError
        Si le backend demande est inconnu ou ses credentials manquent.
    """
    if backend is None:
        backend = os.getenv("BROKER_BACKEND", "dry_run").lower()

    if backend == "dry_run":
        # Import local pour eviter cycle d'import (dry_run_broker importe ce module).
        from src.execution.dry_run_broker import DryRunBroker

        return DryRunBroker(**kwargs)

    if backend == "alpaca":
        from src.execution.alpaca_broker import AlpacaBroker

        return AlpacaBroker(**kwargs)

    raise BrokerError(f"Backend broker inconnu : '{backend}'. Valeurs possibles : 'dry_run', 'alpaca'.")


def utcnow_iso() -> str:
    """Helper : timestamp ISO 8601 UTC (utilise par les brokers pour `submitted_at`)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
