"""execution -- Couche d'execution des ordres (broker abstraction).

Cette couche implemente le pattern Adapter (Gamma 1994) pour decoupler le
pipeline de signaux du fournisseur de courtage reel. Le pipeline emet des
intentions (`OrderIntent`) ; un broker concret les traduit en appels API.

Brokers disponibles :
- DryRunBroker : aucun ordre reel emis, journalisation JSONL append-only.
                 C'est le defaut (BROKER_BACKEND=dry_run ou non set).
- AlpacaBroker  : passe des ordres sur Alpaca paper trading API (sandbox).
                  Active via BROKER_BACKEND=alpaca + ALPACA_API_KEY+SECRET.

Selection du backend via :
    from src.execution import get_broker
    broker = get_broker()  # lit BROKER_BACKEND depuis env
    result = broker.place_order(intent)

Cf. ADR-011 (a creer) pour la justification de l'abstraction broker.
"""

from src.execution.broker_protocol import (  # noqa: F401
    AccountState,
    BrokerError,
    BrokerProtocol,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    Position,
    get_broker,
)
from src.execution.dry_run_broker import DryRunBroker  # noqa: F401

# AlpacaBroker import paresseux (alpaca-py est dep optionnelle).
__all__ = [
    "AccountState",
    "BrokerError",
    "BrokerProtocol",
    "DryRunBroker",
    "OrderIntent",
    "OrderResult",
    "OrderSide",
    "OrderStatus",
    "Position",
    "get_broker",
]
