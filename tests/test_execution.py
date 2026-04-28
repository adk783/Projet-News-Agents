"""Tests pour la couche d'execution (broker abstraction).

Couvre :
- Protocol BrokerProtocol (DryRunBroker et AlpacaBroker satisfont l'interface).
- DryRunBroker : log JSONL append-only, thread-safety, etat de compte fictif.
- AlpacaBroker : mock complet du SDK alpaca-py (place/get/cancel + parsing status).
- Factory get_broker() : selection via env BROKER_BACKEND.
- OrderIntent immutable (frozen dataclass).
- Validation : qty<=0 leve BrokerError.
"""

from __future__ import annotations

import json
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

from src.execution import (
    AccountState,
    BrokerError,
    BrokerProtocol,
    DryRunBroker,
    OrderIntent,
    OrderResult,
    OrderSide,
    OrderStatus,
    Position,
    get_broker,
)


# =============================================================================
# Tests OrderIntent / OrderResult dataclasses
# =============================================================================
class TestDataModels:
    """Verifie les invariants des modeles de donnees."""

    def test_order_intent_is_frozen(self):
        """OrderIntent doit etre immutable (frozen dataclass)."""
        intent = OrderIntent(ticker="AAPL", side=OrderSide.BUY, qty=10)
        with pytest.raises(Exception):  # FrozenInstanceError
            intent.qty = 100  # type: ignore[misc]

    def test_order_intent_default_target_price_none(self):
        """target_price=None par defaut -> market order."""
        intent = OrderIntent(ticker="AAPL", side=OrderSide.BUY, qty=10)
        assert intent.target_price is None

    def test_order_side_enum_values(self):
        """OrderSide doit avoir BUY et SELL."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"


# =============================================================================
# Tests DryRunBroker
# =============================================================================
class TestDryRunBroker:
    """Le DryRunBroker ecrit en JSONL append-only et n'appelle aucun reseau."""

    def test_satisfies_broker_protocol(self, tmp_path):
        """DryRunBroker doit etre conforme au Protocol (duck-typing)."""
        broker = DryRunBroker(log_path=str(tmp_path / "dry.jsonl"))
        assert isinstance(broker, BrokerProtocol)
        assert broker.name == "dry_run"

    def test_place_order_writes_jsonl_line(self, tmp_path):
        """place_order doit produire une ligne JSONL valide."""
        log_path = tmp_path / "trades.jsonl"
        broker = DryRunBroker(log_path=str(log_path))
        intent = OrderIntent(
            ticker="AAPL",
            side=OrderSide.BUY,
            qty=10,
            target_price=180.50,
            signal_source="test",
            confidence=0.85,
        )

        result = broker.place_order(intent)

        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1

        payload = json.loads(lines[0])
        assert payload["ticker"] == "AAPL"
        assert payload["side"] == "BUY"
        assert payload["qty"] == 10
        assert payload["target_price"] == 180.50
        assert payload["signal_source"] == "test"
        assert payload["confidence"] == 0.85
        assert payload["status"] == "SIMULATED"

        # Et le retour OrderResult est bien construit.
        assert result.status == OrderStatus.SIMULATED
        assert result.filled_qty == 10
        assert result.order_id.startswith("dryrun_")

    def test_place_order_appends_not_overwrites(self, tmp_path):
        """Plusieurs appels successifs -> append (pas d'ecrasement)."""
        log_path = tmp_path / "appended.jsonl"
        broker = DryRunBroker(log_path=str(log_path))
        for i in range(3):
            broker.place_order(
                OrderIntent(
                    ticker=f"TKR{i}",
                    side=OrderSide.BUY,
                    qty=1,
                )
            )
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3

    def test_place_order_qty_zero_raises(self, tmp_path):
        """qty <= 0 doit lever BrokerError."""
        broker = DryRunBroker(log_path=str(tmp_path / "x.jsonl"))
        with pytest.raises(BrokerError):
            broker.place_order(
                OrderIntent(
                    ticker="AAPL",
                    side=OrderSide.BUY,
                    qty=0,
                )
            )

    def test_place_order_qty_negative_raises(self, tmp_path):
        """qty negatif aussi rejete."""
        broker = DryRunBroker(log_path=str(tmp_path / "x.jsonl"))
        with pytest.raises(BrokerError):
            broker.place_order(
                OrderIntent(
                    ticker="AAPL",
                    side=OrderSide.SELL,
                    qty=-5,
                )
            )

    def test_get_position_always_returns_none(self, tmp_path):
        """DryRunBroker ne tracke pas les positions (always None)."""
        broker = DryRunBroker(log_path=str(tmp_path / "x.jsonl"))
        broker.place_order(OrderIntent(ticker="AAPL", side=OrderSide.BUY, qty=10))
        assert broker.get_position("AAPL") is None

    def test_get_account_returns_constant_state(self, tmp_path):
        """get_account() retourne l'etat fictif configure."""
        broker = DryRunBroker(
            log_path=str(tmp_path / "x.jsonl"),
            cash=50_000.0,
            buying_power=100_000.0,
        )
        acc = broker.get_account()
        assert acc.cash == 50_000.0
        assert acc.buying_power == 100_000.0

    def test_cancel_always_returns_true(self, tmp_path):
        """cancel_order() succes systematique en dry-run."""
        broker = DryRunBroker(log_path=str(tmp_path / "x.jsonl"))
        assert broker.cancel_order("any_id") is True

    def test_thread_safe_concurrent_writes(self, tmp_path):
        """4 threads x 10 ordres = 40 lignes JSONL atomiques (pas de corruption)."""
        log_path = tmp_path / "concurrent.jsonl"
        broker = DryRunBroker(log_path=str(log_path))

        def worker(thread_id: int):
            for i in range(10):
                broker.place_order(
                    OrderIntent(
                        ticker=f"T{thread_id}",
                        side=OrderSide.BUY,
                        qty=1,
                        signal_source=f"thread_{thread_id}",
                    )
                )

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Chaque ligne doit etre du JSON valide (pas de corruption).
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 40
        for line in lines:
            payload = json.loads(line)  # leve si corrompu
            assert "ticker" in payload

    def test_creates_parent_directory(self, tmp_path):
        """Le repertoire parent du log doit etre cree si absent."""
        log_path = tmp_path / "deep" / "nested" / "trades.jsonl"
        broker = DryRunBroker(log_path=str(log_path))
        broker.place_order(OrderIntent(ticker="X", side=OrderSide.BUY, qty=1))
        assert log_path.exists()


# =============================================================================
# Tests AlpacaBroker (mocks SDK)
# =============================================================================
class TestAlpacaBroker:
    """AlpacaBroker mocke le SDK alpaca-py."""

    def _setup_alpaca_mocks(self):
        """Cree les modules mock alpaca.* en mémoire pour permettre les imports."""
        # On utilise sys.modules pour injecter des fakes au moment ou
        # AlpacaBroker fait ses imports paresseux.
        alpaca = MagicMock()
        alpaca_trading = MagicMock()
        alpaca_trading_client = MagicMock()
        alpaca_trading_enums = MagicMock()
        alpaca_trading_requests = MagicMock()

        # Mocks pour OrderSide enum.
        class _FakeOrderSide:
            BUY = "BUY_SIDE"
            SELL = "SELL_SIDE"

        class _FakeTimeInForce:
            DAY = "DAY"

        alpaca_trading_enums.OrderSide = _FakeOrderSide
        alpaca_trading_enums.TimeInForce = _FakeTimeInForce

        # Mock des request classes (juste des constructeurs qui captent les kwargs).
        alpaca_trading_requests.MarketOrderRequest = MagicMock(side_effect=lambda **kw: kw)
        alpaca_trading_requests.LimitOrderRequest = MagicMock(side_effect=lambda **kw: kw)

        return {
            "alpaca": alpaca,
            "alpaca.trading": alpaca_trading,
            "alpaca.trading.client": alpaca_trading_client,
            "alpaca.trading.enums": alpaca_trading_enums,
            "alpaca.trading.requests": alpaca_trading_requests,
        }

    def test_missing_alpaca_lib_raises_clear_error(self):
        """Sans alpaca-py installe, on doit avoir un message d'install clair."""
        from src.execution.alpaca_broker import AlpacaBroker

        # On simule l'absence du module en mettant None dans sys.modules.
        with patch.dict(sys.modules, {"alpaca.trading.client": None}):
            with pytest.raises(BrokerError) as exc_info:
                AlpacaBroker(api_key="x", secret_key="y")
            assert "alpaca-py" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)

    def test_missing_credentials_raises(self):
        """Sans api_key/secret_key, lever BrokerError."""
        from src.execution.alpaca_broker import AlpacaBroker

        mocks = self._setup_alpaca_mocks()
        with patch.dict(sys.modules, mocks):
            with patch.dict("os.environ", {}, clear=True):
                with pytest.raises(BrokerError) as exc_info:
                    AlpacaBroker()
                assert "ALPACA_API_KEY" in str(exc_info.value)

    def test_place_market_order_calls_market_request(self):
        """target_price=None -> MarketOrderRequest (pas LimitOrderRequest)."""
        from src.execution.alpaca_broker import AlpacaBroker

        mocks = self._setup_alpaca_mocks()
        with patch.dict(sys.modules, mocks):
            broker = AlpacaBroker(api_key="x", secret_key="y", paper_trading=True)

            # Mock la reponse submit_order
            fake_order = MagicMock()
            fake_order.id = "alp_12345"
            fake_order.status = "accepted"
            fake_order.filled_qty = 0
            fake_order.filled_avg_price = 0.0
            fake_order.client_order_id = "client_xyz"
            broker._client.submit_order.return_value = fake_order

            intent = OrderIntent(ticker="AAPL", side=OrderSide.BUY, qty=10)
            result = broker.place_order(intent)

            # Verifier que MarketOrderRequest a ete utilise
            assert mocks["alpaca.trading.requests"].MarketOrderRequest.called
            assert not mocks["alpaca.trading.requests"].LimitOrderRequest.called
            assert result.status == OrderStatus.PENDING
            assert result.order_id == "alp_12345"

    def test_place_limit_order_calls_limit_request(self):
        """target_price=180.50 -> LimitOrderRequest avec ce prix."""
        from src.execution.alpaca_broker import AlpacaBroker

        mocks = self._setup_alpaca_mocks()
        with patch.dict(sys.modules, mocks):
            broker = AlpacaBroker(api_key="x", secret_key="y", paper_trading=True)

            fake_order = MagicMock(
                id="alp_lim_1",
                status="filled",
                filled_qty=10,
                filled_avg_price=180.45,
                client_order_id="x",
            )
            broker._client.submit_order.return_value = fake_order

            intent = OrderIntent(
                ticker="AAPL",
                side=OrderSide.BUY,
                qty=10,
                target_price=180.50,
            )
            result = broker.place_order(intent)

            assert mocks["alpaca.trading.requests"].LimitOrderRequest.called
            limit_kwargs = mocks["alpaca.trading.requests"].LimitOrderRequest.call_args.kwargs
            assert limit_kwargs["limit_price"] == 180.50
            assert result.status == OrderStatus.FILLED
            assert result.filled_avg_price == 180.45

    def test_qty_zero_raises_before_alpaca_call(self):
        """qty <= 0 leve avant tout appel Alpaca."""
        from src.execution.alpaca_broker import AlpacaBroker

        mocks = self._setup_alpaca_mocks()
        with patch.dict(sys.modules, mocks):
            broker = AlpacaBroker(api_key="x", secret_key="y")
            with pytest.raises(BrokerError):
                broker.place_order(
                    OrderIntent(
                        ticker="AAPL",
                        side=OrderSide.BUY,
                        qty=0,
                    )
                )
            # Et l'API Alpaca n'a pas ete appelee
            broker._client.submit_order.assert_not_called()

    def test_get_position_returns_none_when_404(self):
        """Si Alpaca renvoie 'position does not exist' -> retourne None."""
        from src.execution.alpaca_broker import AlpacaBroker

        mocks = self._setup_alpaca_mocks()
        with patch.dict(sys.modules, mocks):
            broker = AlpacaBroker(api_key="x", secret_key="y")
            broker._client.get_open_position.side_effect = RuntimeError("position does not exist for AAPL")
            assert broker.get_position("AAPL") is None

    def test_get_position_returns_position_when_found(self):
        """Position existante -> retourne Position avec les bonnes valeurs."""
        from src.execution.alpaca_broker import AlpacaBroker

        mocks = self._setup_alpaca_mocks()
        with patch.dict(sys.modules, mocks):
            broker = AlpacaBroker(api_key="x", secret_key="y")
            fake_pos = MagicMock(
                qty=10,
                avg_entry_price=180.0,
                market_value=1850.0,
                unrealized_pl=50.0,
            )
            broker._client.get_open_position.return_value = fake_pos

            pos = broker.get_position("AAPL")
            assert pos is not None
            assert pos.qty == 10
            assert pos.avg_entry_price == 180.0
            assert pos.market_value == 1850.0

    def test_cancel_order_failure_returns_false(self):
        """Si cancel echoue, on retourne False (pas de raise)."""
        from src.execution.alpaca_broker import AlpacaBroker

        mocks = self._setup_alpaca_mocks()
        with patch.dict(sys.modules, mocks):
            broker = AlpacaBroker(api_key="x", secret_key="y")
            broker._client.cancel_order_by_id.side_effect = RuntimeError("not found")
            assert broker.cancel_order("nope") is False


# =============================================================================
# Tests factory get_broker
# =============================================================================
class TestGetBroker:
    """get_broker() doit selectionner le bon backend."""

    def test_default_is_dry_run(self, tmp_path):
        """Sans BROKER_BACKEND, le defaut est dry_run."""
        with patch.dict("os.environ", {}, clear=True):
            broker = get_broker(log_path=str(tmp_path / "x.jsonl"))
            assert broker.name == "dry_run"

    def test_explicit_dry_run(self, tmp_path):
        """backend='dry_run' explicite."""
        broker = get_broker(backend="dry_run", log_path=str(tmp_path / "x.jsonl"))
        assert isinstance(broker, DryRunBroker)

    def test_unknown_backend_raises(self):
        """Backend inconnu -> BrokerError clair."""
        with pytest.raises(BrokerError) as exc_info:
            get_broker(backend="ibkr_v999")
        assert "ibkr_v999" in str(exc_info.value)
        assert "dry_run" in str(exc_info.value)
        assert "alpaca" in str(exc_info.value)

    def test_env_var_selects_backend(self, tmp_path):
        """BROKER_BACKEND=dry_run via env var."""
        with patch.dict(
            "os.environ",
            {
                "BROKER_BACKEND": "dry_run",
                "DRY_RUN_LOG_PATH": str(tmp_path / "env.jsonl"),
            },
        ):
            broker = get_broker()
            assert broker.name == "dry_run"
