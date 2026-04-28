"""
src/strategy — Couche d'ancrage stratégique réel

Modules :
  investor_profile  : Profil investisseur (MiFID II / Arrow-Pratt)
  portfolio_state   : État du portefeuille en temps réel (Markowitz)
  position_sizer    : Sizing Kelly/Half-Kelly (Thorp 1956, 2008)
  strategy_context  : Injection de contexte dans les prompts du débat
"""

from .investor_profile import (
    LOSS_AVERSION_LAMBDA,
    RISK_YOLO_THRESHOLDS,
    InvestmentHorizon,
    InvestmentObjective,
    InvestorProfile,
    RiskTolerance,
    load_investor_profile,
    save_investor_profile,
)
from .portfolio_constraints import (
    ConstraintResult,
    CorrelationResult,
    check_pairwise_correlation,
    check_sector_concentration,
)
from .portfolio_state import (
    PortfolioState,
    Position,
    load_portfolio_state,
    refresh_portfolio_prices,
    save_portfolio_state,
)
from .position_sizer import (
    PositionSizeResult,
    calculate_position_size,
)
from .strategy_context import build_strategy_context

__all__ = [
    "InvestorProfile",
    "RiskTolerance",
    "InvestmentHorizon",
    "InvestmentObjective",
    "load_investor_profile",
    "save_investor_profile",
    "LOSS_AVERSION_LAMBDA",
    "RISK_YOLO_THRESHOLDS",
    "Position",
    "PortfolioState",
    "load_portfolio_state",
    "save_portfolio_state",
    "refresh_portfolio_prices",
    "PositionSizeResult",
    "calculate_position_size",
    "build_strategy_context",
    "ConstraintResult",
    "CorrelationResult",
    "check_sector_concentration",
    "check_pairwise_correlation",
]
