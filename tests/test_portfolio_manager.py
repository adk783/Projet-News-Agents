import pytest
from unittest.mock import patch, MagicMock
from src.execution.portfolio_manager import run_portfolio_manager

@patch("src.execution.portfolio_manager.logger")
@patch("src.execution.portfolio_manager.get_signals_last_24h")
@patch("src.execution.portfolio_manager.get_macro_context")
def test_portfolio_manager_no_signals(mock_get_macro, mock_get_signals, mock_logger):
    """Test que l'agent quitte silencieusement s'il n'y a aucun signal."""
    mock_get_signals.return_value = {}
    
    # Mock du contexte macro pour éviter qu'il ne crash avant
    mock_macro = MagicMock()
    mock_get_macro.return_value = mock_macro
    
    run_portfolio_manager()
    
    mock_logger.info.assert_any_call("Aucun signal Achat/Vente généré dans les dernières 24h.")

@patch("src.execution.portfolio_manager.logger")
@patch("src.execution.portfolio_manager.get_signals_last_24h")
@patch("src.execution.portfolio_manager.format_macro_for_prompt")
@patch("src.execution.portfolio_manager.get_macro_context")
@patch("src.execution.portfolio_manager.LLMClient")
def test_portfolio_manager_aggregation(mock_llm_client, mock_get_macro, mock_format_macro, mock_get_signals, mock_logger):
    """Test que l'agent agrège correctement les signaux avec l'API LLM."""
    
    # Mock des signaux en base de données
    mock_get_signals.return_value = {
        "AAPL": [
            {"signal_final": "Achat", "consensus_rate": 0.9, "impact_strength": 0.8, "title": "Good News", "argument_dominant": "Earnings beat"},
            {"signal_final": "Vente", "consensus_rate": 0.7, "impact_strength": 0.5, "title": "Bad News", "argument_dominant": "CEO left"}
        ]
    }
    
    # Mock du contexte macro
    mock_get_macro.return_value = MagicMock()
    mock_format_macro.return_value = "VIX: 15.0 | SPY: +2.5%"
    
    # Mock du LLM
    mock_llm_instance = MagicMock()
    mock_llm_client.from_env.return_value = mock_llm_instance
    
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "<status><decision>Achat</decision><reasoning>Earnings are more important than CEO</reasoning></status>"
    mock_llm_instance.complete_raw.return_value = (mock_response, "groq", "llama-70b")
    
    run_portfolio_manager()
    
    # Vérification que le LLM a bien été appelé
    mock_llm_instance.complete_raw.assert_called_once()
    
    # Vérification des logs (Décision finale correctement parsée)
    mock_logger.info.assert_any_call("[AAPL] Décision Finale PM : Achat | Raison : Earnings are more important than CEO")
