"""
portfolio_manager.py - Agent d'agrégation de portefeuille

Se réveille juste avant l'ouverture du marché (ou à la demande), récupère tous les 
signaux générés par les agents durant les dernières 24 heures pour chaque ticker,
et demande à un LLM "Executive" (Consensus) de prendre une décision finale d'investissement 
pour la journée afin d'éviter d'envoyer des ordres contradictoires au broker.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
import os
from typing import Dict, List

from src.utils.llm_client import LLMClient
from src.knowledge.macro_context import get_macro_context, format_macro_for_prompt
from src.config import DRY_RUN

logger = logging.getLogger(__name__)

PORTFOLIO_MANAGER_PROMPT = """You are the Chief Investment Officer (CIO) / Portfolio Manager.
Your analysts have tracked {ticker} overnight and produced the following trading signals based on different news articles:

{signals}

Your job is to read all these conflicting or reinforcing signals, weigh their respective market impact, and make a FINAL executive trading decision for today's market open.
Take into account the current macro-economic environment when making your decision:
=== MACRO CONTEXT ===
{macro_context}
=====================

If there is a strong "Vente" due to systemic/macro risks, it might override a minor "Achat" from an earnings beat, or vice-versa.

You MUST reply with a strict XML block:
<status agent="PortfolioManager">
  <decision>[Achat|Vente|Neutre]</decision>
  <reasoning>[Explain your final aggregated decision in 2-3 sentences max]</reasoning>
</status>
"""

def get_signals_last_24h(db_path: str = "data/news_database.db") -> Dict[str, List[dict]]:
    """Récupère tous les signaux finaux (Achat/Vente) des dernières 24h groupés par ticker."""
    yesterday = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute(
        """
        SELECT ticker, title, signal_final, argument_dominant, consensus_rate, impact_strength
        FROM articles
        WHERE signal_final IN ('Achat', 'Vente')
          AND date_utc >= ?
        ORDER BY ticker, date_utc DESC
        """,
        (yesterday,)
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    signals_by_ticker = {}
    for row in rows:
        t = row["ticker"]
        if t not in signals_by_ticker:
            signals_by_ticker[t] = []
        signals_by_ticker[t].append(dict(row))
        
    return signals_by_ticker


def run_portfolio_manager():
    """Exécute l'agent d'agrégation et logue le résultat final."""
    logger.info("Démarrage de l'agent Portfolio Manager (Agrégation des signaux de la nuit)...")
    
    signals_by_ticker = get_signals_last_24h()
    if not signals_by_ticker:
        logger.info("Aucun signal Achat/Vente généré dans les dernières 24h.")
        return
        
    llm = LLMClient.from_env()
    
    logger.info("Récupération de la météo macro-économique...")
    macro_snap = get_macro_context(force_refresh=False)
    macro_text = format_macro_for_prompt(macro_snap)
    
    for ticker, signals in signals_by_ticker.items():
        logger.info(f"Analyse des signaux pour {ticker} ({len(signals)} signaux en conflit/renfort)")
        
        # Formatage des signaux pour le prompt
        signals_text = ""
        for i, s in enumerate(signals, 1):
            signals_text += f"Signal {i} : {s['signal_final']} (Confiance: {s['consensus_rate']}, Impact: {s['impact_strength']})\n"
            signals_text += f"Titre : {s['title']}\n"
            signals_text += f"Argument : {s['argument_dominant']}\n\n"
            
        prompt = PORTFOLIO_MANAGER_PROMPT.format(ticker=ticker, signals=signals_text, macro_context=macro_text)
        
        try:
            # On utilise les modèles les plus intelligents (70B+) pour cette tâche d'arbitrage
            response, provider, model = llm.complete_raw(
                messages=[{"role": "user", "content": prompt}],
                model_preference=["consensus", "nim_llama_3_1_70b", "mistral_large", "groq"],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content
            
            # Extraction basique du XML
            decision = "Neutre"
            reasoning = ""
            
            if "<decision>" in content and "</decision>" in content:
                decision = content.split("<decision>")[1].split("</decision>")[0].strip()
            if "<reasoning>" in content and "</reasoning>" in content:
                reasoning = content.split("<reasoning>")[1].split("</reasoning>")[0].strip()
                
            logger.info(f"[{ticker}] Décision Finale PM : {decision} | Raison : {reasoning}")
            
            # Simulation d'exécution
            if DRY_RUN and decision in ("Achat", "Vente"):
                from src.utils.dry_run_logger import log_dry_run_order
                log_dry_run_order(
                    ticker=ticker,
                    signal=decision,
                    prix=0.0, # À l'ouverture, on ne connait pas le prix exact
                    montant=1000, # Montant fixe pour POC
                    risque_yolo="MOYEN",
                    market_regime="PRE_OPEN",
                    vix=0.0,
                    yield_curve_spread=0.0,
                    notes=f"Agrégation PM de {len(signals)} signaux. Raison: {reasoning}",
                    extras={"provider_pm": provider, "model_pm": model}
                )
                logger.info(f"Ordre {decision} {ticker} placé en file d'attente (Dry Run).")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'agrégation pour {ticker} : {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_portfolio_manager()
