import logging
import os
import sys

# Configurer le logging pour voir les résultats
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ajouter la racine du projet au PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.knowledge import (
    DOC_TYPE_ARTICLE,
    RAGDocument,
    format_fundamentals_for_prompt,
    format_macro_for_prompt,
    get_edgar_client,
    get_fundamentals,
    get_macro_context,
    get_rag_store,
)


def test_knowledge_layer():
    print("\n" + "=" * 50)
    print("[TEST] DE LA COUCHE DE CONNAISSANCE (KNOWLEDGE LAYER)")
    print("=" * 50)

    ticker = "AAPL"

    # ---------------------------------------------------------
    print("\n[1] TEST MACRO-ECONOMIE")
    # ---------------------------------------------------------
    try:
        macro = get_macro_context()
        print(format_macro_for_prompt(macro))
    except Exception as e:
        print(f"[ERREUR] Erreur Macro: {e}")

    # ---------------------------------------------------------
    print("\n[2] TEST FONDAMENTAUX (yfinance)")
    # ---------------------------------------------------------
    try:
        fund = get_fundamentals(ticker)
        print(format_fundamentals_for_prompt(fund))
    except Exception as e:
        print(f"[ERREUR] Erreur Fondamentaux: {e}")

    # ---------------------------------------------------------
    print("\n[3] TEST SEC EDGAR (Fact-Checking & Insiders)")
    # ---------------------------------------------------------
    try:
        edgar = get_edgar_client()
        # Test CIK lookup
        cik = edgar.get_cik(ticker)
        print(f"[OK] CIK pour {ticker} : {cik}")

        # Test Insider
        print("Recherche des Form 4 (Insiders)...")
        insiders = edgar.get_insider_activity(ticker, days_back=30)
        print(f"Signal Insider: {insiders.signal} | Note: {insiders.note}")

        # Test Fact Check factice
        print("Simulation Fact-Check (8-K)...")
        verif, raison, pen = edgar.fact_check_article(
            ticker, "Apple announces new CFO and restates earnings", "http://example.com/news", days_back=7
        )
        print(f"Verifié: {verif} | Raison: {raison} | Pénalité: {pen}x")
    except Exception as e:
        print(f"[ERREUR] Erreur EDGAR: {e}")

    # ---------------------------------------------------------
    print("\n[4] TEST RAG (ChromaDB Vector Store)")
    # ---------------------------------------------------------
    try:
        rag = get_rag_store()
        # Insertion d'un faux document
        rag.add_document(
            RAGDocument(
                doc_id="test_doc_1",
                ticker=ticker,
                text="Apple a publié de très bons résultats la semaine dernière, dépassant les attentes sur l'iPhone.",
                doc_type=DOC_TYPE_ARTICLE,
                date_iso="2026-04-10T10:00:00Z",  # ~11 jours avant
            )
        )

        # Recherche
        print("Recherche sémantique dans le RAG...")
        results = rag.query(ticker, "Quels sont les résultats d'Apple sur l'iPhone ?", k=1)
        print(rag.format_for_prompt(results))
    except Exception as e:
        print(f"[ERREUR] Erreur RAG: {e}")

    print("\n" + "=" * 50)
    print("[OK] TEST TERMINE")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    test_knowledge_layer()
