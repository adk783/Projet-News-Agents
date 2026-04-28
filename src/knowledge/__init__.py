"""
src/knowledge — Couche de Connaissance Enrichie

Modules :
  rag_store     : Base vectorielle ChromaDB + décroissance temporelle (Kanhabua & Nørvåg 2010)
  sec_edgar     : API EDGAR publique — fact-checking réel 8-K + Form 4 insiders
  fundamentals  : Données fondamentales enrichies via yfinance
  macro_context : VIX, Yield Curve, Fed Rate, DXY (FRED + yfinance)
"""

from .fundamentals import (
    FundamentalsData,
    format_fundamentals_for_prompt,
    get_fundamentals,
)
from .macro_context import (
    MacroSnapshot,
    format_macro_for_prompt,
    get_macro_context,
)
from .rag_store import (
    DOC_TYPE_ARTICLE,
    DOC_TYPE_DEBATE,
    DOC_TYPE_MEMORY,
    DOC_TYPE_SEC_8K,
    DOC_TYPE_SEC_FORM4,
    LocusRAGStore,
    RAGDocument,
    get_rag_store,
)
from .sec_edgar import (
    EdgarFilingResult,
    InsiderActivity,
    SecEdgarClient,
    get_edgar_client,
)

__all__ = [
    "LocusRAGStore",
    "get_rag_store",
    "RAGDocument",
    "SecEdgarClient",
    "EdgarFilingResult",
    "InsiderActivity",
    "get_edgar_client",
    "FundamentalsData",
    "get_fundamentals",
    "format_fundamentals_for_prompt",
    "MacroSnapshot",
    "get_macro_context",
    "format_macro_for_prompt",
]
