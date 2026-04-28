"""
discovery — Universe selection autonome pour le pipeline news-agents.

Cette couche permet au systeme de choisir lui-meme les tickers a analyser
chaque jour, plutot que de prendre une liste fixee a la main. C'est ce qui
rend possible une "weekly run" en autopilote pour recolter un dataset.

Modules :
  - ticker_discovery : engine multi-signal qui scoore les tickers candidats
  - data_harvester   : persistance JSONL versionnee pour l'analyse a posteriori
  - orchestrator     : boucle quotidienne qui combine discovery + pipeline + harvest
"""

from .data_harvester import (  # noqa: F401
    DataHarvester,
    HarvestRecord,
)
from .orchestrator import (  # noqa: F401
    DailyHarvestOrchestrator,
    OrchestratorReport,
)
from .ticker_discovery import (  # noqa: F401
    DEFAULT_SIGNAL_WEIGHTS,
    DiscoveryReport,
    DiscoveryScore,
    DiscoverySource,
    TickerDiscoveryEngine,
    UniverseFilter,
)
