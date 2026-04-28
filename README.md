# Locus — Systeme Expert Multi-Agents pour Trading Event-Driven

> Generateur de signaux de trading **long/short event-driven** a partir d'actualites
> financieres temps-reel, fonde sur un **debat multi-agent LLM** discipline par
> des modules statistiques bayesiens et causaux, et evalue par une couche
> d'audit scientifique de **17 metriques formelles**.

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](.github/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-375%20passing-brightgreen)](tests/)
[![License](https://img.shields.io/badge/license-Proprietary-red)](#)

---

## Pourquoi ce projet ?

L'analyse de news financieres par LLM avec un score "sentiment positif/negatif"
est aujourd'hui **trop simpliste pour fonder une strategie quantitative**.
Trois problemes structurels :

1. **Vocabulaire piege** : un *miss* trimestriel n'est pas un *beat*, et un LLM
   generaliste n'a pas la connaissance de domaine pour le distinguer correctement.
2. **Biais haussier systemique** : les LLMs RLHF sont entraines a etre positifs
   par defaut — un classique **bull bias cognitif**.
3. **Lecture multi-perspectives** : la meme nouvelle peut etre haussiere pour
   les uns et baissiere pour les autres (cf. annonce de buyback en plein
   ralentissement macro).

**Locus** repond a ces problemes par une architecture en 3 couches :

- Un **comite de 3 LLMs independants** (Cerebras / Groq / Mistral) qui debattent
  contradictoirement avant de produire un signal.
- Une **couche statistique bayesienne** qui quantifie la variance epistemique
  vs aleatorique de l'agregat (Lakshminarayanan 2017).
- Un **framework d'evaluation a 17 metriques formelles** (calibration, NLI
  faithfulness, counterfactual invariance, abnormal returns Fama-French,
  Romano-Wolf stepdown).

---

## Architecture

```
┌────────────────────── COLLECTE & PRE-TRAITEMENT ──────────────────────┐
│  RSS feeds  ──▶ sanitizer (anti-prompt-injection)                     │
│              ──▶ temporal_fence (anti-survivorship-bias)              │
│              ──▶ DistilRoBERTa filter (pertinence financiere)         │
│              ──▶ ABSA taxonomique (Earnings / M&A / Reg / Risk)       │
└────────────────────────┬──────────────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────────────┐
│                 DEBAT MULTI-AGENT (AutoGen, 3 tours)                  │
│                                                                       │
│   Agent Haussier (Cerebras Llama-3.3-70b)                             │
│        │  cherche obstinement l'opportunite de croissance             │
│        ▼                                                              │
│   Agent Baissier (Groq Llama-4-Scout-17b)                             │
│        │  joue l'avocat du diable, identifie les bulles macro         │
│        ▼                                                              │
│   Agent Neutre (Mistral-Small)                                        │
│        │  arbitre, evalue la "YOLO Confidence", produit le signal     │
└────────────────────────┬──────────────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────────────┐
│             COUCHE BAYESIENNE & DECISION                              │
│   Bayesian Aggregator (Beta-Binomial) ──▶ Var_epi vs Var_ale          │
│   YOLO Risk Classifier (auto-exec / warn / human-approval)            │
│   Position Sizer (Kelly net de frais, scale par variance)             │
│   Garde-fous : kill-switch VIX, cap sectoriel 30%, DRY_RUN            │
└────────────────────────┬──────────────────────────────────────────────┘
                         │
                         ▼
                  Signal final + sizing
                  (DB SQLite + JSONL audit trail)
```

A chaque etape, des **metriques formelles** (calibration, alpha ajuste FF3,
counterfactual invariance) sont emises plutot que les confiances LLM
auto-rapportees, qui sont notoirement non-calibrees.

---

## Quick Start

### Pre-requis

- **Python 3.10 ou 3.11**
- 3 cles API :
  - [Groq](https://console.groq.com) (agent Baissier)
  - [Cerebras](https://cloud.cerebras.ai) (agent Haussier)
  - [Mistral](https://console.mistral.ai) (agent Neutre)

### Installation

```bash
# Cloner et installer en mode editable
git clone <repo-url> locus && cd locus
pip install -e ".[dev]"  # inclut pytest, ruff, mypy

# Configurer les cles API (3 obligatoires + 13 optionnelles)
cp .env.example .env
# editer .env avec vos cles
```

### Configuration

Le projet repose sur 3 clés d'API obligatoires pour le système multi-agent, et plusieurs clés optionnelles pour l'exécution et l'audit.
Commencez par dupliquer le fichier d'exemple :
`cp .env.example .env`

**Clés obligatoires (Modèles fondateurs) :**
- **Groq** (`GROQ_API_KEY`) : Utilisé pour l'Agent Baissier. Inscription sur `console.groq.com`. **Gratuit** (quotas très généreux pour les tests).
- **Cerebras** (`CEREBRAS_API_KEY`) : Utilisé pour l'Agent Haussier. Inscription sur `cloud.cerebras.ai`. **Gratuit** (quotas généreux).
- **Mistral** (`MISTRAL_API_KEY`) : Utilisé pour l'Agent Neutre. Inscription sur `console.mistral.ai`. **Payant** mais abordable (prévoir un budget de $1 à $2 pour réaliser un cycle de test complet intensif).

**Clés optionnelles (Audit et Données) :**
- **NVIDIA NIM** (`NVIDIA_NIM_API_KEY`) : Modèle de fallback et d'audit avancé. Inscription sur `build.nvidia.com`. **Gratuit** (1000 crédits offerts).
- **Finnhub** (`FINNHUB_API_KEY`) & **FRED** (`FRED_API_KEY`) : Calendriers macro. **Gratuit**.
- **Alpaca** (`ALPACA_API_KEY`) : Pour le mode Paper-Trading en direct. **Gratuit** en mode sandbox.

### Verification

```bash
# Suite de tests (375 tests, < 2 min sans LLM)
pytest tests/ -v

# Lint + type check
ruff check src/ tests/ eval/
mypy src/discovery/ src/utils/logger.py src/utils/llm_client.py
```

---

## Les 3 modes d'execution

### Mode 1 — Live (un cycle a la demande)

```bash
# Pipeline complet : collecte news + debat + signal
./run.ps1 -Tickers AAPL MSFT GOOGL

# Ou en Python direct
python -m src.pipelines.agent_pipeline
```

Sortie : table `articles` dans `data/news_database.db` avec colonnes
`signal_final`, `confiance`, `montant_euros`, `risk_level`, etc.

### Mode 2 — Paper trading (boucle, dry-run)

```bash
# Loop 30min, signaux loggues mais zero mutation portefeuille
python run_paper_trading.py
```

A chaque cycle, `run_paper_trading.py` :

1. **Selectionne dynamiquement les tickers** via `TickerDiscoveryEngine`
   (volume anomaly + earnings calendar via yfinance, scoore l'univers de 50 leaders S&P 500).
   Si la discovery retourne 0 candidats, fallback sur 5 leaders. Override
   manuel possible via `LOCUS_TICKERS="AAPL TSLA"`.
2. **Ingestion news** (`src/pipelines/news_pipeline.py`) sur les tickers selectionnes.
3. **Si marche ouvert** (Lun-Ven 15h30-22h Paris) : analyse multi-agents (`src/pipelines/agent_pipeline.py`).
4. **Calibration nocturne** : entre 02:30 et 03:00, refit du calibrateur Platt+Isotonic
   (`scripts/fit_calibration_nightly.py`), idempotent (sentinel `data/.last_calibration_date`).
5. **Audit hebdomadaire** : le dimanche a 23h, `scripts/audit_hebdomadaire.py`
   (counterfactual + event study + regimes).

Variables d'environnement (toutes optionnelles) :

| Variable | Defaut | Role |
|---|---|---|
| `DRY_RUN` | 1 | Force le mode simulation (pose par le script) |
| `LLM_DAILY_BUDGET_USD` | 5.0 | Budget LLM journalier |
| `LOCUS_TICKERS` | (vide) | Override manuel : liste de tickers separes par espace |
| `LOCUS_TOP_N` | 10 | Nombre de tickers selectionnes par cycle |
| `LOCUS_INTERVAL_MIN` | 30 | Intervalle entre cycles |

Inspection :

```bash
cat logs/dry_run_trades.jsonl | jq .            # ordres hypothetiques
cat reports/llm_cost_daily/$(date +%F).json     # cout LLM du jour
```

### Mode 3 — Autopilot (ticker discovery + harvest hebdomadaire)

```bash
# Decouvre tickers a chaud (trending news, volume anomaly, social spike,
# earnings calendar, SEC filings, big movers) et harvest les signaux
python -c "
from src.discovery import DailyHarvestOrchestrator, TickerDiscoveryEngine, DataHarvester
from src.knowledge.cold_start import ColdStartManager
from src.pipelines.agent_pipeline import run_agent_pipeline

orch = DailyHarvestOrchestrator(
    discovery_engine=TickerDiscoveryEngine.default(),
    harvester=DataHarvester('data/harvest'),
    analysis_pipeline_fn=run_agent_pipeline,
    cold_start_manager=ColdStartManager(),
)
report = orch.run_daily_harvest(top_n=10)
print(report)
"
```

Idempotent : un re-run le meme jour skip les tickers deja traites.

---

## Dashboard de supervision

```bash
streamlit run dashboard/app.py
```

Tableau de bord Streamlit a 3 pages, lit en read-only la base SQLite, les
fichiers JSON d'etat et les sorties d'evaluation :

- **Recommandations** : signaux courants (Acheter / Vendre / Attendre) avec
  filtres (ticker, decision, regime, coherence min). Detail par article :
  cards meta (action, decision, coherence FinBERT/ABSA, amplitude attendue),
  argument dominant et transcription complete du debat 3 agents.
- **Performance** : metriques scientifiques pour decider si le systeme est
  digne de confiance — Sharpe median walk-forward, hit rate, stability score,
  distribution des Sharpes par fenetre, performance par regime de marche
  (BULL/BEAR/SIDEWAYS), Sharpe net par tier de courtier (Retail/SemiPro/Pro)
  avec break-even accuracy, calibration ECE. Etat live du portefeuille fictif
  (capital initial, P&L, drawdown, cash investissable, positions ouvertes,
  exposition sectorielle). Cadre strategique (capital, profil, regles MiFID II).
- **Surveillance** : sante technique (kill-switch VIX, budget IA quotidien,
  journal d'evenements).

Source de verite : `data/news_database.db`, `data/portfolio_state.json`,
`data/investor_profile.json`, `logs/dry_run_trades.jsonl`,
`reports/llm_cost_daily/`, `eval/eval_results/`. Le dashboard ne mute jamais
ces fichiers — il les affiche.

---

## Framework d'evaluation (Layer 7)

Le module `eval/run_eval.py` produit un **PDF analytique** avec 17 metriques :

| # | Metrique | Module | Reference |
|---|---|---|---|
| 1 | NLI Faithfulness (entailment) | `eval/evaluate_faithfulness.py` | Honovich 2022 |
| 2 | Brier Score | `eval/evaluate_calibration.py` | Brier 1950 |
| 3 | KL Divergence | `eval/evaluate_calibration.py` | Kullback-Leibler 1951 |
| 4 | ECE post-Platt | `src/utils/calibration.py` | Guo 2017 |
| 5 | Self-Consistency (vote majoritaire) | `eval/evaluate_consistency.py` | Wang et al. 2022 |
| 6 | Counterfactual Invariance (MSS 20-dim) | `src/utils/counterfactual.py`, `src/utils/minimal_sufficient_statistic.py` | Veitch 2021 |
| 7 | Abnormal Returns (CAR[-1,+5]) | `eval/evaluate_event_study.py` | MacKinlay 1997 |
| 8 | Newey-West HAC | `eval/evaluate_event_study.py` | Newey-West 1987 |
| 9 | Romano-Wolf q-value | `eval/evaluate_event_study.py` | Romano-Wolf 2005 |
| 10 | Stationary Bootstrap | `eval/evaluate_event_study.py` | Politis-Romano 1994 |
| 11 | Adversarial robustness (negation) | `eval/evaluate_adversarial.py` | Wallace 2019 |
| 12 | Adversarial robustness (truncation) | `eval/evaluate_adversarial.py` | — |
| 13 | Execution costs (Almgren-Chriss) | `eval/evaluate_execution_costs.py` | Almgren-Chriss 2001 |
| 14 | Alpha decay (latency) | `eval/evaluate_latency.py` | — |
| 15 | Market regime robustness | `eval/evaluate_market_regimes.py` | — |
| 16 | Sharpe / Sortino / Calmar | `eval/evaluate_historical_backtest.py` | — |
| 17 | Edge empirique par régime (CAR) | `scripts/evaluate_empirical_edge.py` | — |
| 18 | F1 / Accuracy financial relevance | `eval/evaluate_unit_llm.py` | — |

```bash
# Genere le rapport complet (~5 min, $0.50 de LLM)
python eval/run_eval.py --all

# Sortie : reports/eval_YYYY-MM-DD.json + reports/eval_YYYY-MM-DD.pdf
```

---

## Stack technique

- **Orchestration** : AutoGen (debat 3 tours), LangGraph (pipeline state machine)
- **LLMs** : Cerebras Llama-3.3-70b, Groq Llama-4-Scout-17b, Mistral-Small —
  3 fournisseurs distincts pour limiter la correlation des erreurs entre agents.
  **+ NVIDIA NIM** (4e jambe de fallback, optionnel) : 100+ modeles dont
  Llama 3.1 405B, DeepSeek-R1, Nemotron via build.nvidia.com (free tier)
- **NLP** : DistilRoBERTa (pertinence financiere binaire), ABSA custom
- **RAG** : ChromaDB + sentence-transformers (all-MiniLM-L6-v2)
- **Donnees** : yfinance (S&P 500 benchmark), SEC EDGAR, FRED (optionnel)
- **Stats** : NumPy pur (Platt, isotonic PAVA, Newey-West, bootstrap) — pas de
  scikit-learn, pour rester leger et auditable
- **Stockage** : SQLite (live) + JSONL append-only (audit trail, harvest)
- **Tests** : pytest + pytest-cov, 375 tests passants
- **Execution** : pattern Adapter avec `DryRunBroker` (default) + `AlpacaBroker` (paper trading)
- **Qualite** : ruff (lint), mypy (types stricts sur les modules recents),
  GitHub Actions (CI sur 3.10 + 3.11)

---

## Documentation

| Fichier | Public cible |
|---|---|
| [`README.md`](README.md) | Decouverte projet (vous etes ici) |
| [`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md) | Onboarding LLM/ingenieur (10 min) |
| [`ARCHITECTURE_DECISIONS.md`](ARCHITECTURE_DECISIONS.md) | Decisions techniques (ADRs) |
| [`CRITIQUE_AND_ROADMAP.md`](CRITIQUE_AND_ROADMAP.md) | Audit interne + dette technique |

---

## Roadmap

### Phase 5 (en cours)

- [ ] Migration des appels OpenAI direct dans agent_debat (AutoGen) vers LLMClient
- [x] Migration complete print -> logging dans le code historique (~252 print restants)
- [x] CI/CD automatisée avec GitHub Actions (pytest + ruff) pour garantir la qualité Python
- [ ] Dashboard temps-reel WebSocket (en complement du polling Streamlit existant)

### Phases livrees

- **Phase 4** (avr. 2026) — Production-ready : LLMClient avec fallback inter-provider,
  RotatingFileHandler, anonymizer dynamique (DB-loaded), block length Politis-White,
  walk-forward OOS strict (Bailey-Lopez 2014), broker Alpaca via Adapter pattern,
  pip install -e + nettoyage sys.path, 12 ADRs documentees, 375 tests.

- **Phase 3** (avr. 2026) — Garde-fous operationnels : DRY_RUN, kill-switch VIX,
  ADV dynamique, cap sectoriel, LLM cost tracker, refit calibration nocturne.
- **Phase 2** (avr. 2026) — Refactor pragmatique : separation live/offline,
  caches anti-redondance, config centralisee, P&L backtest restaure.
- **Phase 1** (avr. 2026) — Rigueur scientifique : Event Study FF3,
  Bayesian Aggregator, Counterfactual Invariance, calibration Platt+Iso,
  Almgren-Chriss execution costs.

Detail dans [`CRITIQUE_AND_ROADMAP.md`](CRITIQUE_AND_ROADMAP.md).

---

## Auteur

Projet concu en tant que **Proof-of-Concept architectural** explorant
l'ingenierie de prompt et l'architecture cognitive multi-agent dans un
contexte quantitatif. Il privilegie la rigueur d'evaluation (17 metriques
formelles, papiers academiques references) sur l'optimisation ingenierie
courte-vue.
