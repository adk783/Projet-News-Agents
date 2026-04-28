# Projet News-Agents — Guide d'onboarding pour un LLM

> Ce document est conçu pour qu'un LLM (ou un ingénieur) arrive dans ce repo
> et puisse se repérer en 10 minutes. Il liste : ce que fait le système,
> quelle partie vit où, pourquoi chaque décision a été prise, et quelles
> métriques quantifient chaque couche.

---

## 1. Ce que fait le projet

**Objectif** : générer des signaux de trading **event-driven long/short** à partir
d'articles financiers en s'appuyant sur un **débat multi-agent LLM** discipliné
par des modules statistiques bayésiens et causaux.

Pipeline de bout-en-bout :

```
news brute
  └─▶ sanitizer (anti-prompt-injection, obfuscation B64/entropie)
       └─▶ temporal_fence (anti-survivorship-bias)
            └─▶ anonymizer (optionnel, mode eval)
                 └─▶ DistilRoBERTa filter (pertinence financière)
                      └─▶ ABSA taxonomique (aspects + sentiment)
                           └─▶ Débat AutoGen : Haussier × Baissier × Neutre (3 tours)
                                └─▶ Consensus LLM (juge avec Reasoning Tree)
                                     └─▶ Bayesian Aggregator (Module ②)
                                          └─▶ YOLO Risk Classifier + filtre régime
                                               └─▶ Position Sizer (Kelly net de frais)
```

À chaque étape, des **métriques scientifiques** sont émises (calibration,
invariance contrefactuelle, alpha ajusté au facteur FF3) plutôt que des
confidences LLM auto-rapportées.

---

## 2. Architecture — cartographie des fichiers

### 2.1 Pipelines & Agents

| Fichier | Rôle | Notes critiques |
|---|---|---|
| `src/pipelines/agent_pipeline.py` | Orchestrateur principal (séquentiel, SQLite) | Wiring Bayes + Anonymizer + Temporal Fence (cf §3.4) |
| `src/pipelines/news_pipeline.py`  | Collecte RSS + stockage DB | En amont du pipeline agent |
| `src/agents/agent_filtrage.py`    | DistilRoBERTa binary filter | Financial relevance score |
| `src/agents/agent_absa.py`        | Aspect-Based Sentiment Analysis | Taxonomie d'aspects financiers |
| `src/agents/agent_debat.py`       | Débat multi-agent 3 tours + Consensus | **Fix** : plus de duplicate ÉTAPE 4 |
| `src/agents/agent_memoire.py`     | Mémoire inter-session (consolidation nightly) | |

LLMs utilisés dans le débat (configurés dans `agent_debat.py` via ADR-017) :
- **Haussier** : NIM Nemotron-Mini-4B (NVIDIA)
- **Baissier** : NIM Ministral-14B (Mistral)
- **Neutre**   : NIM Qwen3-Next-80B (Alibaba)
- **Consensus**: Llama 3.3 70B (Groq) — juge

### 2.2 Couche Statistique & Décisionnelle (NOUVEAU)

| Module | Fichier | Papier de référence |
|---|---|---|
| **Bayesian Aggregator** | `src/utils/bayesian_aggregator.py` | Raftery 2005, Lakshminarayanan 2017 |
| **Minimal Sufficient Statistic** | `src/utils/minimal_sufficient_statistic.py` | Fisher 1922, Pearl 2009 |
| **Counterfactual Invariance** | `src/utils/counterfactual.py` | Veitch et al. 2021 |
| **Calibration (Platt + Iso + ECE)** | `src/utils/calibration.py` | Platt 1999, Zadrozny 2002, Guo 2017 |
| **Execution Costs (Almgren-Chriss)** | `src/utils/execution_costs.py` | Almgren-Chriss 2001, Kissell 2014 |
| **Event Study (F-F 3 factor)** | `eval/evaluate_event_study.py` | MacKinlay 1997, Romano-Wolf 2005 |

### 2.3 Sécurité & Garde-fous

| Module | Fichier | But |
|---|---|---|
| Prompt-Injection | `src/utils/security_sanitizer.py` | Patterns + Base64-decode + entropie Shannon |
| Anonymizer | `src/utils/anonymizer.py` | 17 tickers (6 Big Tech + 11 delisted/stressed) |
| Temporal Fence | `src/utils/temporal_fence.py` | Détecte articles post-faillite (`LEH`, `SIVB`, `FRC`, `ENRNQ`...) |
| YOLO Classifier | `src/utils/yolo_classifier.py` | Gate auto-exec / warning / human-approval |
| HTC Calibrator | `src/utils/htc_calibrator.py` | Trajectoire 3D de confiance du débat |

### 2.4 Strategy Layer

| Fichier | Rôle |
|---|---|
| `src/strategy/position_sizer.py` | Kelly-half net de frais, scalé par variance Bayésienne ; guard `HOLD_SYSTEMIC`/`HOLD_SECTOR_CAP` |
| `src/strategy/portfolio_state.py` | État portefeuille, drawdown, concentration sectorielle |
| `src/strategy/investor_profile.py` | Profil Arrow-Pratt (risk aversion, horizon) |
| `src/strategy/portfolio_constraints.py` **(Phase 3)** | Cap sectoriel 30 % (DeMiguel et al. 2009), refus → `HOLD_SECTOR_CAP` |

### 2.5 Knowledge Layer

| Fichier | Rôle |
|---|---|
| `src/knowledge/edgar_client.py` | Fact-checking SEC EDGAR |
| `src/knowledge/macro_context.py` | VIX / SPY / taux Fed, mis en cache par session |
| `src/knowledge/rag_store.py` | Retrieval temporellement pondéré (décroissance log) |
| `src/knowledge/fundamentals.py` | Ratios yfinance |
| `src/knowledge/liquidity.py` **(Phase 3)** | ADV + σ dynamiques yfinance (TTL 24 h), branchés dans le position sizer |

### 2.6 Garde-fous opérationnels (Phase 3)

| Fichier | Rôle |
|---|---|
| `src/utils/dry_run_logger.py` | Paper trading (`DRY_RUN=1`) → JSONL append-only des ordres hypothétiques, zéro mutation portfolio |
| `src/utils/llm_cost_tracker.py` | Compteur singleton (token × prix), `BudgetExceededError` si `LLM_DAILY_BUDGET_USD` dépassé, snapshot quotidien JSON |
| `scripts/fit_calibration_nightly.py` | Refit Platt + Isotonic sur historique trades (outcome via yfinance T+5), `models/calibrator.pkl` + metadata JSON |

---

## 3. Améliorations récentes (audit Quant Tier-1 d'avril 2026)

### 3.1 Module ① — Event Study Fama-French 3 factor

**Fichier** : `eval/evaluate_event_study.py`
**Problème résolu** : `evaluate_historical_backtest.py` était un Monte-Carlo
tautologique (`np.random.rand() < 0.55`). Il simulait l'edge au lieu de le tester.

**Implémentation** :
- Estimation F-F 3-factor (Mkt-Rf, SMB, HML) sur fenêtre [−240, −20]
  → Proxies ETF (SPY/IWM/IWB/IWD/IWF/BIL) ou Ken French data library
- Calcul du **CAR[−1, +5]** (Cumulative Abnormal Return, Brown & Warner 1985)
- **Newey-West HAC** pour l'écart-type robuste à l'auto-corrélation
- **Stationary bootstrap** (Politis-Romano 1994), 10 000 rééchantillonnages
- **Romano-Wolf stepdown** pour la correction de multiple testing
- CLI : `python -m eval.evaluate_event_study --from-db --limit 100`

**Métrique** : t-stat Newey-West + p-value bootstrap + q-value Romano-Wolf.
**Verdict** : si q < 0.05 après Romano-Wolf, l'alpha est statistiquement
significatif à 95 % en presence-of-multiple-testing.

### 3.2 Module ② — Bayesian Aggregator

**Fichier** : `src/utils/bayesian_aggregator.py`
**Problème résolu** : les `[confiance: 0.85]` auto-rapportées par les LLMs
ne sont pas calibrées et étaient moyennées heuristiquement par un simple
`consensus_rate`. La variance entre agents n'était pas exploitée.

**Implémentation** :
- Chaque agent → `AgentPosterior(alpha, beta)` via mapping `thesis × confidence`
- Agrégation **Beta-Binomial hiérarchique** : α_agg = α_prior + Σ α_k
- Décomposition variance : `Var[p] = Var_epi (désaccord) + Var_ale (confiance intra)`
  — Lakshminarayanan et al. 2017 — Deep Ensembles
- CI 95 % du posterior via Beasley-Springer-Moro inverse-normale
- `kelly_scale = 2|Δ| / (1 + κ·Var_total)` — Medo-Pignatti 2013

**Sortie branchée sur** : `position_sizer.calculate_position_size(bayesian_consensus=...)`
qui applique `f_applied *= kelly_scale`.

**Métrique** : CI95 width du posterior, séparation Var_epi vs Var_ale.

### 3.3 Module ③ — Counterfactual Invariance + Minimal Sufficient Statistic

**Fichiers** :
- `src/utils/minimal_sufficient_statistic.py`
- `src/utils/counterfactual.py`
- ~~`eval/evaluate_counterfactual_invariance.py`~~ (supprime lors du nettoyage Phase 4 — voir ADR-010 ; les modules `utils` restent disponibles pour usage offline ad-hoc)

**Problème résolu** : look-ahead bias implicite dans les poids du LLM. Le
modèle "sait" que AAPL en 2024 est une valeur sûre — il triche via son prior.

**Implémentation MSS** :
Vecteur 20-dim déterministe T(x) = { event_type_id, polarity, magnitude
bucket, flags booléens (earnings/MNA/regulatory/...), n_entities, ambiguity,
spread, factual_density }. Sans nom de marque, CEO, date.

**Implémentation CI** :
Pour chaque article, 11 counterfactuals sont générés :
- `ticker_swap` (×6, swap vers AAPL/MSFT/TSLA/AMZN/GOOGL/NVDA/META)
- `date_shift_-1y`
- `magnitude_eps5` (bruit ±5 % relatif sur les pourcentages)
- `sector_swap`
- `ceo_removed`
- `style_neutral` (retrait d'adjectifs hyperboliques)

Score CI = 0.5 · MSS-preservation + 0.5 · signal-invariance.

**Métrique cibles** (Veitch 2021) :
- CI ≥ 0.90 → robuste, décision basée sur le contenu causal
- 0.75–0.89 → biais marginaux
- < 0.75   → le LLM triche via ses priors

**Smoke test** : sur un article AAPL réaliste, le pipeline atteint **CI = 0.955
(MSS-pres = 0.909, 11 perturbations)** en mode offline (sans appels LLM).

### 3.4 Fixes & durcissements secondaires

| # | Fichier | Changement |
|---|---|---|
| F1 | `src/agents/agent_debat.py` | Suppression du duplicate `=== ÉTAPE 4 ===` (→ `ÉTAPE 5`) |
| F2 | `src/strategy/position_sizer.py` | Clamp p ∈ [0.15, 0.85] (au lieu de [0.45, 0.75]) + Kelly net de frais + pénalité `1/(1+κ·Var)` |
| F3 | `src/utils/yolo_classifier.py` | SIDEWAYS défini en **vol-adjusted** (0.5·σ) au lieu de seuil fixe ±3 % |
| F4 | `src/utils/anonymizer.py` | +11 tickers delisted/stressed (LEH, BSC, SIVB, FRC, WCAGY, BYND, PTON, RIVN, ENRNQ, GME, META, JPM) |
| F5 | `src/pipelines/agent_pipeline.py` | Wiring `temporal_fence` (step 0.5) + `anonymizer` (step 0.7, via `EVAL_ANONYMIZE=1`) + `bayesian_aggregator` après débat |
| F6 | `src/pipelines/agent_pipeline.py` | Calcul `spy_20d_vol` en fallback yfinance pour régime vol-adjusted |
| F7 | `src/utils/execution_costs.py` (NOUVEAU) | Mode **cheap** par défaut (commissions IB + spread) + Almgren-Chriss **opt-in** |
| F8 | `src/pipelines/agent_pipeline.py` | Log mode `"w"` → `"a"` (fin de l'écrasement systématique de l'audit trail) |
| F9 | `src/agents/agent_absa.py` | Cache LRU 256 entrées sur `hash(content[:500])` (évite appels LLM redondants) |
| F10 | `src/knowledge/macro_context.py` | Cache TTL 15 min sur `get_macro_context()` (évite 6 appels yfinance/article) |
| F11 | `src/config.py` (NOUVEAU) | Constantes cross-module centralisées + override par env vars |

---

### 3.5 Phase 3 — Garde-fous opérationnels (avril 2026)

Cette phase clôt les risques systémiques identifiés par l'audit Phase 2
(cf. `CRITIQUE_AND_ROADMAP.md` §1.3-1.4) : **kill-switch**, **DRY_RUN**,
**monitoring coûts**, **cap sectoriel**, **ADV dynamique**, **refit
calibration**. C'est ce qui sépare un démonstrateur académique d'un prototype
que l'on peut brancher en paper trading sans crainte de bug coûteux.

| # | Feature | Déclencheur | Comportement |
|---|---|---|---|
| 1 | **DRY_RUN** | env `DRY_RUN=1` | Le pipeline déroule tout (débat, sizing, YOLO) mais n'écrit ni sur `portfolio_state` ni sur les colonnes de trading. Chaque ordre hypothétique est loggué en JSONL (`logs/dry_run_trades.jsonl`) avec sizing, p_win, VIX, yield spread, régime. Colonne DB `articles.dry_run=1` pour audit. |
| 2 | **VIX kill-switch** | `macro_snap.vix ≥ VIX_KILL_SWITCH_THRESHOLD` (défaut 45.0) | Banner WARN au démarrage si franchi ; tous les signaux forcés à `HOLD_SYSTEMIC` ; `position_sizer` retourne `nb_actions=0` / `action_type=TENIR`. Colonne DB `articles.kill_switch_active=1`. Whaley (2009) ; benchmarks COVID 2020 (VIX 82), Lehman 2008 (VIX 80). |
| 3 | **ADV dynamique** | systématique, cache 24 h | `get_liquidity_profile(ticker)` → `LiquidityProfile(adv_volume, sigma_daily, source, fetched_at)`. yfinance `info['averageVolume']` + σ sur 3mo log-returns. Clamps : ADV ∈ [10k, 5B], σ ∈ [0.001, 0.15]. Remplace l'ancien default `5M / 0.018`. |
| 4 | **Cap sectoriel** | signal ∈ (Achat, Vente) ET montant > 0 | `check_sector_concentration(portfolio, secteur, montant)` refuse si `exposition_projetée > MAX_SECTOR_EXPOSURE_PCT` (défaut 30 %). Refus → `signal_final="HOLD_SECTOR_CAP"`, `montant_euros=0`, `action_type="REJECT_SECTOR"`. DeMiguel-Garlappi-Uppal (2009). |
| 5 | **LLM cost tracker** | chaque appel LLM | `track_llm_call(model, prompt, completion)` (ABSA) ou estimation char/4 (débat multi-agent AutoGen). Cumul singleton thread-safe. Warn 80 %, hard-stop 100 % via `BudgetExceededError` → `break` propre de la boucle (articles suivants retraités demain). Dump `reports/llm_cost_daily/YYYY-MM-DD.json` à la fin de la session. |
| 6 | **Calibration nocturne** | cron/Task Scheduler 02:30 | `scripts/fit_calibration_nightly.py` collecte les trades `(p_raw, outcome)` sur 90 j via SQLite + yfinance T+5. Fit Platt + Isotonic (`fit_best_calibrator` de `utils/calibration.py`), garde le meilleur par Brier. Min 30 trades. Sauvegarde `models/calibrator.pkl` + `models/calibrator.meta.json` (ECE before/after, flag `ece_target_met`). Le pipeline live peut charger via `load_calibrator()`. |

**Migrations DB** : `dry_run INT DEFAULT 0` et `kill_switch_active INT DEFAULT 0`
sur la table `articles`, idempotentes (`CREATE TABLE IF NOT EXISTS` + `PRAGMA`
check des colonnes avant `ALTER TABLE`).

**Variables d'environnement ajoutées** (toutes surchargeables) :

| Var | Défaut | Effet |
|---|---|---|
| `DRY_RUN` | `0` | `1` → active le paper trading |
| `DRY_RUN_LOG_PATH` | `logs/dry_run_trades.jsonl` | Chemin du journal JSONL |
| `VIX_KILL_SWITCH_THRESHOLD` | `45.0` | Seuil VIX au-delà duquel on coupe |
| `LLM_DAILY_BUDGET_USD` | `5.0` | `0` pour désactiver le garde-fou |
| `LLM_COST_LOG_DIR` | `reports/llm_cost_daily` | Dossier des snapshots quotidiens |
| `MAX_SECTOR_EXPOSURE_PCT` | `0.30` | Cap sectoriel (DeMiguel 2009) |

### 3.6 Ligne de conduite "Production-Pragmatique"

Après une première passe sur-académique, le pipeline a été scindé explicitement
en **couche live** et **couche audit hors-ligne**. Règle : _"si ça coûte plus de
1 seconde ou 1 appel LLM supplémentaire par article, c'est hors-ligne"._

#### Live (dans le pipeline, appelé à chaque article)
- DistilRoBERTa filter
- ABSA (avec cache LRU)
- Débat multi-agent (4 LLMs, 3 tours)
- Bayesian Aggregator
- YOLO + Position Sizer + Execution costs (cheap mode)
- Sanitizer + Temporal fence + Anonymizer

Coût par article : **~8-15 s, 15-20 appels LLM**.

#### Offline (audit hebdomadaire, `scripts/audit_hebdomadaire.py`)
- Counterfactual Invariance sur les 50 pires trades (~2200 appels LLM)
- Event Study Fama-French + Romano-Wolf (yfinance only, gratuit)
- Backtest P&L classique (Sharpe, Drawdown, courbe de capital)
- Calibration refit (Platt + Isotonic)
- Régime analysis

Coût hebdo : **~30-40 min, ~$5 de LLM + yfinance gratuit**.

Plan Windows Task Scheduler (ou cron) :
```
Dimanche 03:00 → python -m scripts.audit_hebdomadaire
```

#### Pourquoi cette séparation ?

Le Counterfactual Invariance appelé sur 100 articles/jour = 100 × 11 × 4 LLMs =
**3300 appels API par jour**, insoutenable. Mais c'est un outil de **debug** :
on ne le lance que sur les 50 pires trades de la semaine pour détecter les
LLMs qui "trichent" via leurs priors.

L'Event Study (alpha, CAR, Newey-West, Romano-Wolf) répond à _"notre edge
est-il statistiquement significatif ?"_ — utile pour un quant.
Le Backtest P&L (Sharpe, DD, courbe de capital sur 100 k€) répond à _"combien
on gagne et peut-on dormir la nuit ?"_ — utile pour un gérant.
Les deux cohabitent dans `scripts/audit_hebdomadaire.py` — complémentaires, pas
substituables.

#### Almgren-Chriss : opt-in, pas default

Le modèle d'impact `λ · σ · sqrt(Q/ADV)` n'a de sens que si `Q/ADV > 0.5%`.
Pour du retail (<100 k$ par ordre sur du large-cap), l'impact est négligeable
devant le spread. Default `use_almgren_chriss=False` : on paie seulement la
commission IB + half-spread. Le modèle avancé reste dispo pour les cas > 0.5%
ADV ou pour les audits de cost breakdown.

---

## 4. Métriques agrégées par couche

| Couche | Métrique principale | Cible opérationnelle | Où la mesurer |
|---|---|---|---|
| 0. Sécurité | injection-detection rate | ≥ 99 % sur benchmark adversarial | `eval/benchmark_robustness.py` |
| 1. Filtrage ML | F1 financial relevance | ≥ 0.85 | `eval/evaluate_unit_llm.py` |
| 2. ABSA | accuracy aspect-label | ≥ 0.75 | `eval/evaluate_faithfulness.py` |
| 3. Débat | agents confidence dispersion (Var_epi) | 0.02 ≤ Var_epi ≤ 0.12 (sain) | `src/utils/bayesian_aggregator.py` |
| 4. Consensus | ECE (Expected Calibration Error) | ≤ 0.05 post-Platt | `src/utils/calibration.py` + `eval/evaluate_calibration.py` |
| 5. MSS / CI | Counterfactual Invariance | ≥ 0.90 | `src/utils/counterfactual.py` (usage programmatique) |
| 6. YOLO | false-positive auto-exec rate | ≤ 5 % | `src/utils/yolo_classifier.py` + audit log |
| 7. Sizing | Kelly-net-fees vs Kelly-gross | `f_net / f_gross` ∈ [0.6, 0.95] | `src/utils/execution_costs.py` |
| 8. Alpha | Romano-Wolf q-value CAR[-1,+5] | q < 0.05 | `eval/evaluate_event_study.py` |
| 9. Execution | break-even accuracy | p* < 0.55 | `execution_costs.break_even_accuracy()` |

---

## 5. Variables d'environnement

### Core

| Var | Rôle |
|---|---|
| `GROQ_API_KEY` | Llama 4 Scout / Llama 3.3 70B |
| `CEREBRAS_API_KEY` | Llama 3.1 8B (rapide) |
| `MISTRAL_API_KEY` | Mistral Small |
| `EVAL_ANONYMIZE=1` | Active l'anonymisation dans le pipeline (mode audit) |

### Garde-fous Phase 3

| Var | Défaut | Rôle |
|---|---|---|
| `DRY_RUN` | `0` | `1` = paper trading, zéro mutation portefeuille |
| `DRY_RUN_LOG_PATH` | `logs/dry_run_trades.jsonl` | JSONL append-only des ordres hypothétiques |
| `VIX_KILL_SWITCH_THRESHOLD` | `45.0` | VIX ≥ seuil → tous signaux forcés `HOLD_SYSTEMIC` |
| `LLM_DAILY_BUDGET_USD` | `5.0` | Dépassement → `BudgetExceededError`, interrupt propre |
| `LLM_COST_LOG_DIR` | `reports/llm_cost_daily` | Snapshots JSON quotidiens |
| `MAX_SECTOR_EXPOSURE_PCT` | `0.30` | Cap sectoriel (DeMiguel et al. 2009) |

---

## 6. Commandes utiles

```bash
# --- LIVE ---
python -m src.pipelines.agent_pipeline          # pipeline temps réel
python -m src.pipelines.news_pipeline           # collecte RSS

# --- PAPER TRADING (Phase 3) ---
DRY_RUN=1 python -m src.pipelines.agent_pipeline
# Inspecter les ordres hypothétiques :
#   cat logs/dry_run_trades.jsonl | jq .
# Inspecter le coût LLM du jour :
#   cat reports/llm_cost_daily/$(date +%F).json

# --- AUDIT HEBDO (offline, dim. 03:00) ---
python -m scripts.audit_hebdomadaire            # tout
python -m scripts.audit_hebdomadaire --skip-cf  # sans counterfactual (gratuit)
python -m scripts.audit_hebdomadaire --cf-pipeline  # avec vrais appels LLM (cher)

# --- REFIT CALIBRATION NOCTURNE (Phase 3, cron/Task Scheduler 02:30) ---
python -m scripts.fit_calibration_nightly --horizon-days 5 --lookback-days 90

# --- AUDITS UNITAIRES ---
python -m eval.evaluate_event_study --from-db --limit 100
python -m eval.evaluate_historical_backtest                  # P&L + Sharpe + DD
# Counterfactual Invariance : runner CLI supprime Phase 4 (cf. ADR-010).
# Pour calculer le score CI sur un article : usage programmatique via
#   from src.utils.counterfactual import score_invariance
#   from src.utils.minimal_sufficient_statistic import extract_mss
python -m eval.evaluate_calibration

# --- SMOKE TESTS Phase 3 ---
python -m src.utils.dry_run_logger              # écrit une ligne JSONL de démo
python -m src.utils.llm_cost_tracker            # dump snapshot daily JSON
python -m src.strategy.portfolio_constraints    # OK / REFUSED / Energy OK
python -m src.knowledge.liquidity               # AAPL ADV + sigma

# --- CONFIG ---
python -m src.config   # dump la config courante (pour reports)
```

---

## 7. Dépendances scientifiques principales

Aucune dépendance à scikit-learn ou statsmodels : tous les algorithmes
(Platt, isotonic PAVA, Newey-West, stationary bootstrap, inverse-normale)
sont implémentés en NumPy / Python pur pour rester léger et auditable.

```
numpy, pandas, yfinance, requests, python-dotenv
transformers (DistilRoBERTa, ABSA), torch
autogen, langgraph
sqlite3 (stdlib)
```

---

## 8. Dette technique identifiée (non-bloquante)

Voir `CRITIQUE_AND_ROADMAP.md` pour le détail priorisé. **Phase 3 a clôturé
les 6 verrous les plus critiques** — ne restent que les items sans impact
sur la sécurité du paper trading :

1. ~~**Calibration en ligne**~~ → **résolu Phase 3** (`scripts/fit_calibration_nightly.py`)
2. ~~**ADV dynamique**~~ → **résolu Phase 3** (`src/knowledge/liquidity.py`)
3. **Stationary Bootstrap** dans Event Study : block length = 5 en dur. Devrait utiliser Politis-White (2004) pour calibrer `p`.
4. **RotatingFileHandler** : `agent_pipeline.log` append sans rotation — à migrer vers `RotatingFileHandler(maxBytes=10MB, backupCount=5)`.
5. **Anonymizer scalability** : 17 tickers hardcodés — migrer vers liste générée depuis le DB.
6. **Rate limiting LLM** : garde-fou budgétaire en place (Phase 3), mais pas encore de `@rate_limited(rpm=100)` ni de retry-backoff standardisé sur les 429.
7. **CI/CD minimal** : pas encore de `.github/workflows/test.yml`.

---

## 9. Changelog (avril 2026)

### Phase 1 — Audit Quant Tier-1 (research-grade)
- Module ① Event Study (F-F 3 factor + Newey-West + Romano-Wolf)
- Module ② Bayesian Aggregator (Beta-Binomial + variance décomposée)
- Module ③ Counterfactual Invariance + MSS 20-dim
- Calibration submodule (Platt + Isotonic PAVA + ECE + Brier)
- Execution costs (Almgren-Chriss + Kissell timing)
- Position sizer : Kelly net de frais, scaling bayésien, clamp relâché
- SIDEWAYS unifié (vol-adjusted σ-based)
- 11 tickers delisted ajoutés à l'anonymizer
- Pipeline : temporal_fence + anonymizer + bayes branching câblés
- Fix duplicate `=== ÉTAPE 4 ===` dans PROMPT_CONSENSUS

### Phase 2 — Refactoring Production-Pragmatique
- **Counterfactual débranché du live** : banner offline-only + guard `--confirm-cost`
- **Backtest P&L restauré** : Calmar/Sortino + courbe de capital CSV (complément Event Study)
- **`scripts/audit_hebdomadaire.py`** : orchestrateur hebdo (~$5/semaine)
- **Execution costs simplifié** : cheap mode par défaut (IB commissions + spread),
  Almgren-Chriss derrière `use_almgren_chriss=True`
- **Log append** : `mode="a"` au lieu de `"w"` (plus d'écrasement d'audit trail)
- **Cache LRU ABSA** : 256 entrées sur hash(content[:500])
- **Cache TTL macro** : 15 min (évite 6 appels yfinance/article)
- **`src/config.py`** : constantes cross-module + override env vars
- **`CRITIQUE_AND_ROADMAP.md`** : revue critique consolidée + roadmap

### Phase 3 — Garde-fous opérationnels
- **DRY_RUN / paper trading** : env var `DRY_RUN=1` → JSONL append-only, zéro mutation portefeuille. `src/utils/dry_run_logger.py`, colonne DB `articles.dry_run`.
- **Kill-switch VIX** : `VIX_KILL_SWITCH_THRESHOLD=45.0` (Whaley 2009). VIX ≥ seuil → tous signaux forcés `HOLD_SYSTEMIC`. Guard ajouté dans `position_sizer` pour que le sizing retourne `TENIR` immédiatement. Colonne DB `articles.kill_switch_active`.
- **ADV dynamique** : `src/knowledge/liquidity.py` — yfinance `averageVolume` + σ sur 3mo log-returns, cache 24 h. Remplace le default 5 M actions / 1.8 % (biaisait fortement les small-caps).
- **Cap sectoriel 30 %** : `src/strategy/portfolio_constraints.py` — `check_sector_concentration()` post-sizing, refus → `HOLD_SECTOR_CAP`. DeMiguel, Garlappi, Uppal (2009).
- **LLM cost tracker** : `src/utils/llm_cost_tracker.py` — singleton thread-safe, budget `LLM_DAILY_BUDGET_USD=5.0`, `BudgetExceededError` attrapé par le pipeline → `break` propre. Snapshot JSON quotidien (`reports/llm_cost_daily/YYYY-MM-DD.json`), warn à 80 %.
- **Calibration nocturne** : `scripts/fit_calibration_nightly.py` — collecte (p_raw, outcome) sur 90 j via yfinance T+5, fit Platt + Isotonic, garde le meilleur par Brier, persiste `models/calibrator.pkl` + metadata.
- Migrations DB idempotentes (`dry_run`, `kill_switch_active`).
- 4 smoke tests unitaires validés (dry_run_logger, llm_cost_tracker, portfolio_constraints, liquidity). Pipeline import : OK.

### Phase finale — Optimisation Modèles & Cleanup
- **Débat Multi-Agent (ADR-017)** : Remplacement des agents par des modèles NIM très rapides (1.0s à 3.8s) offrant une diversité épistémique maximale (NVIDIA, Mistral, Alibaba, Meta).
- **Cleanup du code** : Nettoyage via Ruff (328 erreurs de lint corrigées), suppression d'imports inutilisés, et refactoring des imports. 
- Validation globale : Tous les tests passent avec succès, documentation à jour, architecture validée.

---

**Dernière mise à jour** : 2026-04-27 (Phase finale — Optimisation Modèles & Cleanup)
