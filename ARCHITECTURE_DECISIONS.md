# Architecture Decision Records (ADRs)

> Document interne tracant les **decisions techniques structurantes**, le
> **contexte** dans lequel elles ont ete prises, les **alternatives evaluees**
> et leurs **consequences**. Chaque entree est immutable une fois acceptee
> (revision = nouvelle ADR qui supersede l'ancienne).

Format inspire de [Michael Nygard's ADR template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

---

## Sommaire

| # | Titre | Statut | Date |
|---|---|---|---|
| 001 | Choisir 3 fournisseurs LLM distincts pour le debat | Accepte | 2026-04 |
| 002 | Migration `print()` -> `logging` : progressive, pas big-bang | Accepte | 2026-04-25 |
| 003 | Couche d'abstraction LLM via fallback en cascade | Accepte | 2026-04-25 |
| 004 | Persistance JSONL append-only pour le harvest | Accepte | 2026-04 |
| 005 | Counterfactual Invariance hors live (cout LLM) | Accepte | 2026-04 |
| 006 | Almgren-Chriss en opt-in (cheap mode par defaut) | Accepte | 2026-04 |
| 007 | NumPy pur sans scikit-learn ni statsmodels | Accepte | 2026-04 |
| 008 | Imports : `from src.x` plutot que relatifs ou sys.path | Accepte | 2026-04-25 |
| 009 | mypy strict zone progressive (pas de big-bang typage) | Accepte | 2026-04-25 |
| 010 | Suppression du sous-systeme `agent_factory` / `prompt_registry` | Accepte | 2026-04-25 |
| 011 | Couche d'execution broker via Adapter pattern (DryRun / Alpaca) | Accepte | 2026-04-26 |
| 012 | Anonymizer dynamique (DB-loaded) en complement du hardcode | Accepte | 2026-04-26 |
| 013 | Ajout de NVIDIA NIM comme 4e jambe de fallback LLM | Accepte | 2026-04-26 |
| 014 | Model Routing : "right model for right task" via BEST_MODELS_BY_TASK | Accepte | 2026-04-26 |
| 015 | ReasoningAuditor : couche d'audit post-debat via modele "thinking" | Accepte | 2026-04-26 |
| 016 | Modeles NIM reserves aux taches batch (latence prohibitive en live) | Revise par ADR-017 | 2026-04-26 |
| 017 | Reintegration des modeles NIM rapides pour le debat principal | Accepte | 2026-04-27 |

---

## ADR-001 — Choisir 3 fournisseurs LLM distincts pour le debat

**Statut** : Accepte
**Date** : 2026-04
**Contexte** :
Le debat multi-agent peut techniquement utiliser le meme LLM pour les 3 roles
(Haussier, Baissier, Neutre) avec des prompts differents. Mais les LLMs ont
des biais cognitifs systemiques (tendance haussiere RLHF, hallucinations
similaires sur les memes patterns).

**Decision** :
Utiliser **3 fournisseurs distincts** pour les 3 agents :
- Haussier : Cerebras (Llama-3.3-70b)
- Baissier : Groq (Llama-4-Scout-17b)
- Neutre : Mistral-Small

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| 1 LLM x 3 prompts | Simple, 1 cle API, debat reproductible | Erreurs correlees, biais commun, pas de "fresh perspective" |
| 3 LLMs meme famille (Llama) | Diversite limitee, cout reduit | Architectures voisines, biais entrainement similaires |
| **3 fournisseurs distincts** | Diversite epistemique reelle, fail-over inter-provider | 3 cles API, plus complexe, $$ |

**Consequences** :
- (+) Diversite des erreurs : un consensus 2/3 a plus de signal qu'un consensus
  monolithique.
- (+) Resilience : Groq down -> on peut reaffecter dynamiquement (cf. ADR-003).
- (-) Coordination des prompts plus delicate (rate limits, timeouts hetereogenes).
- (-) Couts : ~3x un setup mono-provider (mitige par budget LLM_DAILY_BUDGET_USD).

---

## ADR-002 — Migration `print()` -> `logging` progressive

**Statut** : Accepte
**Date** : 2026-04-25
**Contexte** :
Le code historique utilise massivement `print()` (252 occurrences au moment de
l'audit). Probleme : pas de niveau, pas d'agregation, encodage Windows cp1252
fragile sur les caracteres non-ASCII.

**Decision** :
- Creer un module `src/utils/logger.py` avec `get_logger(name)` central.
- Tous les **nouveaux modules** doivent l'utiliser exclusivement.
- Le code historique migre **au cas par cas** lors de toute modification
  ulterieure, **pas en big-bang**.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Big-bang (tout migrer maintenant) | Coherence immediate | 252 `print()` a migrer, risque de regressions, +1 semaine |
| Status quo | Aucun travail | Probleme persiste, dette s'accumule |
| **Progressive + module standard** | Standard etabli, dette quantifiable | Coexistence temporaire `print` + `log` |

**Consequences** :
- (+) Standard clair pour le nouveau code.
- (+) Pas de risque de regression sur le code historique stable.
- (+) Format JSON disponible pour ingestion machine (autopilot week-long).
- (-) Coherence stylistique sacrifiee a court terme.

**Critere de cloture** :
Cette ADR sera revisee quand `print()` aura disparu de `src/`. Outil de
mesure : `grep -rn "print(" src/ | wc -l`.

---

## ADR-003 — Couche d'abstraction LLM via fallback en cascade

**Statut** : Accepte
**Date** : 2026-04-25
**Contexte** :
La logique "essaie Groq, sinon Mistral, sinon Cerebras" etait dupliquee dans 5+
fichiers (`agent_absa.py`, `agent_memoire.py`, `agent_debat.py`,
`context_compressor.py`, `agent_filtrage_api.py`). Si Groq tombe, l'agent
Baissier crash et stoppe le debat entier — pas de fallback inter-provider.

**Decision** :
Creer `src/utils/llm_client.py` avec une classe `LLMClient` qui :
1. Detecte les fournisseurs disponibles via env vars.
2. Tente une chaine de fournisseurs (configurable via `model_preference`).
3. Retry exponentiel borne (max 3 tentatives par provider, cap 8s).
4. Bascule vers le provider suivant si retry epuise.
5. Leve `AllProvidersFailedError` si toute la chaine echoue.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Status quo (logique copiee-collee) | Aucun travail | Bugs subtils, pas de fallback, SPOF par agent |
| Strategy pattern + DI | Tres flexible | Overengineering pour 3 providers |
| **Fallback en cascade unifie** | Standard simple, testable, factorisable | Migration progressive necessaire |

**Consequences** :
- (+) Resilience : un seul provider down ne tue pas le pipeline.
- (+) Code historique non-casse (le client est opt-in).
- (+) 18 tests unitaires + mode `stub_response` deterministe pour les tests.
- (-) Migration des 5+ fichiers historiques reste a faire (suivi sous Phase 4).

---

## ADR-004 — Persistance JSONL append-only pour le harvest

**Statut** : Accepte
**Date** : 2026-04
**Contexte** :
Le systeme autopilot (`DailyHarvestOrchestrator`) doit persister les signaux
generes pour reproductibilite scientifique. Choix entre SQLite (deja utilise
pour le live) et JSONL.

**Decision** :
- **JSONL append-only** avec rotation journaliere (`harvest_YYYY-MM-DD.jsonl`).
- Schema versionne (`HARVEST_SCHEMA_VERSION = 1` dans chaque ligne).
- SQLite reserve au pipeline live (mutations frequentes, requetes complexes).

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| SQLite | Requetes SQL puissantes, deja en place | Mutation possible -> risque audit trail, locks concurrents |
| Parquet | Compression, query analytique | Overhead, non-streamable |
| **JSONL append-only** | Append atomique, durable, streamable, simple `cat`/`jq` | Pas de query SQL native (acceptable hors-ligne) |

**Consequences** :
- (+) Reproductibilite : on peut rejouer un etat passe a partir des fichiers.
- (+) Audit trail incassable : append-only.
- (+) Hash SHA256 de l'univers garantit qu'on ne reecrit pas l'histoire.
- (-) Pas de query SQL ad-hoc — on charge en pandas pour analyse.

---

## ADR-005 — Counterfactual Invariance hors live (cout LLM)

**Statut** : Accepte
**Date** : 2026-04
**Contexte** :
Le module Counterfactual Invariance (Veitch 2021) genere 11 perturbations par
article (ticker_swap x6, date_shift, magnitude_eps5, sector_swap, ceo_removed,
style_neutral). Pour chaque perturbation, 4 LLMs sont appeles. Pour 100
articles/jour : 100 x 11 x 4 = **4400 appels API/jour**, insoutenable.

**Decision** :
- Counterfactual Invariance **debranchee du pipeline live**.
- Deplacee dans `scripts/audit_hebdomadaire.py`, applique sur les **50 pires
  trades de la semaine** (debug ciblé).
- Le runner CLI dedie (`eval/evaluate_counterfactual_invariance.py`) a ete
  supprime ulterieurement (cf. ADR-010), les modules `src/utils/counterfactual.py`
  et `src/utils/minimal_sufficient_statistic.py` restent disponibles
  programmatiquement.
- Guard `--confirm-cost` requis lors de l'audit hebdo pour confirmer le cost > $1.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Live integral | Robustesse maximale | $$$ (4400 calls/j), bloquant |
| Sample 10% live | Cout / 10 | Statistique faible, biais sampling |
| **Hors live + audit hebdo** | Cout maitrise (~$5/sem), focus sur les pires cas | Pas de detection en temps reel |

**Consequences** :
- (+) Cout LLM maitrise : ~$5/semaine au lieu de $20/jour.
- (+) Focus debug : les 50 pires trades sont les plus pertinents pour CI.
- (-) Pas de detection real-time des LLMs qui "trichent" via priors —
  acceptable pour Phase 3 (paper trading).

---

## ADR-006 — Almgren-Chriss en opt-in (cheap mode par defaut)

**Statut** : Accepte
**Date** : 2026-04
**Contexte** :
Le modele d'impact Almgren-Chriss (`lambda * sigma * sqrt(Q/ADV)`) est
academiquement rigoureux mais surdimensionne pour du retail (<100k USD par
ordre sur du large-cap). L'impact y est negligeable devant le spread.

**Decision** :
- **Cheap mode par defaut** : commission IB + half-spread (3.3 bps round-trip
  pour du large-cap retail).
- **Almgren-Chriss derriere `use_almgren_chriss=True`**, automatiquement
  declenche si `Q/ADV > ALMGREN_CHRISS_AUTO_TRIGGER_ADV_PCT` (defaut 0.5%).

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Almgren-Chriss toujours | Rigueur academique max | Sur-estime cout x10 pour retail, biais sizing |
| Cheap mode toujours | Realiste pour retail | Inadequat pour ordres > 0.5% ADV |
| **Cheap mode + AC opt-in conditionnel** | Realisme retail + scalabilite institutional | Complexite legere |

**Consequences** :
- (+) Sizing realiste pour le mode majoritaire (retail).
- (+) AC reste disponible pour audits "et si ordre x10 ?".
- (-) 2 chemins de calcul a maintenir (acceptable, branche `if` simple).

---

## ADR-007 — NumPy pur sans scikit-learn ni statsmodels

**Statut** : Accepte
**Date** : 2026-04
**Contexte** :
Le projet implemente Platt scaling, isotonic PAVA, Newey-West HAC, stationary
bootstrap, inverse-normale Beasley-Springer-Moro. Toutes ces methodes existent
dans scikit-learn et statsmodels.

**Decision** :
**Implementer en NumPy pur** plutot qu'importer scikit-learn / statsmodels.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| scikit-learn + statsmodels | Standards, rapide | +200 Mo deps, pas auditable, version pinning casse-tete |
| **NumPy pur** | Leger, auditable, reproductible | Plus de code a maintenir (~500 lignes) |

**Consequences** :
- (+) Auditabilite : chaque algorithme est lisible, references cite (Platt 1999,
  Politis-Romano 1994).
- (+) Reproductibilite : pas de dependance ABI sur scikit-learn version X.
- (+) CI legere : pas besoin d'installer scikit-learn pour la matrice de tests.
- (-) Code custom = bugs custom (mitige par tests dedies).
- (-) Pas de bench public (mais cross-checked sur petits cas vs sklearn).

---

## ADR-008 — Imports : `from src.x` plutot que relatifs ou sys.path

**Statut** : Accepte
**Date** : 2026-04-25
**Contexte** :
Le projet melange 3 styles d'imports :
- Relatifs : `from .x import y` (4 fichiers, surtout `__init__.py`)
- Absolus : `from src.x import y` (10 fichiers)
- `sys.path.insert(0, ...)` (8 fichiers, dont du code prod)

Cela revele plusieurs couches d'evolution sans refactor unifie.

**Decision** :
- **Convention absolue `from src.x import y` partout** dans le code de prod.
- `__init__.py` peuvent garder `from .x import y` (idiomatic dans les packages).
- `sys.path.insert(0, ...)` retire systematiquement par installation editable
  (`pip install -e .`) declenche par `pyproject.toml`.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Status quo | Aucun travail | Confusion, fragilite execution depuis differents CWD |
| Tout-relatif | Coherent dans un package | Ne marche pas hors mode editable |
| **Absolu uniforme + pip install -e** | Robuste, IDE-friendly, coherent | Migration progressive requise |

**Consequences** :
- (+) Imports robustes peu importe le CWD.
- (+) Tests pytest et `python -m src.x.y` fonctionnent uniformement.
- (-) Migration des 8 fichiers `sys.path.insert` reste a faire (Phase 4).

---

## ADR-009 — mypy strict zone progressive (pas de big-bang typage)

**Statut** : Accepte
**Date** : 2026-04-25
**Contexte** :
Le code historique a des annotations partielles (parametres typees mais pas
les retours, ou inversement). Forcer `disallow_untyped_defs = True` partout
demande des semaines de travail.

**Decision** :
- **Strict zone** progressive : seuls `src/discovery/`, `src/utils/logger.py`,
  `src/utils/llm_client.py` (modules audites recents) sont en mode strict.
- Reste du code en mode tolerant (`disallow_untyped_defs = false`).
- Tout nouveau module va dans la strict zone par defaut.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Big-bang (`strict = true` global) | Coherence immediate | 6 semaines de travail estimees, blocage produit |
| Pas de mypy | Aucun travail | Pas de filet sur les types |
| **Strict zone progressive** | ROI immediat sur le neuf, dette quantifiable | Coexistence temporaire de niveaux |

**Consequences** :
- (+) ROI typage immediat sur les modules critiques.
- (+) CI verte sans bloquer les contributions historiques.
- (+) Standard etabli pour le nouveau code.
- (-) Code historique reste partiellement non-type (acceptable, voir ADR-002).

**Critere de cloture** :
Cette ADR sera revisee quand 80% des modules seront en strict zone.

---

## ADR-010 — Suppression du sous-systeme `agent_factory` / `prompt_registry`

**Statut** : Accepte
**Date** : 2026-04-25
**Contexte** :
Une architecture alternative avait ete prototypee :
- `src/utils/prompt_registry.py` : registre central de prompts YAML versionnees
- `src/utils/agent_factory.py` : factory dynamique d'agents bases sur le registre
- `src/utils/debate_termination.py` : protocole de terminaison de debat
- `src/agents/agent_reflector.py` : agent meta de reflection
- `prompts/` (7 YAML) : prompts externalisees

Cette architecture n'a jamais ete branchee dans le pipeline live (les agents
hardcodent encore leurs prompts). Aucun import depuis le code de prod actif.
Test associe : `test_prompt_registry.py` (22 tests, modules dead).

**Decision** :
**Supprimer integralement** le sous-systeme :
- `src/utils/prompt_registry.py`
- `src/utils/agent_factory.py`
- `src/utils/debate_termination.py`
- `src/agents/agent_reflector.py`
- `tests/test_prompt_registry.py`
- `prompts/` (7 fichiers YAML)

Soit ~1200 lignes de code mort eliminees.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Garder "au cas ou" | Aucun travail destructif | Code mort biaise les futurs lecteurs, dette caché |
| Resusciter (cabler dans pipeline) | Architecture plus propre | 2-3 semaines de refactor, risque regressions |
| **Supprimer** | Code mort = 0, lisibilite +++ | Si on en avait besoin plus tard, regrets |

**Consequences** :
- (+) Lisibilite du projet ameliore : moins de fichiers, moins de "qu'est-ce que ca fait ?"
- (+) Surface d'attaque reduite (moins de YAMLs a auditer pour prompt-injection).
- (+) Tests : 22 tests dead supprimes, suite plus rapide.
- (-) Si on veut un registre de prompts plus tard, on reimplemente from scratch
  (acceptable : YAML + dataclass = 30 minutes de travail).

**Reference** :
Suppression effectuee lors du nettoyage Phase 4 (cf. commit cleanup avril 2026).

---

## ADR-011 — Couche d'execution broker via Adapter pattern

**Statut** : Accepte
**Date** : 2026-04-26
**Contexte** :
Le pipeline produit des signaux `Achat / Vente / Neutre` avec sizing, mais
n'avait aucune couche pour les transformer en ordres reels. Le mode `DRY_RUN`
loggait directement en JSONL via `dry_run_logger.py`, fortement couple au
pipeline. Pas d'extensibilite : impossible d'ajouter un broker reel sans
toucher 5+ fichiers.

**Decision** :
Creer un nouveau package `src/execution/` implementant le pattern Adapter
(Gamma 1994) :
- **`broker_protocol.py`** : `BrokerProtocol` (PEP 544, Protocol/duck-typing)
  + dataclasses `OrderIntent`, `OrderResult`, `Position`, `AccountState`.
- **`dry_run_broker.py`** : implementation par defaut, JSONL append-only,
  thread-safe.
- **`alpaca_broker.py`** : implementation Alpaca paper/live trading, SDK
  alpaca-py en dep optionnelle (lazy import).
- **`get_broker()`** : factory selectionnee par env var `BROKER_BACKEND`.

Le pipeline ne sait pas quel broker il utilise — c'est purement decouple.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Status quo (DRY_RUN dans le pipeline) | Aucun travail | Pas extensible, couple |
| Heritage classique (BaseBroker abstract) | Patterne enseigne | Verbose, force ABC |
| **Protocol PEP 544 + Adapter** | Duck-typing, simple, testable | Necessite Python 3.8+ (OK) |
| Service mesh / gRPC | Multi-language | Overkill pour 1 user |

**Consequences** :
- (+) Pipeline decouple : ajouter Interactive Brokers = nouveau fichier, 0 modif pipeline.
- (+) Tests : 25 tests unitaires (mocks complets du SDK Alpaca, thread-safety).
- (+) Mode dry_run conserve par defaut (pas de risque de trade reel accidentel).
- (+) Kill-switch VIX (Phase 3) reste compatible : applique au signal AVANT le broker.
- (-) Le `dry_run_logger.py` historique devient redondant — A migrer ulterieurement
  vers `DryRunBroker` (Phase 5).

---

## ADR-012 — Anonymizer dynamique (DB-loaded) en complement du hardcode

**Statut** : Accepte
**Date** : 2026-04-26
**Contexte** :
L'anonymizer (`src/utils/anonymizer.py`) avait 17 tickers hardcodes (Big Tech +
delisted importants) avec mappings premium (CEO, produits, lieux). Probleme :
non scalable. Pour anonymiser un ticker hors de cette liste, on retombait sur
"TICKER_X / CorpX" qui ne remplace rien dans le texte.

C5 dans `CRITIQUE_AND_ROADMAP.md` : "Anonymizer hardcode" = 🟡 Basse priorite.

**Decision** :
Garder le hardcode (mappings premium pour Big Tech) + ajouter un mecanisme
d'enrichissement dynamique :
- `extend_db_from_database(db_path)` : SELECT DISTINCT ticker FROM articles
  -> generation d'un alias deterministe par hash SHA256 (prefixe `DYN_`).
- Idempotent (cache thread-safe) : 1 SELECT par session.
- Defensive : DB absente / corrompue / sans table -> retourne 0, pas de crash.
- `reset_dynamic_cache()` + `reset_entity_database()` pour les tests.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Tout hardcode (status quo) | Mappings premium | Non scalable |
| Tout dynamique (DB) | Scalable | Perd les mappings premium (CEO, produits) |
| **Hybride : hardcode + extend DB** | Premium + scalable | Code legerement plus complexe |

**Consequences** :
- (+) Anonymizer scalable a tout univers (S&P 500, Russell 2000).
- (+) Hardcode preserve pour les Big Tech (qualite des alias).
- (+) Hash SHA256 deterministe : meme ticker -> meme alias, reproductibilite.
- (+) 14 tests unitaires (alias generation, idempotence, defensive DB handling).
- (-) L'enrichissement dynamique ne genere que ticker -> alias minimal (pas de
  CEO, produits, lieux). Pour ces metadata, il faudrait du LLM ou une autre source.

---

## ADR-013 — Ajout de NVIDIA NIM comme 4e jambe de fallback LLM

**Statut** : Accepte
**Date** : 2026-04-26
**Contexte** :
La chaine de fallback LLM (cf. ADR-001) compte 3 providers : Groq, Mistral,
Cerebras. NVIDIA NIM (build.nvidia.com) propose en parallele une API gratuite
**OpenAI-compatible** donnant acces a 100+ modeles hostes sur DGX Cloud, dont :

- **Llama 3.1 405B** (modele geant non disponible gratuitement sur Groq/Mistral/Cerebras)
- **DeepSeek-R1** : reasoning explicite via chain-of-thought, prometteur pour le debat
- **Nemotron** : famille NVIDIA optimisee, reglages fins post-training
- **Codestral, Phi-4, GLM-5, Qwen 2.5...**

**Decision** :
Ajouter NIM comme **4e provider** dans `PROVIDERS` (cf. `src/utils/llm_client.py`)
en queue de la chaine de fallback par defaut : `[groq, mistral, cerebras, nvidia_nim]`.

Cle API gratuite via NVIDIA Developer Program (env var `NVIDIA_NIM_API_KEY`,
prefixe `nvapi-`). Endpoint : `https://integrate.api.nvidia.com/v1`.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Status quo (3 providers) | Aucun travail | SPOF si les 3 tombent simultanement |
| NIM en 1ere position | Acces immediat aux gros modeles | Latence/SLA inconnus, ramene NIM en chemin critique |
| **NIM en 4e position (queue)** | Resilience accrue, opt-in via env var, modeles uniques disponibles via `model_preference` explicite | Latence supplementaire si les 3 premiers echouent (acceptable, c'est un fallback) |
| 5+ providers (OpenAI, Anthropic, etc.) | Encore plus de resilience | Couteux ($), explosion combinatoire des tests |

**Consequences** :
- (+) Resilience : les 4 providers tombent simultanement = scenario quasi-impossible (4 backbones distincts).
- (+) Diversite epistemique +33% (3 -> 4 fournisseurs distincts).
- (+) Acces gratuit a DeepSeek-R1 / Nemotron / Llama 405B pour experimentations
  ciblees (`model_preference=["nvidia_nim"]` + `default_model` surcharge).
- (+) Migration transparente : le code historique qui ne specifie pas
  `model_preference` profite automatiquement de la 4e jambe.
- (-) 40 RPM par defaut sur tier free (suffisant pour usage normal, augmentable
  a 200 RPM sur demande forum).
- (-) SLA non garanti (tier free) — mais c'est un fallback, pas le 1er choix.

**Reference** :
- API: https://integrate.api.nvidia.com/v1 (OpenAI Chat Completions compatible)
- Catalogue modeles: https://build.nvidia.com/models
- Forum (rate limit): https://forums.developer.nvidia.com (tag NIM)

---

## ADR-014 — Model Routing : "right model for right task"

**Statut** : Accepte
**Date** : 2026-04-26
**Contexte** :
Avec NIM (ADR-013), on a soudain acces a 100+ modeles dont certains sont
**objectivement meilleurs** pour des taches specifiques (Llama 3.1 405B sur
LongBench, Nemotron-4 sur structured output, Qwen3-thinking sur reasoning).

Mais "soudain choisir n'importe quel modele a chaque appel" mene au chaos :
- Choix par intuition au lieu de benchmark
- Pas de tracabilite ("pourquoi 405B ici ?")
- Pas de moyen de tester l'impact d'un changement

**Decision** :
Codifier les choix dans un **registre central** `BEST_MODELS_BY_TASK` :
```python
BEST_MODELS_BY_TASK = {
    "reasoning_audit":      ("nvidia_nim", "qwen/qwen3-next-80b-a3b-thinking"),
    "long_summarization":   ("nvidia_nim", "meta/llama-3.1-405b-instruct"),
    "structured_extraction": ("nvidia_nim", "nvidia/nemotron-4-340b-instruct"),
    "cheap_classification": ("nvidia_nim", "meta/llama-3.1-8b-instruct"),
    "code_extraction":      ("nvidia_nim", "qwen/qwen2.5-coder-32b-instruct"),
    "general_strong":       ("nvidia_nim", "meta/llama-3.3-70b-instruct"),
}
```
Chaque entree cite le **benchmark public** qui justifie le choix (cf. commentaires
dans `src/utils/llm_client.py`).

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Hardcoder le modele dans chaque appel | Explicite | Dispersion, pas de single source of truth |
| Choix dynamique par LLM router (e.g., RouteLLM) | Adaptatif | Overhead, complexite, opaque |
| **Registre central + best_model_for_task()** | Tracable, refactorable, testable | Update manuel quand nouveau modele |
| Tout en `default_model` du provider | Simple | Casse l'idee "1 provider, plusieurs modeles" |

**Consequences** :
- (+) Choix de modele = decision **explicite**, justifiee par reference.
- (+) Refactor possible : si Llama 3.2 sort meilleur sur structured, on change 1 ligne.
- (+) Verifie empiriquement : tests live (2026-04-26) ont valide les choix
  (Qwen3-thinking 3.3s avec CoT, Llama 405B 4.6s reponse directe, etc.).
- (+) Le registre est une **specification testable** (`tests/test_llm_client.py`
  verifie que reasoning_audit pointe vers un modele "thinking" capable).
- (-) Pas dynamique : si un modele se degrade, il faut un audit manuel pour
  changer le registre. Mitigation : tests live periodiques (Phase 5).
- (-) Limite a NIM aujourd'hui (les 6 entrees pointent toutes sur nvidia_nim).
  C'est volontaire : l'objectif initial est d'exploiter le catalogue NIM.
  Pour un autre provider (ex: Groq Llama 4), il faudrait soit etendre le
  registre soit faire un appel `model_override` direct.

**Reference verification empirique** :
Bench live le 2026-04-26 sur 4 modeles reasoning candidats, sur la question
"AAPL beat EPS 12% mais Q4 guidance -5% : signal dominant ?" :

| Modele | Latence | Sortie utile |
|---|---|---|
| qwen/qwen3-next-80b-a3b-thinking | 3.3s | CoT visible + reponse claire ✅ |
| meta/llama-3.1-405b-instruct | 4.6s | Reponse directe propre ✅ |
| mistralai/magistral-small-2506 | 6.9s | Tronque a 200 tokens (pense trop) ❌ |
| nvidia/llama-3.3-nemotron-super-49b-v1.5 | 6.4s | Tronque a 200 tokens ❌ |

Confirme le choix : Qwen3-thinking pour reasoning_audit, Llama 405B pour long_summarization.

---

## ADR-015 — ReasoningAuditor : couche d'audit post-debat via modele "thinking"

**Statut** : Accepte
**Date** : 2026-04-26
**Contexte** :
L'audit existant (`agent_critic`, `agent_verifier`) regarde **arguments
individuels** : "claim X est-il supporte par evidence Y ?".

Manque : un audit de la **STRUCTURE LOGIQUE GLOBALE** du debat 3-agents.
Detecter par exemple :
- L'agent Bull rejette systematiquement les counter-evidences (confirmation_bias)
- L'agent Bear ajoute "well in this case..." quand corner (ad_hoc_rescue)
- Les 3 tours fixent sur le 1er chiffre cite (anchoring)

Ces fallacies argumentatives sont visibles a l'echelle du debat entier,
pas argument-par-argument.

**Decision** :
Creer `src/agents/agent_reasoning_auditor.py` qui :
1. Recoit le scratchpad XML complet (3 tours, agents Bull/Bear/Neutre).
2. L'envoie a un modele **thinking** (Qwen3-thinking via NIM par defaut).
3. Demande une analyse structuree en JSON : taxonomie de fallacies + severity.
4. Retourne un `ReasoningAudit` immutable + serialisable.
5. **Opt-in** via env var `ENABLE_REASONING_AUDITOR=1`. **Pas active par defaut.**

Recommande pour :
- Audit hebdomadaire (`scripts/audit_hebdomadaire.py`) sur les 50 pires trades.
- Decisions HIGH_RISK (YOLO classifier `requires_human=True`).
- **PAS** dans le pipeline live (latence 30s, cout LLM, pas adapte a 100 articles/jour).

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Status quo (Critic/Verifier seulement) | Couvre l'argument-level | Manque le debate-level |
| Reformer le Critic existant | 1 module | Critic actuel utilise n'importe quel LLM, pas optimal pour reasoning |
| **Nouveau ReasoningAuditor dedie thinking** | Specialise, opt-in | 1 module supplementaire |
| Audit humain | Qualite max | Pas scalable |

**Consequences** :
- (+) Detection de biais cognitifs **mesurables** : test live (2026-04-26)
  sur scratchpad biaise -> 2 fallacies detectees (ad_hoc_rescue + cherry_picking),
  severity moderate, adjustment -0.5.
- (+) Sortie structuree JSON -> integrable dans la DB pour analyse longitudinale
  ("quels agents bias-prone, quelle frequence par fallacy_type ?").
- (+) Degrade gracieusement : si LLM down ou JSON invalide -> retourne CLEAN
  (pas de crash de pipeline).
- (+) 18 tests unitaires, dont robustesse face a JSON invalide / severity
  inconnue / fallacy non taxonomique.
- (-) Latence 30s sur Qwen3-thinking (le modele "pense" beaucoup avant de
  produire le JSON). Acceptable hors-live.
- (-) max_tokens=4096 par defaut (modele thinking gourmand en tokens). Calibre
  empiriquement pour eviter de tronquer le JSON final.

**Reference scientifique** :
- Wei, J. et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models." NeurIPS.
- Du, Y. et al. (2023). "Improving Factuality and Reasoning in Language Models
  through Multiagent Debate." NeurIPS.
- DeepSeek-AI (2025). "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs
  via Reinforcement Learning."
- Mercier & Sperber (2011). "Why do humans reason ? Arguments for an argumentative
  theory." Behavioral and Brain Sciences. (justification : le raisonnement humain
  est souvent post-hoc rationalization, d'ou la valeur d'un auditeur externe).

---

## ADR-016 — Modeles NIM reserves aux taches batch (latence prohibitive en live)

**Statut** : Accepte
**Date** : 2026-04-26
**Contexte** :
Apres l'integration de NIM (ADR-013) et la creation du registre de model
routing (ADR-014), une question s'est posee : **doit-on remplacer un ou
plusieurs agents du debat principal (Bull/Bear/Neutre) par des modeles
NIM plus puissants (Llama 405B, Nemotron-Super, DeepSeek) ?**

L'intuition initiale etait : "remplacer Bull (Cerebras 8B) par Nemotron-Super
ou Llama 405B (NIM) augmenterait la qualite ET la diversite epistemique
(les 3 agents actuels sont 2 Llama Meta + 1 Mistral, peu diversifies)."

**Methode de decision : bench live (faits, pas opinions)**.
Le 2026-04-26, sur le prompt typique d'un agent Bull (3 phrases sur AAPL) :

| Modele | Latence | Qualite |
|---|---|---|
| Cerebras Llama-3.1-8B (BULL ACTUEL) | < 1s | Suffisante |
| Groq Llama-3.3-70B (CONSENSUS ACTUEL) | ~2-3s | Bonne |
| **NIM Llama-3.3-70B** | **34s** | Bonne |
| **NIM Nemotron-Super-49B-v1.5** | 9.6s | **Output vide** (mode thinking) |
| **NIM Llama-3.1-405B** | **110s** | Excellente |
| **NIM DeepSeek-V3.2** | timeout | — |

**Constat empirique** : NIM est entre **10x et 100x plus lent** que Cerebras
ou Groq. Pour un debat de 9 calls sequentiels (3 agents × 3 tours), passer
**1 seul agent** sur NIM ferait monter la latence du debat de ~20s a ~5 min
par article. Pour 50-100 articles/jour : **inacceptable**.

**Decision** :
1. **NE PAS remplacer** les agents du debat principal par des modeles NIM.
   Le bench montre que la diversite epistemique theoriquement gagnee est
   annihilee par la latence pratique.
2. **NE PAS ajouter** de 4eme agent NIM au debat (latence × 1.33 + NIM lent
   = explosion des couts en temps).
3. **MIGRER** les taches **batch / asynchrones** vers NIM, ou la latence
   est un non-probleme :
   - **`agent_memoire`** (consolidation hebdo `AutoDream`) :
     Groq Llama-3.3-70B -> **NIM Llama-3.1-405B**. Run nightly, 1 ticker
     ~2 min, qualite de synthese significativement meilleure (Meta Llama 3.1
     paper, LongBench benchmark).
   - **`ReasoningAuditor`** (cf. ADR-015) : Qwen3-thinking via NIM, deja en place.
4. **CREER `HighQualityConsensus`** (`agent_high_quality_consensus.py`) :
   second avis via Llama-3.1-405B, **opt-in** via env var, declenche
   uniquement sur HIGH_RISK (YOLO `ELEVE` ou audit `MODERATE`/`SEVERE`),
   typiquement 1-5 fois/jour. Latence 21-110s acceptable a cette frequence.

**Alternatives evaluees** :

| Option | Pro | Contra |
|---|---|---|
| Remplacer Bull par NIM Llama-405B | Diversite epistemique, qualite | Latence × 100 = 5 min/article |
| Ajouter 4eme agent NIM (debat 4-agents) | Plus de perspectives | × 1.33 tours + NIM lent = pire |
| **Migrer les taches batch + HQC opt-in** | Qualite gagnee la ou ca compte, latence ignoree | Demande discipline (ne pas activer en live) |
| Status quo (rien de NIM dans le debat) | Aucun travail | Manque l'opportunite d'utiliser des gros modeles |

**Consequences** :
- (+) Le debat live garde sa latence de ~20s/article (Cerebras + Groq + Mistral),
  scalable a 100+ articles/jour.
- (+) `agent_memoire` produit des memoires de meilleure qualite chaque
  nuit (Llama 405B sur LongBench bat le 70B sur les taches longues).
- (+) `HighQualityConsensus` agit comme un "second-opinion expert" sur les
  cas tendus, exactement la ou ca compte le plus.
- (+) Test live `HighQualityConsensus` (2026-04-26) : sur un signal Achat
  ambigu (consensus 0.55), Llama 405B a **confirme l'Achat mais releve la
  confiance a 0.65** avec un raisonnement nuance francais. Pas de rubber-stamp.
- (-) On n'utilise PAS les modeles NIM pour le live (perte d'opportunite
  apparente). C'est un choix delibere fonde sur les chiffres.
- (-) `agent_memoire` depend maintenant de NIM en 1ere intention (avec fallback
  Groq/Mistral/Cerebras). Si NIM down 1 nuit, on retombe sur la qualite
  precedente — pas de regression nette.

**Reference scientifique** :
- Du, Y. et al. (2023). "Improving Factuality and Reasoning in Language Models
  through Multiagent Debate." NeurIPS. (3-5 agents = optimum, au-dela gain marginal nul).
- Wang, Z. et al. (2023). "Self-Consistency Improves Chain-of-Thought Reasoning
  in Language Models." ICLR. (justification du HQC : second avis comme vote pondere).
- Meta AI (2024). "The Llama 3 Herd of Models." arXiv:2407.21783. (405B vs 70B
  benchmarks justifiant `agent_memoire` migration).

**Lecon meta** :
Cette ADR illustre l'importance de **mesurer empiriquement** avant de migrer.
L'intuition "modele plus gros = meilleur" etait correcte sur la qualite, mais
ignorait le facteur latence. Le bench live a transforme une "evidence intuitive"
en decision tracable. C'est exactement la rigueur que la couche Layer 7
(metriques formelles) cherche a apporter au reste du projet.

---

## Process pour ajouter une ADR

1. Copier la derniere ADR comme template, incrementer le numero.
2. Remplir : Contexte (probleme), Decision (choix), Alternatives evaluees
   (table comparative), Consequences (+/-).
3. Statut initial : `Propose`.
4. Apres revue (auto-revue ou peer) : passer a `Accepte`.
5. Ajouter une ligne au sommaire en haut du fichier.

**Regle d'or** : une ADR ne se modifie PAS apres acceptation. Une revision = une
nouvelle ADR qui declare *Supersedes ADR-NNN*.

---

## ADR-017 — Reintegration des modeles NIM rapides pour le debat principal

**Statut** : Accepte
**Date** : 2026-04-27
**Supersedes** : ADR-016 (partiellement)
**Contexte** :
L'ADR-016 stipulait de ne pas utiliser NIM pour le debat en direct car les gros modeles (Llama 405B) prenaient plus de 100s. Cependant, de nouveaux benchmarks sur des modeles NIM optimises (Nemotron-Mini-4B, Ministral-14B, Qwen3-Next-80B) montrent qu'ils peuvent repondre en 1.0s a 3.8s, ce qui est parfaitement compatible avec un pipeline en temps reel.

**Decision** :
Integrer ces modeles NIM legers et performants dans le debat :
- **Bull** : NIM Nemotron-Mini-4B (NVIDIA, 1.0s, IFEval=88.0)
- **Bear** : NIM Ministral-14B (Mistral, 2.3s, AIME=85)
- **Neutre** : NIM Qwen3-Next-80B (Alibaba, 3.8s, MMLU-Pro=80.6)
- **Consensus** : Groq Llama-3.3-70B (Meta, 3-5s) reste inchange.

**Consequences** :
- (+) Diversite epistemique maximale (NVIDIA + Mistral + Alibaba + Meta).
- (+) Latence maitrisee (1-4s par appel), rendant le pipeline scalable.
- (+) Exploitation optimale de l'API NVIDIA NIM pour le live et l'asynchrone.
- (-) Dependance accrue envers NVIDIA NIM (gere par la chaine de fallback existante).

---

## ADR-018 — Validation de la diversité épistémique et Monkey-Patch Fallback Résilient (AutoGen v0.4)

**Statut** : Accepte
**Date** : 2026-04-28
**Supersedes** : N/A
**Contexte** :
Le pipeline utilisait `Ministral-14B` (via NVIDIA NIM) comme agent Baissier. Lors d'un benchmark de raisonnement financier en conditions réelles, ce modèle a échoué à produire un JSON/XML valide en raison d'une taille insuffisante face à un prompt complexe (token collapse). De plus, l'API Groq a atteint sa limite quotidienne gratuite (100k tokens), provoquant un crash `RateLimitError (429)` de tout le système en production. Autogen v0.4 (`OpenAIChatCompletionClient`) ne gère pas nativement de fallback transparent intra-agent avec une liste de `model_clients`.

**Decision** :
1. Remplacer l'agent Baissier défaillant par `openai/gpt-oss-120b` (via Groq), une nouvelle famille de modèles testée avec succès (latence 1.02s, raisonnement implacable, strict respect XML).
2. Ajouter Mistral Large (`mistral-large-latest` via MISTRAL API) en Fallback direct sur ce poste.
3. Implémenter un Monkey-Patch asynchrone dynamique dans `agent_debat.py` : au lieu de crasher si le premier provider renvoie une erreur 429, la méthode `create()` intercepte l'erreur et bascule silencieusement sur le provider de secours (Fallback Chain), assurant l'indestructibilité du pipeline.

**Alternatives evaluees** :
- **Utiliser les modèles Phi-4 (Microsoft)** : `microsoft/phi-4-multimodal-instruct` a été testé (1.73s, excellente qualité). Cependant, l'API NIM a renvoyé des erreurs 400 DEGRADED lors des stress-tests (hébergement instable). Option écartée temporairement.
- **Désactiver Groq (Hardcode)** : Simple mais on perd la latence de 1s de Groq pour demain.
- **Monkey-Patching (Choisi)** : Élégant, abstrait la complexité à Autogen, et relance automatiquement Groq quand la limite quotidienne se reset, sans intervention humaine.

**Consequences** :
- (+) Résilience totale : un quota API (Groq) dépassé ou une API instable n'interrompt plus la boucle de Paper Trading.
- (+) Diversité préservée : L'introduction du modèle `gpt-oss-120b` apporte une nouvelle perspective critique.
- (-) Complexité technique : Le monkey-patching requiert de bien tester les flux asynchrones, validé par les 375 tests pytest.

---

## ADR-019 — Agent Portfolio Manager (Agrégation Pré-Ouverture)

**Statut** : Accepte
**Date** : 2026-04-28
**Supersedes** : N/A
**Contexte** :
Dans l'architecture initiale, l'agent_pipeline exécutait des ordres séquentiels de paper-trading au fil de l'eau, même lorsque le marché était fermé (la nuit). De plus, si deux news nocturnes donnaient des signaux contradictoires (un achat à 23h et une vente à 11h), l'algorithme risquait de passer deux ordres opposés à l'ouverture du marché (15h30), payant le spread et s'annulant. L'exécution à 15h30 de 150 articles d'un coup créait également un goulot d'étranglement (Rate Limits).

**Decision** :
1. **Traitement asynchrone nocturne** : `run_paper_trading.py` lance désormais l'analyse multi-agents (`agent_pipeline.py`) en continu, même lorsque le marché est fermé, mais les ordres ne sont plus émis. Les signaux sont juste stockés en base (état PENDING).
2. **Création du Portfolio Manager** : Un nouveau script `src/execution/portfolio_manager.py` se réveille une fois par jour entre 15h00 et 15h30.
3. **Agrégation LLM Top-Down** : L'agent Portfolio Manager aspire tous les signaux de la nuit pour un ticker, et reçoit le contexte macro-économique (S&P 500, VIX, Régime). Le modèle Llama-3.1-70B (avec fallback asynchrone) arbitre le conflit et émet l'ordre net final.

**Alternatives evaluees** :
- **Agrégation mathématique simple** : Faire la somme des scores de conviction de la nuit. Écartée car elle ne permet pas de comprendre la nuance (ex: une énorme news macro l'emporte sur 3 petites news positives de revenus).
- **Exécution directe** : Envoyer les ordres à Alpaca pendant la nuit pour les mettre en file d'attente. Écarté car Alpaca exécuterait tous les ordres contradictoires à l'ouverture l'un après l'autre.

**Consequences** :
- (+) Élimination du goulot d'étranglement de 15h30 (les analyses sont étalées sur les 15h de fermeture du marché).
- (+) Réduction drastique des frais de transaction simulés (un seul ordre net envoyé).
- (+) Prise de décision plus intelligente (intégration de la météo macro globale à la lecture de plusieurs news isolées).
