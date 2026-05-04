# Projet News Agents — Version finale

Synthèse de toutes les branches du projet en une seule architecture cohérente.

## Architecture (5 étages)

```
1. COLLECTE (news_pipeline.py)
   yfinance.news → newspaper3k → fallback trafilatura → SQLite

2. FILTRAGE cascade (agent_filtrage.py)
   2a. Pré-filtre keywords (yfinance.info)            ─┐
   2b. Confirmation LLM Ollama llama3.2:3b YES/NO     ─┘ bypass si match titre

3. PROCESSING multi-features (processing_pipeline.py)
   ├─ PolarityAgent       FinBERT
   ├─ UncertaintyAgent    heuristique L&M
   ├─ LitigiousAgent      lexique L&M (TF-IDF+Ridge si modèle dispo)
   ├─ FundamentalAgent    lexique fundamentals (TF-IDF+Ridge si dispo)
   └─ SentimentAgent LLM  Ollama phi4-mini JSON
   + features dérivées : risk_adjusted_sentiment,
                         headline_conviction,
                         fundamental_impact

4. AGRÉGATION par ticker (agent_agregateur.py)
   fenêtre 48h, moyenne pondérée par |score|, neutrals exclus,
   confidence ∈ {insufficient, low, normal, high}

5. ORCHESTRATION (orchestrateur.py + status_manager.py + dashboard.py)
   - boucle articles is_analyzed=0 → filtrage → processing → agrégation
   - pipeline_status.json pour suivi temps réel
   - dashboard Flask Chart.js (templates/dashboard.html)
```

## Provenance des idées

| Étage | Provenance |
|---|---|
| 1 — Collecte yfinance + newspaper3k + trafilatura | branche `main` |
| 2a — Pré-filtre keywords (`build_keywords`, `is_relevant`) | branche `filtrage-keywords` |
| 2b — Filtre LLM YES/NO few-shot llama3.2:3b | branche `Antoinev2` |
| 3 — Séparation news/processing | branche `poc-processing-lorenzo` |
| 3 — PolarityAgent FinBERT | branches `poc-processing-lorenzo` + `samuel` |
| 3 — Uncertainty / Litigious / Fundamental + features dérivées | branche `samuel` |
| 3 — SentimentAgent LLM JSON structuré | branche `Antoinev2` |
| 4 — Agrégation 48h pondérée \|score\|, neutrals exclus, confidence | branche `Antoinev2` |
| 5 — Orchestrateur + status_manager + dashboard Flask | branche `Antoinev2` |
| 5 — Logging propre avec handlers | branche `POC-Filtrage-Agents` |

## Lancement

### Test rapide via dashboard (recommandé)

**Checklist avant de cliquer**
- [ ] `pip install -r requirements.txt`
- [ ] Ollama tourne (`ollama serve`)
- [ ] Les 2 modèles sont tirés : `ollama pull llama3.2:3b` et `ollama pull phi4-mini`

**Test**
```bash
python dashboard.py
```
→ ouvre http://127.0.0.1:5000

Dans l'UI :
1. Le champ tickers est pré-rempli `AAPL, MSFT, TSLA`
2. Clic **▶ Lancer le Sourcing** → news_pipeline.py en arrière-plan, barre de progression "Scan : AAPL"
3. Quand terminé, clic **▶ Lancer l'Orchestrateur** → vérification Ollama → "Chargement des modèles…" → analyse article par article
4. Cards des tickers, camembert sentiment et liste d'articles se remplissent en live

### Lancement manuel (sans dashboard)

```bash
ollama serve
ollama pull llama3.2:3b phi4-mini
pip install -r requirements.txt

python news_pipeline.py --tickers AAPL MSFT GOOGL
python orchestrateur.py
python dashboard.py            # juste pour visualiser
```

### Si Ollama est down ou un modèle manque

L'orchestrateur s'arrête avec un message explicite :
```
[OLLAMA] Ollama indisponible sur http://localhost:11434 (...)
```
ou
```
[OLLAMA] Modèle 'phi4-mini' non trouvé dans Ollama (...)
```
L'erreur est aussi écrite dans `pipeline_status.json` (clé `error`) — visible dans le dashboard.

## Schéma SQLite

- `articles` : `url, ticker, sector, industry, date_utc, title, content, json_brut, is_analyzed`
- `article_scores` : `url, ticker, polarity, polarity_conf, uncertainty, legal_risk, fundamental_strength, sentiment, score, reasoning, risk_adjusted_sentiment, headline_conviction, fundamental_impact, analyzed_at`
- `ticker_scores` : `id, ticker, score_global, sentiment_global, nb_articles, nb_neutral, confidence, window_start, window_end, calculated_at`
