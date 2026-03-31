# BRIEF POUR POWERPOINT — Pipeline d'Analyse Financière Multi-Agents par IA

> Ce document est un brief destiné à une IA pour générer un PowerPoint de présentation de projet.
> Merci de créer un PowerPoint professionnel, moderne et visuel (fond sombre, accents colorés) à partir des informations ci-dessous. Le ton doit être sérieux mais accessible. Utilise des icônes, schémas et mises en page dynamiques.

---

## SLIDE 1 — Page de titre
- **Titre :** Pipeline d'Analyse Financière Multi-Agents par IA
- **Sous-titre :** Extraction automatique du sentiment et de l'incertitude dans les articles financiers
- **Contexte :** Projet universitaire — Groupe : [NOM DU GROUPE]

---

## SLIDE 2 — Le Problème
- Chaque jour, des milliers d'articles financiers sont publiés (Reuters, Bloomberg, Yahoo Finance, etc.)
- Un trader ou analyste ne peut PAS tout lire manuellement
- **Question clé :** Comment automatiser l'extraction d'informations utiles pour la prise de décision ?
- Deux informations cruciales à extraire :
  1. **Le Sentiment** (Positif / Négatif / Neutre) → L'article est-il une bonne ou mauvaise nouvelle ?
  2. **L'Incertitude** (score 0 à 1) → L'article est-il factuel et sûr, ou flou et spéculatif ?

---

## SLIDE 3 — Notre Solution : Architecture Multi-Agents
- Schéma de l'architecture en 3 étapes (pipeline) :

```
Internet (Articles)
      │
      ▼
┌─────────────────────┐
│  news_pipeline.py   │  ← ÉTAPE 1 : Scraping
│  (Le Récolteur)     │     Récupère les articles via API
└─────────┬───────────┘
          │ Stockage SQLite
          ▼
┌─────────────────────────────┐
│   processing_pipeline.py    │  ← ÉTAPE 3 : Orchestrateur
│   (Le Chef d'Orchestre)     │     Coordonne les 2 agents
└──────┬──────────┬───────────┘
       │          │
       ▼          ▼
┌────────────┐ ┌──────────────────┐
│ polarity   │ │ uncertainty      │  ← ÉTAPE 2 : Les Agents IA
│ _agent.py  │ │ _agent.py        │
│ (FinBERT)  │ │ (FinBERT + LoRA) │
└────────────┘ └──────────────────┘
       │          │
       ▼          ▼
┌─────────────────────────────┐
│     Base de données         │
│  Table : article_scores     │
│  (polarity, uncertainty)    │
└─────────────────────────────┘
```

---

## SLIDE 4 — Agent 1 : L'Expert en Sentiment (polarity_agent.py)
- **Modèle utilisé :** FinBERT (ProsusAI/finbert)
  - Pré-entraîné sur des millions de documents financiers
  - Modèle de référence en NLP financier
- **Fonctionnement :** Prend un texte → Classe en Positif (+1), Négatif (-1) ou Neutre (0)
- **Score de confiance** associé à chaque prédiction (0 à 1)
- Aucun entraînement supplémentaire nécessaire → Modèle prêt à l'emploi

---

## SLIDE 5 — Agent 2 : L'Expert en Incertitude (uncertainty_agent.py)
- **Problème :** Il n'existe aucun modèle prêt à l'emploi pour prédire l'incertitude financière avec un score continu de 0 à 1
- **Notre innovation :** Nous avons créé et entraîné notre propre modèle !
- **Méthode en 3 étapes :**
  1. **Lexique Loughran-McDonald** : Dictionnaire académique de référence (~247 mots d'incertitude financière : *volatile, uncertain, risk, speculate, doubt...*)
  2. **Weak Labeling** : Étiquetage automatique des articles en comptant le ratio de mots d'incertitude → Score cible entre 0 et 1
  3. **Fine-tuning LoRA** : On prend FinBERT et on ajoute de petits adaptateurs (LoRA) sur les couches d'attention pour lui apprendre cette nouvelle compétence en n'entraînant que 0.2% des paramètres

---

## SLIDE 6 — Focus Technique : LoRA (Low-Rank Adaptation)
- FinBERT a ~110 millions de paramètres
- Ré-entraîner tout le modèle serait trop coûteux
- **LoRA** : On "gèle" le modèle et on ajoute seulement de petites matrices entraînables sur les couches d'attention (Query et Value)
- Résultat : Seuls ~150 000 paramètres entraînés (0.14% du total)
- **Avantage :** Entraînement rapide (~30 secondes sur GPU), performances conservées
- Configuration : rang=16, alpha=32, dropout=0.05, 20 époques, learning rate=5e-5, cosine scheduler

---

## SLIDE 7 — Données d'Entraînement
- **94 échantillons** au total :
  - 67 vrais articles financiers scrappés (Yahoo Finance, Reuters, etc.)
  - 27 exemples synthétiques soigneusement rédigés couvrant haute, moyenne et basse incertitude
- **Distribution des labels :**
  - Score min : 0.00 (textes factuels, chiffres précis)
  - Score max : 1.00 (textes très spéculatifs, pleins de doute)
  - Score moyen : 0.27
- **Validation :** Split 80/20 train/eval, eval loss finale = 0.07

---

## SLIDE 8 — Résultats : Analyse de Sentiment
- **67 articles analysés automatiquement**
- Répartition des sentiments :
  - 🔴 Négatif (-1) : 21 articles (31%)
  - ⚪ Neutre (0) : 29 articles (43%)
  - 🟢 Positif (+1) : 17 articles (26%)
- Confiance moyenne élevée (>0.80)
- **Exemples concrets :**
  - ✅ "Apple Stock Rises as iPhone Upgrade Boom" → Positif (conf: 0.954)
  - ✅ "Dow Jones Futures Waver After Trump Threats" → Négatif (conf: 0.925)
  - ✅ "Musk says SpaceX to build chip factories" → Neutre (conf: 0.915)

---

## SLIDE 9 — Résultats : Scores d'Incertitude
- **Scores variés de 0.00 à 0.57** — le modèle distingue clairement les articles factuels des articles spéculatifs
- **Exemples concrets :**

| Article | Score | Interprétation |
|---------|-------|----------------|
| "SoundHound's Enterprise AI Momentum" | 0.00 🟢 | Très factuel |
| "KLAC vs. Advanced Energy: AI Stock Buy?" | 0.04 🟢 | Factuel |
| "Elon Musk's Terafab bet: what it means" | 0.18 🟡 | Légère incertitude |
| "Microsoft Stock Rises Despite Cloud Concerns" | 0.52 🟠 | Incertain |
| "Alibaba Q3 Disappoint: Hold or Fold?" | 0.56 🔴 | Forte incertitude |
| "Big Tech's Cause for Hope: Link Broken" | 0.57 🔴 | Forte incertitude |

---

## SLIDE 10 — Stack Technique
- **Langage :** Python 3
- **IA / Deep Learning :** PyTorch, HuggingFace Transformers, PEFT (LoRA)
- **Modèle de base :** ProsusAI/FinBERT
- **NLP :** Tokenizer BERT, lexique Loughran-McDonald
- **Base de données :** SQLite
- **Scraping :** API NewsAPI / RSS + Trafilatura (fallback)
- **GPU :** NVIDIA RTX 4080 (accélération CUDA, FP16)
- **Versioning :** Git + GitHub (branches par développeur)

---

## SLIDE 11 — Ce qui a été fait (Contributions personnelles)
1. **Refactoring de l'architecture** : Séparation du code monolithique en modules spécialisés (agents autonomes + orchestrateur)
2. **Création de l'agent d'incertitude de A à Z** : Conception, weak labeling, fine-tuning LoRA, inférence
3. **Séparation de l'agent de polarité** dans un module dédié, réutilisable
4. **Orchestrateur de traitement** : Pipeline qui coordonne les deux agents et sauvegarde tous les résultats dans une table dédiée
5. **Rate limiting et robustesse** : Gestion des timeouts, fallback scraper, validation du contenu minimum

---

## SLIDE 12 — Améliorations Futures
- **Plus de données** : Scraper des milliers d'articles pour ré-entraîner le modèle d'incertitude avec plus de précision
- **Nouveaux agents** : Agent de détection de sujets (topic modeling), agent de résumé automatique
- **Dashboard temps réel** : Interface web pour visualiser les scores en live
- **Alertes automatiques** : Notifications quand un article très incertain ou très négatif est détecté
- **Backtesting** : Corréler les scores de sentiment/incertitude avec les mouvements réels du marché

---

## SLIDE 13 — Conclusion
- ✅ Pipeline **fonctionnel de bout en bout** : scraping → analyse IA → stockage
- ✅ **Deux agents IA spécialisés**, dont un fine-tuné from scratch avec LoRA
- ✅ Résultats **cohérents et exploitables** sur 67 articles réels
- ✅ Architecture **modulaire et évolutive** (ajout facile de nouveaux agents)
- 🎯 **Phrase clé :** *"Nous avons construit un système multi-agents d'intelligence artificielle capable d'analyser automatiquement l'actualité financière pour en extraire le sentiment et le niveau d'incertitude, en utilisant des techniques avancées de NLP et de fine-tuning (LoRA) sur un modèle pré-entraîné."*
