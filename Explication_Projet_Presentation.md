# 📊 Présentation du Projet : Pipeline d'Analyse Financière par IA

Ce document a été conçu spécialement pour t'aider à **comprendre** et **présenter** le projet très simplement. Il est rédigé sans jargon compliqué, avec des exemples clairs.

---

## 1. 🌟 Le But du Projet (Le "Pourquoi")

Imagine que tu es un trader ou un banquier. Chaque jour, des milliers d'articles financiers sont publiés. C'est impossible tout lire ! 
Le but de ce projet est de **créer un robot (un "pipeline") qui lit les articles d'actualité financière à notre place et nous donne deux informations cruciales :**

1. **La Polarité (Le Sentiment) :** Est-ce que l'article est Positif (bonne nouvelle), Négatif (mauvaise nouvelle), ou Neutre ?
2. **L'Incertitude :** Est-ce que le texte est flou, hésitant, rempli de doutes (ex: "il se pourrait que les marchés s'effondrent si...") ou bien est-ce qu'il est très factuel et sûr ?

**L'intérêt ?** Pouvoir prendre des décisions rapides sans avoir à lire tous les journaux.

---

## 2. 🏗️ L'Architecture Générale (Comment ça marche ?)

Pour que notre système soit propre et facile à maintenir, je l'ai récemment **restructuré**. Au lieu d'avoir un énorme fichier brouillon qui fait tout, nous avons maintenant divisé le projet en **trois étapes distinctes**, un peu comme une usine avec différents ateliers :

### 🏭 Étape 1 : Le Récolteur d'Articles (`news_pipeline.py`)
C'est le journaliste. Son seul travail est d'aller chercher (scraper) des articles sur internet à propos de certaines entreprises (ex: Apple, Google) et de les ranger dans notre base de données (`news_database.db`).

### 🧠 Étape 2 : Les "Cerveaux" ou Agents IA
Ce sont nos spécialistes, chacun dans son domaine. 
- **`polarity_agent.py` (L'Expert en Sentiment) :** Il prend un texte et dit "+1" (Positif), "-1" (Négatif) ou "0" (Neutre). Pour ça, il utilise une intelligence artificielle appelée *FinBERT*, qui a déjà lu des millions de documents financiers dans sa vie.
- **`uncertainty_agent.py` (L'Expert en Doute) :** Lui, il cherche le niveau d'incertitude sur une échelle de 0 à 1 (0 = aucuns doutes, 1 = incertitude totale). Comme il n'existait aucun modèle prêt à l'emploi pour ça, **nous avons entraîné notre propre modèle IA !**

### ⚙️ Étape 3 : L'Orchestrateur (`processing_pipeline.py`)
C'est le chef d'orchestre. Il prend les articles récoltés à l'Étape 1, les donne à lire aux deux cerveaux de l'Étape 2, et enregistre leurs notes (scores) dans une nouvelle table de résultats (`article_scores`) dans notre base de données.

---

## 3. 🛠️ Ce qui a été changé ou "Ce que j'ai fait" (À dire lors de la présentation)

Lors de la présentation, si on te demande ce qui a été fait ou amélioré récemment, voici comment l'expliquer simplement :

* **Séparation des responsabilités (Refactoring) :** 
  *"Avant, tout était mélangé. Le code qui allait chercher les articles essayait aussi de les analyser. J'ai séparé l'architecture proprement. Maintenant on a des fichiers dédiés pour chaque tâche : un pour récupérer les news, un chef d'orchestre (`processing_pipeline`), et surtout, j'ai créé des « Agents » autonomes (`polarity_agent` et `uncertainty_agent`). C'est beaucoup plus professionnel et évolutif."*

* **Création d'un Agent d'Incertitude (Innovation) :**
  *"Puisqu'il n'y avait pas de modèle qui prédisait l'incertitude avec un score de 0 à 1, j'ai décidé de créer le nôtre. J'ai utilisé une technique appelée le 'Fine-tuning avec LoRA'. En gros, on a pris une IA existante (FinBERT) et on lui a appris une nouvelle compétence : compter et repérer les mots de doute (grâce à un dictionnaire financier officiel nommé Loughran-McDonald). Ensuite on l'a entraînée (entraîner le modèle) pour qu'elle puisse donner ce score toute seule."*

* **Correction et finalisation du flux de données :**
  *"J'ai aussi réparé le chef d'orchestre (`processing_pipeline.py`) pour qu'il sauvegarde les deux scores correctement dans une seconde table de notre base de données, sans écraser les articles d'origine."*

---

## 4. 📈 Analyse des Résultats (Le traitement a-t-il bien marché ?)

J'ai testé notre base de données à la fin du traitement, et voici le bilan : **Les résultats sont logiques et très encourageants !**

### Pour la Polarité (Le Sentiment) : ✅ Excellent
Sur nos articles actuels, le modèle a classé de manière très équilibrée :
* 🔴 4 articles négatifs (-1)
* 🟢 4 articles positifs (1)
* ⚪ 2 articles neutres (0)
**Explication pour le jury :** Le modèle a une très bonne compréhension, il ne met pas tout au hasard. Il détecte vraiment quand les nouvelles sont bonnes ou mauvaises.

### Pour l'Incertitude : 🚧 Prometteur mais en phase d'apprentissage
L'IA nous donne bien un score continu entre 0 et 1.
Actuellement, les scores trouvés vont de **0.40 (minimum) à 0.52 (maximum)** avec une moyenne autour de **0.47**.
**Est-ce bien ? Oui et Non.**
* **Oui, car ça marche :** L'IA ne crash pas et donne des valeurs logiques autour du point médian (0.50).
* **Mais c'est "timide" :** L'écart entre le plus petit et le plus grand score est très faible. 
**Comment l'expliquer au jury :** *"L'agent d'incertitude vient tout juste d'être créé. On l'a entraîné sur un tout petit échantillon de textes (13 exemples générés) juste pour valider que la mécanique fonctionne (Proof of Concept). C'est pourquoi ses prédictions sont encore un peu 'timides' et restent autour de 0.50 au lieu d'aller de 0 à 1. La prochaine étape logique est simplement de le nourrir avec de **vrais articles en masse** pour relancer un entraînement, et là il deviendra hyper précis."*

---

## 🎯 Résumé éclair pour conclure ta présentation :
👉 **"Nous avons un pipeline fonctionnel de bout-en-bout : il récupère les news en continu, utilise deux intelligences artificielles distinctes (dont une que nous avons fine-tunée nous-mêmes de zéro) pour extraire le sentiment et le doute, et sauvegarde les résultats proprement pour permettre aux traders de prendre leurs décisions."**
