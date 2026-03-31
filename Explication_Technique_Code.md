# 🧠 Plongée Technique : Comment l'Agent d'Incertitude a-t-il appris à lire ? (Explication du Code)

Tu as vu la vue d'ensemble, super ! Maintenant, si quelqu'un te pose une question pointue : *"Mais comment avez-vous entraîné (fine-tuné) cette IA concrètement ?"*, voici l'explication pas à pas, garantie sans prise de tête.

---

## 1. Le Point de Départ : L'Étudiant Doué (FinBERT)
Le code commence avec une IA qui s'appelle **FinBERT**. 
* **C'est quoi ?** C'est un modèle d'Intelligence Artificielle créé par Google (basé sur BERT) qui a été forcé à lire des millions de documents financiers chiants (des rapports d'entreprises, des bilans comptables). 
* **Le problème :** Cet "étudiant" connaît tout le vocabulaire financier, il sait si une phrase est positive ou négative, mais **personne ne lui a jamais appris à donner une note de 0 à 1 sur le niveau de doute (l'incertitude)**.
* **Notre but :** Lui apprendre cette compétence spécifique.

---

## 2. Le Dictionnaire Officiel (Loughran-McDonald)
Pour apprendre à notre IA à repérer le doute, on a besoin d'une règle stricte. 
Dans le code, on télécharge ce qu'on appelle le **Lexique Loughran-McDonald**. 
* **Pourquoi ?** C'est un dictionnaire créé par des chercheurs en finance. Il contient la liste exacte (environ 300 mots) des mots qui expriment le doute ou le risque en bourse en anglais (ex: *fluctuate, maybe, risk, volatile, uncertain*...).
* **Dans le code :** La fonction `download_lm_uncertainty_lexicon()` récupère cette liste de mots.

---

## 3. L'Astuce Magique : Le "Weak Labeling" (L'étiquetage automatique)
Pour entraîner une IA, il faut normalement des humains qui lisent 10 000 textes et notent "Incertitude : 0.8" à la main. C'est long et cher. Nous avons utilisé une astuce de hacker appelée le **Weak Labeling**.
* **Comment on a fait ?** On écrit une fonction mathématique très simple (`compute_uncertainty_score`) :
   1. On compte tous les mots du texte.
   2. On regarde combien de ces mots font partie de notre dictionnaire du doute.
   3. On fait le ratio (ex: 2 mots de doute / 100 mots = 2%).
   4. On transforme ce pourcentage en une note entre 0 et 1.
* **Le résultat :** L'ordinateur "corrige" lui-même les textes d'entraînement et attribue une note (un "label"). L'IA FinBERT va donc pouvoir s'entraîner en s'exerçant sur ces corrections !

---

## 4. L'Entraînement "Chirurgical" : LoRA (Low-Rank Adaptation)
C'est la partie la plus technique du code, mais la plus brillante. 
FinBERT a environ **110 Millions de neurones** (paramètres). Ré-entraîner 110 millions de neurones pour une si petite tâche prendrait des semaines et des ordinateurs surpuissants.

* **La solution dans notre code : on utilise `LoRA`.**
* **C'est quoi LoRA ?** Imagine que le cerveau de l'IA (FinBERT) est un énorme livre figé. Au lieu de gommer et réécrire le livre (trop long), **LoRA ajoute de petits post-its** sur certaines pages clés (les matrices d'attention) pour modifier la façon de penser du livre.
* **Le gain :** Grâce à LoRA, au lieu d'entraîner 110 millions de neurones, le code n'en entraîne que **26 000** (soit 0,2%). C'est ultra-rapide et ça peut se faire sur un petit ordinateur de bureau !

---

## 5. Le Score Final (La "Tête de Régression" et la Sigmoïde)
Une fois que FinBERT a lu le texte grâce à ses post-its LoRA, il doit recracher un chiffre.
* **La Tête de Régression :** Au lieu d'avoir 3 portes de sortie (Positif, Négatif, Neutre), on a détruit ces portes dans le code pour n'en construire qu'une seule (`num_labels=1`). L'IA crache donc une seule valeur.
* **La Sigmoïde (`torch.sigmoid`) :** Mais l'IA pourrait recracher le chiffre "150" ou "-12". Pour garantir que le score sera **toujours compris strictement entre 0 et 1** (0 = sûr, 1 = incertain), on passe le chiffre dans un tuyau mathématique appelé la Sigmoïde.

---

### Résumé pour le jury (en 1 phrase) :
> *"Pour l'agent d'incertitude, nous avons pris un modèle de langage financier existant (FinBERT) et nous l'avons fine-tuné avec la méthode d'optimisation LoRA, en utilisant un jeu de données que nous avons auto-étiqueté mathématiquement grâce au lexique universitaire de référence Loughran-McDonald."* (Effet garanti ! 😎)
