\# Benchmarks du projet



Ce dossier contient les scripts utilisés pour comparer les différentes méthodes du projet.



\## SentFin v2



Le dossier `sentfin` contient les scripts de benchmark sur SentFin v2.

Ce benchmark compare les méthodes sur une tâche commune de sentiment financier :

positive / neutral / negative.



\## Benchmark maison



Le dossier `benchmark\_maison` contient le benchmark construit à partir des articles récupérés par notre pipeline.

Chaque article est annoté avec :

\- relevance\_gold : 1 si l'article est pertinent pour le ticker, 0 sinon

\- sentiment\_gold : positive / neutral / negative



Les scripts ramènent les sorties des différentes branches vers un format commun :

\- pred\_relevance

\- pred\_sentiment



Attention : certains modules complets des branches ne sont pas évalués ici.

Par exemple, pour Samuel on évalue surtout le PolarityAgent.

Pour POC-Filtrage-Agents, on évalue la partie locale DistilRoBERTa, pas la partie complète avec API/ABSA.

