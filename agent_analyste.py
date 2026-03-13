import logging
from typing import Literal

from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.func import task, entrypoint

logger = logging.getLogger(__name__)

class DecisionFinanciere(BaseModel):
    ticker: str = Field(description="Le symbole boursier de l'entreprise (ex: AAPL)")
    analogie_historique: str = Field(description="Le précédent historique comparable identifié (entreprise, date, impact cours)")
    justification: str = Field(description="Le raisonnement étape par étape reliant l'analogie à la situation actuelle")
    signal: Literal["Achat", "Vente", "Neutre"] = Field(description="La décision finale")

llm = ChatOllama(
    model="deepseek-r1:7b",
    temperature=0.0,
    format="json",
    num_gpu=0
)

analyste_structure = llm.with_structured_output(DecisionFinanciere)

prompt_ad_fcot = ChatPromptTemplate.from_messages([
    ("system", """Tu es un analyste financier quantitatif expert.
Ta méthode est l'AD-FCoT (Analogy-Driven Financial Chain-of-Thought), appliquée en 3 phases strictes :

<phase_1_recherche>
Identifie UN précédent historique précis : entreprise comparable, date, type d'événement, impact mesuré sur le cours.
</phase_1_recherche>

<phase_2_comparaison>
Compare point par point l'événement passé à la situation actuelle :
- Type d'événement identique ou similaire ?
- Secteur et taille d'entreprise comparables ?
- Réaction initiale du marché dans le cas historique ?
</phase_2_comparaison>

<phase_3_decision>
Conclus avec un signal clair basé UNIQUEMENT sur cette logique.
Le signal doit être exactement l'un de ces trois mots : Achat, Vente, Neutre.
</phase_3_decision>

Tu dois impérativement respecter le schéma JSON demandé."""),
    ("human", "L'entreprise {ticker} fait face à l'actualité suivante :\n\n{actualite}\n\nApplique ta méthode AD-FCoT.")
])

chaine_analyse = prompt_ad_fcot | analyste_structure

@task
def analyser_impact_financier(texte_article: str, ticker_symbol: str) -> dict:
    """Appelle le modèle via Ollama et retourne la décision structurée."""
    logger.info("[Analyste] Analyse AD-FCoT pour %s...", ticker_symbol)
    resultat = chaine_analyse.invoke({
        "ticker": ticker_symbol,
        "actualite": texte_article
    })
    return resultat.model_dump()

@entrypoint()
def workflow_analyser_actualite(inputs: dict) -> dict:
    """
    Point d'entrée LangGraph.
    Attend un dict : {"texte_article": str, "ticker_symbol": str}
    """
    return analyser_impact_financier(inputs["texte_article"], inputs["ticker_symbol"]).result()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    test_news = "Le PDG historique d'Apple démissionne soudainement suite à un désaccord avec le conseil d'administration sur la stratégie d'IA."
    decision = workflow_analyser_actualite.invoke({"texte_article": test_news, "ticker_symbol": "AAPL"})
    print("\nRésultat structuré obtenu :")
    print(decision)