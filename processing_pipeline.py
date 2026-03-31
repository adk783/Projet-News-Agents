import sqlite3
import os
from polarity_agent import PolarityAgent
from uncertainty_agent import UncertaintyAgent

def run_processing_pipeline():
    
    # 1) Chargement de Modele
    print("Initialisation des agents...")
    polarity_agent = PolarityAgent()
    
    print("Chargement de l'agent d'incertitude...")
    uncertainty_agent = None
    if os.path.exists("./uncertainty_model"):
        uncertainty_agent = UncertaintyAgent(model_path="./uncertainty_model")
    else:
        print("ATTENTION: Modèle uncertainty_model introuvable. Exécutez d'abord l'entraînement.")

    # 2) Connexion à la base source
    conn = sqlite3.connect("news_database.db")
    cursor = conn.cursor()

    print("Connexion à la base réussie.")

    # 3) Créer la table de sortie si elle n'existe pas
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS article_scores (
            url TEXT PRIMARY KEY,
            polarity INTEGER,
            polarity_conf REAL,
            uncertainty REAL
        )
    """)
    conn.commit()

    print("Table article_scores prête.")

    # 4) Lire les articles source
    cursor.execute("SELECT url, title, content FROM articles")
    articles = cursor.fetchall()

    print(f"Nombre d'articles récupérés : {len(articles)}")

    # 5) Scorer chaque article
    for url, title, content in articles:
        text_to_analyze = content if content else title

        # On coupe si le texte est trop long pour éviter les soucis
        text_to_analyze = text_to_analyze[:1500]

        #-------------Agent----------Polarity----------------
        polarity, conf_pol, label = polarity_agent.predict(text_to_analyze)
        #------------------------------------------------------

        #-------------Agent---------Uncertainty----------------
        uncertainty = 0.0
        if uncertainty_agent:
            uncertainty = uncertainty_agent.predict(text_to_analyze)
        #------------------------------------------------------

        #Insertion dans la table
        cursor.execute("""
            INSERT OR REPLACE INTO article_scores (url, polarity, polarity_conf, uncertainty)
            VALUES (?, ?, ?, ?)
        """, (url, polarity, conf_pol, uncertainty))

        print("\n------------------------------")
        print(f"Article traité : {title}")
        print(f"  label = {label}, polarity = {polarity}, confidence = {conf_pol:.3f}")
        print(f"  uncertainty = {uncertainty:.4f}")

    conn.commit()
    conn.close()

    print("Scores générés et enregistrés.")
    print("Fin du processing.")


if __name__ == "__main__":
    run_processing_pipeline()