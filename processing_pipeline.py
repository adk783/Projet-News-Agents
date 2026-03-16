import sqlite3
from transformers import pipeline
from uncertainty_agent import compute_uncertainty


def run_processing_pipeline():
    
    # 1) Chargement de Modele
    # FinBERT
    sentiment_model = pipeline(
        "text-classification",
        model="ProsusAI/finbert"
    )
    # Agent-Uncertainty
    #.....

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

        #-------------FinBert----------Polarity----------------
        result = sentiment_model(text_to_analyze)[0]

        label = result["label"]
        score = result["score"]

        # Conversion label -> polarity
        if label == "positive":
            polarity = 1
        elif label == "negative":
            polarity = -1
        else:
            polarity = 0

        conf_pol = score
        #------------------------------------------------------

        #-------------Agent---------Uncertainty----------------
        uncertainty = compute_uncertainty(text_to_analyze)
        #------------------------------------------------------

        #Insertion dans la table
        cursor.execute("""
            INSERT OR REPLACE INTO article_scores (url, polarity, polarity_conf, uncertainty)
            VALUES (?, ?, ?, ?)
        """, (url, polarity, conf_pol, uncertainty))

        print("\n------------------------------")
        print(f"Article traité : {title}")
        print(f"  label = {label}, polarity = {polarity}, confidence = {conf_pol:.3f}")

    conn.commit()
    conn.close()

    print("Scores générés et enregistrés.")
    print("Fin du processing.")


if __name__ == "__main__":
    run_processing_pipeline()