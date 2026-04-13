import os
import sqlite3

from fundamental_strength_agent import FundamentalStrengthAgent
from litigious_agent import LitigiousAgent
from polarity_agent import PolarityAgent
from uncertainty_agent import UncertaintyAgent


def ensure_column(cursor, table_name, column_name, column_definition):
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = [row[1] for row in cursor.fetchall()]
    if column_name not in existing_columns:
        cursor.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )


def run_processing_pipeline():
    print("Initialisation des agents...")
    polarity_agent = PolarityAgent()

    print("Chargement de l'agent d'incertitude...")
    uncertainty_agent = None
    if os.path.exists("./uncertainty_model"):
        uncertainty_agent = UncertaintyAgent(model_path="./uncertainty_model")
    else:
        print("ATTENTION: Modele uncertainty_model introuvable. Executez d'abord l'entrainement.")

    print("Chargement de l'agent litigious...")
    litigious_agent = LitigiousAgent(
        model_path="./litigious_model",
        fallback_to_heuristic=True,
    )

    print("Chargement de l'agent fundamental strength...")
    fundamental_strength_agent = FundamentalStrengthAgent(
        model_path="./fundamental_strength_model",
        fallback_to_heuristic=True,
    )

    conn = sqlite3.connect("news_database.db")
    cursor = conn.cursor()
    print("Connexion a la base reussie.")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS article_scores (
            url TEXT PRIMARY KEY,
            polarity INTEGER,
            polarity_conf REAL,
            uncertainty REAL,
            litigious REAL,
            fundamental_strength REAL
        )
        """
    )
    ensure_column(cursor, "article_scores", "litigious", "REAL")
    ensure_column(cursor, "article_scores", "fundamental_strength", "REAL")
    conn.commit()

    print("Table article_scores prete.")

    cursor.execute("SELECT url, title, content FROM articles")
    articles = cursor.fetchall()
    print(f"Nombre d'articles recuperes : {len(articles)}")

    for url, title, content in articles:
        text_to_analyze = content if content else title
        text_to_analyze = text_to_analyze[:1500]

        polarity, conf_pol, label = polarity_agent.predict(text_to_analyze)

        uncertainty = 0.0
        if uncertainty_agent:
            uncertainty = uncertainty_agent.predict(text_to_analyze)

        litigious = litigious_agent.predict(text_to_analyze)
        fundamental_strength = fundamental_strength_agent.predict(text_to_analyze)

        cursor.execute(
            """
            INSERT OR REPLACE INTO article_scores (
                url,
                polarity,
                polarity_conf,
                uncertainty,
                litigious,
                fundamental_strength
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (url, polarity, conf_pol, uncertainty, litigious, fundamental_strength),
        )

        print("\n------------------------------")
        print(f"Article traite : {title}")
        print(f"  label = {label}, polarity = {polarity}, confidence = {conf_pol:.3f}")
        print(f"  uncertainty = {uncertainty:.4f}")
        print(f"  litigious = {litigious:.4f}")
        print(f"  fundamental_strength = {fundamental_strength:.4f}")

    conn.commit()
    conn.close()

    print("Scores generes et enregistres.")
    print("Fin du processing.")


if __name__ == "__main__":
    run_processing_pipeline()
