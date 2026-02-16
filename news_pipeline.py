import yfinance as yf
from newspaper import Article
import sqlite3
import json
from datetime import datetime, timezone

def run_news_pipeline():
    # --- PRÉPARATION : LIVRAISON (Stockage Local SQLite) ---
    # Création du fichier et de la table (avec 'IF NOT EXISTS' pour ne rien écraser)
    conn = sqlite3.connect('news_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            url TEXT PRIMARY KEY, 
            ticker TEXT,
            date_utc TEXT,
            title TEXT,
            content TEXT,
            json_brut TEXT
        )
    ''') # L'URL sert de PRIMARY KEY pour bloquer les doublons automatiquement
    conn.commit()

    ticker_symbol = "AAPL"
    print(f"--- 1. DÉTECTION : Scan des news pour {ticker_symbol} ---")
    # Utilisation du module yfinance pour le ticker ciblé
    stock = yf.Ticker(ticker_symbol)
    news_list = stock.news 

    for news_item in news_list:
        url = news_item.get('link')
        title = news_item.get('title')
        timestamp = news_item.get('providerPublishTime')

        # --- 3a. NORMALISATION : Conversion de la date en UTC ---
        date_utc = ""
        if timestamp:
            # yfinance donne un timestamp Unix, on le convertit en format lisible UTC
            date_utc = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()

        print(f"\nTraitement de l'article : {title}")
        
        # --- 2. EXTRACTION : Corps de l'article avec newspaper3k ---
        content = ""
        try:
            article = Article(url)
            article.download()
            article.parse()
            content = article.text # Extraction propre du corps de l'article
            print("  -> Extraction réussie.")
        except Exception as e:
            print(f"  -> Erreur d'extraction (blocage ou format) : {e}")
            continue # Si le site bloque, on passe simplement à la news suivante

        # --- 3b. NORMALISATION : Création du JSON Standard ---
        data_dict = {
            "ticker": ticker_symbol,
            "title": title,
            "date_utc": date_utc,
            "url": url,
            "content": content
        }
        json_standard = json.dumps(data_dict, ensure_ascii=False)

        # --- 4. LIVRAISON : Ajout à la base SQLite ---
        try:
            # INSERT OR IGNORE permet d'ignorer la ligne si l'URL existe déjà !
            cursor.execute('''
                INSERT OR IGNORE INTO articles (url, ticker, date_utc, title, content, json_brut)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (url, ticker_symbol, date_utc, title, content, json_standard))
            
            # On vérifie si une nouvelle ligne a vraiment été ajoutée
            if cursor.rowcount > 0:
                print("  -> LIVRAISON : Nouvel article sauvegardé en base de données !")
            else:
                print("  -> LIVRAISON : Article déjà présent en base (ignoré).")
            
            conn.commit()
        except Exception as e:
            print(f"  -> Erreur lors de la sauvegarde : {e}")

    # Fermeture de la connexion à la fin du scan
    conn.close()
    print("\n--- Fin du Pipeline ---")

# Lancement du script
if __name__ == "__main__":
    run_news_pipeline()