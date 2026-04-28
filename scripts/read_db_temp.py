import sqlite3
from pathlib import Path

import pandas as pd

db_path = Path("data/news_database.db")
if db_path.exists():
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("PRAGMA table_info(articles)", conn)
        print("=== SCHEMA DE LA TABLE articles ===")
        for _, row in df.iterrows():
            print(f"- {row['name']} ({row['type']})")

        print("\n=== Market Regime Count ===")
        try:
            regimes = pd.read_sql_query(
                "SELECT market_regime, count(*) as n FROM articles GROUP BY market_regime", conn
            )
            print(regimes)
        except Exception as e:
            print(f"Error querying market_regime: {e}")

        print("\n=== Signal Count ===")
        try:
            signals = pd.read_sql_query("SELECT signal_final, count(*) as n FROM articles GROUP BY signal_final", conn)
            print(signals)
        except Exception as e:
            print(f"Error querying signal_final: {e}")
else:
    print("DATABASE NOT FOUND")
