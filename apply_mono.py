import re

with open("human_review.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. SQL Persistence (get_articles)
code = code.replace(
'''    c.execute("""
        SELECT a.url, a.title, a.ticker, a.content, s.polarity, s.polarity_conf, s.uncertainty
        FROM article_scores s
        JOIN articles a ON s.url = a.url
        ORDER BY s.uncertainty DESC
    """)''',
'''    c.execute(\'\'\'
        CREATE TABLE IF NOT EXISTS human_reviews (
            url TEXT PRIMARY KEY,
            human_status TEXT,
            human_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    \'\'\')
    c.execute("""
        SELECT a.url, a.title, a.ticker, a.content, s.polarity, s.polarity_conf, s.uncertainty,
               hr.human_status, hr.human_score
        FROM article_scores s
        JOIN articles a ON s.url = a.url
        LEFT JOIN human_reviews hr ON s.url = hr.url
        ORDER BY s.uncertainty DESC
    """)'''
)

code = code.replace(
'''            "human_score": None,
            "human_status": "pending",  # pending, approved, modified, rejected''',
'''            "human_status": r[7] if r[7] else "pending",
            "human_score": r[8] if r[7] else None,'''
)

# 2. SQL Persistence (do_POST)
code = code.replace(
'''            # Update cache
            if ReviewHandler.articles_cache and idx < len(ReviewHandler.articles_cache):
                ReviewHandler.articles_cache[idx]['human_status'] = data['status']
                ReviewHandler.articles_cache[idx]['human_score'] = data.get('human_score')''',
'''            # Update cache
            if ReviewHandler.articles_cache and idx < len(ReviewHandler.articles_cache):
                article = ReviewHandler.articles_cache[idx]
                article['human_status'] = data['status']
                article['human_score'] = data.get('human_score')
                try:
                    import sqlite3
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute(\'\'\'
                        INSERT OR REPLACE INTO human_reviews (url, human_status, human_score, timestamp)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    \'\'\', (article['url'], data['status'], data.get('human_score')))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    pass'''
)

# 3. CSS Monochrome transformation (Light theme)
css_original = '''        :root {
            --bg-primary: #0f0f1a;
            --bg-secondary: #1a1a2e;
            --bg-card: #16213e;
            --bg-hover: #1f2b4d;
            --accent: #6c63ff;
            --accent-light: #8b83ff;
            --green: #00d4aa;
            --green-dim: rgba(0,212,170,0.15);
            --red: #ff6b6b;
            --red-dim: rgba(255,107,107,0.15);
            --orange: #ffa726;
            --orange-dim: rgba(255,167,38,0.15);
            --blue: #42a5f5;
            --text-primary: #e8e8f0;
            --text-secondary: #9090a8;
            --text-dim: #606080;
            --border: rgba(108,99,255,0.2);
            --glow: rgba(108,99,255,0.3);
        }'''

css_mono = '''        :root {
            --bg-primary: #f0f0f0;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --bg-hover: #e0e0e0;
            --accent: #000000;
            --accent-light: #444444;
            --green: #000000;
            --green-dim: rgba(0,0,0,0.08);
            --red: #000000;
            --red-dim: rgba(0,0,0,0.08);
            --orange: #000000;
            --orange-dim: rgba(0,0,0,0.08);
            --blue: #000000;
            --text-primary: #000000;
            --text-secondary: #333333;
            --text-dim: #666666;
            --border: rgba(0,0,0,0.2);
            --glow: rgba(0,0,0,0.0);
        }'''
code = code.replace(css_original, css_mono)

# 4. Remove all emojis from HTML
import emoji
code = emoji.replace_emoji(code, replace='')

# 5. Fix "Approuver" logic and UI wording
code = code.replace("✅ Approuver", "Le modele a raison")
code = code.replace("✏️ Modifier", "Je veux corriger")
code = code.replace("❌ Rejeter", "Rejeter l'article")
code = code.replace(
'''                <p style="font-size:11px; color:var(--text-secondary); margin-bottom:12px; font-style:italic;">
                    ⚠️ Bougez le curseur pour corriger. C'est sauvegardé automatiquement ! Pas besoin de cliquer sur un bouton de validation, passez juste à l'article "Suivant →".
                </p>''', "")
# We will just inject the help text without emoji
code = code.replace('<label>Score d\'incertitude corrigé :</label>', 
'<label>Score d\'incertitude corrigé :</label><p style="font-size:11.5px; margin-bottom:12px; font-style:italic;">Ajustez pour sauvegarder. Ne cliquez pas sur Approuver après !</p>')

# Fix colors in JS charts
code = code.replace("['#00d4aa', '#00d4aa', '#42a5f5', '#42a5f5', '#ffa726', '#ffa726', '#ff6b6b', '#ff6b6b', '#ff4444', '#ff4444']", 
"['#333', '#444', '#555', '#666', '#777', '#888', '#999', '#aaa', '#bbb', '#ccc']")

with open("human_review.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Done!")
