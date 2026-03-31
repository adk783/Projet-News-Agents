import re

with open('human_review.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. SQL Table create & Left Join
sql_old = '''    c.execute("""
        SELECT a.url, a.title, a.ticker, a.content, s.polarity, s.polarity_conf, s.uncertainty
        FROM article_scores s
        JOIN articles a ON s.url = a.url
        ORDER BY s.uncertainty DESC
    """)'''

sql_new = '''    c.execute("""
        CREATE TABLE IF NOT EXISTS human_reviews (
            url TEXT PRIMARY KEY,
            human_status TEXT,
            human_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    c.execute("""
        SELECT a.url, a.title, a.ticker, a.content, s.polarity, s.polarity_conf, s.uncertainty,
               hr.human_status, hr.human_score
        FROM article_scores s
        JOIN articles a ON s.url = a.url
        LEFT JOIN human_reviews hr ON s.url = hr.url
        ORDER BY s.uncertainty DESC
    """)'''
text = text.replace(sql_old, sql_new)

# Update the parsing
text = text.replace(
'''            "human_score": None,
            "human_status": "pending",  # pending, approved, modified, rejected''',
'''            "human_score": r[8] if r[7] else None,
            "human_status": r[7] if r[7] else "pending",'''
)

# 2. Add SQL Post save
sql_save_old = '''            # Update cache
            if ReviewHandler.articles_cache and idx < len(ReviewHandler.articles_cache):
                ReviewHandler.articles_cache[idx]['human_status'] = data['status']
                ReviewHandler.articles_cache[idx]['human_score'] = data.get('human_score')'''

sql_save_new = '''            # Update cache
            if ReviewHandler.articles_cache and idx < len(ReviewHandler.articles_cache):
                article = ReviewHandler.articles_cache[idx]
                article['human_status'] = data['status']
                article['human_score'] = data.get('human_score')
                try:
                    import sqlite3
                    conn = sqlite3.connect(DB_PATH)
                    c = conn.cursor()
                    c.execute("""
                        INSERT OR REPLACE INTO human_reviews (url, human_status, human_score, timestamp)
                        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    """, (article['url'], data['status'], data.get('human_score')))
                    conn.commit()
                    conn.close()
                except Exception as e:
                    pass'''
text = text.replace(sql_save_old, sql_save_new)

# 3. CSS to Monochrome
css_old = '''        :root {
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

css_new = '''        :root {
            --bg-primary: #f0f0f0;
            --bg-secondary: #ffffff;
            --bg-card: #ffffff;
            --bg-hover: #e0e0e0;
            --accent: #222222;
            --accent-light: #555555;
            --green: #000000;
            --green-dim: rgba(0,0,0,0.06);
            --red: #000000;
            --red-dim: rgba(0,0,0,0.06);
            --orange: #000000;
            --orange-dim: rgba(0,0,0,0.06);
            --blue: #000000;
            --text-primary: #000000;
            --text-secondary: #333333;
            --text-dim: #555555;
            --border: rgba(0,0,0,0.15);
            --glow: rgba(0,0,0,0);
        }'''
text = text.replace(css_old, css_new)

# 4. Remove Emojis Manually
emoji_list = ['✅', '✏️', '❌', '🎲', '📥', '📊', '📰', '🔍', '⏱️', '⏳']
for e in emoji_list:
    text = text.replace(e, '')

# Cleanup empty spaces
text = text.replace('  Approuvé', 'Approuvé')
text = text.replace('  Modifié', 'Modifié')
text = text.replace('  Rejeté', 'Rejeté')
text = text.replace('  Approuver', 'Approuver')
text = text.replace('  Modifier', 'Modifier')
text = text.replace('  Rejeter', 'Rejeter')
text = text.replace('  En attente', 'En attente')

# Update bad wording of Approuver/Modifier
text = text.replace(">Approuver<", ">Le modele a raison<")
text = text.replace(">Modifier<", ">Je veux corriger<")
text = text.replace(">Rejeter<", ">Rejeter l'article<")

text = text.replace("✅ Approuver", "Le modele a raison")
text = text.replace("✏️ Modifier", "Je veux corriger")
text = text.replace("❌ Rejeter", "Rejeter l'article")

# Inject warning msg for slider without quotes inside strings
warning = '<label>Score d\\'incertitude corrigé :</label><p style="font-size:11px; margin-bottom:12px; font-style:italic; color:#555;">(C\\'est sauvegarde auto. Cliquez juste sur Suivant au lieu de valider apres.)</p>'
text = text.replace("<label>Score d'incertitude corrigé :</label>", warning)

# JS Chart Colors array
old_array = "['#00d4aa', '#00d4aa', '#42a5f5', '#42a5f5', '#ffa726', '#ffa726', '#ff6b6b', '#ff6b6b', '#ff4444', '#ff4444']"
new_array = "['#ccc', '#ccc', '#aaa', '#aaa', '#888', '#888', '#555', '#555', '#222', '#222']"
text = text.replace(old_array, new_array)

with open('human_review.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Succesfully rewrote human_review.py")
