import re

with open('human_review.py', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add SQL Table create & Left Join
text = text.replace(
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

text = text.replace(
'''            "human_score": None,
            "human_status": "pending",  # pending, approved, modified, rejected''',
'''            "human_status": r[7] if r[7] else "pending",
            "human_score": r[8] if r[7] else None,'''
)

# 2. Add SQL Post save
text = text.replace(
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

# 3. CSS to Monochrome
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
            --bg-primary: #ffffff;
            --bg-secondary: #f4f4f4;
            --bg-card: #ffffff;
            --bg-hover: #e0e0e0;
            --accent: #000000;
            --accent-light: #555555;
            --green: #000000;
            --green-dim: rgba(0,0,0,0.05);
            --red: #000000;
            --red-dim: rgba(0,0,0,0.05);
            --orange: #000000;
            --orange-dim: rgba(0,0,0,0.05);
            --blue: #000000;
            --text-primary: #000000;
            --text-secondary: #444444;
            --text-dim: #777777;
            --border: rgba(0,0,0,0.15);
            --glow: rgba(0,0,0,0);
        }'''
text = text.replace(css_original, css_mono)

# 4. Remove emojis with simple regex
emoji_pattern = re.compile(u'[\\U00010000-\\U0010ffff]', flags=re.UNICODE)
text = emoji_pattern.sub(r'', text)
text = text.replace('✅', '').replace('✏️', '').replace('❌', '').replace('🎲', '').replace('📥', '').replace('📊', '').replace('📰', '').replace('🔍', '').replace('⏱️', '')

# Remove random spaces left by emojis
text = text.replace('  Approuver', 'Approuver')
text = text.replace('  Modifier', 'Modifier')
text = text.replace('  Rejeter', 'Rejeter')

# Update button texts for clarity
text = text.replace(">Approuver<", ">Le modele a raison<")
text = text.replace(">Modifier<", ">Je veux corriger<")
text = text.replace(">Rejeter<", ">Rejeter l'article<")

# Add the warning snippet
text = text.replace('<label>Score d\\'incertitude corrigé :</label>',
'<label>Score d\\'incertitude corrigé :</label><p style="font-size:11px; margin-bottom:12px; font-style:italic; color:#555;">(Sauvegarde auto. Cliquez juste sur Suivant apres.)</p>')

text = text.replace("['#00d4aa', '#00d4aa', '#42a5f5', '#42a5f5', '#ffa726', '#ffa726', '#ff6b6b', '#ff6b6b', '#ff4444', '#ff4444']",
"['#ccc', '#bbb', '#aaa', '#999', '#888', '#777', '#666', '#555', '#444', '#333']")

with open('human_review.py', 'w', encoding='utf-8') as f:
    f.write(text)

print("Done monochrome update!")
