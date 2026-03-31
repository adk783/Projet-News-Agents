import sqlite3
conn=sqlite3.connect('news_database.db')
cursor=conn.cursor()
cursor.execute('SELECT * FROM human_reviews')
rows=cursor.fetchall()
print(f'Total reviews: {len(rows)}')
for r in rows:
    print(r)

conn.close()
