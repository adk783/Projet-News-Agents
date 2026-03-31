import sqlite3
import statistics

conn = sqlite3.connect('news_database.db')
c = conn.cursor()

c.execute('SELECT COUNT(*) FROM articles')
print('Articles:', c.fetchone()[0])

c.execute('SELECT COUNT(*) FROM article_scores')
print('Scores:', c.fetchone()[0])

c.execute('SELECT MIN(uncertainty), MAX(uncertainty), AVG(uncertainty) FROM article_scores')
r = c.fetchone()
print(f'Uncertainty min: {r[0]}')
print(f'Uncertainty max: {r[1]}')
print(f'Uncertainty avg: {r[2]}')

c.execute('SELECT uncertainty FROM article_scores')
scores = [r[0] for r in c.fetchall()]
print(f'Std dev: {statistics.stdev(scores):.6f}')
srt = sorted(scores)
n = len(srt)
print(f'Q1: {srt[n//4]:.4f}')
print(f'Q2: {srt[n//2]:.4f}')
print(f'Q3: {srt[3*n//4]:.4f}')

# Distribution bins
bins = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
for lo, hi in bins:
    count = sum(1 for s in scores if lo <= s < hi)
    print(f'  [{lo:.1f}, {hi:.1f}): {count}')

# Write to file for easy reading
with open('stats_output.txt', 'w') as f:
    f.write(f'Articles: {n}\n')
    f.write(f'Min: {min(scores)}\n')
    f.write(f'Max: {max(scores)}\n')
    f.write(f'Mean: {statistics.mean(scores):.6f}\n')
    f.write(f'Stdev: {statistics.stdev(scores):.6f}\n')
    f.write(f'All scores sorted:\n')
    for s in srt:
        f.write(f'  {s}\n')

conn.close()
print('Done - see stats_output.txt')
