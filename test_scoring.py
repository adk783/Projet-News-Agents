"""Quick test: compare old vs new uncertainty score distribution on real articles."""
import sqlite3
import re
import math
import statistics
import sys

sys.path.insert(0, '.')

# Old function
def old_compute_uncertainty_score(text, lexicon):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) == 0:
        return 0.0
    uncertainty_count = sum(1 for w in words if w in lexicon)
    raw_ratio = uncertainty_count / len(words)
    SATURATION_RATIO = 0.08
    score = min(raw_ratio / SATURATION_RATIO, 1.0)
    return round(score, 4)

from uncertainty_agent import compute_uncertainty_score, download_lm_uncertainty_lexicon

lexicon = download_lm_uncertainty_lexicon()

conn = sqlite3.connect('news_database.db')
c = conn.cursor()
c.execute('SELECT content FROM articles WHERE content IS NOT NULL AND LENGTH(content) > 100')
articles = [r[0] for r in c.fetchall()]
conn.close()

old_scores = [old_compute_uncertainty_score(a, lexicon) for a in articles]
new_scores = [compute_uncertainty_score(a, lexicon) for a in articles]

lines = []
lines.append(f"COMPARISON: OLD vs NEW scoring on {len(articles)} real articles")
lines.append(f"")
lines.append(f"  Metric               OLD        NEW")
lines.append(f"  ----------------------------------------")
lines.append(f"  Min              {min(old_scores):>10.4f} {min(new_scores):>10.4f}")
lines.append(f"  Max              {max(old_scores):>10.4f} {max(new_scores):>10.4f}")
lines.append(f"  Mean             {statistics.mean(old_scores):>10.4f} {statistics.mean(new_scores):>10.4f}")
lines.append(f"  Stdev            {statistics.stdev(old_scores):>10.4f} {statistics.stdev(new_scores):>10.4f}")
srt_old = sorted(old_scores)
srt_new = sorted(new_scores)
n = len(articles)
lines.append(f"  Q1               {srt_old[n//4]:>10.4f} {srt_new[n//4]:>10.4f}")
lines.append(f"  Median           {srt_old[n//2]:>10.4f} {srt_new[n//2]:>10.4f}")
lines.append(f"  Q3               {srt_old[3*n//4]:>10.4f} {srt_new[3*n//4]:>10.4f}")
lines.append(f"")

bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
labels_b = ['[0.0-0.2)', '[0.2-0.4)', '[0.4-0.6)', '[0.6-0.8)', '[0.8-1.0]']
lines.append(f"  Bin          OLD count  NEW count")
lines.append(f"  --------------------------------")
for (lo, hi), label in zip(bins, labels_b):
    old_c = sum(1 for s in old_scores if lo <= s < hi)
    new_c = sum(1 for s in new_scores if lo <= s < hi)
    lines.append(f"  {label:<12} {old_c:>10} {new_c:>10}")

output = "\n".join(lines)
print(output)
with open('test_output.txt', 'w', encoding='utf-8') as f:
    f.write(output)
