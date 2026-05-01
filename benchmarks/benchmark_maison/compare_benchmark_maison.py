import json
import glob
import pandas as pd

rows = []

for file in glob.glob("benchmark_maison_*_summary.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows.append({
        "method": data.get("method"),
        "total_rows": data.get("total_rows"),
        "relevance_accuracy": data.get("relevance_accuracy"),
        "relevance_f1": data.get("relevance_f1"),
        "sentiment_accuracy": data.get("sentiment_accuracy"),
        "sentiment_macro_f1": data.get("sentiment_macro_f1"),
        "avg_time_sec": data.get("avg_inference_time_sec"),
        "output_file": data.get("output_file"),
    })

df = pd.DataFrame(rows)

if df.empty:
    print("Aucun fichier benchmark_maison_*_summary.json trouvé.")
    exit()

df = df.sort_values(
    by=["sentiment_macro_f1", "relevance_f1"],
    ascending=False
)

print("\n===== COMPARAISON BENCHMARK MAISON =====\n")
print(df.to_string(index=False))

df.to_csv("benchmark_maison_comparison_results.csv", index=False, encoding="utf-8-sig")

print("\nFichier créé : benchmark_maison_comparison_results.csv")