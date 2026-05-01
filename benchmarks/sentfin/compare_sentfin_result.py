import json
import glob
import pandas as pd

rows = []

for file in glob.glob("*_summary.json"):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows.append({
        "model": data.get("model"),
        "total_rows": data.get("total_rows"),
        "valid_predictions": data.get("valid_predictions"),
        "unknown_predictions": data.get("unknown_predictions"),
        "accuracy": data.get("accuracy"),
        "macro_f1": data.get("macro_f1"),
        "avg_time_sec": data.get("avg_inference_time_sec"),
        "output_file": data.get("output_file"),
    })

df = pd.DataFrame(rows)

df = df.sort_values(by=["macro_f1", "accuracy"], ascending=False)

print("\n===== COMPARAISON SENTFIN V2 =====\n")
print(df.to_string(index=False))

df.to_csv("sentfin_v2_comparison_results.csv", index=False, encoding="utf-8-sig")

print("\nFichier créé : sentfin_v2_comparison_results.csv")