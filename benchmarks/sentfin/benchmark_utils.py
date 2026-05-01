import argparse
import json
import os
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


LABELS = ["positive", "neutral", "negative"]


def normalize_label(label):
    """
    Normalise les sorties de tous les modèles vers :
    positive / neutral / negative / unknown
    """
    label = str(label).strip().lower()

    mapping = {
        "positive": "positive",
        "pos": "positive",
        "bullish": "positive",
        "achat": "positive",
        "buy": "positive",
        "1": "positive",
        "1.0": "positive",

        "neutral": "neutral",
        "neutre": "neutral",
        "none": "neutral",
        "null": "neutral",
        "0": "neutral",
        "0.0": "neutral",

        "negative": "negative",
        "neg": "negative",
        "bearish": "negative",
        "vente": "negative",
        "sell": "negative",
        "-1": "negative",
        "-1.0": "negative",
    }

    return mapping.get(label, "unknown")


def build_input(text, entity=None):
    """
    Entrée commune pour les modèles Hugging Face.
    SentFin contient souvent une entité cible.
    On l'ajoute pour aider le modèle à comprendre sur quelle entreprise juger le sentiment.
    """
    text = "" if pd.isna(text) else str(text)
    entity = "" if entity is None or pd.isna(entity) else str(entity)

    if entity:
        return f"Target entity: {entity}\nText: {text}"
    return text


def load_sentfin(input_file, limit=None):
    df = pd.read_csv(input_file, encoding="utf-8-sig")

    required_cols = ["text", "entity", "gold_sentiment"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {input_file}: {missing}. "
            f"Colonnes trouvées: {list(df.columns)}"
        )

    df["text"] = df["text"].fillna("").astype(str)
    df["entity"] = df["entity"].fillna("").astype(str)
    df["gold_sentiment"] = df["gold_sentiment"].apply(normalize_label)

    df = df[df["gold_sentiment"].isin(LABELS)].copy()

    if limit is not None:
        df = df.head(int(limit)).copy()

    return df.reset_index(drop=True)


def evaluate_and_save(df, pred_col, output_file, model_name=None):
    eval_df = df[df[pred_col].isin(LABELS)].copy()
    unknown_df = df[~df[pred_col].isin(LABELS)].copy()

    total = len(df)
    valid = len(eval_df)
    unknown = len(unknown_df)

    if valid == 0:
        print("Aucune prédiction valide. Vérifie le parsing ou les labels.")
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        return

    y_true = eval_df["gold_sentiment"]
    y_pred = eval_df[pred_col]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=LABELS,
        zero_division=0
    )

    avg_time = None
    if "inference_time_sec" in eval_df.columns:
        avg_time = float(eval_df["inference_time_sec"].mean())

    print("\n==============================")
    print(f"RESULTATS : {model_name or pred_col}")
    print("==============================")
    print(f"Total lignes            : {total}")
    print(f"Prédictions valides     : {valid}")
    print(f"Prédictions unknown     : {unknown}")
    print(f"Accuracy                : {acc:.4f}")
    print(f"Macro-F1                : {macro_f1:.4f}")

    if avg_time is not None:
        print(f"Temps moyen/article     : {avg_time:.6f} sec")

    print("\nMatrice de confusion :")
    print(confusion_matrix(y_true, y_pred, labels=LABELS))

    print("\nRapport détaillé :")
    print(classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0
    ))

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nPrédictions sauvegardées dans : {output_file}")

    summary = {
        "model": model_name or pred_col,
        "output_file": output_file,
        "total_rows": total,
        "valid_predictions": valid,
        "unknown_predictions": unknown,
        "accuracy": round(float(acc), 6),
        "macro_f1": round(float(macro_f1), 6),
        "avg_inference_time_sec": None if avg_time is None else round(avg_time, 6),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    summary_file = output_file.replace(".csv", "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Résumé sauvegardé dans : {summary_file}")


def add_common_args(parser):
    parser.add_argument("--input", default="sentfin_v2_clean.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser