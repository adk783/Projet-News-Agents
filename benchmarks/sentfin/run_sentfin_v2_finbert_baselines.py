import argparse
import time

from transformers import pipeline

from benchmark_utils import (
    add_common_args,
    build_input,
    evaluate_and_save,
    load_sentfin,
    normalize_label,
)


MODELS = {
    "finbert": "ProsusAI/finbert",
    "finbert_twitter": "nickmuchi/finbert-tone-finetuned-fintwitter-classification",
}


def run_model(model_key, input_file, output_file, limit):
    if model_key not in MODELS:
        raise ValueError(f"Modèle inconnu: {model_key}. Choix possibles: {list(MODELS.keys())}")

    model_name = MODELS[model_key]
    pred_col = f"pred_{model_key}"

    df = load_sentfin(input_file, limit=limit)

    print(f"Chargement du modèle Hugging Face : {model_name}")
    clf = pipeline("text-classification", model=model_name)

    predictions = []
    raw_labels = []
    raw_scores = []
    times = []

    for i, row in df.iterrows():
        model_input = build_input(row["text"], row["entity"])

        start = time.perf_counter()
        result = clf(model_input, truncation=True)[0]
        elapsed = time.perf_counter() - start

        raw_label = result.get("label", "")
        raw_score = result.get("score", None)
        pred = normalize_label(raw_label)

        predictions.append(pred)
        raw_labels.append(raw_label)
        raw_scores.append(raw_score)
        times.append(elapsed)

        if (i + 1) % 25 == 0:
            print(f"{i + 1}/{len(df)} lignes traitées...")

    df[pred_col] = predictions
    df[f"{model_key}_raw_label"] = raw_labels
    df[f"{model_key}_raw_score"] = raw_scores
    df["inference_time_sec"] = times

    evaluate_and_save(df, pred_col, output_file, model_name=model_key)


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--model", choices=list(MODELS.keys()), required=True)
    args = parser.parse_args()

    output_file = args.output or f"sentfin_v2_{args.model}_predictions.csv"
    run_model(args.model, args.input, output_file, args.limit)


if __name__ == "__main__":
    main()