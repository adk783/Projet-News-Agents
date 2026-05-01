import argparse
import os
import sys
import time

from benchmark_utils import (
    add_common_args,
    evaluate_and_save,
    load_sentfin,
    normalize_label,
)


def load_antoine_agent(repo_root):
    repo_root = os.path.abspath(repo_root)
    sys.path.insert(0, repo_root)

    from agent_sentiment import analyser_article
    return analyser_article


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--repo-root", required=True, help="Chemin vers le repo GitHub en branche Antoinev2")
    args = parser.parse_args()

    output_file = args.output or "sentfin_v2_antoinev2_predictions.csv"
    pred_col = "pred_antoinev2"

    df = load_sentfin(args.input, limit=args.limit)

    analyser_article = load_antoine_agent(args.repo_root)

    predictions = []
    raw_sentiments = []
    scores = []
    reasonings = []
    times = []
    errors = []

    for i, row in df.iterrows():
        entity = row["entity"]
        text = row["text"]

        title = str(text)[:250]
        content = str(text)

        start = time.perf_counter()

        try:
            result = analyser_article(
                url=f"sentfin_row_{i}",
                ticker=entity,
                title=title,
                content=content
            )

            elapsed = time.perf_counter() - start

            raw_sentiment = result.get("sentiment", "unknown")
            pred = normalize_label(raw_sentiment)

            predictions.append(pred)
            raw_sentiments.append(raw_sentiment)
            scores.append(result.get("score", None))
            reasonings.append(result.get("reasoning", ""))
            errors.append("")

        except Exception as e:
            elapsed = time.perf_counter() - start

            predictions.append("unknown")
            raw_sentiments.append("unknown")
            scores.append(None)
            reasonings.append("")
            errors.append(str(e))

        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(df)} lignes traitées...")

    df[pred_col] = predictions
    df["antoinev2_raw_sentiment"] = raw_sentiments
    df["antoinev2_score"] = scores
    df["antoinev2_reasoning"] = reasonings
    df["antoinev2_error"] = errors
    df["inference_time_sec"] = times

    evaluate_and_save(df, pred_col, output_file, model_name="antoinev2_phi4_mini")


if __name__ == "__main__":
    main()