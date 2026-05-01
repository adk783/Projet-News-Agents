import argparse
import json
import os
import sys
import time
from datetime import datetime

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


SENTIMENT_LABELS = ["positive", "neutral", "negative"]


POSITIVE_KEYWORDS = [
    "beat", "beats", "growth", "strong", "profit", "surge", "record",
    "upgrade", "gain", "gains", "improve", "improved", "partnership",
    "expands", "expansion", "success", "successful", "increase",
    "increases", "higher", "rise", "rises", "bullish", "outperform"
]

NEGATIVE_KEYWORDS = [
    "lawsuit", "probe", "investigation", "decline", "miss", "misses",
    "cut", "cuts", "layoff", "layoffs", "downgrade", "drop", "falls",
    "fine", "fined", "risk", "warning", "weak", "weaker", "loss",
    "losses", "decrease", "decreases", "bearish", "underperform"
]

COMPANY_KEYWORDS = {
    "AAPL": ["apple", "aapl", "iphone", "ipad", "mac", "tim cook"],
    "TSLA": ["tesla", "tsla", "elon musk", "musk", "model 3", "model y"],
    "JPM": ["jpmorgan", "jpm", "jp morgan", "jamie dimon", "dimon", "chase"],

    # Tickers ajoutés pour le benchmark maison à 130 articles
    "AMZN": ["amazon", "amzn", "aws", "andy jassy", "jassy", "prime", "alexa"],
    "NVDA": ["nvidia", "nvda", "jensen huang", "huang", "geforce", "cuda", "gpu", "ai chip", "chips"],
    "MSFT": ["microsoft", "msft", "satya nadella", "nadella", "azure", "windows", "copilot", "openai"],
    "GOOGL": ["alphabet", "google", "googl", "goog", "sundar pichai", "pichai", "gemini", "youtube", "waymo"],
    "META": ["meta", "facebook", "instagram", "whatsapp", "mark zuckerberg", "zuckerberg", "reels"],
    "AMD": ["amd", "advanced micro devices", "lisa su", "radeon", "ryzen", "epyc", "mi300"],
    "NFLX": ["netflix", "nflx", "reed hastings", "ted sarandos", "streaming"],
    "BAC": ["bank of america", "bac", "bofa", "bofA", "brian moynihan", "moynihan"],
    "WMT": ["walmart", "wmt", "sam's club", "sams club", "doug mcmillon", "mcmillon"],
    "DIS": ["disney", "dis", "espn", "abc", "pixar", "marvel", "bob iger", "iger"],
    "KO": ["coca-cola", "coca cola", "coke", "ko", "james quincey", "quincey"],
    "PFE": ["pfizer", "pfe", "albert bourla", "bourla", "eliquis", "covid vaccine"],
}


def normalize_sentiment(label):
    label = str(label).strip().lower()

    mapping = {
        "positive": "positive",
        "pos": "positive",
        "bullish": "positive",
        "buy": "positive",
        "achat": "positive",
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
        "sell": "negative",
        "vente": "negative",
        "-1": "negative",
        "-1.0": "negative",
    }

    return mapping.get(label, "unknown")


def normalize_relevance(value):
    value = str(value).strip().lower()

    if value in ["1", "1.0", "true", "yes", "y", "relevant", "pertinent"]:
        return 1

    if value in ["0", "0.0", "false", "no", "n", "irrelevant", "non pertinent"]:
        return 0

    return None


def read_benchmark_csv(path):
    """
    Lecture robuste : marche avec CSV séparé par virgule ou point-virgule.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", sep=None, engine="python")

    required = ["ticker", "title", "content", "relevance_gold", "sentiment_gold"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Colonnes manquantes dans {path}: {missing}\n"
            f"Colonnes trouvées: {list(df.columns)}"
        )

    if "article_id" not in df.columns:
        df["article_id"] = range(1, len(df) + 1)

    df["ticker"] = df["ticker"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["content"] = df["content"].fillna("").astype(str)

    df["relevance_gold_norm"] = df["relevance_gold"].apply(normalize_relevance)
    df["sentiment_gold_norm"] = df["sentiment_gold"].apply(normalize_sentiment)

    df = df[df["relevance_gold_norm"].notna()].copy()
    df["relevance_gold_norm"] = df["relevance_gold_norm"].astype(int)

    return df.reset_index(drop=True)


def build_text(row, max_chars=None):
    text = f"{row['title']}\n\n{row['content']}".strip()

    if max_chars is not None:
        text = text[:max_chars]

    return text


def predict_relevance_keywords(ticker, title, content):
    import re

    ticker = str(ticker).upper().strip()
    text = f"{title} {content}".lower()

    keywords = COMPANY_KEYWORDS.get(ticker, [ticker.lower()])

    for kw in keywords:
        kw = str(kw).lower().strip()
        if not kw:
            continue

        # Pour les tickers très courts comme KO ou DIS, on impose une vraie frontière de mot
        # afin d'éviter les faux positifs dans des mots comme discussed, known, etc.
        pattern = r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            return 1

    return 0


def predict_sentiment_keywords(title, content):
    text = f"{title} {content}".lower()

    pos_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg_count = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)

    if pos_count > neg_count:
        return "positive"

    if neg_count > pos_count:
        return "negative"

    return "neutral"


def evaluate_and_save(df, method, output_file):
    print("\n==============================")
    print(f"RESULTATS BENCHMARK MAISON : {method}")
    print("==============================")

    rel_eval = df[df["pred_relevance"].isin([0, 1])].copy()

    y_rel = rel_eval["relevance_gold_norm"]
    pred_rel = rel_eval["pred_relevance"].astype(int)

    rel_acc = accuracy_score(y_rel, pred_rel)
    rel_f1 = f1_score(y_rel, pred_rel, zero_division=0)

    print(f"Nombre total d'articles     : {len(df)}")
    print(f"Pertinence - Accuracy       : {rel_acc:.4f}")
    print(f"Pertinence - F1             : {rel_f1:.4f}")

    print("\nMatrice de confusion pertinence [0, 1] :")
    print(confusion_matrix(y_rel, pred_rel, labels=[0, 1]))

    sent_eval = df[
        df["sentiment_gold_norm"].isin(SENTIMENT_LABELS)
        & df["pred_sentiment"].isin(SENTIMENT_LABELS)
    ].copy()

    if len(sent_eval) > 0:
        y_sent = sent_eval["sentiment_gold_norm"]
        pred_sent = sent_eval["pred_sentiment"]

        sent_acc = accuracy_score(y_sent, pred_sent)
        sent_macro_f1 = f1_score(
            y_sent,
            pred_sent,
            average="macro",
            labels=SENTIMENT_LABELS,
            zero_division=0
        )

        print(f"\nSentiment - Articles évalués : {len(sent_eval)}")
        print(f"Sentiment - Accuracy         : {sent_acc:.4f}")
        print(f"Sentiment - Macro-F1         : {sent_macro_f1:.4f}")

        print("\nMatrice de confusion sentiment [positive, neutral, negative] :")
        print(confusion_matrix(y_sent, pred_sent, labels=SENTIMENT_LABELS))

        print("\nRapport détaillé sentiment :")
        print(classification_report(
            y_sent,
            pred_sent,
            labels=SENTIMENT_LABELS,
            zero_division=0
        ))
    else:
        sent_acc = None
        sent_macro_f1 = None
        print("\nAucun sentiment évaluable. Vérifie sentiment_gold.")

    avg_time = None
    if "inference_time_sec" in df.columns:
        avg_time = float(df["inference_time_sec"].mean())
        print(f"\nTemps moyen/article          : {avg_time:.6f} sec")

    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\nPrédictions sauvegardées dans : {output_file}")

    summary = {
        "method": method,
        "output_file": output_file,
        "total_rows": int(len(df)),
        "relevance_accuracy": round(float(rel_acc), 6),
        "relevance_f1": round(float(rel_f1), 6),
        "sentiment_accuracy": None if sent_acc is None else round(float(sent_acc), 6),
        "sentiment_macro_f1": None if sent_macro_f1 is None else round(float(sent_macro_f1), 6),
        "avg_inference_time_sec": None if avg_time is None else round(float(avg_time), 6),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    summary_file = output_file.replace(".csv", "_summary.json")

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Résumé sauvegardé dans : {summary_file}")


def run_baseline_neutral(df):
    start = time.perf_counter()

    df["pred_relevance"] = 1
    df["pred_sentiment"] = "neutral"
    df["method_raw_output"] = "baseline_neutral"

    elapsed = time.perf_counter() - start
    df["inference_time_sec"] = elapsed / max(len(df), 1)

    return df


def run_baseline_keywords(df):
    predictions_rel = []
    predictions_sent = []

    start_total = time.perf_counter()

    for _, row in df.iterrows():
        pred_rel = predict_relevance_keywords(row["ticker"], row["title"], row["content"])
        pred_sent = predict_sentiment_keywords(row["title"], row["content"])

        predictions_rel.append(pred_rel)
        predictions_sent.append(pred_sent)

    elapsed = time.perf_counter() - start_total

    df["pred_relevance"] = predictions_rel
    df["pred_sentiment"] = predictions_sent
    df["method_raw_output"] = "baseline_keywords"
    df["inference_time_sec"] = elapsed / max(len(df), 1)

    return df


def run_hf_classifier(df, model_name, method_name):
    from transformers import pipeline

    print(f"Chargement du modèle : {model_name}")
    clf = pipeline("text-classification", model=model_name)

    predictions_rel = []
    predictions_sent = []
    raw_labels = []
    raw_scores = []
    times = []

    for i, row in df.iterrows():
        # Les modèles FinBERT/FinBERT Twitter ont une limite autour de 512 tokens.
        # Le benchmark maison contient des articles complets, donc on tronque le texte
        # avant l'appel modèle pour éviter les erreurs de taille de tenseur.
        text = build_text(row, max_chars=1800)
        model_input = f"Target entity: {row['ticker']}\nText: {text}"

        pred_rel = predict_relevance_keywords(row["ticker"], row["title"], row["content"])

        start = time.perf_counter()
        result = clf(model_input, truncation=True, max_length=512)[0]
        elapsed = time.perf_counter() - start

        raw_label = result.get("label", "")
        raw_score = result.get("score", None)

        pred_sent = normalize_sentiment(raw_label)

        predictions_rel.append(pred_rel)
        predictions_sent.append(pred_sent)
        raw_labels.append(raw_label)
        raw_scores.append(raw_score)
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(df)} articles traités...")

    df["pred_relevance"] = predictions_rel
    df["pred_sentiment"] = predictions_sent
    df["method_raw_output"] = raw_labels
    df["method_score"] = raw_scores
    df["inference_time_sec"] = times

    return df


def run_samuel(df, repo_root=None):
    """
    Teste le PolarityAgent de la branche Samuel si disponible.
    Sinon fallback vers ProsusAI/finbert.
    """
    if repo_root:
        repo_root = os.path.abspath(repo_root)
        sys.path.insert(0, repo_root)

    try:
        from polarity_agent import PolarityAgent
        agent = PolarityAgent()
        print("PolarityAgent Samuel chargé.")
        use_samuel = True
    except Exception as e:
        print("Impossible de charger PolarityAgent Samuel.")
        print(f"Erreur : {e}")
        print("Fallback vers ProsusAI/finbert.")
        use_samuel = False

    if not use_samuel:
        return run_hf_classifier(df, "ProsusAI/finbert", "samuel_fallback_finbert")

    predictions_rel = []
    predictions_sent = []
    raw_outputs = []
    times = []

    for i, row in df.iterrows():
        text = build_text(row, max_chars=1800)
        model_input = f"Target entity: {row['ticker']}\nText: {text}"

        pred_rel = predict_relevance_keywords(row["ticker"], row["title"], row["content"])

        start = time.perf_counter()

        try:
            result = agent.predict(model_input)
            elapsed = time.perf_counter() - start

            if isinstance(result, tuple):
                # Cas attendu : polarity, confidence, label
                if len(result) >= 3:
                    raw_label = result[2]
                else:
                    raw_label = result[0]
            elif isinstance(result, dict):
                raw_label = result.get("label") or result.get("sentiment") or result.get("polarity")
            else:
                raw_label = result

            pred_sent = normalize_sentiment(raw_label)
            raw_outputs.append(str(result))

        except Exception as e:
            elapsed = time.perf_counter() - start
            pred_sent = "unknown"
            raw_outputs.append(f"error: {e}")

        predictions_rel.append(pred_rel)
        predictions_sent.append(pred_sent)
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(df)} articles traités...")

    df["pred_relevance"] = predictions_rel
    df["pred_sentiment"] = predictions_sent
    df["method_raw_output"] = raw_outputs
    df["inference_time_sec"] = times

    return df


def run_poc_filtrage(df, threshold=0.70):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import pipeline

    model_name = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"

    print(f"Chargement du modèle POC-Filtrage-Agents : {model_name}")
    clf = pipeline("text-classification", model=model_name, truncation=True, device="cpu")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    predictions_rel = []
    predictions_sent = []
    raw_outputs = []
    confidence_scores = []
    times = []

    for i, row in df.iterrows():
        text = build_text(row)

        pred_rel = predict_relevance_keywords(row["ticker"], row["title"], row["content"])

        start = time.perf_counter()

        chunks = splitter.split_text(text)
        scores = {"positive": 0.0, "negative": 0.0}
        best_score = 0.0
        best_label = "neutral"

        for chunk in chunks:
            result = clf(chunk, truncation=True)[0]
            label = normalize_sentiment(result.get("label", "neutral"))
            score = float(result.get("score", 0.0))

            if label in ["positive", "negative"] and score > threshold:
                scores[label] += score

                if score > best_score:
                    best_score = score
                    best_label = label

        if scores["positive"] == 0.0 and scores["negative"] == 0.0:
            pred_sent = "neutral"
        else:
            pred_sent = "positive" if scores["positive"] >= scores["negative"] else "negative"

        elapsed = time.perf_counter() - start

        predictions_rel.append(pred_rel)
        predictions_sent.append(pred_sent)
        raw_outputs.append(best_label)
        confidence_scores.append(best_score)
        times.append(elapsed)

        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(df)} articles traités...")

    df["pred_relevance"] = predictions_rel
    df["pred_sentiment"] = predictions_sent
    df["method_raw_output"] = raw_outputs
    df["method_score"] = confidence_scores
    df["inference_time_sec"] = times

    return df


def run_antoinev2(df, repo_root=None):
    """
    Teste la vraie méthode de la branche Antoinev2.

    Cette fonction charge agent_sentiment.py depuis la branche Antoinev2,
    puis appelle directement analyser_article(url, ticker, title, content),
    comme dans le code du projet.

    La branche Antoinev2 produit surtout un sentiment. Pour garder une
    comparaison commune avec les autres méthodes sur le benchmark maison,
    la pertinence est évaluée avec la même règle par mots-clés que pour
    FinBERT, Samuel et POC-Filtrage.
    """
    if repo_root:
        repo_root = os.path.abspath(repo_root)
        sys.path.insert(0, repo_root)

    try:
        from agent_sentiment import analyser_article
        print("Agent Antoinev2 chargé depuis agent_sentiment.py.")
    except Exception as e:
        raise ImportError(
            "Impossible d'importer agent_sentiment.py depuis la branche Antoinev2.\n"
            "Vérifie le chemin donné avec --repo-root et vérifie que tu es bien sur la branche Antoinev2.\n"
            f"Erreur originale : {e}"
        )

    predictions_rel = []
    predictions_sent = []
    raw_outputs = []
    raw_scores = []
    times = []

    for i, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        title = str(row["title"]).strip()
        content = str(row["content"]).strip()
        url = str(row["url"]).strip() if "url" in df.columns else f"benchmark_maison_row_{i}"

        pred_rel = predict_relevance_keywords(ticker, title, content)

        start = time.perf_counter()

        try:
            result = analyser_article(
                url=url,
                ticker=ticker,
                title=title,
                content=content
            )
            elapsed = time.perf_counter() - start

            if isinstance(result, dict):
                raw_sentiment = (
                    result.get("sentiment")
                    or result.get("label")
                    or result.get("prediction")
                    or result.get("polarity")
                    or "unknown"
                )
                raw_score = result.get("score", None)
            else:
                raw_sentiment = result
                raw_score = None

            pred_sent = normalize_sentiment(raw_sentiment)
            raw_outputs.append(str(result))
            raw_scores.append(raw_score)

        except Exception as e:
            elapsed = time.perf_counter() - start
            pred_sent = "unknown"
            raw_outputs.append(f"error: {e}")
            raw_scores.append(None)

        predictions_rel.append(pred_rel)
        predictions_sent.append(pred_sent)
        times.append(elapsed)

        if (i + 1) % 5 == 0:
            avg_time = sum(times) / len(times)
            remaining = len(df) - (i + 1)
            estimated_min = remaining * avg_time / 60
            print(
                f"{i + 1}/{len(df)} articles traités "
                f"| {avg_time:.2f}s/article "
                f"| reste estimé : {estimated_min:.1f} min"
            )

    df["pred_relevance"] = predictions_rel
    df["pred_sentiment"] = predictions_sent
    df["method_raw_output"] = raw_outputs
    df["method_score"] = raw_scores
    df["inference_time_sec"] = times

    return df

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", default="benchmark_v1_articles.csv")
    parser.add_argument("--method", required=True, choices=[
        "baseline_neutral",
        "baseline_keywords",
        "finbert",
        "finbert_twitter",
        "samuel",
        "poc_filtrage",
        "antoinev2",
    ])
    parser.add_argument("--output", default=None)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.70)

    args = parser.parse_args()

    df = read_benchmark_csv(args.input)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print(f"Articles chargés : {len(df)}")
    print(f"Méthode testée : {args.method}")

    if args.method == "baseline_neutral":
        df = run_baseline_neutral(df)

    elif args.method == "baseline_keywords":
        df = run_baseline_keywords(df)

    elif args.method == "finbert":
        df = run_hf_classifier(df, "ProsusAI/finbert", "finbert")

    elif args.method == "finbert_twitter":
        df = run_hf_classifier(
            df,
            "nickmuchi/finbert-tone-finetuned-fintwitter-classification",
            "finbert_twitter"
        )

    elif args.method == "samuel":
        df = run_samuel(df, repo_root=args.repo_root)

    elif args.method == "poc_filtrage":
        df = run_poc_filtrage(df, threshold=args.threshold)

    elif args.method == "antoinev2":
        df = run_antoinev2(df, repo_root=args.repo_root)

    output_file = args.output or f"benchmark_maison_{args.method}_predictions.csv"

    evaluate_and_save(df, args.method, output_file)


if __name__ == "__main__":
    main()