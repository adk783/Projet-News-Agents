import json
import logging
import os
import sys
from pathlib import Path

# Activer les logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SetupBenchmarks")

DATA_DIR = Path(__file__).parent / "data"


def check_and_create_financial_phrasebank():
    fpb_path = DATA_DIR / "financial_phrasebank.jsonl"
    prep_script = Path(__file__).parent / "prepare_financial_phrasebank.py"
    if fpb_path.exists():
        logger.info(f"[OK] FinancialPhraseBank détecté : {fpb_path.name}")
    else:
        logger.warning("[-] FinancialPhraseBank manquant.")
        if prep_script.exists():
            logger.info("-> Exécution du script de préparation local...")
            os.system(f"python {prep_script}")
        else:
            logger.error("-> Script prepare_financial_phrasebank.py introuvable.")


def fetch_fiqa():
    out_path = DATA_DIR / "fiqa_sentiment.jsonl"
    if out_path.exists():
        logger.info(f"[OK] FiQA Sentiment détecté : {out_path.name}")
        return

    logger.info("[+] Téléchargement de FiQA Sentiment via HuggingFace...")
    try:
        from datasets import load_dataset

        # TheFinAI/fiqa-sentiment-classification gives scores between -1 and 1
        ds = load_dataset("TheFinAI/fiqa-sentiment-classification", split="train")

        with open(out_path, "w", encoding="utf-8") as f:
            count = 0
            for item in ds:
                score = float(item["score"])
                if score > 0.1:
                    label = "positive"
                elif score < -0.1:
                    label = "negative"
                else:
                    label = "neutral"

                record = {"sentence": item["sentence"], "label": label}
                f.write(json.dumps(record) + "\n")
                count += 1
        logger.info(f"    -> FiQA généré: {count} exemples.")
    except ImportError:
        logger.error("La librairie 'datasets' est requise. (pip install datasets)")
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement de FiQA: {e}")


def fetch_sentfin():
    out_path = DATA_DIR / "sentfin.jsonl"
    if out_path.exists():
        logger.info(f"[OK] SEntFiN détecté : {out_path.name}")
        return

    logger.info("[+] Téléchargement de SEntFiN via HuggingFace...")
    try:
        from datasets import load_dataset

        # temetnosce01/phrasebank_and_sentfin regroupe FPB et SEntFiN
        ds = load_dataset("temetnosce01/phrasebank_and_sentfin", split="train")

        label_map = {0: "negative", 1: "neutral", 2: "positive"}
        count = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for item in ds:
                if "SentFIN" in str(item.get("Source", "")):
                    lbl = label_map.get(item["Labels"], "neutral")
                    record = {"sentence": item["Sentences"], "label": lbl}
                    f.write(json.dumps(record) + "\n")
                    count += 1
        logger.info(f"    -> SEntFiN généré: {count} exemples.")
    except ImportError:
        logger.error("La librairie 'datasets' est requise. (pip install datasets)")
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement de SEntFiN: {e}")


def notify_manual_datasets():
    logger.info("=" * 50)
    logger.info("Note concernant Kaggle Market Events et FinMarBa :")
    logger.info("Ces bases de données exigent une clé d'API Kaggle ou une authentification.")
    logger.info("Afin de les évaluer, placez vous-même les fichiers 'kaggle_market_events.jsonl' et")
    logger.info("'finmarba.jsonl' dans le dossier 'eval/lm_tasks/data/'. Mappez-les au format:")
    logger.info('{"sentence": "...", "label": "positive|neutral|negative"}')
    logger.info("L'évaluateur les captera automatiquement.")
    logger.info("=" * 50)


def main():
    logger.info("=== Vérification et génération des vrais datasets ===")

    # Nettoyage des vieux mocks pour forcer le téléchargement si nécessaire
    for mock_file in ["fiqa_sentiment.jsonl", "sentfin.jsonl", "kaggle_market_events.jsonl", "finmarba.jsonl"]:
        path = DATA_DIR / mock_file
        # Check if it was purely our mock by seeing total rows
        if path.exists():
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            # If the file has exactly 3/4 lines, it's our mock, delete it to install the real one
            if len(lines) <= 5:
                logger.info(f"[*] Suppression du mock : {mock_file}")
                path.unlink()

    check_and_create_financial_phrasebank()

    # Check if datasets library is installed, if not try to install it
    try:
        import datasets
    except ImportError:
        logger.warning("Installation de 'datasets' requise pour le téléchargement depuis HF Hub...")
        os.system(f"{sys.executable} -m pip install datasets")

    fetch_fiqa()
    fetch_sentfin()
    notify_manual_datasets()
    logger.info("=== Setup terminé. Prêt pour l'évaluation réelle. ===")


if __name__ == "__main__":
    main()
