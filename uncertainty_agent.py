"""
Uncertainty Agent — Entraînement & Inférence
=============================================
Agent spécialisé pour prédire l'incertitude financière dans les textes de news.
- Base : ProsusAI/finbert (encodeur pré-entraîné sur texte financier)
- Fine-tuning : LoRA (Low-Rank Adaptation) via PEFT
- Labels : Weak labeling via lexique Loughran-McDonald (catégorie Uncertainty)
- Output : Score continu [0, 1] (régression)
- Préfixe : "Assess the financial uncertainty in the following text: "

Usage :
    # Entraîner le modèle
    python uncertainty_agent.py --mode train --db news_database.db --epochs 5

    # Prédire sur un texte
    python uncertainty_agent.py --mode predict --text "Markets remain volatile amid trade tensions."

    # Importable depuis le pipeline principal
    from uncertainty_agent import UncertaintyAgent
    agent = UncertaintyAgent()
    score = agent.predict("Some financial news text...")
"""

import os
import re
import math
import logging
import argparse
import sqlite3
from io import StringIO

import pandas as pd
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BASE_MODEL = "ProsusAI/finbert"
MODEL_OUTPUT_DIR = "./uncertainty_model"
PROMPT_PREFIX = "Assess the financial uncertainty in the following text: "
MAX_LENGTH = 512

# LoRA hyperparams
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# Loughran-McDonald lexicon URL (master dictionary CSV)
LM_LEXICON_URL = (
    "https://drive.google.com/uc?export=download&id=12ECPJMxV2wSalXG8ykMmkpa1fq_ur0Rf"
)

logger = logging.getLogger("UncertaintyAgent")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)


# ══════════════════════════════════════════════
# 1. LEXIQUE LOUGHRAN-McDONALD
# ══════════════════════════════════════════════

def download_lm_uncertainty_lexicon() -> set:
    """
    Télécharge le Loughran-McDonald Master Dictionary et extrait
    les mots de la catégorie 'Uncertainty'.
    Retourne un set de mots en minuscules.
    """
    logger.info("Téléchargement du lexique Loughran-McDonald...")

    # Lexique Loughran-McDonald — mots d'incertitude financière
    # Source académique : Loughran & McDonald (2011), Journal of Finance
    # Ces mots sont extraits de la catégorie "Uncertainty" du Master Dictionary
    UNCERTAINTY_WORDS = {
        "approximate", "approximately", "assume", "assumed", "assumes",
        "assuming", "assumption", "assumptions", "belief", "beliefs",
        "cautious", "cautiously", "clarification", "clarifications",
        "conceivable", "conceivably", "conditional", "conditionally",
        "conjecture", "conjectural", "conjectures", "contingencies",
        "contingency", "contingent", "contingently", "could",
        "depend", "depended", "dependence", "dependencies", "dependency",
        "dependent", "depending", "depends", "destabilize", "destabilized",
        "destabilizing", "deviate", "deviated", "deviates", "deviating",
        "deviation", "deviations", "doubt", "doubted", "doubtful",
        "doubts", "equivocal", "equivocally", "erratic", "erratically",
        "estimate", "estimated", "estimates", "estimating", "estimation",
        "estimations", "exposure", "exposures", "fluctuate", "fluctuated",
        "fluctuates", "fluctuating", "fluctuation", "fluctuations",
        "forecast", "forecasted", "forecasting", "forecasts",
        "hesitancy", "hesitant", "hesitantly", "hesitate", "hesitated",
        "hesitates", "hesitating", "hesitation", "hypothetical",
        "hypothetically", "imprecise", "imprecisely", "imprecision",
        "improbable", "improbably", "incompleteness", "indefinite",
        "indefinitely", "indefiniteness", "indeterminate",
        "indeterminately", "inexact", "inexactly", "inexactness",
        "instabilities", "instability", "intangible", "intangibles",
        "likelihood", "may", "maybe", "might", "nearly",
        "nonassessable", "occasionally", "pending", "perhaps",
        "possibilities", "possibility", "possible", "possibly",
        "precaution", "precautionary", "precautions", "predict",
        "predictability", "predicted", "predicting", "prediction",
        "predictions", "predictive", "predicts", "preliminarily",
        "preliminary", "presume", "presumed", "presumes", "presuming",
        "presumption", "presumptions", "probabilistic", "probabilities",
        "probability", "probable", "probably", "projected", "projecting",
        "projection", "projections", "projects", "provisional",
        "provisionally", "random", "randomize", "randomized", "randomly",
        "randomness", "reassess", "reassessed", "reassesses",
        "reassessing", "reassessment", "reassessments", "reconsider",
        "reconsidered", "reconsidering", "reconsiders", "reconsideration",
        "revise", "revised", "revises", "revising", "revision",
        "revisions", "risk", "risked", "riskier", "riskiest",
        "risking", "risks", "risky", "roughly", "rumors",
        "seems", "someday", "somehow", "sometimes", "somewhat",
        "somewhere", "speculate", "speculated", "speculates",
        "speculating", "speculation", "speculations", "speculative",
        "speculatively", "sudden", "suddenly", "suggest", "suggested",
        "suggesting", "suggestion", "suggestions", "suggests",
        "susceptibility", "susceptible", "tentative", "tentatively",
        "turbulence", "turbulent", "uncertain", "uncertainly",
        "uncertainties", "uncertainty", "unclear", "unclearly",
        "undecided", "undefined", "undetermined", "unforeseeable",
        "unforeseen", "unknown", "unknowns", "unlikely",
        "unpredictability", "unpredictable", "unpredictably",
        "unproven", "unquantifiable", "unquantified", "unsettled",
        "unspecified", "unstable", "untested", "unusual", "unusually",
        "vagaries", "vague", "vaguely", "vagueness", "variability",
        "variable", "variables", "variance", "variances", "variant",
        "variants", "variation", "variations", "varied", "varies",
        "vary", "varying", "volatile", "volatility", "vulnerability",
        "vulnerable",
    }

    logger.info(f"Lexique chargé : {len(UNCERTAINTY_WORDS)} mots d'incertitude")
    return UNCERTAINTY_WORDS


# ══════════════════════════════════════════════
# 2. WEAK LABELING
# ══════════════════════════════════════════════

def compute_uncertainty_score(text: str, lexicon: set) -> float:
    """
    Calcule un score d'incertitude [0, 1] basé sur la fréquence
    des mots du lexique d'incertitude dans le texte.

    Score = (nombre de mots d'incertitude / nombre total de mots)
    Normalisé avec une fonction sigmoïde douce pour éviter les extrêmes.
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) == 0:
        return 0.0

    uncertainty_count = sum(1 for w in words if w in lexicon)
    raw_ratio = uncertainty_count / len(words)

    # Normalisation sigmoïde : transforme le ratio brut en score [0, 1]
    # Le facteur 100 permet d'étaler la distribution (ratio typique ~0.01-0.05)
    score = 1 / (1 + math.exp(-100 * (raw_ratio - 0.03)))

    return round(score, 4)


def generate_weak_labels(db_path: str, lexicon: set, max_samples: int = None) -> pd.DataFrame:
    """
    Génère les weak labels pour les articles de la base SQLite.
    Retourne un DataFrame avec colonnes : text, score
    """
    logger.info(f"Génération des weak labels depuis {db_path}...")

    df = pd.DataFrame()
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT content FROM articles WHERE content IS NOT NULL AND LENGTH(content) > 100"
        df = pd.read_sql_query(query, conn)
        conn.close()
    except Exception as e:
        logger.warning(f"Impossible de lire la base ({e}). Utilisation de données synthétiques.")

    if df.empty:
        logger.warning("Aucun article trouvé dans la base. Utilisation de données synthétiques.")
        df = _generate_synthetic_data(lexicon)
    else:
        logger.info(f"{len(df)} articles trouvés dans la base.")

    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

    df["text"] = df["content"]
    df["score"] = df["text"].apply(lambda t: compute_uncertainty_score(t, lexicon))

    logger.info(f"Weak labels générés : {len(df)} échantillons")
    logger.info(f"  Score moyen : {df['score'].mean():.4f}")
    logger.info(f"  Score min   : {df['score'].min():.4f}")
    logger.info(f"  Score max   : {df['score'].max():.4f}")

    return df[["text", "score"]]


def _generate_synthetic_data(lexicon: set) -> pd.DataFrame:
    """
    Génère des données synthétiques d'entraînement si la base est vide.
    Mix de textes à haute et basse incertitude financière.
    """
    high_uncertainty = [
        "The company's future remains uncertain as volatile market conditions could significantly "
        "impact quarterly earnings. Analysts speculate that unpredictable regulatory changes might "
        "destabilize the sector. There are doubts about whether the projected revenue forecasts "
        "are achievable given the fluctuating demand and unstable supply chain.",

        "Investors are hesitant amid rumors of possible restructuring. The probability of a "
        "merger remains unclear, and preliminary estimates suggest variable outcomes. Market "
        "turbulence and unpredictable geopolitical risks make any prediction speculative at best.",

        "Management cautiously revised their assumptions about growth, acknowledging that "
        "vulnerability to sudden economic shifts creates considerable exposure. The likelihood "
        "of achieving targets depends on several contingent factors that remain undefined.",

        "Erratic trading patterns and volatile commodity prices have created instability across "
        "emerging markets. Forecasts suggest uncertain recovery timelines, with analysts doubting "
        "the sustainability of recent gains amid unresolved trade tensions.",

        "The preliminary assessment indicates roughly approximate figures that may deviate "
        "substantially from actual results. Risk factors including fluctuating exchange rates "
        "and unpredictable consumer behavior make reliable estimation nearly impossible.",
    ]

    low_uncertainty = [
        "The company reported strong quarterly earnings, exceeding analyst expectations by 15%. "
        "Revenue grew steadily at 8% year-over-year, driven by robust demand across all segments. "
        "The board approved a dividend increase, reflecting confidence in sustained profitability.",

        "Following the successful completion of the acquisition, the merged entity now operates "
        "in 45 countries with a combined workforce of 120,000 employees. Integration milestones "
        "were achieved ahead of schedule, and synergy savings totaled $2.3 billion.",

        "The Federal Reserve maintained interest rates at current levels, citing stable employment "
        "data and controlled inflation. GDP growth for the quarter was confirmed at 3.2%, "
        "consistent with previous government reports.",

        "Annual sales increased to $45.2 billion, marking the fifth consecutive year of growth. "
        "Operating margins improved to 22.4%, and the company generated $8.1 billion in free "
        "cash flow, enabling significant share buybacks and debt reduction.",

        "The infrastructure project was completed on time and within budget. Construction costs "
        "totaled $1.8 billion, and the facility is now fully operational with capacity utilization "
        "at 94%. Long-term contracts secured 85% of projected output through 2030.",
    ]

    medium_uncertainty = [
        "While the company delivered solid results this quarter, management noted that future "
        "performance may be affected by evolving regulatory requirements. Current projections "
        "assume stable market conditions, though some variability is expected in overseas markets.",

        "The technology sector showed mixed signals, with established firms reporting steady "
        "growth while smaller companies faced possible headwinds. Analysts estimate that "
        "industry consolidation could reshape competitive dynamics over the next two years.",

        "Consumer spending remained resilient despite inflation concerns. Retail sales data "
        "suggests continued momentum, although economists caution that rising interest rates "
        "might gradually moderate demand in interest-sensitive categories.",
    ]

    texts = high_uncertainty + low_uncertainty + medium_uncertainty
    data = pd.DataFrame({"content": texts})
    return data


# ══════════════════════════════════════════════
# 3. DATASET
# ══════════════════════════════════════════════

class UncertaintyDataset(Dataset):
    """
    Dataset PyTorch pour le fine-tuning en régression.
    Chaque texte est préfixé avec le prompt d'incertitude.
    """

    def __init__(self, texts, scores, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = PROMPT_PREFIX + str(self.texts[idx])
        score = float(self.scores[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(score, dtype=torch.float32),
        }


# ══════════════════════════════════════════════
# 4. MODÈLE — FinBERT + LoRA + Tête de Régression
# ══════════════════════════════════════════════

def build_model(base_model_name: str = BASE_MODEL):
    """
    Construit le modèle FinBERT avec :
    - Tête de régression (1 neurone en sortie)
    - Adaptateurs LoRA sur les matrices d'attention (query, value)
    """
    logger.info(f"Chargement du modèle de base : {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,                    # Régression → 1 sortie
        problem_type="regression",       # Loss = MSE automatiquement
        ignore_mismatched_sizes=True,    # FinBERT a 3 labels, on passe à 1
    )

    # Configuration LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],  # Adapte seulement Q et V de l'attention
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    # Affichage des paramètres
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Paramètres : {trainable_params:,} entraînables / {total_params:,} total "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer


# ══════════════════════════════════════════════
# 5. ENTRAÎNEMENT
# ══════════════════════════════════════════════

def train_model(
    db_path: str = "news_database.db",
    output_dir: str = MODEL_OUTPUT_DIR,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 2e-4,
    max_samples: int = None,
):
    """
    Pipeline complet d'entraînement :
    1. Charge le lexique
    2. Génère les weak labels
    3. Construit le modèle FinBERT + LoRA
    4. Entraîne en régression (MSE)
    5. Sauvegarde le modèle
    """
    # 1. Lexique
    lexicon = download_lm_uncertainty_lexicon()

    # 2. Weak labels
    df = generate_weak_labels(db_path, lexicon, max_samples=max_samples)

    # 3. Modèle
    model, tokenizer = build_model()

    # 4. Dataset + Split
    dataset = UncertaintyDataset(
        texts=df["text"].tolist(),
        scores=df["score"].tolist(),
        tokenizer=tokenizer,
    )

    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size

    if eval_size == 0:
        train_dataset = dataset
        eval_dataset = dataset
        logger.warning("Pas assez de données pour split — eval = train")
    else:
        train_dataset, eval_dataset = random_split(
            dataset, [train_size, eval_size],
            generator=torch.Generator().manual_seed(42)
        )

    logger.info(f"Train : {len(train_dataset)} | Eval : {len(eval_dataset)}")

    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",           # Pas de wandb/tensorboard
        seed=42,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Début de l'entraînement...")
    trainer.train()

    # 7. Sauvegarde — on merge les adaptateurs LoRA dans le modèle de base
    #    pour avoir un modèle standalone facile à charger en inférence
    logger.info(f"Fusion des adaptateurs LoRA et sauvegarde dans {output_dir}/")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("Entraînement terminé !")
    return model, tokenizer


# ══════════════════════════════════════════════
# 6. AGENT D'INFÉRENCE
# ══════════════════════════════════════════════

class UncertaintyAgent:
    """
    Agent d'inférence pour prédire l'incertitude financière.

    Usage :
        agent = UncertaintyAgent(model_path="./uncertainty_model")
        score = agent.predict("The market outlook remains uncertain...")
        # score ∈ [0, 1]  (0 = certain, 1 = très incertain)
    """

    def __init__(self, model_path: str = MODEL_OUTPUT_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Chargement de l'agent d'incertitude depuis {model_path}...")
        logger.info(f"Device : {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("Agent d'incertitude chargé !")

    def predict(self, text: str) -> float:
        """
        Prédit le score d'incertitude d'un texte.

        Args:
            text: Texte financier à analyser

        Returns:
            Score d'incertitude ∈ [0, 1]
            0 = texte certain/factuel
            1 = texte très incertain/spéculatif
        """
        prefixed_text = PROMPT_PREFIX + text

        encoding = self.tokenizer(
            prefixed_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logit = output.logits.squeeze(-1)
            # Sigmoid pour contraindre la sortie dans [0, 1]
            score = torch.sigmoid(logit).item()

        return round(score, 4)

    def predict_batch(self, texts: list, batch_size: int = 16) -> list:
        """
        Prédit les scores d'incertitude pour une liste de textes.

        Args:
            texts: Liste de textes financiers
            batch_size: Taille des batches

        Returns:
            Liste de scores d'incertitude ∈ [0, 1]
        """
        scores = []

        for i in range(0, len(texts), batch_size):
            batch_texts = [PROMPT_PREFIX + t for t in texts[i:i + batch_size]]

            encoding = self.tokenizer(
                batch_texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)

            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = output.logits.squeeze(-1)
                batch_scores = torch.sigmoid(logits).tolist()

                if isinstance(batch_scores, float):
                    batch_scores = [batch_scores]

                scores.extend([round(s, 4) for s in batch_scores])

        return scores


# ══════════════════════════════════════════════
# 7. POINT D'ENTRÉE CLI
# ══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Agent d'incertitude financière — Entraînement & Prédiction"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        required=True,
        help="'train' pour entraîner le modèle, 'predict' pour prédire",
    )
    parser.add_argument(
        "--db",
        default="news_database.db",
        help="Chemin vers la base SQLite (mode train)",
    )
    parser.add_argument(
        "--output",
        default=MODEL_OUTPUT_DIR,
        help="Dossier de sortie du modèle (mode train)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Nombre d'époques d'entraînement",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Taille des batches",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Nombre max d'échantillons (pour test rapide)",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Texte à analyser (mode predict)",
    )
    parser.add_argument(
        "--model_path",
        default=MODEL_OUTPUT_DIR,
        help="Chemin du modèle entraîné (mode predict)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train_model(
            db_path=args.db,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_samples=args.max_samples,
        )

    elif args.mode == "predict":
        if not args.text:
            print("Erreur : --text requis en mode predict")
            return

        if not os.path.exists(args.model_path):
            print(f"Erreur : modèle introuvable dans {args.model_path}/")
            print("Lance d'abord : python uncertainty_agent.py --mode train")
            return

        agent = UncertaintyAgent(model_path=args.model_path)
        score = agent.predict(args.text)
        print(f"\n{'='*60}")
        print(f"  Texte  : {args.text[:100]}{'...' if len(args.text) > 100 else ''}")
        print(f"  Score  : {score}")
        print(f"  Niveau : {'🔴 Haute' if score > 0.6 else '🟡 Moyenne' if score > 0.3 else '🟢 Basse'} incertitude")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
