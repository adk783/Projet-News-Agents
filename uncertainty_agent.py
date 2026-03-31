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
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

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
    Calcule un score d'incertitude [0, 1] basé sur plusieurs signaux :
    1. Ratio de mots d'incertitude (lexique Loughran-McDonald)
    2. Présence de phrases conditionnelles (if/whether/could/might...)
    3. Hedging patterns (expressions de couverture)
    4. Ancrage factuel (chiffres concrets réduisent l'incertitude)

    Distribution cible :
    - Articles factuels (résultats, chiffres) → 0.00 - 0.20
    - Articles mixtes (analyse + données) → 0.20 - 0.50
    - Articles spéculatifs modérés → 0.50 - 0.75
    - Articles très incertains → 0.75 - 1.00
    """
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if len(words) == 0:
        return 0.0

    text_lower = text.lower()
    n_words = len(words)

    # ── Signal 1 : Ratio de mots d'incertitude (poids 50%) ──
    uncertainty_count = sum(1 for w in words if w in lexicon)
    raw_ratio = uncertainty_count / n_words
    # Saturation à 2.5% (réaliste pour des articles financiers réels)
    SATURATION_RATIO = 0.025
    lexicon_score = min(raw_ratio / SATURATION_RATIO, 1.0)

    # ── Signal 2 : Phrases conditionnelles et modales (poids 25%) ──
    CONDITIONAL_PATTERNS = [
        r'\bif\b', r'\bwhether\b', r'\bcould\b', r'\bmight\b', r'\bwould\b',
        r'\bshould\b', r'\bperhaps\b', r'\bpossibly\b', r'\bpotentially\b',
        r'\bit remains to be seen\b', r'\bremains unclear\b', r'\bhard to say\b',
        r'\bdifficult to predict\b', r'\btime will tell\b',
        r'\bon the other hand\b', r'\bhowever\b', r'\balthough\b',
        r'\bdespite\b', r'\bnevertheless\b', r'\byet\b',
        r'\bwe believe\b', r'\bwe expect\b', r'\bwe anticipate\b',
        r'\banalysts expect\b', r'\banalysts predict\b',
        r'\bgoing forward\b', r'\bin the near term\b', r'\bin the long run\b',
    ]
    conditional_hits = sum(1 for pat in CONDITIONAL_PATTERNS if re.search(pat, text_lower))
    # Normaliser : 6+ patterns trouvés = score conditionnel maximal
    conditional_score = min(conditional_hits / 6.0, 1.0)

    # ── Signal 3 : Hedging / couverture (poids 15%) ──
    HEDGING_PATTERNS = [
        r'\bto some extent\b', r'\bmore or less\b', r'\bbroadly speaking\b',
        r'\bgenerally\b', r'\btypically\b', r'\btends to\b',
        r'\bmay or may not\b', r'\bnot necessarily\b', r'\bnot always\b',
        r'\bsubject to\b', r'\bcontingent upon\b', r'\bdepending on\b',
        r'\bassuming that\b', r'\bprovided that\b', r'\bunless\b',
        r'\bbarring\b', r'\babsent\b',
    ]
    hedging_hits = sum(1 for pat in HEDGING_PATTERNS if re.search(pat, text_lower))
    hedging_score = min(hedging_hits / 4.0, 1.0)

    # ── Signal 4 : Ancrage factuel (réduit l'incertitude, poids -10%) ──
    # Les articles avec beaucoup de données concrètes sont plus factuels
    number_matches = re.findall(r'\$[\d,.]+|\d+\.?\d*\s*%|\d{1,3}(?:,\d{3})+|\b\d+\.\d+\b', text)
    factual_density = len(number_matches) / max(n_words / 50.0, 1.0)  # chiffres par ~50 mots
    factual_penalty = min(factual_density / 3.0, 0.3)  # max 0.3 de réduction

    # ── Combinaison pondérée ──
    raw_score = (
        0.50 * lexicon_score +
        0.25 * conditional_score +
        0.15 * hedging_score
        - 0.10 * factual_penalty  # les chiffres concrets réduisent l'incertitude
    )

    # Clamp à [0, 1]
    raw_score = max(0.0, min(raw_score, 1.0))

    # ── Transformation non-linéaire (sqrt) pour étaler la distribution ──
    # Sans cette transformation, les scores se concentrent trop au milieu
    score = math.sqrt(raw_score)

    # Clamp final
    score = max(0.0, min(score, 1.0))

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
    Génère des données synthétiques d'entraînement.
    Grand mix de textes à haute, moyenne et basse incertitude financière
    pour créer un signal d'entraînement fort et varié.
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

        "It is unclear whether the proposed legislation could pass. Speculation mounts that "
        "the uncertain regulatory environment might destabilize several vulnerable sectors. "
        "Forecasts vary wildly and assumptions about consumer confidence remain untested.",

        "Doubts persist about the company's ability to meet its projections amid volatile "
        "currency markets. The risks are unpredictable, and the likelihood of a downturn "
        "depends on contingent geopolitical developments that seem increasingly unstable.",

        "There is considerable uncertainty surrounding the merger's outcome. Rumors suggest "
        "possible delays, and estimates of synergy savings are speculative. Turbulent bond "
        "markets and fluctuating interest rates add further unpredictability to forecasts.",

        "Analysts hesitate to issue definitive guidance given the volatile macroeconomic "
        "backdrop. Assumptions underlying current valuations may prove unreliable if "
        "unforeseen risks materialize. The probability of recession remains an uncertain "
        "variable that could suddenly alter investment outlooks.",

        "The company cautioned that preliminary results might deviate from projections due "
        "to unpredictable supply chain disruptions and fluctuating raw material costs. Risk "
        "exposure is heightened by vulnerability to sudden policy shifts and uncertain demand.",
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

        "Tesla delivered 1.2 million vehicles in Q4, a 12% increase from the previous quarter. "
        "Net income rose to $3.4 billion. The Gigafactory in Texas reached full production "
        "capacity, and the company confirmed its 2027 product roadmap on schedule.",

        "Apple reported record iPhone sales of 89 million units. Services revenue reached "
        "$25.3 billion, growing 18% year-over-year. The company authorized an additional "
        "$110 billion share repurchase program and raised its quarterly dividend by 4%.",

        "The S&P 500 closed at a new all-time high of 5,842 points. All eleven sectors "
        "finished in positive territory. Trading volume was 11.2 billion shares, above the "
        "20-day average. The VIX index dropped to 12.3, its lowest level this year.",

        "Microsoft announced Q3 revenue of $62.0 billion, up 17% year-over-year. Azure cloud "
        "revenue grew 29%. Operating income increased to $27.6 billion with margins expanding "
        "to 44.5%. The company raised full-year guidance based on strong enterprise demand.",

        "Alphabet reported advertising revenue of $68.1 billion, beating estimates by $2.4 billion. "
        "YouTube ad revenue surged 21% to $9.2 billion. Google Cloud turned profitable for the "
        "third consecutive quarter with $1.2 billion in operating income.",
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

        "The pharmaceutical company received FDA approval for its new treatment. Revenue impact "
        "is estimated at $2-4 billion annually, though adoption rates may vary across markets. "
        "Some analysts suggest the competitive landscape could shift if rival treatments emerge.",

        "Bank earnings beat expectations, driven by higher net interest income. However, "
        "management warned that loan growth might slow if economic conditions soften, and "
        "provisions for credit losses were increased as a precautionary measure.",

        "The energy sector rallied on production cuts, though sustainability of price gains "
        "depends on geopolitical factors. OPEC forecasts suggest demand growth, while some "
        "independent analysts predict a possible oversupply scenario by mid-year.",

        "Retail chains reported better-than-expected holiday sales. E-commerce grew 22%, "
        "but brick-and-mortar stores faced variable performance across regions. Inventory "
        "levels suggest cautious optimism, though consumer confidence surveys show mixed signals.",
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
# 5. DATA AUGMENTATION
# ══════════════════════════════════════════════

def augment_text(text: str) -> str:
    """
    Augmentation de texte pour enrichir le dataset :
    - Mélange aléatoire de phrases (shuffle)
    - Suppression aléatoire de mots (dropout 10%)
    """
    import random as _rng

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    # 50% chance de mélanger l'ordre des phrases
    if len(sentences) > 2 and _rng.random() < 0.5:
        _rng.shuffle(sentences)

    result = " ".join(sentences)

    # 10% word dropout (suppression aléatoire de mots)
    words = result.split()
    if len(words) > 10:
        words = [w for w in words if _rng.random() > 0.10]
    result = " ".join(words)

    return result


# ══════════════════════════════════════════════
# 6. ENTRAÎNEMENT
# ══════════════════════════════════════════════

CSV_TRAINING_DATA = "./training_data.csv"

def train_model(
    db_path: str = "news_database.db",
    output_dir: str = MODEL_OUTPUT_DIR,
    epochs: int = 50,
    batch_size: int = 16,
    learning_rate: float = 3e-5,
    max_samples: int = None,
):
    """
    Pipeline complet d'entraînement amélioré :
    1. Charge le lexique
    2. Combine les données : articles réels + CSV synthétique massif + inline
    3. Applique la data augmentation (x3 le dataset)
    4. Construit le modèle FinBERT + LoRA
    5. Entraîne en régression (MSE) — 50 époques, GPU optimisé
    6. Sauvegarde le modèle fusionné
    """
    import random as _rng
    _rng.seed(42)

    # 1. Lexique
    lexicon = download_lm_uncertainty_lexicon()

    # 2a. Weak labels depuis les vrais articles
    df = generate_weak_labels(db_path, lexicon, max_samples=max_samples)

    # 2b. Données synthétiques inline (27 exemples)
    synthetic_df = _generate_synthetic_data(lexicon)
    synthetic_df["text"] = synthetic_df["content"]
    synthetic_df["score"] = synthetic_df["text"].apply(
        lambda t: compute_uncertainty_score(t, lexicon)
    )
    synthetic_df = synthetic_df[["text", "score"]]
    df = pd.concat([df, synthetic_df], ignore_index=True)

    # 2c. Dataset CSV massif (600+ exemples) si disponible
    if os.path.exists(CSV_TRAINING_DATA):
        logger.info(f"Chargement du dataset CSV : {CSV_TRAINING_DATA}")
        csv_df = pd.read_csv(CSV_TRAINING_DATA)
        csv_df["score"] = csv_df["text"].apply(
            lambda t: compute_uncertainty_score(t, lexicon)
        )
        csv_df = csv_df[["text", "score"]]
        df = pd.concat([df, csv_df], ignore_index=True)
        logger.info(f"  {len(csv_df)} textes chargés depuis le CSV")
    else:
        logger.warning(f"CSV non trouvé : {CSV_TRAINING_DATA}. Lancez generate_training_data.py d'abord.")

    logger.info(f"Dataset avant augmentation : {len(df)} échantillons")
    logger.info(f"  Score moyen : {df['score'].mean():.4f}")
    logger.info(f"  Score min   : {df['score'].min():.4f}")
    logger.info(f"  Score max   : {df['score'].max():.4f}")

    # 3. Data augmentation — doubler le dataset avec des variantes
    augmented_texts = []
    augmented_scores = []
    for _, row in df.iterrows():
        # Version originale
        augmented_texts.append(row["text"])
        augmented_scores.append(row["score"])
        # 2 variantes augmentées par texte
        for _ in range(2):
            aug_text = augment_text(row["text"])
            aug_score = compute_uncertainty_score(aug_text, lexicon)
            augmented_texts.append(aug_text)
            augmented_scores.append(aug_score)

    df_augmented = pd.DataFrame({"text": augmented_texts, "score": augmented_scores})
    logger.info(f"Dataset après augmentation (x3) : {len(df_augmented)} échantillons")
    logger.info(f"  Score moyen : {df_augmented['score'].mean():.4f}")
    logger.info(f"  Score min   : {df_augmented['score'].min():.4f}")
    logger.info(f"  Score max   : {df_augmented['score'].max():.4f}")

    # 4. Modèle
    model, tokenizer = build_model()

    # 5. Dataset + Split
    dataset = UncertaintyDataset(
        texts=df_augmented["text"].tolist(),
        scores=df_augmented["score"].tolist(),
        tokenizer=tokenizer,
    )

    train_size = int(0.85 * len(dataset))
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

    # 6. Training arguments — optimisé pour RTX 4080 GPU (16GB VRAM)
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        logger.info(f"🚀 GPU détecté : {torch.cuda.get_device_name(0)}")
        logger.info(f"   VRAM totale : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("⚠️ Pas de GPU détecté — entraînement sur CPU (lent)")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,   # Eval peut être plus gros
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=1,               # Batch direct = batch_size
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,                             # Log plus fréquent
        save_total_limit=2,
        fp16=use_gpu,
        dataloader_num_workers=0,
        report_to="none",
        seed=42,
        disable_tqdm=False,                           # Barre de progression visible
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info(f"Début de l'entraînement — {epochs} époques, batch={batch_size}, GPU={use_gpu}")
    trainer.train()

    # 8. Sauvegarde — on merge les adaptateurs LoRA dans le modèle de base
    logger.info(f"Fusion des adaptateurs LoRA et sauvegarde dans {output_dir}/")
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("✅ Modèle fusionné et sauvegardé !")
    except Exception as e:
        logger.warning(f"Merge LoRA a échoué ({e}), sauvegarde du modèle PEFT directement...")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("✅ Modèle PEFT sauvegardé (sans merge) !")

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
            # Clamp direct : le modèle est entraîné en régression MSE,
            # les logits sont déjà calibrés dans [0, 1]
            score = logit.clamp(0.0, 1.0).item()

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
                # Clamp direct : le modèle est entraîné en régression MSE,
                # les logits sont déjà calibrés dans [0, 1]
                batch_scores = logits.clamp(0.0, 1.0).tolist()

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
        default=50,
        help="Nombre d'époques d'entraînement",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Taille des batches",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
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
