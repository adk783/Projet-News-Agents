"""
Générateur de données d'entraînement massif pour l'agent d'incertitude.
Produit un CSV avec 500+ textes financiers synthétiques couvrant
tout le spectre d'incertitude [0, 1].

Usage:
    python generate_training_data.py
    → Crée training_data.csv dans le dossier courant
"""

import csv
import random
import re
import math

# ─── Templates de phrases financières ───

# TRÈS HAUTE INCERTITUDE (score cible ~0.7-1.0)
VERY_HIGH_TEMPLATES = [
    "It remains highly uncertain whether {company} can achieve its {metric} targets amid volatile {market} conditions. Analysts speculate that unpredictable {risk} could destabilize the entire sector, raising serious doubts about the sustainability of recent gains.",
    "The probability of {company} meeting its forecasts is increasingly questionable. Rumors of possible {event} have created turbulence, and preliminary estimates suggest highly variable outcomes. The likelihood of success depends on several contingent and undefined factors.",
    "{company}'s future is shrouded in uncertainty as erratic {market} patterns and fluctuating {commodity} prices create instability. Forecasts vary wildly, and analysts doubt whether projected {metric} figures are achievable given the unpredictable regulatory environment.",
    "Investors remain hesitant about {company} amid speculation of {event}. The risks are unpredictable, assumptions about {factor} are untested, and vulnerability to sudden {market} shifts makes any projection roughly approximate at best.",
    "There is considerable uncertainty surrounding {company}'s outlook. Volatile {commodity} markets and unpredictable {risk} make reliable estimation nearly impossible. Doubts persist about whether preliminary {metric} assumptions will deviate substantially from actual results.",
    "The speculative nature of {company}'s {event} creates exposure to volatile and uncertain market dynamics. Fluctuating {commodity} costs and unstable {factor} conditions suggest that forecasts might prove unreliable. Risk factors remain largely undefined and unpredictable.",
    "Management cautiously acknowledged that {company}'s vulnerability to sudden economic shifts and erratic {market} behavior creates substantial uncertainty. The likelihood of achieving {metric} targets is contingent on several factors that remain unclear and possibly unresolvable.",
    "{company} faces an uncertain future as turbulent {market} conditions and fluctuating {factor} dynamics raise doubts about projected growth. Speculation about possible {event} has created hesitancy among investors, with risk exposure at unprecedented levels.",
]

# HAUTE INCERTITUDE (score cible ~0.4-0.7)
HIGH_TEMPLATES = [
    "{company} faces uncertain prospects as {market} volatility continues. While recent {metric} showed some improvement, analysts suggest that risks from {risk} could impact future performance. The outlook depends on several factors that remain unclear.",
    "The possibility of {event} at {company} has raised concerns among investors. Forecasts suggest variable outcomes, and the risk of {risk} adds uncertainty to an already fluctuating market environment.",
    "Speculation about {company}'s {metric} trajectory has intensified. While some analysts predict growth, others doubt the assumptions underlying current projections. Market conditions remain somewhat unpredictable.",
    "{company}'s exposure to {risk} could destabilize its {metric} performance. Preliminary estimates suggest the impact might vary considerably depending on {factor} developments that are difficult to forecast.",
    "Analysts express caution about {company}'s near-term outlook, citing volatile {market} conditions and uncertain {factor} trends. The probability of meeting {metric} targets may depend on unpredictable external factors.",
    "Investors are weighing the risks and uncertainties facing {company}. While the company's fundamentals are solid, the volatile {market} environment and possible {event} create a somewhat unpredictable investment landscape.",
]

# INCERTITUDE MOYENNE (score cible ~0.2-0.4)
MEDIUM_TEMPLATES = [
    "{company} reported mixed results, with {metric} meeting expectations but {factor} showing some variability. Management indicated that future performance may be affected by evolving market conditions, though the core business remains stable.",
    "The outlook for {company} is cautiously optimistic. While {metric} growth has been steady, some analysts note that {risk} could introduce variability. Overall, the company's position appears solid with some room for uncertainty.",
    "{company}'s {metric} came in slightly above estimates, though management cautioned that {factor} trends may shift. The company maintains a strong balance sheet, but some risks from {market} dynamics bear watching.",
    "While {company} has demonstrated resilience, the broader {market} environment introduces some variability. {metric} projections assume stable conditions, though {factor} developments could alter the trajectory somewhat.",
    "Analysts view {company} as well-positioned despite some uncertainty in {market} conditions. Recent {metric} data suggests continued momentum, though {risk} factors warrant monitoring over the coming quarters.",
    "{company} delivered a solid quarter, though the {factor} outlook contains some unknowns. Management expects stable {metric} growth but acknowledged that global {market} conditions could introduce modest variability.",
]

# BASSE INCERTITUDE (score cible ~0.05-0.2)
LOW_TEMPLATES = [
    "{company} delivered strong {metric} results, with revenue growing {pct}% year-over-year. The company reaffirmed its guidance for the full year, citing robust demand across key segments and stable operating conditions.",
    "{company}'s {metric} exceeded expectations, driven by continued strength in its core business. Management raised full-year guidance and announced a {pct}% increase in its quarterly dividend, signaling confidence in sustained growth.",
    "The board of {company} approved a new share repurchase program worth ${amount} billion following strong {metric} performance. Operating margins expanded to {pct}%, reflecting efficient cost management and steady demand.",
    "{company} completed its acquisition of {target} ahead of schedule, with integration milestones achieved on time. The combined entity now operates across {num} markets with annual revenue of ${amount} billion.",
    "Annual {metric} for {company} increased to ${amount} billion, marking the {ordinal} consecutive year of growth. The company generated strong free cash flow, enabling debt reduction and strategic investments.",
    "{company} reported record quarterly {metric}, with all business segments contributing to growth. The company's backlog reached ${amount} billion, providing strong revenue visibility for the next {num} quarters.",
]

# TRÈS BASSE INCERTITUDE (score cible ~0.0-0.05)
VERY_LOW_TEMPLATES = [
    "{company} reported Q{q} revenue of ${amount} billion, up {pct}% from the prior year. Net income was ${income} billion. The company declared a quarterly dividend of ${div} per share, payable on {date}.",
    "{company} completed the sale of its {division} division for ${amount} billion in cash. The transaction closed on {date}, and proceeds will be used for debt repayment and share buybacks.",
    "The {index} closed at {points} points, {direction} {pct}% on the day. Trading volume was {volume} billion shares. The {sector} sector led gains with a {spct}% advance.",
    "{company} opened its new {facility} in {city}, creating {num} jobs. The ${amount} billion facility will manufacture {product} at full capacity by {date}.",
    "{company}'s board elected {name} as the new CEO, effective {date}. {name} previously served as COO for {num} years and oversaw the company's expansion into {num2} international markets.",
    "The Federal Reserve held interest rates steady at {rate}% following its {month} meeting. The decision was unanimous. GDP growth was confirmed at {gdp}% for the quarter.",
]

# ─── Fill-in values ───

COMPANIES = [
    "Apple", "Microsoft", "Google", "Amazon", "Tesla", "Meta", "NVIDIA",
    "JPMorgan", "Goldman Sachs", "Bank of America", "Berkshire Hathaway",
    "Johnson & Johnson", "Pfizer", "ExxonMobil", "Chevron", "Walmart",
    "Visa", "Mastercard", "Netflix", "Adobe", "Salesforce", "Intel",
    "AMD", "Qualcomm", "Broadcom", "Oracle", "IBM", "Cisco",
    "The Walt Disney Company", "Coca-Cola", "PepsiCo", "Nike", "Procter & Gamble",
    "UnitedHealth", "Merck", "Alibaba", "Samsung", "Toyota", "Shell",
]

METRICS = [
    "revenue", "earnings", "profit margin", "operating income", "net income",
    "EBITDA", "free cash flow", "EPS", "gross margin", "sales",
    "market share", "user growth", "subscriber count", "order backlog",
]

MARKETS = [
    "equity", "bond", "commodity", "currency", "crypto", "housing",
    "energy", "semiconductor", "technology", "emerging market",
    "fixed income", "derivatives",
]

RISKS = [
    "geopolitical tensions", "regulatory changes", "trade war escalation",
    "interest rate hikes", "inflation spikes", "supply chain disruption",
    "cyberattack threats", "climate regulation", "antitrust action",
    "currency devaluation", "credit default", "pandemic resurgence",
]

EVENTS = [
    "a major restructuring", "a hostile takeover bid", "a stock split",
    "executive departures", "a strategic pivot", "asset write-downs",
    "a debt downgrade", "regulatory investigation", "a merger",
    "layoffs", "a product recall", "patent litigation",
]

FACTORS = [
    "consumer demand", "input cost", "labor market", "exchange rate",
    "raw material pricing", "competitive", "macroeconomic",
    "interest rate", "supply chain", "regulatory", "AI adoption",
]

COMMODITIES = [
    "oil", "gas", "copper", "lithium", "gold", "silver",
    "wheat", "steel", "aluminum", "rare earth", "silicon",
]

NAMES = ["John Smith", "Sarah Chen", "Michael Roberts", "Lisa Park", "David Kumar"]
CITIES = ["Austin", "Dublin", "Singapore", "Munich", "Bangalore", "Shanghai"]
SECTORS = ["technology", "healthcare", "energy", "financial", "consumer", "industrial"]
MONTHS = ["January", "March", "June", "September", "December"]
INDICES = ["S&P 500", "Nasdaq Composite", "Dow Jones", "FTSE 100", "Nikkei 225"]
ORDINALS = ["third", "fourth", "fifth", "sixth", "seventh", "eighth"]
PRODUCTS = ["semiconductors", "batteries", "electric vehicles", "medical devices", "solar panels"]


def fill_template(template: str) -> str:
    """Remplace les placeholders par des valeurs aléatoires."""
    replacements = {
        "{company}": random.choice(COMPANIES),
        "{metric}": random.choice(METRICS),
        "{market}": random.choice(MARKETS),
        "{risk}": random.choice(RISKS),
        "{event}": random.choice(EVENTS),
        "{factor}": random.choice(FACTORS),
        "{commodity}": random.choice(COMMODITIES),
        "{pct}": str(random.randint(3, 35)),
        "{spct}": f"{random.uniform(0.5, 4.5):.1f}",
        "{amount}": f"{random.uniform(1.0, 95.0):.1f}",
        "{income}": f"{random.uniform(0.5, 15.0):.1f}",
        "{div}": f"{random.uniform(0.20, 2.50):.2f}",
        "{num}": str(random.randint(3, 150)),
        "{num2}": str(random.randint(5, 50)),
        "{q}": str(random.randint(1, 4)),
        "{date}": f"{random.choice(MONTHS)} {random.randint(1, 28)}, {random.choice(['2025', '2026'])}",
        "{direction}": random.choice(["up", "down"]),
        "{volume}": f"{random.uniform(5.0, 15.0):.1f}",
        "{points}": str(random.randint(4000, 6500)),
        "{rate}": f"{random.uniform(3.5, 5.75):.2f}",
        "{gdp}": f"{random.uniform(1.5, 4.5):.1f}",
        "{month}": random.choice(MONTHS),
        "{index}": random.choice(INDICES),
        "{sector}": random.choice(SECTORS),
        "{name}": random.choice(NAMES),
        "{city}": random.choice(CITIES),
        "{ordinal}": random.choice(ORDINALS),
        "{facility}": random.choice(["manufacturing plant", "data center", "research lab", "distribution hub"]),
        "{product}": random.choice(PRODUCTS),
        "{division}": random.choice(["media", "logistics", "cloud", "hardware", "retail"]),
        "{target}": random.choice(COMPANIES),
    }

    result = template
    for key, value in replacements.items():
        result = result.replace(key, value)
    return result


def generate_dataset(n_per_category: int = 120) -> list:
    """Génère un dataset équilibré avec des textes pour chaque niveau d'incertitude."""
    data = []
    
    categories = [
        (VERY_HIGH_TEMPLATES, "very_high"),
        (HIGH_TEMPLATES, "high"),
        (MEDIUM_TEMPLATES, "medium"),
        (LOW_TEMPLATES, "low"),
        (VERY_LOW_TEMPLATES, "very_low"),
    ]

    for templates, level in categories:
        for _ in range(n_per_category):
            template = random.choice(templates)
            text = fill_template(template)
            data.append({"text": text, "level": level})

    random.shuffle(data)
    return data


def main():
    random.seed(42)
    
    # 120 textes × 5 niveaux = 600 textes synthétiques
    data = generate_dataset(n_per_category=120)
    
    output_file = "training_data.csv"
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "level"])
        writer.writeheader()
        writer.writerows(data)

    # Stats
    print(f"✅ Dataset généré : {len(data)} textes")
    from collections import Counter
    counts = Counter(d["level"] for d in data)
    for level, count in sorted(counts.items()):
        print(f"  {level}: {count}")
    print(f"\n→ Fichier sauvegardé : {output_file}")


if __name__ == "__main__":
    main()
