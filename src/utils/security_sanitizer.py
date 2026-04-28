import logging
import math
import re

logger = logging.getLogger(__name__)

# Mots-clés et expressions couramment utilisés dans les prompt injections ou jailbreaks
SUSPICIOUS_PATTERNS = [
    r"(?i)ignore\s+((?:all\s+)?previous\s+)?instructions",
    r"(?i)disregard\s+previous",
    r"(?i)system\s+prompt",
    r"(?i)bypass\s+rules",
    r"(?i)you\s+are\s+now",
    r"(?i)output\s+(?:achat|vente|buy|sell|strong buy|strong sell)",
    r"(?i)confidence\s*(?:=|of|to)\s*100%?",
    r"(?i)ignore\s+le\s+contexte",
    r"(?i)oublie\s+(?:tes\s+)?instructions",
    r"(?i)tu\s+dois\s+(?:absolument|maintenant)\s+dire",
    r"(?i)simulate\s+a\s+scenario",
    r"(?i)act\s+as\s+if",
]


def shannon_entropy(data: str) -> float:
    """Calcule l'entropie de Shannon pour detecter du Base64 ou des payloads chiffrés."""
    if not data:
        return 0.0
    entropy = 0
    for x in set(data):
        p_x = float(data.count(x)) / len(data)
        entropy -= p_x * math.log(p_x, 2)
    return entropy


def check_prompt_injection(texte_article: str) -> bool:
    """
    Analyse le texte de l'article pour détecter des tentatives d'injection de prompt.
    Retourne True si une tentative est détectée (article malveillant), False sinon.
    """
    if not texte_article:
        return False

    # Heuristique 1 : Recherche de patterns typiques de jailbreak
    for pattern in SUSPICIOUS_PATTERNS:
        if re.search(pattern, texte_article):
            logger.warning("[Sanitizer] Pattern d'injection détecté : %s", pattern)
            return True

    # Heuristique 2 : Densité anormale de directives impératives
    imperatives = len(re.findall(r"(?i)\b(must|have to|force|obligatory|require|ignore|always)\b", texte_article))
    word_count = len(texte_article.split())
    if word_count > 0 and (imperatives / word_count) > 0.05:  # Plus de 5% de mots impératifs
        logger.warning("[Sanitizer] Densité anormale de directives détectée.")
        return True

    # Heuristique 3 (Critique C) : Détection de payloads obfusqués (Base64 / Hex)
    import base64

    b64_pattern = r"(?:[A-Za-z0-9+/]{4}){10,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?"
    b64_matches = re.findall(b64_pattern, texte_article)
    for match in b64_matches:
        try:
            decoded = base64.b64decode(match).decode("utf-8", errors="ignore")
            for pattern in SUSPICIOUS_PATTERNS:
                if re.search(pattern, decoded):
                    logger.warning("[Sanitizer] Chaine Base64 malveillante detectee (obfuscation confirmee).")
                    return True
        except Exception:
            pass

    # Heuristique 4 : Entropie textuelle
    words = texte_article.split()
    for w in words:
        if len(w) > 30 and shannon_entropy(w) > 4.5:
            # On ignore si c'est une clef ou hash naturel, mais on reste vigilant
            if not re.match(r"^[A-Fa-f0-9]+$", w):  # si ce n'est pas qu'un hash hex
                logger.warning("[Sanitizer] Haute entropie textuelle detectee dans le mot : %s", w[:15] + "...")
                return True

    return False
