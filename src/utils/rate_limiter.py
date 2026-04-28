"""
rate_limiter.py — Exponential backoff / retry pour appels LLM

OBJECTIF
--------
Groq, Cerebras et Mistral renvoient HTTP 429 (Too Many Requests) quand on
dépasse leur quota rpm. Sans retry, un rafale de news fait planter le pipeline
en cascade. Ce module fournit un décorateur `@with_backoff(...)` qui :

  1. Détecte les erreurs retryables (429, timeouts, connection errors,
     APIError transient)
  2. Attend un délai `base_delay × 2^attempt` avec jitter ±20 %
  3. Réessaie jusqu'à `max_retries` (défaut 5)
  4. Abandonne proprement sur les erreurs non retryables (auth, 4xx≠429,
     ValueError applicatives, etc.)

USAGE
-----
    from src.utils.rate_limiter import with_backoff

    @with_backoff(max_retries=5, base_delay=1.0, max_delay=30.0)
    def _call_llm(client, **kwargs):
        return client.chat.completions.create(**kwargs)

RÉFÉRENCES
----------
  Google SRE Book, ch.22 "Addressing Cascading Failures" (jitter nécessaire).
  AWS "Error retries and exponential backoff" (cap exponentiel).
  Kubernetes kube-apiserver : `client-go` workqueue (même forme).
"""

from __future__ import annotations

import logging
import random
import time
from functools import wraps
from typing import Callable, Iterable, TypeVar

logger = logging.getLogger("RateLimiter")

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Détecteurs d'erreurs retryables
# ---------------------------------------------------------------------------

_RATE_LIMIT_NAME_KEYWORDS = ("ratelimit", "rate_limit", "toomanyrequests")
_RATE_LIMIT_MSG_KEYWORDS = ("429", "rate limit", "too many requests", "quota")
_NETWORK_NAME_KEYWORDS = (
    "timeout",
    "connection",
    "apierror",
    "apiconnectionerror",
    "serviceunavailable",
    "serverdisconnected",
    "readtimeout",
    "writetimeout",
    "apistatus",  # openai.APIStatusError (5xx)
)


def _is_rate_limit_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    msg = str(exc).lower()
    if any(k in name for k in _RATE_LIMIT_NAME_KEYWORDS):
        return True
    if any(k in msg for k in _RATE_LIMIT_MSG_KEYWORDS):
        return True
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    return False


def _is_transient_network_error(exc: BaseException) -> bool:
    name = type(exc).__name__.lower()
    if any(k in name for k in _NETWORK_NAME_KEYWORDS):
        return True
    # Aussi : APIError avec code 5xx explicite
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 500 <= status < 600:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None:
        rs = getattr(resp, "status_code", None)
        if isinstance(rs, int) and 500 <= rs < 600:
            return True
    return False


# ---------------------------------------------------------------------------
# Decorateur
# ---------------------------------------------------------------------------


def with_backoff(
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    extra_retry_exceptions: Iterable[type] | None = None,
    retry_on_any_exception: bool = False,
):
    """
    Décorateur exponential-backoff pour appels réseau/LLM.

    Args:
        max_retries : tentatives max (en plus de la 1ère). Total = 1 + max_retries.
        base_delay  : délai initial en secondes (doublé à chaque échec).
        max_delay   : cap sur le délai (empêche l'attente d'exploser).
        jitter      : ajoute ±20 % de bruit uniforme (anti thundering-herd).
        extra_retry_exceptions : types d'exception additionnels à retryer.
        retry_on_any_exception : si True, retry sur toute exception (debug).

    Raises:
        Reraise l'exception d'origine après épuisement des tentatives.
    """
    extra_types = tuple(extra_retry_exceptions or ())

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:
                    is_retryable = (
                        retry_on_any_exception
                        or isinstance(exc, extra_types)
                        or _is_rate_limit_error(exc)
                        or _is_transient_network_error(exc)
                    )
                    if not is_retryable or attempt >= max_retries:
                        # Non retryable, ou budget épuisé : on remonte
                        if attempt >= max_retries and is_retryable:
                            logger.error(
                                "[RateLimiter] %s : abandon après %d retries — %s",
                                fn.__name__,
                                max_retries,
                                str(exc)[:120],
                            )
                        raise
                    delay = min(max_delay, base_delay * (2**attempt))
                    if jitter:
                        delay *= random.uniform(0.8, 1.2)
                    attempt += 1
                    logger.warning(
                        "[RateLimiter] %s (%s) — retry %d/%d dans %.1fs",
                        type(exc).__name__,
                        str(exc)[:80],
                        attempt,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Cas 1 : succès après 2 échecs 429
    counter = {"n": 0}

    class FakeRateLimit(Exception):
        def __init__(self):
            super().__init__("429 Too Many Requests")

    @with_backoff(max_retries=3, base_delay=0.05, max_delay=0.2, jitter=False)
    def unstable():
        counter["n"] += 1
        if counter["n"] < 3:
            raise FakeRateLimit()
        return "OK"

    print("Test 1 (3 tries needed):", unstable(), "attempts=", counter["n"])

    # Cas 2 : non retryable (ValueError) → remonte direct
    counter["n"] = 0

    @with_backoff(max_retries=3, base_delay=0.05)
    def fatal():
        counter["n"] += 1
        raise ValueError("programmer error")

    try:
        fatal()
    except ValueError as e:
        print("Test 2 (no retry on ValueError):", e, "attempts=", counter["n"])

    # Cas 3 : budget épuisé
    counter["n"] = 0

    @with_backoff(max_retries=2, base_delay=0.05, max_delay=0.1, jitter=False)
    def always_429():
        counter["n"] += 1
        raise FakeRateLimit()

    try:
        always_429()
    except FakeRateLimit:
        print("Test 3 (gave up after 2 retries): attempts=", counter["n"])
