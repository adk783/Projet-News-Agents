"""Client LLM unifie avec fallback en cascade entre fournisseurs.

Pourquoi ce module ?
--------------------
Le projet historique appelle directement les SDK Groq/Cerebras/Mistral via
`OpenAI(api_key=..., base_url=...)` repete dans 5+ fichiers (`agent_absa.py`,
`agent_memoire.py`, `agent_debat.py`, `context_compressor.py`,
`agent_filtrage_api.py`). 3 problemes :

1. **SPOF** : si Groq tombe, l'agent Baissier crash et stoppe tout le debat.
2. **Code duplique** : la logique "essaie Groq, sinon Mistral, sinon Cerebras"
   est copiee-collee dans chaque module avec de subtiles divergences.
3. **Pas de retry/backoff** : un 429 (rate limit) tue le pipeline.

Ce module fournit une **abstraction unique** :

>>> from src.utils.llm_client import LLMClient
>>> client = LLMClient.from_env()
>>> response = client.complete(
...     messages=[{"role": "user", "content": "Hello"}],
...     model_preference=["groq", "mistral", "cerebras"],
... )

Avec :
- Fallback automatique entre fournisseurs si l'un echoue (timeout, 5xx, 429).
- Retry exponentiel borne (max 3 tentatives, plafond 8s).
- Tracking des couts via `llm_cost_tracker` (deja en place).
- Logging structure de chaque appel (ProviderName, latence, tokens, errno).
- Mode test deterministe via `LLMClient(stub_response=...)` (pas d'appel reel).

Politique d'adoption
--------------------
- **Nouveau code** : utiliser `LLMClient` exclusivement.
- **Code historique** : migration progressive, chaque module au cas par cas
  (cf. ADR-003 dans ARCHITECTURE_DECISIONS.md). Les patterns hardcodes
  restent fonctionnels.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.utils.logger import get_logger

log = get_logger(__name__)


# =============================================================================
# Configuration des fournisseurs
# =============================================================================
@dataclass(frozen=True)
class ProviderConfig:
    """Configuration immutable d'un fournisseur LLM (endpoint OpenAI-compatible)."""

    name: str  # cle interne (groq / mistral / cerebras)
    env_key: str  # nom de la variable d'env contenant la cle API
    base_url: str  # endpoint OpenAI-compatible
    default_model: str  # modele par defaut si non specifie
    timeout_sec: float = 30.0  # timeout par requete


# Configuration centralisee : ajouter un nouveau fournisseur = 1 ligne ici.
# Ordre = priorite par defaut (mais surchargeable via `model_preference`).
PROVIDERS: dict[str, ProviderConfig] = {
    "groq": ProviderConfig(
        name="groq",
        env_key="GROQ_API_KEY",
        base_url="https://api.groq.com/openai/v1",
        default_model="meta-llama/llama-4-scout-17b-16e-instruct",
        timeout_sec=30.0,
    ),
    "mistral": ProviderConfig(
        name="mistral",
        env_key="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-small-latest",
        timeout_sec=30.0,
    ),
    "cerebras": ProviderConfig(
        name="cerebras",
        env_key="CEREBRAS_API_KEY",
        base_url="https://api.cerebras.ai/v1",
        default_model="llama-3.3-70b",
        timeout_sec=20.0,
    ),
    # NVIDIA NIM (build.nvidia.com) : OpenAI-compatible, free tier 40 RPM,
    # 100+ modeles accessibles (Llama 3.1 405B, DeepSeek-R1, Nemotron, etc.).
    # Place en queue de fallback : moins eprouve que Groq/Mistral/Cerebras
    # mais utile comme 4e jambe de resilience + acces a des modeles uniques.
    # Cle API gratuite via NVIDIA Developer Program (prefixe nvapi-).
    "nvidia_nim": ProviderConfig(
        name="nvidia_nim",
        env_key="NVIDIA_NIM_API_KEY",
        base_url="https://integrate.api.nvidia.com/v1",
        # Llama 3.1 70B Instruct : equilibre qualite/latence sur NIM.
        # Pour debat reasoning : surcharger via `model_preference` avec
        # par exemple "deepseek-ai/deepseek-r1" ou "nvidia/llama-3.3-nemotron-super-49b-v1".
        default_model="meta/llama-3.1-70b-instruct",
        timeout_sec=30.0,
    ),
}


# =============================================================================
# Resultat d'une completion
# =============================================================================
@dataclass
class LLMResponse:
    """Resultat d'un appel LLM, peu importe le fournisseur retenu."""

    content: str  # texte genere
    provider_used: str  # nom du fournisseur ayant repondu
    model_used: str  # nom du modele effectif
    latency_sec: float  # latence totale (incluant retries)
    n_attempts: int  # nombre total de tentatives
    fallback_chain: list[str] = field(default_factory=list)  # historique des tentatives


# =============================================================================
# Exceptions
# =============================================================================
class LLMError(Exception):
    """Erreur generique du client LLM."""


class AllProvidersFailedError(LLMError):
    """Tous les fournisseurs de la chaine de fallback ont echoue."""

    def __init__(self, attempts: list[tuple[str, str]]):
        self.attempts = attempts  # liste de (provider_name, error_message)
        chain = " -> ".join(f"{p}({e[:60]})" for p, e in attempts)
        super().__init__(f"All providers failed. Chain: {chain}")


# =============================================================================
# Client principal
# =============================================================================
class LLMClient:
    """Client LLM unifie avec fallback inter-fournisseur et retry exponentiel.

    Usage canonique
    ---------------
    >>> client = LLMClient.from_env()
    >>> resp = client.complete(
    ...     messages=[{"role": "user", "content": "Bonjour"}],
    ...     model_preference=["groq", "mistral"],
    ...     max_tokens=512,
    ...     temperature=0.7,
    ... )
    >>> print(resp.content, resp.provider_used)

    Mode test
    ---------
    >>> stub = LLMClient(stub_response="reponse fixe")
    >>> resp = stub.complete(messages=[...])
    >>> assert resp.content == "reponse fixe"
    """

    def __init__(
        self,
        api_keys: dict[str, str] | None = None,
        stub_response: str | None = None,
        max_retries: int = 3,
        backoff_base_sec: float = 0.5,
        backoff_cap_sec: float = 8.0,
    ):
        """Initialise le client.

        Parameters
        ----------
        api_keys
            Mapping {provider_name -> api_key}. Si None, lecture depuis env vars.
        stub_response
            Si fourni, retourne immediatement cette reponse SANS appeler de LLM.
            Utile pour les tests deterministes.
        max_retries
            Nombre max de tentatives par fournisseur (au-dela on bascule).
        backoff_base_sec, backoff_cap_sec
            Parametres du backoff exponentiel : `min(cap, base * 2^attempt)`.
        """
        # Subtilite : `if api_keys is None` (et pas `if not api_keys`) car un
        # dict vide {} est valide et signifie "pas de provider configure".
        # Avec `or`, {} aurait declenche le fallback env qui n'est pas voulu.
        self._api_keys = api_keys if api_keys is not None else self._load_keys_from_env()
        self._stub = stub_response
        self._max_retries = max_retries
        self._backoff_base = backoff_base_sec
        self._backoff_cap = backoff_cap_sec

    @classmethod
    def from_env(cls) -> LLMClient:
        """Construit un client en lisant les cles depuis les env vars."""
        return cls()

    @staticmethod
    def _load_keys_from_env() -> dict[str, str]:
        """Lit les cles API depuis l'environnement, ignore les absentes."""
        keys = {}
        for name, cfg in PROVIDERS.items():
            val = os.getenv(cfg.env_key, "").strip()
            if val:
                keys[name] = val
        return keys

    def available_providers(self) -> list[str]:
        """Retourne la liste des fournisseurs avec une cle API configuree."""
        return list(self._api_keys.keys())

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def complete(
        self,
        messages: Sequence[dict[str, str]],
        model_preference: Sequence[str] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model_override: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Genere une completion avec fallback automatique entre fournisseurs.

        Parameters
        ----------
        messages
            Format OpenAI : `[{"role": "user", "content": "..."}, ...]`
        model_preference
            Ordre des fournisseurs a essayer. Default = ["groq", "mistral",
            "cerebras", "nvidia_nim"]. Les fournisseurs sans cle API sont
            automatiquement skip.
        model_override
            Si fourni, override le `default_model` du provider retenu. Permet
            d'utiliser un modele specifique (ex: "deepseek-ai/deepseek-r1" pour
            NIM) sans modifier la config globale.
            **ATTENTION** : le model_override est passe a TOUS les providers de
            la chaine. Si vous voulez un modele specifique a NIM uniquement,
            utiliser `model_preference=["nvidia_nim"]` en complement.
        max_tokens, temperature, **kwargs
            Parametres OpenAI-compatibles passes au SDK.

        Returns
        -------
        LLMResponse
            Reponse + metadonnees de fallback.

        Raises
        ------
        AllProvidersFailedError
            Si tous les fournisseurs de la chaine ont echoue.
        """
        # Mode stub : retourne immediatement (tests).
        if self._stub is not None:
            return LLMResponse(
                content=self._stub,
                provider_used="stub",
                model_used="stub",
                latency_sec=0.0,
                n_attempts=1,
                fallback_chain=["stub"],
            )

        chain = list(model_preference) if model_preference else ["groq", "mistral", "cerebras", "nvidia_nim"]
        # Filtre les fournisseurs sans cle API (au lieu de les essayer en vain).
        chain = [p for p in chain if p in self._api_keys]
        if not chain:
            raise AllProvidersFailedError([("none", "aucun fournisseur configure")])

        attempts: list[tuple[str, str]] = []
        t0 = time.monotonic()
        n_total_attempts = 0

        for provider_name in chain:
            cfg = PROVIDERS[provider_name]
            # Si model_override fourni, on l'utilise ; sinon default du provider.
            effective_model = model_override or cfg.default_model
            try:
                resp_raw, n_tries = self._call_with_retry(
                    cfg=cfg,
                    api_key=self._api_keys[provider_name],
                    messages=list(messages),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model_override=model_override,
                    **kwargs,
                )
                content = (resp_raw.choices[0].message.content or "").strip()
                n_total_attempts += n_tries
                latency = time.monotonic() - t0
                log.info(
                    "llm_complete_ok",
                    extra={
                        "provider": provider_name,
                        "model": effective_model,
                        "latency_sec": round(latency, 3),
                        "n_attempts": n_total_attempts,
                        "fallback_chain": [p for p, _ in attempts] + [provider_name],
                    },
                )
                return LLMResponse(
                    content=content,
                    provider_used=provider_name,
                    model_used=effective_model,
                    latency_sec=latency,
                    n_attempts=n_total_attempts,
                    fallback_chain=[p for p, _ in attempts] + [provider_name],
                )
            except Exception as e:  # noqa: BLE001 - on capture large et on log
                n_total_attempts += self._max_retries
                err_msg = f"{type(e).__name__}: {e}"
                attempts.append((provider_name, err_msg))
                log.warning(
                    "llm_provider_failed",
                    extra={
                        "provider": provider_name,
                        "error": err_msg[:200],
                        "next_provider": chain[chain.index(provider_name) + 1]
                        if chain.index(provider_name) + 1 < len(chain)
                        else None,
                    },
                )
                continue

        # Tous les fournisseurs ont echoue.
        raise AllProvidersFailedError(attempts)

    def complete_raw(
        self,
        messages: Sequence[dict[str, str]],
        model_preference: Sequence[str] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        model_override: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, str, str]:
        """Variante de `complete()` retournant l'objet ChatCompletion brut.

        Necessaire pour les agents qui veulent garder l'acces a `response.usage`
        (cost tracking) ou `response_format`/`tool_calls` non exposes par
        `LLMResponse`.

        Returns
        -------
        (chat_completion, provider_used, model_used)
            chat_completion : objet `openai.types.chat.ChatCompletion` brut
            provider_used   : nom du fournisseur retenu (groq, mistral, cerebras)
            model_used      : nom du modele utilise

        Raises
        ------
        AllProvidersFailedError
            Si tous les fournisseurs de la chaine ont echoue.
        NotImplementedError
            Si appele en mode stub (le stub ne fournit pas d'objet brut).
        """
        if self._stub is not None:
            raise NotImplementedError("complete_raw() incompatible avec le mode stub. Utiliser complete().")

        chain = list(model_preference) if model_preference else ["groq", "mistral", "cerebras", "nvidia_nim"]
        chain = [p for p in chain if p in self._api_keys]
        if not chain:
            raise AllProvidersFailedError([("none", "aucun fournisseur configure")])

        attempts: list[tuple[str, str]] = []
        for provider_name in chain:
            cfg = PROVIDERS[provider_name]
            effective_model = model_override or cfg.default_model
            try:
                resp_raw, _ = self._call_with_retry(
                    cfg=cfg,
                    api_key=self._api_keys[provider_name],
                    messages=list(messages),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model_override=model_override,
                    **kwargs,
                )
                return resp_raw, provider_name, effective_model
            except Exception as e:  # noqa: BLE001
                attempts.append((provider_name, f"{type(e).__name__}: {e}"))
                continue

        raise AllProvidersFailedError(attempts)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _call_with_retry(
        self,
        cfg: ProviderConfig,
        api_key: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        model_override: str | None = None,
        **kwargs: Any,
    ) -> tuple[Any, int]:
        """Appelle un fournisseur avec retry exponentiel.

        Parameters
        ----------
        model_override
            Override le `cfg.default_model`. Permet de cibler un modele
            specifique (ex: "deepseek-ai/deepseek-r1") sans modifier
            la config globale du provider.

        Returns
        -------
        (chat_completion, n_attempts)
            chat_completion : objet brut `openai.types.chat.ChatCompletion`,
                              expose pour preserver `usage`, `tool_calls`, etc.

        Raises
        ------
        Exception
            La derniere exception rencontree apres `max_retries` tentatives.
        """
        # Import paresseux : le SDK openai n'est requis qu'a l'usage.
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout_sec,
        )

        # Determine le modele effectif (override > default).
        effective_model = model_override or cfg.default_model

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = client.chat.completions.create(
                    model=effective_model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs,
                )
                return resp, attempt + 1
            except Exception as e:  # noqa: BLE001
                last_exc = e
                if attempt < self._max_retries - 1:
                    delay = self._compute_retry_delay(e, attempt)
                    log.debug(
                        "llm_retry",
                        extra={
                            "provider": cfg.name,
                            "attempt": attempt + 1,
                            "delay_sec": delay,
                            "error": f"{type(e).__name__}: {str(e)[:100]}",
                        },
                    )
                    time.sleep(delay)

        # Ne devrait jamais arriver (last_exc set au moins 1 fois).
        assert last_exc is not None
        raise last_exc

    def _compute_retry_delay(self, exc: Exception, attempt: int) -> float:
        """Calcule le delai optimal avant retry selon le type d'erreur.

        Strategie differenciee (Grigorik 2012, Google Cloud Retry Design) :
        - **429 Rate Limit** : delai minimum plus long (10s) pour sortir de
          la fenetre de throttling, avec respect du header ``Retry-After``
          si fourni par le serveur. Doubler n'est pas suffisant ici.
        - **Erreur generique** (5xx, timeout, reseau) : backoff exponentiel
          classique ``min(cap, base * 2^attempt)`` depuis 0.5s.

        Le header ``Retry-After`` est prioritaire car le provider sait mieux
        que nous quand sa file sera disponible (Fielding & Reschke, RFC 7231).

        Parameters
        ----------
        exc     : Exception levee par le SDK openai.
        attempt : Numero de tentative (0-indexed), utilise pour le backoff.

        Returns
        -------
        float
            Nombre de secondes a attendre avant le prochain essai.
        """
        # Tente de recuperer le code HTTP depuis les exceptions openai
        # (RateLimitError, APIStatusError...).
        status_code: int | None = getattr(exc, "status_code", None)
        is_rate_limit = status_code == 429 or "rate limit" in str(exc).lower() or "429" in str(exc)

        if is_rate_limit:
            # Cherche le header Retry-After (secondes ou date HTTP).
            retry_after: float | None = None
            response = getattr(exc, "response", None)
            if response is not None:
                headers = getattr(response, "headers", {}) or {}
                ra_val = headers.get("retry-after") or headers.get("Retry-After")
                if ra_val is not None:
                    try:
                        retry_after = float(ra_val)
                    except (ValueError, TypeError):
                        pass  # Ignorer si c'est une date HTTP (rare sur NIM/Groq)

            if retry_after is not None:
                # On respecte ce que le serveur demande, avec un min de 1s.
                delay = max(1.0, retry_after)
                log.info(
                    "llm_rate_limit_retry_after",
                    extra={"retry_after_sec": delay, "attempt": attempt + 1},
                )
            else:
                # Pas de header : backoff 10s * 2^attempt (plus agressif que
                # le backoff generique, car on doit vraiment sortir du throttle).
                delay = min(self._backoff_cap * 5, 10.0 * (2**attempt))
                log.warning(
                    "llm_rate_limit_no_header",
                    extra={"delay_sec": delay, "attempt": attempt + 1},
                )
        else:
            # Erreur generique (reseau, 5xx, timeout) : backoff standard.
            delay = min(self._backoff_cap, self._backoff_base * (2**attempt))

        return delay


# =============================================================================
# Model Routing — registre des meilleurs modeles par tache
# =============================================================================
# Principe scientifique (cf. ADR-014) : a chaque tache, le modele est choisi
# selon des **benchmarks publics** (pas l'intuition). Chaque entree cite la
# reference qui justifie le choix.
#
# Format : task_name -> (provider, model_name)
#
# Pour utiliser :
#     from src.utils.llm_client import LLMClient, BEST_MODELS_BY_TASK
#     provider, model = BEST_MODELS_BY_TASK["reasoning_audit"]
#     resp = client.complete(messages=..., model_preference=[provider], model_override=model)
#
# Verification empirique faite sur la cle NIM live (2026-04-26) :
# tous les modeles cites sont disponibles via integrate.api.nvidia.com.
# =============================================================================
BEST_MODELS_BY_TASK: dict[str, tuple[str, str]] = {
    # Audit du raisonnement post-debat : besoin d'un modele "thinking"
    # qui produit explicitement la chain-of-thought (Wei et al. 2022).
    # Benchmark live (2026-04-26) : 3.3s, CoT visible, 80B params actifs.
    "reasoning_audit": ("nvidia_nim", "qwen/qwen3-next-80b-a3b-thinking"),
    # Summarization haute qualite avec long contexte (consolidation memoire,
    # transcripts d'earnings). Llama 3.1 405B domine sur LongBench / ScrollS
    # (Meta Llama 3.1 paper, juillet 2024).
    "long_summarization": ("nvidia_nim", "meta/llama-3.1-405b-instruct"),
    # Extraction structuree (ABSA, JSON taxonomique). Nemotron-Super 49B
    # (successeur de Nemotron-4 340B, deprecie sur NIM depuis Q1 2026)
    # est entraine pour le suivi d'instructions et la sortie structuree.
    "structured_extraction": ("nvidia_nim", "nvidia/llama-3.3-nemotron-super-49b-v1"),
    # Classification rapide YES/NO (filtrage de pertinence). Llama 3.1 8B
    # est suffisant : tache triviale, on optimise la latence + le cout.
    "cheap_classification": ("nvidia_nim", "meta/llama-3.1-8b-instruct"),
    # Code/JSON parsing (filings SEC structures, tableaux). Qwen2.5-Coder
    # 32B specialise pour le code (Qwen2.5-Coder Technical Report, sept 2024).
    "code_extraction": ("nvidia_nim", "qwen/qwen2.5-coder-32b-instruct"),
    # General-purpose haute qualite via NIM (fallback ou tests).
    "general_strong": ("nvidia_nim", "meta/llama-3.3-70b-instruct"),
}


def best_model_for_task(task: str) -> tuple[str, str]:
    """Retourne (provider, model) pour la tache donnee.

    Parameters
    ----------
    task : str
        Nom de la tache. Doit etre une cle de BEST_MODELS_BY_TASK.

    Returns
    -------
    (provider_name, model_name)
        Tuple a passer comme `model_preference=[provider]`, `model_override=model`.

    Raises
    ------
    KeyError
        Si la tache est inconnue. Le message liste les taches disponibles.
    """
    if task not in BEST_MODELS_BY_TASK:
        raise KeyError(f"Tache inconnue '{task}'. Disponibles : {sorted(BEST_MODELS_BY_TASK)}")
    return BEST_MODELS_BY_TASK[task]


__all__ = [
    "LLMClient",
    "LLMResponse",
    "LLMError",
    "AllProvidersFailedError",
    "ProviderConfig",
    "PROVIDERS",
    "BEST_MODELS_BY_TASK",
    "best_model_for_task",
]
