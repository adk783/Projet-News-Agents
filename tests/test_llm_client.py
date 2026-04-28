"""Tests unitaires pour src.utils.llm_client.

Couvre :
- Mode stub (deterministe, pas d'appel reel).
- Detection des fournisseurs disponibles selon les env vars.
- Fallback en cascade : provider 1 echoue -> provider 2 -> succes.
- Retry exponentiel borne par max_retries.
- AllProvidersFailedError leve si chaine entierement KO.
- Backoff respecte (delais croissants entre tentatives).
- ProviderConfig immutable (frozen dataclass).
- Filtrage des providers sans cle API.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from src.utils.llm_client import (
    PROVIDERS,
    AllProvidersFailedError,
    LLMClient,
    LLMResponse,  # noqa: F401  (export check)
    ProviderConfig,
)

# L'import OpenAI est paresseux dans llm_client._call_with_retry, on patche
# donc directement la classe a sa source `openai.OpenAI` plutot que la
# reference (inexistante) dans le module llm_client.
PATCH_TARGET = "openai.OpenAI"


# =============================================================================
# Mode stub : pas d'appel reel
# =============================================================================
class TestStubMode:
    """Le stub_response permet des tests deterministes hors-ligne."""

    def test_stub_returns_fixed_response(self):
        """Un stub doit toujours renvoyer le meme contenu, sans I/O reseau."""
        client = LLMClient(stub_response="reponse fixe")
        resp = client.complete(messages=[{"role": "user", "content": "ignored"}])
        assert resp.content == "reponse fixe"
        assert resp.provider_used == "stub"
        assert resp.model_used == "stub"
        assert resp.n_attempts == 1
        assert resp.fallback_chain == ["stub"]

    def test_stub_response_zero_latency(self):
        """En mode stub, latence rapportee = 0.0 (pas d'I/O)."""
        client = LLMClient(stub_response="x")
        resp = client.complete(messages=[])
        assert resp.latency_sec == 0.0


# =============================================================================
# Detection des fournisseurs disponibles
# =============================================================================
class TestProviderDetection:
    """available_providers() doit refleter les env vars presentes."""

    def test_no_keys_returns_empty(self):
        """Sans cle API, aucun provider disponible."""
        with patch.dict(os.environ, {}, clear=True):
            client = LLMClient()
            assert client.available_providers() == []

    def test_only_groq_key_returns_groq(self):
        """Une seule cle -> un seul provider."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test"}, clear=True):
            client = LLMClient()
            assert client.available_providers() == ["groq"]

    def test_all_keys_returns_all_providers(self):
        """4 cles -> 4 providers (Groq, Mistral, Cerebras, NVIDIA NIM)."""
        env = {
            "GROQ_API_KEY": "gsk_x",
            "MISTRAL_API_KEY": "mst_x",
            "CEREBRAS_API_KEY": "csk_x",
            "NVIDIA_NIM_API_KEY": "nvapi-x",
        }
        with patch.dict(os.environ, env, clear=True):
            client = LLMClient()
            assert set(client.available_providers()) == {
                "groq",
                "mistral",
                "cerebras",
                "nvidia_nim",
            }

    def test_only_nvidia_nim_key_returns_nim(self):
        """Si seule la cle NIM est presente, le client doit la detecter."""
        with patch.dict(os.environ, {"NVIDIA_NIM_API_KEY": "nvapi-test"}, clear=True):
            client = LLMClient()
            assert client.available_providers() == ["nvidia_nim"]

    def test_empty_string_key_treated_as_absent(self):
        """Une cle vide ne doit PAS compter comme un provider disponible."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "  "}, clear=True):
            client = LLMClient()
            assert "groq" not in client.available_providers()


# =============================================================================
# Fallback en cascade
# =============================================================================
class TestFallbackChain:
    """Si un provider echoue, le client doit basculer vers le suivant."""

    def test_single_provider_succeeds_first_try(self):
        """Cas heureux : 1 provider, succes immediat."""
        client = LLMClient(api_keys={"groq": "gsk_x"})

        with patch(PATCH_TARGET) as mock_openai:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="ok groq"))]
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            resp = client.complete(messages=[{"role": "user", "content": "x"}])

            assert resp.content == "ok groq"
            assert resp.provider_used == "groq"
            assert resp.fallback_chain == ["groq"]
            # Un seul appel OpenAI (pas de retry)
            assert mock_openai.return_value.chat.completions.create.call_count == 1

    def test_fallback_when_first_provider_fails(self):
        """Provider 1 leve une exception sur toutes ses tentatives ->
        bascule vers provider 2 qui reussit."""
        client = LLMClient(
            api_keys={"groq": "gsk_x", "mistral": "mst_x"},
            max_retries=2,
            backoff_base_sec=0.0,  # pas d'attente en test
        )

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 2:
                # Les 2 premieres tentatives (Groq) echouent
                raise RuntimeError("groq is down")
            # 3eme tentative (Mistral) reussit
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="ok mistral"))]
            return mock_resp

        with patch(PATCH_TARGET) as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = side_effect

            resp = client.complete(
                messages=[{"role": "user", "content": "x"}],
                model_preference=["groq", "mistral"],
            )

            assert resp.content == "ok mistral"
            assert resp.provider_used == "mistral"
            assert resp.fallback_chain == ["groq", "mistral"]
            # 2 retries Groq + 1 succes Mistral = 3 appels OpenAI
            assert call_count["n"] == 3

    def test_all_providers_fail_raises(self):
        """Si TOUS les providers echouent, on leve AllProvidersFailedError
        avec l'historique complet."""
        client = LLMClient(
            api_keys={"groq": "gsk_x", "mistral": "mst_x"},
            max_retries=1,
            backoff_base_sec=0.0,
        )

        with patch(PATCH_TARGET) as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("network down")

            with pytest.raises(AllProvidersFailedError) as exc_info:
                client.complete(
                    messages=[{"role": "user", "content": "x"}],
                    model_preference=["groq", "mistral"],
                )

            err = exc_info.value
            assert len(err.attempts) == 2
            assert err.attempts[0][0] == "groq"
            assert err.attempts[1][0] == "mistral"
            assert "network down" in err.attempts[0][1]

    def test_no_configured_providers_raises_immediately(self):
        """Aucune cle API -> AllProvidersFailedError sans tenter d'appel."""
        client = LLMClient(api_keys={})

        with pytest.raises(AllProvidersFailedError) as exc_info:
            client.complete(messages=[{"role": "user", "content": "x"}])

        assert "aucun fournisseur configure" in exc_info.value.attempts[0][1]

    def test_default_chain_includes_nvidia_nim_in_last_position(self):
        """La chaine par defaut doit inclure NIM en queue de fallback."""
        client = LLMClient(
            api_keys={
                "groq": "gsk_x",
                "mistral": "mst_x",
                "cerebras": "csk_x",
                "nvidia_nim": "nvapi-x",
            },
            max_retries=1,
            backoff_base_sec=0.0,
        )

        with patch(PATCH_TARGET) as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("all down")

            with pytest.raises(AllProvidersFailedError) as exc_info:
                client.complete(messages=[{"role": "user", "content": "x"}])

            # 4 tentatives = 1 par provider, NIM en derniere position
            providers_tried = [p for p, _ in exc_info.value.attempts]
            assert providers_tried == ["groq", "mistral", "cerebras", "nvidia_nim"]

    def test_nim_used_when_others_fail(self):
        """Si Groq+Mistral+Cerebras tombent, NIM doit etre invoque et reussir."""
        client = LLMClient(
            api_keys={
                "groq": "gsk_x",
                "mistral": "mst_x",
                "cerebras": "csk_x",
                "nvidia_nim": "nvapi-x",
            },
            max_retries=1,
            backoff_base_sec=0.0,
        )

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] <= 3:  # Groq, Mistral, Cerebras tombent
                raise RuntimeError(f"provider {call_count['n']} down")
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="nim_ok"))]
            return mock_resp

        with patch(PATCH_TARGET) as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = side_effect

            resp = client.complete(messages=[{"role": "user", "content": "x"}])
            assert resp.content == "nim_ok"
            assert resp.provider_used == "nvidia_nim"
            assert resp.fallback_chain == ["groq", "mistral", "cerebras", "nvidia_nim"]

    def test_unconfigured_providers_filtered_from_chain(self):
        """Si on demande [groq, mistral] mais seul Groq est configure,
        Mistral doit etre filtre silencieusement."""
        client = LLMClient(api_keys={"groq": "gsk_x"})

        with patch(PATCH_TARGET) as mock_openai:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            resp = client.complete(
                messages=[],
                model_preference=["groq", "mistral", "cerebras"],
            )
            # Seul Groq est essaye, les 2 autres etaient absents
            assert resp.fallback_chain == ["groq"]


# =============================================================================
# Retry exponentiel
# =============================================================================
class TestRetryBackoff:
    """Le retry doit doubler le delai jusqu'a un cap."""

    def test_retry_succeeds_on_second_attempt(self):
        """Echec puis succes : 2 tentatives, retourne le contenu de la 2eme."""
        client = LLMClient(
            api_keys={"groq": "gsk_x"},
            max_retries=3,
            backoff_base_sec=0.0,
        )

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient")
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="recovered"))]
            return mock_resp

        with patch(PATCH_TARGET) as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = side_effect

            resp = client.complete(messages=[])
            assert resp.content == "recovered"
            assert call_count["n"] == 2

    def test_max_retries_respected(self):
        """Au-dela de max_retries, on bascule provider (ou on leve si seul)."""
        client = LLMClient(
            api_keys={"groq": "gsk_x"},
            max_retries=3,
            backoff_base_sec=0.0,
        )

        call_count = {"n": 0}

        def always_fail(*args, **kwargs):
            call_count["n"] += 1
            raise RuntimeError("permanent")

        with patch(PATCH_TARGET) as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = always_fail

            with pytest.raises(AllProvidersFailedError):
                client.complete(messages=[])

            # max_retries=3 tentatives sur l'unique provider
            assert call_count["n"] == 3

    def test_backoff_delays_growing(self):
        """Les delais entre retries doivent etre croissants (backoff exp)."""
        client = LLMClient(
            api_keys={"groq": "gsk_x"},
            max_retries=4,
            backoff_base_sec=0.1,
            backoff_cap_sec=10.0,
        )

        sleep_calls = []

        def fake_sleep(d):
            sleep_calls.append(d)

        with patch(PATCH_TARGET) as mock_openai, patch("time.sleep", side_effect=fake_sleep):
            mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("fail")

            with pytest.raises(AllProvidersFailedError):
                client.complete(messages=[])

            # 4 tentatives -> 3 sleeps entre tentatives
            # Delais : 0.1, 0.2, 0.4 (exponentiel base 0.1)
            assert len(sleep_calls) == 3
            assert sleep_calls[0] < sleep_calls[1] < sleep_calls[2]
            assert sleep_calls == [0.1, 0.2, 0.4]

    def test_backoff_capped(self):
        """Le delai ne doit pas depasser backoff_cap_sec."""
        client = LLMClient(
            api_keys={"groq": "gsk_x"},
            max_retries=10,
            backoff_base_sec=1.0,
            backoff_cap_sec=2.5,  # cap bas pour tester
        )

        sleep_calls = []

        with patch(PATCH_TARGET) as mock_openai, patch("time.sleep", side_effect=sleep_calls.append):
            mock_openai.return_value.chat.completions.create.side_effect = RuntimeError("fail")

            with pytest.raises(AllProvidersFailedError):
                client.complete(messages=[])

            # Tous les delais doivent etre <= cap (2.5s)
            assert all(d <= 2.5 for d in sleep_calls), f"got {sleep_calls}"


# =============================================================================
# Configuration des fournisseurs
# =============================================================================
class TestProviderConfig:
    """ProviderConfig est immutable et liste 3 fournisseurs canoniques."""

    def test_four_providers_registered(self):
        """Les 4 providers du projet doivent etre dans le registre global :
        Groq, Mistral, Cerebras + NVIDIA NIM (Phase 4 - 4e jambe fallback)."""
        assert set(PROVIDERS.keys()) == {"groq", "mistral", "cerebras", "nvidia_nim"}

    def test_nvidia_nim_provider_config(self):
        """NIM doit pointer vers integrate.api.nvidia.com avec env NVIDIA_NIM_API_KEY."""
        nim = PROVIDERS["nvidia_nim"]
        assert nim.env_key == "NVIDIA_NIM_API_KEY"
        assert nim.base_url == "https://integrate.api.nvidia.com/v1"
        assert "llama" in nim.default_model.lower()

    def test_provider_config_is_frozen(self):
        """ProviderConfig est un frozen dataclass : pas de mutation possible."""
        cfg = PROVIDERS["groq"]
        with pytest.raises((AttributeError, Exception)):  # FrozenInstanceError
            cfg.name = "something"  # type: ignore[misc]

    def test_each_provider_has_required_fields(self):
        """Chaque ProviderConfig doit avoir name, env_key, base_url, default_model."""
        for name, cfg in PROVIDERS.items():
            assert isinstance(cfg, ProviderConfig)
            assert cfg.name == name
            assert cfg.env_key
            assert cfg.base_url.startswith("https://")
            assert cfg.default_model
            assert cfg.timeout_sec > 0


# =============================================================================
# Tests model_override (selection per-call sans toucher la config globale)
# =============================================================================
class TestModelOverride:
    """Le model_override permet d'utiliser un modele specifique par appel."""

    def test_model_override_passed_to_sdk(self):
        """Si model_override fourni, c'est ce modele qui est passe a OpenAI."""
        client = LLMClient(api_keys={"nvidia_nim": "nvapi-x"})

        with patch(PATCH_TARGET) as mock_openai:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            client.complete(
                messages=[{"role": "user", "content": "x"}],
                model_preference=["nvidia_nim"],
                model_override="qwen/qwen3-next-80b-a3b-thinking",
            )

            # Verifier que le SDK a recu le model override, pas le default
            create_kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
            assert create_kwargs["model"] == "qwen/qwen3-next-80b-a3b-thinking"

    def test_model_override_reflected_in_response(self):
        """Le LLMResponse.model_used reflete le model_override."""
        client = LLMClient(api_keys={"nvidia_nim": "nvapi-x"})

        with patch(PATCH_TARGET) as mock_openai:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            resp = client.complete(
                messages=[{"role": "user", "content": "x"}],
                model_preference=["nvidia_nim"],
                model_override="meta/llama-3.1-405b-instruct",
            )

            assert resp.model_used == "meta/llama-3.1-405b-instruct"

    def test_no_override_uses_default_model(self):
        """Sans model_override, le default_model du provider est utilise."""
        client = LLMClient(api_keys={"groq": "gsk_x"})

        with patch(PATCH_TARGET) as mock_openai:
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            client.complete(
                messages=[{"role": "user", "content": "x"}],
                model_preference=["groq"],
            )

            create_kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
            assert create_kwargs["model"] == PROVIDERS["groq"].default_model

    def test_complete_raw_supports_model_override(self):
        """complete_raw() expose aussi le model_override."""
        client = LLMClient(api_keys={"nvidia_nim": "nvapi-x"})

        with patch(PATCH_TARGET) as mock_openai:
            mock_resp = MagicMock()
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            _, provider, model = client.complete_raw(
                messages=[{"role": "user", "content": "x"}],
                model_preference=["nvidia_nim"],
                model_override="nvidia/nemotron-4-340b-instruct",
            )
            assert provider == "nvidia_nim"
            assert model == "nvidia/nemotron-4-340b-instruct"


# =============================================================================
# Tests BEST_MODELS_BY_TASK (model routing)
# =============================================================================
class TestBestModelsByTask:
    """Le registre BEST_MODELS_BY_TASK doit etre coherent et utilisable."""

    def test_registry_has_all_canonical_tasks(self):
        """Toutes les taches utilisees dans le code doivent etre presentes."""
        from src.utils.llm_client import BEST_MODELS_BY_TASK

        expected = {
            "reasoning_audit",
            "long_summarization",
            "structured_extraction",
            "cheap_classification",
            "code_extraction",
            "general_strong",
        }
        assert set(BEST_MODELS_BY_TASK.keys()) >= expected

    def test_each_entry_is_provider_model_tuple(self):
        """Chaque entree doit etre un tuple (provider, model) valide."""
        from src.utils.llm_client import BEST_MODELS_BY_TASK

        for task, (provider, model) in BEST_MODELS_BY_TASK.items():
            assert provider in PROVIDERS, f"{task}: provider '{provider}' not in PROVIDERS"
            assert model and isinstance(model, str), f"{task}: model invalide"

    def test_best_model_for_task_returns_tuple(self):
        from src.utils.llm_client import best_model_for_task

        provider, model = best_model_for_task("reasoning_audit")
        assert provider == "nvidia_nim"
        assert "thinking" in model.lower()

    def test_best_model_for_unknown_task_raises(self):
        from src.utils.llm_client import best_model_for_task

        with pytest.raises(KeyError) as exc_info:
            best_model_for_task("does_not_exist")
        # Le message doit lister les taches disponibles
        assert "reasoning_audit" in str(exc_info.value)

    def test_reasoning_audit_uses_thinking_capable_model(self):
        """La tache reasoning_audit doit etre routee vers un modele 'thinking'."""
        from src.utils.llm_client import BEST_MODELS_BY_TASK

        _, model = BEST_MODELS_BY_TASK["reasoning_audit"]
        # Verifier que c'est un modele avec capacites reasoning explicite
        assert any(kw in model.lower() for kw in ["thinking", "r1", "reasoning", "magistral"])

    def test_long_summarization_uses_large_model(self):
        """Long summarization doit utiliser un gros modele (>= 70B params)."""
        from src.utils.llm_client import BEST_MODELS_BY_TASK

        _, model = BEST_MODELS_BY_TASK["long_summarization"]
        # On verifie que c'est un Llama 3.1+ (les benchmarks justifient).
        assert "llama" in model.lower() or "405b" in model.lower()


# =============================================================================
# complete_raw : objet ChatCompletion brut (preserve usage, tool_calls, etc.)
# =============================================================================
class TestCompleteRaw:
    """complete_raw() retourne le ChatCompletion brut + nom du provider."""

    def test_complete_raw_returns_object_provider_model(self):
        """Heureux : retourne (chat_completion, provider, model)."""
        client = LLMClient(api_keys={"groq": "gsk_x"})

        with patch(PATCH_TARGET) as mock_openai:
            mock_resp = MagicMock()
            mock_resp.usage.total_tokens = 42
            mock_openai.return_value.chat.completions.create.return_value = mock_resp

            raw, provider, model = client.complete_raw(
                messages=[{"role": "user", "content": "x"}],
                model_preference=["groq"],
            )

            assert raw is mock_resp
            assert raw.usage.total_tokens == 42  # acces preserve a usage
            assert provider == "groq"
            assert model == PROVIDERS["groq"].default_model

    def test_complete_raw_falls_back_between_providers(self):
        """complete_raw() doit aussi beneficier du fallback inter-provider."""
        client = LLMClient(
            api_keys={"groq": "gsk_x", "mistral": "mst_x"},
            max_retries=1,
            backoff_base_sec=0.0,
        )

        call_count = {"n": 0}

        def side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("groq down")
            return MagicMock(usage=MagicMock(total_tokens=10))

        with patch(PATCH_TARGET) as mock_openai:
            mock_openai.return_value.chat.completions.create.side_effect = side_effect

            raw, provider, model = client.complete_raw(messages=[])
            assert provider == "mistral"
            assert raw.usage.total_tokens == 10

    def test_complete_raw_in_stub_mode_raises(self):
        """En mode stub, complete_raw() doit lever NotImplementedError
        (le stub ne peut pas fournir un ChatCompletion realiste)."""
        client = LLMClient(stub_response="x")
        with pytest.raises(NotImplementedError):
            client.complete_raw(messages=[])

    def test_complete_raw_no_providers_raises_immediately(self):
        """complete_raw() avec api_keys={} doit lever AllProvidersFailedError."""
        client = LLMClient(api_keys={})
        with pytest.raises(AllProvidersFailedError):
            client.complete_raw(messages=[])

    def test_complete_raw_passes_kwargs_through(self):
        """`response_format`, `tool_choice` etc. doivent etre transmis au SDK."""
        client = LLMClient(api_keys={"groq": "gsk_x"})

        with patch(PATCH_TARGET) as mock_openai:
            mock_openai.return_value.chat.completions.create.return_value = MagicMock()

            client.complete_raw(
                messages=[{"role": "user", "content": "x"}],
                response_format={"type": "json_object"},
            )

            create_kwargs = mock_openai.return_value.chat.completions.create.call_args.kwargs
            assert create_kwargs["response_format"] == {"type": "json_object"}
