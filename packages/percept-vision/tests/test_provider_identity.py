"""Card RB-PROVIDER — provider identity is the NAME; the model id is configuration.

The property under test is architectural, not behavioral: a volatile model id must never become the
thing that selects an adapter, and an explicitly requested model must never be silently swapped.
"""

from __future__ import annotations

import pytest

import percept
from percept import provider as prov
from percept.config import Settings
from percept.provider import (
    ProviderError,
    ProviderSpec,
    looks_like_model_id,
    resolve,
    resolve_vision,
    settings_with_model,
)


# ---- 1. name -> factory; the model rides along as configuration ----------------------------------


def test_provider_name_resolves_to_adapter_and_model_is_carried_as_settings():
    adapter = resolve_vision("openrouter", model="qwen/qwen3-vl-32b-instruct")
    assert type(adapter).__name__ == "OpenRouterVision"
    assert adapter._model == "qwen/qwen3-vl-32b-instruct"


def test_same_provider_different_models_is_one_adapter_identity():
    """Two model ids, ONE adapter identity — the card in one assertion."""
    a = resolve_vision("openrouter", model="qwen/qwen3-vl-32b-instruct")
    b = resolve_vision("openrouter", model="opengvlab/internvl3-78b")
    assert type(a) is type(b)
    assert a._model != b._model
    assert ProviderSpec("openrouter", a._model).identity == ProviderSpec("openrouter", b._model).identity


def test_a_cross_vendor_model_id_does_not_change_the_adapter():
    """An anthropic model id routed via OpenRouter is still the openrouter adapter, not the claude one."""
    adapter = resolve_vision("openrouter", model="anthropic/claude-3-opus")
    assert type(adapter).__name__ == "OpenRouterVision"


def test_model_is_optional_and_settings_are_untouched_without_it():
    base = Settings()
    assert settings_with_model(base, "vision", None) is base
    adapter = resolve_vision("scripted", settings=base)
    assert type(adapter).__module__ == "percept.backends.fake"


def test_settings_with_model_copies_rather_than_mutates():
    base = Settings()
    out = settings_with_model(base, "vision", "qwen/qwen3-vl-32b-instruct")
    assert out is not base
    assert base.models.vision_model == Settings().models.vision_model
    assert out.models.vision_model == "qwen/qwen3-vl-32b-instruct"
    # every other seam's model is carried over untouched
    assert out.models.conductor_model == base.models.conductor_model


def test_specialist_and_conductor_seams_carry_their_own_model_field():
    base = Settings()
    assert settings_with_model(base, "specialist", "m-1").models.specialist_model == "m-1"
    assert settings_with_model(base, "conductor", "m-2").models.conductor_model == "m-2"


def test_seam_without_a_model_id_refuses_rather_than_ignoring_it():
    with pytest.raises(ProviderError, match="carries no model id"):
        resolve("stt", "scripted", model="nova-3")


# ---- 2. a model id is never an adapter name ------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "qwen/qwen3-vl-32b-instruct",
        "anthropic/claude-3-opus",
        "claude-opus-4-8",
        "gemini-2.5-flash",
        "gpt-5-mini",
    ],
)
def test_model_id_in_the_provider_slot_raises_an_actionable_error(name):
    with pytest.raises(ProviderError) as exc:
        resolve_vision(name)
    msg = str(exc.value)
    assert "MODEL id" in msg
    assert f"model={name!r}" in msg  # names the two-argument fix


@pytest.mark.parametrize("name", ["openrouter", "claude", "gemini", "gemini-live", "scripted", "fake"])
def test_registered_provider_names_are_never_treated_as_model_ids(name):
    """A registered name wins the lookup unconditionally — the heuristic can't shadow a real provider."""
    assert name in percept.registry.available("vision")
    resolve_vision(name)  # must not raise ProviderError


def test_looks_like_model_id_does_not_fire_on_plain_provider_names():
    for name in ("openrouter", "claude", "gemini", "gemini-live", "scripted", "fake", "deepgram"):
        assert not looks_like_model_id(name)
    for name in ("qwen/qwen3-vl", "claude-opus-4-8", "gemini-2.5-flash"):
        assert looks_like_model_id(name)
    assert not looks_like_model_id("")


def test_unknown_non_model_name_still_gets_the_registry_error():
    with pytest.raises(ValueError, match="unknown vision backend"):
        resolve_vision("nosuchprovider")


def test_missing_provider_is_refused():
    with pytest.raises(ProviderError, match="no vision provider given"):
        resolve_vision(None)


# ---- 3. an explicit model is honored or refused, never silently swapped --------------------------


def test_a_model_the_provider_cannot_route_is_refused_not_substituted():
    """OpenRouter coerces a non-namespaced id to its own default; asking for one must FAIL loudly."""
    with pytest.raises(ProviderError) as exc:
        resolve_vision("openrouter", model="claude-opus-4-8")
    assert "did not accept" in str(exc.value)


def test_adapter_without_a_reported_model_is_not_second_guessed():
    """No capability manifest exists (deferred) — an adapter that reports no model is simply not checked."""
    scripted = resolve_vision("scripted", model="qwen/qwen3-vl-32b-instruct")
    assert prov._adapter_model(scripted) is None


# ---- manual adapter construction -----------------------------------------------------------------


class _HandBuilt:
    def __init__(self, model: str) -> None:
        self._model = model


def test_a_constructed_adapter_is_used_verbatim():
    obj = _HandBuilt("whatever/i-want")
    assert resolve_vision(obj) is obj


def test_model_alongside_a_constructed_adapter_is_refused_not_ignored():
    obj = _HandBuilt("whatever/i-want")
    with pytest.raises(ProviderError, match="silently ignored"):
        resolve_vision(obj, model="qwen/qwen3-vl-32b-instruct")


# ---- spec + surface ------------------------------------------------------------------------------


def test_provider_spec_requires_a_real_name():
    for bad in ("", "   "):
        with pytest.raises(ProviderError):
            ProviderSpec(bad)


def test_provider_seam_is_on_the_public_surface():
    assert percept.resolve_vision is resolve_vision
    for name in ("resolve_vision", "ProviderSpec", "ProviderError"):
        assert name in percept.__all__


def test_provider_module_imports_no_vendor_sdk():
    """Invariant 7: the resolution seam itself pulls in no vendor client."""
    import inspect

    src = inspect.getsource(prov)
    for vendor in ("import openai", "import anthropic", "import google", "from openai", "from anthropic"):
        assert vendor not in src
