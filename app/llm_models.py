"""
Model catalogue and fallback-chain construction.

Providers are tried in order, and each provider owns an ordered list of
models. A model is abandoned when it fails with a transient error (rate
limit, overload, timeout); the chain then moves to the next model and,
once every model of a provider is exhausted, to the next provider.

All OpenRouter entries default to ``:free`` variants so the agent stays
usable without a paid credit balance.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ordered by preference. The first reachable model wins.
OPENROUTER_MODELS: tuple[str, ...] = (
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "cohere/north-mini-code:free",
    "nvidia/nemotron-3.5-lightning:free",
    "poolside/laguna-s-2.1:free",
    "liquid/lfm-2.5-2.6b:free",
)

GROQ_MODELS: tuple[str, ...] = (
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
)

XAI_MODELS: tuple[str, ...] = (
    "grok-4-fast-non-reasoning",
    "grok-3-mini",
)

# Provider order for the whole chain.
PROVIDER_ORDER: tuple[str, ...] = ("openrouter", "groq", "xai")


@dataclass(frozen=True)
class ModelSpec:
    """A concrete model endpoint plus the provider that serves it."""

    provider: str
    model: str

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def _models_for(provider: str) -> tuple[str, ...]:
    return {
        "openrouter": OPENROUTER_MODELS,
        "groq": GROQ_MODELS,
        "xai": XAI_MODELS,
    }.get(provider, ())


def build_model_chain(
    *,
    openrouter_api_key: str = "",
    groq_api_key: str = "",
    xai_api_key: str = "",
    preferred_model: str = "",
    preferred_provider: str = "",
) -> list[ModelSpec]:
    """
    Build the ordered list of endpoints to try.

    A provider is only included when its API key is present. When a
    preferred model is configured it is promoted to the front of its
    provider's list.
    """
    keys = {
        "openrouter": openrouter_api_key,
        "groq": groq_api_key,
        "xai": xai_api_key,
    }
    chain: list[ModelSpec] = []
    for provider in PROVIDER_ORDER:
        if not keys.get(provider):
            continue
        models = list(_models_for(provider))
        if preferred_provider == provider and preferred_model:
            models = [preferred_model] + [m for m in models if m != preferred_model]
        chain.extend(ModelSpec(provider=provider, model=m) for m in models)
    return chain
