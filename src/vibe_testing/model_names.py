"""Shared model-name canonicalization and display helpers."""

from __future__ import annotations

from typing import Dict

# Explicit alias -> canonical mapping used across Stage 5/6 analysis.
# Keep this dictionary reviewable and update it whenever a new provider/repo
# alias is introduced for an existing underlying model.
MODEL_NAME_CANONICAL_MAP: Dict[str, str] = {
    "gpt5": "gpt-5.1",
    "gpt5.1": "gpt-5.1",
    "gpt-5.1": "gpt-5.1",
    "gpt-5.1-2025-11-13": "gpt-5.1",
    "gpt4": "gpt-4o",
    "gpt4-o": "gpt-4o",
    "gpt-4o": "gpt-4o",
    "gpt-oss": "gpt-oss-20b",
    "gpt-oss-20b": "gpt-oss-20b",
    "gpt-oss-low-effort": "gpt-oss-20b",
    "gpt-oss-low-effort-4bit": "gpt-oss-20b",
    "gpt-oss-medium-effort": "gpt-oss-20b",
    "gpt-oss-high-effort": "gpt-oss-20b",
    "openai/gpt-oss-20b": "gpt-oss-20b",
    "unsloth/gpt-oss-20b": "gpt-oss-20b",
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit": "gpt-oss-20b",
    "gpt-oss-120b": "gpt-oss-120b",
    "gpt-oss-120b-low-effort": "gpt-oss-120b",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "qwen3-4b": "qwen3-4b",
    "qwen/qwen3-4b": "qwen3-4b",
    "unsloth/qwen3-4b": "qwen3-4b",
    "unsloth/qwen3-4b-unsloth-bnb-4bit": "qwen3-4b",
    "qwen3-8b": "qwen3-8b",
    "qwen/qwen3-8b": "qwen3-8b",
    "unsloth/qwen3-8b": "qwen3-8b",
    "unsloth/qwen3-8b-unsloth-bnb-4bit": "qwen3-8b",
    "qwen3-14b": "qwen3-14b",
    "qwen3_14b": "qwen3-14b",
    "qwen3-14b-low-max-tokens": "qwen3-14b",
    "qwen/qwen3-14b": "qwen3-14b",
    "unsloth/qwen3-14b": "qwen3-14b",
    "unsloth/qwen3-14b-unsloth-bnb-4bit": "qwen3-14b",
    "qwen3-32b": "qwen3-32b",
    "qwen3_32b": "qwen3-32b",
    "qwen3-32b-low-max-tokens": "qwen3-32b",
    "qwen/qwen3-32b": "qwen3-32b",
    "unsloth/qwen3-32b": "qwen3-32b",
    "unsloth/qwen3-32b-unsloth-bnb-4bit": "qwen3-32b",
    "gemma3-4b": "gemma-3-4b-it",
    "gemma-3-4b-it": "gemma-3-4b-it",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
    "gemini3-pro": "gemini-3-pro-preview",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
    "gemini3-flash": "gemini-3-flash-preview",
    "gemini-3-flash-preview": "gemini-3-flash-preview",
    "llama3-70b": "llama-3.3-70b-instruct",
    "llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "meta-llama/llama-3.3-70b-instruct": "llama-3.3-70b-instruct",
    "meta-llama/llama-3.1-70b-instruct": "llama-3.1-70b-instruct",
    "meta-llama/llama-3.1-8b-instruct": "llama-3.1-8b-instruct",
    "meta-llama/llama-3-70b-instruct": "llama-3-70b-instruct",
    "claude-3-5-sonnet": "claude-3.5-sonnet",
    "claude-3-5-sonnet-20241022": "claude-3.5-sonnet",
    "claude-3-opus": "claude-3-opus",
    "claude-3-opus-20240229": "claude-3-opus",
    "gpt-4": "gpt-4",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-4-turbo-preview": "gpt-4-turbo",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}

MODEL_NAME_DISPLAY_MAP: Dict[str, str] = {
    "gpt-5.1": "GPT-5.1",
    "gpt-4o": "GPT-4o",
    "gpt-oss-20b": "GPT-OSS-20B",
    "gpt-oss-120b": "GPT-OSS-120B",
    "qwen3-4b": "Qwen3-4B",
    "qwen3-8b": "Qwen3-8B",
    "qwen3-14b": "Qwen3-14B",
    "qwen3-32b": "Qwen3-32B",
    "gemma-3-4b-it": "Gemma-3-4B",
    "gemini-3-pro-preview": "Gemini-3-Pro",
    "gemini-3-flash-preview": "Gemini-3-Flash",
    "llama-3.3-70b-instruct": "Llama-3.3-70B",
    "llama-3.1-70b-instruct": "Llama-3.1-70B",
    "llama-3.1-8b-instruct": "Llama-3.1-8B",
    "llama-3-70b-instruct": "Llama-3-70B",
    "claude-3.5-sonnet": "Claude-3.5-Sonnet",
    "claude-3-opus": "Claude-3-Opus",
    "gpt-4": "GPT-4",
    "gpt-4-turbo": "GPT-4-Turbo",
    "gpt-3.5-turbo": "GPT-3.5-Turbo",
}


def strip_old_prefix(model_name: object) -> str:
    """
    Remove one or more legacy ``OLD_`` prefixes from a model identifier.

    Args:
        model_name (object): Raw model identifier.

    Returns:
        str: Cleaned model identifier with any leading ``OLD_`` prefixes removed.
    """
    cleaned = str(model_name or "").strip()
    while cleaned.lower().startswith("old_"):
        cleaned = cleaned[4:]
    return cleaned


def canonicalize_model_name(model_name: object) -> str:
    """
    Map a raw model identifier to its canonical shared name.

    Args:
        model_name (object): Raw model identifier from configs, artifacts, or CLI input.

    Returns:
        str: Canonical model identifier. Unknown names are returned unchanged except
            for ``OLD_`` prefix stripping.
    """
    raw = strip_old_prefix(model_name)
    if not raw:
        return ""
    lowered = raw.lower()
    canonical = MODEL_NAME_CANONICAL_MAP.get(lowered)
    if canonical is not None:
        return canonical
    if "/" in lowered:
        tail = lowered.split("/")[-1]
        canonical = MODEL_NAME_CANONICAL_MAP.get(tail)
        if canonical is not None:
            return canonical
    return raw


def display_model_name(model_name: object) -> str:
    """
    Translate a raw or canonical model identifier into a display-friendly label.

    Args:
        model_name (object): Raw or canonical model identifier.

    Returns:
        str: Human-readable display name.
    """
    canonical = canonicalize_model_name(model_name)
    if not canonical:
        return ""
    return MODEL_NAME_DISPLAY_MAP.get(canonical, canonical)
