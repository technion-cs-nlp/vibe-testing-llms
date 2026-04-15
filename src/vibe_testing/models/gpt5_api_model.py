"""
Implementation for the new generation of API models like GPT-5.1.

This model primarily uses the `responses.create` endpoint and falls back to
`chat.completions.create` when necessary. It is configured via YAML in a way
that mirrors the GPT-OSS HuggingFace model, so higher-level pipeline code can
remain model-agnostic.
"""

import logging
import os
from typing import Any, Dict, Optional

from openai import OpenAI

from src.vibe_testing.env import load_project_dotenv

from .base import BaseModel
from .generation_failure_tracker import (
    record_generation_failure,
    record_generation_success,
)

logger = logging.getLogger(__name__)


class GPT5APIError(RuntimeError):
    """Raised when GPT5APIModel fails to generate a response."""


class GPT5APIModel(BaseModel):
    """
    A concrete implementation for next-generation language models like GPT-5.1
    that are accessed through the OpenAI API.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the API model client.

        Args:
            model_name (str): The name of the model (e.g., 'gpt-5.1-2025-11-13').
            config (Dict[str, Any]): Configuration dictionary containing API
                                     keys and model-specific parameters.
        """
        super().__init__(model_name, config)

        dotenv_path = load_project_dotenv(override=False)

        api_key_env_var = self.config.get("api_key_env_var", "OPENAI_API_KEY")
        api_key_env = os.getenv(api_key_env_var)
        if not api_key_env:
            env_hint = ""
            if dotenv_path is None:
                env_hint = (
                    " No .env file was found/loaded for this process. "
                    "On Slurm, .env is typically not sourced automatically; "
                    "either export the variable in the job environment or ensure "
                    "the working directory contains a readable .env file."
                )
            else:
                env_hint = (
                    f" A .env file was loaded from '{dotenv_path}', but it did not "
                    f"set '{api_key_env_var}'."
                )

            similar_vars = [
                name
                for name in ("OPENAI_API_KEY", "OPENAI_KEY", "OPEN_AI_KEY")
                if os.getenv(name)
            ]
            if similar_vars and api_key_env_var not in similar_vars:
                env_hint = f"{env_hint} Found other OpenAI-looking env vars set: {similar_vars}."

            raise ValueError(
                f"API key environment variable '{api_key_env_var}' is not set."
                f"{env_hint}"
            )
        self._client = OpenAI(api_key=api_key_env)
        logger.info(
            "Initialized GPT5APIModel for model '%s' using key env var '%s'.",
            self.model_name,
            api_key_env_var,
        )

    def _build_input_text(
        self, prompt: str, json_schema: Optional[Dict[str, Any]]
    ) -> str:
        """
        Build a single text input that mirrors the logical structure of GPT-OSS.
        """
        prompt_params = self.config.get("prompt_params", {})
        developer_message = prompt_params.get("developer_message", "").strip()
        if developer_message:
            prompt_prefix = f"{developer_message}\n\n"
        else:
            prompt_prefix = ""
        if json_schema:
            import json

            schema_str = json.dumps(json_schema, indent=2)
            prompt_prefix = f"{prompt_prefix}# Response Formats\n\n## {json_schema.get('title', 'schema')}\n\n{schema_str}<|end|>\n\n"
        return f"{prompt_prefix}{prompt}"

    def _build_generation_params(
        self, json_schema: Optional[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Construct generation parameters from config and runtime overrides.
        If json_schema is provided, structure the request to use
        the 'text.format' key as required by the Responses API for structured output.
        """
        gen_cfg = self.config.get("generation_config", {}).get("params", {}) or {}
        temperature = gen_cfg.get("temperature", 1.0)
        max_new_tokens = (
            gen_cfg.get("max_new_tokens")
            or gen_cfg.get("max_tokens")
            or gen_cfg.get("max_output_tokens")
            or 1024
        )

        params: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_new_tokens,
        }

        reasoning_cfg = self.config.get("reasoning", {})
        if isinstance(reasoning_cfg, dict) and reasoning_cfg.get("effort"):
            params["reasoning"] = {"effort": reasoning_cfg.get("effort")}

        text_cfg = self.config.get("text", {})
        if isinstance(text_cfg, dict) and text_cfg:
            params["text"] = dict(text_cfg)  # copy so we don’t mutate original

        if json_schema:
            # sanitize it first
            self._ensure_additional_properties_false(json_schema)
            # Build the structured-output schema under text.format
            params.setdefault("text", {})
            params["text"]["format"] = {
                "type": "json_schema",
                "name": "response_schema",
                "strict": True,
                "schema": json_schema,
            }

        # Runtime overrides take precedence
        params.update(kwargs)
        return params

    def _extract_text_from_responses(self, response: Any) -> str:
        """
        Extract textual content from a responses.create result.

        This is written defensively to accommodate minor SDK shape changes.
        """
        output = getattr(response, "output", None)
        if output is None:
            raise GPT5APIError(
                f"Responses API returned object without 'output' for model '{self.model_name}'."
            )

        # output is typically a list; find the first item with a 'content' attribute.
        segments = None
        if isinstance(output, (list, tuple)):
            for item in output:
                content = getattr(item, "content", None)
                if not content and isinstance(item, dict):
                    content = item.get("content")
                if content:
                    segments = content
                    break
        else:
            segments = getattr(output, "content", None)
            if not segments and isinstance(output, dict):
                segments = output.get("content")

        if segments is None:
            raise GPT5APIError(
                f"Responses output for model '{self.model_name}' has no 'content' segments."
            )

        texts = []
        if isinstance(segments, (list, tuple)):
            iterable = segments
        else:
            iterable = [segments]

        for seg in iterable:
            text_obj = getattr(seg, "text", None)
            if text_obj is None and isinstance(seg, dict):
                text_obj = seg.get("text") or seg.get("content")

            if text_obj is None:
                continue

            # Some SDK variants wrap the actual string in a `.value` attribute.
            if hasattr(text_obj, "value"):
                text_val = getattr(text_obj, "value")
            else:
                text_val = text_obj

            if text_val is not None:
                texts.append(str(text_val))

        if not texts:
            raise GPT5APIError(
                f"Could not extract text from responses output for model '{self.model_name}'."
            )
        return "\n".join(texts)

    def _ensure_additional_properties_false(self, schema: Dict[str, Any]) -> None:
        """
        Recursively traverse a JSON Schema dict and set additionalProperties: False
        for every object schema that doesn't already specify it.
        Modifies the dict in-place.
        """
        schema_type = schema.get("type")
        if schema_type == "object":
            # ensure additionalProperties is explicitly False
            if "additionalProperties" not in schema:
                schema["additionalProperties"] = False

            # recurse into properties
            props = schema.get("properties")
            if isinstance(props, dict):
                for key, subschema in props.items():
                    if isinstance(subschema, dict):
                        self._ensure_additional_properties_false(subschema)

            # also if schema defines nested schemas via other keywords (e.g. definitions, items, etc.)
            # but for now we handle the common properties case

        elif schema_type == "array":
            items = schema.get("items")
            if isinstance(items, dict):
                self._ensure_additional_properties_false(items)

        # If there are combinators (anyOf/oneOf/allOf) — you may want to handle them too:
        for comb in ("anyOf", "oneOf", "allOf", "not"):
            if comb in schema and isinstance(schema[comb], list):
                for subschema in schema[comb]:
                    if isinstance(subschema, dict):
                        self._ensure_additional_properties_false(subschema)

    def _build_chat_params(
        self, json_schema: Optional[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Construct parameters for the chat.completions.create fallback.
        """
        gen_cfg = self.config.get("generation_config", {}).get("params", {}) or {}
        temperature = gen_cfg.get("temperature", 1.0)
        max_new_tokens = (
            gen_cfg.get("max_new_tokens")
            or gen_cfg.get("max_tokens")
            or gen_cfg.get("max_output_tokens")
            or 1024
        )
        params: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_new_tokens,
        }
        if json_schema:
            params["response_format"] = {"type": "json_object"}
        params.update(kwargs)
        return params

    def generate(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> str:
        """
        Generate a response using the OpenAI GPT-5.1 API.

        The primary path uses `responses.create`; on failure, we fall back to
        `chat.completions.create`. Errors are logged and re-raised to avoid
        silent failures.
        """
        logger.debug(
            "Generating response for model '%s' (prompt prefix: '%s')",
            self.model_name,
            prompt[:80].replace("\n", " "),
        )

        input_text = self._build_input_text(prompt, json_schema)
        responses_params = self._build_generation_params(json_schema, **kwargs)

        last_error: Optional[Exception] = None

        # OpenAI API does not support seed parameter anymore
        responses_params.pop("seed", None)
        # responses_params.pop("response_format", None)

        # Primary: responses.create
        try:
            logger.debug(
                "Calling OpenAI.responses.create for model '%s' with params keys: %s",
                self.model_name,
                sorted(responses_params.keys()),
            )
            response = self._client.responses.create(
                model=self.model_name,
                input=input_text,
                **responses_params,
            )
            text = self._extract_text_from_responses(response).strip()
            record_generation_success(self.model_name)
            return text
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning(
                "Responses API call failed for model '%s': %s",
                self.model_name,
                exc,
            )

        # Fallback: chat.completions.create
        try:
            prompt_params = self.config.get("prompt_params", {})
            developer_message = prompt_params.get("developer_message", "")
            messages = []
            if developer_message:
                messages.append({"role": "system", "content": developer_message})
            messages.append({"role": "user", "content": prompt})

            chat_params = self._build_chat_params(json_schema, **kwargs)
            logger.debug(
                "Falling back to OpenAI.chat.completions.create for model '%s' "
                "with params keys: %s",
                self.model_name,
                sorted(chat_params.keys()),
            )
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                **chat_params,
            )
            text = response.choices[0].message.content.strip()
            record_generation_success(self.model_name)
            return text
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            logger.error(
                "Chat completions API call also failed for model '%s': %s",
                self.model_name,
                exc,
            )
            prefix = prompt[:200].replace("\n", " ")
            raise GPT5APIError(
                f"GPT5APIModel failed for model '{self.model_name}'. "
                f"Prompt prefix: '{prefix}'. "
                f"Responses error: {repr(last_error)}; chat error: {repr(exc)}"
            ) from exc
