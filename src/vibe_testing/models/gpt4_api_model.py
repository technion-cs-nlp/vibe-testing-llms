"""
Implementation for GPT-4 class API models like gpt-4o.

Unlike GPT-5-style models in this repository, GPT-4o is accessed via the
`chat.completions.create` endpoint and does not use Harmony function calling.
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


class GPT4APIError(RuntimeError):
    """Raised when GPT4APIModel fails to generate a response."""


class GPT4APIModel(BaseModel):
    """
    A concrete implementation for GPT-4 class models accessed through the OpenAI API.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the API model client.

        Args:
            model_name (str): The name of the model (e.g., 'gpt-4o').
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
            raise ValueError(
                f"API key environment variable '{api_key_env_var}' is not set."
                f"{env_hint}"
            )
        self._client = OpenAI(api_key=api_key_env)
        logger.info(
            "Initialized GPT4APIModel for model '%s' using key env var '%s'.",
            self.model_name,
            api_key_env_var,
        )

    def _build_user_content(
        self, prompt: str, json_schema: Optional[Dict[str, Any]]
    ) -> str:
        """
        Builds the user message content, optionally prefixing a schema block.

        Args:
            prompt (str): The user prompt.
            json_schema (Optional[Dict[str, Any]]): Optional JSON schema for structured output.

        Returns:
            str: The content passed as the user message.
        """
        if not json_schema:
            return prompt

        import json

        schema_str = json.dumps(json_schema, indent=2)
        # Note: GPT-4 chat completions does not support strict JSON Schema enforcement
        # in this codebase. We include the schema as context and also enable JSON mode.
        schema_prefix = (
            "# Response Formats\n\n"
            f"## {json_schema.get('title', 'schema')}\n\n"
            f"{schema_str}\n\n"
        )
        return f"{schema_prefix}{prompt}"

    def _build_generation_params(
        self, json_schema: Optional[Dict[str, Any]], **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Construct generation parameters from config and runtime overrides.

        Args:
            json_schema (Optional[Dict[str, Any]]): Optional JSON schema request.
            **kwargs: Per-call overrides.

        Returns:
            Dict[str, Any]: Parameters suitable for chat.completions.create.

        Raises:
            ValueError: If unsupported OpenAI parameters are provided.
        """
        # Guardrails: GPT-4o path does not support Harmony/function calling in this repo.
        unsupported_keys = {"tools", "tool_choice", "functions", "function_call"}
        present_unsupported = sorted(k for k in kwargs.keys() if k in unsupported_keys)
        if present_unsupported:
            raise ValueError(
                "GPT4APIModel does not support Harmony/function calling parameters. "
                f"Unsupported keys: {present_unsupported}. TODO: implement tools support if needed."
            )

        gen_cfg = self.config.get("generation_config", {}).get("params", {}) or {}
        temperature = gen_cfg.get("temperature", 1.0)
        top_p = gen_cfg.get("top_p", 1.0)
        max_tokens = (
            gen_cfg.get("max_tokens")
            or gen_cfg.get("max_output_tokens")
            or gen_cfg.get("max_new_tokens")
            or 1024
        )

        params: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }

        if json_schema:
            # Best-effort JSON mode; schema is provided in the prompt prefix.
            params["response_format"] = {"type": "json_object"}

        # OpenAI chat.completions does not support seed parameter; drop it.
        kwargs.pop("seed", None)

        # Runtime overrides take precedence
        params.update(kwargs)
        return params

    def generate(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> str:
        """
        Generate a response using the OpenAI Chat Completions API.

        Args:
            prompt (str): The input prompt for the model.
            json_schema (Optional[Dict[str, Any]]): Optional schema hint; enables JSON mode.
            **kwargs: Additional generation parameters.

        Returns:
            str: The model's generated output.

        Raises:
            GPT4APIError: If the API call fails or the response is malformed.
        """
        try:
            logger.debug(
                "Generating response for model '%s' (prompt prefix: '%s')",
                self.model_name,
                prompt[:80].replace("\n", " "),
            )

            prompt_params = self.config.get("prompt_params", {}) or {}
            system_message = (
                prompt_params.get("system_message")
                or prompt_params.get("developer_message")
                or ""
            )

            user_content = self._build_user_content(prompt, json_schema)
            messages = []
            if system_message:
                messages.append({"role": "system", "content": system_message})
            messages.append({"role": "user", "content": user_content})

            chat_params = self._build_generation_params(json_schema, **kwargs)

            try:
                logger.debug(
                    "Calling OpenAI.chat.completions.create for model '%s' with params keys: %s",
                    self.model_name,
                    sorted(chat_params.keys()),
                )
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    **chat_params,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Chat completions API call failed for model '%s': %s",
                    self.model_name,
                    exc,
                )
                prefix = prompt[:200].replace("\n", " ")
                raise GPT4APIError(
                    f"GPT4APIModel failed for model '{self.model_name}'. Prompt prefix: '{prefix}'. "
                    f"Error: {repr(exc)}"
                ) from exc

            try:
                content = response.choices[0].message.content
            except Exception as exc:  # noqa: BLE001
                raise GPT4APIError(
                    f"Malformed OpenAI response for model '{self.model_name}': {repr(exc)}"
                ) from exc

            if content is None:
                raise GPT4APIError(
                    f"OpenAI returned empty content for model '{self.model_name}'."
                )

            text = str(content).strip()
            record_generation_success(self.model_name)
            return text
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise
