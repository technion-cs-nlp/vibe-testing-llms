import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from .base import BaseModel
from .generation_failure_tracker import (
    record_generation_failure,
    record_generation_success,
)

logger = logging.getLogger(__name__)


class Gemini3Model(BaseModel):
    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set.")

        # Initialize the Client.
        self._client = genai.Client(
            api_key=api_key  # , http_options=types.HttpOptions(api_version="v1alpha")
        )

        self._sleep_seconds = config.get("sleep_seconds", 60)

    def _prepare_config(
        self, json_schema: Optional[Dict[str, Any]]
    ) -> types.GenerateContentConfig:
        gen_cfg = self.config.get("generation_config", {}).get("params", {})
        think_level = self.config.get("thinking", {}).get("level", "HIGH").upper()

        if json_schema:  # keep only `type` and `properties`
            json_schema_for_generation = self.json_schema_to_gemini_response_schema(
                json_schema
            )
        else:
            json_schema_for_generation = None

        return types.GenerateContentConfig(
            temperature=gen_cfg.get("temperature", 1.0),
            max_output_tokens=gen_cfg.get("max_output_tokens", 5000),
            system_instruction=self.config.get("prompt_params", {}).get(
                "developer_message", ""
            ),
            thinking_config=types.ThinkingConfig(
                thinking_level=getattr(
                    types.ThinkingLevel, think_level, types.ThinkingLevel.LOW
                )
            ),
            response_mime_type="application/json" if json_schema else None,
            response_schema=json_schema_for_generation if json_schema else None,
        )

    def json_schema_to_gemini_response_schema(self, schema: dict) -> dict:
        """
        Convert a (subset of) JSON Schema draft-07-ish dicts into Gemini `response_schema` dicts.

        What it fixes:
        - JSON Schema lowercase types -> Gemini enum types (OBJECT, STRING, ...)
        - Nullable unions like {"type": ["string","null"]} -> {"type":"STRING","nullable": True}
        - Also supports nullable via anyOf/oneOf: [{"type":"string"}, {"type":"null"}]

        What it supports (recursively):
        - object: properties, required
        - array: items, minItems/maxItems
        - scalar: minimum/maximum, minLength/maxLength, enum, title/description

        Raises:
        - TypeError / ValueError for schema shapes that can’t be expressed in Gemini response_schema.
        """
        TYPE = {
            "string": "STRING",
            "number": "NUMBER",
            "integer": "INTEGER",
            "boolean": "BOOLEAN",
            "array": "ARRAY",
            "object": "OBJECT",
            "null": "NULL",
        }
        KEEP = {
            "title",
            "description",
            "minimum",
            "maximum",
            "minItems",
            "maxItems",
            "minLength",
            "maxLength",
            "enum",
        }

        def conv(node: object) -> dict:
            if not isinstance(node, dict):
                raise TypeError(
                    f"Schema node must be an object, got {type(node).__name__}"
                )

            # Drop JSON-Schema-only fields commonly rejected by Gemini's Schema
            node = {k: v for k, v in node.items() if k != "$schema"}

            # Nullable via anyOf/oneOf: [{"type":"<t>"}, {"type":"null"}]
            for key in ("anyOf", "oneOf"):
                if key in node and "type" not in node:
                    opts = node.get(key)
                    if not (isinstance(opts, list) and len(opts) == 2):
                        raise ValueError(
                            f"Unsupported {key}: expected exactly 2 options for nullable union."
                        )
                    null_opt = next(
                        (
                            o
                            for o in opts
                            if isinstance(o, dict) and o.get("type") == "null"
                        ),
                        None,
                    )
                    other = next((o for o in opts if o is not null_opt), None)
                    if null_opt is None or not isinstance(other, dict):
                        raise ValueError(
                            f"Unsupported {key}: only <T>|null unions are supported."
                        )
                    out = conv(other)
                    out["nullable"] = True
                    return out

            out = {k: node[k] for k in KEEP if k in node}

            t = node.get("type")
            nullable = False

            # Nullable via type union: ["string","null"]
            if isinstance(t, list):
                types = [x for x in t if isinstance(x, str)]
                nullable = "null" in types
                non_null = [x for x in types if x != "null"]
                if nullable and len(non_null) == 1:
                    t = non_null[0]
                elif types == ["null"] or types == [
                    "null",
                ]:
                    t = "null"
                else:
                    raise ValueError(
                        f"Unsupported type union {t}. Only ['<T>', 'null'] is supported."
                    )

            out["type"] = TYPE.get(t, "TYPE_UNSPECIFIED")
            if nullable:
                out["nullable"] = True

            if out["type"] == "OBJECT":
                props = node.get("properties") or {}
                if not isinstance(props, dict):
                    raise ValueError('"properties" must be an object when type=object.')
                out["properties"] = {name: conv(sub) for name, sub in props.items()}
                req = node.get("required") or []
                if not (isinstance(req, list) and all(isinstance(x, str) for x in req)):
                    raise ValueError(
                        '"required" must be a list of strings when type=object.'
                    )
                out["required"] = req

            if out["type"] == "ARRAY":
                items = node.get("items")
                out["items"] = (
                    conv(items)
                    if isinstance(items, dict)
                    else {"type": "TYPE_UNSPECIFIED"}
                )

            return out

        result = conv(schema)
        if result.get("type") != "OBJECT":
            raise ValueError("Top-level schema must be type=object.")
        return result

    def generate(
        self,
        prompt_input: Any,
        history: Optional[List[types.Content]] = None,
        json_schema: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Singular instance generation. Clean slate by default.
        Handles both raw strings and GenerationRequest objects from the orchestrator.

        Args:
            prompt_input (Any): The input prompt, can be a string, GenerationRequest object, or dict.
            history (Optional[List[types.Content]]): Optional conversation history.
            json_schema (Optional[Dict[str, Any]]): Optional JSON schema for structured output.
            **kwargs: Additional generation parameters.

        Returns:
            str: The generated text response.

        Raises:
            RuntimeError: If generation fails after retry.
        """
        # 1. UNWRAP the prompt if it's a GenerationRequest object
        if hasattr(prompt_input, "prompt"):
            actual_prompt = prompt_input.prompt
        elif isinstance(prompt_input, dict) and "prompt" in prompt_input:
            actual_prompt = prompt_input["prompt"]
        else:
            actual_prompt = str(prompt_input)

        # 2. Prepare Config (Stateless)
        config = self._prepare_config(json_schema)

        # 3. Build contents list
        contents = []
        if history:
            contents.extend(history)

        contents.append(
            types.Content(role="user", parts=[types.Part(text=actual_prompt)])
        )

        try:
            try:
                response = self._client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
            except Exception as e:
                logger.warning(
                    f"Gemini 3 Generation Failed, trying again in {self._sleep_seconds} seconds..."
                )
                time.sleep(self._sleep_seconds)
                response = self._client.models.generate_content(
                    model=self.model_name, contents=contents, config=config
                )
            text = response.text.strip()
            record_generation_success(self.model_name)
            return text
        except Exception as e:
            record_generation_failure(self.model_name, e)
            logger.error(f"Gemini 3 Generation Failed: {e}")
            if "You exceeded your current quota" in str(e):
                logger.error("You exceeded your current quota. Exiting the program.")
                # sys.exit(1)
            raise RuntimeError(f"Model {self.model_name} failed turn.") from e

    def generate_batch(self, requests: List[Any]) -> List[str]:
        """
        Batch evaluation entry point for Stage 4.
        Ensures a clean slate for every sample in the batch.

        Args:
            requests (List[Any]): List of generation requests.

        Returns:
            List[str]: Generated outputs in the same order as requests.
        """
        results = []
        for req in requests:
            try:
                # Extract schema if present in the request object
                schema = getattr(req, "json_schema", None)
                response = self.generate(req, history=None, json_schema=schema)
                results.append(response)
            except Exception as e:
                logger.error(f"Error generating response for request: {e}")
                results.append(f"ERROR: {str(e)}")
        return results
