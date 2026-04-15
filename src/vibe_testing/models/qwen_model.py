"""Implementation for Qwen models loaded from HuggingFace."""

import importlib
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

from src.vibe_testing.utils import get_models_cache_dir
from .base import BaseModel, GenerationRequest
from .generation_failure_tracker import (
    record_generation_failure,
    record_generation_success,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer  # pragma: no cover

DEFAULT_THINKING_PROMPT = (
    "You are a helpful assistant. "
    "Please first think about the question thoroughly. Consider multiple approaches and show your reasoning. "
    "Wrap your thinking in <think> and </think> tags and then return your final answer."
)


class QwenContextLengthError(RuntimeError):
    """
    Raised when a Qwen+Unsloth request is too close to or over the context limit.
    """

    def __init__(
        self,
        *,
        backend: str,
        max_seq_length: int,
        input_tokens: int,
        remaining_tokens: int,
        required_headroom_tokens: int,
        prompt_preview: Optional[str] = None,
    ) -> None:
        self.backend = backend
        self.max_seq_length = int(max_seq_length)
        self.input_tokens = int(input_tokens)
        self.remaining_tokens = int(remaining_tokens)
        self.required_headroom_tokens = int(required_headroom_tokens)
        self.prompt_preview = prompt_preview
        self.failure_reason = (
            "input_exceeds_context_limit"
            if self.remaining_tokens < 0
            else "insufficient_context_headroom"
        )
        preview_suffix = f" Prompt preview: {prompt_preview!r}" if prompt_preview else ""
        super().__init__(
            "Qwen Unsloth input length validation failed: "
            f"backend={backend} input_tokens={self.input_tokens} "
            f"max_seq_length={self.max_seq_length} "
            f"remaining_tokens={self.remaining_tokens} "
            f"required_headroom_tokens={self.required_headroom_tokens} "
            f"failure_reason={self.failure_reason}."
            f"{preview_suffix}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the validation failure for logs and saved artifacts.
        """
        return {
            "backend": self.backend,
            "max_seq_length": self.max_seq_length,
            "input_tokens": self.input_tokens,
            "remaining_tokens": self.remaining_tokens,
            "required_headroom_tokens": self.required_headroom_tokens,
            "failure_reason": self.failure_reason,
            "prompt_preview": self.prompt_preview,
            "message": str(self),
        }


class QwenModel(BaseModel):
    """
    A concrete implementation for Qwen models that are loaded from the
    HuggingFace Hub. Supports specific Qwen features like thinking tokens.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes and loads the Qwen model and tokenizer.

        Args:
            model_name (str): The HuggingFace model identifier
                              (e.g., 'Qwen/Qwen3-8B').
            config (Dict[str, Any]): Configuration dictionary containing details
                                     like quantization, device mapping, etc.
        """
        super().__init__(model_name, config)
        # NOTE: Avoid importing transformers at module import time so that, when using
        # the Unsloth backend, Unsloth can be imported before transformers.
        self._model: Any | None = None
        self._tokenizer: Any | None = None

        model_params = self.config.get("model", {}).get("params", {})
        self.backend = model_params.get("backend", "hf")
        self._strict_context_limit = self.backend == "unsloth"
        self._max_seq_length = (
            int(model_params.get("max_seq_length", 4096))
            if self._strict_context_limit
            else None
        )
        self._min_context_headroom_tokens = int(
            model_params.get("min_context_headroom_tokens", 100)
        )
        if self._min_context_headroom_tokens < 0:
            raise ValueError(
                "model.params.min_context_headroom_tokens must be >= 0. "
                f"Got: {self._min_context_headroom_tokens}"
            )

        self._load_model()

        # Ensure decoder-only batching uses left padding
        if self._tokenizer.padding_side != "left":
            self._tokenizer.padding_side = "left"

        # Get the </think> token ID for parsing thinking content
        # Try to encode the token first
        encoded = self._tokenizer.encode("</think>", add_special_tokens=False)
        if encoded:
            self._think_end_token_id = encoded[0]
        else:
            # Fallback: try to find the token ID directly (151668 as mentioned in snippet)
            try:
                self._think_end_token_id = self._tokenizer.convert_tokens_to_ids(
                    "</think>"
                )
                if (
                    self._think_end_token_id is None
                    or self._think_end_token_id == self._tokenizer.unk_token_id
                ):
                    raise ValueError("Token not found")
            except (Exception, ValueError):
                # If token not found, use the known ID from snippet
                self._think_end_token_id = 151668
                logger.warning(
                    f"Could not find </think> token, using fallback ID: {self._think_end_token_id}"
                )

        logger.info(f"Initialized QwenModel: {self.model_name}")

    def _load_model(self) -> None:
        """Loads the model backend and tokenizer."""
        if self._model is not None:
            logger.info("Model is already loaded.")
            return

        if self.backend == "unsloth":
            self._load_model_unsloth()
            return

        if self.backend not in {"hf"}:
            raise ValueError(
                f"Unsupported Qwen backend '{self.backend}'. Expected one of: hf, unsloth."
            )
        self._load_model_hf()

    def _load_model_hf(self) -> None:
        """
        Loads the Qwen model via HuggingFace transformers.

        This preserves the existing behavior (device_map, dtype='auto', trust_remote_code).
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_params = self.config.get("model", {}).get("params", {})
        device = model_params.get("device", self.config.get("device", "auto"))
        cache_dir = model_params.get(
            "cache_dir",
            self.config.get("cache_dir", get_models_cache_dir("transformers")),
        )
        trust_remote_code = model_params.get(
            "trust_remote_code", self.config.get("trust_remote_code", True)
        )

        logger.info("Loading tokenizer for %s...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        logger.info("Loading model %s...", self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device,
            dtype="auto",
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

    def _load_model_unsloth(self) -> None:
        """
        Loads the Qwen model via Unsloth (FastLanguageModel) for faster inference.

        Opt-in via config:
          model:
            name: qwen
            params:
              backend: unsloth
              model_name: "<unsloth qwen repo>"
              dtype: null
              max_seq_length: 4096
              load_in_4bit: false
              full_finetuning: false
              cache_dir: "/mnt/nlp/models"
              hf_token: "hf_..."  # optional
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Unsloth backend requires CUDA (torch.cuda.is_available() is False)."
            )

        try:
            unsloth_mod = importlib.import_module("unsloth")
        except Exception as exc:  # noqa: BLE001
            raise ImportError(
                "Unsloth backend requested but 'unsloth' is not installed. "
                "Install it (and its CUDA deps) in your environment, or set model.params.backend='hf'."
            ) from exc

        if not hasattr(unsloth_mod, "FastLanguageModel"):
            raise ImportError(
                "Unsloth import succeeded but FastLanguageModel was not found. "
                "This likely indicates an incompatible 'unsloth' installation."
            )

        FastLanguageModel = getattr(unsloth_mod, "FastLanguageModel")

        model_params = self.config.get("model", {}).get("params", {})
        max_seq_length = int(model_params.get("max_seq_length", 4096))
        load_in_4bit = bool(model_params.get("load_in_4bit", False))
        full_finetuning = bool(model_params.get("full_finetuning", False))
        dtype = model_params.get("dtype", None)
        hf_token = model_params.get("hf_token")
        cache_dir = model_params.get("cache_dir", get_models_cache_dir("transformers"))

        logger.info(
            "Loading Qwen via Unsloth: model_name=%s max_seq_length=%s load_in_4bit=%s full_finetuning=%s",
            self.model_name,
            max_seq_length,
            load_in_4bit,
            full_finetuning,
        )

        from_pretrained_kwargs: Dict[str, Any] = {
            "model_name": self.model_name,
            "dtype": dtype,
            "max_seq_length": max_seq_length,
            "load_in_4bit": load_in_4bit,
            "full_finetuning": full_finetuning,
            "cache_dir": cache_dir,
        }
        if hf_token:
            from_pretrained_kwargs["token"] = hf_token

        model, tokenizer = FastLanguageModel.from_pretrained(**from_pretrained_kwargs)
        FastLanguageModel.for_inference(model)

        self._model = model
        self._tokenizer = tokenizer

        if getattr(self._tokenizer, "padding_side", None) != "left":
            self._tokenizer.padding_side = "left"

    def _build_messages(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, str]]:
        """
        Build the chat-template messages for a prompt.
        """
        prompt_params = self.config.get("prompt", {}).get("params", {})
        system_prompt = prompt_params.get(
            "system_prompt",
            self.config.get("params", {}).get(
                "system_prompt", DEFAULT_THINKING_PROMPT
            ),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if json_schema:
            schema_str = json.dumps(json_schema, indent=2)
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Please provide your response in a single, valid JSON format that "
                        "adheres to the following schema:\n"
                        f"```json\n{schema_str}\n```"
                    ),
                }
            )
        return messages

    def _render_prompt(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Render the final chat-formatted prompt string.
        """
        messages = self._build_messages(prompt, json_schema=json_schema)
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _tokenize_prompt(self, final_prompt: str) -> Any:
        """
        Tokenize a rendered prompt without applying truncation.
        """
        return self._tokenizer(final_prompt, return_tensors="pt")

    def _validate_prompt_length(self, final_prompt: str) -> None:
        """
        Enforce strict prompt-length validation for the Unsloth backend.
        """
        if not self._strict_context_limit:
            return
        if self._max_seq_length is None:
            raise RuntimeError(
                "Qwen strict context validation is enabled but max_seq_length is missing."
            )
        tokenized = self._tokenize_prompt(final_prompt)
        input_tokens = int(tokenized.input_ids.shape[1])
        remaining_tokens = int(self._max_seq_length - input_tokens)
        if remaining_tokens >= self._min_context_headroom_tokens:
            return
        prompt_preview = final_prompt[:200].replace("\n", "\\n")
        raise QwenContextLengthError(
            backend=self.backend,
            max_seq_length=self._max_seq_length,
            input_tokens=input_tokens,
            remaining_tokens=remaining_tokens,
            required_headroom_tokens=self._min_context_headroom_tokens,
            prompt_preview=prompt_preview,
        )

    def validate_generation_request(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """
        Validate a request before it is sent to Qwen generation.
        """
        del kwargs
        final_prompt = self._render_prompt(prompt, json_schema=json_schema)
        self._validate_prompt_length(final_prompt)

    def generate(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        Generates a response using the loaded Qwen model.

        Args:
            prompt (str): The input prompt for the model.
            json_schema (Optional[Dict[str, Any]]): If provided, the schema is added
                                                     to the prompt as a guide.
            **kwargs: Additional generation parameters (e.g., temperature,
                      max_new_tokens).

        Returns:
            str: The model's generated output.
        """
        try:
            logger.info(
                f"Generating response for prompt (first 50 chars): '{prompt[:50]}...'"
            )

            # Local models rely on global seeding.
            kwargs.pop("seed", None)
            final_prompt = self._render_prompt(prompt, json_schema=json_schema)
            self._validate_prompt_length(final_prompt)

            # Get generation config from config file or use defaults for thinking mode
            gen_defaults = self.config.get("generation_config", {}).get("params", {})

            # For thinking mode, use Temperature=0.6, TopP=0.95, TopK=20, MinP=0 (not greedy)
            generation_params = {
                "max_new_tokens": gen_defaults.get(
                    "max_new_tokens", self.config.get("max_length", 32768)
                ),
                "temperature": gen_defaults.get(
                    "temperature", self.config.get("temperature", 0.6)
                ),
                "top_p": gen_defaults.get("top_p", self.config.get("top_p", 0.95)),
                "top_k": gen_defaults.get("top_k", self.config.get("top_k", 20)),
                "min_p": gen_defaults.get("min_p", self.config.get("min_p", 0.0)),
                "do_sample": gen_defaults.get(
                    "do_sample", self.config.get("do_sample", True)
                ),
                **kwargs,
            }

            inputs = self._tokenize_prompt(final_prompt).to(self._model.device)
            input_length = inputs.input_ids.shape[1]

            outputs = self._model.generate(**inputs, **generation_params)

            # Decode the output, skipping special tokens and the prompt
            generated_tokens = outputs[0, input_length:].tolist()

            # Parse thinking content: find </think> token and extract content after it
            content = self._parse_thinking_content(generated_tokens)
            record_generation_success(self.model_name)
            return content
        except QwenContextLengthError:
            raise
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise

    def _parse_thinking_content(self, output_ids: List[int]) -> str:
        """
        Parses thinking content from Qwen3 output tokens.

        Finds the </think> token (ID 151668) and extracts content after it.
        If </think> is not found, returns the full decoded content.

        Args:
            output_ids (List[int]): List of token IDs from the model output.

        Returns:
            str: The final content after thinking tokens, or full content if no </think> found.
        """
        try:
            # Find the last occurrence of </think> token
            index = len(output_ids) - output_ids[::-1].index(self._think_end_token_id)
        except ValueError:
            # </think> token not found, return full content
            index = 0

        # Extract content after </think>
        content_ids = output_ids[index:]
        content = self._tokenizer.decode(content_ids, skip_special_tokens=True).strip(
            "\n"
        )

        return content

    def generate_batch(
        self, requests: List[GenerationRequest], **kwargs: Any
    ) -> List[str]:
        """
        Batched generation for Qwen3 models. Falls back to per-call when schemas differ.

        Args:
            requests: List of generation requests with prompts and optional JSON schemas.
            **kwargs: Additional generation parameters applied to all requests.

        Returns:
            List of generated responses in the same order as requests.
        """
        try:
            if not requests:
                return []

            # If per-request kwargs differ, use sequential fallback to respect overrides
            first_kwargs = requests[0].generation_kwargs
            if not all(req.generation_kwargs == first_kwargs for req in requests):
                return super().generate_batch(requests, **kwargs)

            # Drop seed hint (local models rely on global seeding)
            merged_gen_kwargs = {**first_kwargs, **kwargs}
            merged_gen_kwargs.pop("seed", None)

            # Prepare messages and tokenize for each request
            token_seqs: List[torch.Tensor] = []
            for req in requests:
                final_prompt = self._render_prompt(
                    req.prompt, json_schema=req.json_schema
                )
                self._validate_prompt_length(final_prompt)

                # Tokenize
                tokens = self._tokenize_prompt(final_prompt)
                tensor_ids = tokens.input_ids[0].to(self._model.device)
                token_seqs.append(tensor_ids)

            # Left-pad sequences to batch
            pad_id = (
                self._tokenizer.pad_token_id
                if self._tokenizer.pad_token_id is not None
                else self._tokenizer.eos_token_id
            )
            max_len = max(seq.shape[0] for seq in token_seqs)
            padded_inputs = torch.full(
                (len(token_seqs), max_len),
                pad_id,
                device=self._model.device,
                dtype=torch.long,
            )
            attention_mask = torch.zeros_like(padded_inputs)
            input_lengths: List[int] = []
            for i, seq in enumerate(token_seqs):
                seq_len = seq.shape[0]
                input_lengths.append(seq_len)
                # Left padding: place sequence at the end
                padded_inputs[i, -seq_len:] = seq
                attention_mask[i, -seq_len:] = 1

            # Generation config
            gen_defaults = self.config.get("generation_config", {}).get("params", {})

            generation_params = {
                "max_new_tokens": gen_defaults.get(
                    "max_new_tokens", self.config.get("max_length", 5000)
                ),
                "temperature": gen_defaults.get(
                    "temperature", self.config.get("temperature", 0.6)
                ),
                "top_p": gen_defaults.get("top_p", self.config.get("top_p", 0.95)),
                "top_k": gen_defaults.get("top_k", self.config.get("top_k", 20)),
                "min_p": gen_defaults.get("min_p", self.config.get("min_p", 0.0)),
                "do_sample": gen_defaults.get(
                    "do_sample", self.config.get("do_sample", True)
                ),
                "pad_token_id": pad_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            generation_params.update(merged_gen_kwargs)

            # Generate in batch
            with torch.no_grad():
                try:
                    outputs = self._model.generate(
                        input_ids=padded_inputs,
                        attention_mask=attention_mask,
                        **generation_params,
                    )
                except Exception as e:
                    logger.warning(
                        f"Batch generation failed, falling back to per-call generation: {e}"
                    )
                    logger.debug(f"Batch generation error details: {e}", exc_info=True)
                    torch.cuda.empty_cache()
                    return super().generate_batch(requests, **kwargs)

            # Decode each response, extracting only the generated portion and parsing thinking content
            results: List[str] = []
            for row_idx, output_row in enumerate(outputs):
                prompt_len = input_lengths[row_idx]
                generated_tokens = output_row[prompt_len:].tolist()
                content = self._parse_thinking_content(generated_tokens)
                results.append(content)

            record_generation_success(self.model_name)
            return results
        except QwenContextLengthError:
            raise
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise
