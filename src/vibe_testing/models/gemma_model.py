"""Implementation for Gemma models loaded from HuggingFace."""

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
    from transformers import (
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )  # pragma: no cover

DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class GemmaModel(BaseModel):
    """
    A concrete implementation for Gemma models loaded from HuggingFace.

    This implementation uses Gemma's processor-based chat template flow and
    supports text-only prompting.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initialize and load the Gemma model and processor.

        Args:
            model_name (str): HuggingFace model identifier.
            config (Dict[str, Any]): Model configuration dictionary.
        """
        super().__init__(model_name, config)
        self._model: Any | None = None
        self._processor: Any | None = None
        self._tokenizer: Any | None = None

        self._load_model()
        self._ensure_padding_token()
        self._ensure_left_padding()
        logger.info("Initialized GemmaModel: %s", self.model_name)

    def _load_model(self) -> None:
        """
        Load Gemma model and processor from HuggingFace.

        Raises:
            RuntimeError: If required Gemma classes are unavailable.
        """
        if self._model is not None and self._processor is not None:
            logger.info("Model is already loaded.")
            return

        try:
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "GemmaModel requires transformers support for Gemma3. "
                "Please upgrade/install 'transformers' with Gemma classes available."
            ) from exc

        model_params = self.config.get("model", {}).get("params", {})
        device = model_params.get("device", self.config.get("device", "auto"))
        cache_dir = model_params.get(
            "cache_dir",
            self.config.get("cache_dir", get_models_cache_dir("transformers")),
        )
        trust_remote_code = model_params.get(
            "trust_remote_code", self.config.get("trust_remote_code", True)
        )

        logger.info("Loading processor for %s...", self.model_name)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )
        self._tokenizer = getattr(self._processor, "tokenizer", None)
        if self._tokenizer is None:
            raise ValueError(
                f"Processor for model '{self.model_name}' does not expose a tokenizer."
            )

        logger.info("Loading model %s...", self.model_name)
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=device,
            dtype=torch.bfloat16,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        ).eval()

    def _ensure_padding_token(self) -> None:
        """
        Ensure tokenizer has a usable pad token.

        Raises:
            ValueError: If both pad and EOS tokens are unavailable.
        """
        if self._tokenizer.pad_token_id is not None:
            return

        if self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            logger.info(
                "Tokenizer for '%s' had no pad token; using eos token as pad token.",
                self.model_name,
            )
            return

        raise ValueError(
            f"Tokenizer for model '{self.model_name}' has no pad_token_id and no eos_token. "
            "Cannot continue batched generation safely."
        )

    def _ensure_left_padding(self) -> None:
        """Force left padding for decoder-only batching behavior."""
        if self._tokenizer.padding_side != "left":
            self._tokenizer.padding_side = "left"

    def _build_messages(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Build Gemma chat-template messages for a single generation call.

        Args:
            prompt (str): User prompt text.
            json_schema (Optional[Dict[str, Any]]): Optional JSON schema hint.

        Returns:
            List[Dict[str, Any]]: Messages in processor chat-template format.
        """
        prompt_params = self.config.get("prompt", {}).get("params", {})
        system_prompt = prompt_params.get(
            "system_prompt",
            self.config.get("params", {}).get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        )

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]

        if json_schema:
            schema_str = json.dumps(json_schema, indent=2)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Please provide your response in a single, valid JSON format that "
                                "adheres to the following schema:\n"
                                f"```json\n{schema_str}\n```"
                            ),
                        }
                    ],
                }
            )
        return messages

    def _generation_params(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Build generation parameters from config defaults and runtime overrides.

        Args:
            **kwargs: Runtime generation overrides.

        Returns:
            Dict[str, Any]: Generation parameters for model.generate.
        """
        gen_defaults = self.config.get("generation_config", {}).get("params", {})
        params = {
            "max_new_tokens": gen_defaults.get(
                "max_new_tokens", self.config.get("max_length", 5000)
            ),
            "temperature": gen_defaults.get(
                "temperature", self.config.get("temperature", 0.7)
            ),
            "top_p": gen_defaults.get("top_p", self.config.get("top_p", 0.95)),
            "do_sample": gen_defaults.get(
                "do_sample", self.config.get("do_sample", False)
            ),
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        params.update(kwargs)
        return params

    def generate(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> str:
        """
        Generate a response using Gemma.

        Args:
            prompt (str): Input prompt text.
            json_schema (Optional[Dict[str, Any]]): Optional JSON schema hint.
            **kwargs (Any): Additional generation overrides.

        Returns:
            str: Generated text output.
        """
        try:
            logger.info(
                "Generating response for prompt (first 50 chars): '%s...'", prompt[:50]
            )

            # Local models rely on global seeding.
            kwargs.pop("seed", None)

            messages = self._build_messages(prompt, json_schema=json_schema)
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = inputs.get("input_ids")
            if input_ids is None:
                raise ValueError(
                    "Processor output is missing 'input_ids'; cannot run generation."
                )
            input_length = int(input_ids.shape[-1])

            model_inputs = {
                key: value.to(self._model.device)
                for key, value in inputs.items()
                if hasattr(value, "to")
            }
            outputs = self._model.generate(
                **model_inputs,
                **self._generation_params(**kwargs),
            )
            generated_tokens = outputs[0][input_length:]

            content = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            record_generation_success(self.model_name)
            return content.strip("\n")
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise

    def generate_batch(
        self, requests: List[GenerationRequest], **kwargs: Any
    ) -> List[str]:
        """
        Generate responses for multiple requests in a single batch call.

        Args:
            requests (List[GenerationRequest]): Requests to generate.
            **kwargs (Any): Additional generation overrides.

        Returns:
            List[str]: Generated responses in request order.
        """
        try:
            if not requests:
                return []

            first_kwargs = requests[0].generation_kwargs
            if not all(req.generation_kwargs == first_kwargs for req in requests):
                return super().generate_batch(requests, **kwargs)

            merged_gen_kwargs = {**first_kwargs, **kwargs}
            merged_gen_kwargs.pop("seed", None)

            token_seqs: List[torch.Tensor] = []
            for req in requests:
                messages = self._build_messages(req.prompt, json_schema=req.json_schema)
                tokenized = self._processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                input_ids = tokenized.get("input_ids")
                if input_ids is None:
                    raise ValueError(
                        "Processor output is missing 'input_ids'; cannot run batched generation."
                    )
                token_seqs.append(input_ids[0].to(self._model.device))

            pad_id = self._tokenizer.pad_token_id
            if pad_id is None:
                raise ValueError(
                    f"Tokenizer for model '{self.model_name}' still has no pad_token_id "
                    "after initialization."
                )
            max_len = max(seq.shape[0] for seq in token_seqs)
            padded_inputs = torch.full(
                (len(token_seqs), max_len),
                pad_id,
                device=self._model.device,
                dtype=torch.long,
            )
            attention_mask = torch.zeros_like(padded_inputs)
            for i, seq in enumerate(token_seqs):
                seq_len = seq.shape[0]
                padded_inputs[i, -seq_len:] = seq
                attention_mask[i, -seq_len:] = 1

            generation_params = self._generation_params(**merged_gen_kwargs)
            with torch.no_grad():
                outputs = self._model.generate(
                    input_ids=padded_inputs,
                    attention_mask=attention_mask,
                    **generation_params,
                )

            generated_start = padded_inputs.shape[1]
            results: List[str] = []
            for output_row in outputs:
                generated_tokens = output_row[generated_start:]
                content = self._tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                results.append(content.strip("\n"))

            record_generation_success(self.model_name)
            return results
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise
