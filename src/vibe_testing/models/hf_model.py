"""Implementation for models loaded from HuggingFace."""

import json
import logging
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.vibe_testing.utils import get_models_cache_dir
from .base import BaseModel, GenerationRequest
from .generation_failure_tracker import (
    record_generation_failure,
    record_generation_success,
)

logger = logging.getLogger(__name__)


class HFModel(BaseModel):
    """
    A concrete implementation for language models that are loaded from the
    HuggingFace Hub.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes and loads the HuggingFace model and tokenizer.

        Args:
            model_name (str): The HuggingFace model identifier
                              (e.g., 'codellama/CodeLlama-7b-hf').
            config (Dict[str, Any]): Configuration dictionary containing details
                                     like quantization, device mapping, etc.
        """
        super().__init__(model_name, config)

        device = self.config.get("device", "auto")
        cache_dir = self.config.get("cache_dir", get_models_cache_dir("transformers"))

        logger.info("Loading tokenizer for %s...", self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir
        )

        logger.info("Loading model %s...", self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=device,
            dtype=torch.bfloat16,  # Use bfloat16 for better performance
            cache_dir=cache_dir,
        )
        logger.info("Initialized HFModel: %s", self.model_name)

    def generate(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        Generates a response using the loaded HuggingFace model.

        Args:
            prompt (str): The input prompt for the model.
            json_schema (Optional[Dict[str, Any]]): If provided, the schema is added
                                                     to the prompt as a guide.
            **kwargs: Additional generation parameters (e.g., temperature,
                      max_new_tokens).

        Returns:
            str: The model's generated output.
        """
        logger.debug(
            "Generating response for model '%s' (prompt prefix: '%s')",
            self.model_name,
            prompt[:50].replace("\n", " "),
        )

        # Local HF models rely on global seeding (seed_everything / transformers.set_seed).
        # Do not forward per-call seed into the underlying model.generate.
        kwargs.pop("seed", None)

        final_prompt = prompt
        if json_schema:
            schema_str = json.dumps(json_schema, indent=2)
            final_prompt = (
                f"{prompt}\n\n"
                f"Please provide your response in a single, valid JSON format that adheres to the following schema:\n"
                f"```json\n{schema_str}\n```"
            )

        # Combine default config params with any runtime kwargs
        generation_params = {
            "max_new_tokens": self.config.get("max_length", 1024),
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 1.0),
            "do_sample": self.config.get("do_sample", False),
            **kwargs,
        }

        inputs = self._tokenizer(final_prompt, return_tensors="pt").to(
            self._model.device
        )
        input_length = inputs.input_ids.shape[1]

        try:
            outputs = self._model.generate(**inputs, **generation_params)
            # Decode the output, skipping special tokens and the prompt
            generated_tokens = outputs[0, input_length:]
            text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            record_generation_success(self.model_name)
            return text
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise

    def generate_batch(
        self, requests: List[GenerationRequest], **kwargs: Any
    ) -> List[str]:
        """
        Generate responses for multiple prompts in one pass when schemas and kwargs align.

        Falls back to BaseModel.generate_batch when schemas differ or per-request kwargs
        are not uniform.
        """
        if not requests:
            return []

        # CURRENT -We want batch even with a schema
        # OLD - If any request includes a schema, rely on the safer default implementation.
        # if any(req.json_schema is not None for req in requests):
        #    return super().generate_batch(requests, **kwargs)

        first_kwargs = requests[0].generation_kwargs
        if not all(req.generation_kwargs == first_kwargs for req in requests):
            return super().generate_batch(requests, **kwargs)

        generation_params = {
            "max_new_tokens": self.config.get("max_length", 1024),
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 1.0),
            "do_sample": self.config.get("do_sample", False),
            **first_kwargs,
            **kwargs,
        }

        # Local HF models rely on global seeding; drop per-call seed hints.
        generation_params.pop("seed", None)

        prompts = [req.prompt for req in requests]
        inputs = self._tokenizer(prompts, return_tensors="pt", padding=True).to(
            self._model.device
        )
        # attention_mask sum gives prompt token lengths
        input_lengths = inputs["attention_mask"].sum(dim=1)

        try:
            outputs = self._model.generate(**inputs, **generation_params)

            decoded: List[str] = []
            for row_idx, output_row in enumerate(outputs):
                prompt_len = int(input_lengths[row_idx].item())
                generated_tokens = output_row[prompt_len:]
                decoded.append(
                    self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
                )
            record_generation_success(self.model_name)
            return decoded
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise
