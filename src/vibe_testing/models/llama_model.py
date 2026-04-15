"""Implementation for Llama 3.3 models loaded from HuggingFace."""

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


class LlamaModel(BaseModel):
    """
    A concrete implementation for Llama 3.3 models.
    Uses proper chat templates and bfloat16 precision.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes and loads the Llama model and tokenizer.

        Args:
            model_name (str): The HuggingFace model identifier.
            config (Dict[str, Any]): Configuration dictionary.
        """
        super().__init__(model_name, config)

        device = self.config.get("device", "auto")
        cache_dir = self.config.get("cache_dir", get_models_cache_dir("transformers"))

        # Llama 3.3 specific: Ensure we trust remote code if needed, though usually not for main Llama repo
        trust_remote_code = self.config.get("trust_remote_code", False)

        logger.info(f"Loading tokenizer for {self.model_name}...")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        # DEBUG: Log special tokens to ensure we loaded the right tokenizer
        logger.debug(f"Tokenizer special tokens: {self._tokenizer.special_tokens_map}")

        # CRITICAL: Llama models often have no pad token by default.
        # We set it to eos_token to avoid generation crashes.
        if self._tokenizer.pad_token is None:
            logger.info("Tokenizer has no pad_token. Setting pad_token = eos_token.")
            self._tokenizer.pad_token = self._tokenizer.eos_token

        logger.info(f"Loading model {self.model_name}...")
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device,
                dtype=torch.bfloat16,  # Llama 3 is trained in bfloat16
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            )
        except ImportError as e:
            logger.error(
                "Failed to load model. Ensure 'accelerate' and 'bitsandbytes' (if quantizing) are installed."
            )
            raise e

        logger.info(
            f"Initialized LlamaModel: {self.model_name} on device: {self._model.device}"
        )

        # Ensure decoder-only batching uses left padding to avoid warnings and misalignment
        if self._tokenizer.padding_side != "left":
            self._tokenizer.padding_side = "left"

    def generate(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        Generates a response using the Llama 3.3 chat template.

        Args:
            prompt (str): The user prompt.
            json_schema (Optional[Dict[str, Any]]): JSON schema to enforce structure.
            **kwargs: Generation parameters (temperature, max_new_tokens, etc.)
        """
        try:
            # Local models rely on global seeding; remove 'seed' from kwargs if present
            kwargs.pop("seed", None)

            # 1. Prepare Messages
            # Support both old flat config and new prompt section
            system_prompt = (
                self.config.get("prompt", {})
                .get("params", {})
                .get(
                    "system_prompt",
                    self.config.get("params", {}).get(
                        "system_prompt", "You are a helpful assistant."
                    ),
                )
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

            if json_schema:
                try:
                    schema_str = json.dumps(json_schema, indent=2)
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Please provide your response in a single, valid JSON format that adheres to the following schema:\n"
                                f"```json\n{schema_str}\n```\n"
                                "Do not include any text before or after the JSON."
                            ),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to serialize JSON schema: {e}. Proceeding without it."
                    )

            # 2. Apply Chat Template
            # This handles the specific <|begin_of_text|>, <|start_header_id|> logic for Llama 3
            try:
                final_prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.error(f"Failed to apply chat template: {e}")
                # Fallback to manual formatting if template fails (unlikely but safe)
                final_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"

            # DEBUG: Log the actual prompt to see what the model is receiving
            # Truncate if too long to avoid flooding logs
            log_prompt = (
                final_prompt[:500] + "..." if len(final_prompt) > 500 else final_prompt
            )
            logger.debug(f"Final Input Prompt:\n{log_prompt}")

            # 3. Tokenize
            inputs = self._tokenizer(final_prompt, return_tensors="pt").to(
                self._model.device
            )
            input_length = inputs.input_ids.shape[1]

            # 4. Generate
            # Get defaults from new generation_config section or fallback to flat params
            gen_defaults = self.config.get("generation_config", {}).get("params", {})

            # Merge priorities: kwargs (runtime) > generation_config (yaml) > flat params (yaml legacy) > hard defaults
            generation_params = {
                "max_new_tokens": gen_defaults.get(
                    "max_new_tokens", self.config.get("max_length", 2048)
                ),
                "temperature": gen_defaults.get(
                    "temperature", self.config.get("temperature", 0)
                ),
                "top_p": gen_defaults.get("top_p", self.config.get("top_p", 1)),
                "do_sample": gen_defaults.get(
                    "do_sample", self.config.get("do_sample", False)
                ),
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }
            generation_params.update(kwargs)

            with torch.no_grad():
                outputs = self._model.generate(**inputs, **generation_params)

            # 5. Decode
            generated_tokens = outputs[0, input_length:]
            response = self._tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            logger.debug(f"Raw Model Output:\n{response}")

            record_generation_success(self.model_name)
            return response
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise

    def generate_batch(
        self, requests: List[GenerationRequest], **kwargs: Any
    ) -> List[str]:
        """
        Batched generation for Llama 3.3 models. Falls back to per-call when schemas differ.

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

            # Get system prompt from config
            system_prompt = (
                self.config.get("prompt", {})
                .get("params", {})
                .get(
                    "system_prompt",
                    self.config.get("params", {}).get(
                        "system_prompt", "You are a helpful assistant."
                    ),
                )
            )

            # Prepare messages and tokenize for each request
            token_seqs: List[torch.Tensor] = []
            for req in requests:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": req.prompt},
                ]

                # Add JSON schema as additional user message if provided
                if req.json_schema:
                    try:
                        schema_str = json.dumps(req.json_schema, indent=2)
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Please provide your response in a single, valid JSON format that adheres to the following schema:\n"
                                    f"```json\n{schema_str}\n```\n"
                                    "Do not include any text before or after the JSON."
                                ),
                            }
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to serialize JSON schema: {e}. Proceeding without it."
                        )

                # Apply chat template
                try:
                    final_prompt = self._tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                except Exception as e:
                    logger.error(f"Failed to apply chat template: {e}")
                    # Fallback to manual formatting
                    final_prompt = (
                        f"System: {system_prompt}\nUser: {req.prompt}\nAssistant:"
                    )

                # Tokenize (chat template already includes special tokens, so we don't add extra)
                tokens = self._tokenizer(final_prompt, return_tensors="pt")
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
                    "max_new_tokens", self.config.get("max_length", 2048)
                ),
                "temperature": gen_defaults.get(
                    "temperature", self.config.get("temperature", 0)
                ),
                "top_p": gen_defaults.get("top_p", self.config.get("top_p", 1)),
                "do_sample": gen_defaults.get(
                    "do_sample", self.config.get("do_sample", False)
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

            # Decode each response, extracting only the generated portion
            results: List[str] = []
            for row_idx, output_row in enumerate(outputs):
                prompt_len = input_lengths[row_idx]
                generated_tokens = output_row[prompt_len:]
                response = self._tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                results.append(response)

            record_generation_success(self.model_name)
            return results
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise
