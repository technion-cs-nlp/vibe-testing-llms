"""
Implementation for the GPT-OSS model from HuggingFace, using a chat-based approach.
"""

import gc
import importlib
import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import ConfigDict
import torch
from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    load_harmony_encoding,
    Message,
    Role,
)
from src.vibe_testing.utils import get_models_cache_dir
from .base import BaseModel, GenerationRequest
from .generation_failure_tracker import (
    record_generation_failure,
    record_generation_success,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer  # pragma: no cover


class GptOssModel(BaseModel):
    """
    Model class for gpt-oss that handles prompt formatting, model invocation,
    and response parsing for chat-based generation.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes and loads the GptOssModel.

        Args:
            model_name (str): The HuggingFace model identifier.
            config (Dict[str, Any]): The full configuration dictionary, which may
                                     contain model, prompt, and generation params.
        """
        super().__init__(model_name, config)
        # NOTE: Avoid importing transformers at module import time so that, when using
        # the Unsloth backend, Unsloth can be imported before transformers (per Unsloth
        # optimization guidance).
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.harmony_encoding = None
        self.quantizatied_model_path = "/mnt/nlp/models/gpt-oss-20b-dequantized-bf16"
        self.backend = (
            self.config.get("model", {}).get("params", {}).get("backend", "hf")
        )

        self._load_model()

    def _load_model(self) -> None:
        """Loads the model backend and tokenizer."""
        if self.model is not None:
            logger.info("Model is already loaded.")
            return

        if self.backend == "unsloth":
            self._load_model_unsloth()
            return
        if self.backend not in {"hf", "harmony"}:
            raise ValueError(
                f"Unsupported gpt-oss backend '{self.backend}'. Expected one of: hf, harmony, unsloth."
            )
        self._load_model_harmony()

    def _load_model_harmony(self) -> None:
        """
        Loads the GPT-OSS model via standard HuggingFace transformers and initializes the
        OpenAI Harmony encoding for chat-style prompt formatting and parsing.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_params = self.config.get("model", {}).get("params", {})
        device = model_params.get("device", "auto")
        cache_dir = model_params.get("cache_dir", get_models_cache_dir("transformers"))

        logger.info(f"Loading gpt-oss model '{self.model_name}'...")

        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     llm_int8_target_modules=[
        #         "q_proj",
        #         "k_proj",
        #         "v_proj",
        #         "o_proj",  # Attention
        #         "gate_up_proj",
        #         "down_proj",  # Experts (MoE)
        #         "lm_head",  # Optional: Quantize head for extra savings
        #     ],
        # )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=cache_dir
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=cache_dir,
            # dtype=torch.bfloat16,
            dtype="auto",
            device_map=device,
            # force_download=True,
            # attn_implementation="eager",
        )

        # # 1. Load the raw configuration
        # config = AutoConfig.from_pretrained(
        #     self.quantizatied_model_path, trust_remote_code=True
        # )

        # # 2. SYSTEMIC FIX: Modify the object's internal state directly
        # # We cannot use 'del config.torch_dtype' because it's a property.
        # # We cannot use 'AutoConfig.from_dict' because it's a factory.
        # # So we modify the underlying __dict__ to remove the data causing the issues.

        # # Remove quantization metadata so BnB can take over
        # if "quantization_config" in config.__dict__:
        #     config.__dict__.pop("quantization_config")

        # # Remove precision enforcement
        # # We remove both 'dtype' (the source) and 'torch_dtype' (the property cache)
        # config.__dict__.pop("dtype", None)
        # config.__dict__.pop("torch_dtype", None)

        # # 3. Re-instantiate a clean Config object
        # # This creates a new config without the "baggage" of the old file
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     self.quantizatied_model_path,
        #     config=config,
        #     cache_dir=cache_dir,
        #     # dtype=torch.bfloat16,
        #     # device_map=device,
        #     device_map="auto",
        #     attn_implementation="flash_attention_2",
        #     quantization_config=bnb_config,
        #     trust_remote_code=True,
        # )
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.quantizatied_model_path, cache_dir=cache_dir
        # )
        # Ensure decoder-only batching uses left padding to avoid warnings and misalignment.
        if self.tokenizer.padding_side != "left":
            self.tokenizer.padding_side = "left"
        self.harmony_encoding = load_harmony_encoding(
            HarmonyEncodingName.HARMONY_GPT_OSS
        )
        logger.info(
            f"gpt-oss model '{self.model_name}' loaded successfully on device '{self.model.device}'."
        )

    def _load_model_unsloth(self) -> None:
        """
        Loads GPT-OSS via Unsloth (FastLanguageModel) for faster inference.

        This path is opt-in via config:
          model:
            name: GptOssModel
            params:
              backend: unsloth
              model_name: "unsloth/gpt-oss-20b"
              max_seq_length: 4096
              dtype: null
              load_in_4bit: false
              full_finetuning: false
              # hf_token: "hf_..."  # optional, for gated models

        Notes:
        - This backend requires CUDA.
        - Harmony formatting/parsing is not used on this backend. GPT-OSS uses
          tokenizer.apply_chat_template(..., reasoning_effort=...).
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Unsloth backend requires CUDA (torch.cuda.is_available() is False)."
            )

        if "gpt-oss" not in self.model_name:
            raise ValueError(
                "Unsloth backend is only supported for GPT-OSS models. "
                f"Got model_name='{self.model_name}'. "
                "Use an Unsloth GPT-OSS repo like 'unsloth/gpt-oss-20b' or "
                "'unsloth/gpt-oss-20b-unsloth-bnb-4bit'."
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
            "Loading GPT-OSS via Unsloth: model_name=%s max_seq_length=%s load_in_4bit=%s full_finetuning=%s",
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

        self.model = model
        self.tokenizer = tokenizer
        self.harmony_encoding = None

        if getattr(self.tokenizer, "padding_side", None) != "left":
            self.tokenizer.padding_side = "left"

        logger.info(
            "Unsloth model '%s' loaded successfully on CUDA.",
            self.model_name,
        )

    def generate(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        Generates text using the loaded gpt-oss model.

        Args:
            prompt (str): The user input prompt.
            json_schema (Optional[Dict[str, Any]]): A JSON schema to enforce on the output.
            **kwargs: Additional generation parameters to override config.

        Returns:
            str: The generated output string from the 'final' channel.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError(
                "Model and tokenizer must be loaded before generating text."
            )

        if self.backend == "unsloth":
            return self._generate_unsloth(
                prompt=prompt, json_schema=json_schema, **kwargs
            )

        if self.harmony_encoding is None:
            raise ValueError(
                "Harmony encoding must be loaded for the HF/Harmony backend. "
                "If you intended to use Unsloth, set model.params.backend='unsloth'."
            )

        # Local GPT-OSS (HF) models also rely on global seeding. Do not pass
        # a per-call seed down into the underlying generate call.
        kwargs.pop("seed", None)

        prompt_params = self.config.get("prompt", {}).get("params", {})
        system_message = prompt_params.get("system_message", "")
        developer_message = prompt_params.get("developer_message", "")

        if json_schema:
            schema_string = json.dumps(json_schema)
            # ahem JSON schema is not a valid JSON with comments...
            schema_name = json_schema.get("title", "schema")
            developer_message += (
                f"\n\n# Response Formats\n\n## {schema_name}\n\n{schema_string}<|end|>"
            )

        conversation = Conversation.from_messages(
            [
                Message.from_role_and_content(Role.SYSTEM, system_message),
                Message.from_role_and_content(Role.DEVELOPER, developer_message),
                Message.from_role_and_content(Role.USER, prompt),
            ]
        )

        input_tokens = self.harmony_encoding.render_conversation_for_completion(
            conversation, Role.ASSISTANT
        )

        inputs = torch.tensor([input_tokens], device=self.model.device)
        input_length = inputs.shape[1]

        # Combine generation configs
        generation_config_params = self.config.get("generation_config", {}).get(
            "params", {}
        )
        final_gen_config = generation_config_params.copy()
        for key, value in kwargs.items():
            final_gen_config[key] = value

        # Add harmony-specific stop tokens
        stop_token_ids = self.tokenizer.convert_tokens_to_ids(
            ["<|return|>", "<|call|>"]
        )
        if "eos_token_id" not in final_gen_config:
            final_gen_config["eos_token_id"] = stop_token_ids
        elif isinstance(final_gen_config["eos_token_id"], int):
            final_gen_config["eos_token_id"] = [
                final_gen_config["eos_token_id"]
            ] + stop_token_ids
        else:  # is a list
            final_gen_config["eos_token_id"].extend(stop_token_ids)
            # Remove duplicates
            final_gen_config["eos_token_id"] = list(
                set(final_gen_config["eos_token_id"])
            )

        with torch.no_grad():
            try:
                outputs = self.model.generate(input_ids=inputs, **final_gen_config)
            except Exception as e:
                try:
                    # empty pytorch cache
                    torch.cuda.empty_cache()
                    outputs = self.model.generate(input_ids=inputs, **final_gen_config)
                except Exception as e:
                    logger.error(f"Error generating output: {e}")
                    logger.error(f"Input prompt: {prompt}")
                    logger.error(f"Final generation config: {final_gen_config}")
                    logger.error(f"Retunring empty reponse!")
                    if json_schema:
                        empty_response = json.dumps(json_schema)
                    else:
                        empty_response = "FAILED TO PARSE GENERATED TOKENS"
                    record_generation_failure(self.model_name, e)
                    return empty_response
                logger.error(f"Error generating output: {e}")
                logger.error(f"Input prompt length: {input_length}")
                logger.error(f"Input prompt tail 100 characters: |||{prompt[-100:]}|||")
                logger.error(f"Final generation config: {final_gen_config}")
                logger.error(f"Retunring empty reponse!")
                if json_schema:
                    empty_response = json.dumps(json_schema)
                else:
                    empty_response = "FAILED TO PARSE GENERATED TOKENS"
                record_generation_failure(self.model_name, e)
                return empty_response
                # raise e

        generated_tokens = outputs[0, input_length:]

        # Parse the generated tokens to extract the final response from harmony format
        try:
            try:
                parsed_messages = (
                    self.harmony_encoding.parse_messages_from_completion_tokens(
                        generated_tokens.tolist(), Role.ASSISTANT
                    )
                )
            except Exception as e:
                if "unexpected tokens remaining in message header" in str(e):
                    # raw_tokenized_messages = self.tokenizer.batch_decode(
                    #     generated_tokens.tolist()
                    # )
                    # index_of_json = raw_tokenized_messages.index("json")
                    # parsed_messages = [
                    #     Message.from_role_and_content(
                    #         Role.ASSISTANT,
                    #         "".join(raw_tokenized_messages[index_of_final:]),
                    #     )
                    # ]
                    # return "".join(raw_tokenized_messages[index_of_json + 2 :])
                    logger.error(f"Error parsing generated tokens: {e}")
                    logger.error(
                        f"Returning FAILED TO PARSE GENERATED TOKENS response!"
                    )
                    record_generation_failure(self.model_name, e)
                    return "FAILED TO PARSE GENERATED TOKENS"
        except Exception as e:
            logger.error(f"Error parsing generated tokens")
            logger.error(
                f"Manully decoded tokens using tokenizer: {self.tokenizer.batch_decode(generated_tokens)}"
            )
            record_generation_failure(self.model_name, e)
            raise e

        final_response = None
        for msg in parsed_messages:
            if msg.channel == "final":
                final_response = msg.content
                break

        if final_response is None:
            logger.error("No final response found in the generated tokens.")
            logger.error(f"Input prompt: {prompt}")
            logger.error(f"Final generation config: {final_gen_config}")
            logger.error(f"Parsed messages: {parsed_messages}")
            logger.error(f"Last message content: {parsed_messages[-1].content}")
            logger.error(f"Returning empty response!")
            exc = ValueError("No final response found in the generated tokens.")
            record_generation_failure(self.model_name, exc)
            raise exc
        text = final_response[0].text.strip()
        record_generation_success(self.model_name)
        return text

    def _generate_unsloth(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> str:
        """
        Generates text using the Unsloth backend.

        This backend uses the GPT-OSS chat template and supports `reasoning_effort`
        via tokenizer.apply_chat_template(..., reasoning_effort=...).
        """
        assert self.model is not None
        assert self.tokenizer is not None

        # Local models rely on global seeding; do not forward per-call seed hints.
        kwargs.pop("seed", None)

        prompt_params = self.config.get("prompt", {}).get("params", {})
        system_message = prompt_params.get("system_message", "")
        developer_message = prompt_params.get("developer_message", "")
        reasoning_effort = prompt_params.get("reasoning_effort", "low")

        schema_suffix = ""
        if json_schema:
            schema_suffix = (
                "\n\n"
                "Please provide your response in a single, valid JSON format that adheres to the following schema.\n"
                "Do not include any text before or after the JSON.\n"
                f"{json.dumps(json_schema, indent=2)}"
            )

        def _cuda_is_usable() -> bool:
            """
            Guard CUDA moves in environments without actual GPUs.

            Some CI environments (and unit tests) monkeypatch `torch.cuda.is_available()`
            but still run without a functional CUDA runtime. Calling `.to("cuda")` in
            those cases triggers CUDA lazy init and raises.
            """
            try:
                return (
                    bool(torch.cuda.is_available())
                    and int(torch.cuda.device_count()) > 0
                )
            except Exception:  # noqa: BLE001
                return False

        target_device = "cuda" if _cuda_is_usable() else "cpu"

        try:
            messages: List[Dict[str, str]] = []
            combined_system = "\n\n".join(
                [part for part in [system_message, developer_message] if part]
            ).strip()
            if combined_system:
                messages.append({"role": "system", "content": combined_system})
            messages.append({"role": "user", "content": f"{prompt}{schema_suffix}"})

            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                reasoning_effort=reasoning_effort,
            )
            if hasattr(inputs, "to"):
                inputs = inputs.to(target_device)
            elif isinstance(inputs, dict):
                inputs = {
                    k: (v.to(target_device) if hasattr(v, "to") else v)
                    for k, v in inputs.items()
                }
            else:
                raise TypeError(
                    "tokenizer.apply_chat_template returned an unsupported type. "
                    f"Expected a BatchEncoding-like object or dict, got: {type(inputs)}"
                )
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise RuntimeError(
                "Failed to apply GPT-OSS chat template for Unsloth backend. "
                "Verify you are using an Unsloth GPT-OSS repo and a compatible tokenizer."
            ) from exc

        input_length = int(inputs["input_ids"].shape[1])

        generation_config_params = self.config.get("generation_config", {}).get(
            "params", {}
        )
        final_gen_config = generation_config_params.copy()
        final_gen_config.update(kwargs)

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        final_gen_config.setdefault("pad_token_id", pad_id)

        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **final_gen_config)
        except Exception as exc:  # noqa: BLE001
            record_generation_failure(self.model_name, exc)
            raise

        generated_tokens = outputs[0, input_length:]
        decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
        text, extracted_ok, extraction_exc = self._extract_gpt_oss_final_or_return_raw(
            decoded,
            context="unsloth_single",
            count_failure=True,
        )
        if extracted_ok:
            record_generation_success(self.model_name)
        else:
            assert extraction_exc is not None
            # We intentionally return the raw decoded generation so downstream stages can persist
            # artifacts for debugging, while the global failure tracker still enforces a hard stop
            # on repeated failures.
            logger.exception(
                "GPT-OSS final-channel extraction failed (context=%s). Returning raw decoded output. "
                "model=%s decoded_prefix=%r decoded_suffix=%r",
                "unsloth_single",
                self.model_name,
                decoded[:250].replace("\n", "\\n"),
                decoded[-250:].replace("\n", "\\n"),
                exc_info=extraction_exc,
            )
        return text

    @staticmethod
    def _extract_gpt_oss_final_from_decoded(decoded: str) -> str:
        """
        Extract the GPT-OSS final-channel payload from a decoded generation.

        GPT-OSS Unsloth generations include explicit channel markers. We only want the
        content between:

          <|start|>assistant<|channel|>final<|message|>
          ...
          <|return|>

        Args:
            decoded (str): Decoded generation string with special tokens preserved
                (i.e., decoded with skip_special_tokens=False).

        Returns:
            str: The extracted final-channel message content.

        Raises:
            ValueError: If required markers are missing or malformed.
        """
        if decoded is None:
            raise ValueError("Decoded output is None.")

        final_start = "<|start|>assistant<|channel|>final<|message|>"
        end_markers = ("<|return|>", "<|end|>")

        start_idx = decoded.find(final_start)
        if start_idx == -1:
            prefix = decoded[:250].replace("\n", "\\n")
            raise ValueError(
                "Failed to locate GPT-OSS final channel start marker in decoded output. "
                f"Expected marker='{final_start}'. Decoded prefix='{prefix}'"
            )

        content_start = start_idx + len(final_start)
        end_idx = -1
        for marker in end_markers:
            end_idx = decoded.find(marker, content_start)
            if end_idx != -1:
                break

        if end_idx == -1:
            snippet = decoded[content_start : content_start + 250].replace("\n", "\\n")
            end_snippet = decoded[-250:].replace("\n", "\\n")
            raise ValueError(
                "Failed to locate GPT-OSS final channel end marker in decoded output. "
                f"Expected one of {end_markers}.\n\nFinal-content prefix='{snippet}' \n\nFinal-content suffix='{end_snippet}'"
            )

        return decoded[content_start:end_idx]

    def _extract_gpt_oss_final_or_return_raw(
        self,
        decoded: str,
        *,
        context: str,
        count_failure: bool,
    ) -> tuple[str, bool, Optional[BaseException]]:
        """
        Extract the GPT-OSS final-channel content, falling back to raw decoded text on failure.

        This helper exists because some evaluation loops intentionally catch broad exceptions and
        continue. If we raise on parsing/extraction failures, we may prevent downstream stages from
        writing artifacts needed for debugging. Instead, we:
        - Return the raw decoded output on extraction failure.
        - Record the failure in the global failure tracker (optionally) so repeated failures still
          abort the run deterministically.

        Args:
            decoded (str): Decoded generation string with special tokens preserved.
            context (str): Short context label for logging/debugging (e.g., "unsloth_single").
            count_failure (bool): Whether to record this failure in the global tracker.

        Returns:
            tuple[str, bool, Optional[BaseException]]:
                - text: Extracted final text if successful, otherwise the raw decoded output.
                - extracted_ok: True if extraction succeeded, False otherwise.
                - exc: The exception on failure (best-effort), else None.
        """
        try:
            return self._extract_gpt_oss_final_from_decoded(decoded).strip(), True, None
        except Exception as exc:  # noqa: BLE001
            if count_failure:
                record_generation_failure(self.model_name, exc)
            # Return raw decoded so callers can still persist outputs for debugging.
            return (decoded or "").strip(), False, exc

    def _trim_batch_after_return(
        self,
        outputs: torch.Tensor,
        input_lengths: list[int],
        return_id: int,
        eot_id: int,
    ) -> List[torch.Tensor]:  # [B, T]
        """
        Returns a list of 1D tensors, one per batch item,
        trimmed after the first <|return|>, keeping the first eos and dropping any trailing tokens.
        """

        B, T = outputs.shape
        trimmed: List[torch.Tensor] = []

        for i in range(B):
            # Only look at generated portion
            gen_tokens = outputs[i]
            gen_start = input_lengths[i]
            gen_segment = gen_tokens[gen_start:]

            # Find first <|return|>
            return_positions = (gen_segment == return_id).nonzero(as_tuple=True)[0]
            if len(return_positions) == 0:
                # No return token; keep generated segment as-is.
                trimmed.append(gen_segment)
                continue

            first_return = return_positions[0].item()
            after_return = gen_segment[first_return:]

            # Find first eos after return
            eos_positions = (after_return == eot_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) == 0:
                # No eos after return; keep from return onward.
                kept = gen_segment[:]
            else:
                first_eos = eos_positions[0].item()
                end_idx = first_return + first_eos  # position of eos
                # Exclude the eos itself to avoid parser header errors.
                kept = gen_segment[:end_idx]

            trimmed.append(kept)

        return trimmed

    def _extract_and_sanitize_completion_tokens(
        self,
        sequences: torch.Tensor,  # [B, T_out]
        prompt_len: int,  # max_len (padded prompt length)
        stop_ids: set[int],  # {return_id, call_id}
        pad_id: int,
    ) -> List[List[int]]:
        """
        Extract per-row completion tokens and remove batch padding / eos noise so Harmony can parse reliably.
        """
        assert self.tokenizer is not None

        eos_id = self.tokenizer.eos_token_id
        results: List[List[int]] = []

        B, _ = sequences.shape
        for i in range(B):
            # Generated portion plus batch padding
            toks = sequences[i, prompt_len:].tolist()

            # Strip trailing padding (most of the "many end_of_text" tokens come from here)
            while toks and toks[-1] == pad_id:
                toks.pop()

            # If eos is distinct from pad, strip trailing eos noise too
            if eos_id is not None and eos_id != pad_id:
                while toks and toks[-1] == eos_id:
                    toks.pop()

            # Truncate at the first real Harmony stop token (<|return|> or <|call|>)
            for j, t in enumerate(toks):
                if t in stop_ids:
                    toks = toks[: j + 1]
                    break

            # Final cleanup: strip any trailing eos/pad that is not an actual stop token
            while toks and toks[-1] in {pad_id, eos_id} and toks[-1] not in stop_ids:
                toks.pop()

            results.append(toks)

        return results

    def generate_batch(
        self, requests: List[GenerationRequest], **kwargs: Any
    ) -> List[str]:
        """
        Batched generation for gpt-oss. Falls back to per-call when schemas differ.
        """
        if not requests:
            return []
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded.")

        if self.backend == "unsloth":
            return self._generate_batch_unsloth(requests=requests, **kwargs)

        if self.harmony_encoding is None:
            raise ValueError("Harmony encoding must be loaded.")

        # If per-request kwargs differ, use sequential fallback to respect overrides.
        first_kwargs = requests[0].generation_kwargs
        if not all(req.generation_kwargs == first_kwargs for req in requests):
            return super().generate_batch(requests, **kwargs)

        # Drop seed hint
        merged_gen_kwargs = {**first_kwargs, **kwargs}
        merged_gen_kwargs.pop("seed", None)

        # Render conversations to token IDs
        token_seqs: List[torch.Tensor] = []
        for req in requests:
            prompt_params = self.config.get("prompt", {}).get("params", {})
            system_message = prompt_params.get("system_message", "")
            developer_message = prompt_params.get("developer_message", "")
            if req.json_schema:
                schema_string = json.dumps(req.json_schema)
                schema_name = req.json_schema.get("title", "schema")
                developer_message += f"\n\n# Response Formats\n\n## {schema_name}\n\n{schema_string}<|end|>"

            conversation = Conversation.from_messages(
                [
                    Message.from_role_and_content(Role.SYSTEM, system_message),
                    Message.from_role_and_content(Role.DEVELOPER, developer_message),
                    Message.from_role_and_content(Role.USER, req.prompt),
                ]
            )
            tokens = self.harmony_encoding.render_conversation_for_completion(
                conversation, Role.ASSISTANT
            )
            tensor_ids = torch.tensor(
                tokens, device=self.model.device, dtype=torch.long
            )
            token_seqs.append(tensor_ids)
        # Left-pad to batch
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        max_len = max(seq.shape[0] for seq in token_seqs)
        padded_inputs = torch.full(
            (len(token_seqs), max_len),
            pad_id,
            device=self.model.device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(padded_inputs)
        for i, seq in enumerate(token_seqs):
            padded_inputs[i, -seq.shape[0] :] = seq
            attention_mask[i, -seq.shape[0] :] = 1
        # Generation starts after the padded prompt length
        input_lengths: List[int] = [max_len for _ in token_seqs]

        # Generation config
        generation_config_params = self.config.get("generation_config", {}).get(
            "params", {}
        )
        final_gen_config = generation_config_params.copy()
        final_gen_config.update(merged_gen_kwargs)

        # # EOS handling
        # stop_token_ids = self.tokenizer.convert_tokens_to_ids(
        #     ["<|return|>", "<|call|>"]
        # )
        # if "eos_token_id" not in final_gen_config:
        #     final_gen_config["eos_token_id"] = stop_token_ids
        # elif isinstance(final_gen_config["eos_token_id"], int):
        #     final_gen_config["eos_token_id"] = [
        #         final_gen_config["eos_token_id"]
        #     ] + stop_token_ids
        # else:
        #     final_gen_config["eos_token_id"].extend(stop_token_ids)
        #     final_gen_config["eos_token_id"] = list(
        #         set(final_gen_config["eos_token_id"])
        #     )

        # EOS handling (Harmony-aware) + pad_token_id for stable batch padding behavior
        stop_token_ids = self.tokenizer.convert_tokens_to_ids(
            ["<|return|>", "<|call|>"]
        )
        if any(x is None for x in stop_token_ids):
            raise ValueError(
                f"Failed to map Harmony stop tokens to ids: {stop_token_ids}"
            )

        return_id, call_id = stop_token_ids
        stop_ids = {return_id, call_id}

        # Ensure HF generate pads consistently (batch outputs are padded to max length)
        final_gen_config["pad_token_id"] = pad_id

        # Prefer Harmony stop tokens as EOS so generation stops cleanly
        final_gen_config["eos_token_id"] = [return_id, call_id]

        outputs = None
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=padded_inputs,
                    attention_mask=attention_mask,
                    **final_gen_config,
                )
            except Exception as e:
                logger.warning(
                    f"Batch generation failed, falling back to per-call generation.",
                )
                logger.debug(f"Batch generation error message: {e}")
                torch.cuda.empty_cache()
                # outputs = self.model.generate(
                #     input_ids=padded_inputs,
                #     attention_mask=attention_mask,
                #     **final_gen_config,
                # )
                return super().generate_batch(requests, **kwargs)

        # results: List[str] = []
        # trimmed_batch = self._trim_batch_after_return(
        #     outputs, input_lengths, stop_token_ids[0], self.tokenizer.eos_token_id
        # )
        # for row_idx, output_row in enumerate(trimmed_batch):
        #     tokens = output_row.tolist()
        #     # Drop leading pad/eos noise
        #     while tokens and tokens[0] in {pad_id, self.tokenizer.eos_token_id}:
        #         tokens.pop(0)
        #     # Keep the first eos as a delimiter; drop everything after it
        #     if self.tokenizer.eos_token_id in tokens:
        #         eos_pos = tokens.index(self.tokenizer.eos_token_id)
        #         tokens = tokens[: eos_pos + 1]
        #     if not tokens:
        #         logger.error(
        #             "Empty token sequence after cleanup for batch item %s", row_idx
        #         )
        #         results.append("{}")
        #         continue

        #     parsed_messages = (
        #         self.harmony_encoding.parse_messages_from_completion_tokens(
        #             tokens, Role.ASSISTANT
        #         )
        #     )
        #     final_response = None
        #     for msg in parsed_messages:
        #         if msg.channel == "final":
        #             final_response = msg.content
        #             break
        #     if final_response is None:
        #         logger.error("No final response found for batch item %s", row_idx)
        #         results.append("{}")
        #     else:
        #         results.append(final_response[0].text.strip())
        # return results

        # Extract and sanitize per-row completion tokens (removes batch padding eos noise)
        completion_rows = self._extract_and_sanitize_completion_tokens(
            sequences=outputs,
            prompt_len=max_len,
            stop_ids=stop_ids,
            pad_id=pad_id,
        )

        results: List[str] = []
        for row_idx, tokens in enumerate(completion_rows):
            # Drop any leading pad noise (rare, but safe)
            while tokens and tokens[0] == pad_id:
                tokens.pop(0)

            if not tokens:
                logger.error(
                    "Empty token sequence after sanitization for batch item %s", row_idx
                )
                results.append("FAILED TO PARSE GENERATED TOKENS")
                continue

            try:
                parsed_messages = (
                    self.harmony_encoding.parse_messages_from_completion_tokens(
                        tokens, Role.ASSISTANT
                    )
                )
            except Exception:
                logger.error(
                    "Harmony parsing failed for batch item %s. Falling back to per-call for this item.",
                    row_idx,
                    exc_info=True,
                )
                # Per-item fallback, keeps batch mostly fast but robust
                results.append(
                    self.generate(
                        prompt=requests[row_idx].prompt,
                        json_schema=requests[row_idx].json_schema,
                        **merged_gen_kwargs,
                    )
                )
                continue

            final_response = None
            for msg in parsed_messages:
                if msg.channel == "final":
                    final_response = msg.content
                    break

            if final_response is None:
                logger.error("No final response found for batch item %s", row_idx)
                results.append("FAILED TO PARSE GENERATED TOKENS")
            else:
                results.append(final_response[0].text.strip())

        return results

    def _generate_batch_unsloth(
        self, requests: List[GenerationRequest], **kwargs: Any
    ) -> List[str]:
        """
        Batched generation for the Unsloth backend.

        This follows the same batching approach used by other decoder-only local models:
        left-pad input IDs, generate in one call, and slice by prompt length.
        """
        assert self.model is not None
        assert self.tokenizer is not None

        # If per-request kwargs differ, use sequential fallback to respect overrides.
        first_kwargs = requests[0].generation_kwargs
        if not all(req.generation_kwargs == first_kwargs for req in requests):
            return super().generate_batch(requests, **kwargs)

        merged_gen_kwargs = {**first_kwargs, **kwargs}
        merged_gen_kwargs.pop("seed", None)

        prompt_params = self.config.get("prompt", {}).get("params", {})
        system_message = prompt_params.get("system_message", "")
        developer_message = prompt_params.get("developer_message", "")
        reasoning_effort = prompt_params.get("reasoning_effort", "low")
        combined_prefix = "\n\n".join(
            [part for part in [system_message, developer_message] if part]
        )

        def _cuda_is_usable() -> bool:
            """
            Guard CUDA moves in environments without actual GPUs.

            See `_generate_unsloth` for rationale.
            """
            try:
                return (
                    bool(torch.cuda.is_available())
                    and int(torch.cuda.device_count()) > 0
                )
            except Exception:  # noqa: BLE001
                return False

        # Prefer the model's device when usable; fall back to CPU for safety.
        target_device = self.model.device
        if str(target_device).startswith("cuda") and not _cuda_is_usable():
            target_device = torch.device("cpu")

        token_seqs: List[torch.Tensor] = []
        mask_seqs: List[torch.Tensor] = []
        input_lengths: List[int] = []
        for req in requests:
            schema_suffix = ""
            if req.json_schema:
                schema_suffix = (
                    "\n\n"
                    "Please provide your response in a single, valid JSON format that adheres to the following schema.\n"
                    "Do not include any text before or after the JSON.\n"
                    f"{json.dumps(req.json_schema, indent=2)}"
                )

            full_prompt = (
                f"{combined_prefix}\n\n{req.prompt}{schema_suffix}"
                if combined_prefix
                else f"{req.prompt}{schema_suffix}"
            )
            try:
                messages: List[Dict[str, str]] = []
                if combined_prefix.strip():
                    messages.append(
                        {"role": "system", "content": combined_prefix.strip()}
                    )
                messages.append(
                    {"role": "user", "content": f"{req.prompt}{schema_suffix}"}
                )

                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                    reasoning_effort=reasoning_effort,
                )
            except Exception as exc:  # noqa: BLE001
                record_generation_failure(self.model_name, exc)
                raise RuntimeError(
                    "Failed to apply GPT-OSS chat template during Unsloth batched generation. "
                    "Verify you are using an Unsloth GPT-OSS repo and a compatible tokenizer."
                ) from exc

            input_ids = inputs["input_ids"][0].to(target_device)
            attention_mask = inputs["attention_mask"][0].to(target_device)
            token_seqs.append(input_ids)
            mask_seqs.append(attention_mask)
            input_lengths.append(int(input_ids.shape[0]))

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        if pad_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id set.")

        max_len = max(seq.shape[0] for seq in token_seqs)
        padded_inputs = torch.full(
            (len(token_seqs), max_len),
            pad_id,
            device=target_device,
            dtype=torch.long,
        )
        attention_mask = torch.zeros_like(padded_inputs)
        for i, seq in enumerate(token_seqs):
            padded_inputs[i, -seq.shape[0] :] = seq
            attention_mask[i, -seq.shape[0] :] = mask_seqs[i]

        generation_config_params = self.config.get("generation_config", {}).get(
            "params", {}
        )
        final_gen_config = generation_config_params.copy()
        final_gen_config.update(merged_gen_kwargs)
        final_gen_config.setdefault("pad_token_id", pad_id)

        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids=padded_inputs,
                    attention_mask=attention_mask,
                    **final_gen_config,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Error generating for input shape: {padded_inputs.shape}")
                logger.error(
                    f"Model quantization method: {self.model.quantization_method}"
                )
                record_generation_failure(self.model_name, exc)
                raise

        results: List[str] = []
        any_extraction_failed = False
        first_extraction_exc: Optional[BaseException] = None
        for row_idx, output_row in enumerate(outputs):
            prompt_len = input_lengths[row_idx]
            generated_tokens = output_row[prompt_len:]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
            item_context = f"unsloth_batch_item_{row_idx}"
            text, extracted_ok, extraction_exc = (
                self._extract_gpt_oss_final_or_return_raw(
                    decoded,
                    context=item_context,
                    # Record at most once per batch call (see below).
                    count_failure=False,
                )
            )
            results.append(text)
            if not extracted_ok:
                any_extraction_failed = True
                if first_extraction_exc is None:
                    first_extraction_exc = extraction_exc
                assert extraction_exc is not None
                logger.exception(
                    "GPT-OSS final-channel extraction failed for batch item %s (context=%s). Returning raw decoded output. "
                    "model=%s decoded_prefix=%r decoded_suffix=%r",
                    row_idx,
                    item_context,
                    self.model_name,
                    decoded[:250].replace("\n", "\\n"),
                    decoded[-250:].replace("\n", "\\n"),
                    exc_info=extraction_exc,
                )

        if any_extraction_failed:
            # Count this as a generation failure (once per batch call) so repeated systemic failures
            # still abort the run, but we also return raw decoded outputs for debugging/persistence.
            record_generation_failure(
                self.model_name,
                (
                    first_extraction_exc
                    if first_extraction_exc is not None
                    else RuntimeError(
                        "GPT-OSS final-channel extraction failed in batch (unknown error)."
                    )
                ),
            )
        else:
            record_generation_success(self.model_name)
        return results

    def cleanup(self) -> None:
        """Explicitly releases model and tokenizer from memory."""
        logger.info("Cleaning up gpt-oss model resources...")
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.harmony_encoding is not None:
            del self.harmony_encoding
            self.harmony_encoding = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("gpt-oss model resources cleaned up successfully.")

    def __del__(self):
        """Destructor to ensure cleanup on garbage collection."""
        self.cleanup()
