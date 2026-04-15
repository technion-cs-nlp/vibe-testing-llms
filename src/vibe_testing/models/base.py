"""Abstract base class for all language models."""

import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """
    Encapsulates a single prompt request for batch generation.

    Args:
        prompt (str): The input prompt.
        json_schema (Optional[Dict[str, Any]]): Optional JSON schema for structured output.
        generation_kwargs (Dict[str, Any]): Per-request generation overrides.
    """

    prompt: str
    json_schema: Optional[Dict[str, Any]] = None
    generation_kwargs: Dict[str, Any] = field(default_factory=dict)


class BaseModel(ABC):
    """
    Abstract base class defining the interface for all models in the pipeline.
    """

    def __init__(self, model_name: str, config: Dict[str, Any]):
        """
        Initializes the model.

        Args:
            model_name (str): The name of the model.
            config (Dict[str, Any]): A dictionary containing model-specific
                                     configuration.
        """
        self.model_name = model_name
        self.config = config

        # Configure the global consecutive-generation-failure threshold based on
        # this model's config (default: 100). This ensures consistent behavior
        # even when models are constructed directly (not via load_model_from_config).
        try:
            from src.vibe_testing.models.generation_failure_tracker import (
                configure_failure_threshold_from_model_config,
            )

            configure_failure_threshold_from_model_config(
                config,
                source=model_name,
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                f"Failed to configure generation failure threshold for model '{model_name}': {exc}"
            ) from exc

    @abstractmethod
    def generate(
        self, prompt: str, json_schema: Optional[Dict[str, Any]] = None, **kwargs
    ) -> str:
        """
        Generates a response to a given prompt.

        Args:
            prompt (str): The input prompt for the model.
            json_schema (Optional[Dict[str, Any]]): A JSON schema to enforce on the output.
            **kwargs: Additional generation parameters (e.g., temperature,
                max_new_tokens, seed). Implementations should treat a ``seed``
                argument, when present, as a per-call hint for making the
                generation as deterministic as the backend allows.

        Returns:
            str: The model's generated output.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def generate_batch(
        self, requests: List[GenerationRequest], **kwargs: Any
    ) -> List[str]:
        """
        Generates outputs for multiple prompts.

        This default implementation iterates over `generate` to preserve backward
        compatibility. Model implementations can override for true batching.

        Args:
            requests (List[GenerationRequest]): Batched generation requests.
            **kwargs: Additional parameters applied to every request.

        Returns:
            List[str]: Generated outputs in the same order as `requests`.
        """
        outputs: List[str] = []
        for req in requests:
            merged_kwargs = {**req.generation_kwargs, **kwargs}
            outputs.append(
                self.generate(req.prompt, json_schema=req.json_schema, **merged_kwargs)
            )
        return outputs

    def __repr__(self) -> str:
        """Provides a string representation of the model."""
        return f"{self.__class__.__name__}(model_name='{self.model_name}')"
