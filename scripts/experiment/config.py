"""
Experiment configuration loading and validation.

This module handles loading experiment YAML configs with include resolution,
providing access to persona, model, and judge configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from src.vibe_testing.pairwise_judgment_types import (
    PAIRWISE_JUDGMENT_TYPE_PERSONA,
    normalize_pairwise_judgment_type,
)

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when experiment configuration is invalid."""

    pass


@dataclass
class ExperimentConfig:
    """
    Parsed and resolved experiment configuration.

    Attributes:
        name: Human-readable experiment name.
        base_dir: Base directory for all experiment outputs.
        defaults: Default settings (benchmarks, num_samples, etc.).
        generator: Name of the model used for dataset generation.
        personas: Dict mapping persona names to their configs.
        models: Dict mapping model names to their configs.
        use_personas: List of persona names to include in experiment.
        use_models: List of model names to evaluate.
        use_judges: List of model names to use as judges.
    """

    name: str
    base_dir: Path
    defaults: Dict[str, Any]
    generator: str
    personas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    use_personas: List[str] = field(default_factory=list)
    use_models: List[str] = field(default_factory=list)
    use_judges: List[str] = field(default_factory=list)
    pairwise_judgment_types: List[str] = field(default_factory=list)
    _config_dir: Path = field(default=Path("."), repr=False)

    @classmethod
    def load(cls, config_path: str | Path) -> "ExperimentConfig":
        """
        Load experiment config from YAML file with include resolution.

        Args:
            config_path: Path to the experiment YAML config file.

        Returns:
            ExperimentConfig: Parsed configuration object.

        Raises:
            ConfigError: If the config file is invalid or missing required fields.
            FileNotFoundError: If the config file doesn't exist.
        """
        config_path = Path(config_path).resolve()
        config_dir = config_path.parent

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info("Loading experiment config from %s", config_path)

        with open(config_path, "r") as f:
            raw = yaml.safe_load(f)

        if not raw:
            raise ConfigError(f"Empty config file: {config_path}")

        # Validate required fields
        required_fields = ["name", "base_dir", "generator"]
        for field_name in required_fields:
            if field_name not in raw:
                raise ConfigError(f"Missing required field: {field_name}")

        # Resolve includes
        personas: Dict[str, Dict[str, Any]] = {}
        models: Dict[str, Dict[str, Any]] = {}

        for include_path in raw.get("include", []):
            full_path = config_dir / include_path
            if not full_path.exists():
                raise ConfigError(f"Include file not found: {full_path}")

            logger.debug("Loading include: %s", full_path)
            with open(full_path, "r") as f:
                included = yaml.safe_load(f)

            if included:
                if "personas" in included:
                    personas.update(included["personas"])
                if "models" in included:
                    models.update(included["models"])

        # Validate generator exists in models
        generator_name = raw["generator"]
        if generator_name not in models:
            raise ConfigError(
                f"Generator model '{generator_name}' not found in included models. "
                f"Available: {list(models.keys())}"
            )

        # Get use_* lists with defaults
        use_personas = raw.get("use_personas", list(personas.keys()))
        use_models = raw.get("use_models", list(models.keys()))
        use_judges = raw.get("use_judges", [generator_name])
        raw_pairwise_judgment_types = raw.get(
            "pairwise_judgment_types", [PAIRWISE_JUDGMENT_TYPE_PERSONA]
        )
        if not isinstance(raw_pairwise_judgment_types, list) or not all(
            isinstance(item, str) for item in raw_pairwise_judgment_types
        ):
            raise ConfigError(
                "pairwise_judgment_types must be a list of strings when provided."
            )
        pairwise_judgment_types: List[str] = []
        for item in raw_pairwise_judgment_types:
            normalized = normalize_pairwise_judgment_type(item)
            if normalized not in pairwise_judgment_types:
                pairwise_judgment_types.append(normalized)

        # Validate all referenced names exist
        for persona in use_personas:
            if persona not in personas:
                raise ConfigError(
                    f"Persona '{persona}' not found. Available: {list(personas.keys())}"
                )

        for model in use_models:
            if model not in models:
                raise ConfigError(
                    f"Model '{model}' not found. Available: {list(models.keys())}"
                )

        for judge in use_judges:
            if judge not in models:
                raise ConfigError(
                    f"Judge model '{judge}' not found. Available: {list(models.keys())}"
                )

        # Parse defaults
        defaults = raw.get("defaults", {})
        if "benchmarks" not in defaults:
            defaults["benchmarks"] = ["mbpp_plus"]
        if "num_samples" not in defaults:
            defaults["num_samples"] = 20
        if "num_variations" not in defaults:
            defaults["num_variations"] = 2
        if "filter_model" not in defaults:
            defaults["filter_model"] = "none"
        if "dataset_type" not in defaults:
            defaults["dataset_type"] = "function"
        # Optional default prompt types for downstream stages. When omitted,
        # stages operate on all prompt types together using the legacy layout.
        prompt_types = defaults.get("prompt_types")
        if prompt_types is not None:
            if not isinstance(prompt_types, list) or not all(
                isinstance(pt, str) for pt in prompt_types
            ):
                raise ConfigError(
                    "defaults.prompt_types, when provided, must be a list of strings."
                )

        return cls(
            name=raw["name"],
            base_dir=Path(raw["base_dir"]),
            defaults=defaults,
            generator=generator_name,
            personas=personas,
            models=models,
            use_personas=use_personas,
            use_models=use_models,
            use_judges=use_judges,
            pairwise_judgment_types=pairwise_judgment_types,
            _config_dir=config_dir,
        )

    def get_persona_config(self, name: str) -> Dict[str, Any]:
        """
        Get the full configuration for a persona.

        Args:
            name: Persona name.

        Returns:
            Dict containing persona config with 'name' field added.

        Raises:
            KeyError: If persona not found.
        """
        if name not in self.personas:
            raise KeyError(f"Persona not found: {name}")
        config = self.personas[name].copy()
        config["name"] = name
        return config

    def get_model_config(self, name: str) -> Dict[str, Any]:
        """
        Get the full configuration for a model.

        Args:
            name: Model name.

        Returns:
            Dict containing model config with 'name' field added.

        Raises:
            KeyError: If model not found.
        """
        if name not in self.models:
            raise KeyError(f"Model not found: {name}")
        config = self.models[name].copy()
        config["name"] = name
        return config

    def get_generator_config(self) -> Dict[str, Any]:
        """Get the generator model configuration."""
        return self.get_model_config(self.generator)

    def get_prompt_types(self, stage: str) -> Optional[List[str]]:
        """
        Get the default prompt types for a given stage, if configured.

        Args:
            stage: Logical stage name (e.g., 'objective', 'subjective', 'pairwise').

        Returns:
            Optional list of prompt-type strings. When None, callers should
            fall back to legacy behavior (all prompt types together).
        """
        # For now we expose a single experiment-wide default list, but this
        # helper centralizes the lookup so we can add per-stage overrides in
        # the future without touching all call sites.
        prompt_types = self.defaults.get("prompt_types")
        if not prompt_types:
            return None
        return list(prompt_types)

    def get_pairwise_judgment_types(self) -> List[str]:
        """
        Get the pairwise judgment types configured for Stage 5b.

        Returns:
            List[str]: Normalized judgment-type tokens.
        """
        if not self.pairwise_judgment_types:
            return [PAIRWISE_JUDGMENT_TYPE_PERSONA]
        return list(self.pairwise_judgment_types)

    def filter_personas_by_tags(self, tags: List[str]) -> List[str]:
        """
        Filter persona names by tags.

        Args:
            tags: List of tags to filter by (OR logic - any tag matches).

        Returns:
            List of persona names that have at least one matching tag.
        """
        return self._filter_by_tags(self.personas, self.use_personas, tags)

    def filter_models_by_tags(self, tags: List[str]) -> List[str]:
        """
        Filter model names by tags.

        Args:
            tags: List of tags to filter by (OR logic - any tag matches).

        Returns:
            List of model names that have at least one matching tag.
        """
        return self._filter_by_tags(self.models, self.use_models, tags)

    def filter_judges_by_tags(self, tags: List[str]) -> List[str]:
        """
        Filter judge model names by tags.

        Args:
            tags: List of tags to filter by (OR logic - any tag matches).

        Returns:
            List of judge model names that have at least one matching tag.
        """
        return self._filter_by_tags(self.models, self.use_judges, tags)

    @staticmethod
    def _filter_by_tags(
        definitions: Dict[str, Dict[str, Any]],
        names: List[str],
        tags: List[str],
    ) -> List[str]:
        """
        Filter names by tags from their definitions.

        Args:
            definitions: Dict mapping names to their configs (with 'tags' field).
            names: List of names to filter.
            tags: Tags to match (OR logic).

        Returns:
            Filtered list of names.
        """
        if not tags:
            return names

        tag_set: Set[str] = set(tags)
        result = []
        for name in names:
            item_tags = set(definitions.get(name, {}).get("tags", []))
            if item_tags & tag_set:  # Intersection - any tag matches
                result.append(name)
        return result

    def get_dataset_id(self, persona: str) -> str:
        """
        Get or generate a dataset ID for a persona.

        The dataset ID is derived from the experiment name to ensure
        consistency across runs.

        Args:
            persona: Persona name.

        Returns:
            Dataset ID string.
        """
        # Use experiment name as base for consistent dataset IDs
        return f"dataset_{self.name}"

    def __repr__(self) -> str:
        return (
            f"ExperimentConfig(name={self.name!r}, "
            f"personas={self.use_personas}, "
            f"models={self.use_models}, "
            f"judges={self.use_judges}, "
            f"pairwise_judgment_types={self.pairwise_judgment_types})"
        )
