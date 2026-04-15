"""Logic for creating a structured user profile from raw text."""

import json
import logging
from typing import Any, Dict

from jsonschema import ValidationError

from .data_utils import UserProfile, UserProfilePersona
from .models.base import BaseModel
from .utils import parse_and_validate_json

logger = logging.getLogger(__name__)


class Profiler:
    """
    Handles the conversion of raw user descriptions into structured
    UserProfile objects.
    """

    def __init__(self, model: BaseModel):
        """
        Initializes the Profiler.

        Args:
            model: An instance of a language model (e.g., APIModel or HFModel)
                   to be used for parsing the user description.
        """
        self._model = model
        logger.info(f"Initialized Profiler with model: {model}")

    def create_profile(
        self, user_config: dict, generation_kwargs: Dict[str, Any] = None
    ) -> UserProfile:
        """
        Creates a structured user profile from a user configuration dictionary
        by using an LLM to interpret the free-text description.

        Args:
            user_config (dict): A dictionary loaded from a user config YAML.
            generation_kwargs (Dict[str, Any], optional): A dictionary of generation
                parameters to override the model's defaults. Defaults to None.

        Returns:
            UserProfile: A Pydantic UserProfile object.
        """
        if generation_kwargs is None:
            generation_kwargs = {}

        user_id = user_config.get("user_id", "unknown_user")
        description = user_config.get("description", "")
        logger.info(f"Creating profile for user_id: {user_id}")

        prompt_template = self._model.config.get("profiling_prompt")
        if not prompt_template:
            logger.error("'profiling_prompt' not found in the model configuration.")
            raise ValueError("profiling_prompt not found in config")

        prompt = prompt_template.format(description=description)
        logger.debug(f"Generated prompt for user profiling:\n{prompt}")

        try:
            response_str = self._model.generate(prompt, **generation_kwargs)
            logger.debug(f"Received raw response from model: {response_str}")

            profile_data = parse_and_validate_json(
                response_str, UserProfilePersona.model_json_schema()
            )
            profile_data["user_id"] = user_id
            profile_data["description"] = description
            return UserProfile(**profile_data)
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            logger.warning(
                "Failed to parse profiling response for user %s. Falling back to user config. Error: %s",
                user_id,
                exc,
            )
            fallback_profile = dict(user_config)
            fallback_profile.setdefault("user_id", user_id)
            fallback_profile.setdefault("description", description)
            return UserProfile(**fallback_profile)
