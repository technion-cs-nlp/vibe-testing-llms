"""Logic for personalizing tasks to create a vibe-testing dataset."""

import itertools
import json
import logging
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import os

import tqdm

from .data_utils import (
    UserProfile,
    BenchmarkSample,
    VibeTask,
    ChangeIdentificationOutput,
    ChangeOption,
    PersonalizedSample,
    PromptVariation,
    VerificationOutput,
    FieldChanges,
)
from .models.base import BaseModel, GenerationRequest
from .utils import parse_and_validate_json, load_json, format_test_cases_for_prompt

logger = logging.getLogger(__name__)


class TaskPersonalizer:
    """
    Adapts benchmark samples into personalized VibeTasks based on a user profile
    by using an LLM to generate and verify prompt variations.
    """

    def __init__(
        self,
        model: BaseModel,
        generation_kwargs: Dict[str, Any] = None,
        batch_size: int = 1,
        artifacts_dir: Optional[str] = None,
    ):
        """
        Initializes the TaskPersonalizer.

        Args:
            model: An instance of a language model for generation tasks.
            generation_kwargs: Default generation parameters for the model.
            artifacts_dir: Optional directory for saving metadata artifacts produced during
                personalization (e.g., identified change options). This directory is
                intentionally separate from dataset sample JSONs to avoid collisions with
                downstream globbing logic.
        """
        self._model = model
        self._generation_kwargs = generation_kwargs or {}
        self._batch_size = max(1, int(batch_size))
        self._artifacts_dir = (
            Path(artifacts_dir).expanduser() if artifacts_dir else None
        )

        # Load prompts from the model's configuration
        prompt_config = self._model.config.get("personalization_prompts", {})
        self._identify_changes_template = prompt_config.get("identify_changes")
        self._compose_prompt_template = prompt_config.get("compose_prompt")
        self._compose_prompt_template_humaneval_plus = prompt_config.get(
            "compose_prompt_humaneval_plus"
        )
        self._verify_prompt_template = prompt_config.get("verify_prompt")

        if not all(
            [
                self._identify_changes_template,
                self._compose_prompt_template,
                self._verify_prompt_template,
            ]
        ):
            raise ValueError(
                "Personalization prompts not fully configured in model config."
            )

        # Load schemas
        schema_dir = os.path.join(os.path.dirname(__file__), "schemas")
        self._identify_changes_schema = load_json(
            os.path.join(schema_dir, "identify_changes_schema.json")
        )
        self._compose_prompt_schema = load_json(
            os.path.join(schema_dir, "compose_prompt_schema.json")
        )
        self._verify_prompt_schema = load_json(
            os.path.join(schema_dir, "verify_prompt_schema.json")
        )

        logger.info(f"Initialized TaskPersonalizer with model: {model}")

    @staticmethod
    def _sanitize_artifact_token(value: str) -> str:
        """
        Sanitize user-provided tokens for safe filenames.

        Args:
            value: Raw token (e.g., user_id).

        Returns:
            A filesystem-safe token.
        """
        cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", (value or "").strip()).strip("-")
        return cleaned or "unknown"

    @staticmethod
    def _next_versioned_path(parent_dir: Path, base_name: str, ext: str) -> Path:
        """
        Resolve a next available versioned filename inside a directory.

        Args:
            parent_dir: Directory to place the artifact.
            base_name: Base filename without version/ext.
            ext: File extension without leading dot.

        Returns:
            Path: A versioned path like ``{base_name}_v00.{ext}``.

        Raises:
            RuntimeError: If no free version slot is available.
        """
        for version in range(100):
            candidate = parent_dir / f"{base_name}_v{version:02d}.{ext}"
            if not candidate.exists():
                return candidate
        raise RuntimeError(
            f"Unable to allocate a versioned artifact filename for '{base_name}' in {parent_dir}."
        )

    def _persist_identified_changes(
        self,
        *,
        profile: UserProfile,
        parsed_json: Dict[str, Any],
        raw_response: str,
    ) -> None:
        """
        Persist the identified changes output for later reference.

        This intentionally writes into a dedicated metadata directory so Stage 4 / experiment
        scanners that glob ``dataset_dir/*.json`` do not treat this file as a dataset sample.

        Args:
            profile: User profile used for identifying changes.
            parsed_json: Parsed JSON payload (validated against schema).
            raw_response: Raw model output string used to produce ``parsed_json``.

        Raises:
            RuntimeError: If the artifact cannot be written.
        """
        if self._artifacts_dir is None:
            return

        try:
            self._artifacts_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to create personalization artifacts directory: {self._artifacts_dir}"
            ) from exc

        safe_user_id = self._sanitize_artifact_token(profile.user_id or "unknown_user")
        base_name = f"identify_changes_user-{safe_user_id}"
        output_path = self._next_versioned_path(self._artifacts_dir, base_name, "json")

        payload = {
            "user_id": profile.user_id,
            "model_name": getattr(self._model, "model_name", None),
            "generation_kwargs": dict(self._generation_kwargs),
            "identify_changes": parsed_json,
            "raw_response": raw_response,
        }

        tmp_path = output_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, output_path)
        except Exception as exc:  # noqa: BLE001
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"Failed to write identified changes artifact to {output_path}"
            ) from exc

        logger.info("Saved identified change options to %s", output_path)

    def _select_compose_prompt_template(self, sample: BenchmarkSample) -> str:
        """
        Select the compose prompt template based on the benchmark.

        HumanEval+ personalization must be a prefix-only rewrite, which is later
        concatenated with the original prompt. This is controlled by a dedicated
        prompt template key: `compose_prompt_humaneval_plus`.
        """
        if sample.source_benchmark == "humaneval_plus":
            if not self._compose_prompt_template_humaneval_plus:
                raise ValueError(
                    "Model config missing personalization_prompts.compose_prompt_humaneval_plus "
                    "required for humaneval_plus."
                )
            return self._compose_prompt_template_humaneval_plus
        return self._compose_prompt_template

    @staticmethod
    def _finalize_modified_prompt(sample: BenchmarkSample, modified_prompt: str) -> str:
        """
        Finalize the stored modified prompt text for a benchmark.

        For HumanEval+, we store the final personalized prompt as:
            f\"{prefix}\\n{original_prompt}\"
        where `prefix` is what the personalization model generated.
        """
        if sample.source_benchmark != "humaneval_plus":
            return modified_prompt

        prefix = (modified_prompt or "").strip()
        original = (sample.prompt or "").strip()
        if not prefix:
            return original
        if not original:
            return prefix
        return f"{prefix}\n{original}"

    def _identify_change_options(
        self, profile: UserProfile
    ) -> ChangeIdentificationOutput:
        """Subtask 1: Identify possible changes based on the user profile."""
        logger.info(f"Identifying change options for profile: {profile.user_id}")

        prompt = self._identify_changes_template.format(
            user_profile_json=profile.model_dump_json(indent=2)
        )

        response_str = self._model.generate(
            prompt, json_schema=self._identify_changes_schema, **self._generation_kwargs
        )
        logger.debug(f"Raw response for change identification:\n{response_str}")

        parsed_json = parse_and_validate_json(
            response_str, schema=self._identify_changes_schema
        )
        self._persist_identified_changes(
            profile=profile,
            parsed_json=parsed_json,
            raw_response=response_str,
        )
        return ChangeIdentificationOutput(**parsed_json)

    def _compose_modified_prompts_batch(self, prompts: List[str]) -> List[str]:
        """
        Subtask 2: Compose modified prompts in batch.

        Args:
            prompts: Rendered prompt strings to send to the model.

        Returns:
            List[str]: Model outputs aligned with the input order.
        """
        requests = [
            GenerationRequest(
                prompt=p,
                json_schema=self._compose_prompt_schema,
                generation_kwargs=self._generation_kwargs,
            )
            for p in prompts
        ]
        raw_outputs = self._model.generate_batch(requests)
        parsed_outputs: List[str] = []
        for output in raw_outputs:
            parsed_outputs.append(
                self._unwrap_composed_output(output, self._compose_prompt_schema)
            )
        return parsed_outputs

    @staticmethod
    def _unwrap_composed_output(payload: Any, compose_schema: Dict[str, Any]) -> str:
        """
        Normalize the composed prompt output to a plain string.

        Args:
            payload: Raw model response for the composed prompt.
            compose_schema: JSON schema used for validation.

        Returns:
            The extracted modified prompt text, or the raw payload on failure.
        """
        if payload is None:
            return ""

        if isinstance(payload, dict):
            return str(payload.get("modified_prompt", ""))

        if isinstance(payload, str):
            stripped = payload.strip()
            if stripped.startswith("{"):
                try:
                    parsed_json = parse_and_validate_json(
                        stripped, schema=compose_schema
                    )
                    return str(parsed_json.get("modified_prompt", stripped))
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Failed to parse composed prompt JSON; returning raw. Error: %s",
                        exc,
                    )
            return stripped

        return str(payload)

    def _verify_variations_batch(
        self, original_prompts: List[str], modified_prompts: List[str]
    ) -> List[VerificationOutput]:
        """
        Subtask 3: Verify multiple prompt variations in batch.
        """
        prompts = []
        for original_prompt, modified_prompt in zip(original_prompts, modified_prompts):
            prompts.append(
                self._verify_prompt_template.format(
                    original_prompt=original_prompt,
                    modified_prompt=modified_prompt,
                )
            )

        requests = [
            GenerationRequest(
                prompt=p,
                json_schema=self._verify_prompt_schema,
                generation_kwargs=self._generation_kwargs,
            )
            for p in prompts
        ]
        responses = self._model.generate_batch(requests)

        verified: List[VerificationOutput] = []
        for response_str in responses:
            parsed_json = parse_and_validate_json(
                response_str, schema=self._verify_prompt_schema
            )
            verified.append(VerificationOutput(**parsed_json))
        return verified

    def _verify_variation(
        self, original_prompt: str, modified_prompt: str
    ) -> VerificationOutput:
        """Subtask 3: Verify the modified prompt preserves the original's intent."""
        logger.info("Verifying prompt variation.")
        prompt = self._verify_prompt_template.format(
            original_prompt=original_prompt,
            modified_prompt=modified_prompt,
        )

        response_str = self._model.generate(
            prompt, json_schema=self._verify_prompt_schema, **self._generation_kwargs
        )
        logger.debug(f"Raw response for verification:\n{response_str}")

        parsed_json = parse_and_validate_json(
            response_str, schema=self._verify_prompt_schema
        )
        return VerificationOutput(**parsed_json)

    def build_dataset(
        self,
        samples: List[BenchmarkSample],
        profile: UserProfile,
        num_variations_per_sample: int,
    ) -> List[PersonalizedSample]:
        """
        Builds a full vibe-testing dataset for a given user profile.

        This orchestrates the three subtasks:
        1. Identifies all possible change options from the profile.
        2. For each sample, creates N unique compositions of changes.
        3. For each composition, generates and verifies a new prompt.

        Args:
            samples: A list of selected benchmark samples.
            profile: The user profile.
            num_variations_per_sample: The target number of variations for each sample.

        Returns:
            A list of PersonalizedSample objects, each containing the original
            sample and its generated variations.
        """
        change_options_output = self._identify_change_options(profile)
        options_by_field = change_options_output.changes_by_field

        personalized_samples: List[PersonalizedSample] = []
        # Go over each sample in the dataset with tqdm
        for sample in tqdm.tqdm(samples, desc="Processing samples", total=len(samples)):
            sample_id = sample.sample_id
            logger.info(f"--- Processing sample: {sample_id} ---")
            variations: List[PromptVariation] = []

            # Generate unique combinations of changes
            # This is a simple strategy: pick one change from each field category
            change_lists = [fc.options for fc in options_by_field]
            all_possible_combos = list(itertools.product(*change_lists))

            # If there are fewer possible combos than requested, just use all of them
            num_to_generate = min(num_variations_per_sample, len(all_possible_combos))

            # Randomly sample unique combinations
            selected_combos = random.sample(all_possible_combos, num_to_generate)

            # Prebuild prompts for composition
            compose_payloads = []
            test_list = sample.metadata.get("prompt_tests", {})
            plus_test_list = sample.metadata.get("tests", {}).get(
                "test", ""
            )  # this are the plus tests
            test_list = test_list  # + [plus_test_list]
            # prompt_with_tests = sample.prompt + format_test_cases_for_prompt(test_list)
            compose_template = self._select_compose_prompt_template(sample)

            for i, combo in enumerate(selected_combos):
                variation_id = f"{sample_id}-{profile.user_id}-var{i+1}"
                applied_change_options = list(combo)
                applied_change_names = [opt.name for opt in applied_change_options]
                changes_description = "\n".join(
                    f"- {c.name}: {c.change} (e.g., '{c.example}')"
                    for c in applied_change_options
                )
                # if compose_template has {evaluation_tests}, add it to the render_prompt
                if "{evaluation_tests}" in compose_template:
                    render_prompt = compose_template.format(
                        original_prompt=sample.prompt,  # prompt without tests
                        changes_description=changes_description,
                        evaluation_tests=format_test_cases_for_prompt(test_list),
                    )
                else:
                    render_prompt = compose_template.format(
                        original_prompt=sample.prompt,  # prompt without tests,
                        changes_description=changes_description,
                    )

                compose_payloads.append(
                    {
                        "variation_id": variation_id,
                        "applied_changes": applied_change_names,
                        "render_prompt": render_prompt,
                    }
                )

            # Compose in batches
            composed_outputs: List[str] = []
            for start in range(0, len(compose_payloads), self._batch_size):
                chunk = compose_payloads[start : start + self._batch_size]
                prompts = [item["render_prompt"] for item in chunk]
                composed_outputs.extend(self._compose_modified_prompts_batch(prompts))

            # Verify in batches, aligned to compose_payloads order
            verification_inputs = []
            for payload, modified_prompt in zip(compose_payloads, composed_outputs):
                finalized_prompt = self._finalize_modified_prompt(
                    sample, modified_prompt
                )
                verification_inputs.append(
                    {
                        "variation_id": payload["variation_id"],
                        "applied_changes": payload["applied_changes"],
                        "modified_prompt": finalized_prompt,
                    }
                )

            verifications: List[VerificationOutput] = []
            for start in range(0, len(verification_inputs), self._batch_size):
                chunk = verification_inputs[start : start + self._batch_size]
                original_prompts = [sample.prompt for _ in chunk]
                modified_prompts = [item["modified_prompt"] for item in chunk]
                verifications.extend(
                    self._verify_variations_batch(original_prompts, modified_prompts)
                )

            for payload, verification in zip(verification_inputs, verifications):
                variation = PromptVariation(
                    variation_id=payload["variation_id"],
                    applied_changes=payload["applied_changes"],
                    modified_prompt=payload["modified_prompt"],
                    verification=verification,
                )
                variations.append(variation)
                logger.info(
                    f"Successfully created and verified variation {payload['variation_id']}"
                )

            personalized_samples.append(
                PersonalizedSample(original_sample=sample, variations=variations)
            )
            logger.info(
                f"Finished processing sample {sample_id}, created {len(variations)} variations."
            )

        return personalized_samples
