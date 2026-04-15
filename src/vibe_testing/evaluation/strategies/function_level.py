"""
Strategy for MBPP+-style function-level evaluation.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.vibe_testing.code_processing.sanitizer import CodeSanitizer
from src.vibe_testing.data_utils import (
    FunctionEvaluationSample,
    FunctionEvaluationTests,
)
from src.vibe_testing.evaluation.strategies.base import (
    EvaluationContext,
    EvaluationStrategy,
    ExecutionResult,
    GenerationResult,
    PreparedInput,
    StrategyResult,
)
from src.vibe_testing.execution import SandboxExecutor, SandboxResult
from src.vibe_testing.reporting.passk import PassKReporter
from src.vibe_testing.utils import save_json, format_test_cases_for_prompt
from src.vibe_testing.pathing import format_artifact_name, normalize_token
from src.vibe_testing.evaluation.test_extractors import TestExtractor
from src.vibe_testing.models.base import GenerationRequest


class FunctionLevelStrategy(EvaluationStrategy):
    """
    Implements Paradigm A (function-level rigor) evaluation workflow.
    """

    def __init__(
        self,
        context: EvaluationContext,
        sanitizer: Optional[CodeSanitizer] = None,
        executor: Optional[SandboxExecutor] = None,
        llm_retry_executor: Optional[SandboxExecutor] = None,
        reporter: Optional[PassKReporter] = None,
        logger: Optional[logging.Logger] = None,
        passk: Sequence[int] = (1,),
    ):
        super().__init__(context, logger=logger)
        self.sanitizer = sanitizer or CodeSanitizer(self.logger)
        self.executor = executor or SandboxExecutor()
        extra = context.extra or {}
        self._enable_llm_code_extraction = bool(
            extra.get("enable_llm_code_extraction", False)
        )
        self._llm_code_extraction_prompt = extra.get("llm_code_extraction_prompt")
        self._llm_code_extraction_generation_kwargs = (
            extra.get("llm_code_extraction_generation_kwargs") or {}
        )
        self._llm_code_extraction_retry_on_timeout = int(
            max(0, extra.get("llm_code_extraction_retry_on_timeout", 1))
        )
        self._llm_code_extraction_retry_timeout = int(
            max(1, extra.get("llm_code_extraction_retry_timeout", 60))
        )
        self._llm_retry_executor = llm_retry_executor or SandboxExecutor(
            timeout=self._llm_code_extraction_retry_timeout
        )
        self._cumulative_failures_counters: Dict[str, int] = {
            # Final-result counters (backward compatible).
            "assertion_errors": 0,
            "execution_errors": 0,
            "timeouts": 0,
            "assertion_errors_plus": 0,
            "execution_errors_plus": 0,
            "timeouts_plus": 0,
            # Primary-run counters (before any LLM extraction).
            "primary_assertion_errors": 0,
            "primary_execution_errors": 0,
            "primary_timeouts": 0,
            "primary_assertion_errors_plus": 0,
            "primary_execution_errors_plus": 0,
            "primary_timeouts_plus": 0,
            # LLM-extraction metadata counters.
            "llm_extraction_attempts": 0,
            "llm_extraction_used_for_final": 0,
            "llm_extraction_generation_errors": 0,
            "llm_extraction_no_code_blocks": 0,
            # LLM retry (last attempt) failure counters.
            "llm_retry_assertion_errors": 0,
            "llm_retry_execution_errors": 0,
            "llm_retry_timeouts": 0,
            "llm_retry_assertion_errors_plus": 0,
            "llm_retry_execution_errors_plus": 0,
            "llm_retry_timeouts_plus": 0,
        }
        run_name = extra.get("run_name", "no_name")
        evaluated_model_label = extra.get("evaluated_model_label")
        model_config_path = extra.get("model_config_path")
        self.reporter = reporter or PassKReporter(
            passk,
            run_name=run_name,
            model_name=evaluated_model_label,
            model_config_path=model_config_path,
        )
        self._run_name = run_name

        self.logger.info(
            "LLM code extraction enabled=%s retry_on_timeout=%s retry_timeout=%ss prompt_template=%s",
            self._enable_llm_code_extraction,
            self._llm_code_extraction_retry_on_timeout,
            self._llm_code_extraction_retry_timeout,
            "provided" if bool(self._llm_code_extraction_prompt) else "default",
        )

    @staticmethod
    def _default_llm_code_extraction_prompt() -> str:
        """
        Default prompt template used for evaluation-only code extraction fallback.

        Returns:
            str: Prompt template string with {language}, {entry_point}, and {raw_output}.
        """
        return (
            "You are given a raw model response to a coding task.\n"
            "Your job is to extract ONLY the final runnable solution code.\n\n"
            "Requirements:\n"
            "- Language: {language}\n"
            "- The solution MUST define the required entry point function: {entry_point}\n"
            "- Output MUST be ONLY a single markdown code block (no prose).\n"
            "- The code block MUST contain only valid {language} code.\n"
            "- Do NOT include tests, examples, or explanations.\n\n"
            "Raw response:\n"
            "-----\n"
            "{raw_output}\n"
            "-----\n\n"
            "Output:\n"
            "```{language}\n"
            "<put only the solution code here>\n"
            "```\n"
        )

    def _should_attempt_llm_code_extraction(
        self, result: Optional[SandboxResult]
    ) -> bool:
        """
        Decide whether to attempt LLM-assisted code extraction.

        We only attempt fallback on execution-type failures (not assertion failures),
        and only when enabled via context.extra.
        """
        if not self._enable_llm_code_extraction:
            return False
        if result is None:
            return False
        return result.status in {"timeout", "error", "stdin_error", "execution_error"}

    def _get_llm_extraction_trigger(
        self,
        base_result: Optional[SandboxResult],
        plus_result: Optional[SandboxResult],
    ) -> Tuple[Optional[SandboxResult], Optional[str]]:
        """
        Decide which (if any) SandboxResult should trigger LLM code extraction.

        We only trigger on execution-type failures (timeout/error/stdin_error),
        not assertion failures.

        NOTE: We intentionally do NOT trigger extraction based on plus-suite
        failures. The plus suite can be substantially harder / broader than base,
        and we want the fallback to focus on "get the basic solution runnable"
        rather than over-fitting to extra tests.
        """
        if base_result is not None and not base_result.passed:
            if self._failure_type_from_status(base_result.status) in {
                "timeout",
                "execution",
            }:
                return base_result, "base"
        return None, None

    @staticmethod
    def _failure_type_from_status(status: Optional[str]) -> Optional[str]:
        """
        Map SandboxResult.status to a canonical failure type.

        Returns:
            Optional[str]: One of {"assertion", "execution", "timeout"} or None.
        """
        if not status:
            return None
        if status == "assertion_error":
            return "assertion"
        if status == "timeout":
            return "timeout"
        if status in {"error", "stdin_error"}:
            return "execution"
        # Backward compatibility: some legacy callers used "execution_error".
        if status == "execution_error":
            return "execution"
        return "execution"

    def _increment_failure_counter(
        self, counters: Dict[str, int], prefix: str, result: Optional[SandboxResult]
    ) -> None:
        """
        Increment failure counters for a given SandboxResult.
        """
        if result is None or result.passed:
            return
        failure_type = self._failure_type_from_status(getattr(result, "status", None))
        if failure_type == "assertion":
            counters[f"{prefix}assertion_errors"] += 1
        elif failure_type == "timeout":
            counters[f"{prefix}timeouts"] += 1
        else:
            counters[f"{prefix}execution_errors"] += 1

    def _run_with_sanitized_blocks(
        self,
        code_blocks: List[str],
        sample: FunctionEvaluationSample,
        is_ds1000: bool,
        executor: Optional[SandboxExecutor] = None,
    ) -> Dict[str, Any]:
        """
        Execute base/plus tests for a list of sanitized code blocks and return a structured record.
        """
        executor_to_use = executor or self.executor
        selected_code_block = None
        base_result = None
        plus_result = None

        if code_blocks:
            for code_block in code_blocks:
                try:
                    if is_ds1000:
                        test_result = self._execute_ds1000(
                            code_block, sample.tests.base_tests
                        )
                    else:
                        test_result = executor_to_use.run(
                            code_block, sample.tests.base_tests
                        )
                    # If the sandbox timed out, do not attempt any additional blocks or re-runs.
                    # This prevents accidental double-execution and keeps retry-on-timeout logic
                    # predictable.
                    if test_result.status == "timeout":
                        selected_code_block = code_block
                        base_result = test_result
                        break
                    if test_result.passed:
                        selected_code_block = code_block
                        base_result = test_result
                        break
                except Exception as exc:
                    self.logger.debug(
                        "Code block failed to execute during selection: %s", exc
                    )
                    continue

            if selected_code_block is None:
                selected_code_block = max(code_blocks, key=len)
                if is_ds1000:
                    base_result = self._execute_ds1000(
                        selected_code_block, sample.tests.base_tests
                    )
                else:
                    base_result = executor_to_use.run(
                        selected_code_block, sample.tests.base_tests
                    )
        else:
            selected_code_block = ""
            base_result = SandboxResult(
                passed=False,
                message="Evaluation failed: no sanitized code blocks produced.",
                stdout="",
                stderr="",
                exception="ValueError('no code blocks')",
                status="error",
            )

        # IMPORTANT: Only run the plus suite if base passed.
        # This avoids wasted work (and noisy logs) when base failed due to syntax/runtime errors.
        if base_result and base_result.passed:
            combined_tests = sample.tests.base_tests + sample.tests.plus_tests
            if sample.tests.plus_tests:
                if is_ds1000:
                    plus_result = self._execute_ds1000(
                        selected_code_block, combined_tests
                    )
                else:
                    plus_result = executor_to_use.run(
                        selected_code_block, combined_tests
                    )
            else:
                plus_result = base_result
        else:
            plus_result = base_result

        return {
            "sanitized_code_blocks": code_blocks,
            "selected_code_block": selected_code_block,
            "selected_block_index": (
                code_blocks.index(selected_code_block)
                if selected_code_block in code_blocks
                else -1
            ),
            "base_result": base_result,
            "plus_result": plus_result,
        }

    def _attempt_llm_code_extraction(
        self,
        problem_prompt: str,
        raw_output: str,
        entry_point: str,
        language: str = "python",
    ) -> Dict[str, Any]:
        """
        Ask the evaluated model to extract runnable solution code from its own raw output.
        This is evaluation-only and does not replace the stored raw output.
        """
        prompt_template = (
            self._llm_code_extraction_prompt
            or self._default_llm_code_extraction_prompt()
        )
        # Support both:
        # - our internal template placeholders (language/entry_point/raw_output), and
        # - external templates (e.g., model config extract_code_prompt) that expect
        #   explicit PROBLEM_PROMPT / RAW_RESPONSE blocks.
        try:
            prompt = prompt_template.format(
                language=language, entry_point=entry_point, raw_output=raw_output
            )
        except Exception:
            prompt = prompt_template
        prompt = (
            f"{prompt.rstrip()}\n\n"
            f"PROBLEM_PROMPT:\n---\n{problem_prompt}\n---\n\n"
            f"RAW_RESPONSE:\n---\n{raw_output}\n---\n\n"
            f"REQUIRED_ENTRYPOINT: {entry_point}\n"
        )

        record: Dict[str, Any] = {
            "attempted": True,
            "prompt": prompt,
            "generation_kwargs": dict(self._llm_code_extraction_generation_kwargs),
            "raw_extraction_output": "",
            "sanitized_extraction_blocks": [],
            "error": None,
        }

        try:
            extraction_output = self.model.generate(
                prompt, **self._llm_code_extraction_generation_kwargs
            )
            record["raw_extraction_output"] = extraction_output
            sanitized = self.sanitizer.sanitize(extraction_output, entry_point)
            record["sanitized_extraction_blocks"] = sanitized
            return record
        except SystemExit as exc:
            # Do not allow evaluation-only recovery to abort the run.
            record["error"] = f"SystemExit during LLM code extraction: {exc}"
            return record
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            return record

    def _is_ds1000_format(self, sample: FunctionEvaluationSample) -> bool:
        """
        Check if the sample uses DS-1000 format (code_context-based tests).

        Args:
            sample (FunctionEvaluationSample): The sample to check.

        Returns:
            bool: True if this is DS-1000 format, False otherwise.
        """
        # Check source benchmark
        if sample.metadata.get("source_benchmark") == "ds1000":
            return True

        # Check if tests contain test_execution function (DS-1000 marker)
        for test in sample.tests.base_tests:
            if "test_execution" in test and "code_context" in test.lower():
                return True

        return False

    def _execute_ds1000(self, sanitized_code: str, tests: List[str]) -> SandboxResult:
        """
        Execute DS-1000 format tests.

        DS-1000 uses code_context with test_execution function that expects
        the solution as a string. This method wraps the execution to provide
        the solution code string to test_execution.

        Args:
            sanitized_code (str): The sanitized solution code.
            tests (List[str]): The test list (contains code_context).

        Returns:
            SandboxResult: The execution result.
        """
        if not tests:
            return SandboxResult(True, "No tests provided.")

        # The tests contain code_context which defines test_execution.
        # We need to provide the solution code as a string.
        # Create a wrapper that sets solution_code_string before running tests.
        wrapper_code = f"""
# Store solution code as string for DS-1000 test_execution
solution_code_string = {repr(sanitized_code)}

# Now execute the tests (which include code_context and test_execution call)
"""

        # Combine wrapper + tests
        full_test_script = wrapper_code + "\n".join(tests)

        # Execute using the executor's infrastructure
        # We'll use exec directly since we need to provide the solution string
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        old_stdin = sys.stdin
        sys.stdin = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                stderr_buffer
            ):
                exec(full_test_script, {})
            return SandboxResult(
                True,
                "All DS-1000 tests passed.",
                stdout_buffer.getvalue(),
                stderr_buffer.getvalue(),
                status="ok",
            )
        except AssertionError as exc:
            self.logger.debug(
                "AssertionError during DS-1000 execution: %s\nCode:\n%s",
                exc,
                full_test_script,
            )
            return SandboxResult(
                False,
                f"DS-1000 test failed: {exc}",
                stdout_buffer.getvalue(),
                stderr_buffer.getvalue(),
                exception=repr(exc),
                status="assertion_error",
            )
        except Exception as exc:
            self.logger.warning(
                "Exception during DS-1000 execution: %s\nCode:\n%s",
                exc,
                full_test_script,
            )
            return SandboxResult(
                False,
                f"DS-1000 execution error: {type(exc).__name__}: {exc}",
                stdout_buffer.getvalue(),
                stderr_buffer.getvalue(),
                exception=repr(exc),
                status="error",
            )
        finally:
            sys.stdin = old_stdin

    def _extract_entry_point_from_tests(self, test_list: List[str]) -> str:
        """
        Extract the function name (entry point) from test assertions.

        Args:
            test_list (List[str]): List of test assertion strings.

        Returns:
            str: The function name found in the tests.
        """
        if not test_list:
            raise ValueError("Cannot extract entry point from empty test list.")

        # Pattern to match function calls in assert statements
        # Matches: assert function_name(...) or function_name(...) == ...
        func_call_pattern = re.compile(r"(?:assert\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(")

        for test in test_list:
            match = func_call_pattern.search(test)
            if match:
                return match.group(1)

        raise ValueError(f"Could not extract entry point from tests: {test_list}")

    def _transform_personalized_sample(
        self, sample: Dict[str, Any]
    ) -> FunctionEvaluationSample:
        """
        Transform a PersonalizedSample structure into a FunctionEvaluationSample.

        Args:
            sample (Dict[str, Any]): A PersonalizedSample dictionary with
                'original_sample' and 'variations' keys.

        Returns:
            FunctionEvaluationSample: The transformed sample ready for evaluation.
        """
        original_sample = sample.get("original_sample", {})
        variation = sample.get("active_variation")
        if not original_sample:
            raise ValueError("PersonalizedSample missing 'original_sample' field.")

        # Extract basic fields
        sample_id = original_sample.get("sample_id")
        prompt = original_sample.get("prompt")
        original_metadata = original_sample.get("metadata", {})
        metadata = deepcopy(original_metadata)

        if not sample_id:
            raise ValueError("Original sample missing 'sample_id' field.")
        if not prompt:
            raise ValueError("Original sample missing 'prompt' field.")

        # Extract tests from metadata using the modular test extractor.
        # For mbpp_plus we need special handling to separate basic vs. plus tests:
        # - original_test_list -> basic tests / base Pass@k
        # - tests.test (EvalPlus harness) -> rigorous plus test suite
        tests_metadata = original_metadata.get("tests", {})
        source_benchmark = original_sample.get("source_benchmark")
        # Preserve benchmark identity in metadata so later stages (like prompt
        # construction) can apply benchmark-specific policies.
        if source_benchmark and "source_benchmark" not in metadata:
            metadata["source_benchmark"] = source_benchmark

        is_mbpp_plus = source_benchmark == "mbpp_plus"

        if is_mbpp_plus and "original_test_list" in original_metadata:
            # Use the explicit original_test_list as the base test suite.
            test_list = original_metadata.get("original_test_list") or []
            extracted_entry_point = tests_metadata.get("entry_point")
            if extracted_entry_point:
                entry_point = extracted_entry_point
            else:
                try:
                    entry_point = self._extract_entry_point_from_tests(test_list)
                except ValueError:
                    # Fall back to a metadata-provided entry point or a safe default.
                    entry_point = original_metadata.get("entry_point") or "solution"
                    self.logger.debug(
                        "mbpp_plus sample %s: could not extract entry point from "
                        "original_test_list, using default '%s'.",
                        sample_id,
                        entry_point,
                    )
        else:
            try:
                test_list, extracted_entry_point = TestExtractor.extract_tests(
                    tests_metadata, sample_id, source_benchmark
                )
            except ValueError as e:
                self.logger.error(
                    f"Failed to extract tests for sample {sample_id}: {e}"
                )
                raise

            # Extract entry point from tests (if not already extracted)
            if extracted_entry_point:
                entry_point = extracted_entry_point
            else:
                # Try to extract from test_list (for assertion-based tests)
                try:
                    entry_point = self._extract_entry_point_from_tests(test_list)
                except ValueError:
                    # For DS-1000 and other formats without clear entry points,
                    # use a default or extract from metadata
                    entry_point = original_metadata.get("entry_point") or "solution"
                    self.logger.debug(
                        f"Could not extract entry point from tests for sample {sample_id}. "
                        f"Using default: {entry_point}"
                    )

        # Apply variation overrides if present
        variant_label = "original"
        normalized_sample_id = sample_id
        if variation:
            # Check if variation has explicit prompt_type or variant_label
            # This allows control prompts to be properly labeled
            explicit_label = variation.get("prompt_type") or variation.get(
                "variant_label"
            )
            if explicit_label:
                variant_label = str(explicit_label).lower()
            else:
                variant_label = "personalized"  # Default for personalized variations

            prompt_override = self._extract_variation_prompt_text(variation)
            if prompt_override:
                prompt = prompt_override
            variation_id = variation.get("variation_id") or "variation"
            normalized_sample_id = f"{sample_id}::variation::{variation_id}"
            metadata.update(
                {
                    "variant_label": variant_label,
                    "variation_id": variation_id,
                    "applied_changes": variation.get("applied_changes", []),
                    "verification": variation.get("verification"),
                }
            )
        else:
            metadata.setdefault("variant_label", variant_label)

        # Build FunctionEvaluationTests.
        # For mbpp_plus, attach plus tests separately so Pass@k
        # can report base-only vs. base+plus metrics.
        plus_tests: List[str] = []
        if is_mbpp_plus:
            # Optional explicit plus_test_list field (if present).
            mbpp_plus_extra_tests = original_metadata.get("plus_test_list") or []
            if isinstance(mbpp_plus_extra_tests, list):
                plus_tests.extend(mbpp_plus_extra_tests)

            # The EvalPlus robustness harness is stored under tests_metadata["test"].
            mbpp_plus_script = tests_metadata.get("test")
            if isinstance(mbpp_plus_script, str) and mbpp_plus_script.strip():
                plus_tests.append(mbpp_plus_script)

        tests = FunctionEvaluationTests(
            entry_point=entry_point,
            base_tests=test_list,
            plus_tests=plus_tests,
        )

        # Build FunctionEvaluationSample
        return FunctionEvaluationSample(
            sample_id=normalized_sample_id,
            prompt=prompt,
            tests=tests,
            metadata=metadata,
        )

    @staticmethod
    def _extract_variation_prompt_text(variation: Dict[str, Any]) -> Optional[str]:
        """
        Normalize the prompt text stored in a variation entry.

        This helper is intentionally conservative: it only unwraps structured
        representations when it can confidently recover a `modified_prompt`
        string. Otherwise, it returns the raw payload unchanged to preserve
        backward compatibility.

        Supported `variation["modified_prompt"]` shapes:
        - dict: `{"modified_prompt": "<text>"}`
        - JSON string: `'{"modified_prompt": "<text>"}'`
        - Fenced JSON code block string:
          ```json
          {"modified_prompt": "<text>"}
          ```

        Args:
            variation (Dict[str, Any]): Variation record that may contain
                a `modified_prompt` field in one of the supported shapes.

        Returns:
            Optional[str]: The extracted prompt text, or None when no prompt is present.
        """
        logger = logging.getLogger("FunctionLevelStrategy")
        raw_prompt = variation.get("modified_prompt")
        if raw_prompt is None:
            return None
        if isinstance(raw_prompt, dict):
            return raw_prompt.get("modified_prompt")
        if isinstance(raw_prompt, str):
            stripped = raw_prompt.strip()

            # Some upstream components store the JSON payload inside a fenced
            # markdown code block (e.g. ```json ... ```). Try to unwrap it, but
            # only when it successfully parses into a dict with `modified_prompt`.
            if stripped.startswith("```") and "```" in stripped[3:]:
                match = re.search(
                    r"```(?:json)?\s*([\s\S]*?)\s*```",
                    stripped,
                    flags=re.IGNORECASE,
                )
                if match:
                    fenced_payload = match.group(1).strip()
                    if fenced_payload.startswith("{"):
                        try:
                            parsed = json.loads(fenced_payload)
                            if isinstance(parsed, dict) and parsed.get(
                                "modified_prompt"
                            ):
                                return parsed["modified_prompt"]
                        except json.JSONDecodeError:
                            logger.debug(
                                "Failed to parse fenced variation prompt JSON for variation %s.",
                                variation.get("variation_id"),
                            )

            if stripped.startswith("{"):
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict) and parsed.get("modified_prompt"):
                        return parsed["modified_prompt"]
                except json.JSONDecodeError:
                    logger.debug(
                        "Failed to parse variation prompt JSON for variation %s.",
                        variation.get("variation_id"),
                    )
            return raw_prompt
        return str(raw_prompt)

    def prepare_inputs(self, sample: Dict[str, Any]) -> PreparedInput:
        """
        Validate and normalize the raw sample dictionary with Pydantic.

        Handles both PersonalizedSample format (from vibe dataset) and direct
        FunctionEvaluationSample format.
        """
        # Check if this is a PersonalizedSample (has 'original_sample' key)
        if "original_sample" in sample:
            # Transform PersonalizedSample to FunctionEvaluationSample
            normalized_sample = self._transform_personalized_sample(sample)
        else:
            # Assume it's already in FunctionEvaluationSample format
            try:
                # Add in the end a reqest to add final code under <final_code> tag
                # sample["prompt"] = (
                #     sample["prompt"]
                #     + "\n\nIn the end of your answer, add a running final code under <final_code> tag and don't add anything else beside the final code."
                # )
                normalized_sample = FunctionEvaluationSample(**sample)
            except Exception as e:
                self.logger.error(
                    "Failed to parse sample as FunctionEvaluationSample. "
                    "Expected either PersonalizedSample (with 'original_sample') or "
                    f"FunctionEvaluationSample format. Error: {e}"
                )
                raise

        num_completions = (
            normalized_sample.metadata.get("num_completions")
            or self.context.extra.get("num_completions")
            or 1
        )
        num_completions = max(1, int(num_completions))
        generation_kwargs = (
            normalized_sample.metadata.get("generation_kwargs")
            or self.context.extra.get("generation_kwargs")
            or {}
        )
        payload = {
            "sample": normalized_sample,
            "num_completions": int(num_completions),
            "generation_kwargs": generation_kwargs,
        }
        return PreparedInput(sample_id=normalized_sample.sample_id, payload=payload)

    def run_generation(self, prepared: PreparedInput) -> GenerationResult:
        """
        Generate K completions for the provided prompt.
        """
        # Delegate to the batch pathway for consistent handling.
        return self.run_generation_batch([prepared])[0]

    def run_generation_batch(
        self, prepared_batch: List[PreparedInput]
    ) -> List[GenerationResult]:
        """
        Generate completions for a batch of prepared inputs using model.generate_batch.
        """
        requests: List[GenerationRequest] = []
        sample_index: List[str] = []
        prompt_lookup: Dict[str, str] = {}
        meta_lookup: Dict[str, Dict[str, Any]] = {}

        for prepared in prepared_batch:
            sample: FunctionEvaluationSample = prepared.payload["sample"]
            num_completions: int = prepared.payload["num_completions"]
            generation_kwargs: Dict[str, Any] = prepared.payload["generation_kwargs"]

            source_benchmark = (sample.metadata.get("source_benchmark") or "").lower()

            # HumanEval prompts already contain all relevant examples/context and
            # should not be modified by appending tests at evaluation time.
            if source_benchmark in {"humaneval", "humaneval_plus"}:
                prompt_with_tests = sample.prompt
            else:
                # Append test cases to the prompt if available.
                test_list = sample.metadata.get("prompt_tests")
                if not test_list:
                    test_list = sample.tests.base_tests + sample.tests.plus_tests
                prompt_with_tests = sample.prompt + format_test_cases_for_prompt(
                    test_list
                )

            prompt_lookup[sample.sample_id] = prompt_with_tests
            meta_lookup[sample.sample_id] = {
                "sample": sample,
                "num_completions": num_completions,
            }

            for _ in range(num_completions):
                requests.append(
                    GenerationRequest(
                        prompt=prompt_with_tests,
                        generation_kwargs=dict(generation_kwargs),
                    )
                )
                sample_index.append(sample.sample_id)

        debug_forced_raw_output = None
        try:
            debug_forced_raw_output = (self.context.extra or {}).get(
                "debug_force_raw_output"
            )
        except Exception:
            debug_forced_raw_output = None

        if debug_forced_raw_output is not None:
            self.logger.warning(
                "DEBUG: Bypassing model generation and forcing raw_output for all requests."
            )
            raw_outputs = [str(debug_forced_raw_output)] * len(requests)
        else:
            raw_outputs = self.model.generate_batch(requests)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for idx, raw_output in enumerate(raw_outputs):
            sample_id = sample_index[idx]
            grouped.setdefault(sample_id, []).append(
                {"index": len(grouped.get(sample_id, [])), "raw_output": raw_output}
            )

        results: List[GenerationResult] = []
        for prepared in prepared_batch:
            sample: FunctionEvaluationSample = prepared.payload["sample"]
            generations = grouped.get(sample.sample_id, [])
            metadata = {
                "prompt": prompt_lookup[sample.sample_id],
                "num_completions": meta_lookup[sample.sample_id]["num_completions"],
                "sample": sample,
            }
            results.append(
                GenerationResult(
                    sample_id=sample.sample_id,
                    generations=generations,
                    metadata=metadata,
                )
            )
        return results

    def execute(self, generated: GenerationResult) -> ExecutionResult:
        """
        Sanitize generations and execute them against base/plus test suites.
        """
        sample_obj = generated.metadata.get("sample")
        if not isinstance(sample_obj, FunctionEvaluationSample):
            raise ValueError(
                "FunctionLevelStrategy expected sample metadata during execution."
            )
        sample: FunctionEvaluationSample = sample_obj

        # Check if this is DS-1000 format (has code_context in tests)
        is_ds1000 = self._is_ds1000_format(sample)

        records: List[Dict[str, Any]] = []
        base_pass_flags: List[bool] = []
        plus_pass_flags: List[bool] = []
        failures_counters: Dict[str, int] = {
            k: 0 for k in self._cumulative_failures_counters
        }
        for generation in generated.generations:
            raw_output: str = generation["raw_output"]
            if "<final_code>" in raw_output:
                raw_output_to_sanitize = raw_output.split("<final_code>")[0].strip()
            else:
                raw_output_to_sanitize = raw_output
            sanitized_code_blocks = self.sanitizer.sanitize(
                raw_output_to_sanitize, sample.tests.entry_point
            )

            primary_run = self._run_with_sanitized_blocks(
                sanitized_code_blocks, sample, is_ds1000
            )
            primary_selected_code_block = primary_run["selected_code_block"]
            primary_selected_block_index = primary_run["selected_block_index"]
            primary_base_result = primary_run["base_result"]
            primary_plus_result = primary_run["plus_result"]

            # Track primary failures (even if recovery later succeeds).
            self._increment_failure_counter(
                failures_counters, "primary_", primary_base_result
            )
            if sample.tests.plus_tests:
                if primary_plus_result is not None and not primary_plus_result.passed:
                    failure_type = self._failure_type_from_status(
                        primary_plus_result.status
                    )
                    if failure_type == "assertion":
                        failures_counters["primary_assertion_errors_plus"] += 1
                    elif failure_type == "timeout":
                        failures_counters["primary_timeouts_plus"] += 1
                    else:
                        failures_counters["primary_execution_errors_plus"] += 1

            # Final results default to the primary run (unless a fallback succeeds).
            selected_code_block = primary_selected_code_block
            selected_block_index = primary_selected_block_index
            base_result = primary_base_result
            plus_result = primary_plus_result

            trigger_result, trigger_phase = self._get_llm_extraction_trigger(
                base_result, plus_result
            )

            llm_extraction_record: Dict[str, Any] = {
                "enabled": self._enable_llm_code_extraction,
                "attempted": False,
                "trigger_status": trigger_result.status if trigger_result else None,
                "trigger_phase": trigger_phase,
                "used_for_final": False,
                "retry_on_timeout": self._llm_code_extraction_retry_on_timeout,
                "extraction": None,
                "retry": None,
                "retry_attempts": [],
            }

            # Evaluation-only fallback: ask the model to extract runnable code from its own response,
            # then re-sanitize and retry execution. This does NOT replace raw_output.
            if self._should_attempt_llm_code_extraction(trigger_result):
                llm_extraction_record["attempted"] = True
                failures_counters["llm_extraction_attempts"] += 1
                effective_prompt = str(
                    generated.metadata.get("prompt") or sample.prompt
                )
                extraction = self._attempt_llm_code_extraction(
                    effective_prompt,
                    raw_output_to_sanitize,
                    sample.tests.entry_point,
                    language="python",
                )
                llm_extraction_record["extraction"] = extraction
                if extraction.get("error"):
                    failures_counters["llm_extraction_generation_errors"] += 1

                sanitized_retry_blocks = (
                    extraction.get("sanitized_extraction_blocks") or []
                )
                if not sanitized_retry_blocks:
                    failures_counters["llm_extraction_no_code_blocks"] += 1
                max_attempts = 1 + self._llm_code_extraction_retry_on_timeout
                retry_base = None
                retry_plus = None
                retry_selected = None
                retry_selected_idx = None

                for attempt_idx in range(max_attempts):
                    attempt_start = time.time()
                    retry_run = self._run_with_sanitized_blocks(
                        sanitized_retry_blocks,
                        sample,
                        is_ds1000,
                        executor=self._llm_retry_executor,
                    )
                    attempt_duration = time.time() - attempt_start
                    retry_base = retry_run["base_result"]
                    retry_plus = retry_run["plus_result"]
                    retry_selected = retry_run["selected_code_block"]
                    retry_selected_idx = retry_run["selected_block_index"]

                    attempt_payload = {
                        "attempt_index": attempt_idx,
                        "timeout_seconds": self._llm_code_extraction_retry_timeout,
                        "duration_seconds": attempt_duration,
                        "sanitized_code_blocks": sanitized_retry_blocks,
                        "selected_code_block": retry_selected,
                        "selected_block_index": retry_selected_idx,
                        "base_result": asdict(retry_base) if retry_base else None,
                        "plus_result": asdict(retry_plus) if retry_plus else None,
                    }
                    llm_extraction_record["retry_attempts"].append(attempt_payload)

                    if retry_base and retry_base.passed:
                        break
                    if retry_base is None or retry_base.status != "timeout":
                        # Only retry on timeout; other failures stop immediately.
                        break
                    # Log which attempt timed out for easier debugging.
                    self.logger.warning(
                        "LLM-extracted retry timed out (attempt %s/%s, timeout=%ss, duration=%.2fs) for sample_id=%s",
                        attempt_idx + 1,
                        max_attempts,
                        self._llm_code_extraction_retry_timeout,
                        attempt_duration,
                        sample.sample_id,
                    )

                llm_extraction_record["retry"] = (
                    llm_extraction_record["retry_attempts"][-1]
                    if llm_extraction_record["retry_attempts"]
                    else None
                )

                if retry_base and retry_base.passed:
                    llm_extraction_record["used_for_final"] = True
                    failures_counters["llm_extraction_used_for_final"] += 1
                    selected_code_block = retry_selected
                    selected_block_index = (
                        retry_selected_idx if retry_selected_idx is not None else -1
                    )
                    base_result = retry_base
                    plus_result = retry_plus
                else:
                    # Count the final (last) retry failure separately.
                    if retry_base is not None and not retry_base.passed:
                        failure_type = self._failure_type_from_status(retry_base.status)
                        if failure_type == "assertion":
                            failures_counters["llm_retry_assertion_errors"] += 1
                        elif failure_type == "timeout":
                            failures_counters["llm_retry_timeouts"] += 1
                        else:
                            failures_counters["llm_retry_execution_errors"] += 1

                    if (
                        sample.tests.plus_tests
                        and retry_plus is not None
                        and not retry_plus.passed
                    ):
                        failure_type = self._failure_type_from_status(retry_plus.status)
                        if failure_type == "assertion":
                            failures_counters["llm_retry_assertion_errors_plus"] += 1
                        elif failure_type == "timeout":
                            failures_counters["llm_retry_timeouts_plus"] += 1
                        else:
                            failures_counters["llm_retry_execution_errors_plus"] += 1

            # Final-result failure counters (backward compatible).
            if base_result is not None and not base_result.passed:
                failure_type = self._failure_type_from_status(base_result.status)
                if failure_type == "assertion":
                    failures_counters["assertion_errors"] += 1
                elif failure_type == "timeout":
                    failures_counters["timeouts"] += 1
                else:
                    failures_counters["execution_errors"] += 1
            if plus_result is not None and not plus_result.passed:
                failure_type = self._failure_type_from_status(plus_result.status)
                if failure_type == "assertion":
                    failures_counters["assertion_errors_plus"] += 1
                elif failure_type == "timeout":
                    failures_counters["timeouts_plus"] += 1
                else:
                    failures_counters["execution_errors_plus"] += 1

            base_pass_flags.append(base_result.passed if base_result else False)
            plus_pass_flags.append(plus_result.passed if plus_result else False)
            records.append(
                {
                    "index": generation["index"],
                    "raw_output": raw_output,
                    "primary_run": {
                        "sanitized_code_blocks": sanitized_code_blocks,
                        "selected_code_block": primary_selected_code_block,
                        "selected_block_index": primary_selected_block_index,
                        "base_result": (
                            asdict(primary_base_result) if primary_base_result else None
                        ),
                        "plus_result": (
                            asdict(primary_plus_result) if primary_plus_result else None
                        ),
                    },
                    "sanitized_code_blocks": sanitized_code_blocks,
                    "selected_code_block": selected_code_block,
                    "selected_block_index": selected_block_index,
                    "base_result": asdict(base_result) if base_result else None,
                    "plus_result": asdict(plus_result) if plus_result else None,
                    "llm_code_extraction": llm_extraction_record,
                }
            )

        # Get the prompt that was actually sent to the model (with tests appended)
        prompt = generated.metadata.get("prompt", sample.prompt)

        artifacts = {
            "records": records,
            "base_pass_flags": base_pass_flags,
            "plus_pass_flags": plus_pass_flags,
            "tests": sample.tests.model_dump(),
            "sample_metadata": sample.metadata,
            "prompt": prompt,
            "failures_counters": failures_counters,
        }

        # Update cumulative counters and log progress after each evaluated sample.
        for key, value in failures_counters.items():
            self._cumulative_failures_counters[key] = int(
                self._cumulative_failures_counters.get(key, 0) + int(value)
            )
        artifacts["cumulative_failures_counters"] = dict(
            self._cumulative_failures_counters
        )

        # self.logger.info("Failures counters (this sample): %s"s, failures_counters) # no need for that
        cumulative_failures_counters_preety_print_string = f"\nCumulative failures counters: assertion_errors: {self._cumulative_failures_counters['assertion_errors']}, execution_errors: {self._cumulative_failures_counters['execution_errors']}, timeouts: {self._cumulative_failures_counters['timeouts']}, assertion_errors_plus: {self._cumulative_failures_counters['assertion_errors_plus']}, execution_errors_plus: {self._cumulative_failures_counters['execution_errors_plus']}, timeouts_plus: {self._cumulative_failures_counters['timeouts_plus']}"
        self.logger.info(
            "Failures counters (cumulative so far): %s",
            cumulative_failures_counters_preety_print_string,
        )
        cumulative_llm_retry_counters_preety_print_string = f"\nCumulative LLM retry counters: llm_retry_assertion_errors: {self._cumulative_failures_counters['llm_retry_assertion_errors']}, llm_retry_timeouts: {self._cumulative_failures_counters['llm_retry_timeouts']}, llm_retry_execution_errors: {self._cumulative_failures_counters['llm_retry_execution_errors']}, llm_retry_assertion_errors_plus: {self._cumulative_failures_counters['llm_retry_assertion_errors_plus']}, llm_retry_timeouts_plus: {self._cumulative_failures_counters['llm_retry_timeouts_plus']}, llm_retry_execution_errors_plus: {self._cumulative_failures_counters['llm_retry_execution_errors_plus']}"
        self.logger.info(
            "LLM retry counters (cumulative so far): %s",
            cumulative_llm_retry_counters_preety_print_string,
        )
        return ExecutionResult(
            sample_id=generated.sample_id, artifacts=artifacts, metrics={}
        )

    def collect_metrics(self, executed: ExecutionResult) -> StrategyResult:
        """
        Convert execution traces into Pass@k metrics via the reporter.
        """
        base_flags = executed.artifacts["base_pass_flags"]
        plus_flags = executed.artifacts["plus_pass_flags"]
        records = executed.artifacts["records"]

        # Extract failure type information from records
        base_assertion_failures = []
        base_execution_failures = []
        plus_assertion_failures = []
        plus_execution_failures = []

        for record in records:
            base_result = record.get("base_result", {})
            plus_result = record.get("plus_result", {})

            # Track base test failures
            if not base_result.get("passed", True):
                base_status = base_result.get("status", "error")
                if base_status == "assertion_error":
                    base_assertion_failures.append(True)
                    base_execution_failures.append(False)
                else:
                    base_assertion_failures.append(False)
                    base_execution_failures.append(True)
            else:
                base_assertion_failures.append(False)
                base_execution_failures.append(False)

            # Track plus test failures
            if not plus_result.get("passed", True):
                plus_status = plus_result.get("status", "error")
                if plus_status == "assertion_error":
                    plus_assertion_failures.append(True)
                    plus_execution_failures.append(False)
                else:
                    plus_assertion_failures.append(False)
                    plus_execution_failures.append(True)
            else:
                plus_assertion_failures.append(False)
                plus_execution_failures.append(False)

        sample_metrics = self.reporter.record(
            executed.sample_id,
            base_flags,
            plus_flags,
            base_assertion_failures=base_assertion_failures,
            base_execution_failures=base_execution_failures,
            plus_assertion_failures=plus_assertion_failures,
            plus_execution_failures=plus_execution_failures,
        )
        return StrategyResult(
            sample_id=executed.sample_id,
            metrics=sample_metrics,
            artifacts=executed.artifacts,
        )

    def report(
        self, results: List[StrategyResult], prompt_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Write per-sample traces and aggregated Pass@k summaries to disk.
        """
        detail_dir = Path(self.context.output_dir) / "function_eval"
        detail_dir.mkdir(parents=True, exist_ok=True)
        model_name = getattr(self.model, "model_name", None)
        model_config_path = None
        if self.context.extra:
            model_config_path = self.context.extra.get("model_config_path")
        for result in results:
            path = self.get_expected_output_path(result.sample_id)
            payload: Dict[str, Any] = {
                "sample_id": result.sample_id,
                "metrics": result.metrics,
                "artifacts": result.artifacts,
            }
            if model_name:
                payload["model_name"] = model_name
            if model_config_path:
                payload["model_config_path"] = model_config_path
            save_json(payload, str(path))
        export_paths = self.reporter.export(str(detail_dir), prompt_type=prompt_type)
        return export_paths

    def get_expected_output_path(self, sample_id: str) -> Optional[Path]:
        """
        Return the Path where the result for a specific sample_id would be stored.
        """
        detail_dir = Path(self.context.output_dir) / "function_eval"
        detail = f"sample-{normalize_token(sample_id.replace('::', '-'))}"
        filename = format_artifact_name(
            artifact_type="function_eval",
            evaluation_type="objective",
            detail=detail,
            version=0,
            ext="json",
            run_name=self._run_name,
        )
        return detail_dir / filename
