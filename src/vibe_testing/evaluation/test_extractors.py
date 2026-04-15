"""
Test extraction utilities for different dataset formats.

This module provides format-agnostic test extraction from various benchmark
dataset formats, allowing evaluation strategies to work with different test
representations (test_list, code_context, etc.).
"""

from typing import Any, Dict, List, Optional


class TestExtractor:
    """
    Extracts and normalizes tests from different dataset formats.

    Supports multiple test formats:
    - test_list: List of assertion strings (MBPP, HumanEval)
    - code_context: DS-1000 execution-based format
    - test: Single test string (HumanEval alternative)
    """

    @staticmethod
    def extract_tests(
        tests_metadata: Dict[str, Any],
        sample_id: str,
        source_benchmark: Optional[str] = None,
    ) -> tuple[List[str], Optional[str]]:
        """
        Extract tests from metadata in a format-agnostic way.

        Args:
            tests_metadata (Dict[str, Any]): The 'tests' dictionary from sample metadata.
            sample_id (str): Sample identifier for error messages.
            source_benchmark (Optional[str]): Source benchmark name to help determine format.

        Returns:
            tuple[List[str], Optional[str]]: A tuple of (test_list, entry_point).
                test_list: List of test strings ready for execution.
                entry_point: Function name to extract, or None if not applicable.

        Raises:
            ValueError: If no valid test format is found.
        """
        if not tests_metadata:
            raise ValueError(f"Sample {sample_id} has no 'tests' field in metadata.")

        entry_point = tests_metadata.get("entry_point")

        # Try test_list format (MBPP, HumanEval)
        test_list = tests_metadata.get("test_list")
        if test_list and isinstance(test_list, list) and len(test_list) > 0:
            return test_list, entry_point

        # Try code_context format (DS-1000)
        code_context = tests_metadata.get("code_context")
        if code_context and isinstance(code_context, str):
            return (
                TestExtractor._extract_from_code_context(code_context, sample_id),
                entry_point,
            )

        # Try single test string format (HumanEval alternative)
        test_string = tests_metadata.get("test")
        if test_string and isinstance(test_string, str):
            return [test_string], entry_point

        # If we have a source benchmark hint, provide more specific error
        benchmark_hint = f" (from {source_benchmark})" if source_benchmark else ""
        raise ValueError(
            f"Sample {sample_id}{benchmark_hint} has no recognized test format in metadata.tests. "
            f"Expected one of: 'test_list' (list), 'code_context' (str), or 'test' (str). "
            f"Found keys: {list(tests_metadata.keys())}"
        )

    @staticmethod
    def _extract_from_code_context(code_context: str, sample_id: str) -> List[str]:
        """
        Convert DS-1000 code_context format to executable test list.

        DS-1000 uses a code_context string that contains:
        - Helper functions (generate_test_case, exec_test, etc.)
        - An exec_context template with [insert] placeholder
        - A test_execution function that validates the solution

        The test_execution function expects a solution string and replaces [insert]
        in exec_context with it, then executes and validates.

        This method creates a wrapper that:
        1. Includes the code_context (defines test_execution)
        2. Calls test_execution with solution_code_string (provided by executor)

        Args:
            code_context (str): The code_context string from DS-1000.
            sample_id (str): Sample identifier for error messages.

        Returns:
            List[str]: A list containing the code_context and a test wrapper.
        """
        # The executor will provide solution_code_string before executing tests.
        # We just need to call test_execution with it.
        wrapper = f"""{code_context}

# DS-1000 test execution wrapper
# Call test_execution with the solution code string (provided by executor)
try:
    if 'test_execution' not in globals():
        raise AssertionError("DS-1000 test_execution function not found in code_context")
    
    if 'solution_code_string' not in globals() and 'solution_code_string' not in locals():
        raise NameError("solution_code_string not found. This should be provided by the executor.")
    
    test_execution(solution_code_string)
except NameError as e:
    raise AssertionError(f"DS-1000 test requires solution code string: {{e}}")
except Exception as e:
    raise AssertionError(f"DS-1000 test execution failed: {{e}}")
"""

        return [wrapper]
