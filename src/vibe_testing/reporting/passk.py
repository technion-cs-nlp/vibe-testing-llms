"""
Utilities for computing Pass@k metrics on function-level benchmarks.
"""

from __future__ import annotations

import csv
import math
import os
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

from src.vibe_testing.utils import save_json
from src.vibe_testing.pathing import format_artifact_name


def _pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """
    Compute Pass@k following the standard OpenAI/HumanEval unbiased estimator.

    This estimator is:
        1 - C(n - c, k) / C(n, k)
    where:
        n = number of sampled solutions
        c = number of correct solutions among them
        k = requested Pass@k
    """
    if num_samples == 0:
        return 0.0
    if num_samples < k:
        return float(num_correct > 0)
    if num_correct == 0:
        return 0.0
    combos = math.comb(num_samples, k)
    fail_combos = math.comb(num_samples - num_correct, k)
    return 1.0 - fail_combos / combos


def _empirical_pass_at_k(pass_flags: List[bool], k: int) -> float:
    """
    Compute the empirical Pass@k from the observed attempt ordering.

    This is simply whether any of the first k attempts passed. It is useful as a
    diagnostic baseline, but it is not the standard unbiased estimator when
    multiple samples are available.
    """
    if not pass_flags:
        return 0.0
    k = max(1, int(k))
    return float(any(bool(flag) for flag in pass_flags[:k]))


class PassKReporter:
    """
    Tracks per-sample Pass@k metrics for both base and rigorous test suites.
    """

    def __init__(
        self,
        ks: Sequence[int],
        run_name: str = "no_name",
        model_name: Optional[str] = None,
        model_config_path: Optional[str] = None,
    ):
        """
        Args:
            ks (Sequence[int]): The Pass@k values to compute (e.g., [1, 5, 10]).
        """
        if not ks:
            raise ValueError("At least one k value is required for Pass@k reporting.")
        self.ks = sorted(set(int(k) for k in ks))
        self._records: List[Dict[str, Any]] = []
        self._run_name = run_name
        self._model_name = model_name
        self._model_config_path = model_config_path

    def record(
        self,
        sample_id: str,
        base_passes: Iterable[bool],
        plus_passes: Iterable[bool],
        base_assertion_failures: Optional[Iterable[bool]] = None,
        base_execution_failures: Optional[Iterable[bool]] = None,
        plus_assertion_failures: Optional[Iterable[bool]] = None,
        plus_execution_failures: Optional[Iterable[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Store per-sample pass/fail traces and compute Pass@k for both suites.
        
        Args:
            sample_id: Unique identifier for the sample
            base_passes: Boolean flags indicating base test suite pass/fail per attempt
            plus_passes: Boolean flags indicating plus test suite pass/fail per attempt
            base_assertion_failures: Boolean flags indicating assertion failures in base tests
            base_execution_failures: Boolean flags indicating execution failures in base tests
            plus_assertion_failures: Boolean flags indicating assertion failures in plus tests
            plus_execution_failures: Boolean flags indicating execution failures in plus tests
        """
        base_flags = list(bool(flag) for flag in base_passes)
        plus_flags = list(bool(flag) for flag in plus_passes)
        
        # Convert failure type flags to lists, defaulting to all False if not provided
        base_assertion = (
            list(bool(flag) for flag in base_assertion_failures)
            if base_assertion_failures is not None
            else [False] * len(base_flags)
        )
        base_execution = (
            list(bool(flag) for flag in base_execution_failures)
            if base_execution_failures is not None
            else [False] * len(base_flags)
        )
        plus_assertion = (
            list(bool(flag) for flag in plus_assertion_failures)
            if plus_assertion_failures is not None
            else [False] * len(plus_flags)
        )
        plus_execution = (
            list(bool(flag) for flag in plus_execution_failures)
            if plus_execution_failures is not None
            else [False] * len(plus_flags)
        )
        
        # Compute failure counts
        base_assertion_count = sum(base_assertion)
        base_execution_count = sum(base_execution)
        plus_assertion_count = sum(plus_assertion)
        plus_execution_count = sum(plus_execution)
        
        sample_metrics: Dict[str, Any] = {
            "sample_id": sample_id,
            "base": {
                "per_attempt": base_flags,
                "pass_at_k": self._curve(base_flags),
                "assertion_failures": base_assertion_count,
                "execution_failures": base_execution_count,
            },
            "plus": {
                "per_attempt": plus_flags,
                "pass_at_k": self._curve(plus_flags),
                "assertion_failures": plus_assertion_count,
                "execution_failures": plus_execution_count,
            },
        }
        if self._model_name is not None:
            sample_metrics["model_name"] = self._model_name
        self._records.append(sample_metrics)
        return sample_metrics

    def summarize(self) -> Dict[str, Any]:
        """
        Aggregate Pass@k across all recorded samples.
        """
        if not self._records:
            return {
                "num_samples": 0,
                "base": {},
                "plus": {},
                "unbiased_pass_at_k": {"base": {}, "plus": {}},
                "empirical_pass_at_k": {"base": {}, "plus": {}},
                "failure_counts": {"base": {}, "plus": {}},
            }
        summary = {
            "num_samples": len(self._records),
            "base": {},
            "plus": {},
            "unbiased_pass_at_k": {"base": {}, "plus": {}},
            "empirical_pass_at_k": {"base": {}, "plus": {}},
            "failure_counts": {"base": {}, "plus": {}},
        }
        for suite in ("base", "plus"):
            per_k_values = {str(k): [] for k in self.ks}
            per_k_empirical = {str(k): [] for k in self.ks}
            total_assertion_failures = 0
            total_execution_failures = 0
            for record in self._records:
                for k, value in record[suite]["pass_at_k"].items():
                    per_k_values[k].append(value)
                flags = record[suite].get("per_attempt") or []
                if isinstance(flags, list):
                    for k in self.ks:
                        per_k_empirical[str(k)].append(
                            _empirical_pass_at_k([bool(v) for v in flags], k)
                        )
                # Aggregate failure counts
                total_assertion_failures += record[suite].get("assertion_failures", 0)
                total_execution_failures += record[suite].get("execution_failures", 0)
            suite_unbiased = {
                k: mean(values) if values else 0.0 for k, values in per_k_values.items()
            }
            summary[suite] = suite_unbiased
            summary["unbiased_pass_at_k"][suite] = suite_unbiased
            summary["empirical_pass_at_k"][suite] = {
                k: mean(values) if values else 0.0
                for k, values in per_k_empirical.items()
            }
            summary["failure_counts"][suite] = {
                "assertion_failures": total_assertion_failures,
                "execution_failures": total_execution_failures,
            }
        return summary

    def export(
        self,
        output_dir: str,
        filename_prefix: str = "function_eval",
        prompt_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Persist per-sample metrics and aggregated Pass@k to JSON and CSV files.
        """
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summarize()
        payload: Dict[str, Any] = {"samples": self._records, "summary": summary}
        if self._model_name is not None:
            payload["model_name"] = self._model_name
        if self._model_config_path is not None:
            payload["model_config_path"] = self._model_config_path
        
        detail = f"{filename_prefix}-metrics"
        if prompt_type:
            detail = f"{filename_prefix}-{prompt_type}-metrics"
            
        json_name = format_artifact_name(
            artifact_type="function_eval",
            evaluation_type="metrics",
            detail=detail,
            version=0,
            ext="json",
            run_name=self._run_name,
        )
        csv_name = format_artifact_name(
            artifact_type="function_eval",
            evaluation_type="metrics",
            detail=detail,
            version=0,
            ext="csv",
            run_name=self._run_name,
        )
        json_path = target_dir / json_name
        csv_path = target_dir / csv_name
        save_json(payload, str(json_path))
        self._write_csv(str(csv_path))
        return {"json": str(json_path), "csv": str(csv_path), "summary": summary}

    def _curve(self, pass_flags: List[bool]) -> Dict[str, float]:
        """
        Compute Pass@k values for the recorded attempts.
        """
        n = len(pass_flags)
        c = sum(pass_flags)
        return {str(k): _pass_at_k(n, c, k) for k in self.ks}

    def _write_csv(self, path: str) -> None:
        """
        Write a per-sample CSV summarizing Pass@k values and failure counts.
        """
        fieldnames = (
            ["sample_id"]
            + [f"base_pass@{k}" for k in self.ks]
            + [f"plus_pass@{k}" for k in self.ks]
            + [
                "base_assertion_failures",
                "base_execution_failures",
                "plus_assertion_failures",
                "plus_execution_failures",
            ]
        )
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._records:
                row = {"sample_id": record["sample_id"]}
                for k, value in record["base"]["pass_at_k"].items():
                    row[f"base_pass@{k}"] = value
                for k, value in record["plus"]["pass_at_k"].items():
                    row[f"plus_pass@{k}"] = value
                row["base_assertion_failures"] = record["base"].get(
                    "assertion_failures", 0
                )
                row["base_execution_failures"] = record["base"].get(
                    "execution_failures", 0
                )
                row["plus_assertion_failures"] = record["plus"].get(
                    "assertion_failures", 0
                )
                row["plus_execution_failures"] = record["plus"].get(
                    "execution_failures", 0
                )
                writer.writerow(row)
