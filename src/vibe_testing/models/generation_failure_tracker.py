"""
Global consecutive generation failure tracking.

This module provides a single, process-wide counter that tracks consecutive
generation failures across *all* model wrappers in `src.vibe_testing.models`.

Why this exists:
- Some evaluation loops intentionally catch `Exception` and continue, which can
  silently produce many bad/placeholder generations.
- We want a hard safety stop when the system is likely in a bad state
  (e.g., invalid credentials, rate limits, backend outage).

Policy:
- On any successful generation, reset the global consecutive-failure counter to 0.
- On any generation failure, increment the counter.
- If failures exceed the configured threshold (default: 100), abort the entire
  run immediately with a clear error message.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Mapping, Optional

logger = logging.getLogger(__name__)


class ConsecutiveGenerationFailuresAbort(SystemExit):
    """
    Raised when the global consecutive generation failure threshold is exceeded.

    We intentionally inherit from SystemExit (a BaseException) so that broad
    `except Exception:` handlers do not swallow this abort signal.
    """


@dataclass(frozen=True)
class FailureStreakState:
    """
    Snapshot of the global failure streak.

    Args:
        consecutive_failures: Number of consecutive failures since the last success.
        last_model_name: Model name that caused the last recorded failure.
        last_error_repr: repr() of the last failure exception (best-effort).
        threshold: The maximum allowed consecutive failures before aborting.
    """

    consecutive_failures: int
    last_model_name: Optional[str]
    last_error_repr: Optional[str]
    threshold: int


_LOCK = threading.Lock()
_CONSECUTIVE_FAILURES = 0
_LAST_MODEL_NAME: Optional[str] = None
_LAST_ERROR_REPR: Optional[str] = None
# Maximum allowed consecutive failures before aborting. We abort on the
# (_THRESHOLD + 1)th failure to match the policy described in the module docstring.
_DEFAULT_THRESHOLD = 100
_THRESHOLD = _DEFAULT_THRESHOLD


def configure_failure_threshold(
    threshold: int, *, source: Optional[str] = None
) -> None:
    """
    Configure the global consecutive-generation-failure threshold.

    This intentionally does NOT consult environment variables. The threshold is meant
    to be controlled via model configuration (YAML), so runs are reproducible and
    provenance is explicit.

    Args:
        threshold (int): Maximum allowed consecutive failures (>= 0). The abort is
            triggered on the (threshold + 1)-th consecutive failure.
        source (Optional[str]): Optional provenance string (e.g., config path) for logs.

    Raises:
        ValueError: If threshold is negative.
        TypeError: If threshold is not an int.
    """

    if not isinstance(threshold, int):
        raise TypeError(f"threshold must be int. Got: {type(threshold).__name__}")
    if threshold < 0:
        raise ValueError(f"threshold must be >= 0. Got: {threshold}")

    global _THRESHOLD
    with _LOCK:
        previous = _THRESHOLD
        _THRESHOLD = threshold

    if previous != threshold:
        logger.info(
            "Configured global generation failure threshold: %d -> %d%s",
            previous,
            threshold,
            f" (source={source})" if source else "",
        )


def configure_failure_threshold_from_model_config(
    model_config: Mapping[str, Any],
    *,
    default: int = _DEFAULT_THRESHOLD,
    source: Optional[str] = None,
) -> int:
    """
    Configure the global threshold based on a model configuration mapping.

    Expected key (preferred):
        model.params.max_consecutive_generation_failures: int

    Backward-compatible alias (also accepted):
        max_consecutive_generation_failures: int

    Args:
        model_config (Mapping[str, Any]): Loaded model config dictionary.
        default (int): Default threshold to use if not specified in the config.
        source (Optional[str]): Optional provenance string for logs.

    Returns:
        int: The resolved threshold value that was applied.

    Raises:
        ValueError: If the configured value is present but invalid.
        TypeError: If the configured value is present but not an int.
    """

    value: Any = None
    try:
        model_section = model_config.get("model")
        if isinstance(model_section, Mapping):
            params = model_section.get("params")
            if isinstance(params, Mapping):
                value = params.get("max_consecutive_generation_failures")
        if value is None:
            value = model_config.get("max_consecutive_generation_failures")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Failed to read max_consecutive_generation_failures from model config: {exc}"
        ) from exc

    if value is None:
        configure_failure_threshold(int(default), source=source)
        return int(default)

    if isinstance(value, bool):
        raise TypeError(
            "max_consecutive_generation_failures must be an int (not bool). "
            f"Got: {value!r}"
        )
    if not isinstance(value, int):
        raise TypeError(
            "max_consecutive_generation_failures must be an int. "
            f"Got: {type(value).__name__} ({value!r})"
        )
    if value < 0:
        raise ValueError(
            "max_consecutive_generation_failures must be >= 0. " f"Got: {value}"
        )

    configure_failure_threshold(value, source=source)
    return value


def reset_failure_streak() -> None:
    """
    Reset the global failure streak to a clean state.

    This is primarily intended for tests or process-level reinitialization.
    """

    global _CONSECUTIVE_FAILURES, _LAST_MODEL_NAME, _LAST_ERROR_REPR
    with _LOCK:
        _CONSECUTIVE_FAILURES = 0
        _LAST_MODEL_NAME = None
        _LAST_ERROR_REPR = None


def get_failure_streak_state() -> FailureStreakState:
    """
    Get the current global failure streak state (for debugging/tests).

    Returns:
        FailureStreakState: An immutable snapshot of the current streak state.
    """

    with _LOCK:
        return FailureStreakState(
            consecutive_failures=_CONSECUTIVE_FAILURES,
            last_model_name=_LAST_MODEL_NAME,
            last_error_repr=_LAST_ERROR_REPR,
            threshold=_THRESHOLD,
        )


def record_generation_success(model_name: str) -> None:
    """
    Record a successful generation and reset the global consecutive-failure streak.

    Args:
        model_name (str): The model that succeeded (used only for debug logging).
    """

    global _CONSECUTIVE_FAILURES, _LAST_MODEL_NAME, _LAST_ERROR_REPR
    with _LOCK:
        if _CONSECUTIVE_FAILURES:
            logger.info(
                "Generation success for model '%s' resets consecutive failure streak (%d -> 0).",
                model_name,
                _CONSECUTIVE_FAILURES,
            )
        _CONSECUTIVE_FAILURES = 0
        _LAST_MODEL_NAME = None
        _LAST_ERROR_REPR = None


def record_generation_failure(model_name: str, exc: BaseException) -> None:
    """
    Record a failed generation attempt and enforce the global abort policy.

    Args:
        model_name (str): The model that failed.
        exc (BaseException): The exception (or error) representing the failure.

    Raises:
        ConsecutiveGenerationFailuresAbort: When consecutive failures exceed the
            configured threshold.
    """

    global _CONSECUTIVE_FAILURES, _LAST_MODEL_NAME, _LAST_ERROR_REPR
    error_repr = None
    try:
        error_repr = repr(exc)
    except Exception:  # noqa: BLE001
        error_repr = "<unreprable error>"

    with _LOCK:
        _CONSECUTIVE_FAILURES += 1
        _LAST_MODEL_NAME = model_name
        _LAST_ERROR_REPR = error_repr

        failure_count = _CONSECUTIVE_FAILURES
        abort_on = _THRESHOLD + 1

        logger.warning(
            "Generation failure streak: %d/%d (model='%s', error=%s)",
            failure_count,
            abort_on,
            model_name,
            error_repr,
        )

        if failure_count >= abort_on:
            message = (
                "Aborting run: exceeded global consecutive generation failure threshold. "
                f"Failures in a row: {failure_count} (threshold={_THRESHOLD}). "
                f"Last failing model: '{model_name}'. Last error: {error_repr}"
            )
            logger.critical(message)
            raise ConsecutiveGenerationFailuresAbort(message)
