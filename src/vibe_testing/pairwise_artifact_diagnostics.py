"""Shared diagnostics for pairwise artifact loading failures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class PairwiseArtifactLoadContext:
    """Structured provenance for a pairwise artifact load attempt."""

    artifact_path: str
    failure_stage: str
    record_index: Optional[int] = None
    record_count: Optional[int] = None
    payload_type: Optional[str] = None
    task_id: Optional[str] = None
    raw_task_id: Optional[str] = None
    model_a_name: Optional[str] = None
    model_b_name: Optional[str] = None
    judge_model_name: Optional[str] = None
    judge_token: Optional[str] = None
    persona: Optional[str] = None
    source_persona: Optional[str] = None
    prompt_type: Optional[str] = None
    pairwise_judgment_type: Optional[str] = None
    generator_model: Optional[str] = None
    filter_model: Optional[str] = None
    tie_breaker_mode: Optional[str] = None

    def merged(self, **updates: Optional[str]) -> "PairwiseArtifactLoadContext":
        """Return a copy with non-empty update fields applied."""

        normalized_updates = {
            key: value
            for key, value in updates.items()
            if value is not None and value != ""
        }
        if not normalized_updates:
            return self
        return replace(self, **normalized_updates)

    def as_dict(self) -> Dict[str, Any]:
        """Return the context as a plain dict."""

        return asdict(self)


class PairwiseArtifactLoadError(ValueError):
    """Raised when a discovered pairwise artifact cannot be loaded safely."""

    def __init__(
        self,
        *,
        context: PairwiseArtifactLoadContext,
        message: str,
        cause_type: Optional[str] = None,
        cause_message: Optional[str] = None,
    ) -> None:
        self.context = context
        self.message = str(message)
        self.cause_type = str(cause_type) if cause_type else None
        self.cause_message = str(cause_message) if cause_message else None
        super().__init__(self.render())

    def render(self) -> str:
        """Render the diagnostic as a bounded one-line message."""

        details = {
            "artifact": str(Path(self.context.artifact_path)),
            "failure_stage": self.context.failure_stage,
            "payload_type": self.context.payload_type,
            "record_count": self.context.record_count,
            "record_index": self.context.record_index,
            "task_id": self.context.task_id,
            "raw_task_id": self.context.raw_task_id,
            "model_a_name": self.context.model_a_name,
            "model_b_name": self.context.model_b_name,
            "judge_model_name": self.context.judge_model_name,
            "judge_token": self.context.judge_token,
            "persona": self.context.persona,
            "source_persona": self.context.source_persona,
            "prompt_type": self.context.prompt_type,
            "pairwise_judgment_type": self.context.pairwise_judgment_type,
            "generator_model": self.context.generator_model,
            "filter_model": self.context.filter_model,
            "tie_breaker_mode": self.context.tie_breaker_mode,
        }
        if self.cause_type:
            details["cause_type"] = self.cause_type
        if self.cause_message:
            details["cause_message"] = self.cause_message
        detail_text = " ".join(
            f"{key}={value!r}" for key, value in details.items() if value is not None
        )
        return f"{self.message} {detail_text}".strip()


def wrap_pairwise_artifact_load_error(
    exc: Exception,
    *,
    context: PairwiseArtifactLoadContext,
    message: str,
) -> PairwiseArtifactLoadError:
    """Normalize arbitrary exceptions into a structured pairwise load error."""

    if isinstance(exc, PairwiseArtifactLoadError):
        merged_context = exc.context.merged(**context.as_dict())
        combined_message = exc.message
        if message and message not in combined_message:
            combined_message = f"{message} {combined_message}"
        return PairwiseArtifactLoadError(
            context=merged_context,
            message=combined_message,
            cause_type=exc.cause_type,
            cause_message=exc.cause_message,
        )

    return PairwiseArtifactLoadError(
        context=context,
        message=message,
        cause_type=type(exc).__name__,
        cause_message=str(exc),
    )
