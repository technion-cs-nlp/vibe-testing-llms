"""Schemas for standalone human annotation workflows."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.vibe_testing.analysis.io import PAIRWISE_DIMENSIONS
from src.vibe_testing.human_annotation.sample_type_utils import (
    candidate_field_names,
    field_map_label,
    field_map_signature,
    normalize_allowed_field_values,
    normalize_field_map,
    validate_candidate_fields,
)

logger = logging.getLogger(__name__)


def _json_ready(value: Any) -> Any:
    """
    Recursively convert dataclass-friendly values into JSON-friendly values.

    Args:
        value (Any): Arbitrary Python object.

    Returns:
        Any: JSON-friendly representation.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _ensure_list(value: Any, field_name: str) -> List[str]:
    """
    Normalize a scalar-or-list value into a list of strings.

    Args:
        value (Any): Source value from configuration.
        field_name (str): Field name used for error messages.

    Returns:
        List[str]: Normalized string list.

    Raises:
        ValueError: If the input type is unsupported.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [value]
    raise ValueError(f"{field_name} must be a string or list of strings.")


def _resolve_filter_alias(
    data: Dict[str, Any],
    canonical_key: str,
    legacy_key: str,
    default: Any,
) -> Any:
    """
    Resolve a filter-config value from canonical and legacy keys.

    Args:
        data (Dict[str, Any]): Raw filter config block.
        canonical_key (str): Preferred config key.
        legacy_key (str): Backward-compatible legacy key.
        default (Any): Default value when neither key is provided.

    Returns:
        Any: Resolved config value.

    Raises:
        ValueError: If canonical and legacy keys are both provided with different values.
    """
    has_canonical = canonical_key in data
    has_legacy = legacy_key in data
    if has_canonical and has_legacy and data[canonical_key] != data[legacy_key]:
        raise ValueError(
            f"filters.{canonical_key} and filters.{legacy_key} were both provided "
            "with different values. Please keep only one."
        )
    if has_canonical:
        return data[canonical_key]
    if has_legacy:
        logger.warning(
            "filters.%s is deprecated; please use filters.%s instead.",
            legacy_key,
            canonical_key,
        )
        return data[legacy_key]
    return default


def _candidate_filter_field_names() -> set[str]:
    """
    List candidate fields that may appear in field-based filter configs.

    Args:
        None.

    Returns:
        set[str]: Supported candidate field names.
    """
    return candidate_field_names()


@dataclass(frozen=True)
class SamplingTarget:
    """
    Explicit quota for a subset of candidate records.

    Attributes:
        name: Human-readable label for the target.
        where: Equality filters applied to candidate fields.
        n_samples: Number of samples to draw for this target.
    """

    name: str
    where: Dict[str, List[str]]
    n_samples: int

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SamplingTarget":
        """
        Build a sampling target from raw config data.

        Args:
            data (Dict[str, Any]): Raw YAML object.

        Returns:
            SamplingTarget: Parsed target definition.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        name = str(data.get("name") or "unnamed_target")
        raw_where = data.get("where") or {}
        if not isinstance(raw_where, dict) or not raw_where:
            raise ValueError(
                "Each allocation target must define a non-empty 'where' map."
            )
        where: Dict[str, List[str]] = {}
        for key, value in raw_where.items():
            where[str(key)] = _ensure_list(
                value, f"allocation.targets[{name}].where.{key}"
            )
        n_samples = int(data.get("n_samples", 0))
        if n_samples < 1:
            raise ValueError(
                f"allocation target '{name}' must set n_samples >= 1, got {n_samples}."
            )
        return cls(name=name, where=where, n_samples=n_samples)


@dataclass(frozen=True)
class SourceConfig:
    """
    Configuration for discovering source pairwise artifacts.

    Attributes:
        base_results_dir: Root results directory containing persona folders.
        personas: Optional persona allowlist.
        prompt_types: Optional prompt-type allowlist.
        model_pairs: Optional model-pair allowlist in `a_vs_b` form.
        judges: Optional judge allowlist.
        pairwise_judgment_types: Optional pairwise judgment-type allowlist.
    """

    base_results_dir: Path
    personas: List[str] = field(default_factory=list)
    prompt_types: List[str] = field(default_factory=list)
    model_pairs: List[str] = field(default_factory=list)
    judges: List[str] = field(default_factory=list)
    pairwise_judgment_types: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SourceConfig":
        """
        Parse source discovery config.

        Args:
            data (Dict[str, Any]): Raw YAML object.

        Returns:
            SourceConfig: Parsed source configuration.

        Raises:
            ValueError: If the base directory is missing.
        """
        base_results_dir = data.get("base_results_dir")
        if not base_results_dir:
            raise ValueError("source.base_results_dir is required.")
        return cls(
            base_results_dir=Path(str(base_results_dir)).expanduser(),
            personas=_ensure_list(data.get("personas"), "source.personas"),
            prompt_types=_ensure_list(data.get("prompt_types"), "source.prompt_types"),
            model_pairs=_ensure_list(data.get("model_pairs"), "source.model_pairs"),
            judges=_ensure_list(data.get("judges"), "source.judges"),
            pairwise_judgment_types=_ensure_list(
                data.get("pairwise_judgment_types"),
                "source.pairwise_judgment_types",
            ),
        )


@dataclass(frozen=True)
class MaxOutputCharsOverride:
    """
    Candidate-scoped override for the response-length filter.

    Attributes:
        where: Candidate-field equality filters used to match a subset.
        max_output_chars: Maximum allowed output length for matching candidates.
    """

    where: Dict[str, List[str]]
    max_output_chars: int

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], field_name: str
    ) -> "MaxOutputCharsOverride":
        """
        Parse one max-output-chars override rule from YAML.

        Args:
            data (Dict[str, Any]): Raw override mapping.
            field_name (str): Parent field path for error messages.

        Returns:
            MaxOutputCharsOverride: Parsed override rule.

        Raises:
            ValueError: If the override shape or values are invalid.
        """
        if not isinstance(data, dict):
            raise ValueError(f"{field_name} must be a mapping.")
        raw_where = data.get("where") or {}
        if not isinstance(raw_where, dict) or not raw_where:
            raise ValueError(f"{field_name}.where must be a non-empty mapping.")
        where: Dict[str, List[str]] = {}
        valid_fields = _candidate_filter_field_names()
        for key, value in raw_where.items():
            key_text = str(key)
            if key_text not in valid_fields:
                raise ValueError(
                    f"{field_name}.where.{key_text} is not a valid candidate field."
                )
            where[key_text] = _ensure_list(value, f"{field_name}.where.{key_text}")
        max_output_chars = data.get("max_output_chars")
        if max_output_chars is None:
            raise ValueError(f"{field_name}.max_output_chars is required.")
        max_output_chars = int(max_output_chars)
        if max_output_chars < 1:
            raise ValueError(f"{field_name}.max_output_chars must be >= 1.")
        return cls(where=where, max_output_chars=max_output_chars)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the override into JSON-friendly data.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serializable override payload.
        """
        return {
            "where": _json_ready(self.where),
            "max_output_chars": self.max_output_chars,
        }


@dataclass(frozen=True)
class SampleTypeSpec:
    """
    Exact sample-type selector expressed as candidate-field equality filters.

    Attributes:
        where: Candidate field values that define the sample type.
    """

    where: Dict[str, List[str]]

    @property
    def label(self) -> str:
        """
        Build a stable readable label for this selector.

        Args:
            None.

        Returns:
            str: Canonical selector label.
        """
        return field_map_label(self.where)

    @property
    def signature(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """
        Build a hashable canonical signature for this selector.

        Args:
            None.

        Returns:
            tuple[tuple[str, tuple[str, ...]], ...]: Canonical selector signature.
        """
        return field_map_signature(self.where)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], field_name: str) -> "SampleTypeSpec":
        """
        Parse one exact sample-type selector from YAML.

        Args:
            data (Dict[str, Any]): Raw selector mapping.
            field_name (str): Parent field path for error messages.

        Returns:
            SampleTypeSpec: Parsed selector.

        Raises:
            ValueError: If the selector is empty or contains invalid fields.
        """
        if not isinstance(data, dict):
            raise ValueError(f"{field_name} must be a mapping.")
        raw_where = data.get("where", data)
        if not isinstance(raw_where, dict) or not raw_where:
            raise ValueError(f"{field_name} must define at least one candidate field.")
        where: Dict[str, List[str]] = {}
        for key, value in raw_where.items():
            key_text = str(key)
            if key_text not in _candidate_filter_field_names():
                raise ValueError(
                    f"{field_name}.{key_text} is not a valid candidate field."
                )
            where[key_text] = normalize_allowed_field_values(
                key_text,
                _ensure_list(value, f"{field_name}.{key_text}"),
            )
        return cls(where=where)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the selector into a JSON-friendly mapping.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serializable selector payload.
        """
        return _json_ready(normalize_field_map(self.where))


@dataclass(frozen=True)
class FilterConfig:
    """
    Filtering rules applied before sampling.

    Attributes:
        max_output_chars: Optional maximum length allowed for both compared outputs.
        max_output_chars_overrides: Ordered per-subset overrides for the output-length
            filter. When multiple rules match, later rules win.
        require_no_overall_ties: When true, reject records with tied overall winner.
        require_no_dimension_ties: When true, reject records with any tied dimension.
        require_judge_consensus: When true, apply per-item judge consensus filtering.
        min_judges_in_pool: Minimum number of judge rows that must remain in the
            agreement pool before consensus is evaluated.
        min_consensus_judges: Minimum number of judges that must share the same
            agreement signature. When omitted, all judges in the agreement pool
            must agree.
        consensus_scope: Agreement signature type. ``"full_signature"``
            requires matching overall winner plus per-dimension winners.
            ``"overall_winner"`` ignores per-dimension mismatches.
        exclude_tied_overall_from_consensus: When true, tied/empty overall-winner
            judgments are removed from the agreement pool before counting consensus.
        exclude_prior_selection_plans: When true, remove items already present in
            prior `selection_plan.csv` files before sampling.
        prior_selection_plan_paths: Prior `selection_plan.csv` paths consulted when
            `exclude_prior_selection_plans` is enabled.
        include_sample_types: Exact sample-type selectors applied before sampling.
        include_prior_sample_ids: Explicit repeated `source_key` values to include.
        include_from_selection_plan_paths: Prior `selection_plan.csv` paths whose
            `source_key`s should be repeated.
        include_from_manifest_paths: Prior study manifests whose sampled `source_key`s
            should be repeated.
        repeat_mode: Repeat-inclusion policy. `include_only` restricts the eligible
            pool to repeated items only; `pin_and_fill` guarantees repeated items are
            included first, then fills the remaining quota with fresh items.
        allow: Field-based allowlist filters.
        deny: Field-based denylist filters.
    """

    max_output_chars: Optional[int] = None
    max_output_chars_overrides: List[MaxOutputCharsOverride] = field(
        default_factory=list
    )
    require_no_overall_ties: bool = False
    require_no_dimension_ties: bool = False
    require_judge_consensus: bool = False
    min_judges_in_pool: int = 2
    min_consensus_judges: Optional[int] = None
    consensus_scope: str = "full_signature"
    exclude_tied_overall_from_consensus: bool = False
    exclude_prior_selection_plans: bool = False
    prior_selection_plan_paths: List[Path] = field(default_factory=list)
    include_sample_types: List[SampleTypeSpec] = field(default_factory=list)
    include_prior_sample_ids: List[str] = field(default_factory=list)
    include_from_selection_plan_paths: List[Path] = field(default_factory=list)
    include_from_manifest_paths: List[Path] = field(default_factory=list)
    repeat_mode: str = "include_only"
    allow: Dict[str, List[str]] = field(default_factory=dict)
    deny: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterConfig":
        """
        Parse filtering configuration from YAML.

        Args:
            data (Dict[str, Any]): Raw YAML object.

        Returns:
            FilterConfig: Parsed filter configuration.
        """
        allow: Dict[str, List[str]] = {}
        deny: Dict[str, List[str]] = {}
        for name, target in (("allow", allow), ("deny", deny)):
            raw_block = data.get(name) or {}
            if not isinstance(raw_block, dict):
                raise ValueError(f"filters.{name} must be a mapping.")
            for key, value in raw_block.items():
                target[str(key)] = _ensure_list(value, f"filters.{name}.{key}")
        max_output_chars = data.get("max_output_chars")
        if max_output_chars is not None:
            max_output_chars = int(max_output_chars)
            if max_output_chars < 1:
                raise ValueError("filters.max_output_chars must be >= 1 when provided.")
        raw_max_output_chars_overrides = data.get("max_output_chars_overrides") or []
        if not isinstance(raw_max_output_chars_overrides, list):
            raise ValueError("filters.max_output_chars_overrides must be a list.")
        max_output_chars_overrides = [
            MaxOutputCharsOverride.from_dict(
                item,
                f"filters.max_output_chars_overrides[{index}]",
            )
            for index, item in enumerate(raw_max_output_chars_overrides)
        ]
        raw_include_sample_types = data.get("include_sample_types") or []
        if not isinstance(raw_include_sample_types, list):
            raise ValueError("filters.include_sample_types must be a list.")
        include_sample_types = [
            SampleTypeSpec.from_dict(item, f"filters.include_sample_types[{index}]")
            for index, item in enumerate(raw_include_sample_types)
        ]
        min_judges_in_pool = int(
            _resolve_filter_alias(
                data,
                canonical_key="min_judges_in_pool",
                legacy_key="min_judges_for_agreement",
                default=2,
            )
        )
        if min_judges_in_pool < 1:
            raise ValueError("filters.min_judges_in_pool must be >= 1.")
        min_consensus_judges = _resolve_filter_alias(
            data,
            canonical_key="min_consensus_judges",
            legacy_key="min_agreeing_judges",
            default=None,
        )
        if min_consensus_judges is not None:
            min_consensus_judges = int(min_consensus_judges)
            if min_consensus_judges < 1:
                raise ValueError("filters.min_consensus_judges must be >= 1.")
        consensus_scope = str(
            _resolve_filter_alias(
                data,
                canonical_key="consensus_scope",
                legacy_key="judge_agreement_scope",
                default="full_signature",
            )
        ).strip()
        if consensus_scope not in {"full_signature", "overall_winner"}:
            raise ValueError(
                "filters.consensus_scope must be one of "
                "['full_signature', 'overall_winner']."
            )
        repeat_mode = str(data.get("repeat_mode") or "include_only").strip()
        if repeat_mode not in {"include_only", "pin_and_fill"}:
            raise ValueError(
                "filters.repeat_mode must be one of ['include_only', 'pin_and_fill']."
            )
        return cls(
            max_output_chars=max_output_chars,
            max_output_chars_overrides=max_output_chars_overrides,
            require_no_overall_ties=bool(data.get("require_no_overall_ties", False)),
            require_no_dimension_ties=bool(
                data.get("require_no_dimension_ties", False)
            ),
            require_judge_consensus=bool(
                _resolve_filter_alias(
                    data,
                    canonical_key="require_judge_consensus",
                    legacy_key="require_all_judges_agree",
                    default=False,
                )
            ),
            min_judges_in_pool=min_judges_in_pool,
            min_consensus_judges=min_consensus_judges,
            consensus_scope=consensus_scope,
            exclude_tied_overall_from_consensus=bool(
                _resolve_filter_alias(
                    data,
                    canonical_key="exclude_tied_overall_from_consensus",
                    legacy_key="exclude_tied_judges_for_agreement",
                    default=False,
                )
            ),
            exclude_prior_selection_plans=bool(
                data.get("exclude_prior_selection_plans", False)
            ),
            prior_selection_plan_paths=[
                Path(str(path)).expanduser()
                for path in _ensure_list(
                    data.get("prior_selection_plan_paths"),
                    "filters.prior_selection_plan_paths",
                )
            ],
            include_sample_types=include_sample_types,
            include_prior_sample_ids=_ensure_list(
                data.get("include_prior_sample_ids"),
                "filters.include_prior_sample_ids",
            ),
            include_from_selection_plan_paths=[
                Path(str(path)).expanduser()
                for path in _ensure_list(
                    data.get("include_from_selection_plan_paths"),
                    "filters.include_from_selection_plan_paths",
                )
            ],
            include_from_manifest_paths=[
                Path(str(path)).expanduser()
                for path in _ensure_list(
                    data.get("include_from_manifest_paths"),
                    "filters.include_from_manifest_paths",
                )
            ],
            repeat_mode=repeat_mode,
            allow=allow,
            deny=deny,
        )

    @property
    def require_all_judges_agree(self) -> bool:
        """Backward-compatible alias for `require_judge_consensus`."""
        return self.require_judge_consensus

    @property
    def min_judges_for_agreement(self) -> int:
        """Backward-compatible alias for `min_judges_in_pool`."""
        return self.min_judges_in_pool

    @property
    def min_agreeing_judges(self) -> Optional[int]:
        """Backward-compatible alias for `min_consensus_judges`."""
        return self.min_consensus_judges

    @property
    def judge_agreement_scope(self) -> str:
        """Backward-compatible alias for `consensus_scope`."""
        return self.consensus_scope

    @property
    def exclude_tied_judges_for_agreement(self) -> bool:
        """Backward-compatible alias for `exclude_tied_overall_from_consensus`."""
        return self.exclude_tied_overall_from_consensus

    def to_public_dict(self) -> Dict[str, Any]:
        """
        Serialize the filter config using canonical public field names.

        Args:
            None.

        Returns:
            Dict[str, Any]: JSON-friendly filter config payload.
        """
        return {
            "max_output_chars": self.max_output_chars,
            "max_output_chars_overrides": [
                override.to_dict() for override in self.max_output_chars_overrides
            ],
            "require_no_overall_ties": self.require_no_overall_ties,
            "require_no_dimension_ties": self.require_no_dimension_ties,
            "require_judge_consensus": self.require_judge_consensus,
            "min_judges_in_pool": self.min_judges_in_pool,
            "min_consensus_judges": self.min_consensus_judges,
            "consensus_scope": self.consensus_scope,
            "exclude_tied_overall_from_consensus": (
                self.exclude_tied_overall_from_consensus
            ),
            "exclude_prior_selection_plans": self.exclude_prior_selection_plans,
            "prior_selection_plan_paths": _json_ready(self.prior_selection_plan_paths),
            "include_sample_types": [
                selector.to_dict() for selector in self.include_sample_types
            ],
            "include_prior_sample_ids": list(self.include_prior_sample_ids),
            "include_from_selection_plan_paths": _json_ready(
                self.include_from_selection_plan_paths
            ),
            "include_from_manifest_paths": _json_ready(
                self.include_from_manifest_paths
            ),
            "repeat_mode": self.repeat_mode,
            "allow": _json_ready(self.allow),
            "deny": _json_ready(self.deny),
            "legacy_aliases": {
                "require_all_judges_agree": self.require_judge_consensus,
                "min_judges_for_agreement": self.min_judges_in_pool,
                "min_agreeing_judges": self.min_consensus_judges,
                "judge_agreement_scope": self.consensus_scope,
                "exclude_tied_judges_for_agreement": (
                    self.exclude_tied_overall_from_consensus
                ),
            },
        }


@dataclass(frozen=True)
class AllocationConfig:
    """
    Sampling-allocation rules.

    Attributes:
        random_seed: Seed used for deterministic sampling.
        stride: Keep every Nth row after stable sorting.
        stride_offset: Starting offset applied before stride selection.
        total_samples: Optional global number of samples to draw.
        equal_allocation_by: Optional fields whose full joint cross-product strata
            must receive equal counts in the final sampled subset. This guarantees
            balance within each relevant subset, and sampling fails loudly when the
            eligible source-item pool is missing required strata or cannot satisfy
            the per-stratum quota.
        targets: Optional explicit per-cell quotas.
    """

    random_seed: int = 0
    stride: int = 1
    stride_offset: int = 0
    total_samples: Optional[int] = None
    equal_allocation_by: List[str] = field(default_factory=list)
    targets: List[SamplingTarget] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllocationConfig":
        """
        Parse allocation configuration from YAML.

        Args:
            data (Dict[str, Any]): Raw YAML object.

        Returns:
            AllocationConfig: Parsed allocation configuration.

        Raises:
            ValueError: If the allocation block is inconsistent.
        """
        stride = int(data.get("stride", 1))
        if stride < 1:
            raise ValueError("allocation.stride must be >= 1.")
        stride_offset = int(data.get("stride_offset", 0))
        if stride_offset < 0:
            raise ValueError("allocation.stride_offset must be >= 0.")
        total_samples = data.get("total_samples")
        if total_samples is not None:
            total_samples = int(total_samples)
            if total_samples < 1:
                raise ValueError("allocation.total_samples must be >= 1 when provided.")
        equal_allocation_by = _ensure_list(
            data.get("equal_allocation_by"), "allocation.equal_allocation_by"
        )
        raw_targets = data.get("targets") or []
        if not isinstance(raw_targets, list):
            raise ValueError("allocation.targets must be a list when provided.")
        targets = [SamplingTarget.from_dict(item) for item in raw_targets]
        if total_samples is None and not targets:
            raise ValueError(
                "allocation must define either total_samples or at least one explicit target."
            )
        return cls(
            random_seed=int(data.get("random_seed", 0)),
            stride=stride,
            stride_offset=stride_offset,
            total_samples=total_samples,
            equal_allocation_by=equal_allocation_by,
            targets=targets,
        )


@dataclass(frozen=True)
class AnnotatorConfig:
    """
    Annotator assignment settings.

    Attributes:
        annotator_ids: Explicit annotator identifiers.
        items_per_annotator: Optional target count per annotator.
        anchor_count: Number of shared anchor items assigned to every annotator.
        overlap_count: Additional shared overlap items assigned to every annotator.
        annotators_per_regular_item: Number of annotators that should receive each
            non-shared regular item. The legacy behavior is `1`.
        calibration_sample_types: Exact selectors used for calibration items that
            are sampled separately from the regular packet template.
        excluded_manifest_paths: Prior manifest files whose source items should be excluded.
        balance_by: Candidate fields that define packet-level sample types.
        balance_scope: Whether balancing considers only unique items or the full packet.
        balance_mode: Whether infeasible packet balancing should fail loudly or degrade
            to best-effort assignment.
    """

    annotator_ids: List[str]
    items_per_annotator: Optional[int] = None
    anchor_count: int = 0
    overlap_count: int = 0
    annotators_per_regular_item: int = 1
    calibration_sample_types: List[SampleTypeSpec] = field(default_factory=list)
    excluded_manifest_paths: List[Path] = field(default_factory=list)
    balance_by: List[str] = field(default_factory=list)
    balance_scope: str = "unique_only"
    balance_mode: str = "strict"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotatorConfig":
        """
        Parse annotator assignment configuration.

        Args:
            data (Dict[str, Any]): Raw YAML object.

        Returns:
            AnnotatorConfig: Parsed annotator configuration.

        Raises:
            ValueError: If annotator IDs are missing.
        """
        annotator_ids = _ensure_list(
            data.get("annotator_ids"), "annotators.annotator_ids"
        )
        if not annotator_ids:
            raise ValueError(
                "annotators.annotator_ids must contain at least one annotator."
            )
        items_per_annotator = data.get("items_per_annotator")
        if items_per_annotator is not None:
            items_per_annotator = int(items_per_annotator)
            if items_per_annotator < 1:
                raise ValueError("annotators.items_per_annotator must be >= 1.")
        annotators_per_regular_item = int(data.get("annotators_per_regular_item", 1))
        if annotators_per_regular_item < 1:
            raise ValueError("annotators.annotators_per_regular_item must be >= 1.")
        balance_by = _ensure_list(data.get("balance_by"), "annotators.balance_by")
        if balance_by:
            balance_by = validate_candidate_fields(balance_by, "annotators.balance_by")
        raw_calibration_sample_types = data.get("calibration_sample_types") or []
        if not isinstance(raw_calibration_sample_types, list):
            raise ValueError("annotators.calibration_sample_types must be a list.")
        calibration_sample_types = [
            SampleTypeSpec.from_dict(
                selector, f"annotators.calibration_sample_types[{index}]"
            )
            for index, selector in enumerate(raw_calibration_sample_types)
        ]
        balance_scope = str(data.get("balance_scope") or "unique_only").strip()
        if balance_scope not in {"unique_only", "all_items"}:
            raise ValueError(
                "annotators.balance_scope must be one of ['unique_only', 'all_items']."
            )
        balance_mode = str(data.get("balance_mode") or "strict").strip()
        if balance_mode not in {"strict", "best_effort"}:
            raise ValueError(
                "annotators.balance_mode must be one of ['strict', 'best_effort']."
            )
        return cls(
            annotator_ids=annotator_ids,
            items_per_annotator=items_per_annotator,
            anchor_count=int(data.get("anchor_count", 0)),
            overlap_count=int(data.get("overlap_count", 0)),
            annotators_per_regular_item=annotators_per_regular_item,
            calibration_sample_types=calibration_sample_types,
            excluded_manifest_paths=[
                Path(str(path)).expanduser()
                for path in _ensure_list(
                    data.get("excluded_manifest_paths"),
                    "annotators.excluded_manifest_paths",
                )
            ],
            balance_by=balance_by,
            balance_scope=balance_scope,
            balance_mode=balance_mode,
        )


@dataclass(frozen=True)
class QualtricsConfig:
    """
    Qualtrics rendering configuration.

    Attributes:
        survey_title: Survey title used in generated text files.
        intro_text: Short introductory participant instructions (legacy/simple flow).
        opening_per_item_text: Optional text shown at the top of each comparison page
            (before the persona and prompt).
        task_overview_text: Longer task instructions shown on an explanation page.
        toy_example_prompt: Toy example prompt shown on the explanation page.
        toy_example_response_a: Toy example response A shown on the explanation page.
        toy_example_response_b: Toy example response B shown on the explanation page.
        dimension_short_reminders: Optional one-sentence reminders per dimension.
            When omitted, reminders are derived from the first line of the judge guidance.
        render_responses_side_by_side: When true, try rendering Response A/B next to
            each other; otherwise fall back to stacked boxes.
        outro_text: Final completion message.
        dimensions: Pairwise dimensions to ask about for each item.
        include_confidence_question: Whether to add an overall confidence question.
        include_rationale_question: Whether to add an overall rationale text box.
    """

    survey_title: str
    intro_text: str
    outro_text: str
    dimensions: List[str] = field(default_factory=lambda: list(PAIRWISE_DIMENSIONS))
    include_confidence_question: bool = True
    include_rationale_question: bool = True
    opening_per_item_text: str = ""
    task_overview_text: str = ""
    toy_example_prompt: str = ""
    toy_example_response_a: str = ""
    toy_example_response_b: str = ""
    dimension_short_reminders: Dict[str, str] = field(default_factory=dict)
    render_responses_side_by_side: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QualtricsConfig":
        """
        Parse Qualtrics rendering settings.

        Args:
            data (Dict[str, Any]): Raw YAML object.

        Returns:
            QualtricsConfig: Parsed rendering configuration.
        """
        dimensions = _ensure_list(data.get("dimensions"), "qualtrics.dimensions")
        normalized_dimensions = dimensions or list(PAIRWISE_DIMENSIONS)
        unknown = sorted(set(normalized_dimensions) - set(PAIRWISE_DIMENSIONS))
        if unknown:
            raise ValueError(
                f"qualtrics.dimensions contains unsupported dimensions: {unknown}"
            )
        return cls(
            survey_title=str(
                data.get("survey_title") or "Pairwise Coding Output Study"
            ),
            intro_text=str(
                data.get("intro_text")
                or "Compare Response A and Response B for each question."
            ),
            opening_per_item_text=str(data.get("opening_per_item_text") or ""),
            task_overview_text=str(
                data.get("task_overview_text")
                or "You will see a prompt, a persona, and two responses (A and B).\n"
                "First, choose which response is better overall for that persona.\n"
                "Then, choose which response is better for each vibe dimension.\n"
                "Choose 'Tie' only if the two responses are genuinely equivalent."
            ),
            toy_example_prompt=str(
                data.get("toy_example_prompt")
                or "Example prompt: “Write a Python function that returns the sum of two integers.”"
            ),
            toy_example_response_a=str(
                data.get("toy_example_response_a") or "def add(a, b):\n    return a + b"
            ),
            toy_example_response_b=str(
                data.get("toy_example_response_b")
                or "def add_numbers(x, y):\n    return x + y"
            ),
            dimension_short_reminders=dict(data.get("dimension_short_reminders") or {}),
            render_responses_side_by_side=bool(
                data.get("render_responses_side_by_side", True)
            ),
            outro_text=str(
                data.get("outro_text")
                or "Thank you for completing the annotation task."
            ),
            dimensions=normalized_dimensions,
            include_confidence_question=bool(
                data.get("include_confidence_question", True)
            ),
            include_rationale_question=bool(
                data.get("include_rationale_question", True)
            ),
        )


@dataclass(frozen=True)
class OutputConfig:
    """
    Output locations and export naming settings.

    Attributes:
        workspace_dir: Working directory for plans, manifests, and Qualtrics files.
        export_base_dir: Root results directory for exported human Stage-5b artifacts.
        study_name: Human-readable study label.
        judge_name_prefix: Prefix used for exported human judge identifiers.
    """

    workspace_dir: Path
    export_base_dir: Path
    study_name: str
    judge_name_prefix: str = "human"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputConfig":
        """
        Parse output configuration from YAML.

        Args:
            data (Dict[str, Any]): Raw YAML object.

        Returns:
            OutputConfig: Parsed output configuration.

        Raises:
            ValueError: If required paths are missing.
        """
        workspace_dir = data.get("workspace_dir")
        export_base_dir = data.get("export_base_dir")
        study_name = data.get("study_name")
        if not workspace_dir or not export_base_dir or not study_name:
            raise ValueError(
                "outputs.workspace_dir, outputs.export_base_dir, and outputs.study_name are required."
            )
        return cls(
            workspace_dir=Path(str(workspace_dir)).expanduser(),
            export_base_dir=Path(str(export_base_dir)).expanduser(),
            study_name=str(study_name),
            judge_name_prefix=str(data.get("judge_name_prefix") or "human"),
        )


@dataclass(frozen=True)
class HumanAnnotationConfig:
    """
    Top-level standalone human annotation configuration.

    Attributes:
        source: Source artifact discovery rules.
        filters: Candidate filtering rules.
        allocation: Sampling allocation rules.
        annotators: Annotator assignment settings.
        qualtrics: Qualtrics rendering settings.
        outputs: Output-path and naming settings.
        config_path: Original config path for provenance.
    """

    source: SourceConfig
    filters: FilterConfig
    allocation: AllocationConfig
    annotators: AnnotatorConfig
    qualtrics: QualtricsConfig
    outputs: OutputConfig
    config_path: Optional[Path] = None

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], config_path: Optional[Path] = None
    ) -> "HumanAnnotationConfig":
        """
        Build the top-level configuration from a YAML dictionary.

        Args:
            data (Dict[str, Any]): Raw YAML object.
            config_path (Optional[Path]): Source file path for provenance.

        Returns:
            HumanAnnotationConfig: Parsed configuration object.
        """
        return cls(
            source=SourceConfig.from_dict(data.get("source") or {}),
            filters=FilterConfig.from_dict(data.get("filters") or {}),
            allocation=AllocationConfig.from_dict(data.get("allocation") or {}),
            annotators=AnnotatorConfig.from_dict(data.get("annotators") or {}),
            qualtrics=QualtricsConfig.from_dict(data.get("qualtrics") or {}),
            outputs=OutputConfig.from_dict(data.get("outputs") or {}),
            config_path=config_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the configuration into plain Python objects.

        Args:
            None.

        Returns:
            Dict[str, Any]: Serializable configuration payload.
        """
        payload = _json_ready(asdict(self))
        payload["filters"] = self.filters.to_public_dict()
        if self.config_path is not None:
            payload["config_path"] = str(self.config_path)
        return payload


@dataclass(frozen=True)
class PairwiseCandidateRecord:
    """
    Normalized pairwise item discovered from a Stage-5b artifact.

    Attributes:
        source_key: Stable identity key for cross-judge grouping.
        artifact_path: Source artifact file path.
        artifact_index: Record index inside the source file.
        persona: Source persona.
        prompt_type: Source prompt type.
        pairwise_judgment_type: Source judgment type.
        judge_dir_name: Judge token from the directory name.
        judge_model_name: Judge identity stored in the record payload.
        generator_model: Generator model token.
        filter_model: Filter model token.
        model_a_name: Canonical model A name from the record.
        model_b_name: Canonical model B name from the record.
        model_pair: Stable `a_vs_b` pair token.
        task_id: Base task identifier.
        variant_id: Variant-specific identifier.
        raw_task_id: Original task identifier from the artifact.
        input_text: Source prompt shown to the compared models.
        model_a_output: Output text from model A.
        model_b_output: Output text from model B.
        overall_winner: Stored overall winner value.
        dimension_results: Raw dimension-results object.
        win_counts: Stored win-count object.
        metadata: Additional source metadata.
    """

    source_key: str
    artifact_path: str
    artifact_index: int
    persona: str
    prompt_type: str
    pairwise_judgment_type: str
    judge_dir_name: str
    judge_model_name: str
    generator_model: str
    filter_model: str
    model_a_name: str
    model_b_name: str
    model_pair: str
    task_id: str
    variant_id: str
    raw_task_id: str
    input_text: str
    model_a_output: str
    model_b_output: str
    overall_winner: Optional[str]
    dimension_results: Dict[str, Dict[str, Any]]
    win_counts: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def output_char_len_a(self) -> int:
        """
        Get the length of model A output.

        Args:
            None.

        Returns:
            int: Character length.
        """
        return len(self.model_a_output or "")

    @property
    def output_char_len_b(self) -> int:
        """
        Get the length of model B output.

        Args:
            None.

        Returns:
            int: Character length.
        """
        return len(self.model_b_output or "")


@dataclass(frozen=True)
class SampledAnnotationItem:
    """
    Sampled item selected for human annotation.

    Attributes:
        item_id: Stable human-annotation item identifier.
        candidate: Source pairwise candidate record.
        selection_target: Allocation target or group name that selected the item.
        selection_rank: Position inside the selected sample list.
        selection_metadata: Trace metadata describing the sampling decision.
    """

    item_id: str
    candidate: PairwiseCandidateRecord
    selection_target: str
    selection_rank: int
    selection_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnnotatorAssignment:
    """
    Assignment of a sampled item to a specific annotator.

    Attributes:
        annotator_id: Annotator identifier.
        item: Sampled item assigned to this annotator.
        assignment_rank: Position in the annotator's task list.
        presentation_swapped: Whether the human form swaps A/B display positions.
        assignment_role: Role label such as `anchor`, `overlap`, or `unique`.
    """

    annotator_id: str
    item: SampledAnnotationItem
    assignment_rank: int
    presentation_swapped: bool
    assignment_role: str


@dataclass(frozen=True)
class ProcessedAnnotationRecord:
    """
    Normalized processed response for one annotator-item pair.

    Attributes:
        annotator_id: Annotator identifier written into processed outputs. This may
            be the manifest slot id or a runtime override provided during
            `process-results`.
        item_id: Human-annotation item identifier.
        task_id: Base task identifier.
        variant_id: Variant-specific identifier.
        raw_task_id: Original task identifier.
        persona: Source persona.
        prompt_type: Source prompt type.
        pairwise_judgment_type: Source pairwise judgment type.
        model_a_name: Canonical model A name.
        model_b_name: Canonical model B name.
        model_a_output: Source output for model A.
        model_b_output: Source output for model B.
        input_text: Source prompt text.
        overall_choice: Human overall choice in canonical `A`, `B`, or `tie` form.
        overall_confidence: Optional human confidence.
        overall_rationale: Optional human rationale.
        dimension_choices: Per-dimension canonical choices.
        metadata: Additional study metadata.
    """

    annotator_id: str
    item_id: str
    task_id: str
    variant_id: str
    raw_task_id: str
    persona: str
    prompt_type: str
    pairwise_judgment_type: str
    model_a_name: str
    model_b_name: str
    model_a_output: str
    model_b_output: str
    input_text: str
    overall_choice: str
    overall_confidence: Optional[str]
    overall_rationale: str
    dimension_choices: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StudyArtifactPaths:
    """
    Canonical file locations within a study workspace.

    Attributes:
        study_dir: Root workspace directory for the study.
        selection_manifest_path: JSON manifest for sampled items and assignments.
        selection_plan_csv_path: Flat CSV plan for auditing.
        filter_summary_path: JSON summary of rejected candidates.
        qualtrics_dir: Directory containing generated participant files.
        processed_json_path: JSON payload of processed annotations.
        processed_csv_path: CSV payload of processed annotations.
        stats_json_path: JSON summary of study statistics.
        analysis_summary_json_path: JSON summary of descriptive annotation analysis.
    """

    study_dir: Path
    selection_manifest_path: Path
    selection_plan_csv_path: Path
    filter_summary_path: Path
    qualtrics_dir: Path
    processed_json_path: Path
    processed_csv_path: Path
    stats_json_path: Path
    analysis_summary_json_path: Path
