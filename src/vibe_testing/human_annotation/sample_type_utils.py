"""Shared helpers for sample-type selection and balancing."""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from src.vibe_testing.model_names import canonicalize_model_name
from src.vibe_testing.pathing import canonicalize_pairwise_model_routing


def candidate_field_names() -> set[str]:
    """
    List candidate fields that may be used in selection and balancing configs.

    Args:
        None.

    Returns:
        set[str]: Supported candidate field names.
    """
    return {
        "source_key",
        "artifact_path",
        "artifact_index",
        "persona",
        "prompt_type",
        "pairwise_judgment_type",
        "judge_dir_name",
        "judge_model_name",
        "generator_model",
        "filter_model",
        "model_a_name",
        "model_b_name",
        "model_pair",
        "task_id",
        "variant_id",
        "raw_task_id",
        "input_text",
        "model_a_output",
        "model_b_output",
        "overall_winner",
    }


def validate_candidate_fields(fields: Sequence[str], field_name: str) -> list[str]:
    """
    Validate and normalize a sequence of candidate field names.

    Args:
        fields (Sequence[str]): Candidate field names.
        field_name (str): Parent config field path for error messages.

    Returns:
        list[str]: Normalized field-name list.

    Raises:
        ValueError: If any field is unsupported.
    """
    valid_fields = candidate_field_names()
    normalized = [str(field) for field in fields]
    invalid = [field for field in normalized if field not in valid_fields]
    if invalid:
        raise ValueError(
            f"{field_name} contains unsupported candidate fields: {sorted(invalid)}"
        )
    return normalized


def normalize_candidate_field_value(field_name: str, value: object) -> str:
    """
    Normalize a candidate field value into the comparison namespace used by selectors.

    Args:
        field_name (str): Candidate field name.
        value (object): Raw field value.

    Returns:
        str: Comparable normalized value.

    Raises:
        ValueError: If a `model_pair` selector is malformed.
    """
    text = str(value)
    if field_name in {"model_a_name", "model_b_name"}:
        return canonicalize_model_name(text)
    if field_name == "model_pair":
        if "_vs_" not in text:
            raise ValueError(
                "Sample-type field 'model_pair' must use 'model_a_vs_model_b' format."
            )
        model_a_name, model_b_name = text.split("_vs_", 1)
        return canonicalize_pairwise_model_routing(
            model_a_name, model_b_name
        ).canonical_model_pair
    return text


def normalize_allowed_field_values(
    field_name: str, values: Sequence[object]
) -> list[str]:
    """
    Normalize allowed selector values for one candidate field.

    Args:
        field_name (str): Candidate field name.
        values (Sequence[object]): Raw allowed values.

    Returns:
        list[str]: Normalized comparable values.
    """
    return [
        normalize_candidate_field_value(field_name, value) for value in values
    ]


def candidate_field_value(candidate: Any, field_name: str) -> str:
    """
    Read a comparable string field from a candidate-like object.

    Args:
        candidate (Any): Candidate-like object.
        field_name (str): Field name to read.

    Returns:
        str: Comparable field value.

    Raises:
        ValueError: If the field is not present.
    """
    if not hasattr(candidate, field_name):
        raise ValueError(f"Unknown candidate field requested: {field_name}")
    return normalize_candidate_field_value(field_name, getattr(candidate, field_name))


def normalize_field_map(where: Dict[str, Sequence[object]]) -> Dict[str, list[str]]:
    """
    Normalize a selector field map into the comparison namespace used by candidates.

    Args:
        where (Dict[str, Sequence[object]]): Raw selector field map.

    Returns:
        Dict[str, list[str]]: Normalized field map.
    """
    return {
        field_name: normalize_allowed_field_values(field_name, allowed_values)
        for field_name, allowed_values in where.items()
    }


def field_map_signature(
    where: Dict[str, Sequence[object]],
) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """
    Build a hashable canonical signature for a selector field map.

    Args:
        where (Dict[str, Sequence[object]]): Selector field map.

    Returns:
        Tuple[Tuple[str, Tuple[str, ...]], ...]: Canonical signature.
    """
    normalized_where = normalize_field_map(where)
    return tuple(
        sorted(
            (field_name, tuple(normalized_where[field_name]))
            for field_name in normalized_where
        )
    )


def field_map_label(where: Dict[str, Sequence[object]]) -> str:
    """
    Build a stable readable label for a selector field map.

    Args:
        where (Dict[str, Sequence[object]]): Selector field map.

    Returns:
        str: Stable label such as `persona=A::prompt_type=original`.
    """
    normalized_where = normalize_field_map(where)
    return "::".join(
        f"{field_name}={','.join(normalized_where[field_name])}"
        for field_name in sorted(normalized_where)
    )


def matches_field_map(candidate: Any, where: Dict[str, Sequence[str]]) -> bool:
    """
    Check whether a candidate matches a field-value map.

    Args:
        candidate (Any): Candidate-like object.
        where (Dict[str, Sequence[str]]): Allowed values by field.

    Returns:
        bool: True when all fields match.
    """
    normalized_where = normalize_field_map(where)
    for field_name, allowed_values in normalized_where.items():
        if candidate_field_value(candidate, field_name) not in set(allowed_values):
            return False
    return True


def sample_type_key(candidate: Any, fields: Sequence[str]) -> Tuple[str, ...]:
    """
    Build a stable sample-type tuple from configured fields.

    Args:
        candidate (Any): Candidate-like object.
        fields (Sequence[str]): Fields that define sample type.

    Returns:
        Tuple[str, ...]: Stable sample-type tuple.
    """
    return tuple(candidate_field_value(candidate, field_name) for field_name in fields)


def sample_type_map(candidate: Any, fields: Sequence[str]) -> Dict[str, str]:
    """
    Build a field-value map for the configured sample type.

    Args:
        candidate (Any): Candidate-like object.
        fields (Sequence[str]): Fields that define sample type.

    Returns:
        Dict[str, str]: Field-value sample-type mapping.
    """
    return {
        field_name: candidate_field_value(candidate, field_name) for field_name in fields
    }


def sample_type_label(candidate: Any, fields: Sequence[str]) -> str:
    """
    Build a readable stable sample-type label.

    Args:
        candidate (Any): Candidate-like object.
        fields (Sequence[str]): Fields that define sample type.

    Returns:
        str: Stable label like `persona=A::prompt_type=original`.
    """
    return "::".join(
        f"{field_name}={candidate_field_value(candidate, field_name)}"
        for field_name in fields
    )
