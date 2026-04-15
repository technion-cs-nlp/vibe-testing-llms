"""Qualtrics artifact generation for standalone human annotation studies."""

from __future__ import annotations

import html
import logging
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
import yaml

from src.vibe_testing.human_annotation.config import build_study_artifact_paths
from src.vibe_testing.human_annotation.filters import resolve_max_output_chars
from src.vibe_testing.human_annotation.schemas import (
    AnnotatorAssignment,
    FilterConfig,
    HumanAnnotationConfig,
    QualtricsConfig,
    SampledAnnotationItem,
)
from src.vibe_testing.subjective_evaluation.pairwise_judges import (
    get_dimension_guidance_text,
)
from src.vibe_testing.utils import save_json

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
OVERALL_PROMPT = "Which response better overall serves this user persona?"
MATRIX_PROMPT = (
    "Which response better serves this user persona for the following given aspects?"
)
CONFIDENCE_PROMPT = (
    "How confident are you in your preference decisions for this comparison?"
)
RATIONALE_PROMPT = "Briefly explain your preference decisions for this comparison."
_MATRIX_DIMENSION_DESCRIPTIONS = {
    "clarity": "How clear, structured, and readable the response is.",
    "tone_style_fit": (
        "Whether the writing style matches the user's preferences and the task."
    ),
    "workflow_fit": "How well the response fits the user's workflow.",
    "cognitive_load": (
        "How mentally taxing it is for this user to process and apply the response."
    ),
    "context_awareness": (
        "How well the response tracks the conversation and constraints from the prompt."
    ),
    "persona_consistency": (
        "Whether the response adheres to a specified role/persona when a persona "
        "is part of the evaluation setup."
    ),
    "anthropomorphism": "Perceived human-likeness versus roboticness.",
}


def _html_escape(text: str) -> str:
    """
    Escape untrusted text for HTML display.

    Args:
        text (str): Arbitrary text.

    Returns:
        str: HTML-escaped text.
    """
    return html.escape(text or "", quote=False)


def _html_with_breaks(text: str) -> str:
    """
    Render plain text as HTML using `<br>` for new lines while preserving
    leading indentation on each line.

    Args:
        text (str): Plain text.

    Returns:
        str: HTML string using `<br>` as line breaks.
    """
    rendered_lines: List[str] = []
    for raw_line in (text or "").splitlines():
        indent_parts: List[str] = []
        content_start = 0
        for content_start, char in enumerate(raw_line):
            if char == " ":
                indent_parts.append("&nbsp;")
                continue
            if char == "\t":
                indent_parts.append("&nbsp;" * 4)
                continue
            break
        else:
            content_start = len(raw_line)
        rendered_lines.append(
            "".join(indent_parts) + _html_escape(raw_line[content_start:])
        )
    if not rendered_lines:
        return ""
    return "<br>".join(rendered_lines)


def _render_underlined_title(title: str) -> str:
    """
    Render an underlined title using only Qualtrics-safe tags.

    Args:
        title (str): Section title.

    Returns:
        str: HTML string.
    """
    return f"<u><b>{_html_escape(title)}</b></u>"


def _render_section(title: str, body_text: str, *, italic_body: bool) -> str:
    """
    Render a titled section using only `<br>`, `<b>`, `<i>`, and `<u>`.

    Args:
        title (str): Section title.
        body_text (str): Plain text body.
        italic_body (bool): Whether to italicize the body.

    Returns:
        str: HTML string.
    """
    title_html = _render_underlined_title(title)
    body_html = _html_with_breaks(body_text)
    if italic_body:
        return "<br>".join([title_html, f"<i>{body_html}</i>"])
    return "<br>".join([title_html, body_html])


def _format_dimension_title(dimension: str) -> str:
    """
    Format a human-readable dimension title.

    Args:
        dimension (str): Dimension token.

    Returns:
        str: Title-cased display name.
    """
    return dimension.replace("_", " ").title()


def _normalized_after_measures(text: str) -> str:
    """
    Extract the explanation fragment after the word 'measures' from the first line.

    Args:
        text (str): Dimension guidance text.

    Returns:
        str: Normalized fragment (sentence-cased, with THIS user -> this user).
    """
    first_line = (text or "").splitlines()[0] if text else ""
    if "measures" not in first_line:
        return first_line.strip()
    after = first_line.split("measures", 1)[1].strip()
    # after = after.replace("THIS user", "this user")
    if not after:
        return after
    return after[0].upper() + after[1:]


def _dimension_reference_opening_line(guidance_text: str) -> str:
    """
    Build the opening line for the dimension reference page.

    Requirement: remove leading dimension name text, start with 'Measures', and
    keep the rest as-is (with THIS user normalized).

    Args:
        guidance_text (str): Full judge guidance text.

    Returns:
        str: Opening line HTML.
    """
    after = _normalized_after_measures(guidance_text)
    return f"Measures {after[0].lower() + after[1:] if after else ''}".strip()


def _dimension_short_reminder(dimension: str, guidance_text: str) -> str:
    """
    Build a short one-sentence reminder for a dimension.

    Requirement: keep only the fragment after 'measures', and normalize casing.

    Args:
        dimension (str): Dimension token.
        guidance_text (str): Full judge guidance text.

    Returns:
        str: One-sentence reminder.
    """
    return _normalized_after_measures(guidance_text)


def _format_dimension_reference_line(text: str) -> str:
    """
    Format one long-guidance line for the dimension reference page.

    Args:
        text (str): Raw guidance line.

    Returns:
        str: HTML-safe line with key comparison cues underlined.
    """
    formatted = _html_escape(text)
    for cue in ["Compare:", "Also compare:", "Prefer:", "Important:"]:
        formatted = formatted.replace(cue, f"<u>{cue}</u>", 1)
    return formatted


def _load_persona_description(persona: str) -> str:
    """
    Load persona description text from `configs/user_profiles/<persona>.yaml`.

    Args:
        persona (str): Persona name used in the human-annotation study.

    Returns:
        str: Persona context block.

    Raises:
        FileNotFoundError: If the persona YAML file is missing.
        ValueError: If the persona YAML cannot be parsed.
    """
    path = _PROJECT_ROOT / "configs" / "user_profiles" / f"{persona}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Persona config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Persona config must be a mapping: {path}")
    description = str(payload.get("description") or "").strip()
    persona_description = str(payload.get("persona_description") or "").strip()
    if persona_description and description:
        return f'{persona_description}\n\nDescription: "{description}"'
    return persona_description or description or persona


def _choice_label(swapped: bool, canonical_choice: str) -> str:
    """
    Convert a canonical A/B/tie choice into participant-facing labels.

    Args:
        swapped (bool): Whether displayed A/B positions are swapped.
        canonical_choice (str): Canonical `A`, `B`, or `tie` value.

    Returns:
        str: Participant-facing label.
    """
    if canonical_choice == "tie":
        return "Tie / Equal"
    if canonical_choice == "A":
        return "Response B" if swapped else "Response A"
    if canonical_choice == "B":
        return "Response A" if swapped else "Response B"
    raise ValueError(f"Unsupported canonical choice label: {canonical_choice}")


def _render_item_prompt(
    assignment: AnnotatorAssignment, qualtrics_config: QualtricsConfig
) -> str:
    """
    Render the participant-facing prompt block for a sampled item.

    Args:
        assignment (AnnotatorAssignment): Annotator assignment.

    Returns:
        str: Multi-line display text for the item.
    """
    item = assignment.item
    candidate = item.candidate
    display_a = (
        candidate.model_b_output
        if assignment.presentation_swapped
        else candidate.model_a_output
    )
    display_b = (
        candidate.model_a_output
        if assignment.presentation_swapped
        else candidate.model_b_output
    )
    persona_text = _load_persona_description(candidate.persona)

    # Qualtrics HTML restrictions: use only `<br>`, `<b>`, `<i>`, `<u>`.
    # Titles are underlined; response bodies are italicized.
    persona_render_text = f"{persona_text}"
    persona_section = _render_section(
        "User Persona", persona_render_text, italic_body=False
    )
    prompt_section = _render_section(
        "Question (Prompt)", candidate.input_text, italic_body=False
    )
    response_a_section = _render_section(
        "--- Response A ---", display_a, italic_body=True
    )
    response_b_section = _render_section(
        "--- Response B ---", display_b, italic_body=True
    )

    sections: List[str] = []
    if qualtrics_config.opening_per_item_text.strip():
        sections.append(
            _html_with_breaks(qualtrics_config.opening_per_item_text) + "<hr>"
        )
    sections.extend(
        [
            "=" * 59 + "<br>" + persona_section,
            prompt_section + "<br>" + "=" * 59,
            response_a_section + "<br>" + "=" * 59,
            response_b_section + "<br>" + "=" * 59 + "<hr>",
        ]
    )
    return "<br><br>".join(sections)


def _question_ids(assignment: AnnotatorAssignment, dimension: str) -> Dict[str, str]:
    """
    Build deterministic question ids for one annotator-item-dimension block.

    Args:
        assignment (AnnotatorAssignment): Annotator assignment.
        dimension (str): Pairwise dimension identifier.

    Returns:
        Dict[str, str]: Stable question ids for the item.
    """
    # Question IDs are derived from the assignment rank so result processing can
    # deterministically reconstruct the same Qualtrics columns later.
    prefix = f"I{assignment.assignment_rank:03d}"
    dim_short = dimension.upper()[:6]
    return {
        "overall": f"{prefix}_OVR",
        "confidence": f"{prefix}_CONF",
        "rationale": f"{prefix}_WHY",
        "dimension": f"{prefix}_{dim_short}",
        "dimensions_matrix": f"{prefix}_DIMMAT",
    }


def _render_single_choice_question(question_id: str, prompt: str) -> List[str]:
    """
    Render a single-choice Qualtrics question.

    Args:
        question_id (str): Stable Qualtrics question id.
        prompt (str): Participant-facing question text.

    Returns:
        List[str]: Advanced Format lines.
    """
    return [
        f"[[ID:{question_id}]]",
        "[[Question:MC:SingleAnswer]]",
        prompt,
        "[[Choices]]",
        "Response A",
        "Tie",
        "Response B",
    ]


def _render_dimensions_matrix_question(
    matrix_id: str, dimensions: Sequence[str]
) -> List[str]:
    """
    Render a single matrix question for all vibe dimensions.

    Args:
        matrix_id (str): Stable Qualtrics question id.
        dimensions (Sequence[str]): Dimension tokens (row labels).

    Returns:
        List[str]: Advanced Format lines.
    """
    return [
        f"[[ID:{matrix_id}]]",
        "[[Question:Matrix]]",
        MATRIX_PROMPT,
        "[[Choices]]",
        *[
            "<br>".join(
                [
                    f"<b>{_html_escape(_format_dimension_title(dim))}</b>",
                    f"<i>{_html_escape(_MATRIX_DIMENSION_DESCRIPTIONS.get(dim, _dimension_short_reminder(dim, get_dimension_guidance_text(dim))))}</i>",
                ]
            )
            for dim in dimensions
        ],
        "[[AdvancedAnswers]]",
        "[[Answer]]",
        "Response A",
        "[[Answer]]",
        "Tie",
        "[[Answer]]",
        "Response B",
    ]


def _render_confidence_question(question_id: str) -> List[str]:
    """
    Render a confidence Qualtrics question.

    Args:
        question_id (str): Stable Qualtrics question id.

    Returns:
        List[str]: Advanced Format lines.
    """
    return [
        f"[[ID:{question_id}]]",
        "[[Question:MC:SingleAnswer]]",
        CONFIDENCE_PROMPT,
        "[[Choices]]",
        "Low",
        "Medium",
        "High",
    ]


def _render_rationale_question(question_id: str) -> List[str]:
    """
    Render a free-text rationale question.

    Args:
        question_id (str): Stable Qualtrics question id.

    Returns:
        List[str]: Advanced Format lines.
    """
    return [
        f"[[ID:{question_id}]]",
        "[[Question:TE:Essay]]",
        RATIONALE_PROMPT,
    ]


def _build_selection_plan_rows(
    assignments: Sequence[AnnotatorAssignment],
    qualtrics_config: QualtricsConfig,
    filter_config: FilterConfig,
) -> List[Dict[str, Any]]:
    """
    Build flat audit rows for the sampled assignments.

    Args:
        assignments (Sequence[AnnotatorAssignment]): Annotator assignments.
        qualtrics_config (QualtricsConfig): Qualtrics rendering configuration.
        filter_config (FilterConfig): Filter configuration used for candidate eligibility.

    Returns:
        List[Dict[str, Any]]: Flat audit-plan rows.
    """
    rows: List[Dict[str, Any]] = []
    for assignment in assignments:
        candidate = assignment.item.candidate
        row = {
            "annotator_id": assignment.annotator_id,
            "assignment_rank": assignment.assignment_rank,
            "assignment_role": assignment.assignment_role,
            "item_role": assignment.item.selection_metadata.get("item_role", ""),
            "presentation_swapped": assignment.presentation_swapped,
            "item_id": assignment.item.item_id,
            "selection_target": assignment.item.selection_target,
            "source_key": candidate.source_key,
            "task_id": candidate.task_id,
            "variant_id": candidate.variant_id,
            "raw_task_id": candidate.raw_task_id,
            "persona": candidate.persona,
            "prompt_type": candidate.prompt_type,
            "pairwise_judgment_type": candidate.pairwise_judgment_type,
            "model_a_name": candidate.model_a_name,
            "model_b_name": candidate.model_b_name,
            "model_pair": candidate.model_pair,
            "judge_model_name": candidate.judge_model_name,
            "artifact_path": candidate.artifact_path,
            "artifact_index": candidate.artifact_index,
            "generator_model": candidate.generator_model,
            "filter_model": candidate.filter_model,
            "sample_type_fields": assignment.item.selection_metadata.get(
                "sample_type_fields", ""
            ),
            "sample_type_key": assignment.item.selection_metadata.get(
                "sample_type_key", ""
            ),
            "repeat_status": assignment.item.selection_metadata.get(
                "repeat_status", "fresh"
            ),
            "repeat_source_key": assignment.item.selection_metadata.get(
                "repeat_source_key", candidate.source_key
            ),
            "repeat_source_plan": assignment.item.selection_metadata.get(
                "repeat_source_plan", ""
            ),
            "annotators_per_regular_item": assignment.item.selection_metadata.get(
                "annotators_per_regular_item", ""
            ),
            "output_char_len_a": candidate.output_char_len_a,
            "output_char_len_b": candidate.output_char_len_b,
            "applied_max_output_chars": resolve_max_output_chars(
                candidate, filter_config
            ),
            "dimensions": ",".join(qualtrics_config.dimensions),
        }
        row.update(
            {
                key: value
                for key, value in assignment.item.selection_metadata.items()
                if key
                not in {
                    "selection_target",
                    "random_seed",
                    "marginal_balance_audit",
                    "included_type_audit",
                }
            }
        )
        rows.append(row)
    return rows


def _build_manifest_payload(
    config: HumanAnnotationConfig,
    sampled_items: Sequence[SampledAnnotationItem],
    assignments: Sequence[AnnotatorAssignment],
    qualtrics_files: Dict[str, str],
) -> Dict[str, Any]:
    """
    Build the authoritative manifest payload for a study workspace.

    Args:
        config (HumanAnnotationConfig): Study configuration.
        sampled_items (Sequence[SampledAnnotationItem]): Sampled items.
        assignments (Sequence[AnnotatorAssignment]): Annotator assignments.
        qualtrics_files (Dict[str, str]): Annotator-to-file mapping.

    Returns:
        Dict[str, Any]: Manifest payload.
    """
    return {
        "config": config.to_dict(),
        "sampled_items": [asdict(item) for item in sampled_items],
        "assignments": [asdict(item) for item in assignments],
        "qualtrics_files": qualtrics_files,
    }


def generate_qualtrics_artifacts(
    config: HumanAnnotationConfig,
    sampled_items: Sequence[SampledAnnotationItem],
    assignments: Sequence[AnnotatorAssignment],
) -> Dict[str, Path]:
    """
    Generate plan, manifest, and Qualtrics files for a study workspace.

    Args:
        config (HumanAnnotationConfig): Study configuration.
        sampled_items (Sequence[SampledAnnotationItem]): Sampled items.
        assignments (Sequence[AnnotatorAssignment]): Annotator assignments.

    Returns:
        Dict[str, Path]: Generated artifact paths keyed by logical name.
    """
    paths = build_study_artifact_paths(config)
    paths.study_dir.mkdir(parents=True, exist_ok=True)
    paths.qualtrics_dir.mkdir(parents=True, exist_ok=True)

    grouped: Dict[str, List[AnnotatorAssignment]] = defaultdict(list)
    for assignment in assignments:
        grouped[assignment.annotator_id].append(assignment)

    qualtrics_files: Dict[str, str] = {}
    for annotator_id, annotator_assignments in grouped.items():
        n_questions_per_item = 1 + len(config.qualtrics.dimensions)
        if config.qualtrics.include_confidence_question:
            n_questions_per_item += 1
        if config.qualtrics.include_rationale_question:
            n_questions_per_item += 1
        lines: List[str] = [
            "[[AdvancedFormat]]",
            f"[[Block:{config.qualtrics.survey_title}]]",
            f"[[ED:annotator_id:{annotator_id}]]",
            f"[[ED:study_name:{config.outputs.study_name}]]",
            "[[Question:Text]]",
            config.qualtrics.intro_text,
            "[[PageBreak]]",
        ]

        total_comparisons = len(annotator_assignments)

        # Explanation page (task instructions + toy example) if provided.
        if (
            config.qualtrics.task_overview_text
            or config.qualtrics.toy_example_prompt
            or config.qualtrics.toy_example_response_a
            or config.qualtrics.toy_example_response_b
        ):
            overview = "<br>".join(
                [
                    _html_with_breaks(config.qualtrics.task_overview_text),
                    "",
                    '<b>IMPORTANT:</b><br> - Focus ONLY on the given aspect, not overall quality.<br> - Consider the user\'s background, preferences, and context.<br> - "Tie" is acceptable only if both responses are genuinely equivalent for the given aspect.<br>Below is an example of user persona, question, responses, and annotations. You can use it to help you understand the task.<hr><br>',
                    "",
                ]
            )
            example_prompt_box = _render_section(
                "Question (Prompt)",
                config.qualtrics.toy_example_prompt,
                italic_body=False,
            )
            example_a_box = _render_section(
                "--- Response A ---",
                config.qualtrics.toy_example_response_a,
                italic_body=True,
            )
            example_b_box = _render_section(
                "--- Response B ---",
                config.qualtrics.toy_example_response_b,
                italic_body=True,
            )
            example_user = "<b>User Persona:</b> CS student. Prefers short, step-by-step explanations and a friendly tone."
            # example_explanation = "<b>Example explanation:</b><br>For this user persona (CS student), Response B is better in terms of clarity. So we can report that 'Response B' is better for the `Clarity` dimension."
            example_explanation = """<b>Annotations</b><br><br>

                Below are annotations for the given user persona on seven aspects:<br><br>

                1. <b>Clarity</b><br>
                <i>How clear, structured, and readable the response is.</i><br>
                <u>Winner</u>: Response <b>B</b>.<br>
                <u>Reason</u>: It is easier to follow because it includes a simple example that makes the answer clearer.<br><br>

                2. <b>Tone Style Fit</b><br>
                <i>Whether the writing style matches the user's preferences and the task context.</i><br>
                <u>Winner</u>: Response <b>A</b>.<br>
                <u>Reason</u>: Its friendly and supportive wording fits this beginner user better.<br><br>

                3. <b>Workflow Fit</b><br>
                <i>How well the response fits the user's workflow in terms of time, steps, and iteration cost.</i><br>
                <u>Winner</u>: Tie.<br>
                <u>Reason</u>: Both responses give a correct function and a brief explanation, so the user can move forward equally well with either one.<br><br>

                4. <b>Cognitive Load</b><br>
                <i>How mentally taxing it is for the user to process and apply the response.</i><br>
                <u>Winner</u>: Response <b>B</b>.<br>
                <u>Reason</u>: The example makes it easier for a beginner to understand the code with less mental effort.<br><br>

                5. <b>Context Awareness</b><br>
                <i>How well the response tracks the conversational state and constraints from the prompt and prior turns.</i><br>
                <u>Winner</u>: Tie.<br>
                <u>Reason</u>: Both responses follow the prompt by providing a Python function and a brief explanation.<br><br>

                6. <b>Persona Consistency</b><br>
                <i>Whether the response adheres to a specified role/persona across turns when a persona is part of the evaluation setup.</i><br>
                <u>Winner</u>: Tie.<br>
                <u>Reason</u>: No special assistant persona or fixed role is specified in this example, so this dimension is neutral here.<br><br>

                7. <b>Anthropomorphism</b><br>
                <i>Perceived human-likeness versus roboticness in text interaction.</i><br>
                <u>Winner</u>: Response <b>A</b>.<br>
                <u>Reason</u>: It sounds more natural and human in interaction, while Response B feels more plain and templated.<br><br>

                <hr><br>
                <b>Note:</b><br>
                Beyond prompts and responese, judgments depend also on the user persona. The same two responses can be judged differently for a different user.
                For example, if the user were an experienced developer who prefers very direct answers, Response B might be better for "Tone Style Fit" because it is more minimal."""
            explanation_html = "<br><br>".join(
                [
                    "<b>Task explanation</b>",
                    overview,
                    "<br>",
                    "<b>Example</b>",
                    example_user,
                    example_prompt_box,
                    example_a_box,
                    example_b_box,
                    "=" * 80 + "<br>",
                    example_explanation,
                ]
            )
            lines.extend(
                [
                    "[[Question:Text]]",
                    explanation_html,
                    "[[PageBreak]]",
                ]
            )

        # Dimension reference page (long guidance shown exactly to LLM judges).
        dimension_blocks: List[str] = [
            "<b>Dimension reference</b><br><br>Below are the full definitions of the dimensions you will be asked to compare the responses on. You can review them before you start the task to help you verify your understanding. You will have short reminders for their meaning in each comparison."
            + "<hr>"
        ]
        for dimension in config.qualtrics.dimensions:
            guidance = get_dimension_guidance_text(dimension)
            opening = _dimension_reference_opening_line(guidance)
            remainder_lines = (guidance or "").splitlines()[1:]
            remainder_html = "<br>".join(
                _format_dimension_reference_line(line) for line in remainder_lines
            )
            block_html = "<br>".join(
                [
                    f"<b>{_html_escape(_format_dimension_title(dimension))}</b>",
                    _html_escape(opening),
                    remainder_html,
                ]
            )
            dimension_blocks.append(block_html)
        lines.extend(
            [
                "[[Question:Text]]",
                "<br><br>".join(dimension_blocks),
                f"<hr><br><br>You will now start the task. You will be asked to annotate {total_comparisons} comparisons. You can pause and return to this form at any time.",
                "[[PageBreak]]",
            ]
        )
        for assignment in sorted(
            annotator_assignments, key=lambda item: item.assignment_rank
        ):
            item_question_ids = _question_ids(assignment, "overall")
            item_header_html = _render_item_prompt(assignment, config.qualtrics)
            lines.extend(
                [
                    "[[Question:Text]]",
                    "<br><br>".join(
                        [
                            f"<b>Comparison {assignment.assignment_rank}</b>",
                            item_header_html,
                        ]
                    ),
                ]
            )
            lines.extend(
                _render_dimensions_matrix_question(
                    item_question_ids["dimensions_matrix"], config.qualtrics.dimensions
                )
            )

            # add a text block with only <hr> as line breaks to separate the dimensions matrix question from the overall question
            lines.extend(
                [
                    "[[Question:Text]]",
                    "<hr>",
                ]
            )

            lines.extend(
                _render_single_choice_question(
                    item_question_ids["overall"],
                    OVERALL_PROMPT,
                )
            )
            if config.qualtrics.include_confidence_question:
                lines.extend(
                    _render_confidence_question(item_question_ids["confidence"])
                )
            if config.qualtrics.include_rationale_question:
                lines.extend(_render_rationale_question(item_question_ids["rationale"]))

            lines.append("[[PageBreak]]")
        lines.extend(
            [
                f"[[Block:{config.qualtrics.survey_title} Completion]]",
                "[[Question:Text]]",
                config.qualtrics.outro_text,
            ]
        )
        output_path = paths.qualtrics_dir / f"{annotator_id}_qualtrics.txt"
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        qualtrics_files[annotator_id] = str(output_path)
        logger.info(
            "Generated Qualtrics file for %s: path=%s items=%d questions=%d",
            annotator_id,
            output_path,
            len(annotator_assignments),
            len(annotator_assignments) * n_questions_per_item,
        )

    plan_rows = _build_selection_plan_rows(
        assignments, config.qualtrics, config.filters
    )
    pd.DataFrame(plan_rows).to_csv(paths.selection_plan_csv_path, index=False)
    save_json(
        _build_manifest_payload(config, sampled_items, assignments, qualtrics_files),
        str(paths.selection_manifest_path),
    )

    logger.info(
        "Generated %d Qualtrics participant files under %s with %d selection-plan rows",
        len(qualtrics_files),
        paths.qualtrics_dir,
        len(plan_rows),
    )
    return {
        "selection_manifest": paths.selection_manifest_path,
        "selection_plan_csv": paths.selection_plan_csv_path,
        "qualtrics_dir": paths.qualtrics_dir,
    }
