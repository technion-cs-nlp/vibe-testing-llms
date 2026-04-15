"""CLI script for annotating CSV examples using project LLM wrappers.

This script loads an input CSV containing model outputs (e.g., "Gemini JSON"
and "GPT-5.2 JSON"), sends each row to a configured model for annotation using
an annotation prompt, and writes an annotated CSV with an additional
``annotation`` column.
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# Ensure the project root (which contains the `src` package) is on sys.path
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.vibe_testing.models.base import GenerationRequest
from src.vibe_testing.utils import load_model_from_config


logger = logging.getLogger(__name__)


ANNOTATION_PROMPT: str = """
**SYSTEM MESSAGE**

You are an expert NLP researcher annotating empirical examples for an ACL paper.
Your job is to **carefully annotate vibe-testing examples** based **only on the information explicitly present in the input JSON**.

⚠️ **Critical rules**:

* Do **NOT** infer facts that are not directly stated.
* Do **NOT** guess the user's intent, persona, or evaluation criteria unless clearly supported by the quote or metadata.
* If something is unclear or missing, explicitly mark it as `"uncertain"` or `"not stated"`.
* Do **NOT** add benchmark claims unless the quote explicitly mentions benchmarks.
* Prefer *under-annotation* over speculation.
* If the example does **not** fully satisfy a label, say so.

Your goal is **accuracy and restraint**, not completeness.

---

**USER MESSAGE**

You will be given a single JSON object representing a candidate *vibe-testing episode* extracted from the web.

Your task is to annotate it using the schema below.

### **Definition Reminder**

A *vibe-testing episode* involves:

1. Evaluation of one or more AI models
2. A concrete task or usage scenario
3. Subjective, qualitative judgments (e.g., tone, clarity, reliability, "feels", aesthetic judgment, frustration)

---

### **Annotation Schema**

Return a **single JSON object** with the following fields:

```json
{{
  "is_vibe_testing": "yes | no | uncertain",
  "confidence_in_vibe_testing": "high | medium | low",

  "task_type_normalized": "<short normalized label or 'not stated'>",

  "models_mentioned": ["<model names copied verbatim from input>"],

  "vibe_dimensions": [
    "<select ONLY from the list below>"
  ],

  "vibe_dimensions_justification": {{
    "<dimension>": "<short justification citing exact words or phrases>"
  }},

  "vibe_language_cited": [
    "<exact words or short phrases from the quote>"
  ],

  "evaluation_basis": "qualitative | mixed | unclear",

  "benchmark_reference": "explicit | none | uncertain",

  "anthropomorphic_language": "yes | no",

  "brand_or_expectation_based_judgment": "yes | no | uncertain",

  "notes_or_uncertainties": "<brief note explaining any ambiguity or missing information>"
}}
```

---

### **Allowed `vibe_dimensions` (closed set)**

Only use dimensions **clearly supported by the text**:

* `clarity`
* `instruction_following`
* `reliability`
* `consistency`
* `aesthetic_quality`
* `hallucination_or_artifacts`
* `safety_or_censorship`
* `frustration_or_loss_of_control`
* `creativity`
* `efficiency_or_workflow_fit`
* `tone_or_personality`

If none apply confidently, return an **empty list**.

---

### **Strict Annotation Rules**

* **Do not infer user persona** unless explicitly stated.
* **Do not infer benchmarks** unless named (e.g., "benchmarks", "leaderboard", "score").
* **Do not merge dimensions** unless both are clearly present.
* **Every vibe dimension must be justified** using *exact quoted language*.
* If the episode involves subjective judgment but lacks a clear task, mark `is_vibe_testing = uncertain`.

---

### **Input Example**

Here is the JSON object to annotate:

```json
{example_json}
```
---

### **Output Requirements**

* Output **only valid JSON**
* No commentary outside the JSON
* No additional fields
* Be conservative and precise

"""


def configure_logging(verbosity: int) -> None:
    """
    Configure the root logger for the script.

    Args:
        verbosity (int): Verbosity level from CLI (0=WARNING, 1=INFO, 2=DEBUG).

    Returns:
        None

    Raises:
        ValueError: If verbosity is negative.
    """
    if verbosity < 0:
        raise ValueError("verbosity must be non-negative")

    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    logger.debug("Logging configured with level %s", logging.getLevelName(level))


def resolve_project_root() -> Path:
    """
    Resolve the project root directory based on this file location.

    Returns:
        Path: Absolute path to the project root directory.

    Raises:
        RuntimeError: If the project root cannot be resolved.
    """
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent
    if not project_root.exists():
        raise RuntimeError(f"Failed to resolve project root from {this_file}")
    return project_root


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    """
    Load all rows from a CSV file as dictionaries.

    Args:
        csv_path (Path): Path to the input CSV file.

    Returns:
        List[Dict[str, str]]: List of row dictionaries keyed by column names.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        OSError: For general file I/O issues.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    logger.info("Loading CSV file from %s", csv_path)
    rows: List[Dict[str, str]] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except OSError as exc:
        logger.error("Failed to read CSV file %s: %s", csv_path, exc)
        raise

    logger.info("Loaded %d rows from CSV", len(rows))
    return rows


def build_row_context(row: Dict[str, str]) -> str:
    """
    Build a human-readable context string for a CSV row.

    This function emphasizes the "Gemini JSON" and "GPT-5.2 JSON" columns if
    present, and falls back to including all columns otherwise.

    Args:
        row (Dict[str, str]): Single CSV row.

    Returns:
        str: Formatted context string describing the row.

    Raises:
        None
    """
    preferred_keys = ["Gemini JSON", "GPT-5.2 JSON"]
    parts: List[str] = []

    for key in preferred_keys:
        if key in row and row[key]:
            parts.append(f"{key}:\n{row[key]}")

    if not parts:
        # Fallback: dump all columns in a readable way.
        for key, value in row.items():
            parts.append(f"{key}: {value}")

    return "\n\n".join(parts)


def build_full_prompt(base_prompt: str, row: Dict[str, str]) -> str:
    """
    Combine the annotation prompt template with the row context.

    Args:
        base_prompt (str): Base annotation prompt that explains the task.
        row (Dict[str, str]): Single CSV row to annotate.

    Returns:
        str: Full prompt string to send to the model.

    Raises:
        None
    """
    row_context = build_row_context(row)
    full_prompt = (
        f"{base_prompt.strip()}\n\n"
        "---\n"
        "Example to annotate:\n"
        f"{row_context}\n"
        "---"
    )
    return full_prompt


def init_model(api_choice: str, project_root: Path) -> Any:
    """
    Initialize a model instance using the project's config loading utility.

    Args:
        api_choice (str): API choice string, either 'GPT5' or 'Gemini'.
        project_root (Path): Absolute path to the project root.

    Returns:
        Any: Instantiated model object compatible with BaseModel.

    Raises:
        ValueError: If api_choice is not supported.
        SystemExit: If model configuration is invalid (propagated from loader).
    """
    api_choice_upper = api_choice.upper()
    if api_choice_upper == "GPT5":
        config_path = project_root / "configs" / "model_configs" / "gpt5_config.yaml"
    elif api_choice_upper == "GEMINI":
        config_path = project_root / "configs" / "model_configs" / "gemini3-flash.yaml"
    else:
        raise ValueError("api_choice must be either 'GPT5' or 'Gemini'")

    logger.info("Loading model from config %s", config_path)
    model = load_model_from_config(str(config_path))
    logger.info("Initialized model instance: %r", model)
    return model


def parse_annotation_json(annotation_text: str) -> Dict[str, str]:
    """
    Parse the JSON annotation response and format it into CSV-friendly columns.

    Args:
        annotation_text (str): Raw annotation text from the model (should be JSON).

    Returns:
        Dict[str, str]: Dictionary with parsed annotation fields formatted for CSV.
    """
    parsed_fields: Dict[str, str] = {}

    # Try to extract JSON from the response (may be wrapped in markdown code blocks)
    json_text = annotation_text.strip()

    # Remove markdown code fences if present
    if json_text.startswith("```json"):
        json_text = json_text[7:]
    elif json_text.startswith("```"):
        json_text = json_text[3:]
    if json_text.endswith("```"):
        json_text = json_text[:-3]
    json_text = json_text.strip()

    try:
        annotation_data = json.loads(json_text)
        # Ensure we have a dictionary, not a list or other type
        if not isinstance(annotation_data, dict):
            raise ValueError(
                f"Expected JSON object, got {type(annotation_data).__name__}"
            )
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse annotation JSON: %s", exc)
        # Store raw annotation and error info
        parsed_fields["annotation_raw"] = annotation_text
        parsed_fields["annotation_parse_error"] = str(exc)
        # Set all expected fields to empty/error values
        for field in [
            "is_vibe_testing",
            "confidence_in_vibe_testing",
            "task_type_normalized",
            "models_mentioned",
            "vibe_dimensions",
            "vibe_dimensions_justification",
            "vibe_language_cited",
            "evaluation_basis",
            "benchmark_reference",
            "anthropomorphic_language",
            "brand_or_expectation_based_judgment",
            "notes_or_uncertainties",
        ]:
            parsed_fields[field] = ""
        return parsed_fields

    # Extract and format each field according to the schema
    parsed_fields["is_vibe_testing"] = str(annotation_data.get("is_vibe_testing", ""))
    parsed_fields["confidence_in_vibe_testing"] = str(
        annotation_data.get("confidence_in_vibe_testing", "")
    )
    parsed_fields["task_type_normalized"] = str(
        annotation_data.get("task_type_normalized", "")
    )

    # Format arrays as comma-separated values (or JSON if complex)
    models_mentioned = annotation_data.get("models_mentioned", [])
    if isinstance(models_mentioned, list):
        parsed_fields["models_mentioned"] = ", ".join(
            str(m) for m in models_mentioned if m
        )
    else:
        parsed_fields["models_mentioned"] = str(models_mentioned)

    vibe_dimensions = annotation_data.get("vibe_dimensions", [])
    if isinstance(vibe_dimensions, list):
        parsed_fields["vibe_dimensions"] = ", ".join(
            str(d) for d in vibe_dimensions if d
        )
    else:
        parsed_fields["vibe_dimensions"] = str(vibe_dimensions)

    vibe_language_cited = annotation_data.get("vibe_language_cited", [])
    if isinstance(vibe_language_cited, list):
        parsed_fields["vibe_language_cited"] = ", ".join(
            str(l) for l in vibe_language_cited if l
        )
    else:
        parsed_fields["vibe_language_cited"] = str(vibe_language_cited)

    # Format objects as JSON strings
    vibe_dimensions_justification = annotation_data.get(
        "vibe_dimensions_justification", {}
    )
    if isinstance(vibe_dimensions_justification, dict):
        parsed_fields["vibe_dimensions_justification"] = json.dumps(
            vibe_dimensions_justification, ensure_ascii=False
        )
    else:
        parsed_fields["vibe_dimensions_justification"] = str(
            vibe_dimensions_justification
        )

    parsed_fields["evaluation_basis"] = str(annotation_data.get("evaluation_basis", ""))
    parsed_fields["benchmark_reference"] = str(
        annotation_data.get("benchmark_reference", "")
    )
    parsed_fields["anthropomorphic_language"] = str(
        annotation_data.get("anthropomorphic_language", "")
    )
    parsed_fields["brand_or_expectation_based_judgment"] = str(
        annotation_data.get("brand_or_expectation_based_judgment", "")
    )
    parsed_fields["notes_or_uncertainties"] = str(
        annotation_data.get("notes_or_uncertainties", "")
    )

    return parsed_fields


def annotate_rows(
    rows: Iterable[Dict[str, str]],
    model: Any,
    base_prompt_template: str,
    api_choice: str = "GPT5",
    sleep_seconds: float = 0.5,
) -> List[Dict[str, str]]:
    """
    Annotate each CSV row using the provided model and base prompt template.

    Args:
        rows (Iterable[Dict[str, str]]): Input row dictionaries.
        model (Any): Model instance supporting ``generate_batch``.
        base_prompt_template (str): Base annotation prompt template with {example_json} placeholder.
        api_choice (str): API choice ("GPT5" or "Gemini") to determine which JSON column to use.
            Defaults to "GPT5" for backward compatibility with older callers/tests.
        sleep_seconds (float): Delay between model calls to avoid rate limits.

    Returns:
        List[Dict[str, str]]: New list of rows with parsed annotation fields as separate columns
        (is_vibe_testing, confidence_in_vibe_testing, task_type_normalized, models_mentioned,
        vibe_dimensions, vibe_dimensions_justification, vibe_language_cited, evaluation_basis,
        benchmark_reference, anthropomorphic_language, brand_or_expectation_based_judgment,
        notes_or_uncertainties, annotation_raw).

    Raises:
        None
    """
    annotated_rows: List[Dict[str, str]] = []

    for index, row in enumerate(rows, start=1):
        logger.info("Processing row %d", index)

        # Extract the appropriate JSON column for this row
        if api_choice == "Gemini":
            example_json = row.get("Gemini JSON", "")
        elif api_choice == "GPT5":
            example_json = row.get("GPT-5.2 JSON", "")
        else:
            example_json = ""

        # Format the prompt template with this row's JSON
        base_prompt = base_prompt_template.format(example_json=example_json)
        full_prompt = build_full_prompt(base_prompt, row)

        annotation_text = ""
        try:
            request = GenerationRequest(prompt=full_prompt)
            # Use generate_batch for compatibility with both GPT5APIModel and Gemini3Model.
            outputs = model.generate_batch([request])
            if not outputs:
                raise RuntimeError("Model returned no outputs.")
            annotation_text = str(outputs[0]).strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Annotation failed for row %d: %s", index, exc)
            annotation_text = f"ERROR: {exc}"

        # Parse the annotation JSON into separate CSV columns
        parsed_annotation = parse_annotation_json(annotation_text)

        # Create new row with original columns plus parsed annotation fields
        new_row = dict(row)
        new_row.update(parsed_annotation)

        # Also keep raw annotation for debugging
        # Backward-compatible alias: older callers/tests expect an "annotation" column.
        new_row["annotation"] = annotation_text
        new_row["annotation_raw"] = annotation_text

        annotated_rows.append(new_row)

        # Simple rate limiting backoff.
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return annotated_rows


def write_csv_rows(csv_path: Path, rows: List[Dict[str, str]]) -> None:
    """
    Write annotated rows to a CSV file.

    Args:
        csv_path (Path): Destination CSV path.
        rows (List[Dict[str, str]]): Rows to write.

    Returns:
        None

    Raises:
        OSError: If writing fails due to I/O issues.
    """
    if not rows:
        logger.warning("No rows to write to %s", csv_path)
        return

    # Collect all possible fieldnames from all rows to ensure consistency
    all_fieldnames = set()
    for row in rows:
        all_fieldnames.update(row.keys())
    fieldnames = sorted(all_fieldnames)

    logger.info("Writing %d rows to %s", len(rows), csv_path)
    try:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                # Ensure all fields are present (fill missing ones with empty string)
                complete_row = {field: row.get(field, "") for field in fieldnames}
                writer.writerow(complete_row)
    except OSError as exc:
        logger.error("Failed to write CSV file %s: %s", csv_path, exc)
        raise


def compute_output_path(input_path: Path, model_api: str = "") -> Path:
    """
    Compute the annotated CSV output path by appending '_annotated' to the name.

    Args:
        input_path (Path): Original CSV path.
        model_api (str): Optional API choice to include in filename. Defaults to
            empty string, which omits the API tag for backward compatibility.

    Returns:
        Path: Derived output CSV path.

    Raises:
        None
    """
    stem = input_path.stem
    suffix = input_path.suffix or ".csv"
    if model_api:
        return input_path.with_name(f"{stem}_{model_api}_annotated{suffix}")
    return input_path.with_name(f"{stem}_annotated{suffix}")


def compute_json_output_path(input_path: Path, model_api: str) -> Path:
    """
    Compute the JSON output path based on the input CSV path.

    Args:
        input_path (Path): Original CSV path.
        model_api (str): API choice to include in filename.

    Returns:
        Path: Derived output JSON path.

    Raises:
        None
    """
    stem = input_path.stem
    return input_path.with_name(f"{stem}_{model_api}_annotated.json")


def create_json_objects(annotated_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Convert annotated CSV rows into JSON objects for output.

    Each JSON object includes:
    - The parsed annotation JSON (as a structured object)
    - Name, Source, and Link fields from the original CSV
    - An index number according to the order in the CSV file

    Args:
        annotated_rows (List[Dict[str, str]]): Annotated rows from CSV.

    Returns:
        List[Dict[str, Any]]: List of JSON objects ready for output.

    Raises:
        None
    """
    json_objects: List[Dict[str, Any]] = []

    for index, row in enumerate(annotated_rows, start=1):
        # Extract annotation JSON from the parsed fields
        annotation_data: Dict[str, Any] = {}

        # Simple string fields
        if row.get("is_vibe_testing"):
            annotation_data["is_vibe_testing"] = row["is_vibe_testing"]
        if row.get("confidence_in_vibe_testing"):
            annotation_data["confidence_in_vibe_testing"] = row[
                "confidence_in_vibe_testing"
            ]
        if row.get("task_type_normalized"):
            annotation_data["task_type_normalized"] = row["task_type_normalized"]
        if row.get("evaluation_basis"):
            annotation_data["evaluation_basis"] = row["evaluation_basis"]
        if row.get("benchmark_reference"):
            annotation_data["benchmark_reference"] = row["benchmark_reference"]
        if row.get("anthropomorphic_language"):
            annotation_data["anthropomorphic_language"] = row[
                "anthropomorphic_language"
            ]
        if row.get("brand_or_expectation_based_judgment"):
            annotation_data["brand_or_expectation_based_judgment"] = row[
                "brand_or_expectation_based_judgment"
            ]
        if row.get("notes_or_uncertainties"):
            annotation_data["notes_or_uncertainties"] = row["notes_or_uncertainties"]

        # Array fields (convert from comma-separated back to arrays)
        if row.get("models_mentioned"):
            models = [
                m.strip() for m in row["models_mentioned"].split(",") if m.strip()
            ]
            if models:
                annotation_data["models_mentioned"] = models

        if row.get("vibe_dimensions"):
            dimensions = [
                d.strip() for d in row["vibe_dimensions"].split(",") if d.strip()
            ]
            if dimensions:
                annotation_data["vibe_dimensions"] = dimensions

        if row.get("vibe_language_cited"):
            language = [
                l.strip() for l in row["vibe_language_cited"].split(",") if l.strip()
            ]
            if language:
                annotation_data["vibe_language_cited"] = language

        # Object fields (parse JSON strings back to objects)
        if row.get("vibe_dimensions_justification"):
            try:
                justification = json.loads(row["vibe_dimensions_justification"])
                if justification:
                    annotation_data["vibe_dimensions_justification"] = justification
            except json.JSONDecodeError:
                # If parsing fails, skip this field
                pass

        # Create the output JSON object
        json_obj: Dict[str, Any] = {
            "index": index,
            "annotation": annotation_data,
        }

        # Add Name, Source, and Link from original CSV if present
        if row.get("Name"):
            json_obj["Name"] = row["Name"]
        if row.get("Source"):
            json_obj["Source"] = row["Source"]
        if row.get("Link"):
            json_obj["Link"] = row["Link"]

        json_objects.append(json_obj)

    return json_objects


def write_json_output(json_path: Path, json_objects: List[Dict[str, Any]]) -> None:
    """
    Write JSON objects to a JSON file.

    Args:
        json_path (Path): Destination JSON file path.
        json_objects (List[Dict[str, Any]]): List of JSON objects to write.

    Returns:
        None

    Raises:
        OSError: If writing fails due to I/O issues.
        ValueError: If JSON serialization fails.
    """
    if not json_objects:
        logger.warning("No JSON objects to write to %s", json_path)
        return

    logger.info("Writing %d JSON objects to %s", len(json_objects), json_path)
    try:
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(json_objects, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        logger.error("Failed to write JSON file %s: %s", json_path, exc)
        raise
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialize JSON: %s", exc)
        raise ValueError(f"JSON serialization failed: {exc}") from exc


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the annotation script.

    Args:
        argv (Optional[List[str]]): Optional list of CLI arguments for testing.

    Returns:
        argparse.Namespace: Parsed arguments.

    Raises:
        SystemExit: If arguments are invalid.
    """
    parser = argparse.ArgumentParser(
        description="Annotate CSV examples using GPT5 or Gemini models."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="examples_annotations/Examples Vibe-Testing.csv",
        help="Path to the input CSV file to annotate.",
    )
    parser.add_argument(
        "--api",
        type=str,
        choices=["GPT5", "Gemini"],
        default="GPT5",
        help="Which API/model configuration to use.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Seconds to sleep between API calls to avoid rate limits.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (use -vv for DEBUG).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point for the annotation script.

    Args:
        argv (Optional[List[str]]): Optional list of arguments for testing.

    Returns:
        int: Process exit code (0 on success, non-zero on failure).

    Raises:
        None
    """
    args = parse_args(argv)
    configure_logging(args.verbose)

    try:
        project_root = resolve_project_root()
    except RuntimeError as exc:
        logger.error("Could not resolve project root: %s", exc)
        return 1

    input_csv_path = (project_root / args.input).resolve()
    try:
        rows = load_csv_rows(input_csv_path)
    except FileNotFoundError as exc:
        logger.error("Input CSV not found: %s", exc)
        return 1
    except OSError as exc:
        logger.error("Failed to load input CSV: %s", exc)
        return 1

    try:
        model = init_model(args.api, project_root)
    except (ValueError, SystemExit) as exc:
        logger.error("Failed to initialize model: %s", exc)
        return 1

    annotated_rows = annotate_rows(
        rows=rows,
        model=model,
        base_prompt_template=ANNOTATION_PROMPT,
        api_choice=args.api,
        sleep_seconds=args.sleep_seconds,
    )

    # Write CSV output
    output_csv_path = compute_output_path(input_csv_path, args.api)
    try:
        write_csv_rows(output_csv_path, annotated_rows)
    except OSError:
        return 1

    # Create and write JSON output
    json_objects = create_json_objects(annotated_rows)
    output_json_path = compute_json_output_path(input_csv_path, args.api)
    try:
        write_json_output(output_json_path, json_objects)
    except (OSError, ValueError):
        return 1

    logger.info(
        "Annotation completed. Output written to %s and %s",
        output_csv_path,
        output_json_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
