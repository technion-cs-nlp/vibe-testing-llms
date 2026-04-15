"""Common utility functions for the vibe-testing pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from dotenv import load_dotenv
from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate

from src.vibe_testing.pathing import RunContext, build_run_context
from src.vibe_testing.env import load_project_dotenv

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"
_ENV_INITIALIZED = False
_HM_HOME_PATH: Optional[Path] = None
FENCE_RE_JSON = re.compile(r"```(?:json)\s*(.*?)\s*```", re.I | re.S)
FENCE_RE_ANY = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)\s*```", re.S)


def extract_code_from_markdown(text: str) -> str:
    """
    Extracts a code block from a markdown-formatted string.

    Args:
        text (str): The string, which may contain a markdown code fence.

    Returns:
        str: The extracted code, or the original string if no fence is found.
    """
    match = FENCE_RE_ANY.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


# Regex: match triple backticks, optional language marker, then capture everything until closing ```
FENCE_RE_ANY = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


def ensure_environment(should_print: bool = False) -> Path:
    """
    Loads the .env file, validates HM_HOME, and configures cache directories.

    Returns:
        Path: The resolved HM_HOME directory path.

    Raises:
        EnvironmentError: If HM_HOME is not set, or if it/its cache subdirectories
            cannot be found or created.
    """
    global _ENV_INITIALIZED, _HM_HOME_PATH
    if _ENV_INITIALIZED and _HM_HOME_PATH is not None:
        return _HM_HOME_PATH

    if ENV_FILE.exists():
        load_dotenv(dotenv_path=ENV_FILE, override=False)

    hm_home_value = os.environ.get("HM_HOME")
    if not hm_home_value:
        raise EnvironmentError(
            "HM_HOME environment variable is not set. Please configure it in the .env file."
        )

    hm_home_path = Path(hm_home_value).expanduser()
    try:
        if not hm_home_path.is_dir():
            # Only attempt creation if it doesn't exist.
            hm_home_path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise EnvironmentError(
            f"HM_HOME directory '{hm_home_path}' is missing or inaccessible, "
            f"and could not be created: {exc}"
        ) from exc

    datasets_cache = hm_home_path / "datasets"
    models_cache = hm_home_path / "models"
    hf_home = hm_home_path / "hf"

    for directory in (datasets_cache, models_cache, hf_home, hf_home / "hub"):
        try:
            if not directory.is_dir():
                directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise EnvironmentError(
                f"Required cache directory '{directory}' is missing or inaccessible, "
                f"and could not be created: {exc}"
            ) from exc

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(datasets_cache))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub"))
    # os.environ.setdefault("TRANSFORMERS_CACHE", str(models_cache / "transformers")) # transformer_cache will deprecate in the future, use HF_HOME instead
    # delete TRANSFORMERS_CACHE from the environment with ''

    # Print the environment variables
    os.environ.pop("TRANSFORMERS_CACHE", None)
    # if should_print:
    #     print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    #     print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE')}")
    #     print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
    #     print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}")

    _HM_HOME_PATH = hm_home_path
    _ENV_INITIALIZED = True
    return hm_home_path


def hm_path(*relative_parts: str) -> str:
    """
    Builds an absolute path rooted at HM_HOME.

    Args:
        *relative_parts (str): Path components relative to HM_HOME.

    Returns:
        str: The combined absolute path.
    """
    base_path = ensure_environment()
    return str(base_path.joinpath(*relative_parts))


def get_datasets_cache_dir() -> str:
    """
    Provides the HM_HOME-backed cache directory for datasets.

    Returns:
        str: Absolute path to the datasets cache directory.
    """
    return hm_path("datasets")


def get_models_cache_dir(subdir: Optional[str] = None) -> str:
    """
    Provides the HM_HOME-backed cache directory for models.

    Args:
        subdir (Optional[str]): Optional child directory name.

    Returns:
        str: Absolute path to the requested models cache directory.
    """
    relative_parts = ("models",) + ((subdir,) if subdir else tuple())
    return hm_path(*relative_parts)


def extract_all_code_blocks(text: str, required_entry_point: str) -> List[str]:
    """
    Extracts all code blocks from a markdown-formatted string.
    Works with any language (e.g., ```python, ```java, or plain ```).

    Args:
        text (str): Input string that may contain markdown code fences.
        required_entry_point (str): Required function name for execution.
            This parameter is kept for backward compatibility; filtering and
            entry-point enforcement are handled by the caller (e.g., CodeSanitizer).

    Returns:
        List[str]: List of extracted code blocks. Empty list if none found.
    """
    matches = FENCE_RE_ANY.findall(text)
    return [m.strip() for m in matches]
    # concatenated_code = "\n".join([m.strip() for m in matches])
    # return concatenated_code


def format_test_cases_for_prompt(test_list: List[str]) -> str:
    """
    Formats a list of test cases into a string that can be appended to a prompt.

    Args:
        test_list (List[str]): List of test assertion strings.

    Returns:
        str: Formatted test cases string, or empty string if no tests provided.
    """
    if not test_list:
        return ""

    test_section = "\n\nTest cases:\n"
    for i, test in enumerate(test_list, 1):
        test_section += f"{i}. {test}\n"

    return test_section


def setup_logger(
    log_path: str, debug: bool = False, logger_name: Optional[str] = None
) -> logging.Logger:
    """
    Configures and returns a logger that writes to both console and a file.

    Args:
        log_path (str): The file path for the log file.
        debug (bool): Whether to enable debug-level logging.
        logger_name (Optional[str]): Optional explicit logger name. If not
            provided, defaults to this module's logger.

    Returns:
        logging.Logger: The configured logger instance.
    """
    name = logger_name or __name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    # If this logger already has handlers attached, assume it is fully
    # configured and avoid adding duplicate handlers that would lead to
    # repeated log lines (especially when stages are run in-process).
    if logger.handlers:
        return logger

    # Console handler
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stdout_handler.setFormatter(stdout_formatter)

    # File handler
    # create file if it doesn't exist
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w") as f:
            f.write("")
    file_handler = logging.FileHandler(filename=log_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)

    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)

    return logger


_STAGE_LOGGER_NAMES: Dict[str, str] = {
    "pipeline_orchestrator": "vibe.pipeline",
    "stage_1_profile_user": "vibe.stage_1_profile_user",
    "stage_2_select_samples": "vibe.stage_2_select_samples",
    "stage_3_build_vibe_dataset": "vibe.stage_3_build_vibe_dataset",
    "stage_4_evaluate_vibe_dataset": "vibe.stage_4_evaluate_vibe_dataset",
    "stage_5_subjective_vibe_evaluation": "vibe.stage_5_subjective_vibe_evaluation",
    "stage_6_analyze_results": "vibe.stage_6_analyze_results",
}


def get_stage_logger_name(stage: str) -> str:
    """
    Resolves a canonical logger name for a given pipeline stage.

    Args:
        stage (str): Identifier of the pipeline stage (e.g., \"stage_1_profile_user\"
            or \"pipeline_orchestrator\").

    Returns:
        str: The fully qualified logger name to use for that stage.
    """
    return _STAGE_LOGGER_NAMES.get(stage, f"vibe.{stage}")


def archive_existing_directory(
    target_dir: Union[str, Path],
    logger: logging.Logger,
    timestamp: Optional[str] = None,
) -> Optional[Path]:
    """
    Moves every non-archive entry inside target_dir into target_dir/old/<timestamp>.

    Args:
        target_dir (Union[str, Path]): Directory whose current contents should be archived.
        logger (logging.Logger): Logger used to emit informational messages.
        timestamp (Optional[str]): Optional timestamp override to simplify testing.

    Returns:
        Optional[Path]: Path to the archive directory if any items were moved, otherwise None.
    """
    path = Path(target_dir)
    if not path.exists():
        logger.debug("Directory %s does not exist; nothing to archive.", path)
        return None

    live_entries = [entry for entry in path.iterdir() if entry.name != "old"]
    if not live_entries:
        logger.debug("Directory %s has no active artifacts to archive.", path)
        return None

    archive_root = path / "old"
    archive_root.mkdir(parents=True, exist_ok=True)

    ts_value = timestamp or datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    archive_dir = archive_root / ts_value
    suffix = 1
    while archive_dir.exists():
        archive_dir = archive_root / f"{ts_value}_{suffix:02d}"
        suffix += 1
    archive_dir.mkdir(parents=True, exist_ok=False)

    for entry in live_entries:
        destination = archive_dir / entry.name
        shutil.move(str(entry), str(destination))
        logger.debug("Moved %s -> %s", entry, destination)

    logger.info(
        "Archived %s items from %s into %s.", len(live_entries), path, archive_dir
    )
    return archive_dir


def load_config(path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str, indent: int = 2):
    """Saves a Python object to a JSON file."""
    path_str = str(path)
    if "MagicMock" in path_str:
        raise ValueError(
            f"Output path appears to be a mock object: {path_str!r}. "
            "This typically occurs when resolve_run_context is patched in tests but "
            "artifact_path.return_value is not configured to return a real Path. "
            "Configure mock_run_context.artifact_path.return_value = Path(...) in the test."
        )
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Created directory for {path}")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: str) -> Any:
    """Loads a JSON object from a file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: List[Dict[str, Any]], path: str):
    """Saves a list of dictionaries to a JSONL file."""
    path_str = str(path)
    if "MagicMock" in path_str:
        raise ValueError(
            f"Output path appears to be a mock object: {path_str!r}. "
            "This typically occurs when resolve_run_context is patched in tests but "
            "artifact_path.return_value is not configured to return a real Path. "
            "Configure mock_run_context.artifact_path.return_value = Path(...) in the test."
        )
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Created directory for {path}")
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Loads data from a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """
    Adds arguments that are common across all pipeline scripts to a parser.

    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to.

    Returns:
        argparse.ArgumentParser: The parser with common arguments added.
    """
    parser.add_argument(
        "--run-dir",
        type=str,
        required=False,
        help="(Deprecated) Explicit run directory path. Overrides structured layout if provided.",
    )
    parser.add_argument(
        "--run-base-dir",
        type=str,
        default="Runs",
        help="Base directory for structured runs (default: Runs).",
    )
    parser.add_argument(
        "--user-group",
        type=str,
        required=True,
        help="User persona or cohort identifier (e.g., novice_user).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name used in the run (e.g., llama3-8b).",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=False,
        help="Optional run identifier (defaults to autogenerated timestamp).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="no_name",
        help="Human-friendly label prepended to timestamp-derived run IDs.",
    )
    parser.add_argument(
        "--selection-method",
        type=str,
        default=None,
        help="Selection strategy label used to organize Stage 2 outputs.",
    )
    parser.add_argument(
        "--generator-model-name",
        type=str,
        default=None,
        help="Generator/personalization model label for Stage 3/4/5 directories.",
    )
    parser.add_argument(
        "--filter-model-name",
        type=str,
        default=None,
        help="Filter/ranking model label for Stage 3/4/5 directories.",
    )
    parser.add_argument(
        "--evaluated-model-name",
        type=str,
        default=None,
        help="Evaluated model label for Stage 4/5/6 directories.",
    )
    parser.add_argument(
        "--judge-model-name",
        type=str,
        default=None,
        help="Judge model label for Stage 5/6 directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, validate inputs and print actions without executing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across runs (default: 42).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log level for stage console/file logging (default: INFO).",
    )
    return parser


def resolve_run_context(
    stage: str, args: argparse.Namespace, ensure_tree: bool = True
) -> RunContext:
    """
    Build a RunContext from parsed CLI arguments.

    Args:
        stage (str): Stage identifier (e.g., stage_1_profile_user).
        args (argparse.Namespace): Parsed arguments including run metadata.
        ensure_tree (bool): Whether to create directories on disk.

    Returns:
        RunContext: Canonical directory context for the run.
    """
    if args.run_dir:
        logger.warning(
            "Using deprecated --run-dir argument; structured naming is bypassed."
        )
        return build_run_context(
            stage=stage,
            user_group=args.user_group,
            model_name=args.model_name,
            run_id=args.run_id,
            base_dir=args.run_base_dir,
            ensure_tree_flag=ensure_tree,
            root_override=args.run_dir,
            run_name=args.run_name,
        )

    return build_run_context(
        stage=stage,
        user_group=args.user_group,
        model_name=args.model_name,
        run_id=args.run_id,
        base_dir=args.run_base_dir,
        ensure_tree_flag=ensure_tree,
        run_name=args.run_name,
    )


def _minor_repair(text: str) -> str:
    """Tiny, safe repairs: remove trailing commas before } or ]."""
    return re.sub(r",\s*(?=[}\]])", "", text)


def parse_and_validate_json(
    s: str,
    schema: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Extracts JSON from a string, parses it, and validates it against a schema.

    This function is designed to be robust to common LLM output issues, such as:
    - JSON being embedded within markdown code fences.
    - Extraneous text before or after the JSON object.
    - Trailing commas that would make the JSON invalid.

    Args:
        s (str): The source string that may contain the JSON.
        schema (Optional[Dict[str, Any]]): A JSON Schema dictionary to validate against.

    Returns:
        Any: The parsed and validated Python object.

    Raises:
        json.JSONDecodeError: If the string contains no valid JSON.
        jsonschema.ValidationError: If the parsed JSON does not match the schema.
    """
    # 1. Check for markdown fences and extract content if present
    fence_match = FENCE_RE_JSON.search(s) or FENCE_RE_ANY.search(s)
    if fence_match:
        content_to_parse = fence_match.group(1).strip()
    else:
        # If no fences, assume the whole string might be JSON, but needs cleaning.
        # Find the first '{' or '[' and the last '}' or ']'
        start_brace = s.find("{")
        start_bracket = s.find("[")
        end_brace = s.rfind("}")
        end_bracket = s.rfind("]")

        start = -1
        if start_brace != -1 and start_bracket != -1:
            start = min(start_brace, start_bracket)
        elif start_brace != -1:
            start = start_brace
        else:
            start = start_bracket

        end = max(end_brace, end_bracket)

        if start != -1 and end != -1 and end > start:
            content_to_parse = s[start : end + 1]
        else:
            content_to_parse = s  # Fallback to the original string

    # 2. Attempt to parse the content, with a repair attempt for trailing commas
    try:
        # First, try to parse the string as is
        parsed = json.loads(content_to_parse)
    except json.JSONDecodeError as e:
        # If it fails, attempt a minor repair and retry
        try:
            repaired_str = _minor_repair(content_to_parse)
            parsed = json.loads(repaired_str)
        except json.JSONDecodeError:
            # If the repair also fails, raise the original error for a clearer message
            print(f"Error: Failed to parse JSON: {e}")
            print(f"Content to parse: {content_to_parse}")
            print(f"Repaired string: {repaired_str}")
            raise e

    # 3. Validate against the schema if one is provided
    if schema:
        jsonschema_validate(instance=parsed, schema=schema)

    return parsed


def load_model_from_config(config: Union[str, dict]):
    """
    Loads a model instance from a configuration file path or a dictionary.

    Args:
        config (Union[str, dict]): Path to the YAML config file or a loaded config dictionary.

    Returns:
        An instance of a model (e.g., APIModel, HFModel), or None if config is invalid.

    Raises:
        SystemExit: If the model configuration is invalid or essential keys are missing.
    """
    if isinstance(config, str):
        if not os.path.exists(config):
            print(f"Error: Model config file not found at {config}")
            sys.exit(1)
        model_config = load_config(config)
        dotenv_start_dir = Path(config).resolve().parent
    elif isinstance(config, dict):
        model_config = config
        dotenv_start_dir = None
    else:
        raise TypeError("config must be a file path (str) or a dictionary.")

    # Load `.env` (if present) before any model initialization.
    #
    # Important: do NOT call ensure_environment() here. That function also enforces
    # HM_HOME, which is not required for API-only flows (e.g., listing tasks,
    # running API models on CPU-only nodes, etc.). We only need `.env` loading
    # for API keys and other optional configuration.
    load_project_dotenv(override=False, start_dir=dotenv_start_dir)

    # Configure the global consecutive-generation-failure threshold from the model
    # configuration (default: 100). This is intentionally driven solely by config,
    # not environment variables, to keep runs reproducible and provenance explicit.
    try:
        from src.vibe_testing.models.generation_failure_tracker import (
            configure_failure_threshold_from_model_config,
        )

        configure_failure_threshold_from_model_config(
            model_config,
            source=config if isinstance(config, str) else None,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Failed to configure generation failure threshold from model config: {exc}"
        ) from exc

    # This logic is based on how models are loaded in the stage scripts
    model_type = model_config.get("model_type")
    model_name = model_config.get("model_name")

    if "model" in model_config and isinstance(model_config["model"], dict):
        model_type = model_config["model"].get("name", model_type)
        if "params" in model_config["model"] and isinstance(
            model_config["model"]["params"], dict
        ):
            model_name = model_config["model"]["params"].get("model_name", model_name)

    if not model_type:
        print("Error: Could not determine model_type from config.")
        sys.exit(1)
    if not model_name:
        print("Error: Could not determine model_name from config.")
        sys.exit(1)

    # Lazy-import only the selected model implementation. This avoids importing
    # transformers-heavy modules when they're not needed, and ensures Unsloth can be
    # imported before transformers when using the Unsloth backend.

    if model_type == "api":
        from src.vibe_testing.models.api_model import APIModel

        return APIModel(model_name, model_config)

    if model_type == "hf":
        from src.vibe_testing.models.hf_model import HFModel

        return HFModel(model_name, model_config)

    if model_type == "qwen":
        backend = model_config.get("model", {}).get("params", {}).get("backend", "hf")
        if backend == "unsloth":
            # Best-effort: import Unsloth early on GPU runs to allow it to apply
            # transformers optimizations before any transformers imports occur.
            try:
                import torch  # type: ignore

                if torch.cuda.is_available():
                    import unsloth  # type: ignore  # noqa: F401
            except Exception:
                pass

        from src.vibe_testing.models.qwen_model import QwenModel

        return QwenModel(model_name, model_config)

    if model_type == "gemma":
        from src.vibe_testing.models.gemma_model import GemmaModel

        return GemmaModel(model_name, model_config)

    if model_type in {"gpt5", "gpt5_api"}:
        from src.vibe_testing.models.gpt5_api_model import GPT5APIModel

        return GPT5APIModel(model_name, model_config)

    if model_type in {"gpt4", "gpt4o"}:
        from src.vibe_testing.models.gpt4_api_model import GPT4APIModel

        return GPT4APIModel(model_name, model_config)

    if model_type == "llama":
        from src.vibe_testing.models.llama_model import LlamaModel

        return LlamaModel(model_name, model_config)

    if model_type == "gemini3":
        try:
            from src.vibe_testing.models.gemini3_model import Gemini3Model
        except ImportError:
            print(
                "Error: Gemini3Model is not available. Please install required dependencies."
            )
            sys.exit(1)
        return Gemini3Model(model_name, model_config)

    if model_type == "GptOssModel":
        backend = model_config.get("model", {}).get("params", {}).get("backend", "hf")
        if backend == "unsloth":
            # Import Unsloth before any transformers imports to ensure Unsloth can apply
            # its optimizations (per Unsloth guidance).
            try:
                import unsloth  # type: ignore  # noqa: F401
            except Exception:
                # Keep the detailed error handling in GptOssModel, but avoid importing
                # other transformers-heavy modules here.
                pass

        from src.vibe_testing.models.gpt_oss_model import GptOssModel

        return GptOssModel(model_name, model_config)

    print(f"Error: Unknown model_type in config: {model_type}")
    sys.exit(1)


def seed_everything(seed: int) -> None:
    """
    Seed all known random number generators for reproducibility.

    Args:
        seed (int): Random seed to apply across Python, NumPy, PyTorch, and Transformers.
    """
    logger.info("Seeding all RNGs with seed=%s", seed)

    # Python's built-in RNG
    random.seed(seed)

    # Hash-based operations
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy (optional)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except ImportError:
        logger.debug("NumPy not installed; skipping NumPy seeding.")

    # PyTorch (optional)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("PyTorch not installed; skipping torch seeding.")

    # Hugging Face transformers (optional)
    try:
        from transformers import set_seed as hf_set_seed  # type: ignore

        hf_set_seed(seed)
    except ImportError:
        logger.debug("Transformers not installed; skipping transformers.set_seed.")
