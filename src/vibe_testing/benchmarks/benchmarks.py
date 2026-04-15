from __future__ import annotations

import ast
import json
from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Union

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    concatenate_datasets,
    interleave_datasets,
    Value,
)

# --- Only import IterableDataset for type checkers to avoid runtime/type warnings
if TYPE_CHECKING:
    # Precise type for static analysis
    from datasets import IterableDataset as HFIterableDataset
else:
    # Runtime-friendly fallback (keeps annotations resolvable as strings)
    HFIterableDataset = Any

from src.vibe_testing.utils import get_datasets_cache_dir

IterableDataset = HFIterableDataset

# A unified “dataset-like” alias for return / parameter types
_DSLike = Union[Dataset, "HFIterableDataset"]


PostProc = Callable[[_DSLike], _DSLike]
FilterFn = Callable[[dict], bool]

_BENCHMARK_ALIASES: Dict[str, str] = {
    # Common variants / typos for HumanEval+.
    "humanevalplus": "humaneval_plus",
    "humaneval_plus": "humaneval_plus",
    "huamneval_plus": "humaneval_plus",
    "huamnevalplus": "humaneval_plus",
}


def _normalize_benchmark_name(name: str) -> str:
    """
    Normalize benchmark identifiers to registry keys.

    This keeps CLI/configs forgiving to common naming variants while still
    failing loudly for unknown benchmarks.
    """
    key = name.lower().strip()
    return _BENCHMARK_ALIASES.get(key, key)


@dataclass(frozen=True)
class BenchmarkSpec:
    hf_id: str
    default_split: Optional[str] = None
    default_config: Optional[str] = None
    note: str = ""
    postprocess: Optional[PostProc] = None


def materialize_iterable(
    ds: "HFIterableDataset", *, limit: Optional[int] = None
) -> Dataset:
    """
    Turn an IterableDataset into a map-style Dataset by consuming it.
    If `limit` is provided, only take the first N examples.

    Note: IterableDatasets don’t support len()/select(); this function
    consumes the stream and builds a regular Dataset you can slice.
    """
    from datasets import Dataset as HFDataset

    if limit is None:

        def gen():
            for ex in ds:
                yield ex

    else:

        def gen():
            for i, ex in enumerate(ds):
                if i >= limit:
                    break
                yield ex

    # A simple, reliable way to cache/convert (see HF & SO guidance)
    mat = HFDataset.from_generator(gen)
    return mat


def _apps_postprocess(
    ds: Union[Dataset, IterableDataset],
) -> Union[Dataset, IterableDataset]:
    """
    APPS stores 'solutions' and 'input_output' as JSON strings.
    Parse them into Python objects as shown in the dataset card.
    """
    import json

    def _mapper(ex):
        if ex.get("solutions"):
            try:
                ex["solutions"] = json.loads(ex["solutions"])
            except Exception:
                pass
        if ex.get("input_output"):
            try:
                ex["input_output"] = json.loads(ex["input_output"])
            except Exception:
                pass
        return ex

    return ds.map(_mapper)


# ---- Benchmark Registry ------------------------------------------------------

_REGISTRY: Dict[str, BenchmarkSpec] = {
    # Function-level Python problems, single test split
    # Usage: load_dataset("openai/openai_humaneval", split="test")
    "humaneval": BenchmarkSpec(
        hf_id="openai/openai_humaneval",
        default_split="test",
        note="OpenAI HumanEval (Python). Single 'test' split.",
    ),
    # MBPP (and 'sanitized' variant). Default split is "test".
    # Usage: load_dataset("mbpp") or load_dataset("mbpp", "sanitized")
    # "mbpp": BenchmarkSpec(
    #     hf_id="mbpp",  # short-id resolves to a maintained mirror; supports config "sanitized"
    #     default_split="test",
    #     default_config=None,
    #     note="MBPP. Optional config='sanitized' for the cleaned subset.",
    # ),
    # MBPP+ (EvalPlus)
    # Usage: load_dataset("evalplus/mbppplus")
    "mbpp_plus": BenchmarkSpec(
        hf_id="evalplus/mbppplus",
        default_split="test",
        note="MBPP+ (EvalPlus). Robustified MBPP tests. Uses 'test' script.",
    ),
    # HumanEval+ (EvalPlus)
    # Usage: load_dataset("evalplus/humanevalplus")
    "humaneval_plus": BenchmarkSpec(
        hf_id="evalplus/humanevalplus",
        default_split="test",
        note="HumanEval+ (EvalPlus). Robustified HumanEval tests. Uses 'test' script.",
    ),
    # APPS (Intro/Interview/Competition difficulties)
    # Usage: load_dataset("codeparrot/apps"); filter via difficulties arg if you want
    "apps": BenchmarkSpec(
        hf_id="codeparrot/apps",
        default_split=None,  # has train & test; caller can choose; we'll default to 'train' below
        note="APPS (train/test). Includes optional 'difficulties'=['introductory'|'interview'|'competition'].",
        postprocess=_apps_postprocess,
    ),
    # DS-1000 (test split only)
    # Usage: load_dataset("xlangai/DS-1000")["test"]
    "ds1000": BenchmarkSpec(
        hf_id="xlangai/DS-1000",
        default_split="test",
        note="DS-1000 simplified format. Single 'test' split.",
    ),
    # SWE-bench Verified (human-validated 500 samples, test split)
    # Usage: load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    "swe_bench_verified": BenchmarkSpec(
        hf_id="princeton-nlp/SWE-bench_Verified",
        default_split="test",
        note="SWE-bench Verified subset. Single 'test' split.",
    ),
    # BigCodeBench (the dataset uses version string as the split; latest on card is v0.1.4)
    # Usage: load_dataset("bigcode/bigcodebench", split="v0.1.4")
    "bigcodebench": BenchmarkSpec(
        hf_id="bigcode/bigcodebench",
        default_split="v0.1.4",
        note="BigCodeBench. Pass split='v0.1.4' (dataset has no standard train/test).",
    ),
    # HumanEval-X (multilingual; requires a language config, default 'python') – 'test' split only
    # Usage: load_dataset("THUDM/humaneval-x", "python", split="test")
    "humaneval_x": BenchmarkSpec(
        hf_id="THUDM/humaneval-x",
        default_config="python",
        default_split="test",
        note="HumanEval-X (configs: python | cpp | java | js | go). Single 'test' split.",
    ),
    # MBXP & Multi-HumanEval via AmazonScience/mxeval
    # Usage examples:
    #   load_dataset("AmazonScience/mxeval", "mbxp", split="python")
    #   load_dataset("AmazonScience/mxeval", "multi-humaneval", split="python")
    "mbxp": BenchmarkSpec(
        hf_id="AmazonScience/mxeval",
        default_config="mbxp",
        default_split="python",
        note="MBXP from MxEval (configs: 'mbxp' | 'multi-humaneval' | 'mathqa-x'; split is language, e.g., 'python').",
    ),
    # CodeXGLUE representatives (you can add more as needed)
    # Code-to-Text (docstring generation)
    # Usage: load_dataset("google/code_x_glue_ct_code_to_text", "python", split="train")
    "codexglue_code_to_text": BenchmarkSpec(
        hf_id="google/code_x_glue_ct_code_to_text",
        default_config="python",
        default_split="train",
        note="CodeXGLUE code-to-text (configs: python/java/javascript/php/ruby).",
    ),
    # Text-to-Code (NL2Code)
    # Usage: load_dataset("google/code_x_glue_tc_text_to_code", split="train")
    "codexglue_text_to_code": BenchmarkSpec(
        hf_id="google/code_x_glue_tc_text_to_code",
        default_config=None,
        default_split="train",
        note="CodeXGLUE text-to-code (Parquet).",
    ),
    # Code-to-Code translation (Java<->C#)
    # Usage: load_dataset("google/code_x_glue_cc_code_to_code_trans", split="train")
    "codexglue_code_to_code_trans": BenchmarkSpec(
        hf_id="google/code_x_glue_cc_code_to_code_trans",
        default_split="train",
        note="CodeXGLUE code-to-code translation (Java↔C#).",
    ),
    # CodeSearchNet (docs subset shown in card; returns IterableDataset)
    # Usage: load_dataset("irds/codesearchnet", "docs")
    "codesearchnet_docs": BenchmarkSpec(
        hf_id="irds/codesearchnet",
        default_config="docs",
        default_split=None,
        note="CodeSearchNet docs (IRDS). IterableDataset; iterate or cast as needed.",
    ),
}


def load_benchmark(
    name: str,
    *,
    split: Optional[str] = None,
    config: Optional[str] = None,
    streaming: Optional[bool] = None,
    filter_fn: Optional[FilterFn] = None,
    postprocess: Optional[PostProc] = None,
    select_fields: Optional[Iterable[str]] = None,
    limit: Optional[int] = None,
    **hf_kwargs: Any,
) -> Union[Dataset, IterableDataset]:
    """
    Load a coding benchmark by name and optionally:
      - choose split/config
      - filter rows (callable returning bool)
      - postprocess rows (map)
      - column-select & limit

    Args:
        name: one of the keys in the registry above (case-insensitive).
        split, config: override defaults from the registry.
        streaming: forward to datasets.load_dataset (for very large sets).
        filter_fn: `lambda example: True|False` to keep rows.
        postprocess: callable(ds) -> ds, applied after filtering. Overrides the registry's postprocessor if provided.
        select_fields: list of column names to keep.
        limit: keep only the first N rows (after filtering/postprocess).
        **hf_kwargs: forwarded to `load_dataset` (e.g., difficulties=["competition"] for APPS).

    Returns:
        A `datasets.Dataset` or `IterableDataset` ready to use.
    """
    key = _normalize_benchmark_name(name)
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown benchmark '{name}'. Available: {sorted(_REGISTRY.keys())}"
        )

    spec = _REGISTRY[key]

    # Decide config/split
    cfg = config if config is not None else spec.default_config
    spl = split if split is not None else spec.default_split

    # Reasonable defaults where split is missing
    if spl is None and key == "apps":
        # APPS has train and test; default to 'train' if not specified
        spl = "train"

    # Build kwargs for HF loader
    kwargs = dict(hf_kwargs)
    kwargs.setdefault("cache_dir", get_datasets_cache_dir())
    if cfg is not None:
        kwargs["name"] = cfg
    if streaming is not None:
        kwargs["streaming"] = streaming
    if spl is not None:
        kwargs["split"] = spl

    print(f"Loading dataset {spec.hf_id} with kwargs: {kwargs}")
    print(f"HF_HOME: {os.environ.get('HF_HOME')}")
    print(f"HF_DATASETS_CACHE: {os.environ.get('HF_DATASETS_CACHE')}")
    print(f"HF_HUB_CACHE: {os.environ.get('HF_HUB_CACHE')}")
    # print(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE')}") # transformer_cache will deprecate in the future, use HF_HOME instead
    os.environ.pop("TRANSFORMERS_CACHE", None)
    ds = load_dataset(spec.hf_id, **kwargs)

    # Some loaders return a DatasetDict if split=None; prefer a single Dataset
    if isinstance(ds, DatasetDict):
        # If user didn’t specify split and dataset has one obvious split, pick it.
        if spl is None and len(ds.keys()) == 1:
            ds = next(iter(ds.values()))
        else:
            # Prefer 'test' when present (evaluation-oriented)
            for pref in ("test", "validation", "dev", "train"):
                if pref in ds:
                    ds = ds[pref]
                    break
            else:
                # Fall back to first split
                ds = next(iter(ds.values()))

    # Filtering
    if filter_fn is not None:
        ds = ds.filter(filter_fn)

    # Post-process (dataset-specific first unless overridden)
    pp = postprocess if postprocess is not None else spec.postprocess
    if pp is not None:
        ds = pp(ds)

    # Column selection
    if select_fields is not None:
        cols = [c for c in select_fields if c in ds.column_names]
        ds = ds.remove_columns([c for c in ds.column_names if c not in cols])

    # Row limit
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    return ds


DSLike = Union[Dataset, HFIterableDataset]
Row = Dict[str, Any]
Adapter = Callable[[Row], Row]


# ---------- helpers ----------


def materialize_iterable(
    ds: HFIterableDataset, *, limit: Optional[int] = None
) -> Dataset:
    """
    Consume an IterableDataset into a map-style Dataset.
    If limit is provided, take only the first N rows.
    """
    from datasets import Dataset as HFDataset

    def gen():
        for i, ex in enumerate(ds):
            if limit is not None and i >= limit:
                break
            yield ex

    return HFDataset.from_generator(gen)


def _ensure_map_style(ds: DSLike, *, materialize_limit: Optional[int]) -> Dataset:
    """Return a map-style Dataset, materializing if needed."""
    if isinstance(ds, Dataset):
        return ds
    # Iterable → map
    return materialize_iterable(ds, limit=materialize_limit)


# ---------- unified schema & adapters ----------

UNIFIED_COLUMNS: List[str] = [
    "benchmark",  # str: canonical benchmark name
    "task_id",  # str: stable identifier within the benchmark
    "prompt",  # str: the text/code prompt for generation
    "ground_truth",  # str|None: the canonical solution or reference output
    "language",  # str|None: e.g., 'python','java','cpp', etc.
    "difficulty",  # str|None: e.g., 'easy','medium','hard','competition'
    "inputs",  # dict|None: structured inputs (if any)
    "tests",  # dict|None: test spec / unit test text / metadata
    "metadata",  # dict|None: any extra info
]


def _with_defaults(benchmark: str, row: Row) -> Row:
    """Fill missing unified fields with defaults and set benchmark."""
    base = {
        "benchmark": benchmark,
        "task_id": None,
        "prompt": None,
        "ground_truth": None,
        "language": None,
        "difficulty": None,
        "inputs": None,
        "tests": None,
        "metadata": None,
    }
    base.update(row)
    return base


# — Adapters (small and explicit). Adjust/extend as you add benchmarks. —


def adapt_humaneval(ex: Row) -> Row:
    # openai/openai_humaneval: typical keys: task_id, prompt, canonical_solution, test
    return _with_defaults(
        "humaneval",
        {
            "task_id": ex.get("task_id"),
            "prompt": ex.get("prompt"),
            "ground_truth": ex.get("canonical_solution"),
            "language": "python",
            "tests": {"test": ex.get("test")},  # tests as dict payload
            "metadata": {"entry_point": ex.get("entry_point")},
        },
    )


def adapt_mbpp(ex: Row, *, sanitized: bool = False) -> Row:
    # mbpp(/sanitized): problem statement fields include 'text', 'prompt', 'code', 'test_list' (variant-dependent)
    return _with_defaults(
        "mbpp_sanitized" if sanitized else "mbpp",
        {
            "task_id": ex.get("task_id") or ex.get("id") or "",
            "prompt": ex.get("text") or ex.get("prompt"),
            "ground_truth": ex.get("code"),
            "language": "python",
            "tests": {"test_list": ex.get("test_list"), "test": ex.get("test")},
        },
    )


def adapt_mbpp_plus(ex: Row) -> Row:
    # mbpp_plus: has 'task_id', 'code', 'prompt', 'test', 'test_list'
    # We rely on 'test' (full script) for execution, but extract entry_point from 'code' (solution).

    code = ex.get("code")
    entry_point = None
    if code:
        try:
            tree = ast.parse(code)
            # Find the last function definition
            func_nodes = [
                node
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if func_nodes:
                entry_point = func_nodes[-1].name
        except Exception:
            # If parsing fails, keep entry_point unset. Downstream logic may still
            # extract it from tests metadata or fall back to a safe default.
            entry_point = None

    if ex.get("task_id") in [94, 721, 722, 723, 754]:
        pass

    return _with_defaults(
        "mbpp_plus",
        {
            "task_id": ex.get("task_id") or str(ex.get("id")) or "",
            "prompt": ex.get("prompt"),
            "ground_truth": ex.get("code"),
            "language": "python",
            "tests": {
                "test": ex.get("test"),
                "entry_point": entry_point,
            },
            "metadata": {
                "source_file": ex.get("source_file"),
                "test_imports": ex.get("test_imports"),
                "original_test_list": ex.get("test_list"),
                "prompt_tests": ex.get("test_list"),
            },
        },
    )


def adapt_humaneval_plus(ex: Row) -> Row:
    """
    Adapt EvalPlus HumanEval+ rows into the unified schema.

    HumanEval+ uses the same core fields as HumanEval (prompt, canonical_solution,
    entry_point) but ships a single robustified `test` script that includes both
    base and plus tests. Downstream evaluation should treat this as a base test
    suite (i.e., no explicit plus/base split).
    """
    return _with_defaults(
        "humaneval_plus",
        {
            "task_id": ex.get("task_id"),
            "prompt": ex.get("prompt"),
            "ground_truth": ex.get("canonical_solution"),
            "language": "python",
            "tests": {
                "test": ex.get("test"),
                "entry_point": ex.get("entry_point"),
            },
            "metadata": {"entry_point": ex.get("entry_point")},
        },
    )


def adapt_apps(ex: Row) -> Row:
    # codeparrot/apps: keys include 'question', 'solutions' (list), 'input_output' (dict), 'difficulty'
    solutions = ex.get("solutions", [])
    return _with_defaults(
        "apps",
        {
            "task_id": ex.get("problem_id") or ex.get("id") or "",
            "prompt": ex.get("question"),
            "ground_truth": solutions[0] if solutions else None,
            "language": "python",
            "difficulty": ex.get("difficulty"),
            "inputs": ex.get("input_output"),
            "metadata": {"solutions": solutions},
        },
    )


def adapt_ds1000(ex: Row) -> Row:
    # xlangai/DS-1000: keys include 'prompt', 'metadata', 'code_context'
    md = ex.get("metadata") or {}
    # Prefer explicit None checks instead of `or` so that integer IDs like 0
    # are treated as valid (DS-1000 uses 0-based problem_id).
    primary_id = md.get("id")
    if primary_id is None:
        primary_id = ex.get("id")

    problem_id = ex.get("problem_id")
    if problem_id is None:
        problem_id = md.get("problem_id")

    instance_id = md.get("instance_id")
    if instance_id is None:
        instance_id = ex.get("instance_id")

    if primary_id is not None:
        task_id = primary_id
    elif problem_id is not None:
        task_id = problem_id
    elif instance_id is not None:
        task_id = instance_id
    else:
        # Fallback to row index if no identifier is present
        task_id = getattr(ex, "index", None)

    return _with_defaults(
        "ds1000",
        {
            "task_id": task_id,
            "prompt": ex.get("prompt"),
            "language": "python",
            "metadata": md,
            "tests": {"code_context": ex.get("code_context")},
        },
    )


def adapt_swebench_verified(ex: Row) -> Row:
    # princeton-nlp/SWE-bench_Verified: repo/instance_id/problem_statement/base_commit/patch/test_patch/...
    return _with_defaults(
        "swe_bench_verified",
        {
            "task_id": ex.get("instance_id"),
            "prompt": ex.get("problem_statement"),
            "ground_truth": ex.get("patch"),
            "language": "python",
            "metadata": {
                "repo": ex.get("repo"),
                "base_commit": ex.get("base_commit"),
                "version": ex.get("version"),
                "created_at": ex.get("created_at"),
                "difficulty": ex.get("difficulty"),
            },
            "tests": {
                "test_patch": ex.get("test_patch"),
                "FAIL_TO_PASS": ex.get("FAIL_TO_PASS"),
                "PASS_TO_PASS": ex.get("PASS_TO_PASS"),
            },
            "difficulty": ex.get("difficulty"),
        },
    )


def adapt_bigcodebench(ex: Row) -> Row:
    # bigcode/bigcodebench (split like 'v0.1.4'): fields include prompt, language(s), etc.
    return _with_defaults(
        "bigcodebench",
        {
            "task_id": ex.get("task_id") or ex.get("id") or "",
            "prompt": ex.get("prompt"),
            "ground_truth": ex.get("canonical_solution"),
            "language": ex.get("language"),
            "metadata": {
                k: ex.get(k)
                for k in ex.keys()
                if k not in {"task_id", "prompt", "language"}
            },
        },
    )


def adapt_humaneval_x(ex: Row, lang: str) -> Row:
    return _with_defaults(
        "humaneval_x",
        {
            "task_id": ex.get("task_id"),
            "prompt": ex.get("prompt"),
            "ground_truth": ex.get("canonical_solution"),
            "language": lang,
            "tests": {"test": ex.get("test")},
            "metadata": {"entry_point": ex.get("entry_point")},
        },
    )


def adapt_mbxp(ex: Row, lang: str) -> Row:
    # AmazonScience/mxeval with config='mbxp' and split=<language>
    return _with_defaults(
        "mbxp",
        {
            "task_id": ex.get("task_id") or ex.get("id"),
            "prompt": ex.get("prompt"),
            "ground_truth": ex.get("canonical_solution"),
            "language": lang,
            "tests": {"test": ex.get("test")},
            "metadata": {"source": ex.get("source")},
        },
    )


def adapt_cxg_code_to_text(ex: Row, lang: str) -> Row:
    return _with_defaults(
        "codexglue_code_to_text",
        {
            "task_id": ex.get("id") or "",
            "prompt": ex.get("code") or ex.get("func_code_string"),
            "ground_truth": ex.get("docstring") or ex.get("func_documentation_string"),
            "language": lang,
            "metadata": {},
        },
    )


def adapt_cxg_text_to_code(ex: Row) -> Row:
    return _with_defaults(
        "codexglue_text_to_code",
        {
            "task_id": ex.get("id") or "",
            "prompt": ex.get("nl"),
            "ground_truth": ex.get("code"),
            "language": ex.get("language") or "python",
            "metadata": {},  # some splits carry code targets
        },
    )


def adapt_cxg_code_to_code_trans(ex: Row) -> Row:
    return _with_defaults(
        "codexglue_code_to_code_trans",
        {
            "task_id": ex.get("id") or "",
            "prompt": ex.get("code"),
            "ground_truth": ex.get("target"),
            "language": ex.get("lang") or ex.get("src_lang"),
            "metadata": {
                "target_lang": ex.get("tgt_lang"),
            },
        },
    )


def adapt_codesearchnet_docs(ex: Row) -> Row:
    # irds/codesearchnet 'docs' is iterable; fields include 'docstring','func_name','language','repo', etc.
    return _with_defaults(
        "codesearchnet_docs",
        {
            "task_id": ex.get("id") or "",
            "prompt": ex.get("docstring") or ex.get("text"),
            "ground_truth": ex.get("code"),
            "language": ex.get("language"),
            "metadata": {"repo": ex.get("repo"), "func_name": ex.get("func_name")},
        },
    )


# Adapter registry: name -> adapter factory (optionally parameterized)
ADAPTERS: Dict[str, Callable[..., Adapter]] = {
    "humaneval": lambda: adapt_humaneval,
    "mbpp": lambda sanitized=False: (lambda ex: adapt_mbpp(ex, sanitized=sanitized)),
    "mbpp_plus": lambda: adapt_mbpp_plus,
    "humaneval_plus": lambda: adapt_humaneval_plus,
    "apps": lambda: adapt_apps,
    "ds1000": lambda: adapt_ds1000,
    "swe_bench_verified": lambda: adapt_swebench_verified,
    "bigcodebench": lambda: adapt_bigcodebench,
    "humaneval_x": lambda lang="python": (lambda ex: adapt_humaneval_x(ex, lang=lang)),
    "mbxp": lambda lang="python": (lambda ex: adapt_mbxp(ex, lang=lang)),
    "codexglue_code_to_text": lambda lang="python": (
        lambda ex: adapt_cxg_code_to_text(ex, lang=lang)
    ),
    "codexglue_text_to_code": lambda: adapt_cxg_text_to_code,
    "codexglue_code_to_code_trans": lambda: adapt_cxg_code_to_code_trans,
    "codesearchnet_docs": lambda: adapt_codesearchnet_docs,
}


# ---------- main: load & unify ----------


def load_and_unify_benchmarks(
    names: Iterable[str],
    *,
    # hooks to your earlier loader
    loader: Callable[..., DSLike],
    # adapter kwargs per benchmark (e.g., {"mbpp": {"sanitized": True}, "humaneval_x": {"lang":"java"}})
    adapter_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    # how to deal with IterableDataset
    materialize_if_iterable: bool = True,
    materialize_limit: Optional[int] = None,
    # concatenate vs interleave if any are iterable
    iterable_union_mode: str = "materialize",  # "materialize" or "interleave"
) -> DSLike:
    """
    Load multiple benchmarks and unify them into a single dataset with a common schema.

    - Uses small per-benchmark adapters to produce {UNIFIED_COLUMNS}.
    - If any dataset is IterableDataset:
        * materialize_if_iterable=True -> convert to map-style (optionally with a limit)
        * iterable_union_mode="interleave" -> keep iterables and interleave them

    Returns:
        Dataset (map-style) when everything is materialized; otherwise IterableDataset if interleaving.
    """
    adapter_overrides = adapter_overrides or {}
    adapted: List[DSLike] = []
    any_iterable = False

    if isinstance(names, str):
        name_list = [n for n in names.split(",") if n.strip()]
    else:
        name_list = list(names)

    for raw_name in name_list:
        name = _normalize_benchmark_name(raw_name)
        raw = loader(name)  # load_benchmark(name, ...) fits here
        is_iterable = not isinstance(raw, Dataset)
        any_iterable = any_iterable or is_iterable

        # pick adapter
        if name not in ADAPTERS:
            raise KeyError(
                f"No adapter registered for '{name}'. Add one to ADAPTERS."
            )
        adapter_factory = ADAPTERS[name]
        adapter_kwargs = adapter_overrides.get(name, {})
        adapter = adapter_factory(**adapter_kwargs)

        # Ensure map-style if we plan to concatenate
        ds_map = raw
        if materialize_if_iterable and is_iterable:
            ds_map = _ensure_map_style(raw, materialize_limit=materialize_limit)

        # Map to unified rows and keep only UNIFIED_COLUMNS
        def _mapper(ex: Row) -> Row:
            row = adapter(ex)
            # guarantee column presence/dtypes by forcing defaults
            filled = _with_defaults(row.get("benchmark", name), row)

            # Ensure task_id is always a string to prevent schema conflicts
            if filled.get("task_id") is not None:
                filled["task_id"] = str(filled["task_id"])

            # Serialize complex/heterogeneous dictionary fields to JSON strings
            for field in ["inputs", "tests", "metadata"]:
                if filled.get(field) is not None:
                    filled[field] = json.dumps(filled[field])

            return {k: filled.get(k) for k in UNIFIED_COLUMNS}

        ds_map = ds_map.map(
            _mapper,
            remove_columns=[
                c
                for c in getattr(ds_map, "column_names", [])
                if c not in UNIFIED_COLUMNS
            ],
        )

        # Explicitly cast task_id to string in the dataset features
        if "task_id" in ds_map.column_names:
            ds_map = ds_map.cast_column("task_id", Value("string"))

        adapted.append(ds_map)

    # Combine
    if any_iterable and not materialize_if_iterable:
        # Keep them iterable and interleave
        # (docs show interleave works for Dataset and IterableDataset)
        # https://huggingface.co/docs/datasets/en/stream
        return interleave_datasets(adapted)  # type: ignore[arg-type]

    # Otherwise, concatenate as a single map-style Dataset
    # https://huggingface.co/docs/datasets/en/process
    return concatenate_datasets(adapted)  # columns already aligned
