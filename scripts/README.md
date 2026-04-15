# Vibe Testing Pipeline Scripts

This directory documents the current experiment workflow for the Vibe Testing pipeline. The supported path is the YAML-based orchestrator in `run_experiment.py`, which handles configuration, completion detection, filtering, and task-based parallelism.

---

## Experiment Workflow

The orchestrator runs the paper pipeline in a small number of stages:

1. **Profile user**: build a structured user profile from a natural-language description.
2. **Select samples**: choose benchmark tasks for the target personas and settings.
3. **Build vibe dataset**: generate personalized and control prompt variants.
4. **Objective evaluation**: run the evaluated models and score correctness.
5. **Pairwise evaluation (5b)**: compare two models side by side.
6. **Analysis**: aggregate results and export summary tables and figures.

Use `run_experiment.py` for normal runs. The lower-level `stage_*.py` scripts are mainly useful for debugging or focused manual experiments.

---

## Example Config

Start from `configs/experiments/example_experiment.yaml`. A minimal commented config looks like this:

```yaml
# Human-readable experiment name used in logs and outputs.
name: "gpt5_persona_comparison"

# Root directory where this experiment writes artifacts.
base_dir: "runs/experiments/gpt5_comparison"

# Reusable persona and model definitions.
include:
  - "components/personas.yaml"
  - "components/models.yaml"

# Defaults shared across stages unless overridden.
defaults:
  benchmarks:
    - "mbpp_plus"              # Benchmark family to evaluate.
  num_samples: 20              # Number of tasks to sample per persona.
  num_variations: 2            # Number of personalized rewrites per task.
  filter_model: "none"         # Optional filter model for prompt screening.
  dataset_type: "function"     # Function-level evaluation.
  prompt_types:
    - "original"               # Original benchmark prompt.
    - "personalized"           # Full user-aware rewrite.
    - "control"                # Non-personal control rewrite.

# Model used for stages 1-3: profiling and dataset construction.
generator: "gpt5"

# Personas to include in the experiment matrix.
use_personas:
  - "researcher_user"
  - "novice_user"

# Models to evaluate objectively in stage 4.
use_models:
  - "gpt-oss-low-effort"
  - "gpt5"

# Judge models used for pairwise comparison.
use_judges:
  - "gpt5"
```

Component registries live in `configs/experiments/components/`:

- `personas.yaml`
- `models.yaml`

---

## Common Commands

```bash
# Run the full experiment.
python scripts/run_experiment.py configs/experiments/example_experiment.yaml

# Preview the task plan without executing anything.
python scripts/run_experiment.py configs/experiments/example_experiment.yaml --dry-run

# Inspect what has already completed.
python scripts/run_experiment.py configs/experiments/example_experiment.yaml --status

# Restrict the run to one persona.
python scripts/run_experiment.py configs/experiments/example_experiment.yaml --only-personas researcher_user

# Run only API-backed or only local models.
python scripts/run_experiment.py configs/experiments/example_experiment.yaml --model-tags api
python scripts/run_experiment.py configs/experiments/example_experiment.yaml --model-tags local

# Re-run only selected stages.
python scripts/run_experiment.py configs/experiments/example_experiment.yaml --stages dataset objective

# Force a rerun even if outputs already exist.
python scripts/run_experiment.py configs/experiments/example_experiment.yaml --force

# List tasks or run a specific task ID.
python scripts/run_experiment.py configs/experiments/example_experiment.yaml --list-tasks
python scripts/run_experiment.py configs/experiments/example_experiment.yaml --task-id 5
python scripts/run_experiment.py configs/experiments/example_experiment.yaml --task-id 3-7
```

---

## Stage Reference

| Stage | Script | Description |
| --- | --- | --- |
| 1 | `stage_1_profile_user.py` | Parse a user description into a structured profile. |
| 2 | `stage_2_select_samples.py` | Select benchmark samples for the experiment. |
| 3 | `stage_3_build_vibe_dataset.py` | Generate personalized and control prompt variants. |
| 4 | `stage_4_evaluate_vibe_dataset.py` | Run objective evaluation. |
| 5b | `stage_5b_pairwise_comparison.py` | Compare two models head to head. |
| 6 | `stage_6_analyze_results.py` | Aggregate results and generate analysis outputs. |

---

## Manual Stage Runs

If you need to debug a single stage directly, use the stage scripts in this directory with explicit input paths. For most users and most experiments, `run_experiment.py` should remain the default entry point.
