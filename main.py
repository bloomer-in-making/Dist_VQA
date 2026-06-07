from __future__ import annotations

import argparse
import importlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

try:
    import yaml
except ModuleNotFoundError:
    yaml = None


ROOT = Path(__file__).resolve().parent
DEFAULT_TASKS = ("xyz", "qaoa", "xyz_vop", "qaoa_vop")
PHASE_CHOICES = ("all", "phase1", "phase2", "randinit")


def parse_args() -> argparse.Namespace:
    """Define and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="VQA two-stage pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        choices=DEFAULT_TASKS,
        help="Run a single task only (optional when a pipeline config is used)",
    )
    parser.add_argument(
        "--phase",
        choices=PHASE_CHOICES,
        default="all",
        help="Select which phase of the 2-phase experiment to run",
    )
    parser.add_argument(
        "--config",
        help="Path to the config file. A pipeline YAML with phase definitions may be given",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override config values (e.g. --set runner.epochs=5)",
    )
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    """Load a JSON/YAML config file and return it as a dict."""
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")

    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML not installed: run pip install pyyaml and try again")
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge nested dictionaries."""
    merged = deepcopy(base)
    for key, value in overrides.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def parse_override(item: str) -> Dict[str, Any]:
    """Convert a --set KEY=VALUE item into a dict patch."""
    if "=" not in item:
        raise ValueError(f"invalid --set argument format: {item}")
    path, raw_value = item.split("=", 1)
    value = _infer_value(raw_value)
    keys = path.split(".")
    override: Dict[str, Any] = {}
    current = override
    for key in keys[:-1]:
        current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return override


def _infer_value(value: str) -> Any:
    """Infer a string as bool/int/float (in that order) and cast accordingly."""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def apply_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    """Apply multiple --set patches in sequence."""
    result = deepcopy(config)
    for item in overrides:
        patch = parse_override(item)
        result = deep_update(result, patch)
    return result


def build_runner(task: str, config: Dict[str, Any]):
    """Import the per-task runner module and call its builder."""
    module = importlib.import_module(f"tasks.{task}.runner")
    return module.build_runner(config)


def run_single(task: str, task_config: Dict[str, Any]) -> None:
    """Build a runner from a single-task config and run it."""
    runner = build_runner(task,task_config)
    runner.run()


def build_phase_payload(
    config_path: Path,
    defaults: Dict[str, Any],
    phase_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge per-phase external/inline configs with defaults into a run payload."""
    external_config = {}
    if path := phase_config.get("config_path"):
        external_config = load_config(_resolve_relative_path(config_path, path))

    inline_config = phase_config.get("config", {})
    merged = deep_update(defaults, external_config)
    merged = deep_update(merged, inline_config)
    return merged


def select_phases(phases: Dict[str, Any], selection: str) -> Iterable[str]:
    """Determine the list of phase names to run based on the --phase option."""
    if selection == "all":
        return list(phases.keys())
    if selection not in phases:
        raise ValueError(f"requested phase({selection}) is not in the config.")
    return [selection]


def resolve_config_path(task: str | None, user_config: str | None) -> Path:
    """Resolve the config file path from --config or the task default path."""
    if user_config:
        return Path(user_config)
    if task is None:
        raise ValueError("Either --config or --task must be specified.")
    return ROOT / "configs" / f"{task}.yaml"


def _resolve_relative_path(base_config: Path, target: str) -> Path:
    """Resolve a relative path in a phase config to an actual file path."""
    candidate = Path(target)
    if candidate.is_absolute():
        return candidate
    return base_config.parent / candidate


def main() -> None:
    """Entry point: load config, branch on phase, run tasks."""
    args = parse_args()
    config_path = resolve_config_path(args.task, args.config)
    raw_config = load_config(config_path)
    config = apply_overrides(raw_config, args.set)

    phases = config.get("phases")
    if phases:
        for phase_name in select_phases(phases, args.phase):
            phase_cfg = phases[phase_name]
            phase_task = phase_cfg.get("task") or args.task
            if not phase_task:
                raise ValueError(f"no task specified for {phase_name}.")

            task_payload = build_phase_payload(
                config_path=config_path,
                defaults=config.get("defaults", {}),
                phase_config=phase_cfg,
            )
            run_single(phase_task, task_payload)
    else:
        task_name = args.task or config.get("task")
        if not task_name:
            raise ValueError("A task name is required for a single run.")
        run_single(task_name, config)


if __name__ == "__main__":
    main()
