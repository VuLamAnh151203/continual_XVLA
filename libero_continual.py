#!/usr/bin/env python3
"""Run the LIBERO continual-learning workflow extracted from the notebook.

All user-facing configuration is read from environment variables. The companion
run_libero.sh file provides Kaggle/Linux defaults.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import importlib.util
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an environment variable cannot be parsed."""


class TeeStream:
    """Mirror writes to the original stream and a log file."""

    def __init__(self, stream: Any, log_file: Any) -> None:
        self.stream = stream
        self.log_file = log_file
        self.encoding = getattr(stream, "encoding", "utf-8")

    def write(self, text: str) -> int:
        written = self.stream.write(text)
        self.log_file.write(text)
        return written

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def parse_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean value, got {raw!r}")


def parse_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def parse_csv(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ConfigError(f"{name} must contain at least one value")
    return values


def parse_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return list(default)

    raw = raw.strip()
    try:
        if raw.startswith("["):
            values = json.loads(raw)
        else:
            values = [item.strip() for item in raw.split(",") if item.strip()]
        parsed = [int(value) for value in values]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigError(
            f"{name} must be a comma-separated integer list or JSON list, got {raw!r}"
        ) from exc

    if not parsed:
        raise ConfigError(f"{name} must contain at least one task id")
    return parsed


def parse_json_map(name: str, default: dict[str, str]) -> dict[str, str]:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return dict(default)

    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{name} must be valid JSON, got {raw!r}") from exc

    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a JSON object")
    if not all(isinstance(key, str) and isinstance(val, str) for key, val in value.items()):
        raise ConfigError(f"{name} must map strings to strings")
    return value


def cli_bool(value: bool) -> str:
    return str(value).lower()


def task_ids_arg(task_ids: list[int]) -> str:
    return json.dumps(task_ids, separators=(",", ":"))


def command_to_string(command: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


@dataclass(frozen=True)
class Config:
    workdir: Path
    dataset_input_root: Path
    dataset_work_root: Path
    output_root: Path
    run_name: str
    run_output_root: Path
    results_file: Path
    eval_results_file: Path
    run_log_file: Path
    libero_config_path: Path
    libero_datasets: Path
    libero_benchmark_root: str
    write_libero_config: bool
    base_model: str
    eval_policy_path: str
    suites: list[str]
    convert_suites: list[str]
    dataset_repo_ids: dict[str, str]
    train_task_ids: list[int]
    test_task_ids: list[int]
    train_steps: int
    batch_size: int
    device: str
    control_mode: str
    n_eval_episodes: int
    eval_batch_size: int
    install_deps: bool
    copy_datasets: bool
    convert_datasets: bool
    run_train: bool
    run_eval: bool
    patch_checkpoint: bool
    train_all_suites: bool
    dry_run: bool
    cudnn_benchmark: str
    cublas_workspace_config: str
    env_max_parallel_tasks: int
    eval_freq: int
    policy_push_to_hub: bool
    policy_dtype: str
    optimizer_lr: str
    scheduler_warmup_steps: int
    scheduler_decay_steps: int
    scheduler_decay_lr: str
    action_mode: str
    policy_num_image_views: int
    policy_empty_cameras: int
    freeze_vision_encoder: bool
    freeze_language_encoder: bool
    train_policy_transformer: bool
    train_soft_prompts: bool
    peft_method_type: str
    peft_r: int
    peft_target_modules: str
    merge_lora_between_suites: bool
    rename_map: dict[str, str]
    tokenizer_name: str
    tokenizer_max_length: int
    tokenizer_task_key: str
    domain_id: int
    normalizer_eps: str
    image_shape: list[int]
    state_shape: list[int]
    action_shape: list[int]
    preprocessor_device: str
    postprocessor_device: str


def load_config() -> Config:
    workdir = Path(env("WORKDIR", "/kaggle/working")).expanduser()
    output_root = Path(env("OUTPUT_ROOT", str(workdir / "outputs" / "continual_learning"))).expanduser()
    run_name = env("RUN_NAME", datetime.now().strftime("%Y%m%d_%H%M%S"))
    dataset_work_root = Path(env("DATASET_WORK_ROOT", str(workdir / "IPEC-COMMUNITY"))).expanduser()
    libero_dataset_parent = Path(env("LIBERO_DATASET_PARENT", str(workdir / "libero"))).expanduser()

    dataset_repo_ids = {
        "libero_spatial": env(
            "LIBERO_SPATIAL_REPO_ID",
            "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
        ),
        "libero_goal": env(
            "LIBERO_GOAL_REPO_ID",
            "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
        ),
        "libero_10": env(
            "LIBERO_10_REPO_ID",
            "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
        ),
        "libero_object": env(
            "LIBERO_OBJECT_REPO_ID",
            "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
        ),
    }

    suites = parse_csv(
        "SUITES",
        ["libero_spatial", "libero_goal", "libero_10", "libero_object"],
    )
    convert_suites = parse_csv(
        "CONVERT_SUITES",
        ["libero_10", "libero_object", "libero_goal", "libero_spatial"],
    )

    missing_suites = [suite for suite in set(suites + convert_suites) if suite not in dataset_repo_ids]
    if missing_suites:
        raise ConfigError(
            "Missing dataset repo id env vars for suites: " + ", ".join(sorted(missing_suites))
        )

    return Config(
        workdir=workdir,
        dataset_input_root=Path(
            env(
                "DATASET_INPUT_ROOT",
                "/kaggle/input/datasets/anhvlm/libero-dataset/libero_dataset/IPEC-COMMUNITY",
            )
        ).expanduser(),
        dataset_work_root=dataset_work_root,
        output_root=output_root,
        run_name=run_name,
        run_output_root=output_root / "runs" / run_name,
        results_file=Path(env("RESULTS_FILE", str(output_root / "results.json"))).expanduser(),
        eval_results_file=Path(
            env("EVAL_RESULTS_FILE", str(output_root / "evaluation_results.json"))
        ).expanduser(),
        run_log_file=Path(env("RUN_LOG_FILE", str(output_root / "run.log"))).expanduser(),
        libero_config_path=Path(env("LIBERO_CONFIG_PATH", str(workdir / ".libero"))).expanduser(),
        libero_datasets=Path(env("LIBERO_DATASETS", str(libero_dataset_parent / "datasets"))).expanduser(),
        libero_benchmark_root=env("LIBERO_BENCHMARK_ROOT", ""),
        write_libero_config=parse_bool("WRITE_LIBERO_CONFIG", True),
        base_model=env("BASE_MODEL", "lerobot/xvla-base"),
        eval_policy_path=env("EVAL_POLICY_PATH", ""),
        suites=suites,
        convert_suites=convert_suites,
        dataset_repo_ids=dataset_repo_ids,
        train_task_ids=parse_int_list("TRAIN_TASK_IDS", list(range(8))),
        test_task_ids=parse_int_list("TEST_TASK_IDS", [8, 9]),
        train_steps=parse_int("TRAIN_STEPS", 8000),
        batch_size=parse_int("BATCH_SIZE", 2),
        device=env("DEVICE", "cuda"),
        control_mode=env("CONTROL_MODE", "absolute"),
        n_eval_episodes=parse_int("N_EVAL_EPISODES", 5),
        eval_batch_size=parse_int("EVAL_BATCH_SIZE", 1),
        install_deps=parse_bool("INSTALL_DEPS", True),
        copy_datasets=parse_bool("COPY_DATASETS", True),
        convert_datasets=parse_bool("CONVERT_DATASETS", True),
        run_train=parse_bool("RUN_TRAIN", True),
        run_eval=parse_bool("RUN_EVAL", False),
        patch_checkpoint=parse_bool("PATCH_CHECKPOINT", True),
        train_all_suites=parse_bool("TRAIN_ALL_SUITES", False),
        dry_run=parse_bool("DRY_RUN", False),
        cudnn_benchmark=env("CUDNN_BENCHMARK", "0"),
        cublas_workspace_config=env("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
        env_max_parallel_tasks=parse_int("ENV_MAX_PARALLEL_TASKS", 1),
        eval_freq=parse_int("EVAL_FREQ", 0),
        policy_push_to_hub=parse_bool("POLICY_PUSH_TO_HUB", False),
        policy_dtype=env("POLICY_DTYPE", ""),
        optimizer_lr=env("OPTIMIZER_LR", "1e-4"),
        scheduler_warmup_steps=parse_int("SCHEDULER_WARMUP_STEPS", 1000),
        scheduler_decay_steps=parse_int("SCHEDULER_DECAY_STEPS", 30000),
        scheduler_decay_lr=env("SCHEDULER_DECAY_LR", "2.5e-6"),
        action_mode=env("ACTION_MODE", "ee6d"),
        policy_num_image_views=parse_int("POLICY_NUM_IMAGE_VIEWS", 3),
        policy_empty_cameras=parse_int("POLICY_EMPTY_CAMERAS", 1),
        freeze_vision_encoder=parse_bool("FREEZE_VISION_ENCODER", False),
        freeze_language_encoder=parse_bool("FREEZE_LANGUAGE_ENCODER", False),
        train_policy_transformer=parse_bool("TRAIN_POLICY_TRANSFORMER", True),
        train_soft_prompts=parse_bool("TRAIN_SOFT_PROMPTS", True),
        peft_method_type=env("PEFT_METHOD_TYPE", "LORA"),
        peft_r=parse_int("PEFT_R", 64),
        peft_target_modules=env("PEFT_TARGET_MODULES", "all-linear"),
        merge_lora_between_suites=parse_bool("MERGE_LORA_BETWEEN_SUITES", True),
        rename_map=parse_json_map(
            "RENAME_MAP",
            {"observation.images.wrist_image": "observation.images.image2"},
        ),
        tokenizer_name=env("TOKENIZER_NAME", "facebook/bart-large"),
        tokenizer_max_length=parse_int("TOKENIZER_MAX_LENGTH", 50),
        tokenizer_task_key=env("TOKENIZER_TASK_KEY", "task"),
        domain_id=parse_int("DOMAIN_ID", 3),
        normalizer_eps=env("NORMALIZER_EPS", "1e-08"),
        image_shape=parse_int_list("IMAGE_SHAPE", [3, 224, 224]),
        state_shape=parse_int_list("STATE_SHAPE", [8]),
        action_shape=parse_int_list("ACTION_SHAPE", [20]),
        preprocessor_device=env("PREPROCESSOR_DEVICE", env("DEVICE", "cuda")),
        postprocessor_device=env("POSTPROCESSOR_DEVICE", "cpu"),
    )


def safe_config_for_results(config: Config) -> dict[str, Any]:
    return {
        "workdir": str(config.workdir),
        "dataset_input_root": str(config.dataset_input_root),
        "dataset_work_root": str(config.dataset_work_root),
        "output_root": str(config.output_root),
        "run_name": config.run_name,
        "run_output_root": str(config.run_output_root),
        "results_file": str(config.results_file),
        "eval_results_file": str(config.eval_results_file),
        "run_log_file": str(config.run_log_file),
        "libero_config_path": str(config.libero_config_path),
        "libero_datasets": str(config.libero_datasets),
        "base_model": config.base_model,
        "suites": config.suites,
        "convert_suites": config.convert_suites,
        "dataset_repo_ids": config.dataset_repo_ids,
        "train_task_ids": config.train_task_ids,
        "test_task_ids": config.test_task_ids,
        "train_steps": config.train_steps,
        "batch_size": config.batch_size,
        "optimizer_lr": config.optimizer_lr,
        "scheduler_warmup_steps": config.scheduler_warmup_steps,
        "scheduler_decay_steps": config.scheduler_decay_steps,
        "scheduler_decay_lr": config.scheduler_decay_lr,
        "policy_num_image_views": config.policy_num_image_views,
        "policy_empty_cameras": config.policy_empty_cameras,
        "merge_lora_between_suites": config.merge_lora_between_suites,
        "device": config.device,
        "control_mode": config.control_mode,
        "n_eval_episodes": config.n_eval_episodes,
        "eval_batch_size": config.eval_batch_size,
        "train_all_suites": config.train_all_suites,
        "dry_run": config.dry_run,
    }


def print_config(config: Config) -> None:
    print("========== LIBERO continual run config ==========")
    for key, value in safe_config_for_results(config).items():
        print(f"{key}: {value}")
    print("toggles:", {
        "install_deps": config.install_deps,
        "copy_datasets": config.copy_datasets,
        "convert_datasets": config.convert_datasets,
        "run_train": config.run_train,
        "run_eval": config.run_eval,
        "patch_checkpoint": config.patch_checkpoint,
    })
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN is not set; continuing because some runs use cached/local assets.")
    print("=================================================")


def run_command(command: list[str], config: Config, capture: bool = False) -> subprocess.CompletedProcess[str]:
    print("+ " + command_to_string(command), flush=True)
    if config.dry_run:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    kwargs: dict[str, Any] = {"check": True}
    if capture:
        kwargs.update({"capture_output": True, "text": True})
    return subprocess.run([str(part) for part in command], **kwargs)


def run_command_streaming(command: list[str], config: Config) -> subprocess.CompletedProcess[str]:
    print("+ " + command_to_string(command), flush=True)
    if config.dry_run:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    process = subprocess.Popen(
        [str(part) for part in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    def pump(stream: Any, sink: Any, parts: list[str]) -> None:
        try:
            for line in iter(stream.readline, ""):
                parts.append(line)
                print(line, end="", file=sink, flush=True)
        finally:
            stream.close()

    stdout_thread = threading.Thread(
        target=pump, args=(process.stdout, sys.stdout, stdout_parts), daemon=True
    )
    stderr_thread = threading.Thread(
        target=pump, args=(process.stderr, sys.stderr, stderr_parts), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()

    stdout = "".join(stdout_parts)
    stderr = "".join(stderr_parts)
    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code, [str(part) for part in command], output=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(command, return_code, stdout=stdout, stderr=stderr)


def install_dependencies(config: Config) -> None:
    package_groups = [
        ["robosuite==1.4.1"],
        ["bddl", "easydict", "cloudpickle", "num2words"],
        ["imageio[ffmpeg]"],
        ["lerobot[libero]"],
        ["lerobot[xvla]"],
    ]
    for packages in package_groups:
        run_command([sys.executable, "-m", "pip", "install", *packages], config)


def find_libero_benchmark_root() -> Path:
    try:
        spec = importlib.util.find_spec("libero.libero")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Cannot find the 'libero' package. Leave INSTALL_DEPS=1 for the first run, "
            "or install dependencies before running with INSTALL_DEPS=0."
        ) from exc

    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "Cannot locate libero.libero package files. Check that LIBERO installed correctly."
        )
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def ensure_libero_config(config: Config) -> None:
    config_file = config.libero_config_path / "config.yaml"
    if not config.write_libero_config and config_file.exists():
        return

    benchmark_root = (
        Path(config.libero_benchmark_root).expanduser().resolve()
        if config.libero_benchmark_root
        else find_libero_benchmark_root()
    )

    config.libero_config_path.mkdir(parents=True, exist_ok=True)
    config.libero_datasets.mkdir(parents=True, exist_ok=True)
    config_file.write_text(
        "\n".join(
            [
                f"assets: {benchmark_root / 'assets'}",
                f"bddl_files: {benchmark_root / 'bddl_files'}",
                f"benchmark_root: {benchmark_root}",
                f"datasets: {config.libero_datasets}",
                f"init_states: {benchmark_root / 'init_files'}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Created LIBERO config: {config_file}")
    print(f"LIBERO datasets path: {config.libero_datasets}")


def copy_datasets(config: Config) -> None:
    print(f"Copying datasets from {config.dataset_input_root} to {config.dataset_work_root}")
    if config.dry_run:
        return
    if not config.dataset_input_root.exists():
        raise FileNotFoundError(
            f"DATASET_INPUT_ROOT does not exist: {config.dataset_input_root}"
        )
    config.dataset_work_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(config.dataset_input_root, config.dataset_work_root, dirs_exist_ok=True)


def dataset_root(config: Config, suite: str) -> Path:
    repo_id = config.dataset_repo_ids[suite]
    return config.dataset_work_root / repo_id.split("/")[-1]


def require_dataset_root(config: Config, suite: str) -> Path:
    root = dataset_root(config, suite)
    if not config.dry_run and not root.exists():
        raise FileNotFoundError(
            f"Dataset root for {suite} does not exist: {root}. "
            "Enable COPY_DATASETS or adjust DATASET_WORK_ROOT."
        )
    return root


def convert_datasets(config: Config) -> None:
    for suite in config.convert_suites:
        repo_id = config.dataset_repo_ids[suite]
        root = require_dataset_root(config, suite)
        run_command(
            [
                sys.executable,
                "-m",
                "lerobot.scripts.convert_dataset_v21_to_v30",
                f"--repo-id={repo_id}",
                "--push-to-hub",
                "False",
                "--root",
                str(root),
            ],
            config,
        )


def checkpoint_path(config: Config, suite: str) -> Path:
    return config.run_output_root / f"train_{suite}" / "checkpoints" / "last" / "pretrained_model"


def merged_checkpoint_path(config: Config, suite: str) -> Path:
    return config.run_output_root / f"merged_{suite}" / "pretrained_model"


def train_suite(config: Config, suite: str, policy_path: str | Path) -> Path:
    output_dir = config.run_output_root / f"train_{suite}"
    root = require_dataset_root(config, suite)
    rename_map = json.dumps(config.rename_map, separators=(",", ":"))

    command = [
        "lerobot-train",
        f"--policy.path={policy_path}",
        f"--dataset.repo_id={config.dataset_repo_ids[suite]}",
        f"--dataset.root={root}",
        "--env.type=libero",
        f"--env.max_parallel_tasks={config.env_max_parallel_tasks}",
        f"--eval_freq={config.eval_freq}",
        f"--env.task={suite}",
        f"--env.task_ids={task_ids_arg(config.train_task_ids)}",
        f"--output_dir={output_dir}",
        f"--job_name=cl_{suite}",
        f"--steps={config.train_steps}",
        f"--batch_size={config.batch_size}",
        f"--policy.device={config.device}",
        f"--policy.push_to_hub={cli_bool(config.policy_push_to_hub)}",
        f"--policy.optimizer_lr={config.optimizer_lr}",
        f"--policy.scheduler_warmup_steps={config.scheduler_warmup_steps}",
        f"--policy.scheduler_decay_steps={config.scheduler_decay_steps}",
        f"--policy.scheduler_decay_lr={config.scheduler_decay_lr}",
        f"--policy.action_mode={config.action_mode}",
        f"--policy.num_image_views={config.policy_num_image_views}",
        f"--policy.empty_cameras={config.policy_empty_cameras}",
        f"--policy.freeze_vision_encoder={cli_bool(config.freeze_vision_encoder)}",
        f"--policy.freeze_language_encoder={cli_bool(config.freeze_language_encoder)}",
        f"--policy.train_policy_transformer={cli_bool(config.train_policy_transformer)}",
        f"--policy.train_soft_prompts={cli_bool(config.train_soft_prompts)}",
        f"--peft.method_type={config.peft_method_type}",
        f"--peft.r={config.peft_r}",
        f"--peft.target_modules={config.peft_target_modules}",
        f"--rename_map={rename_map}",
    ]

    if config.policy_dtype:
        command.append(f"--policy.dtype={config.policy_dtype}")

    print("\n" + "=" * 60)
    print(f"TRAINING on: {suite}")
    print(f"Dataset: {root}")
    print(f"Starting from: {policy_path}")
    print("=" * 60 + "\n")
    run_command(command, config)
    return checkpoint_path(config, suite)


def merge_lora_checkpoint(
    config: Config,
    base_policy_path: str | Path,
    adapter_checkpoint: Path,
    suite: str,
) -> Path:
    output_path = merged_checkpoint_path(config, suite)
    print(f"Merging LoRA adapter for {suite}")
    print(f"Base policy: {base_policy_path}")
    print(f"Adapter checkpoint: {adapter_checkpoint}")
    print(f"Merged checkpoint: {output_path}")

    if config.dry_run:
        return output_path
    if not adapter_checkpoint.exists():
        raise FileNotFoundError(f"Adapter checkpoint does not exist: {adapter_checkpoint}")

    merge_script = r"""
import json
import shutil
import sys
from pathlib import Path

from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from peft import PeftModel

base_policy_path = sys.argv[1]
adapter = Path(sys.argv[2])
dst = Path(sys.argv[3])

policy = XVLAPolicy.from_pretrained(base_policy_path)


def attach_and_merge(module, adapter_path):
    peft_module = PeftModel.from_pretrained(module, adapter_path)
    if not hasattr(peft_module, "merge_and_unload"):
        raise RuntimeError(
            f"PEFT attached to {type(module).__name__}, but merge_and_unload() is unavailable."
        )
    return peft_module.merge_and_unload()


def try_merge_children(module, adapter_path):
    if not hasattr(module, "named_children"):
        return False

    for name, child in list(module.named_children()):
        try:
            merged_child = attach_and_merge(child, adapter_path)
        except Exception:
            if try_merge_children(child, adapter_path):
                return True
            continue

        setattr(module, name, merged_child)
        print(f"Merged LoRA adapter into child module: {name} ({type(child).__name__})")
        return True

    return False


try:
    policy = attach_and_merge(policy, adapter)
    merged_any = True
    print(f"Merged LoRA adapter into policy: {type(policy).__name__}")
except Exception as exc:
    print(f"Could not attach adapter to policy directly: {exc}")
    merged_any = try_merge_children(policy, adapter)

if not merged_any:
    raise RuntimeError(
        f"Could not attach and merge LoRA adapter from {adapter}. "
        "The adapter may not match the base policy structure."
    )

if dst.exists():
    shutil.rmtree(dst)
dst.mkdir(parents=True, exist_ok=True)
if hasattr(policy, "config"):
    policy.config.use_peft = False
    policy.config.pretrained_path = None
policy.save_pretrained(dst)

config_path = dst / "config.json"
if config_path.exists():
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    cfg["use_peft"] = False
    cfg["pretrained_path"] = None
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)
        handle.write("\n")

for pattern in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
    path = dst / pattern
    if path.exists():
        path.unlink()

for path in adapter.iterdir():
    if path.name.startswith("adapter_"):
        continue
    target = dst / path.name
    if target.exists():
        continue
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)

model_files = list(dst.glob("model*.safetensors")) + list(dst.glob("pytorch_model*.bin"))
if not model_files:
    raise RuntimeError(
        f"Merged checkpoint at {dst} does not contain a full model file. "
        "Expected model*.safetensors or pytorch_model*.bin."
    )
"""

    run_command(
        [
            sys.executable,
            "-c",
            merge_script,
            str(base_policy_path),
            str(adapter_checkpoint),
            str(output_path),
        ],
        config,
    )
    if config.patch_checkpoint:
        patch_checkpoint(config, output_path)
    return output_path


def parse_success_rate(stdout: str) -> float | None:
    success_rate: float | None = None
    for line in stdout.splitlines():
        if "success" not in line.lower():
            continue
        matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if not matches:
            continue
        value = float(matches[-1])
        success_rate = value / 100.0 if "%" in line and value > 1.0 else value
    return success_rate


def evaluate_suite(config: Config, policy_path: str | Path, suite: str) -> float | None:
    command = [
        "lerobot-eval",
        f"--policy.path={policy_path}",
        "--env.type=libero",
        f"--env.task={suite}",
        f"--env.task_ids={task_ids_arg(config.test_task_ids)}",
        f"--env.control_mode={config.control_mode}",
        f"--eval.batch_size={config.eval_batch_size}",
        f"--eval.n_episodes={config.n_eval_episodes}",
        f"--output_dir={config.run_output_root / f'eval_{suite}'}",
    ]

    print(f"Evaluating on: {suite}")
    result = run_command_streaming(command, config)
    combined_output = "\n".join([result.stdout or "", result.stderr or ""])
    success_rate = parse_success_rate(combined_output)
    print(f"Success rate on {suite}: {success_rate}")
    return success_rate


def processor_feature(feature_type: str, shape: list[int]) -> dict[str, Any]:
    return {"type": feature_type, "shape": shape}


def patch_checkpoint(config: Config, base_path: Path) -> None:
    print(f"Patching checkpoint metadata at {base_path}")
    if config.dry_run:
        return
    if not base_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {base_path}")

    config_json = base_path / "config.json"
    if config_json.exists():
        with config_json.open("r", encoding="utf-8") as handle:
            policy_config = json.load(handle)
        policy_config["action_mode"] = config.action_mode
        with config_json.open("w", encoding="utf-8") as handle:
            json.dump(policy_config, handle, indent=2)
            handle.write("\n")
    else:
        print(f"Skipping action_mode patch; missing file: {config_json}")

    postprocessor = {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {
                    "eps": float(config.normalizer_eps),
                    "features": {
                        "action": processor_feature("ACTION", config.action_shape),
                    },
                    "norm_map": {
                        "VISUAL": "MEAN_STD",
                        "STATE": "IDENTITY",
                        "ACTION": "IDENTITY",
                    },
                },
            },
            {
                "registry_name": "device_processor",
                "config": {
                    "device": config.postprocessor_device,
                    "float_dtype": None,
                },
            },
        ],
    }

    preprocessor = {
        "name": "policy_preprocessor",
        "steps": [
            {
                "registry_name": "rename_observations_processor",
                "config": {"rename_map": config.rename_map},
            },
            {"registry_name": "to_batch_processor", "config": {}},
            {
                "registry_name": "tokenizer_processor",
                "config": {
                    "max_length": config.tokenizer_max_length,
                    "task_key": config.tokenizer_task_key,
                    "padding_side": "right",
                    "padding": "max_length",
                    "truncation": True,
                    "tokenizer_name": config.tokenizer_name,
                },
            },
            {
                "registry_name": "xvla_add_domain_id",
                "config": {"domain_id": config.domain_id},
            },
            {
                "registry_name": "device_processor",
                "config": {
                    "device": config.preprocessor_device,
                    "float_dtype": None,
                },
            },
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "eps": float(config.normalizer_eps),
                    "features": {
                        "observation.images.image": processor_feature("VISUAL", config.image_shape),
                        "observation.images.image2": processor_feature("VISUAL", config.image_shape),
                        "observation.state": processor_feature("STATE", config.state_shape),
                        "action": processor_feature("ACTION", config.action_shape),
                    },
                    "norm_map": {
                        "VISUAL": "IDENTITY",
                        "STATE": "IDENTITY",
                        "ACTION": "IDENTITY",
                    },
                },
            },
        ],
    }

    for filename, payload in {
        "policy_postprocessor.json": postprocessor,
        "policy_preprocessor.json": preprocessor,
    }.items():
        with (base_path / filename).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


def save_results(config: Config, results: dict[str, Any]) -> None:
    if config.dry_run:
        print(f"DRY_RUN enabled; not writing {config.results_file}")
        return
    config.results_file.parent.mkdir(parents=True, exist_ok=True)
    with config.results_file.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print(f"Results saved to {config.results_file}")


def save_eval_results(config: Config, results: dict[str, Any]) -> None:
    if config.dry_run:
        print(f"DRY_RUN enabled; not writing {config.eval_results_file}")
        return

    payload = {
        "started_at": results.get("started_at"),
        "updated_at": datetime.now().isoformat(),
        "config": {
            "suites": config.suites,
            "train_task_ids": config.train_task_ids,
            "test_task_ids": config.test_task_ids,
            "n_eval_episodes": config.n_eval_episodes,
            "eval_batch_size": config.eval_batch_size,
            "output_root": str(config.output_root),
            "run_name": config.run_name,
            "run_output_root": str(config.run_output_root),
        },
        "eval_rounds": results.get("eval_rounds", {}),
    }

    config.eval_results_file.parent.mkdir(parents=True, exist_ok=True)
    with config.eval_results_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Evaluation results saved to {config.eval_results_file}")


def save_all_results(config: Config, results: dict[str, Any]) -> None:
    save_results(config, results)
    save_eval_results(config, results)


def configure_runtime_env(config: Config) -> None:
    os.environ["CUDNN_BENCHMARK"] = config.cudnn_benchmark
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.cublas_workspace_config
    os.environ["LIBERO_CONFIG_PATH"] = str(config.libero_config_path)


def run_with_logging(config: Config) -> int:
    if config.dry_run:
        print(f"DRY_RUN enabled; not writing run log {config.run_log_file}")
        return run_workflow(config)

    config.run_log_file.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with config.run_log_file.open("a", encoding="utf-8", buffering=1) as log_file:
        log_file.write("\n\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Run started at {datetime.now().isoformat()}\n")
        log_file.write("=" * 80 + "\n")
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            print(f"Full run log: {config.run_log_file}")
            return run_workflow(config)
        finally:
            print(f"Full run log saved to {config.run_log_file}")
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def run_workflow(config: Config) -> int:
    print_config(config)

    if not config.dry_run:
        config.output_root.mkdir(parents=True, exist_ok=True)
        config.run_output_root.mkdir(parents=True, exist_ok=True)

    if config.install_deps:
        install_dependencies(config)
    if not config.dry_run and (config.convert_datasets or config.run_train or config.run_eval):
        ensure_libero_config(config)
    if config.copy_datasets:
        copy_datasets(config)
    if config.convert_datasets:
        convert_datasets(config)

    results: dict[str, Any] = {
        "started_at": datetime.now().isoformat(),
        "config": safe_config_for_results(config),
        "train_runs": [],
        "eval_rounds": {},
    }

    if config.run_train:
        train_suites = config.suites if config.train_all_suites else config.suites[:1]
        previous_policy: str | Path = config.base_model

        for suite in train_suites:
            adapter_checkpoint = train_suite(config, suite, previous_policy)
            if config.patch_checkpoint:
                patch_checkpoint(config, adapter_checkpoint)

            current_checkpoint = adapter_checkpoint
            merged_checkpoint: Path | None = None
            if config.merge_lora_between_suites:
                merged_checkpoint = merge_lora_checkpoint(
                    config,
                    previous_policy,
                    adapter_checkpoint,
                    suite,
                )
                current_checkpoint = merged_checkpoint

            print(current_checkpoint)

            results["train_runs"].append(
                {
                    "suite": suite,
                    "started_from": str(previous_policy),
                    "adapter_checkpoint": str(adapter_checkpoint),
                    "merged_checkpoint": str(merged_checkpoint) if merged_checkpoint else None,
                    "checkpoint": str(current_checkpoint),
                    "timestamp": datetime.now().isoformat(),
                }
            )

            if config.run_eval:
                round_key = f"after_training_{suite}"
                seen_suites = train_suites[: train_suites.index(suite) + 1]
                results["eval_rounds"][round_key] = {
                    "trained_on": suite,
                    "checkpoint": str(current_checkpoint),
                    "timestamp": datetime.now().isoformat(),
                    "evaluations": {},
                }
                for eval_suite in seen_suites:
                    success_rate = evaluate_suite(config, current_checkpoint, eval_suite)
                    results["eval_rounds"][round_key]["evaluations"][eval_suite] = success_rate
                save_all_results(config, results)

            previous_policy = current_checkpoint

    elif config.run_eval:
        policy_path = config.eval_policy_path or config.base_model
        results["eval_rounds"]["standalone_eval"] = {
            "checkpoint": str(policy_path),
            "timestamp": datetime.now().isoformat(),
            "evaluations": {},
        }
        for suite in config.suites:
            success_rate = evaluate_suite(config, policy_path, suite)
            results["eval_rounds"]["standalone_eval"]["evaluations"][suite] = success_rate

    if config.run_train or config.run_eval:
        save_all_results(config, results)
    else:
        print("RUN_TRAIN and RUN_EVAL are disabled; configuration parsed successfully.")

    print("\n========== FINAL RESULTS ==========")
    print(json.dumps(results, indent=2))
    return 0


def main() -> int:
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_runtime_env(config)
    return run_with_logging(config)


if __name__ == "__main__":
    raise SystemExit(main())
