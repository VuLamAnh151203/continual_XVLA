#!/usr/bin/env bash
set -euo pipefail

SOURCE_PATH="${BASH_SOURCE[0]}"
if [[ "${SOURCE_PATH}" == */* ]]; then
  SCRIPT_DIR="$(cd "${SOURCE_PATH%/*}" && pwd)"
else
  SCRIPT_DIR="$(pwd)"
fi
cd "${SCRIPT_DIR}"

# ============================================================
# USER CONFIGURATION
# ============================================================

# Set this through Kaggle secrets or your shell. Do not hardcode tokens here.
export HF_TOKEN="${HF_TOKEN:-}"
export PYTHON_BIN="${PYTHON_BIN:-python}"

export WORKDIR="${WORKDIR:-/kaggle/working}"
export DATASET_INPUT_ROOT="${DATASET_INPUT_ROOT:-/kaggle/input/datasets/anhvlm/libero-dataset/libero_dataset/IPEC-COMMUNITY}"
export DATASET_WORK_ROOT="${DATASET_WORK_ROOT:-${WORKDIR}/IPEC-COMMUNITY}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/outputs/continual_learning}"
export RUN_NAME="${RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
export RESULTS_FILE="${RESULTS_FILE:-${OUTPUT_ROOT}/results.json}"
export EVAL_RESULTS_FILE="${EVAL_RESULTS_FILE:-${OUTPUT_ROOT}/evaluation_results.json}"
export RUN_LOG_FILE="${RUN_LOG_FILE:-${OUTPUT_ROOT}/run.log}"

# LIBERO asks interactively for this on first import. We create the config
# up front so Kaggle runs do not stop at the Y/N prompt.
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${WORKDIR}/.libero}"
export LIBERO_DATASET_PARENT="${LIBERO_DATASET_PARENT:-${WORKDIR}/libero}"
export LIBERO_DATASETS="${LIBERO_DATASETS:-${LIBERO_DATASET_PARENT}/datasets}"
export LIBERO_BENCHMARK_ROOT="${LIBERO_BENCHMARK_ROOT:-}"
export WRITE_LIBERO_CONFIG="${WRITE_LIBERO_CONFIG:-1}"

export BASE_MODEL="${BASE_MODEL:-lerobot/xvla-base}"
export EVAL_POLICY_PATH="${EVAL_POLICY_PATH:-}"

export SUITES="${SUITES:-libero_spatial,libero_goal,libero_10,libero_object}"
export CONVERT_SUITES="${CONVERT_SUITES:-libero_10,libero_object,libero_goal,libero_spatial}"

export LIBERO_SPATIAL_REPO_ID="${LIBERO_SPATIAL_REPO_ID:-IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot}"
export LIBERO_GOAL_REPO_ID="${LIBERO_GOAL_REPO_ID:-IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot}"
export LIBERO_10_REPO_ID="${LIBERO_10_REPO_ID:-IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot}"
export LIBERO_OBJECT_REPO_ID="${LIBERO_OBJECT_REPO_ID:-IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot}"

export TRAIN_TASK_IDS="${TRAIN_TASK_IDS:-0,1,2,3}"
export TEST_TASK_IDS="${TEST_TASK_IDS:-0,1,2,3}"
export TRAIN_STEPS="${TRAIN_STEPS:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export DEVICE="${DEVICE:-cuda}"
export CONTROL_MODE="${CONTROL_MODE:-absolute}"
export N_EVAL_EPISODES="${N_EVAL_EPISODES:-10}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

export REPLAY_TRANSITIONS="${REPLAY_TRANSITIONS:-0}"
export REPLAY_TRANSITIONS_PER_TASK="${REPLAY_TRANSITIONS_PER_TASK:-1000}"
export REPLAY_SEED="${REPLAY_SEED:-0}"
export REPLAY_ROOT="${REPLAY_ROOT:-${OUTPUT_ROOT}/runs/${RUN_NAME}/replay}"
export REPLAY_STRICT_COUNTS="${REPLAY_STRICT_COUNTS:-0}"

# Workflow toggles. Defaults train the first suite, patch the produced
# checkpoint, and evaluate it on the held-out task ids.
export INSTALL_DEPS="${INSTALL_DEPS:-1}"
export COPY_DATASETS="${COPY_DATASETS:-1}"
export CONVERT_DATASETS="${CONVERT_DATASETS:-1}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_EVAL="${RUN_EVAL:-1}"
export PATCH_CHECKPOINT="${PATCH_CHECKPOINT:-1}"
export TRAIN_ALL_SUITES="${TRAIN_ALL_SUITES:-1}"
export DRY_RUN="${DRY_RUN:-0}"

# ============================================================
# INTERNAL CONSTANTS
# ============================================================

export CUDNN_BENCHMARK="${CUDNN_BENCHMARK:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

export ENV_MAX_PARALLEL_TASKS="${ENV_MAX_PARALLEL_TASKS:-1}"
export EVAL_FREQ="${EVAL_FREQ:-0}"

export POLICY_PUSH_TO_HUB="${POLICY_PUSH_TO_HUB:-0}"
export POLICY_DTYPE="${POLICY_DTYPE:-}"
export OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
export SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-30000}"
export SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-2.5e-6}"
export ACTION_MODE="${ACTION_MODE:-ee6d}"
export POLICY_NUM_IMAGE_VIEWS="${POLICY_NUM_IMAGE_VIEWS:-3}"
export POLICY_EMPTY_CAMERAS="${POLICY_EMPTY_CAMERAS:-1}"
export FREEZE_VISION_ENCODER="${FREEZE_VISION_ENCODER:-0}"
export FREEZE_LANGUAGE_ENCODER="${FREEZE_LANGUAGE_ENCODER:-0}"
export TRAIN_POLICY_TRANSFORMER="${TRAIN_POLICY_TRANSFORMER:-1}"
export TRAIN_SOFT_PROMPTS="${TRAIN_SOFT_PROMPTS:-1}"

export PEFT_METHOD_TYPE="${PEFT_METHOD_TYPE:-LORA}"
export PEFT_R="${PEFT_R:-32}"
export PEFT_TARGET_MODULES="${PEFT_TARGET_MODULES:-all-linear}"
export MERGE_LORA_BETWEEN_SUITES="${MERGE_LORA_BETWEEN_SUITES:-1}"

export RENAME_MAP="${RENAME_MAP:-{\"observation.images.wrist_image\":\"observation.images.image2\"}}"

export TOKENIZER_NAME="${TOKENIZER_NAME:-facebook/bart-large}"
export TOKENIZER_MAX_LENGTH="${TOKENIZER_MAX_LENGTH:-50}"
export TOKENIZER_TASK_KEY="${TOKENIZER_TASK_KEY:-task}"
export DOMAIN_ID="${DOMAIN_ID:-3}"
export NORMALIZER_EPS="${NORMALIZER_EPS:-1e-08}"
export IMAGE_SHAPE="${IMAGE_SHAPE:-3,224,224}"
export STATE_SHAPE="${STATE_SHAPE:-8}"
export ACTION_SHAPE="${ACTION_SHAPE:-20}"
export PREPROCESSOR_DEVICE="${PREPROCESSOR_DEVICE:-${DEVICE}}"
export POSTPROCESSOR_DEVICE="${POSTPROCESSOR_DEVICE:-cpu}"

if [[ -z "${HF_TOKEN}" ]]; then
  echo "Warning: HF_TOKEN is not set. This is OK only if required assets are public or cached." >&2
fi

# LIBERO config is created by libero_continual_replay.py after optional dependency
# installation, so first-time runs do not fail before pip install.

"${PYTHON_BIN}" libero_continual_replay.py
