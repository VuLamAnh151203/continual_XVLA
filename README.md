# Continual X-VLA LIBERO Run Guide

This folder contains the Bash launcher and Python runner for continual X-VLA
fine-tuning on LIBERO.

## Set The Main Paths

In `run_libero.sh`, these two paths are the most important user settings:

```bash
export WORKDIR="${WORKDIR:-/kaggle/working}"
export DATASET_INPUT_ROOT="${DATASET_INPUT_ROOT:-/kaggle/input/datasets/anhvlm/libero-dataset/libero_dataset/IPEC-COMMUNITY}"
```

`WORKDIR` must be a writable working directory. The script writes copied
datasets, checkpoints, logs, and JSON results under this directory.

`DATASET_INPUT_ROOT` must point to the input dataset parent folder that contains
the four LIBERO dataset folders:

```text
IPEC-COMMUNITY/
  libero_spatial_no_noops_1.0.0_lerobot/
  libero_goal_no_noops_1.0.0_lerobot/
  libero_10_no_noops_1.0.0_lerobot/
  libero_object_no_noops_1.0.0_lerobot/
```

Do not set `DATASET_INPUT_ROOT` to one individual suite folder. Set it to the
parent folder that contains all four suite folders.

## Kaggle Example

For Kaggle, keep `WORKDIR` as:

```bash
export WORKDIR="${WORKDIR:-/kaggle/working}"
```

Then set `DATASET_INPUT_ROOT` to your mounted dataset path. Example:

```bash
export DATASET_INPUT_ROOT="${DATASET_INPUT_ROOT:-/kaggle/input/datasets/anhvlm/libero-dataset/libero_dataset/IPEC-COMMUNITY}"
```

If your Kaggle dataset is mounted at a different path, change only this part:

```bash
/kaggle/input/datasets/anhvlm/libero-dataset/libero_dataset/IPEC-COMMUNITY
```

You can check the correct path in a Kaggle notebook with:

```bash
find /kaggle/input -maxdepth 5 -type d -name "IPEC-COMMUNITY"
```

## Local Linux Example

If running outside Kaggle, choose a writable folder for `WORKDIR` and point
`DATASET_INPUT_ROOT` to your local dataset parent:

```bash
export WORKDIR="${WORKDIR:-/home/user/libero_work}"
export DATASET_INPUT_ROOT="${DATASET_INPUT_ROOT:-/home/user/datasets/IPEC-COMMUNITY}"
```

## Related Paths

The launcher derives these paths automatically:

```bash
export DATASET_WORK_ROOT="${DATASET_WORK_ROOT:-${WORKDIR}/IPEC-COMMUNITY}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${WORKDIR}/outputs/continual_learning}"
```

With the default Kaggle paths, this means:

```text
Copied/converted datasets: /kaggle/working/IPEC-COMMUNITY
Checkpoints and results:   /kaggle/working/outputs/continual_learning
Run log:                   /kaggle/working/outputs/continual_learning/run.log
Evaluation JSON:           /kaggle/working/outputs/continual_learning/evaluation_results.json
```

## If The Dataset Is Already In WORKDIR

If your datasets are already available in `DATASET_WORK_ROOT`, you can skip the
copy step:

```bash
export COPY_DATASETS=0
export DATASET_WORK_ROOT="/path/to/IPEC-COMMUNITY"
```

The folder must still contain the same four LIBERO suite folders.

## Run

```bash
bash run_libero.sh
```
