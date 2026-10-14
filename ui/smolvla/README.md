# Piper smolvla Workflow (Local Dataset)

This folder is a standalone setup for training, resuming, serving, and evaluating a `smolvla` policy on your Piper dataset without changing your existing ACT files.

## Defaults in this folder

- `dataset.repo_id`: `local/task1_full`
- `dataset.root`: `/home/physicalai/Desktop/I/V3_data_conversion/custom_datasetv3_folder/task1_full`
- `policy.type`: `smolvla`
- `policy.pretrained_path`: `lerobot/smolvla_base`
- action dim: `7`
- state dim: `7`
- cameras:
  - `observation.images.world`
  - `observation.images.front_left`
  - `observation.images.front_right`
  - `observation.images.wrist`
- smolvla image resolution: `224 x 224` inside the policy
- built-in smolvla normalization:
  - `ACTION=QUANTILES`
  - `STATE=QUANTILES`
  - `VISUAL=IDENTITY`

## 1) Create the smolvla environment

```bash
cd /home/physicalai/Desktop/I/lerobot
bash examples/daksha/smolvla/setup_smolvla_env.sh
```

## 2) Check the dataset before training

```bash
cd /home/physicalai/Desktop/I/lerobot
bash examples/daksha/smolvla/check_dataset_smolvla.sh
```

## 3) Smoke training

```bash
cd /home/physicalai/Desktop/I/lerobot
bash examples/daksha/smolvla/train_smolvla_local.sh
```

This default smoke run uses:

- `output_dir=outputs/smolvla/smolvla_task1_full_smoke`
- `job_name=smolvla_task1_full_smoke`
- `steps=500`
- `batch_size=4`

## 4) Full training

```bash
cd /home/physicalai/Desktop/I/lerobot
OUTPUT_DIR=outputs/smolvla/smolvla_task1_full \
JOB_NAME=smolvla_task1_full \
STEPS=3000 \
BATCH_SIZE=8 \
NUM_WORKERS=2 \
SAVE_FREQ=500 \
LOG_FREQ=20 \
bash examples/daksha/smolvla/train_smolvla_local.sh
```

Optional overrides:

```bash
DATASET_ROOT=/home/physicalai/Desktop/I/V3_data_conversion/custom_datasetv3_folder/task1_full \
PRETRAINED_PATH=lerobot/smolvla_base \
DEVICE=cuda \
DTYPE=bfloat16 \
COMPILE_MODEL=true \
GRADIENT_CHECKPOINTING=true \
FREEZE_VISION_ENCODER=false \
TRAIN_EXPERT_ONLY=false \
bash examples/daksha/smolvla/train_smolvla_local.sh
```

## 5) Resume training

```bash
cd /home/physicalai/Desktop/I/lerobot
RUN_PREFIX=smolvla_task1_full \
bash examples/daksha/smolvla/resume_train_smolvla_local.sh
```

Resume with a new total step target:

```bash
cd /home/physicalai/Desktop/I/lerobot
RUN_PREFIX=smolvla_task1_full \
STEPS=6000 \
SAVE_FREQ=500 \
LOG_FREQ=20 \
bash examples/daksha/smolvla/resume_train_smolvla_local.sh
```

## 6) Offline evaluation on the dataset

```bash
cd /home/physicalai/Desktop/I/lerobot
POLICY_PATH=outputs/smolvla/smolvla_task1_full/checkpoints/last/pretrained_model \
bash examples/daksha/smolvla/eval_smolvla_offline.sh
```

## 7) Start the remote smolvla policy server

```bash
cd /home/physicalai/Desktop/I/lerobot
POLICY_PATH=outputs/smolvla/smolvla_task1_full/checkpoints/last/pretrained_model \
DEVICE=cuda \
bash examples/daksha/smolvla/run_smolvla_zmq_server.sh
```

## 8) Run Piper ROS2 remote evaluation with 4 cameras

In another terminal:

```bash
cd /home/physicalai/Desktop/I/lerobot
source /opt/ros/humble/setup.bash
POLICY_HOST=127.0.0.1 \
POLICY_PORT=5555 \
LANG_INSTRUCTION=task1 \
bash examples/daksha/smolvla/run_ros2_remote_eval_4cam.sh
```

## 9) Run evaluation recording on the real Piper robot

```bash
cd /home/physicalai/Desktop/I/lerobot
EVAL_ROOT=/home/physicalai/Desktop/I/V3_data_conversion/custom_datasetv3_folder/task1_full_eval_smolvla \
POLICY_PATH=outputs/smolvla/smolvla_task1_full/checkpoints/last/pretrained_model \
bash examples/daksha/smolvla/infer_record_piper.sh
```

## Notes

- This workflow prefers `./.venv` if it exists. If not, it falls back to `uv run --extra pi`.
- The ROS2 scripts assume `rclpy` comes from your system ROS install, not pip.
- `infer_record_piper.sh` assumes your Piper robot plugin is already installed and discoverable by LeRobot.
- Training outputs are kept under `outputs/smolvla/` so they stay separate from your ACT runs.
