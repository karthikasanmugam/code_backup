# Daksha VLA Training & Inference UI - Comprehensive Documentation

## 📋 Overview

The Daksha VLA UI provides a comprehensive web-based interface for:

1. **Model Selection**: Choose between ACT and Diffusion policies
2. **Dataset Management**: Browse, select, and validate datasets
3. **Training**: Configure and monitor model training with real-time logs
4. **Inference**: Deploy trained models as ZMQ servers and run inference
5. **Monitoring**: Track system resources and training progress

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd /home/s1/.ihub/vla/lerobot_2/ui/ui
bash start_ui.sh
```

This script will:
- Check Python 3 installation
- Create virtual environment
- Install dependencies
- Verify installations
- Start the Flask server

### 2. Access the UI

Open your browser and navigate to:
```
http://localhost:8104
```

---

## 📦 Project Structure

```
lerobot_2/
├── app.py                      # Flask backend application
└── ui/ui/
    ├── frontend/               # HTML, CSS, JavaScript, and assets
    ├── inference_client.py     # ZMQ inference client
    ├── start_ui.sh             # Startup script
    ├── requirements.txt         # Python dependencies
    └── model_configs.json      # Model configurations & hyperparameters
```

---

## 🎯 Features

### Model Types

#### 1. **ACT (Action Chunking Transformer)**
- **Best for**: Discrete, chunked action prediction
- **Batch Size**: 8 (default)
- **Steps**: 50,000
- **Features**:
  - Action chunking for temporal consistency
  - Multi-camera support
  - Memory efficient

#### 2. **Diffusion Policy**
- **Best for**: Continuous, smooth action trajectories
- **Batch Size**: 4 (default)
- **Steps**: 50,000
- **Features**:
  - Latent diffusion model
  - Smooth trajectory generation
  - Noise scheduling

#### 3. **Pi-0.5**
- **Best for**: Real-time inference
- **Batch Size**: 16 (default)
- **Steps**: 30,000
- **Features**:
  - Lightweight architecture
  - Low-latency inference
  - Efficient memory usage

#### SmolVLA
- **Best for**: Vision-language understanding
- **Batch Size**: 4 (default)
- **Steps**: 3,000 (UI default)
- **Features**:
  - Vision-language alignment
  - Instruction following
  - Multimodal learning

Select **SmolVLA**, choose a dataset, and start training. The UI prepares the
dataset in LeRobot v3 format and invokes `smolvla/train_smolvla_local.sh`, including
its dataset check. The pretrained model defaults to `lerobot/smolvla_base`;
use a local model directory or enable Offline only when the required assets are cached.
The UI exposes batch size, steps, save/log frequency, workers, vision encoder freezing,
and expert-only training. Checkpoints are saved every 500 steps by default.

UI runs are stored at `outputs/daksha/smolvla/<dataset>/<job>_<timestamp>/`, with
weights and processors under `checkpoints/<step>/pretrained_model/` and optimizer
state alongside that directory. The existing retention worker keeps the latest
three complete checkpoints. The Checkpoints tab also discovers standalone runs
in `outputs/smolvla/*` and `outputs/smolvla_daksha`; these are not deletable from the UI.
Resume invokes the SmolVLA resume launcher with the selected checkpoint's saved
configuration and continues in the original output directory.

For inference, select the run in the Inference tab and start the server. The UI
detects SmolVLA from the checkpoint and launches `smolvla/serve_smolvla_zmq.py`.
Task, action horizon, joint ordering, and bounds come from the saved training
configuration and dataset metadata. Keep that dataset available at its saved path.
Start the ROS2 bridge separately using the UI bridge controls.

New UI ACT, Diffusion, and SmolVLA runs pass `dataset.eval_split=0.1` after conversion. LeRobot holds
out the last `ceil(episode_count * 0.1)` episodes per task; frames from an episode
never cross the split. Small datasets therefore have an approximate 90/10 ratio
(six episodes of one task become five training and one validation episode).
The split must leave at least one training episode. The saved `train_config.json`
records the split fraction and episode restrictions; resume preserves these settings.
Validation loss is computed at the save interval. ACT and diffusion use the same
split controls; their remaining training settings are unchanged.

The **Evaluation** tab selects a converted dataset from `DAKSHA_DATASET_ROOT`
(default `/home/s1/Desktop/datasets`), a checkpoint path, and ACT, Diffusion, or
SmolVLA. It measures each policy's native loss on sampled recorded frames from
both splits, using saved processors and temporal windows. Compare losses within
the same policy, not across different policy objectives. This is offline loss,
not robot success rate. Old checkpoints without a training split are explicitly
marked as retrospective evaluation, not held-out validation.

Reports and logs are stored in `outputs/evaluation/<policy>_<timestamp>/` as
`result.json` and `evaluation.log`. Reports include the exact episode lists and
sample counts. Stop training/inference before launching evaluation to free resources.
The Checkpoints tab refreshes every five seconds and on entry. Incomplete SmolVLA
saves (for example, interrupted after writing only config.json) are excluded from
the saved-step list.

---

## 🏋️ Training

### Step 1: Select Model

Click on a model card to select it. Selected model will be highlighted in cyan.

### Step 2: Select Dataset

1. Click "Refresh" to load available datasets
2. Select a dataset from the dropdown or from the Datasets tab
3. Dataset information will display (size, created date, path)

### Step 3: Configure Training

Adjust hyperparameters:

```
• Batch Size (4-256): Number of samples per update
• Learning Rate (1e-6 to 1e-2): Optimizer learning rate
• Steps (1000-500000): Total training iterations
• Policy-specific parameters (chunk size, diffusion steps, etc.)
```

### Step 4: Start Training

1. Enter a job name (optional, auto-generated if empty)
2. Click "Start Training"
3. Monitor progress in real-time:
   - Progress bar shows training completion
   - Current step / total steps displayed
   - Live log streaming from training process
   - CPU/GPU stats update automatically

### Step 5: Monitor

**Training Progress shows:**
- Status indicator (green = running, gray = idle)
- Current step / total steps
- Progress percentage
- Real-time logs

**To Stop Training:**
- Click the "Stop" button
- Process will terminate gracefully

---

## 🚀 Inference

### Step 1: Load Checkpoint

1. Click "Refresh Checkpoints" in Checkpoints tab
2. Select a checkpoint (trained model)
3. Go to Inference tab

### Step 2: Start Server

1. Checkpoint automatically selected
2. Configure server port (default: 5555)
3. Click "Start Server"
4. Wait for "Online" status

### Step 3: Run Inference

**Using Python Client:**

```python
from inference_client import DakshaInferenceClient, InferenceConfig

# Create config
config = InferenceConfig(
    server_host="localhost",
    server_port=5555,
    policy_type="act",
)

# Run inference
with DakshaInferenceClient(config) as client:
    observation = {
        "image_left_c1": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "image_left_c2": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "image_wrist_c1": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "image_wrist_c2": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "state": np.random.randn(16).astype(np.float32),
    }
    action = client.infer(observation)
    print(f"Action: {action}")
```

**Command Line Test:**

```bash
python3 inference_client.py --test --num-tests 5
```

---

## 📊 Dataset Management

### Dataset Requirements

All datasets must contain:

```
dataset_root/
├── episodes/              # Video data
│   ├── episode_0/
│   ├── episode_1/
│   └── ...
├── meta/
│   ├── info.json         # Dataset metadata
│   ├── stats.json        # Action/state statistics
│   ├── tasks.jsonl       # Task definitions
│   └── episodes.jsonl    # Episode information
```

### Required Cameras

For Daksha, datasets must have these camera streams:
- `left_c1` (left arm camera 1)
- `left_c2` (left arm camera 2)
- `wrist_c1` (wrist camera 1)
- `wrist_c2` (wrist camera 2)

### Action/State Dimensions

- **Action Dimension**: 16 DOF (dual-arm control)
- **State Dimension**: 16 DOF (joint positions)
- **Frame Rate**: 10 Hz (default)

---

## ⚙️ Configuration

### Model Configurations (`model_configs.json`)

```json
{
  "models": {
    "policy_type": {
      "name": "Display name",
      "type": "identifier",
      "pretrained": "model path",
      "hyperparameters": {
        "param_name": default_value
      }
    }
  }
}
```

### Training Defaults

Edit in `model_configs.json`:

```json
"training_defaults": {
  "device": "cuda",
  "dtype": "bfloat16",
  "compile_model": true,
  "gradient_checkpointing": true,
  "save_freq": 5000,
  "log_freq": 20,
  "num_workers": 0
}
```

---

## 🔧 API Reference

### Backend Routes

#### Status
```
GET /api/status
→ Returns: training status, inference status, system stats
```

#### Policies
```
GET /api/policies
→ Returns: all available policy types and configurations
```

#### Datasets
```
GET /api/datasets
→ Returns: list of available datasets

POST /api/validate_dataset
→ Validates dataset path and returns statistics
```

#### Training
```
POST /api/train/start
→ Starts training with given configuration

POST /api/train/stop
→ Stops active training

GET /api/train/status
→ Returns training progress

GET /api/train/logs
→ Streams training logs (Server-Sent Events)
```

#### Inference
```
POST /api/inference/start_server
→ Starts inference server with checkpoint

POST /api/inference/stop_server
→ Stops inference server

GET /api/inference/logs
→ Streams server logs

GET /api/checkpoints
→ Lists available checkpoints
```

---

## 🐛 Troubleshooting

### Issue: "Failed to load policies"

**Solution**: Check that `model_configs.json` exists and is valid JSON.

```bash
python3 -m json.tool model_configs.json
```

### Issue: "Port already in use"

**Solution**: Either:
1. Kill existing process: `lsof -i :8104 | kill -9 <PID>`
2. Change port: `DAKSHA_PORT=5001 bash start_ui.sh`

### Issue: "Training won't start"

**Check**:
1. Dataset path is valid: `ls -la /path/to/dataset`
2. Output directory writable: `touch /home/ihub/vla/lerobot/outputs/test`
3. Python environment activated: `which python`

### Issue: "Server connection failed"

**Check**:
1. Server is running (check Inference → Server Status)
2. Port forwarding if remote: `ssh -L 5555:localhost:5555 user@server`
3. Firewall allows connection: `netstat -tuln | grep 5555`

---

## 📈 Monitoring

### System Resources

Monitor in real-time from dashboard:
- **CPU Usage**: Updated every 2 seconds
- **GPU Memory**: Updated every 2 seconds
- **Process Stats**: Shown during training

### Training Logs

Real-time streaming of:
- Training loss
- Validation metrics
- Learning rate schedules
- Checkpoint saves
- Dataset validation output

### Inference Logs

Real-time monitoring of:
- Client connections
- Inference requests/responses
- Error messages
- Performance metrics

---

## 🔐 Security Notes

- **Local Deployment**: SSL/TLS not enabled by default. For production:
  ```bash
  # Use nginx reverse proxy with SSL
  ```

- **Authentication**: No authentication by default. Add if needed:
  ```python
  from flask_httpauth import HTTPBasicAuth
  ```

- **File Access**: Be careful with dataset paths. Validate all user inputs.

---

## 📝 Example Workflows

### Complete Training Workflow

```bash
# 1. Start UI
cd /home/s1/.ihub/vla/lerobot_2/ui/ui
bash start_ui.sh

# 2. In browser: http://localhost:8104
# - Select "ACT" model
# - Select dataset from dropdown
# - Set batch_size=8, steps=50000
# - Click "Start Training"
# - Monitor logs and progress

# 3. After training, go to Inference tab
# - Select trained checkpoint
# - Click "Start Server"
# - Server runs on port 5555
```

### Complete Inference Workflow

```bash
# 1. Start inference server from UI (as above)

# 2. In another terminal, run client:
python3 /home/s1/.ihub/vla/lerobot_2/ui/ui/inference_client.py \
  --host localhost \
  --port 5555 \
  --policy-type act \
  --test \
  --num-tests 5

# 3. Or integrate into your robot code:
from inference_client import DakshaInferenceClient, InferenceConfig
client = DakshaInferenceClient(config)
action = client.infer(observation)
```

---

## 📚 References

- **LeRobot**: https://github.com/huggingface/lerobot
- **Policy Types**: See `model_configs.json` for detailed configurations
- **Dataset Format**: Check `/home/ihub/vla/lerobot/examples/daksha/smolvla/check_dataset_smolvla.py`
- **Training**: Check training scripts in `/home/ihub/vla/lerobot/examples/daksha/{act,diffusion,pi_0_5,smolvla}/`

---

## 💡 Tips & Best Practices

1. **Dataset Validation**: Always run dataset check before training
2. **Hyperparameter Tuning**: Start with defaults, then adjust based on results
3. **Checkpoints**: Save frequently (default: every 5000 steps)
4. **Inference**: Use batch inference for efficiency
5. **Monitoring**: Keep logs window open to catch errors early
6. **GPU Memory**: Reduce batch size if OOM errors occur

---

## Support

For issues and questions:
1. Check logs (both UI and system logs)
2. Verify dataset format
3. Check Python environment
4. Review training hyperparameters

Last Updated: 2026-07-21
