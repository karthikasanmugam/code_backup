# 🚀 Daksha VLA UI - QUICKSTART GUIDE

## What You Have

A **complete web-based training and inference interface** for Daksha robot's vision-language models with support for:

- ✅ **2 Model Types**: ACT, Diffusion
- ✅ **Interactive Training**: Select model → select dataset → configure → train with live logs
- ✅ **Real-time Inference**: Deploy trained models as ZMQ servers
- ✅ **Dataset Management**: Browse and validate datasets
- ✅ **System Monitoring**: CPU/GPU/disk usage tracking
- ✅ **API Integration**: REST API for automation

---

## ⚡ 5-Minute Setup

### Step 1: Run Setup Script (2 minutes)

```bash
cd /home/s1/.ihub/vla/lerobot_2/ui/ui
bash setup.sh
```

This will:
- ✓ Create Python virtual environment
- ✓ Install all dependencies
- ✓ Make scripts executable
- ✓ Verify configuration

### Step 2: Start the UI (1 minute)

```bash
bash start_ui.sh
```

You'll see:
```
════════════════════════════════════════════════════════════════════
Daksha VLA UI - Starting Server
════════════════════════════════════════════════════════════════════
Configuration:
  Host: 0.0.0.0
  Port: 5000
  Debug: false

UI will be available at: http://localhost:8104
```

### Step 3: Open Browser (30 seconds)

Visit: **http://localhost:8104**

Done! 🎉

## Launch with ROS 2

The `daksha_vla_ui` ament CMake package provides a ROS 2 launch entry point for
the existing checkout. It keeps the Flask API and frontend unchanged and requires
the app's existing Python/LeRobot dependencies to be available to the ROS 2 Python.

Add this checkout to a ROS 2 workspace and build it:

```bash
mkdir -p ~/ros2_ws/src
ln -s /home/s1/.ihub/vla/lerobot_2 ~/ros2_ws/src/daksha_vla_ui
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select daksha_vla_ui
source install/setup.bash
```

Start the UI with the defaults configured in `launch/app.launch.py`
(`project_root`, `host`, and `port`):

```bash
ros2 launch daksha_vla_ui app.launch.py
```

Launch arguments can still override those defaults if needed. The launch file
also accepts `debug:=true`; as in direct startup, debug mode is restricted to
loopback. Stop the UI with Ctrl+C. Existing HTTP endpoints,
frontend assets, training/inference scripts, and shutdown handling remain in
the checkout and are invoked by the ROS 2-launched process.

---

## 📖 Complete Workflow

### Training a Model (Start to Finish)

#### 1. **Select Model** (30 seconds)
   - Click on model card (ACT or Diffusion)
   - Card will highlight in cyan when selected

#### 2. **Select Dataset** (30 seconds)
   - Click "Refresh" in "Select Dataset" card
   - Choose from dropdown or click from Datasets tab
   - Dataset info shows size and creation date

#### 3. **Configure Training** (1 minute)
   - Adjust hyperparameters as needed:
     ```
     Default Good Starting Values:
     • ACT: batch_size=8, steps=50000
     • Diffusion: batch_size=4, steps=50000
     • Pi-0.5: batch_size=16, steps=30000
     • SmolVLA: batch_size=4, steps=50000
     ```
   - Enter job name or leave blank for auto-generated

#### 4. **Start Training** (10 seconds)
   - Click "▶️ Start Training"
   - Status changes to "Training (model_type)"

#### 5. **Monitor Progress** (Real-time)
   - **Progress Bar**: Shows % complete
   - **Current Step**: X / Total Steps
   - **Live Logs**: Scrolling output from training
   - **Stats**: CPU/GPU usage updates every 2 seconds

#### 6. **Stop Training** (Optional)
   - Click "⏹️ Stop" button
   - Process terminates gracefully

**Total Training Time**: 
- ACT: 2-6 hours (depends on dataset size)
- Diffusion: 2-6 hours
- Pi-0.5: 1-4 hours
- SmolVLA: 2-6 hours

---

### Running Inference (After Training)

#### 1. **Navigate to Inference Tab** (10 seconds)
   - Click "🚀 Inference" tab
   - Shows Server Management section

#### 2. **Refresh Checkpoints** (30 seconds)
   - Go to "💾 Checkpoints" tab
   - Click "🔄 Refresh Checkpoints"
   - See all trained models listed

#### 3. **Select & Start Server** (30 seconds)
   - Back to Inference tab
   - Checkpoint auto-selected (or choose from dropdown)
   - Click "▶️ Start Server"
   - Status changes to "Online"

#### 4. **Run Inference** (Using Python)

**Quick Test:**
```bash
cd /home/s1/.ihub/vla/lerobot_2/ui/ui
source venv/bin/activate
python3 inference_client.py --test --num-tests 5
```

**In Your Code:**
```python
from inference_client import DakshaInferenceClient, InferenceConfig
import numpy as np

# Configure
config = InferenceConfig(
    server_host="localhost",
    server_port=5555,
    policy_type="act",
)

# Run inference
with DakshaInferenceClient(config) as client:
    # Prepare observation (batch of images + state)
    observation = {
        "image_left_c1": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "image_left_c2": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "image_wrist_c1": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "image_wrist_c2": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        "state": np.random.randn(16).astype(np.float32),
    }
    
    # Get action from model
    action = client.infer(observation)
    print(f"Predicted action: {action}")  # 16-DOF arm control
```

#### 5. **Stop Server** (Optional)
   - Click "⏹️ Stop" button
   - Server goes offline

---

## 💾 Dataset Organization

Make sure your datasets are in this format:

```
my_dataset/
├── episodes/
│   ├── episode_0/
│   │   ├── left_c1.mp4
│   │   ├── left_c2.mp4
│   │   ├── wrist_c1.mp4
│   │   ├── wrist_c2.mp4
│   │   ├── action.npy
│   │   └── state.npy
│   ├── episode_1/
│   └── ...
└── meta/
    ├── info.json
    ├── stats.json
    ├── tasks.jsonl
    └── episodes.jsonl
```

**Required Cameras**:
- `left_c1` - Left arm camera 1
- `left_c2` - Left arm camera 2
- `wrist_c1` - Wrist camera 1
- `wrist_c2` - Wrist camera 2

**Dimensions**:
- Camera frames: 480×640 pixels, RGB
- Actions: 16 DOF (dual arm)
- States: 16 DOF (joint angles)

---

## 🎛️ UI Tabs Explained

### 🏋️ Training Tab
- **Model Selection**: Choose architecture
- **Dataset Selection**: Pick training data
- **Configuration**: Set hyperparameters
- **Progress**: Real-time monitoring

### 🚀 Inference Tab
- **Server Management**: Start/stop inference server
- **Configuration**: Port, checkpoint selection
- **Status**: When server is active/inactive
- **Logs**: Real-time server output

### 📊 Datasets Tab
- **List View**: All available datasets
- **Info**: Size, creation date, path
- **Select**: Quick selection for training

### 💾 Checkpoints Tab
- **List View**: All trained models
- **Info**: Creation date, model path
- **Load**: Select for inference

---

## 🔧 Configuration Files

### Model Configuration (`model_configs.json`)

Define models, hyperparameters, and requirements:

```json
{
  "models": {
    "act": {
      "name": "Action Chunking Transformer",
      "type": "act",
      "pretrained": "facebook/dinov2-base",
      "hyperparameters": {
        "batch_size": 8,
        "learning_rate": 1e-4,
        "steps": 50000
      }
    }
  }
}
```

### Environment Variables

```bash
# Start on different port
DAKSHA_PORT=5001 bash start_ui.sh

# Enable debug mode
DAKSHA_DEBUG=true bash start_ui.sh

# Set custom host
DAKSHA_HOST=0.0.0.0 bash start_ui.sh
```

---

## 📝 Common Tasks

### Change Model During Training?
❌ Not possible - stop training, restart with new model

### Use Bigger Batch Size?
✅ Edit hyperparameter before clicking Start

### Training Too Slow?
✅ Try Pi-0.5 (fastest) or reduce batch size

### GPU Out of Memory?
✅ Reduce batch_size by half, increase gradient_accumulation

### Access UI from Another Computer?
✅ Use server IP instead of localhost:
```
http://your.server.ip:8104
```

### Automate Training?
✅ Use REST API:
```bash
curl -X POST http://localhost:8104/api/train/start \
  -H "Content-Type: application/json" \
  -d '{
    "policy_type": "act",
    "dataset_path": "/path/to/dataset",
    "config": {"batch_size": 8, "steps": 50000}
  }'
```

---

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Port 5000 in use | `DAKSHA_PORT=5001 bash start_ui.sh` |
| "Failed to load policies" | Check `model_configs.json` is valid JSON |
| Training won't start | Verify dataset path exists |
| Server won't connect | Check firewall, verify port |
| Out of memory | Reduce batch_size by half |
| Slow inference | Check server logs, reduce input size |

---

## 📚 File Reference

| File | Purpose |
|------|---------|
| `../../app.py` | Flask backend (APIs, training control) |
| `start_ui.sh` | Main startup script |
| `setup.sh` | One-time setup script |
| `test.sh` | Verify installation |
| `inference_client.py` | Python ZMQ inference client |
| `frontend/templates/index.html` | Web UI layout |
| `frontend/static/js/app.js` | Frontend logic |
| `model_configs.json` | Model definitions |
| `requirements.txt` | Python dependencies |

---

## 🎯 Key Directories

```
/home/ihub/vla/lerobot/
├── examples/daksha/
│   ├── ui/                          ← YOU ARE HERE
│   ├── smolvla/                     ← SmolVLA training (not shown in UI)
│   ├── act/                         ← ACT training
│   ├── diffusion/                   ← Diffusion training
│   └── pi_0_5/                      ← Pi-0.5 training (not shown in UI)
├── outputs/daksha/                  ← All trained checkpoints
└── examples/training/               ← Training scripts reference
```

---

## 🚀 Next Steps

1. ✅ Run setup.sh
2. ✅ Start the UI
3. ✅ Select a model type (try ACT first)
4. ✅ Select a dataset
5. ✅ Click Start Training
6. ✅ Watch the logs
7. ✅ Use checkpoint for inference

---

## 💡 Pro Tips

1. **Start Small**: Use 10,000-20,000 steps for testing, then scale up
2. **Save Frequently**: Training saves every 5000 steps by default
3. **Monitor Resources**: Watch CPU/GPU during training
4. **Batch Inference**: Use `batch_infer()` for faster processing
5. **Multiple Users**: If on same server, use different ports
6. **Backup Checkpoints**: Copy `/home/ihub/vla/lerobot/outputs/daksha` regularly

---

## 🎓 Model Selection Guide

**Choose Based On Your Needs:**

| Model | Speed | Accuracy | Flexibility | Use Case |
|-------|-------|----------|-------------|----------|
| **Pi-0.5** | ⚡⚡⚡ | ✓✓ | ✓ | Real-time robotics |
| **ACT** | ⚡⚡ | ✓✓✓ | ✓✓ | Discrete actions |
| **Diffusion** | ⚡ | ✓✓✓ | ✓✓✓ | Smooth trajectories |
| **SmolVLA** | ⚡ | ✓✓✓ | ✓✓✓ | Language understanding |

---

## 📞 Support Resources

- **Logs**: Check `/tmp/daksha_logs/` for detailed errors
- **Issues**: Read README.md in this directory
- **API**: Check app.py docstrings for route details
- **Inference**: See inference_client.py for examples

---

## ✨ Features Summary

✅ Web-based UI - no command line needed
✅ Live training monitoring  
✅ Dataset browser and validator
✅ Checkpoint management
✅ Real-time log streaming
✅ GPU resource monitoring
✅ No manual config files to edit
✅ REST API for automation
✅ Multi-policy support
✅ Production-ready server

---

**Ready?** 👉 Run `bash setup.sh && bash start_ui.sh`

Enjoy! 🎉
