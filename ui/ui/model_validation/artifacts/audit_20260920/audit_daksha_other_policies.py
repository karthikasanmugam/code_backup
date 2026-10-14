import gc
import json
import os
import sys
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
import numpy as np
import torch

ROOT = Path('/home/s1/.ihub/vla/lerobot')
sys.path.insert(0, str(ROOT/'examples/daksha/ui/model_validation'))
from vla_eval.datasets.lerobot_custom import LeRobotCustomDataset
from vla_eval.models.diffusion.adapter import DiffusionAdapter
from vla_eval.models.smolvla.adapter import SmolVLAAdapter

torch.set_num_threads(4)
dataset = LeRobotCustomDataset(str(ROOT/'examples/daksha/ui/model_validation/sample/place_blue_new_old'))
sample = dataset[0]
dataset.close()
cases = [
    ('diffusion', DiffusionAdapter, 'outputs/daksha/diffusion/place_blue_box/test_diffusion_20260915_102305/checkpoints/000050/pretrained_model'),
    ('smolvla', SmolVLAAdapter, 'outputs/daksha/smolvla/place_blue_new/sample_diffusion_20260912_095245/checkpoints/030000/pretrained_model'),
]
report = {}
for name, cls, path in cases:
    print('LOADING', name, flush=True)
    model = cls(device='cuda', inference_profile='checkpoint')
    try:
        model.load(str(ROOT/path))
        model.reset()
        torch.manual_seed(0)
        prediction = model.predict([sample.observation]*model.get_metadata().obs_horizon)
        report[name] = {'checkpoint':path,'shape':list(prediction.actions.shape),
                        'finite':bool(np.isfinite(prediction.actions).all()),
                        'task':sample.observation.instruction,
                        'latency_s':prediction.inference_latency_s,
                        'runtime':model.get_metadata().extra['runtime']}
        print(name, json.dumps(report[name]), flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        report[name] = {'error':str(exc)}
    del model
    gc.collect()
    torch.cuda.empty_cache()
Path('/tmp/daksha_other_policies_audit.json').write_text(json.dumps(report,indent=2))
