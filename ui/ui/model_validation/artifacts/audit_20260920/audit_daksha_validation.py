import gc
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/home/s1/.ihub/vla/lerobot')
sys.path.insert(0, str(ROOT / 'examples/daksha/ui/model_validation'))
from vla_eval.datasets.lerobot_custom import LeRobotCustomDataset
from vla_eval.models.act.adapter import ACTAdapter
from vla_eval.models.act.preprocessing import observation_to_act_batch
from vla_eval.evaluation.act_episode import calculate_metrics

torch.set_num_threads(4)
print('CUDA available:', torch.cuda.is_available(), flush=True)
dataset = LeRobotCustomDataset('/home/s1/Desktop/datasets/pick_place_brown_300')
groups = {}
for ep in dataset.episodes:
    groups.setdefault(ep.task, []).append(ep.episode_index)
held_out = sorted(i for eps in groups.values() for i in eps[-math.ceil(len(eps)*0.1):])
selected = [held_out[0], held_out[len(held_out)//2], held_out[-1]]
samples = []
for ep in selected:
    frames = dataset.get_episode_frames(ep)
    samples.extend(dataset[frames[i]] for i in [0,len(frames)//2,len(frames)-1])
dataset.close()
report = {'dataset': str(dataset.root), 'held_out_candidates': held_out,
          'sample_frames': [[s.observation.episode_index,s.observation.frame_index] for s in samples],
          'checkpoints': {}}
base = ROOT/'outputs/daksha/act/pick_place_brown_300/demo_1_20260918_094123/checkpoints'
for step in ['006500','015500']:
    adapter = ACTAdapter(device='cuda')
    adapter.load(str(base/step/'pretrained_model'))
    print('LOADED',step, flush=True)
    original, corrected = [], []
    ranges = {}
    for sample in samples:
        original.append(adapter.predict([sample.observation]).actions[0])
        batch = observation_to_act_batch(sample.observation, adapter.cfg.camera_keys)
        for key in adapter.cfg.camera_keys:
            batch[key] = batch[key]/255.0
        batch = adapter.preprocessor(batch)
        if not ranges:
            key = adapter.cfg.camera_keys[0]
            bad = adapter.preprocessor(observation_to_act_batch(sample.observation, adapter.cfg.camera_keys))
            ranges = {'camera':key, 'original':[bad[key].min().item(),bad[key].max().item()],
                      'scaled':[batch[key].min().item(),batch[key].max().item()]}
        with torch.inference_mode():
            chunk = adapter.postprocessor(adapter.policy.predict_action_chunk(batch))
        corrected.append(chunk[0,0].cpu().numpy())
    actual = np.stack([s.ground_truth.action for s in samples])
    result = {'chunk_size':adapter.cfg.chunk_size, 'normalized_image_range':ranges,
              'original_metrics':calculate_metrics(actual,np.stack(original)),
              'scaled_metrics':calculate_metrics(actual,np.stack(corrected)),
              'finite':bool(np.isfinite(original).all() and np.isfinite(corrected).all())}
    report['checkpoints'][step] = result
    print(step,json.dumps(result),flush=True)
    del adapter
    gc.collect()
    torch.cuda.empty_cache()
Path('/tmp/daksha_validation_audit.json').write_text(json.dumps(report,indent=2))
print('REPORT /tmp/daksha_validation_audit.json',flush=True)
