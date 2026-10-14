"""Compare validation profiles on the same recorded observations (no robot control)."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vla_eval.models.diffusion  # noqa: E402, F401
import vla_eval.models.smolvla  # noqa: E402, F401
from vla_eval.datasets.lerobot_custom import LeRobotCustomDataset  # noqa: E402
from vla_eval.models.runtime import PROFILES, load_evaluation_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", choices=("diffusion", "smolvla"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=["checkpoint", "optimized", "fast"])
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    dataset = LeRobotCustomDataset(args.dataset)
    try:
        frames = dataset.get_episode_frames(args.episode)
        samples = [dataset[index] for index in frames[:2]]
    finally:
        dataset.close()
    for profile in args.profiles:
        model = load_evaluation_model(args.model_type, args.checkpoint, args.device, profile)
        horizon = model.get_metadata().obs_horizon
        observations = [sample.observation for sample in samples][-horizon:]
        observations = [observations[0]] * (horizon - len(observations)) + observations
        model.predict(observations)  # Warm kernels; exclude checkpoint loading and warmup.
        latencies, wall_times = [], []
        for repeat in range(args.repeats):
            torch.manual_seed(repeat)
            start = time.perf_counter()
            prediction = model.predict(observations)
            wall_times.append(time.perf_counter() - start)
            latencies.append(prediction.inference_latency_s)
        print(json.dumps({
            "runtime": model.get_metadata().extra["runtime"],
            "repeats": args.repeats,
            "model_median_s": float(np.median(latencies)),
            "predict_median_s": float(np.median(wall_times)),
            "shape": list(prediction.actions.shape),
            "finite": bool(np.isfinite(prediction.actions).all()),
        }), flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
