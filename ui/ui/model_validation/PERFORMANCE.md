# Validation inference profiles

Restart the Daksha UI and refresh the browser. In **Evaluation → Model Validation**,
select **CUDA** and choose the Diffusion / SmolVLA inference profile:

| Profile | Precision | Sampling |
| --- | --- | --- |
| Optimized (default) | BF16 on CUDA compute capability 8+, FP16 on capability 7, FP32 on older GPUs and CPU | Original checkpoint scheduler and number of steps |
| Fast preview | Same automatic precision | Diffusion: DDIM, at most 20 steps. SmolVLA: at most 5 steps |
| Reference | Original model precision without autocast | Original checkpoint scheduler and number of steps |

Optimized precision can cause small numerical differences. Fast preview changes
the sampling procedure and can affect accuracy; compare MAE/RMSE with Reference on
the same held-out episodes before relying on its scores. No checkpoint files,
camera sizes, normalization statistics, action horizons, or training settings are
rewritten. Compilation is disabled in these validation profiles to avoid the
initial compilation cost. ACT keeps its existing inference path.

Every selected frame still gets a fresh prediction; its first predicted action is
compared with the recorded action for that frame. Action chunks are not reused to
artificially improve per-frame latency. Frame stride remains an independent way to
reduce the number of frames evaluated.

The UI loads weights/processors once per job, reuses them across episodes, and
resets episode state each time. Diffusion's decoded observation cache is bounded
to the history window, including episodes whose global indices start above zero.
The selected device also overrides the device saved in the processor pipeline.
SmolVLA uses a checkpoint-local tokenizer if present, otherwise its saved tokenizer
configuration (which may require the normal Hugging Face cache).

Result JSON includes `runtime`, individual `inference_latencies_s`, and
`evaluated_frames_per_second`. Model latency measures the synchronized model call;
evaluated frames/s also includes frame decoding, preprocessing, postprocessing,
and progress updates, but excludes checkpoint loading and dataset initialization.
Dataset latency percentiles use individual frame timings rather than episode means.

To reproduce timing comparisons with a local compatible checkpoint and **v3.0**
dataset, run from the repository root:

```bash
uv run --no-sync python -u examples/daksha/ui/model_validation/scripts/benchmark_inference.py \
  --model-type diffusion \
  --checkpoint /path/to/checkpoints/000050/pretrained_model \
  --dataset /path/to/converted_dataset \
  --episode 0 --device cuda --repeats 3
```

Use `--model-type smolvla` for SmolVLA. This warms up each profile, then times
repeated predictions on the same recorded observation window with fixed seeds.
It checks finite outputs and reports shape, precision, sampling, model latency,
and total adapter latency. This is a timing smoke test, not an accuracy benchmark.
Run profiles sequentially without a concurrent training job for comparable results.

## Local timing smoke test

Measured on the Jetson AGX Orin Developer Kit (64 GB shared memory, CUDA
capability 8.7, PyTorch 2.8.0), using the first observation window of episode 0
from `sample/place_blue_new_old`. Each value is the median of three model calls
after one warmup. These numbers exclude loading and preprocessing and are not
full-episode throughput or accuracy measurements.

| Policy | Reference | Optimized | Fast preview |
| --- | ---: | ---: | ---: |
| Diffusion | 8.561 s | 6.733 s | 1.396 s |
| SmolVLA | 1.506 s | 1.032 s | 0.649 s |

Diffusion used `outputs/daksha/diffusion/place_blue_box/test_diffusion_20260915_102305/checkpoints/000050/pretrained_model`
(100 DDPM steps, or 20 DDIM steps in fast preview). SmolVLA used
`outputs/daksha/smolvla/checkpoints/030000/pretrained_model` (10 steps, or 5 in
fast preview). All profiles returned finite outputs with their expected shapes:
Diffusion `[32, 16]`, SmolVLA `[50, 16]`.
