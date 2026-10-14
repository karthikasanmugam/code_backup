# Diffusion checkpoint adapter

This folder contains the Diffusion adapter matched to the supplied LeRobot checkpoint configuration.

The checkpoint interface is:

- `n_obs_steps = 2`
- `horizon = 64`
- `n_action_steps = 32`
- action dimension = 16
- state = 16
- leader state = 16
- four RGB cameras at `3 x 240 x 320`
- `VISUAL = MEAN_STD`
- `STATE = MIN_MAX`
- `ACTION = MIN_MAX`
- DDPM scheduler, 100 training timesteps, epsilon prediction

The adapter deliberately does **not** hard-code `num_inference_steps=100` because the supplied checkpoint has `num_inference_steps = null`. It lets the installed LeRobot policy/runtime resolve that value.

## Integration

Copy these files into the existing project at the corresponding paths:

- `vla_eval/models/diffusion/config.py`
- `vla_eval/models/diffusion/compatibility.py`
- `vla_eval/models/diffusion/preprocessing.py`
- `vla_eval/models/diffusion/adapter.py`
- `vla_eval/models/diffusion/__init__.py`
- `vla_eval/models/diffusion.yaml`
- `tests/test_diffusion.py`

## Smoke test

PowerShell example:

```powershell
python tests/test_diffusion.py `
  --checkpoint "C:\path\to\pretrained_model" `
  --dataset "D:\Projects\test\sample\place_blue_new_old" `
  --episode 0 `
  --device cuda
```

The UI/evaluation layer can call the adapter exactly like ACT, but it must provide a two-observation history to `predict()`.
