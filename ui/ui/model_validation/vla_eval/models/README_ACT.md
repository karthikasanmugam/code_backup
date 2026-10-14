# ACT integration

The ACT adapter follows the project's `BaseVLA` contract and wraps the installed LeRobot ACT policy.

## Files

- `vla_eval/models/act/config.py` — checkpoint config parsing and compatibility checks.
- `vla_eval/models/act/preprocessing.py` — framework `Observation` -> LeRobot-style batch.
- `vla_eval/models/act/adapter.py` — model loading, reset, inference and `ActionPrediction` conversion.
- `vla_eval/models/act.yaml` — declarative copy of the supplied checkpoint configuration.
- `tests/test_act.py` — checkpoint validation and one real inference smoke test.

## Run

```powershell
python tests\test_act.py --checkpoint "D:\path\to\act_checkpoint" --device cuda
```

The smoke test constructs one synthetic observation with the exact 4 cameras and 16-D state/leader-state required by the supplied checkpoint metadata. It then runs the actual checkpoint and requires a `(50, 16)` action chunk.
