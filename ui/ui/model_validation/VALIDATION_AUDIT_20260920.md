Audit performed on 2026-09-20 against the local source, checkpoint files, and Jetson GPU.

**The current validator is not reliable for ACT or Diffusion accuracy comparisons.** Both adapters omit image scaling that the training dataset loader performs. SmolVLA's image scaling is correct. The evaluator measures offline first-action error against recorded demonstrations; it does not measure robot task success or full action-chunk execution quality.

1. **High: ACT and Diffusion receive images on the wrong scale.**
   `vla_eval/models/act/preprocessing.py:29` explicitly avoids dividing by 255;
   `vla_eval/models/diffusion/preprocessing.py:23` does the same conversion without scaling.
   Training's `src/lerobot/datasets/video_utils.py:223` supplies float images in [0, 1]
   when `return_uint8=False`, as saved in both ACT training configurations.
   `src/lerobot/processor/normalize_processor.py:349` only computes
   `(image - mean) / (std + eps)` for MEAN_STD. The saved pipeline has no pixel-scaling step.
   Both ACT checkpoints use ImageNet means [0.485, 0.456, 0.406] and standard deviations
   [0.229, 0.224, 0.225]. On a real wrist-camera frame, the current normalized range was
   [-2.118, 1136.357]; scaling first gave [-2.118, 2.640]. Divide decoded uint8 pixels
   by 255 before the checkpoint normalizer in both adapters, and add a regression
   check against training preprocessing. Existing ACT/Diffusion scores need rerunning.

2. **Medium: held-out selection does not always reproduce training.**
   `examples/daksha/ui/app.py:1080` compares absolute dataset paths and otherwise uses
   a retrospective 10% split. For these checkpoints, both save `eval_split=0.1`, but
   `015500` saves a path from another machine. The UI marks `006500` held out and
   `015500` retrospective. Both currently select episodes 272 through 302 from the
   local dataset, so this path difference does not change the selected episodes in
   this case. Dataset identity and historical training exclusion for the copied
   checkpoint are not independently proven by matching paths or statistics.
   The helper also ignores `dataset.episodes` and `exclude_episodes`, unlike training's
   `make_train_eval_datasets`, so other runs can select the wrong test episodes.
   Prefer saved split manifests with dataset identity; reproduce episode filters
   when reconstructing older splits. A retrospective subset is not a proven holdout.

3. **Medium: ACT's device override does not reach saved processors.**
   `vla_eval/models/act/adapter.py:104` loads processors without overriding the saved
   CUDA device. Selecting CPU can move inputs back to CUDA or fail without CUDA.
   Diffusion and SmolVLA already pass the selected device to their processor overrides.

4. **Medium: compatibility checks are specific to one Daksha configuration.**
   All three expect the prescribed four cameras, 240x320 RGB images, 16-dimensional
   state/actions, and leader state. Diffusion additionally hardcodes observation
   history 2, horizon 64, action steps 32, backbone, U-Net sizes, and DDPM settings
   in `vla_eval/models/diffusion/config.py:176`. Valid policies trained with other
   supported settings can be rejected. ACT's runtime accepts both supplied chunk
   lengths, but `tests/test_act.py:36` hardcodes length 50 and rejects both checkpoints.
   Read supported dimensions and horizons from checkpoint configuration instead.

5. **Lower priority: measurement and reproducibility limitations.**
   ACT's latency timer (`vla_eval/models/act/adapter.py:186`) lacks CUDA synchronization,
   so it can undercount execution time. Diffusion and SmolVLA synchronize correctly.
   Their production validation path does not set or record random seeds or repeat
   stochastic predictions. Use the Reference profile, fixed seeds, repeated samples,
   and identical frames for checkpoint comparisons. The reported `mean_rmse` is the
   mean of per-joint RMSE values, not pooled RMSE. Aggregate native-unit scores also
   mix arm and gripper coordinates; inspect per-joint results.

The two ACT checkpoints are inference-compatible with the current ACT implementation,
but they are not interchangeable training states merely because they occupy one run folder.

| Saved property | 006500 | 015500 |
| --- | --- | --- |
| Training step | 6,500 | 15,500 |
| Chunk / action steps | 16 / 16 | 32 / 32 |
| Decoder positional embedding | [16, 512] | [32, 512] |
| Learning rate | 0.00001 | 0.0001 |
| Batch size | 1 | 2 |
| Configured total steps | 10,000 | 50,000 |
| Resume flag | true | false |
| Job name | demo_1 | act_daksha |
| Saved dataset root | /home/s1/Desktop/datasets/pick_place_brown_300 | /home/thor/Desktop/Dinesh/pick_place_brown_300 |

Both models have the same configured cameras, feature dimensions, ACT architecture
apart from chunk length, and processor JSON. Action/state means and standard deviations
match exactly, as do image means and standard deviations. The complete statistics
files differ in counts, timestamp statistics, and additional image-statistic entries.
Checkpoint-local weights were loaded; the `006500` config's historical `pretrained_path`
does not redirect this adapter to `last`.

For a diagnostic comparison, each ACT model ran on the same nine real observations:
episodes 272, 287, and 302, at frames 0, 250, and 499. Each observation received a fresh
chunk prediction. The comparison branch only divided images by 255 before applying the
saved processors and model; no source or checkpoint files were changed.

| Checkpoint | Current validator mean MAE | Correct image scale mean MAE | Correct image scale mean RMSE |
| --- | ---: | ---: | ---: |
| 006500 | 0.150252 | 0.032793 | 0.040120 |
| 015500 | 0.151201 | 0.127521 | 0.141800 |

Both produced finite outputs of their configured shapes. `006500` performed better
on this small sample after correcting preprocessing. This is not a full evaluation,
proof of generalization, or a robot deployment recommendation. The other machine's
training implementation was not available to establish its exact historical behavior.

The local Diffusion checkpoint
`outputs/daksha/diffusion/place_blue_box/test_diffusion_20260915_102305/checkpoints/000050/pretrained_model`
and SmolVLA checkpoint
`outputs/daksha/smolvla/place_blue_new/sample_diffusion_20260912_095245/checkpoints/030000/pretrained_model`
also loaded and completed one GPU prediction each using the Reference profile,
seed 0, and the first observation of `sample/place_blue_new_old`.
Diffusion returned finite [32, 16] actions using 100 DDPM steps; SmolVLA returned
finite [50, 16] actions using 10 sampling steps. These are loading/inference smoke
checks, not accuracy evaluations. Diffusion still has the image-scaling defect.
SmolVLA correctly supplies scaled images, the recorded language instruction, saved
tokenization/normalization, and action unnormalization. Its single-frame smoke check
does not establish full-dataset accuracy.

The evaluator's first-action time alignment is consistent with these local policy
implementations. ACT and SmolVLA chunks begin at the current timestep. Diffusion's
`generate_actions()` already slices from `n_obs_steps - 1`, so the evaluator should
not apply a second temporal offset. Left-padding Diffusion observation history at
episode starts also matches its training history convention. This is still an
offline comparison using recorded observations, rather than a closed-loop rollout.

Ten focused runtime/backend pytest tests passed with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
Normal pytest startup is blocked by an incompatible installed ROS `launch_testing`
plugin. Even with plugin autoload disabled, collecting the entire validation test
directory fails because `tests/test_model_registry.py` imports missing
`vla_eval.models.mock`. Passing the focused tests therefore does not establish
preprocessing correctness or full-suite health.

Detailed numeric results and reproducible diagnostic scripts are saved beside this
report under `artifacts/audit_20260920/`. Diagnostic scripts contain local paths and
require the existing LeRobot environment; GPU checks require access to Jetson devices.
