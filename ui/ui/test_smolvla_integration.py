import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app as ui


class SmolVLAIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.state = copy.copy(ui.global_state)
        self.addCleanup(ui.global_state.update, self.state)
        ui.global_state.update(training_active=False, cancel_requested=False,
                               server_process=None, bridge_process=None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        roots = patch.object(ui, "CHECKPOINT_ROOTS", [str(self.root)])
        roots.start()
        self.addCleanup(roots.stop)
        self.client = ui.app.test_client()
        for name in ("TRAIN_LOG", "TRAIN_PREP_LOG", "TRAIN_RAW_LOG", "SERVER_LOG", "BRIDGE_LOG"):
            p = patch.object(ui, name, str(self.root / name))
            p.start()
            self.addCleanup(p.stop)

    def checkpoint(self, policy="smolvla"):
        run = self.root / "standalone" / policy
        model = run / "checkpoints/000500/pretrained_model"
        model.mkdir(parents=True)
        (model / "config.json").write_text(json.dumps({"type": policy}))
        for name in ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
            (model / name).write_text("test fixture")
        (model / "train_config.json").write_text(json.dumps({
            "policy": {"type": policy}, "steps": 3000,
            "dataset": {"root": str(self.root / "dataset")},
        }))
        return run, model

    def test_policy_defaults_and_invalid_requests(self):
        policies = self.client.get("/api/policies").get_json()
        self.assertEqual(set(policies), {"act", "diffusion", "smolvla"})
        for policy in ("act", "diffusion"):
            self.assertNotIn("push_to_hub", policies[policy]["hyperparameters"])
            self.assertNotIn("hub_repo_id", policies[policy]["hyperparameters"])
        defaults = ui.smolvla_training_config({})
        self.assertEqual((defaults["steps"], defaults["batch_size"], defaults["save_freq"]), (3000, 4, 500))
        for config in ({"batch_size": 0}, {"steps": 1.5}, {"offline": "false"},
                       {"pretrained_path": ""}, {"lr": 0.1}, {"steps": None}):
            with self.subTest(config=config):
                response = self.client.post("/api/train/start", json={"policy_type": "smolvla", "config": config})
                self.assertEqual(response.status_code, 400)

    def test_training_launch_environment(self):
        config = ui.smolvla_training_config({"freeze_vision_encoder": True})
        ui.global_state.update(training_active=True, training_process=None,
                               training_cancel_event=None)
        with patch.object(ui, "ensure_dataset_v30"), \
             patch.object(ui, "validate_dataset_with_lerobot", return_value={}), \
             patch.object(ui.threading, "Thread"), \
             patch.object(ui, "resolve_repo_python_executable", return_value=ui.sys.executable), \
             patch.object(ui, "_register_managed_process"), \
             patch.object(ui, "_monitor_training_logs"), \
             patch.object(ui.subprocess, "Popen") as launch:
            ui._prepare_and_train("smolvla", str(self.root), "dataset", config, "job", "job",
                                  str(self.root / "output"), ui.POLICY_CONFIGS["smolvla"],
                                  ui.POLICY_CONFIGS["smolvla"]["script"])
            launch.assert_called_once()
            self.assertTrue(launch.call_args.args[0][1].endswith("smolvla/train_smolvla_local.sh"))
            env = launch.call_args.kwargs["env"]
            self.assertEqual(env["SMOLVLA_PYTHON"], ui.sys.executable)
            for key, value in {"STEPS": "3000", "BATCH_SIZE": "4", "SAVE_FREQ": "500",
                               "PRETRAINED_PATH": "lerobot/smolvla_base", "NUM_WORKERS": "0",
                               "VIDEO_BACKEND": "pyav", "FREEZE_VISION_ENCODER": "true",
                               "TRAIN_EXPERT_ONLY": "false", "HF_HUB_OFFLINE": "0",
                               "EVAL_SPLIT": "0.1", "EVAL_STEPS": "500",
                               "REPAIR_MISSING_QUANTILES": "true", "PUSH_TO_HUB": "false"}.items():
                self.assertEqual(env[key], value, key)
        ui.configure_smolvla_environment(env, {"offline": True})
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")

    def test_recovered_training_status_refreshes_progress(self):
        train_log = self.root / "training.log"
        process = ui.psutil.Process()
        record = {
            "pid": process.pid,
            "pgid": ui.os.getpgid(process.pid),
            "create_time": process.create_time(),
            "policy": "act",
            "stage": "training",
            "output_dir": str(self.root / "output"),
            "total_steps": 49972,
            "current_step": 10,
            "resumed": True,
            "resumed_from_step": 180,
        }
        train_log.write_text(
            "Training resumed (180 / 49972 steps)\n"
            "Training: 1/49792 [00:02, 2.4s/step]\n"
        )
        with patch.object(ui, "PROCESS_STATE_PATH", str(self.root / "managed_processes.json")), \
             patch.object(ui, "TRAIN_LOG", str(train_log)):
            ui._write_managed_processes({"training": record})
            ui.global_state.update(training_process=None, training_active=True,
                                   training_stage="training", total_steps=49972, current_step=180)
            status = self.client.get("/api/train/status").get_json()
        self.assertTrue(status["active"])
        self.assertEqual(status["current_step"], 181)
        self.assertAlmostEqual(status["progress"], 181 / 49972 * 100)

    def test_carriage_return_progress_updates_all_policies_without_displaying_tqdm(self):
        raw_output = (
            "\rTraining:   0%|          | 1/50000 [00:02<34:00:00, 2.4s/step]"
            "\rTraining:   0%|          | 2/50000 [00:04<33:00:00, 2.3s/step]"
            "\nINFO ot_train.py:778 step:2 loss:7.670 grdn:171.775 lr:1.0e-05\n"
        )
        for policy in ("act", "diffusion", "smolvla"):
            with self.subTest(policy=policy):
                train_log = self.root / f"{policy}-training.log"
                ui.global_state.update(total_steps=50000, current_step=0)
                with patch.object(ui, "TRAIN_LOG", str(train_log)), \
                     patch.object(ui, "_managed_process_record", return_value=None):
                    pending = ""
                    for chunk in (raw_output[:23], raw_output[23:71], raw_output[71:]):
                        pending = ui._consume_training_log_chunk(chunk, pending)
                    if pending:
                        ui._consume_training_line(pending)

                self.assertEqual(ui.global_state["current_step"], 2)
                displayed = train_log.read_text()
                self.assertIn("step:2 loss:7.670", displayed)
                self.assertNotIn("Training:", displayed)
                self.assertNotIn("1/50000", displayed)
                self.assertNotIn("2/50000", displayed)

    def test_resumed_progress_offsets_remaining_steps_for_all_policies(self):
        raw_output = (
            "\rTraining:   0%|          | 1/49792 [00:02<34:00:00, 2.4s/step]"
            "\rTraining:   0%|          | 2/49792 [00:04<33:00:00, 2.3s/step]"
            "\nloss_metric step=182 loss=7.670\n"
        )
        for policy in ("act", "diffusion", "smolvla"):
            with self.subTest(policy=policy):
                train_log = self.root / f"{policy}-resume-training.log"
                ui.global_state.update(
                    total_steps=49972, current_step=180,
                    training_resumed_from_step=180,
                )
                with patch.object(ui, "TRAIN_LOG", str(train_log)), \
                     patch.object(ui, "_managed_process_record", return_value=None):
                    pending = ui._consume_training_log_chunk(raw_output)
                    if pending:
                        ui._consume_training_line(pending)

                self.assertEqual(ui.global_state["current_step"], 182)
                displayed = train_log.read_text()
                self.assertIn("loss_metric step=182", displayed)
                self.assertNotIn("Training:", displayed)

    def test_shutdown_keeps_training_alive_by_default(self):
        previous_shutdown_done = ui._shutdown_done
        ui._shutdown_done = False
        self.addCleanup(setattr, ui, "_shutdown_done", previous_shutdown_done)
        with patch.dict(ui.os.environ, {}, clear=True), \
             patch.object(ui, "PROCESS_STATE_PATH", str(self.root / "managed_processes.json")), \
             patch.object(ui, "_stop_managed_process") as stop_process:
            ui.shutdown_all_children("test")
        self.assertNotIn("training", [call.args[0] for call in stop_process.call_args_list])

    def test_training_reset_clears_state_without_deleting_outputs(self):
        output_dir = self.root / "output"
        output_dir.mkdir()
        checkpoint = output_dir / "checkpoint.bin"
        checkpoint.write_text("saved")
        with patch.object(ui, "PROCESS_STATE_PATH", str(self.root / "managed_processes.json")):
            ui.global_state.update(training_active=False, training_process=None,
                                   training_policy="act", training_stage=None,
                                   current_step=500, total_steps=1000,
                                   last_output_dir=str(output_dir), training_cancel_event=None)
            ui.global_state["training_active"] = True
            blocked = self.client.post("/api/train/reset")
            self.assertEqual(blocked.status_code, 409)
            self.assertEqual(ui.global_state["current_step"], 500)
            ui.global_state["training_active"] = False
            response = self.client.post("/api/train/reset")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(ui.global_state["current_step"], 0)
        self.assertEqual(ui.global_state["total_steps"], 0)
        self.assertIsNone(ui.global_state["training_policy"])
        self.assertIsNone(ui.global_state["last_output_dir"])
        self.assertTrue(checkpoint.is_file())
        self.assertEqual(Path(ui.TRAIN_LOG).read_text(), "")

    def test_existing_policies_keep_training_environment(self):
        for policy in ("act", "diffusion"):
            with self.subTest(policy=policy), patch.object(ui, "ensure_dataset_v30"), \
                 patch.object(ui, "validate_dataset_with_lerobot", return_value={}), \
                 patch.object(ui.threading, "Thread"), patch.object(ui, "_monitor_training_logs"), \
                 patch.object(ui, "resolve_repo_python_executable", return_value=ui.sys.executable), \
                 patch.object(ui, "_register_managed_process"), \
                 patch.dict(ui.os.environ, {}, clear=True), patch.object(ui.subprocess, "Popen") as launch:
                ui.global_state.update(training_active=True, training_process=None,
                                       training_cancel_event=None)
                cfg = ui.POLICY_CONFIGS[policy]
                ui._prepare_and_train(policy, str(self.root), "dataset",
                                      {"push_to_hub": True, "hub_repo_id": "user/model"}, "job", "job",
                                      str(self.root / "output"), cfg, cfg["script"])
                env = launch.call_args.kwargs["env"]
                if policy == "act":
                    self.assertEqual(env["SMOLVLA_PYTHON"], ui.sys.executable)
                self.assertEqual(env["HF_HUB_OFFLINE"], "1")
                self.assertNotIn("PRETRAINED_PATH", env)
                self.assertNotIn("REPAIR_MISSING_QUANTILES", env)
                self.assertEqual(env["PUSH_TO_HUB"], "false")
                self.assertNotIn("HUB_REPO_ID", env)
                self.assertEqual(env["EVAL_SPLIT"], "0.1")
                self.assertEqual(env["EVAL_STEPS"], "5000" if policy == "diffusion" else "500")

    def test_inference_routes_from_checkpoint_not_browser(self):
        for policy in ("smolvla", "act", "diffusion"):
            run, model = self.checkpoint(policy)
            meta_dir = self.root / "dataset/meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            (meta_dir / "info.json").write_text(json.dumps({"robot_type": "leader_follower_6dof_4camera"}))
            metadata = {"dataset_root": str(meta_dir.parent), "lang_instruction": "Pick blue box", "action_horizon": 50}
            ui.global_state["server_process"] = None
            with self.subTest(policy=policy), patch.object(ui, "_checkpoint_inference_metadata", return_value=metadata), \
                 patch.object(ui.threading, "Thread"), patch.object(ui.subprocess, "Popen") as launch:
                result = self.client.post("/api/inference/start_server", json={
                    "checkpoint_path": str(run), "policy_type": "generic", "port": 5557,
                })
                self.assertEqual(result.status_code, 200, result.get_json())
                if policy == "diffusion":
                    launch.assert_not_called()
                    self.assertEqual(ui.global_state["inference_metadata"]["pretrained_dir"], str(model))
                    continue
                cmd = launch.call_args.args[0]
                script = "smolvla/serve_smolvla_zmq.py" if policy == "smolvla" else "serve_policy_zmq.py"
                self.assertEqual(cmd[1], f"{ui.DAKSHA_ROOT}/{script}")
                self.assertIn(f"--policy-path={model}", cmd)
                self.assertIn("--task-default=Pick blue box", cmd)
                self.assertIn("--action-horizon=50", cmd)

    def test_standalone_checkpoints_and_resume(self):
        run, model = self.checkpoint()
        with patch.object(ui, "SMOLVLA_OUTPUT_ROOT", str(run.parent)), patch.object(ui, "OUTPUT_ROOT", str(self.root / "ui-runs")):
            items = self.client.get("/api/checkpoints").get_json()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["model"], "smolvla")
        self.assertEqual(items[0]["latest_step"], 500)
        self.assertEqual(items[0]["checkpoint_options"], [{
            "step": 500,
            "label": "000500",
            "path": str(run / "checkpoints/000500"),
        }])
        self.assertFalse(items[0]["can_delete"])
        with patch.object(ui.threading, "Thread") as thread:
            response = self.client.post("/api/train/resume", json={"checkpoint_path": str(run)})
            self.assertEqual(response.status_code, 200, response.get_json())
            self.assertEqual(ui.global_state["training_policy"], "smolvla")
            self.assertEqual(response.get_json()["stage"], "preparing")
            self.assertEqual(thread.call_args.kwargs["args"][0], model)

    def test_default_standalone_output_is_discovered(self):
        run, _ = self.checkpoint()
        with patch.object(ui, "SMOLVLA_DEFAULT_RUN", str(run)), \
             patch.object(ui, "SMOLVLA_OUTPUT_ROOT", str(run.parent)), \
             patch.object(ui, "OUTPUT_ROOT", str(self.root / "ui-runs")):
            items = self.client.get("/api/checkpoints").get_json()
        self.assertEqual([item["path"] for item in items], [str(run)])

    def test_incomplete_smolvla_save_is_not_listed_or_resolved(self):
        run, model = self.checkpoint()
        incomplete = run / "checkpoints/000600/pretrained_model"
        incomplete.mkdir(parents=True)
        (incomplete / "config.json").write_text('{"type":"smolvla"}')
        self.assertEqual(ui._list_run_checkpoint_steps(run), [500])
        self.assertEqual(ui.resolve_pretrained_model_dir(run), model)

    def test_exact_checkpoint_step_resolves_its_own_model(self):
        run, model = self.checkpoint("act")
        newer = run / "checkpoints/000600/pretrained_model"
        newer.mkdir(parents=True)
        (newer / "config.json").write_text('{"type":"act"}')

        self.assertEqual(
            ui.resolve_pretrained_model_dir(run / "checkpoints/000500"),
            model,
        )
        self.assertEqual(ui.resolve_pretrained_model_dir(run), newer)

    def test_resume_uses_shared_backend_for_all_policies(self):
        for policy in ("act", "diffusion", "smolvla"):
            with self.subTest(policy=policy):
                run, model = self.checkpoint(policy)
                ui.global_state.update(
                    training_active=True, training_generation=0,
                    training_stage="preparing", training_policy=policy,
                    current_step=500, total_steps=3000,
                    training_resumed_from_step=500,
                )
                with patch.object(ui.threading, "Thread"), \
                     patch.object(ui, "_monitor_training_logs"), \
                     patch.object(ui, "_register_managed_process"), \
                     patch.object(ui.subprocess, "Popen") as launch:
                    ui._prepare_and_resume_train(model, str(run), 6000)
                cmd = launch.call_args.args[0]
                self.assertEqual(cmd[:3], [ui.sys.executable, "-m", "lerobot.scripts.lerobot_train"])
                self.assertIn("--resume=true", cmd)
                self.assertIn(f"--config_path={model}", cmd)
                self.assertIn(f"--output_dir={run}", cmd)
                self.assertIn("--steps=6000", cmd)
                env = launch.call_args.kwargs["env"]
                self.assertEqual(env["PYTHONUNBUFFERED"], "1")
                self.assertEqual(env["HF_HUB_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
