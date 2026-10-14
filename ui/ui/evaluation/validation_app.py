#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Interactive validation studio for the Daksha B1 GR00T policy — one server that
does the whole loop from the browser, reachable from any machine on the network.

What it does that serve_validation_ui.py does not: nothing has to be prepared in
a terminal first. The page lets you

  * pick a DATASET (every LeRobot dataset found under --dataset-root is listed),
  * pick a CHECKPOINT (every checkpoint dir found under --model-root),
  * pick which EPISODES to validate,
  * press Run — the model is loaded here, rolled out open-loop over the recorded
    observations, and progress streams back to the page,
  * then read, for every joint, the recorded action vs the model prediction as a
    chart, plus the per-joint error table and the per-frame value table.

Results are written to disk (same format validate_prediction_vs_dataset.py
writes: meta.json + episode_<id>.json per run dir), so a finished run reloads
instantly and stays browsable later.

Serve it to the LAN:
    python gr00t/eval/real_robot/daksha_b1/validation_app.py --host 0.0.0.0

Anyone who can reach the port can start a GPU job and read any dataset under the
allowed roots, so on a shared network pass --token <secret> (the page keeps the
token from the URL) and keep --allow-root as tight as you can.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import socket
import threading
import time
import traceback
from typing import Any, Callable
import urllib.parse
import uuid


HERE = Path(__file__).resolve().parent
UI_HTML = HERE / "ui" / "validation_app.html"
REPO_ROOT = HERE.parents[3]  # gr00t/eval/real_robot/daksha_b1 -> repo root

# Directories that never contain a dataset root or a checkpoint but do contain
# thousands of files, so descending into them makes the scan feel broken.
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", "wandb",
    ".cache", ".ipynb_checkpoints", "data", "videos", "images", "meta",
    "site-packages", ".mypy_cache", ".pytest_cache",
}


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def is_dataset(p: Path) -> bool:
    return (p / "meta" / "info.json").is_file()


def is_checkpoint(p: Path) -> bool:
    if not (p / "config.json").is_file():
        return False
    return (
        (p / "model.safetensors.index.json").is_file()
        or any(p.glob("*.safetensors"))
        or any(p.glob("*.bin"))
    )


def find_dirs(
    roots: list[Path],
    match: Callable[[Path], bool],
    depth: int,
    descend_matches: bool = False,
) -> list[Path]:
    """Breadth-limited scan for dirs satisfying `match`.

    `descend_matches` keeps going inside a hit, which is what a training output
    dir needs: it holds a final export (config.json + weights) *and* the
    checkpoint-<n> subdirs, and stopping at the parent would hide every one of
    them.
    """
    found: list[Path] = []
    seen: set[Path] = set()

    def rec(d: Path, level: int) -> None:
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir())
        except (PermissionError, OSError):
            return
        for c in children:
            if c.name.startswith(".") or c.name in SKIP_DIRS:
                continue
            rp = c.resolve()
            if rp in seen:
                continue
            hit = match(c)
            if hit:
                seen.add(rp)
                found.append(c)
                if not descend_matches:
                    continue
            if level < depth:
                rec(c, level + 1)

    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        hit = match(root)
        if hit:
            rp = root.resolve()
            if rp not in seen:
                seen.add(rp)
                found.append(root)
        if not hit or descend_matches:
            rec(root, 1)
    return found


def read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def dataset_summary(p: Path) -> dict:
    """Cheap header for the dataset picker (no episode list, no frames)."""
    info = read_json(p / "meta" / "info.json")
    return {
        "path": str(p),
        "name": p.name,
        "parent": str(p.parent),
        "episodes": info.get("total_episodes", 0),
        "frames": info.get("total_frames", 0),
        "fps": info.get("fps", 30),
        "robot": info.get("robot_type", ""),
        "action_dim": len(info.get("features", {}).get("action", {}).get("names", []) or []),
    }


def dataset_labels(p: Path) -> list[str]:
    """Joint labels straight from the dataset meta, in modality.json key order.

    The authoritative order is the model's action-key order, which is only known
    once a policy is loaded; this is the preview shown before a run.
    """
    info = read_json(p / "meta" / "info.json")
    modality = read_json(p / "meta" / "modality.json")
    flat = info["features"]["action"]["names"]
    labels: list[str] = []
    for key, sl in modality.get("action", {}).items():
        labels.extend(f"{key}:{n}" for n in flat[sl["start"] : sl["end"]])
    return labels


def dataset_episodes(p: Path, limit: int = 5000) -> list[dict]:
    """Episode index + length + task, from episodes.jsonl when it exists."""
    jsonl = p / "meta" / "episodes.jsonl"
    eps: list[dict] = []
    if jsonl.is_file():
        with jsonl.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eps.append(
                    {
                        "index": d.get("episode_index", len(eps)),
                        "length": d.get("length", 0),
                        "task": d.get("task") or d.get("prompt") or "",
                    }
                )
                if len(eps) >= limit:
                    break
    if not eps:
        info = read_json(p / "meta" / "info.json")
        eps = [{"index": i, "length": 0, "task": ""} for i in range(info.get("total_episodes", 0))]
    return eps


def checkpoint_summary(p: Path) -> dict:
    steps = None
    m = re.search(r"checkpoint-(\d+)$", p.name)
    if m:
        steps = int(m.group(1))
    emb = ""
    emb_file = p / "embodiment_id.json"
    if emb_file.is_file():
        try:
            emb = ", ".join(str(k) for k in read_json(emb_file))
        except Exception:
            emb = ""
    name = f"{p.parent.name}/{p.name}" if p.parent.name else p.name
    return {
        "path": str(p),
        # A training output dir with no checkpoint-<n> in its name is the final
        # export of that run, which is easy to mistake for a step checkpoint.
        "name": name if steps is not None else f"{name} (final export)",
        "run": p.parent.name,
        "steps": steps,
        "embodiment": emb,
        "mtime": p.stat().st_mtime,
    }


# --------------------------------------------------------------------------- #
# run bookkeeping
# --------------------------------------------------------------------------- #
def slugify(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "run"


def run_slug(model: str, steps: int, horizon: int) -> str:
    m = Path(model)
    name = f"{m.parent.name}-{m.name}" if m.parent.name else m.name
    return f"{slugify(name)}__h{horizon}_s{steps}"


def runs_root(dataset: Path, override: Path | None) -> Path:
    if override:
        return override / f"validation_{dataset.name}"
    return dataset.parent / f"validation_{dataset.name}"


def list_runs(dataset: Path, override: Path | None) -> list[dict]:
    """Every finished run dir for this dataset, newest first."""
    root = runs_root(dataset, override)
    out: list[dict] = []
    if not root.is_dir():
        return out
    # `root` itself when validate_prediction_vs_dataset.py wrote the dumps flat
    # (the pre-studio layout); newer runs each get their own subdir.
    for d in [root, *root.iterdir()]:
        if not d.is_dir():
            continue
        eps = sorted(int(f.stem.split("_")[1]) for f in d.glob("episode_*.json"))
        if not eps:
            continue
        meta = {}
        mf = d / "meta.json"
        if mf.is_file():
            try:
                meta = read_json(mf)
            except Exception:
                meta = {}
        out.append(
            {
                "slug": "." if d == root else d.name,
                "episodes": eps,
                "model": meta.get("model", ""),
                "steps": meta.get("steps"),
                "action_horizon": meta.get("action_horizon"),
                "embodiment": meta.get("embodiment", ""),
                "summary": meta.get("summary", {}),
                "mtime": d.stat().st_mtime,
            }
        )
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


# --------------------------------------------------------------------------- #
# the job
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: str
    dataset: str
    model: str
    embodiment: str
    episodes: list[int]
    steps: int
    action_horizon: int
    reuse: bool = True
    slug: str = ""
    state: str = "queued"  # queued | running | done | error | cancelled
    stage: str = ""
    done: list[int] = field(default_factory=list)
    current: dict | None = None
    summary: dict = field(default_factory=dict)
    log: list[str] = field(default_factory=list)
    error: str | None = None
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancel: bool = False

    def pct(self) -> float:
        n = max(1, len(self.episodes))
        base = len(self.done) / n
        if self.current and self.current.get("total"):
            base += (self.current["step"] / self.current["total"]) / n
        return min(1.0, base)

    def public(self) -> dict:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "model": self.model,
            "embodiment": self.embodiment,
            "episodes": self.episodes,
            "steps": self.steps,
            "action_horizon": self.action_horizon,
            "slug": self.slug,
            "state": self.state,
            "stage": self.stage,
            "done": self.done,
            "current": self.current,
            "summary": self.summary,
            "log": self.log[-200:],
            "error": self.error,
            "pct": self.pct(),
            "elapsed": (self.finished or time.time()) - self.started,
        }


class Studio:
    """Owns the GPU: one job at a time, one cached policy, one cached loader."""

    def __init__(self, runs_override: Path | None):
        self.runs_override = runs_override
        self.lock = threading.Lock()
        self.job: Job | None = None
        self._policy = None
        self._policy_key: tuple | None = None
        self._loader = None
        self._loader_key: tuple | None = None

    # -- lifecycle ---------------------------------------------------------- #
    def start(self, params: dict) -> Job:
        with self.lock:
            if self.job and self.job.state in ("queued", "running"):
                raise RuntimeError("a validation is already running")
            job = Job(
                id=uuid.uuid4().hex[:8],
                dataset=params["dataset"],
                model=params["model"],
                embodiment=params.get("embodiment", "DAKSHA_B1"),
                episodes=list(params["episodes"]),
                steps=int(params.get("steps", 200)),
                action_horizon=int(params.get("action_horizon", 16)),
                reuse=bool(params.get("reuse", True)),
            )
            job.slug = run_slug(job.model, job.steps, job.action_horizon)
            self.job = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def cancel(self) -> bool:
        job = self.job
        if job and job.state in ("queued", "running"):
            job.cancel = True
            self._log(job, "cancel requested — stopping after the current chunk")
            return True
        return False

    def _log(self, job: Job, msg: str) -> None:
        job.log.append(f"{time.strftime('%H:%M:%S')}  {msg}")
        print(f"[job {job.id}] {msg}", flush=True)

    # -- cached heavy objects ----------------------------------------------- #
    def _release(self) -> None:
        self._policy = None
        self._policy_key = None
        self._loader = None
        self._loader_key = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass

    def _get_policy(self, job: Job, tag):
        key = (job.model, str(tag))
        if self._policy_key == key and self._policy is not None:
            self._log(job, "reusing the already-loaded checkpoint")
            return self._policy
        self._release()
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._log(job, f"loading {job.model} on {device} (first load takes ~1 min)")
        policy = Gr00tPolicy(embodiment_tag=tag, model_path=job.model, device=device)
        self._policy, self._policy_key = policy, key
        return policy

    def _get_loader(self, job: Job, modality):
        key = (job.dataset, job.model)
        if self._loader_key == key and self._loader is not None:
            return self._loader
        from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader

        self._log(job, f"opening dataset {job.dataset}")
        loader = LeRobotEpisodeLoader(
            dataset_path=job.dataset,
            modality_configs=modality,
            video_backend="torchcodec",
            video_backend_kwargs=None,
        )
        self._loader, self._loader_key = loader, key
        return loader

    # -- the rollout -------------------------------------------------------- #
    def _rollout(self, policy, loader, traj_id, tag, steps, horizon, job) -> tuple:
        """Open-loop rollout of one episode; same math as
        validate_prediction_vs_dataset.run_episode, plus progress + cancel."""
        from copy import deepcopy

        from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
        from gr00t.eval.open_loop_eval import parse_action_gr00t, parse_observation_gr00t
        import numpy as np

        traj = loader[traj_id]
        actual = min(steps, len(traj))
        state_keys = loader.modality_configs["state"].modality_keys
        action_keys = loader.modality_configs["action"].modality_keys

        obs_cfg = deepcopy(loader.modality_configs)
        obs_cfg.pop("action")

        preds: list[np.ndarray] = []
        for step in range(0, actual, horizon):
            if job.cancel:
                return None
            job.current = {"episode": traj_id, "step": step, "total": actual}
            dp = extract_step_data(traj, step, obs_cfg, tag)
            obs: dict[str, Any] = {}
            for k, v in dp.states.items():
                obs[f"state.{k}"] = v
            for k, v in dp.images.items():
                obs[f"video.{k}"] = np.array(v)
            for lang_key in loader.modality_configs["language"].modality_keys:
                obs[lang_key] = dp.text
            raw_action, _ = policy.get_action(parse_observation_gr00t(obs, loader.modality_configs))
            chunk = parse_action_gr00t(raw_action)
            for j in range(horizon):
                preds.append(
                    np.concatenate(
                        [
                            np.atleast_1d(np.atleast_1d(chunk[f"action.{k}"])[j])
                            for k in action_keys
                        ],
                        axis=0,
                    )
                )
        job.current = {"episode": traj_id, "step": actual, "total": actual}

        def stack(cols):
            return np.concatenate([np.vstack([a for a in traj[c]]) for c in cols], axis=-1)

        state = stack([f"state.{k}" for k in state_keys])[:actual]
        gt = stack([f"action.{k}" for k in action_keys])[:actual]
        pred = np.array(preds)[:actual]
        return state, gt, pred, len(traj)

    def _run(self, job: Job) -> None:
        try:
            job.state = "running"
            job.stage = "loading model"
            from gr00t.data.embodiment_tags import EmbodimentTag
            from gr00t.eval.real_robot.daksha_b1.validate_prediction_vs_dataset import (
                build_joint_labels,
                per_joint_metrics,
            )
            import numpy as np

            dataset = Path(job.dataset)
            tag = EmbodimentTag.resolve(job.embodiment)
            out_dir = runs_root(dataset, self.runs_override) / job.slug
            out_dir.mkdir(parents=True, exist_ok=True)
            fps = dataset_summary(dataset)["fps"]

            todo = [
                e for e in job.episodes
                if not (job.reuse and (out_dir / f"episode_{e}.json").is_file())
            ]
            for ep in job.episodes:
                f = out_dir / f"episode_{ep}.json"
                if f.is_file() and job.reuse:
                    try:
                        d = read_json(f)
                        job.summary[str(ep)] = d.get("overall", {})
                        job.done.append(ep)
                        self._log(job, f"episode {ep}: reusing cached result")
                    except Exception:
                        todo.append(ep)

            if todo:
                policy = self._get_policy(job, tag)
                modality = policy.get_modality_config()
                loader = self._get_loader(job, modality)
                labels = build_joint_labels(dataset, modality["action"].modality_keys)
                self._log(job, f"{len(loader)} episodes in dataset, {len(labels)} action dims")
            else:
                self._log(job, "every requested episode was already computed")

            for ep in todo:
                if job.cancel:
                    break
                if ep >= len(loader):
                    self._log(job, f"episode {ep} out of range (dataset has {len(loader)}) — skipped")
                    continue
                job.stage = f"episode {ep}"
                self._log(job, f"episode {ep}: rolling out {job.steps} frames")
                t0 = time.time()
                res = self._rollout(policy, loader, ep, tag, job.steps, job.action_horizon, job)
                if res is None:  # cancelled mid-episode
                    break
                state, gt, pred, ep_len = res
                rows = per_joint_metrics(gt, pred, labels)
                overall = {
                    "mae": float(np.mean(np.abs(gt - pred))),
                    "rmse": float(np.sqrt(np.mean((gt - pred) ** 2))),
                    "worst": max(rows, key=lambda r: (r["nmae"] if r["nmae"] == r["nmae"] else -1))["joint"],
                    "frames": int(gt.shape[0]),
                }
                dump = {
                    "traj_id": ep,
                    "labels": labels,
                    "action_horizon": job.action_horizon,
                    "fps": fps,
                    "model": job.model,
                    "dataset": job.dataset,
                    "episode_length": ep_len,
                    "state": np.round(state, 5).tolist() if state.shape == gt.shape else None,
                    "gt": np.round(gt, 5).tolist(),
                    "pred": np.round(pred, 5).tolist(),
                    # NaN is not valid JSON and JSON.parse() rejects it, so the
                    # "joint never moves" case is sent as null.
                    "rows": [
                        {k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
                        for r in rows
                    ],
                    "overall": overall,
                }
                (out_dir / f"episode_{ep}.json").write_text(json.dumps(dump))
                job.summary[str(ep)] = overall
                job.done.append(ep)
                self._log(
                    job,
                    f"episode {ep}: MAE={overall['mae']:.4f} RMSE={overall['rmse']:.4f} "
                    f"worst={overall['worst']} ({time.time() - t0:.0f}s)",
                )

            meta = {
                "model": job.model,
                "dataset": job.dataset,
                "embodiment": str(job.embodiment),
                "action_horizon": job.action_horizon,
                "steps": job.steps,
                "fps": fps,
                "episode_ids": sorted(job.done),
                "summary": job.summary,
                "updated": time.time(),
            }
            # Merge with an earlier run in the same dir so its episodes stay listed.
            mf = out_dir / "meta.json"
            if mf.is_file():
                try:
                    old = read_json(mf)
                    meta["episode_ids"] = sorted(set(old.get("episode_ids", [])) | set(job.done))
                    merged = dict(old.get("summary", {}))
                    merged.update(job.summary)
                    meta["summary"] = merged
                except Exception:
                    pass
            mf.write_text(json.dumps(meta, indent=1))

            job.current = None
            job.stage = ""
            job.state = "cancelled" if job.cancel else "done"
            self._log(job, f"{job.state} — results in {out_dir}")
        except Exception as exc:  # surfaced verbatim in the page
            job.state = "error"
            job.error = f"{type(exc).__name__}: {exc}"
            job.log.append(traceback.format_exc())
            print(traceback.format_exc(), flush=True)
        finally:
            job.finished = time.time()


# --------------------------------------------------------------------------- #
# http
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, args):
        self.args = args
        self.dataset_roots = [Path(r).expanduser() for r in args.dataset_root]
        self.model_roots = [Path(r).expanduser() for r in args.model_root]
        self.allow_roots = [
            Path(r).expanduser().resolve()
            for r in (args.allow_root or [str(Path.home())])
        ]
        self.runs_override = Path(args.runs_dir).expanduser().resolve() if args.runs_dir else None
        self.studio = Studio(self.runs_override)
        self._scan: dict | None = None
        self._scan_at = 0.0

    # -- helpers ------------------------------------------------------------ #
    def safe_path(self, raw: str) -> Path:
        p = Path(raw).expanduser().resolve()
        if not any(p == r or r in p.parents for r in self.allow_roots):
            raise PermissionError(
                f"{p} is outside the allowed roots ({', '.join(map(str, self.allow_roots))})"
            )
        return p

    def scan(self, refresh: bool = False) -> dict:
        if self._scan and not refresh and time.time() - self._scan_at < 120:
            return self._scan
        datasets = []
        for d in find_dirs(self.dataset_roots, is_dataset, self.args.scan_depth):
            try:
                datasets.append(dataset_summary(d))
            except Exception:
                continue
        checkpoints = []
        for c in find_dirs(self.model_roots, is_checkpoint, self.args.scan_depth, descend_matches=True):
            try:
                checkpoints.append(checkpoint_summary(c))
            except Exception:
                continue
        datasets.sort(key=lambda d: d["path"])
        checkpoints.sort(key=lambda c: (c["run"], c["steps"] or 0))
        self._scan = {"datasets": datasets, "checkpoints": checkpoints, "scanned": time.time()}
        self._scan_at = time.time()
        return self._scan

    # -- routes ------------------------------------------------------------- #
    def api(self, method: str, path: str, q: dict, body: dict) -> tuple[int, Any]:
        if path == "/api/state":
            sc = self.scan(refresh=q.get("refresh") == "1")
            return 200, {
                "datasets": sc["datasets"],
                "checkpoints": sc["checkpoints"],
                "scanned": sc["scanned"],
                "roots": {
                    "dataset": [str(r) for r in self.dataset_roots],
                    "model": [str(r) for r in self.model_roots],
                    "allow": [str(r) for r in self.allow_roots],
                },
                "defaults": {
                    "embodiment": self.args.embodiment_tag,
                    "steps": self.args.steps,
                    "action_horizon": self.args.action_horizon,
                    "dataset": self.args.dataset_path or "",
                    "model": self.args.model_path or "",
                },
                "job": self.studio.job.public() if self.studio.job else None,
                "gpu": gpu_name(),
            }

        if path == "/api/dataset":
            ds = self.safe_path(q["path"])
            if not is_dataset(ds):
                return 400, {"error": f"{ds} is not a LeRobot dataset (no meta/info.json)"}
            info = dataset_summary(ds)
            info["episode_list"] = dataset_episodes(ds)
            try:
                info["labels"] = dataset_labels(ds)
            except Exception as e:
                info["labels"] = []
                info["labels_error"] = str(e)
            info["runs"] = list_runs(ds, self.runs_override)
            info["runs_root"] = str(runs_root(ds, self.runs_override))
            return 200, info

        if path == "/api/episode":
            ds = self.safe_path(q["dataset"])
            root = runs_root(ds, self.runs_override).resolve()
            f = (root / q["slug"] / f"episode_{int(q['id'])}.json").resolve()
            # the slug arrives from the browser: keep it inside the runs dir
            if root not in f.parents:
                return 403, {"error": f"{f} is outside {root}"}
            if not f.is_file():
                return 404, {"error": f"{f} not found"}
            return 200, read_json(f)

        if path == "/api/job":
            return 200, {"job": self.studio.job.public() if self.studio.job else None}

        if method == "POST" and path == "/api/run":
            ds = self.safe_path(body["dataset"])
            model = self.safe_path(body["model"])
            if not is_dataset(ds):
                return 400, {"error": f"{ds} is not a LeRobot dataset"}
            if not is_checkpoint(model):
                return 400, {"error": f"{model} is not a checkpoint dir (no config.json + weights)"}
            eps = [int(e) for e in body.get("episodes", [])]
            if not eps:
                return 400, {"error": "pick at least one episode"}
            try:
                job = self.studio.start(
                    {
                        "dataset": str(ds),
                        "model": str(model),
                        "embodiment": body.get("embodiment") or self.args.embodiment_tag,
                        "episodes": eps,
                        "steps": int(body.get("steps") or self.args.steps),
                        "action_horizon": int(body.get("action_horizon") or self.args.action_horizon),
                        "reuse": bool(body.get("reuse", True)),
                    }
                )
            except RuntimeError as e:
                return 409, {"error": str(e)}
            return 200, {"job": job.public()}

        if method == "POST" and path == "/api/cancel":
            return 200, {"cancelled": self.studio.cancel()}

        return 404, {"error": f"no route {path}"}


def gpu_name() -> str:
    """Read the GPU name without importing torch (which would cost ~10s here)."""
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=4,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else ""
    except Exception:
        return ""


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "DakshaValidationStudio/1.0"

        def log_message(self, fmt, *a):  # quieter: one line per API call only
            if "/api/job" not in self.path:
                print(f"{self.address_string()} {fmt % a}", flush=True)

        # -- io ------------------------------------------------------------- #
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: Any) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json; charset=utf-8")

        def _authed(self, q: dict) -> bool:
            if not app.args.token:
                return True
            return (
                q.get("token") == app.args.token
                or self.headers.get("X-Auth-Token") == app.args.token
            )

        # -- routing -------------------------------------------------------- #
        def handle_one(self, method: str) -> None:
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            if not self._authed(q):
                self._json(401, {"error": "bad or missing token — open the URL with ?token=..."})
                return
            if u.path.startswith("/api/"):
                body = {}
                if method == "POST":
                    n = int(self.headers.get("Content-Length") or 0)
                    raw = self.rfile.read(n) if n else b"{}"
                    try:
                        body = json.loads(raw or b"{}")
                    except json.JSONDecodeError:
                        self._json(400, {"error": "body is not JSON"})
                        return
                try:
                    code, payload = app.api(method, u.path, q, body)
                except KeyError as e:
                    code, payload = 400, {"error": f"missing parameter {e}"}
                except PermissionError as e:
                    code, payload = 403, {"error": str(e)}
                except FileNotFoundError as e:
                    code, payload = 404, {"error": str(e)}
                except Exception as e:
                    traceback.print_exc()
                    code, payload = 500, {"error": f"{type(e).__name__}: {e}"}
                self._json(code, payload)
                return
            if u.path in ("/", "/index.html"):
                # Read from disk per request so a UI edit only needs a refresh.
                if not UI_HTML.is_file():
                    self._send(500, f"missing UI file {UI_HTML}".encode(), "text/plain")
                    return
                self._send(200, UI_HTML.read_bytes(), "text/html; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain")

        def do_GET(self):
            self.handle_one("GET")

        def do_POST(self):
            self.handle_one("POST")

    return Handler


def lan_ips() -> list[str]:
    ips: set[str] = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass
    return sorted(ips)


def parse_args():
    p = argparse.ArgumentParser(description="Interactive Daksha B1 validation studio")
    p.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"),
                   help="0.0.0.0 (default) serves every interface; 127.0.0.1 keeps it local.")
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8123)))
    p.add_argument("--token", default=os.environ.get("TOKEN", ""),
                   help="Require ?token=... on every request. Strongly advised with --host 0.0.0.0.")
    p.add_argument("--dataset-root", action="append", default=None,
                   help="Where to look for datasets (repeatable).")
    p.add_argument("--model-root", action="append", default=None,
                   help="Where to look for checkpoints (repeatable).")
    p.add_argument("--allow-root", action="append", default=None,
                   help="Paths the browser is allowed to reach (repeatable, default: $HOME).")
    p.add_argument("--runs-dir", default=None,
                   help="Where results go (default: alongside each dataset, <dataset>/../validation_<name>).")
    p.add_argument("--scan-depth", type=int, default=4)
    p.add_argument("--embodiment-tag", default="DAKSHA_B1")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--action-horizon", type=int, default=16)
    p.add_argument("--dataset-path", default=None, help="Preselect this dataset in the UI.")
    p.add_argument("--model-path", default=None, help="Preselect this checkpoint in the UI.")
    a = p.parse_args()
    home = Path.home()
    if not a.dataset_root:
        a.dataset_root = [str(home / "Desktop")]
    if not a.model_root:
        a.model_root = [str(REPO_ROOT / "output"), str(home / "Desktop")]
    return a


def main():
    args = parse_args()
    app = App(args)
    try:
        httpd = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    except OSError as e:
        raise SystemExit(f"cannot bind {args.host}:{args.port} ({e}). Another server on that port?")
    httpd.daemon_threads = True

    suffix = f"?token={args.token}" if args.token else ""
    print(f"\nDaksha B1 validation studio  —  {args.host}:{args.port}")
    print(f"  dataset roots : {', '.join(args.dataset_root)}")
    print(f"  model roots   : {', '.join(args.model_root)}")
    print(f"  results       : {args.runs_dir or 'alongside each dataset'}")
    print(f"\n  local   http://localhost:{args.port}/{suffix}")
    if args.host in ("0.0.0.0", "::"):
        for ip in lan_ips():
            print(f"  network http://{ip}:{args.port}/{suffix}")
        if not args.token:
            print("\n  ! open to everyone who can reach this port — pass --token <secret> on a shared network")
    print()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
