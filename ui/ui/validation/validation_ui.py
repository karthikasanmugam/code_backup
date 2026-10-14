#!/usr/bin/env python3
"""Web UI for validate_dataset.py.

Point a browser at it, type a dataset path, press Run: you get the summary,
a colour-coded tile per episode, and for any episode you click, the issue list
plus joint traces with the frozen/gap regions shaded, plus the videos.

    python validation_ui.py                       # http://0.0.0.0:8124
    PORT=9000 python validation_ui.py
    python validation_ui.py --dataset /path/to/ds --token secret

Stdlib only, matching custom_biarm_webui/app.py: no Flask, no gradio.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np

UI_ROOT = Path(__file__).resolve().parent
if str(UI_ROOT) not in sys.path:
    sys.path.insert(0, str(UI_ROOT))

import validate_dataset as vd  # noqa: E402

try:
    import pyarrow.parquet as pq
    import pyarrow.types as pat
except ImportError:  # pragma: no cover
    sys.exit("pyarrow is required:  pip install pyarrow")

MAX_PLOT_POINTS = 1500

# ── episode deletion (delete.py) ──────────────────────────────────────────
# delete.py needs the `daksha_data_collection` ROS2 package (to rebuild
# meta/info.json + meta/stats.json via V3DatasetRecorder.resume_existing()
# after removing an episode), which is only importable once both the ROS2
# distro setup *and* this workspace's install overlay are sourced -- neither
# is on this script's own path. Mirrors app.py's build_ros_bridge_cmd() for
# eval_daksha.py: source the setup scripts, then run the system python.
DELETE_SCRIPT = UI_ROOT.parent / "delete.py"
ROS_PYTHON_BIN = os.environ.get("DAKSHA_ROS_PYTHON", "/usr/bin/python3")
ROS_SETUP = os.environ.get("DAKSHA_ROS_SETUP", "/opt/ros/humble/setup.bash")
ROS_WORKSPACE_SETUP = os.environ.get(
    "DAKSHA_ROS_WORKSPACE_SETUP", "/home/s1/.ihub/.final_gen2/install/setup.bash"
)


def build_delete_cmd(dataset_path: str, episode: int) -> list[str]:
    inner = [ROS_PYTHON_BIN, str(DELETE_SCRIPT), "--dataset", dataset_path, "--episode", str(episode)]
    quoted = " ".join(shlex.quote(a) for a in inner)
    script = (
        f'if [ -f "{ROS_SETUP}" ]; then source "{ROS_SETUP}"; fi; '
        f'if [ -f "{ROS_WORKSPACE_SETUP}" ]; then source "{ROS_WORKSPACE_SETUP}"; fi; '
        f"exec {quoted}"
    )
    return ["bash", "-c", script]


# --------------------------------------------------------------------------- #
# http helpers
# --------------------------------------------------------------------------- #


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"{value.__class__.__name__} is not JSON serializable")


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any],
                   status: int = 200) -> None:
    data = json.dumps(payload, default=_json_default).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length else b""
    return json.loads(raw.decode("utf-8")) if raw else {}


# --------------------------------------------------------------------------- #
# run state
# --------------------------------------------------------------------------- #


class RunState:
    """Progress and result of the validation running in the background."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.running = False
        self.stage = "idle"
        self.done = 0
        self.total = 0
        self.started = 0.0
        self.elapsed = 0.0
        self.error: str | None = None
        self.result: dict[str, Any] | None = None
        self.dataset: str | None = None

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "running": self.running,
                "stage": self.stage,
                "done": self.done,
                "total": self.total,
                "elapsed": round(time.time() - self.started, 1) if self.running else self.elapsed,
                "error": self.error,
                "dataset": self.dataset,
                "has_result": self.result is not None,
            }


# --------------------------------------------------------------------------- #
# validation driver
# --------------------------------------------------------------------------- #


def build_args(dataset: Path, opts: dict[str, Any]) -> argparse.Namespace:
    """Validator options, defaults from the CLI parser, overridden by the UI."""
    args = vd.build_parser().parse_args([str(dataset)])
    for key in ("no_video", "deep_video", "strict", "errors_only"):
        if key in opts:
            setattr(args, key, bool(opts[key]))
    for key in ("gap_factor", "jitter_factor", "duration_tol", "short_ratio",
                "dead_eps", "jump_factor", "jump_floor", "range_tol", "dead_ratio"):
        if opts.get(key) not in (None, ""):
            setattr(args, key, float(opts[key]))
    for key in ("freeze_frames", "video_tol", "workers"):
        if opts.get(key) not in (None, ""):
            setattr(args, key, int(opts[key]))
    args.episodes = (opts.get("episodes") or "").strip() or None
    args.json = None
    return args


def select_episodes(ds: vd.Dataset, spec: str | None) -> list[int]:
    eps = vd.discover_episode_indices(ds)
    if not spec:
        return eps
    wanted: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            wanted.update(range(int(a), int(b) + 1))
        elif part:
            wanted.add(int(part))
    return [e for e in eps if e in wanted]


def run_validation(state: RunState, server: "ValidationServer",
                   root: Path, opts: dict[str, Any]) -> None:
    try:
        with state.lock:
            state.running, state.stage = True, "loading metadata"
            state.done, state.total = 0, 0
            state.error, state.result = None, None
            state.started = time.time()
            state.dataset = str(root)

        args = build_args(root, opts)
        ds, issues = vd.load_dataset(root)
        server.cache_dataset(root, ds)
        eps = select_episodes(ds, args.episodes)
        if not eps:
            raise ValueError("no episodes found in that path")

        with state.lock:
            state.total = len(eps)
            state.stage = "checking episodes"

        summaries: list[dict] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            for ep_issues, summ in pool.map(lambda e: vd.check_episode(ds, e, args), eps):
                issues.extend(ep_issues)
                summaries.append(summ)
                with state.lock:
                    state.done += 1

        with state.lock:
            state.stage = "dataset-level checks"
        issues.extend(vd.check_global(ds, eps, summaries, args))
        issues.sort(key=lambda i: (i.episode if i.episode is not None else -1,
                                   0 if i.severity == vd.ERROR else 1, i.code))

        with state.lock:
            state.result = build_result(ds, eps, issues, summaries, args)
            state.stage = "done"
    except Exception as exc:  # surfaced in the UI rather than only the console
        traceback.print_exc()
        with state.lock:
            state.error = f"{type(exc).__name__}: {exc}"
            state.stage = "failed"
    finally:
        with state.lock:
            state.running = False
            state.elapsed = round(time.time() - state.started, 1)


def build_result(ds: vd.Dataset, eps: list[int], issues: list[vd.Issue],
                 summaries: list[dict], args: argparse.Namespace) -> dict[str, Any]:
    by_ep: dict[int | None, list[vd.Issue]] = {}
    for it in issues:
        by_ep.setdefault(it.episode, []).append(it)
    summ_by_ep = {s["episode"]: s for s in summaries}

    episodes = []
    for ep in eps:
        eis = by_ep.get(ep, [])
        n_err = sum(1 for i in eis if i.severity == vd.ERROR)
        n_warn = len(eis) - n_err
        s = summ_by_ep.get(ep, {})
        episodes.append({
            "episode": ep,
            "status": "error" if n_err else ("warn" if n_warn else "clean"),
            "errors": n_err,
            "warnings": n_warn,
            "rows": s.get("rows"),
            "meta_length": s.get("meta_length"),
            "duration_s": s.get("duration_s"),
            "dt_mean_ms": s.get("dt_mean_ms"),
            "dt_max_ms": s.get("dt_max_ms"),
            "video_frames": s.get("video_frames"),
            "issues": [i.as_dict() for i in eis],
        })

    codes: dict[str, dict[str, Any]] = {}
    for it in issues:
        entry = codes.setdefault(it.code, {"code": it.code, "severity": it.severity, "count": 0})
        entry["count"] += 1
        if it.severity == vd.ERROR:
            entry["severity"] = vd.ERROR

    n_err = sum(1 for i in issues if i.severity == vd.ERROR)
    return {
        "dataset": str(ds.root),
        "info": {
            "robot_type": ds.info.get("robot_type"),
            "codebase_version": ds.info.get("codebase_version"),
            "fps": ds.fps,
            "total_episodes": ds.info.get("total_episodes"),
            "total_frames": ds.info.get("total_frames"),
            "tasks": ds.tasks,
            "video_keys": ds.video_keys,
            "state_keys": ds.state_keys,
            "meta_sources": ds.meta_sources,
        },
        "checked_episodes": len(eps),
        "frames": sum(s.get("rows") or 0 for s in summaries),
        "errors": n_err,
        "warnings": len(issues) - n_err,
        "bad_episodes": [e["episode"] for e in episodes if e["status"] == "error"],
        "warn_episodes": [e["episode"] for e in episodes if e["status"] == "warn"],
        "clean_episodes": [e["episode"] for e in episodes if e["status"] == "clean"],
        "dataset_issues": [i.as_dict() for i in by_ep.get(None, [])],
        "codes": sorted(codes.values(), key=lambda c: (-c["count"], c["code"])),
        "episodes": episodes,
        "options": {
            "no_video": args.no_video, "deep_video": args.deep_video,
            "freeze_frames": args.freeze_frames, "gap_factor": args.gap_factor,
            "episodes": args.episodes,
        },
    }


# --------------------------------------------------------------------------- #
# episode detail (traces for the charts)
# --------------------------------------------------------------------------- #


def episode_detail(ds: vd.Dataset, ep: int, freeze_frames: int) -> dict[str, Any]:
    path = vd.data_path(ds, ep)
    if path is None:
        raise FileNotFoundError(f"episode {ep} parquet not found")
    table = pq.read_table(path)
    data = table.to_pydict()
    n = table.num_rows

    vector_cols = []
    for field_ in table.schema:
        if field_.name in ds.video_keys or field_.name in vd.BOOKKEEPING:
            continue
        t = field_.type
        if pat.is_list(t) or pat.is_large_list(t) or pat.is_fixed_size_list(t):
            if pat.is_floating(t.value_type) or pat.is_integer(t.value_type):
                vector_cols.append(field_.name)

    stride = max(1, n // MAX_PLOT_POINTS)
    xs = list(range(0, n, stride))

    traces = []
    for key in vector_cols:
        try:
            arr = np.asarray([np.asarray(v, dtype=np.float64) for v in data[key]])
        except (ValueError, TypeError):
            continue
        if arr.ndim != 2:
            continue
        # Frozen spans are found on the full-resolution data, then the series is
        # decimated for the browser -- decimating first would hide short stalls.
        spans = []
        if n > 1:
            same = np.all(np.isclose(arr[1:], arr[:-1], atol=1e-9), axis=1)
            run_start = None
            for i, s in enumerate(same):
                if s and run_start is None:
                    run_start = i
                elif not s and run_start is not None:
                    if i - run_start + 1 >= freeze_frames:
                        spans.append([run_start, i])
                    run_start = None
            if run_start is not None and len(same) - run_start + 1 >= freeze_frames:
                spans.append([run_start, len(same)])

        names = (ds.features.get(key, {}) or {}).get("names") or []
        dims = []
        for d in range(arr.shape[1]):
            col = arr[:, d]
            dims.append({
                "name": names[d] if d < len(names) else f"dim {d}",
                "min": float(np.nanmin(col)), "max": float(np.nanmax(col)),
                "constant": bool(np.nanmax(col) - np.nanmin(col) < 1e-9),
                "values": [round(float(col[i]), 6) for i in xs],
            })
        traces.append({"key": key, "frames": n, "dims": dims, "frozen_spans": spans})

    dt_ms = []
    gaps = []
    if "timestamp" in data:
        ts = np.asarray(data["timestamp"], dtype=np.float64)
        if ts.size > 1 and np.all(np.isfinite(ts)):
            d = np.diff(ts) * 1000.0
            nominal = 1000.0 / ds.fps
            gaps = [int(i) for i in np.nonzero(d > nominal * 2.5)[0]]
            ds_stride = max(1, d.size // MAX_PLOT_POINTS)
            dt_ms = [round(float(v), 2) for v in d[::ds_stride]]

    videos = []
    for key in ds.video_keys:
        vp = vd.video_path(ds, ep, key)
        if vp is not None:
            videos.append({"key": key, "name": vp.name})

    return {
        "episode": ep,
        "frames": n,
        "x": xs,
        "traces": traces,
        "dt_ms": dt_ms,
        "dt_gaps": gaps,
        "nominal_dt_ms": round(1000.0 / ds.fps, 2),
        "videos": videos,
    }


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #


class ValidationServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr, handler, config: dict[str, Any]):
        super().__init__(addr, handler)
        self.config = config
        self._states: dict[str, RunState] = {}
        self._states_lock = threading.Lock()
        self._ds_cache: dict[str, vd.Dataset] = {}
        self._cache_lock = threading.Lock()

    def state_for(self, key: str) -> RunState:
        """Per-dataset run state, so validating one dataset doesn't block or clobber another."""
        with self._states_lock:
            state = self._states.get(key)
            if state is None:
                state = self._states[key] = RunState()
            return state

    def cache_dataset(self, root: Path, ds: vd.Dataset) -> None:
        with self._cache_lock:
            self._ds_cache[str(root)] = ds

    def invalidate_dataset(self, root: Path) -> None:
        """Drop the cached Dataset (and any stale run result) after an episode delete
        renumbers everything on disk -- cached parquet/video paths would otherwise
        point at the wrong episode."""
        with self._cache_lock:
            self._ds_cache.pop(str(root), None)
        with self._states_lock:
            state = self._states.get(str(root))
        if state is not None:
            with state.lock:
                state.result = None

    def get_dataset(self, root: Path) -> vd.Dataset:
        with self._cache_lock:
            hit = self._ds_cache.get(str(root))
        if hit is not None:
            return hit
        ds, _ = vd.load_dataset(root)
        self.cache_dataset(root, ds)
        return ds

    def resolve(self, raw: str) -> Path:
        """Expand a user-supplied path and refuse anything outside allow_root."""
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = Path(self.config["allow_root"]) / p
        p = p.resolve()
        allow = Path(self.config["allow_root"]).resolve()
        if allow != Path("/") and not (p == allow or allow in p.parents):
            raise PermissionError(f"path is outside --allow-root ({allow}): {p}")
        return p


class Handler(BaseHTTPRequestHandler):
    server: ValidationServer
    server_version = "DatasetValidationUI/1.0"

    def log_message(self, fmt: str, *a: Any) -> None:  # quieter console
        if self.server.config.get("verbose"):
            super().log_message(fmt, *a)

    # -- auth ------------------------------------------------------------- #
    def _authorized(self, query: dict[str, list[str]]) -> bool:
        token = self.server.config.get("token")
        if not token:
            return True
        given = (query.get("token", [""])[0]
                 or self.headers.get("X-Auth-Token", "")
                 or (self.headers.get("Cookie", "").split("token=")[-1].split(";")[0]
                     if "token=" in self.headers.get("Cookie", "") else ""))
        return given == token

    def _state_for_query(self, query: dict[str, list[str]]) -> RunState:
        """The RunState for the dataset named in a request's ?dataset= param.

        Each dataset gets its own RunState (see ValidationServer.state_for) so that
        polling progress/results for one dataset never reads another's -- required for
        validating more than one dataset around the same time (e.g. from separate
        browser tabs deep-linked from the main Daksha UI).
        """
        raw = query.get("dataset", [""])[0]
        root = self.server.resolve(raw) if raw else self.server.config["allow_root"]
        return self.server.state_for(str(root))

    # -- GET -------------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not self._authorized(query):
            _json_response(self, {"ok": False, "error": "bad or missing token"},
                           HTTPStatus.UNAUTHORIZED)
            return

        if parsed.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if self.server.config.get("token"):
                tok = query.get("token", [""])[0]
                if tok:
                    self.send_header("Set-Cookie", f"token={tok}; Path=/; SameSite=Strict")
            self.end_headers()
            self.wfile.write(body)
            return

        if parsed.path == "/api/config":
            _json_response(self, {"ok": True, "config": {
                "default_dataset": self.server.config.get("default_dataset", ""),
                "allow_root": str(self.server.config["allow_root"]),
                "defaults": {
                    k: getattr(vd.build_parser().parse_args(["/x"]), k)
                    for k in ("gap_factor", "freeze_frames", "jump_factor",
                              "short_ratio", "workers")
                },
            }})
            return

        if parsed.path == "/api/progress":
            try:
                state = self._state_for_query(query)
            except (PermissionError, OSError) as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            snap = state.snapshot()
            payload: dict[str, Any] = {"ok": True, "progress": snap}
            if query.get("result", ["0"])[0] == "1":
                with state.lock:
                    payload["result"] = state.result
            _json_response(self, payload)
            return

        if parsed.path == "/api/result":
            try:
                state = self._state_for_query(query)
            except (PermissionError, OSError) as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            with state.lock:
                res = state.result
            if res is None:
                _json_response(self, {"ok": False, "error": "no result yet"},
                               HTTPStatus.NOT_FOUND)
                return
            _json_response(self, {"ok": True, "result": res})
            return

        if parsed.path == "/api/browse":
            self._browse(query)
            return

        if parsed.path == "/api/episode":
            self._episode(query)
            return

        if parsed.path == "/api/video":
            self._video(query)
            return

        _json_response(self, {"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    # -- POST ------------------------------------------------------------- #
    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not self._authorized(query):
            _json_response(self, {"ok": False, "error": "bad or missing token"},
                           HTTPStatus.UNAUTHORIZED)
            return

        if parsed.path == "/api/validate":
            payload = _read_json(self)
            try:
                root = self.server.resolve(payload.get("dataset", ""))
            except (PermissionError, OSError) as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            # Keyed by dataset so validating one doesn't block or get clobbered by another
            # running concurrently (e.g. two datasets opened from the main Daksha UI).
            state = self.server.state_for(str(root))
            if state.snapshot()["running"]:
                _json_response(self, {"ok": False, "error": "this dataset is already being validated"},
                               HTTPStatus.CONFLICT)
                return
            if not (root / "meta").is_dir():
                nested = sorted({c.parent.parent for c in root.rglob("meta/info.json")})[:10]
                _json_response(self, {
                    "ok": False,
                    "error": f"no meta/ directory in {root}",
                    "suggestions": [str(x) for x in nested],
                }, HTTPStatus.BAD_REQUEST)
                return
            t = threading.Thread(target=run_validation,
                                 args=(state, self.server, root, payload),
                                 daemon=True)
            t.start()
            _json_response(self, {"ok": True, "dataset": str(root)})
            return

        if parsed.path == "/api/delete_episode":
            payload = _read_json(self)
            try:
                # Always re-resolved from the dataset path the browser tab is
                # currently tracking (RESULT.dataset / CURRENT_DATASET), never a
                # free-typed one -- so a delete can only ever hit the dataset
                # that's actually loaded, and stays inside --allow-root.
                root = self.server.resolve(payload.get("dataset", ""))
            except (PermissionError, OSError) as exc:
                _json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if not (root / "meta").is_dir():
                _json_response(self, {"ok": False, "error": f"no meta/ directory in {root}"},
                               HTTPStatus.BAD_REQUEST)
                return
            try:
                episode = int(payload.get("episode"))
            except (TypeError, ValueError):
                _json_response(self, {"ok": False, "error": "episode must be an integer"},
                               HTTPStatus.BAD_REQUEST)
                return
            try:
                proc = subprocess.run(
                    build_delete_cmd(str(root), episode),
                    capture_output=True, text=True, timeout=180,
                )
            except subprocess.TimeoutExpired:
                _json_response(self, {"ok": False, "error": "delete.py timed out after 180s"},
                               HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            # Cached Dataset/result are stale the moment any file gets renamed --
            # every episode above the deleted one shifts down by one index.
            self.server.invalidate_dataset(root)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout).strip()
                _json_response(self, {"ok": False, "error": err[-4000:]},
                               HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            _json_response(self, {"ok": True, "episode": episode, "output": proc.stdout.strip()[-2000:]})
            return

        _json_response(self, {"ok": False, "error": "not found"}, HTTPStatus.NOT_FOUND)

    # -- handlers --------------------------------------------------------- #
    def _browse(self, query: dict[str, list[str]]) -> None:
        raw = query.get("path", [str(self.server.config["allow_root"])])[0]
        try:
            root = self.server.resolve(raw)
        except (PermissionError, OSError) as exc:
            _json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if not root.is_dir():
            _json_response(self, {"ok": False, "error": f"not a directory: {root}"},
                           HTTPStatus.NOT_FOUND)
            return
        entries = []
        try:
            for child in sorted(root.iterdir()):
                if child.name.startswith(".") or not child.is_dir():
                    continue
                entries.append({"name": child.name, "path": str(child),
                                "is_dataset": (child / "meta" / "info.json").is_file()})
        except PermissionError:
            pass
        _json_response(self, {"ok": True, "path": str(root),
                              "parent": str(root.parent), "entries": entries})

    def _episode(self, query: dict[str, list[str]]) -> None:
        try:
            root = self.server.resolve(query.get("dataset", [""])[0])
            ep = int(query.get("episode", ["0"])[0])
            freeze = int(query.get("freeze_frames", ["30"])[0])
            ds = self.server.get_dataset(root)
            _json_response(self, {"ok": True, "detail": episode_detail(ds, ep, freeze)})
        except Exception as exc:
            _json_response(self, {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                           HTTPStatus.BAD_REQUEST)

    def _video(self, query: dict[str, list[str]]) -> None:
        """Serve an episode mp4, honouring Range so the browser can scrub."""
        try:
            root = self.server.resolve(query.get("dataset", [""])[0])
            ep = int(query.get("episode", ["0"])[0])
            key = query.get("key", [""])[0]
            ds = self.server.get_dataset(root)
            path = vd.video_path(ds, ep, key)
        except Exception as exc:
            _json_response(self, {"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path is None or not path.is_file():
            _json_response(self, {"ok": False, "error": "video not found"}, HTTPStatus.NOT_FOUND)
            return

        size = path.stat().st_size
        ctype = mimetypes.guess_type(path.name)[0] or "video/mp4"
        rng = self.headers.get("Range", "")
        start, end = 0, size - 1
        partial = False
        if rng.startswith("bytes="):
            spec = rng.split("=", 1)[1].split(",")[0]
            a, _, b = spec.partition("-")
            try:
                if a:
                    start, partial = int(a), True
                if b:
                    end, partial = int(b), True
            except ValueError:
                partial = False
            start = max(0, min(start, size - 1))
            end = max(start, min(end, size - 1))

        length = end - start + 1
        self.send_response(HTTPStatus.PARTIAL_CONTENT if partial else HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as fh:
            fh.seek(start)
            remaining = length
            while remaining > 0:
                chunk = fh.read(min(256 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return  # browser seeked away; not an error
                remaining -= len(chunk)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def _lan_ips() -> list[str]:
    ips = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    return ips


def main() -> int:
    p = argparse.ArgumentParser(description="Web UI for the dataset validator.")
    p.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8124")))
    p.add_argument("--token", default=os.environ.get("TOKEN", ""),
                   help="require ?token=... (set this on a shared network)")
    p.add_argument("--dataset", default=os.environ.get("DATASET_PATH", ""),
                   help="path pre-filled in the form")
    p.add_argument("--allow-root", default=os.environ.get("ALLOW_ROOT", str(Path.home())),
                   help="refuse to read datasets outside this directory")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    config = {
        "token": args.token,
        "default_dataset": args.dataset,
        "allow_root": Path(args.allow_root).expanduser().resolve(),
        "verbose": args.verbose,
    }
    server = ValidationServer((args.host, args.port), Handler, config)
    suffix = f"?token={args.token}" if args.token else ""
    print("dataset validation UI")
    print(f"  allow-root  {config['allow_root']}")
    print(f"  local       http://127.0.0.1:{args.port}/{suffix}")
    for ip in _lan_ips():
        print(f"  lan         http://{ip}:{args.port}/{suffix}")
    if not args.token:
        print("  note        no token set: anyone who can reach this port can read "
              "any dataset under allow-root")
    print("  ctrl-c to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Dataset Validation</title>
<style>
  :root{
    --bg:#F8FAFC; --panel:#FFFFFF; --panel2:#F1F5F9; --line:#CBD5E1;
    --fg:#0F172A; --dim:#64748B; --acc:#2563EB; --acc-hover:#1D4ED8;
    --navy:#1E3A8A; --err:#DC2626; --err-hover:#B91C1C;
    --warn:#B45309; --ok:#15803D;
  }
  /* Dark ("night") palette, mirroring the main Daksha UI's robotic-night theme
     (examples/daksha/ui/templates/index.html) so the two tools look consistent when
     switched together. Applied via body.dark -- see the theme toggle in <header>. */
  body.dark{
    --bg:#0B1220; --panel:#111C33; --panel2:#0F1B33; --line:#294777;
    --fg:#E0ECFF; --dim:#9DB2D0; --acc:#2563EB; --acc-hover:#1D4ED8;
    --navy:#071A45; --err:#F87171; --err-hover:#EF4444;
    --warn:#FBBF24; --ok:#4ADE80;
  }
  body.dark .tile.error{background:#3B151D;border-color:#7F1D1D;color:var(--err)}
  body.dark .tile.warn{background:#3F2D0B;border-color:#92600A;color:var(--warn)}
  body.dark .tile.clean{background:#123226;border-color:#3F8F68;color:var(--ok)}
  body.dark .tag.ERROR{background:#3B151D;color:var(--err)}
  body.dark .tag.WARN{background:#3F2D0B;color:var(--warn)}
  body.dark .tag.CLEAN{background:#123226;color:var(--ok)}
  body.dark .pill.on{background:#172F66;color:#BFDBFE}
  body.dark .err-box{background:#3B151D;border-color:#7F1D1D;color:var(--err)}
  .hdrtop{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:10px}
  .hdrtop h1{margin:0}
  .themebtn{background:var(--panel);color:var(--fg);border:1px solid var(--line);
    border-radius:8px;padding:6px 10px;font-size:14px;line-height:1}
  .themebtn.on{background:var(--acc);border-color:var(--acc);color:#FFFFFF}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
    font:14px/1.5 Inter,Roboto,system-ui,-apple-system,"Segoe UI",sans-serif;font-weight:400}
  a{color:var(--acc)}
  header{position:sticky;top:0;z-index:20;background:var(--navy);color:#FFFFFF;
    border-bottom:1px solid var(--line);padding:12px 18px}
  h1{margin:0 0 10px;font-size:15px;letter-spacing:.5px;font-weight:700}
  h1 span{color:#DBEAFE;font-weight:400}
  .row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
  input[type=text],input[type=number]{background:var(--panel);color:var(--fg);
    border:1px solid var(--line);border-radius:6px;padding:8px 12px;font:inherit}
  input[type=text]{flex:1;min-width:260px}
  button{background:var(--panel);color:var(--fg);border:1px solid var(--line);
    border-radius:6px;padding:8px 16px;font:inherit;cursor:pointer;
    transition:background-color .2s ease,border-color .2s ease,color .2s ease}
  button:hover:not(:disabled){background:#EFF6FF;border-color:var(--acc);color:var(--acc-hover)}
  button:disabled{opacity:.45;cursor:default}
  button.primary{background:var(--acc);border-color:var(--acc);color:#FFFFFF;font-weight:600}
  button.primary:hover:not(:disabled){background:var(--acc-hover);border-color:var(--acc-hover);color:#FFFFFF}
  button.danger{border-color:var(--err);color:var(--err)}
  button.danger:hover:not(:disabled){border-color:var(--err-hover);background:var(--err-hover);color:#FFFFFF}
  label.chk{display:inline-flex;gap:6px;align-items:center;color:#DBEAFE;cursor:pointer}
  main{padding:18px;max-width:1500px;margin:0 auto}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;
    padding:14px 16px;margin-bottom:16px}
  .panel h2{margin:0 0 10px;font-size:12px;text-transform:uppercase;
    letter-spacing:1px;color:var(--dim);font-weight:600}
  .cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px}
  .card{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:12px}
  .card .v{font-size:22px;font-weight:700}
  .card .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.6px}
  .card.err .v{color:var(--err)} .card.warn .v{color:var(--warn)} .card.ok .v{color:var(--ok)}
  .meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
    gap:4px 18px;color:var(--dim);font-size:12px}
  .meta b{color:var(--fg);font-weight:500;overflow-wrap:anywhere}
  .bar{height:6px;background:var(--panel2);border-radius:3px;overflow:hidden;margin-top:8px}
  .bar > i{display:block;height:100%;background:var(--acc);width:0;transition:width .2s}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(66px,1fr));gap:5px}
  .tile{border:1px solid var(--line);border-radius:5px;padding:4px 2px 6px;
    text-align:center;cursor:pointer;font-size:11px;background:var(--panel2);line-height:1.3}
  .tile:hover{border-color:var(--acc)}
  .tile.error{background:#FEF2F2;border-color:#FCA5A5;color:var(--err)}
  .tile.warn{background:#FFFBEB;border-color:#FCD34D;color:var(--warn)}
  .tile.clean{background:#F0FDF4;border-color:#86EFAC;color:var(--ok)}
  .tile.sel{outline:2px solid var(--acc);outline-offset:1px}
  .tile.picked{outline:2px solid var(--acc);outline-offset:1px}
  .tile b{display:block;font-size:12px}
  .tile i{font-style:normal;opacity:.75;font-size:10px}
  /* Own row in normal flow, not absolutely positioned -- an absolutely positioned
     checkbox left no space reserved for it, so the <b> label right after it in the
     DOM painted on top and silently ate clicks meant for the checkbox. */
  .tile .pickrow{text-align:left;height:16px;margin-bottom:1px}
  .tile .pick{margin:0;width:13px;height:13px;cursor:pointer;vertical-align:middle}
  .issue{padding:5px 0;border-bottom:1px solid var(--line);display:flex;gap:9px;
    align-items:flex-start}
  .issue:last-child{border-bottom:none}
  .tag{flex:none;border-radius:4px;padding:1px 7px;font-size:10px;font-weight:700;
    letter-spacing:.5px;margin-top:2px}
  .tag.ERROR{background:#FEE2E2;color:var(--err)}
  .tag.WARN{background:#FEF3C7;color:var(--warn)}
  .tag.CLEAN{background:#DCFCE7;color:var(--ok)}
  .code{color:var(--acc)}
  .pill{display:inline-flex;gap:7px;align-items:center;background:var(--panel2);
    border:1px solid var(--line);border-radius:20px;padding:4px 12px;font-size:12px;
    cursor:pointer;margin:0 6px 6px 0}
  .pill:hover{border-color:var(--acc)}
  .pill.on{border-color:var(--acc);background:#DBEAFE;color:var(--acc-hover)}
  .pill em{font-style:normal;color:var(--dim)}
  .chart{margin-bottom:18px}
  .chart h3{margin:0 0 4px;font-size:13px;font-weight:600}
  .chart .sub{color:var(--dim);font-size:11px;margin-bottom:6px}
  svg.plot{width:100%;height:210px;background:var(--bg);
    border:1px solid var(--line);border-radius:6px;display:block}
  .legend{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
  .lg{font-size:10px;padding:2px 7px;border-radius:4px;border:1px solid var(--line);
    cursor:pointer;background:var(--panel2);display:inline-flex;gap:5px;align-items:center}
  .lg.off{opacity:.35;text-decoration:line-through}
  .lg .sw{width:9px;height:9px;border-radius:2px;flex:none}
  .lg .cst{color:var(--warn)}
  .vids{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
  .vids figure{margin:0}
  .vids figcaption{color:var(--dim);font-size:11px;margin-bottom:4px}
  video{width:100%;border-radius:6px;border:1px solid var(--line);background:#000}
  .err-box{background:#FEF2F2;border:1px solid #FCA5A5;
    color:var(--err);border-radius:6px;padding:10px 12px;margin-bottom:16px;
    white-space:pre-wrap;word-break:break-word}
  .muted{color:var(--dim)}
  details.opts{margin-top:10px}
  details.opts summary{cursor:pointer;color:var(--dim);font-size:12px}
  .optgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
    gap:8px;margin-top:10px}
  .optgrid label{display:flex;flex-direction:column;gap:3px;font-size:11px;color:var(--dim)}
  .sug{margin-top:8px}
  .sug button{font-size:12px;margin:3px 5px 0 0}
  #detail{scroll-margin-top:120px}
  .flexbtw{display:flex;justify-content:space-between;align-items:center;gap:10px;
    flex-wrap:wrap;margin-bottom:10px}
</style>
</head>
<body>
<script>
// Applied synchronously, before <header> is parsed, so there's no light-theme
// flash. ?theme=<family>-day|night comes from the main Daksha UI's deep link
// (see validateDataset() in app.js); a direct visit falls back to this tool's
// own last choice, stored separately since the two UIs run on different
// origins/ports and can't share localStorage.
(function(){
  try {
    var linked = new URLSearchParams(location.search).get("theme");
    var dark = linked !== null ? /night$/.test(linked)
                                : localStorage.getItem("daksha_validation_theme") === "dark";
    if (linked !== null) localStorage.setItem("daksha_validation_theme", dark ? "dark" : "light");
    if (dark) document.body.classList.add("dark");
  } catch (e) {}
})();
</script>

<header>
  <div class="hdrtop">
    <h1>Dataset Validation <span id="hdr"></span></h1>
    <div class="row">
      <button class="themebtn" id="theme-day-btn" title="Light mode">☀️</button>
      <button class="themebtn" id="theme-night-btn" title="Dark mode">🌙</button>
    </div>
  </div>
  <div class="row">
    <input type="text" id="path" placeholder="/path/to/dataset  (the directory containing meta/)">
    <button class="primary" id="run">Run</button>
    <label class="chk"><input type="checkbox" id="novideo"> skip videos (faster)</label>
    <label class="chk"><input type="checkbox" id="deepvideo"> exact video count</label>
  </div>
  <details class="opts">
    <summary>options &amp; thresholds</summary>
    <div class="optgrid">
      <label>episodes subset <input type="text" id="episodes" placeholder="e.g. 0-50,64,88"></label>
      <label>freeze frames <input type="number" id="freeze_frames" step="1" min="2"></label>
      <label>gap factor (x nominal dt) <input type="number" id="gap_factor" step="0.1"></label>
      <label>jump factor (x median step) <input type="number" id="jump_factor" step="1"></label>
      <label>short ratio (of median len) <input type="number" id="short_ratio" step="0.05"></label>
      <label>workers <input type="number" id="workers" step="1" min="1"></label>
    </div>
  </details>
  <div class="bar" id="barwrap" style="display:none"><i id="bar"></i></div>
  <div class="muted" id="status" style="margin-top:6px"></div>
</header>

<main>
  <div id="error" class="err-box" style="display:none"></div>
  <div id="body" style="display:none">

    <div class="panel">
      <div class="cards" id="cards"></div>
      <div class="meta" id="meta" style="margin-top:14px"></div>
    </div>

    <div class="panel" id="dspanel" style="display:none">
      <h2>Dataset-level findings</h2>
      <div id="dsissues"></div>
    </div>

    <div class="panel">
      <h2>Findings by type <span class="muted">(click to filter the grid)</span></h2>
      <div id="codes"></div>
      <button id="delcode" class="danger" style="display:none;margin-top:10px"></button>
    </div>

    <div class="panel">
      <div class="flexbtw">
        <h2 style="margin:0">Episodes</h2>
        <div class="row">
          <span id="filters"></span>
          <button id="copybad">Copy bad list</button>
          <button id="dlreport">Download report.json</button>
          <button id="delbad" class="danger">Delete all bad episodes</button>
        </div>
      </div>
      <div class="flexbtw" id="pickbar">
        <span class="muted" id="pickcount">0 selected <span class="muted">(check tiles below, or ctrl/cmd-click one)</span></span>
        <div class="row">
          <button id="pickall">Select all shown</button>
          <button id="pickclear">Clear selection</button>
          <button id="delpicked" class="danger" disabled>Delete selected</button>
        </div>
      </div>
      <div class="grid" id="grid"></div>
    </div>

    <div class="panel" id="detail" style="display:none"></div>
  </div>
</main>

<script>
const TOKEN = new URLSearchParams(location.search).get("token") || "";
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));

function url(path, params){
  const u = new URL(path, location.origin);
  if (TOKEN) u.searchParams.set("token", TOKEN);
  for (const k in (params||{})) u.searchParams.set(k, params[k]);
  return u.toString();
}
async function api(path, params){
  const r = await fetch(url(path, params));
  return r.json();
}

let RESULT = null, FILTER = "all", CODE_FILTER = null, SELECTED = null, POLL = null;
// Custom multi-select: an arbitrary, hand-picked set of episodes (independent of the
// "all bad" / "all with code X" bulk actions), checked via checkboxes on the grid tiles.
let PICKED = new Set();
// Which dataset this tab is tracking -- the server keeps a separate run/result per
// dataset, so every progress/result/download call must say which one it means.
let CURRENT_DATASET = "";

/* ---------------------------------------------------------------- run --- */

async function boot(){
  const c = await api("/api/config");
  if (c.ok){
    $("path").value = c.config.default_dataset || "";
    const d = c.config.defaults || {};
    $("freeze_frames").value = d.freeze_frames;
    $("gap_factor").value = d.gap_factor;
    $("jump_factor").value = d.jump_factor;
    $("short_ratio").value = d.short_ratio;
    $("workers").value = d.workers;
  }
  // Deep-link support: ?dataset=<path> prefills the path field (overriding the server's
  // default), and &autorun=1 kicks off validation immediately -- this is how the main
  // Daksha UI's "Validate" button opens this tool already pointed at the right dataset.
  const params = new URLSearchParams(location.search);
  const linkedDataset = params.get("dataset");
  if (linkedDataset){
    $("path").value = linkedDataset;
    // Deep-linked from the main Daksha UI: the dataset was already picked there, so the
    // raw filesystem path is just noise here -- hide it entirely rather than display it.
    $("path").style.display = "none";
  }
  CURRENT_DATASET = $("path").value.trim();
  if (params.get("autorun") === "1" && $("path").value.trim()){
    run();
    return;
  }
  const p = await api("/api/progress", {result: 1, dataset: CURRENT_DATASET});
  if (p.ok && p.result){ RESULT = p.result; render(); }
}

function options(){
  const num = (id) => $(id).value === "" ? null : $(id).value;
  return {
    dataset: $("path").value.trim(),
    no_video: $("novideo").checked,
    deep_video: $("deepvideo").checked,
    episodes: $("episodes").value.trim(),
    freeze_frames: num("freeze_frames"),
    gap_factor: num("gap_factor"),
    jump_factor: num("jump_factor"),
    short_ratio: num("short_ratio"),
    workers: num("workers"),
  };
}

async function run(){
  const opts = options();
  if (!opts.dataset){ showError("Enter a dataset path first."); return; }
  CURRENT_DATASET = opts.dataset;
  const dsName = opts.dataset.split("/").filter(Boolean).pop() || opts.dataset;
  $("hdr").textContent = "— " + dsName;
  document.title = dsName + " · Dataset Validation";
  hideError();
  SELECTED = null; CODE_FILTER = null; PICKED.clear();
  $("detail").style.display = "none";
  $("run").disabled = true;
  $("status").textContent = "starting...";
  const r = await fetch(url("/api/validate"), {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(opts)
  }).then(x => x.json());
  if (!r.ok){
    $("run").disabled = false;
    let msg = r.error || "failed to start";
    if (r.suggestions && r.suggestions.length){
      showError(msg, r.suggestions);
      $("status").textContent = "";
      return;
    }
    showError(msg); $("status").textContent = "";
    return;
  }
  $("barwrap").style.display = "block";
  POLL = setInterval(poll, 400);
  poll();
}

async function poll(){
  const p = await api("/api/progress", {dataset: CURRENT_DATASET});
  if (!p.ok) return;
  const s = p.progress;
  const pct = s.total ? Math.round(100 * s.done / s.total) : 0;
  $("bar").style.width = pct + "%";
  $("status").textContent = s.running
    ? `${s.stage} — ${s.done}/${s.total} episodes (${s.elapsed}s)`
    : (s.error ? "" : `${s.stage} — ${s.elapsed}s`);
  if (!s.running){
    clearInterval(POLL); POLL = null;
    $("run").disabled = false;
    $("barwrap").style.display = "none";
    if (s.error){ showError(s.error); return; }
    const r = await api("/api/result", {dataset: CURRENT_DATASET});
    if (r.ok){ RESULT = r.result; hideError(); render(); }
  }
}

/* ------------------------------------------------------------- delete --- */

// Deleting episode N renumbers every episode above N down by one (delete.py
// resequences to keep the dataset gap-free), so a batch must always go from
// the highest original index to the lowest -- otherwise indices requested
// later in the batch no longer point at the episode you meant.
async function deleteEpisodes(indices){
  const ds = RESULT.dataset;
  const ordered = [...new Set(indices)].sort((a, b) => b - a);
  if (!ordered.length) return;
  if (!confirm(`Permanently delete ${ordered.length} episode(s) from\n${ds}\n\nThis cannot be undone.`)) return;

  $("run").disabled = true;
  for (let i = 0; i < ordered.length; i++){
    const ep = ordered[i];
    $("status").textContent = `deleting episode ${String(ep).padStart(6, "0")} (${i + 1}/${ordered.length})...`;
    const r = await fetch(url("/api/delete_episode"), {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({dataset: ds, episode: ep}),
    }).then(x => x.json()).catch(e => ({ok: false, error: String(e)}));
    if (!r.ok){
      $("run").disabled = false;
      $("status").textContent = "";
      showError(`Delete failed on episode ${ep} (${i}/${ordered.length} done): ${r.error}`);
      if (i > 0){ SELECTED = null; PICKED.clear(); $("detail").style.display = "none"; run(); }
      return;
    }
  }
  $("status").textContent = "";
  SELECTED = null;
  PICKED.clear();
  $("detail").style.display = "none";
  run(); // indices shifted dataset-wide -- only a full re-validate reflects reality
}

function showError(msg, suggestions){
  const box = $("error");
  let html = esc(msg);
  if (suggestions && suggestions.length){
    html += '<div class="sug">Datasets found underneath:<br>' +
      suggestions.map(s => `<button data-sug="${esc(s)}">${esc(s)}</button>`).join("") + "</div>";
  }
  box.innerHTML = html;
  box.style.display = "block";
  box.querySelectorAll("[data-sug]").forEach(b => b.onclick = () => {
    $("path").value = b.dataset.sug; hideError(); run();
  });
}
function hideError(){ $("error").style.display = "none"; }

/* ------------------------------------------------------------- render --- */

function render(){
  const r = RESULT;
  $("body").style.display = "block";
  const dsName = r.dataset.split("/").filter(Boolean).pop() || r.dataset;
  $("hdr").textContent = "— " + dsName;
  document.title = dsName + " · Dataset Validation";
  const clean = r.clean_episodes.length;
  $("cards").innerHTML = [
    ["episodes", r.checked_episodes, ""],
    ["frames", r.frames.toLocaleString(), ""],
    ["warnings", r.warnings, "warn"],
    ["bad episodes", r.bad_episodes.length, r.bad_episodes.length ? "err" : "ok"],
    ["clean", clean + "/" + r.checked_episodes, "ok"],
  ].map(([k, v, cls]) =>
    `<div class="card ${cls}"><div class="v">${esc(v)}</div><div class="k">${k}</div></div>`
  ).join("");

  const i = r.info;
  const tasks = Object.values(i.tasks || {});
  $("meta").innerHTML = [
    ["robot", "Daksha"], ["fps", i.fps],
    ["declared episodes", i.total_episodes], ["declared frames", i.total_frames],
    ["state columns", (i.state_keys || []).join(", ") || "-"],
    ["video keys", r.options.no_video ? "skipped" : (i.video_keys || []).length],
    ["task", tasks.length === 1 ? tasks[0] : tasks.length + " tasks"],
  ].map(([k, v]) => `<div>${k}: <b>${esc(v)}</b></div>`).join("");

  const dsp = r.dataset_issues || [];
  $("dspanel").style.display = dsp.length ? "block" : "none";
  $("dsissues").innerHTML = dsp.map(issueHtml).join("");

  $("codes").innerHTML = (r.codes || []).map(c =>
    `<span class="pill ${CODE_FILTER === c.code ? "on" : ""}" data-code="${esc(c.code)}">
       <span class="tag ${c.severity}">${c.severity}</span>${esc(c.code)} <em>${c.count}</em></span>`
  ).join("") || '<span class="muted">nothing found</span>';
  $("codes").querySelectorAll("[data-code]").forEach(p => p.onclick = () => {
    CODE_FILTER = (CODE_FILTER === p.dataset.code) ? null : p.dataset.code;
    render();
  });

  // Bulk-delete-by-finding-type is only offered for ERROR-severity codes --
  // a WARN (jitter, out-of-range, ...) doesn't mean the episode is broken, so
  // it should never be reachable through a "delete this finding" action.
  const activeCode = (r.codes || []).find(c => c.code === CODE_FILTER);
  const delcode = $("delcode");
  if (activeCode && activeCode.severity === "ERROR"){
    const matching = r.episodes.filter(e => e.status === "error"
      && e.issues.some(i => i.code === CODE_FILTER && i.severity === "ERROR")).map(e => e.episode);
    delcode.style.display = "inline-block";
    delcode.textContent = `Delete ${matching.length} episode(s) with ${activeCode.code}`;
    delcode.disabled = !matching.length;
    delcode.onclick = () => deleteEpisodes(matching);
  } else {
    delcode.style.display = "none";
  }

  const counts = {all: r.checked_episodes, error: r.bad_episodes.length,
                  warn: r.warn_episodes.length, clean: clean};
  $("filters").innerHTML = ["all", "error", "warn", "clean"].map(f =>
    `<span class="pill ${FILTER === f ? "on" : ""}" data-f="${f}">${f} <em>${counts[f]}</em></span>`
  ).join("");
  $("filters").querySelectorAll("[data-f]").forEach(p => p.onclick = () => {
    FILTER = p.dataset.f; render();
  });

  $("delbad").textContent = "Delete all bad episodes" + (r.bad_episodes.length ? ` (${r.bad_episodes.length})` : "");
  $("delbad").disabled = !r.bad_episodes.length;
  $("delbad").onclick = () => deleteEpisodes(r.bad_episodes);

  renderGrid();
}

function issueHtml(it){
  return `<div class="issue"><span class="tag ${it.severity}">${it.severity}</span>
    <div><span class="code">${esc(it.code)}</span> ${esc(it.message)}</div></div>`;
}

function renderGrid(){
  let eps = RESULT.episodes;
  if (FILTER !== "all") eps = eps.filter(e => e.status === FILTER);
  if (CODE_FILTER) eps = eps.filter(e => e.issues.some(i => i.code === CODE_FILTER));
  $("grid").innerHTML = eps.length ? eps.map(e => {
    const n = e.errors || e.warnings;
    return `<div class="tile ${e.status} ${SELECTED === e.episode ? "sel" : ""} ${PICKED.has(e.episode) ? "picked" : ""}"
      data-ep="${e.episode}" title="${e.rows} frames">
      <div class="pickrow"><input type="checkbox" class="pick" data-ep="${e.episode}" ${PICKED.has(e.episode) ? "checked" : ""}></div>
      <b>${String(e.episode).padStart(6, "0")}</b>
      <i>${n ? (e.errors ? e.errors + " err" : n + " warn") : "ok"}</i></div>`;
  }).join("") : '<span class="muted">no episodes match this filter</span>';
  $("grid").querySelectorAll(".pick").forEach(cb => cb.onclick = (ev) => {
    ev.stopPropagation();
    togglePick(parseInt(cb.dataset.ep, 10));
  });
  // Scoped to .tile, not the broader "[data-ep]" -- the checkbox above also carries
  // data-ep, and that wider selector used to re-bind onclick on the checkbox too,
  // clobbering its own handler right after it was attached (so clicking a checkbox
  // did nothing; only "Select all shown", which touches PICKED directly, worked).
  $("grid").querySelectorAll(".tile").forEach(t => t.onclick = (ev) => {
    if (ev.target.classList.contains("pick")) return;
    const ep = parseInt(t.dataset.ep, 10);
    if (ev.ctrlKey || ev.metaKey){ togglePick(ep); return; }
    openEpisode(ep);
  });
  updatePickBar(eps);
}

/* ------------------------------------------------------- custom select --- */
// A hand-picked, arbitrary set of episodes -- distinct from "all bad" and "all with
// finding code X" -- checked via tile checkboxes or ctrl/cmd-click, then deleted together.

function togglePick(ep){
  if (PICKED.has(ep)) PICKED.delete(ep); else PICKED.add(ep);
  renderGrid();
}

function updatePickBar(shownEps){
  const n = PICKED.size;
  $("pickcount").innerHTML = n
    ? `<b>${n}</b> selected`
    : `0 selected <span class="muted">(check tiles below, or ctrl/cmd-click one)</span>`;
  $("delpicked").disabled = !n;
  $("delpicked").textContent = n ? `Delete selected (${n})` : "Delete selected";
  $("pickall").onclick = () => { (shownEps || []).forEach(e => PICKED.add(e.episode)); renderGrid(); };
  $("pickclear").onclick = () => { PICKED.clear(); renderGrid(); };
  $("delpicked").onclick = () => deleteEpisodes([...PICKED]);
}

/* ------------------------------------------------------------- detail --- */

async function openEpisode(ep){
  SELECTED = ep;
  renderGrid();
  const box = $("detail");
  box.style.display = "block";
  box.innerHTML = `<h2>Episode ${String(ep).padStart(6, "0")}</h2><div class="muted">loading...</div>`;
  box.scrollIntoView({behavior: "smooth", block: "start"});

  const meta = RESULT.episodes.find(e => e.episode === ep) || {};
  const d = await api("/api/episode", {
    dataset: RESULT.dataset, episode: ep,
    freeze_frames: RESULT.options.freeze_frames
  });
  if (!d.ok){ box.innerHTML = `<h2>Episode ${ep}</h2><div class="err-box">${esc(d.error)}</div>`; return; }
  renderDetail(box, meta, d.detail);
}

function renderDetail(box, meta, det){
  const facts = [
    ["frames", det.frames], ["metadata length", meta.meta_length],
    ["duration", meta.duration_s != null ? meta.duration_s + " s" : "-"],
    ["mean dt", meta.dt_mean_ms != null ? meta.dt_mean_ms + " ms" : "-"],
    ["max dt", meta.dt_max_ms != null ? meta.dt_max_ms + " ms" : "-"],
    ["nominal dt", det.nominal_dt_ms + " ms"],
  ].map(([k, v]) => `<div>${k}: <b>${esc(v)}</b></div>`).join("");

  let html = `<div class="flexbtw">
      <h2 style="margin:0">Episode ${String(det.episode).padStart(6, "0")}
        <span class="tag ${meta.errors ? "ERROR" : (meta.warnings ? "WARN" : "CLEAN")}">
        ${meta.errors ? "ERROR" : (meta.warnings ? "WARN" : "CLEAN")}</span></h2>
      <div class="row">
        <button id="delep" class="danger">Delete episode</button>
        <button id="closedet">close</button>
      </div></div>
    <div class="meta" style="margin-bottom:12px">${facts}</div>`;

  html += (meta.issues && meta.issues.length)
    ? `<div style="margin-bottom:16px">${meta.issues.map(issueHtml).join("")}</div>`
    : '<div class="muted" style="margin-bottom:16px">no findings for this episode</div>';

  if (det.dt_ms && det.dt_ms.length) html += dtChart(det);
  det.traces.forEach((t, i) => { html += traceChart(t, det, i); });

  if (det.videos.length){
    html += '<h2 style="margin-top:18px">Video</h2><div class="vids">' + det.videos.map(v =>
      `<figure><figcaption>${esc(v.key)}</figcaption>
        <video controls preload="metadata" src="${url("/api/video", {
          dataset: RESULT.dataset, episode: det.episode, key: v.key})}"></video></figure>`
    ).join("") + "</div>";
  }
  box.innerHTML = html;
  $("closedet").onclick = () => { box.style.display = "none"; SELECTED = null; renderGrid(); };
  $("delep").onclick = () => deleteEpisodes([det.episode]);
  box.querySelectorAll("[data-toggle]").forEach(el => el.onclick = () => {
    const line = document.getElementById(el.dataset.toggle);
    if (!line) return;
    const off = line.style.display === "none";
    line.style.display = off ? "" : "none";
    el.classList.toggle("off", !off);
  });
}

const W = 1000, H = 200, PAD = 8;

function color(i, n){ return `hsl(${Math.round(360 * i / Math.max(n, 1))} 72% 62%)`; }

function traceChart(trace, det, ci){
  const xs = det.x, n = trace.frames;
  const sx = (fi) => PAD + (W - 2 * PAD) * (fi / Math.max(n - 1, 1));

  // Frozen stretches are the thing you are looking for, so shade them first.
  let shade = (trace.frozen_spans || []).map(([a, b]) =>
    `<rect x="${sx(a).toFixed(1)}" y="0" width="${Math.max(sx(b) - sx(a), 1.5).toFixed(1)}"
       height="${H}" fill="rgba(255,95,86,.20)"/>`).join("");

  const lines = trace.dims.map((d, i) => {
    const span = d.max - d.min;
    const pts = d.values.map((v, k) => {
      // Each dim is min-max scaled on its own: the shape matters here, not the
      // absolute value, and a shared axis would flatten the gripper channels.
      const y = span > 1e-12 ? PAD + (H - 2 * PAD) * (1 - (v - d.min) / span) : H / 2;
      return sx(xs[k]).toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
    return `<polyline id="ln-${ci}-${i}" points="${pts}" fill="none"
      stroke="${color(i, trace.dims.length)}" stroke-width="1.2"
      vector-effect="non-scaling-stroke"/>`;
  }).join("");

  const legend = trace.dims.map((d, i) =>
    `<span class="lg" data-toggle="ln-${ci}-${i}">
      <span class="sw" style="background:${color(i, trace.dims.length)}"></span>
      ${esc(d.name)}${d.constant ? ' <span class="cst">const</span>' : ""}
      <span class="muted">${d.min.toFixed(3)}..${d.max.toFixed(3)}</span></span>`
  ).join("");

  const frozen = (trace.frozen_spans || []).length;
  return `<div class="chart"><h3>${esc(trace.key)}</h3>
    <div class="sub">${trace.dims.length} dims, ${n} frames · each dim min-max scaled
      ${frozen ? `· <span style="color:var(--err)">${frozen} frozen stretch(es) shaded</span>` : ""}</div>
    <svg class="plot" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      ${shade}${lines}</svg>
    <div class="legend">${legend}</div></div>`;
}

function dtChart(det){
  const d = det.dt_ms, n = d.length;
  const nom = det.nominal_dt_ms;
  const top = Math.max(nom * 3, ...d);
  const sx = (i) => PAD + (W - 2 * PAD) * (i / Math.max(n - 1, 1));
  const sy = (v) => PAD + (H - 2 * PAD) * (1 - Math.min(v, top) / top);
  const pts = d.map((v, i) => sx(i).toFixed(1) + "," + sy(v).toFixed(1)).join(" ");
  const thr = sy(nom * 2.5).toFixed(1);
  const base = sy(nom).toFixed(1);
  return `<div class="chart"><h3>frame interval</h3>
    <div class="sub">nominal ${nom} ms (blue) · gap threshold ${(nom * 2.5).toFixed(1)} ms (red)
      · ${det.dt_gaps.length} gap(s)</div>
    <svg class="plot" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
      <line x1="0" y1="${base}" x2="${W}" y2="${base}" stroke="var(--acc)"
        stroke-width="1" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>
      <line x1="0" y1="${thr}" x2="${W}" y2="${thr}" stroke="var(--err)"
        stroke-width="1" stroke-dasharray="4 4" vector-effect="non-scaling-stroke"/>
      <polyline points="${pts}" fill="none" stroke="#7ee787" stroke-width="1.2"
        vector-effect="non-scaling-stroke"/></svg></div>`;
}

/* --------------------------------------------------------------- misc --- */

$("run").onclick = run;
$("path").addEventListener("keydown", e => { if (e.key === "Enter") run(); });
$("copybad").onclick = () => {
  if (!RESULT) return;
  const t = RESULT.bad_episodes.map(e => String(e).padStart(6, "0")).join(", ");
  navigator.clipboard.writeText(t).then(() => {
    $("copybad").textContent = "copied " + RESULT.bad_episodes.length;
    setTimeout(() => $("copybad").textContent = "Copy bad list", 1500);
  });
};
$("dlreport").onclick = () => {
  const a = document.createElement("a");
  a.href = url("/api/result", {dataset: CURRENT_DATASET});
  a.download = "validation_report.json";
  a.click();
};
boot();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    sys.exit(main())
