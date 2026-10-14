#!/usr/bin/env python3
"""Universal validator for LeRobot-style robot datasets.

Point it at a dataset root and it tells you which episodes are broken:

    python validate_dataset.py /path/to/dataset

It auto-detects the layout from meta/info.json and works across the LeRobot
v2.x / v3 / custom_v3_local variants: episode metadata from either
meta/episodes.jsonl or meta/episodes/chunk-*/episodes.parquet, frames from
data/chunk-*/episode_*.parquet, images either inline or as per-key mp4s.

Checks performed
  files       missing / empty / orphan parquet + video, unreadable parquet
  counts      parquet rows vs metadata length vs video frame count,
              episodes shorter than the rest of the dataset
  deletion    holes in the episode numbering, non-contiguous frame_index,
              non-contiguous global index, mid-episode timestamp jumps
  timestamps  NaN, non-monotonic, duplicated, dt far from 1/fps, episode
              duration inconsistent with frame count
  joints      wrong width, NaN/Inf, dead (zero-variance) channels, frozen
              runs of identical frames, values outside meta/stats.json range,
              implausible frame-to-frame jumps
  totals      info.json total_episodes / total_frames vs what is on disk,
              task_index valid and task string consistent

Exit code is 0 when clean, 1 when any ERROR was found (also on any WARN with
--strict), so it drops straight into CI or a shell guard.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pyarrow.parquet as pq
    import pyarrow.types as pat
except ImportError:  # pragma: no cover
    sys.exit("pyarrow is required:  pip install pyarrow")

# Per-frame bookkeeping columns. They are scalars, so they are never treated as
# joint vectors, and they are validated by their own dedicated checks below.
BOOKKEEPING = {
    "timestamp", "frame_index", "episode_index", "index", "task_index",
    "task", "prompt", "task_label", "object_name", "target_side",
    "next.done", "next.reward", "next.success", "frame_id",
}

ERROR = "ERROR"
WARN = "WARN"


# --------------------------------------------------------------------------- #
# terminal helpers
# --------------------------------------------------------------------------- #

_TTY = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


def red(s: str) -> str:
    return _c(s, "31")


def yellow(s: str) -> str:
    return _c(s, "33")


def green(s: str) -> str:
    return _c(s, "32")


def bold(s: str) -> str:
    return _c(s, "1")


def dim(s: str) -> str:
    return _c(s, "2")


def paint(sev: str) -> str:
    return red(sev) if sev == ERROR else yellow(sev)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #


@dataclass
class Issue:
    episode: int | None
    code: str
    severity: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class Dataset:
    root: Path
    info: dict
    fps: float
    episodes: dict[int, dict]           # episode_index -> metadata row
    tasks: dict[int, str]               # task_index -> task string
    features: dict
    video_keys: list[str]
    state_keys: list[str]               # numeric vector columns to inspect
    feature_width: dict[str, int]       # column -> expected vector width
    stats: dict                         # meta/stats.json, may be empty
    chunks_size: int
    data_template: str
    video_template: str
    meta_sources: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_parquet_rows(path: Path) -> list[dict]:
    table = pq.read_table(path)
    cols = table.to_pydict()
    n = table.num_rows
    return [{k: cols[k][i] for k in cols} for i in range(n)]


def load_dataset(root: Path) -> tuple[Dataset, list[Issue]]:
    """Read every metadata file we can find and describe the dataset."""
    issues: list[Issue] = []
    meta = root / "meta"
    if not meta.is_dir():
        # Point at the real roots underneath rather than just refusing.
        nested = sorted({c.parent.parent for c in root.rglob("meta/info.json")})[:10]
        hint = ""
        if nested:
            hint = "\n\ndid you mean one of these?\n" + "\n".join(
                f"  {n}" for n in nested)
        raise SystemExit(f"not a dataset root (no meta/ directory): {root}{hint}")

    info_path = meta / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"missing {info_path}")
    info = json.loads(info_path.read_text())

    fps = float(info.get("fps") or 0) or 30.0
    if not info.get("fps"):
        issues.append(Issue(None, "info-no-fps", WARN, "info.json has no fps; assuming 30"))

    features = info.get("features", {}) or {}
    video_keys = list(info.get("video_keys") or info.get("camera_keys") or [])
    if not video_keys:
        video_keys = [k for k, v in features.items() if v.get("dtype") in ("video", "image")]

    # Joint/state vector columns: 1-D numeric features wider than a scalar.
    # shape [1] entries are per-frame bookkeeping (timestamp, index, ...), not
    # joint data, and string features are never vectors.
    state_keys: list[str] = []
    feature_width: dict[str, int] = {}
    for key, spec in features.items():
        if key in video_keys or key in BOOKKEEPING:
            continue
        if spec.get("dtype") in ("video", "image", "string"):
            continue
        shape = spec.get("shape")
        if isinstance(shape, list) and len(shape) == 1 and isinstance(shape[0], int) \
                and shape[0] > 1:
            state_keys.append(key)
            feature_width[key] = shape[0]

    # --- episode metadata: jsonl and/or parquet, cross-checked ------------- #
    sources: list[str] = []
    episodes: dict[int, dict] = {}
    # Which episode indices each metadata source lists, so a source that has
    # silently fallen behind the others can be named.
    src_ids: dict[str, set[int]] = {}

    jsonl = meta / "episodes.jsonl"
    if jsonl.is_file():
        sources.append("meta/episodes.jsonl")
        src_ids["meta/episodes.jsonl"] = set()
        for row in _read_jsonl(jsonl):
            idx = row.get("episode_index")
            if idx is None:
                continue
            if idx in episodes:
                issues.append(
                    Issue(idx, "meta-duplicate", ERROR,
                          "episode listed more than once in meta/episodes.jsonl")
                )
            episodes[int(idx)] = dict(row)
            src_ids["meta/episodes.jsonl"].add(int(idx))

    ep_parquets = sorted((meta / "episodes").rglob("*.parquet")) if (meta / "episodes").is_dir() else []
    for p in ep_parquets:
        rel_name = str(p.relative_to(root))
        sources.append(rel_name)
        src_ids.setdefault(rel_name, set())
        for row in _read_parquet_rows(p):
            idx = row.get("episode_index")
            if idx is None:
                continue
            idx = int(idx)
            src_ids[rel_name].add(idx)
            if idx in episodes:
                # Cross-check the two metadata sources against each other.
                a, b = episodes[idx].get("length"), row.get("length")
                if a is not None and b is not None and int(a) != int(b):
                    issues.append(
                        Issue(idx, "meta-length-disagree", ERROR,
                              f"metadata sources disagree on length: "
                              f"episodes.jsonl={a} vs episodes.parquet={b}")
                    )
                episodes[idx] = {**row, **episodes[idx]}
            else:
                episodes[idx] = dict(row)

    # An episode listed by one metadata source and not another means a writer
    # updated only part of the metadata -- typically a half-finished deletion.
    if len(src_ids) > 1:
        union: set[int] = set().union(*src_ids.values())
        for name, ids in src_ids.items():
            gone = sorted(union - ids)
            if gone:
                preview = ", ".join(str(g) for g in gone[:20])
                more = f" (+{len(gone) - 20} more)" if len(gone) > 20 else ""
                issues.append(Issue(None, "meta-source-incomplete", ERROR,
                                    f"{name} is missing {len(gone)} episode(s) that other "
                                    f"metadata sources list: {preview}{more}"))

    if not episodes:
        issues.append(Issue(None, "meta-no-episodes", WARN,
                            "no episode metadata found; falling back to files on disk"))

    # --- tasks ------------------------------------------------------------- #
    tasks: dict[int, str] = {}
    tasks_jsonl = meta / "tasks.jsonl"
    tasks_parquet = meta / "tasks.parquet"
    if tasks_jsonl.is_file():
        for row in _read_jsonl(tasks_jsonl):
            if "task_index" in row:
                tasks[int(row["task_index"])] = row.get("task", "")
    elif tasks_parquet.is_file():
        for row in _read_parquet_rows(tasks_parquet):
            if "task_index" in row:
                tasks[int(row["task_index"])] = row.get("task", "")

    stats_path = meta / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}

    ds = Dataset(
        root=root,
        info=info,
        fps=fps,
        episodes=episodes,
        tasks=tasks,
        features=features,
        video_keys=video_keys,
        state_keys=state_keys,
        feature_width=feature_width,
        stats=stats,
        chunks_size=int(info.get("chunks_size") or 1000),
        data_template=info.get("data_path")
        or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        video_template=info.get("video_path")
        or "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4",
        meta_sources=sources,
    )
    return ds, issues


# --------------------------------------------------------------------------- #
# path resolution
# --------------------------------------------------------------------------- #


def data_path(ds: Dataset, ep: int) -> Path | None:
    """Locate an episode's parquet, template first then a glob fallback."""
    rel = ds.data_template.format(
        episode_chunk=ep // ds.chunks_size, episode_index=ep, video_key=""
    )
    p = ds.root / rel
    if p.is_file():
        return p
    hits = sorted((ds.root / "data").rglob(f"episode_{ep:06d}.parquet"))
    return hits[0] if hits else None


def video_path(ds: Dataset, ep: int, key: str) -> Path | None:
    rel = ds.video_template.format(
        episode_chunk=ep // ds.chunks_size, episode_index=ep, video_key=key
    )
    p = ds.root / rel
    if p.is_file():
        return p
    base = ds.root / "videos"
    if not base.is_dir():
        return None
    for cand in base.rglob(f"episode_{ep:06d}.*"):
        if key in str(cand.relative_to(base)):
            return cand
    return None


def discover_episode_indices(ds: Dataset) -> list[int]:
    """Union of episodes named in metadata and parquet files present on disk."""
    found = set(ds.episodes)
    data_dir = ds.root / "data"
    if data_dir.is_dir():
        for p in data_dir.rglob("episode_*.parquet"):
            stem = p.stem.replace("episode_", "")
            if stem.isdigit():
                found.add(int(stem))
    return sorted(found)


# --------------------------------------------------------------------------- #
# video probing
# --------------------------------------------------------------------------- #


def _ffprobe(args: list[str]) -> str | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", *args],
            capture_output=True, text=True, timeout=120,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def video_frame_count(path: Path, deep: bool) -> int | None:
    """Frames in a video. Container metadata first; packet count if absent."""
    common = ["-select_streams", "v:0", "-of", "default=nokey=1:noprint_wrappers=1", str(path)]
    if not deep:
        s = _ffprobe(["-show_entries", "stream=nb_frames", *common])
        if s and s.isdigit() and int(s) > 0:
            return int(s)
    s = _ffprobe(["-count_packets", "-show_entries", "stream=nb_read_packets", *common])
    return int(s) if s and s.isdigit() else None


# --------------------------------------------------------------------------- #
# per-episode validation
# --------------------------------------------------------------------------- #


def _vector_column(col: Any, name: str, ep: int) -> tuple[np.ndarray | None, Issue | None]:
    """Turn a list-typed parquet column into a 2-D float array."""
    try:
        arr = np.asarray([np.asarray(v, dtype=np.float64) for v in col])
    except (ValueError, TypeError):
        return None, Issue(ep, "col-ragged", ERROR,
                           f"{name}: rows have differing widths (ragged column)")
    if arr.ndim != 2:
        return None, Issue(ep, "col-shape", ERROR,
                           f"{name}: expected 2-D (frames x dims), got shape {arr.shape}")
    return arr, None


def check_episode(ds: Dataset, ep: int, args: argparse.Namespace) -> tuple[list[Issue], dict]:
    issues: list[Issue] = []
    summary: dict[str, Any] = {"episode": ep, "rows": None, "meta_length": None}

    meta_row = ds.episodes.get(ep)
    meta_len = int(meta_row["length"]) if meta_row and meta_row.get("length") is not None else None
    summary["meta_length"] = meta_len

    path = data_path(ds, ep)
    if path is None:
        issues.append(Issue(ep, "data-missing", ERROR,
                            "episode parquet is missing (listed in metadata but not on disk)"))
        return issues, summary
    if meta_row is None:
        issues.append(Issue(ep, "meta-orphan", ERROR,
                            f"{path.relative_to(ds.root)} exists but no metadata entry"))
    if path.stat().st_size == 0:
        issues.append(Issue(ep, "data-empty", ERROR, "episode parquet is a 0-byte file"))
        return issues, summary

    try:
        table = pq.read_table(path)
    except Exception as exc:
        issues.append(Issue(ep, "data-unreadable", ERROR, f"cannot read parquet: {exc}"))
        return issues, summary

    cols = set(table.column_names)
    n = table.num_rows
    summary["rows"] = n

    if n == 0:
        issues.append(Issue(ep, "data-no-rows", ERROR, "episode parquet has 0 rows"))
        return issues, summary

    if meta_len is not None and meta_len != n:
        sev = ERROR
        issues.append(Issue(ep, "length-mismatch", sev,
                            f"metadata says {meta_len} frames, parquet has {n} "
                            f"({n - meta_len:+d})"))

    data = table.to_pydict()

    # ---- frame_index / index continuity (partial deletion) ---------------- #
    if "frame_index" in cols:
        fi = np.asarray(data["frame_index"], dtype=np.int64)
        expected = np.arange(n, dtype=np.int64)
        if not np.array_equal(fi, expected):
            bad = int(np.count_nonzero(fi != expected))
            first = int(np.argmax(fi != expected))
            issues.append(Issue(ep, "frame-index-broken", ERROR,
                                f"frame_index is not 0..{n - 1} "
                                f"({bad} positions differ, first at row {first}: "
                                f"got {fi[first]}, expected {first}) "
                                f"-- frames removed mid-episode"))
    if "index" in cols:
        gi = np.asarray(data["index"], dtype=np.int64)
        d = np.diff(gi)
        if n > 1 and not np.all(d == 1):
            holes = int(np.count_nonzero(d != 1))
            issues.append(Issue(ep, "global-index-broken", ERROR,
                                f"global index column has {holes} discontinuity(ies) "
                                f"-- rows deleted after the dataset was written"))
        summary["index_start"] = int(gi[0])
        summary["index_end"] = int(gi[-1])

    if "episode_index" in cols:
        uniq = set(int(v) for v in data["episode_index"])
        if uniq != {ep}:
            issues.append(Issue(ep, "episode-index-wrong", ERROR,
                                f"episode_index column holds {sorted(uniq)}, expected [{ep}]"))

    # ---- tasks ------------------------------------------------------------ #
    if "task_index" in cols:
        tset = set(int(v) for v in data["task_index"])
        unknown = sorted(t for t in tset if ds.tasks and t not in ds.tasks)
        if unknown:
            issues.append(Issue(ep, "task-index-unknown", ERROR,
                                f"task_index {unknown} not present in meta/tasks"))
        if meta_row and meta_row.get("task_index") is not None:
            if tset != {int(meta_row["task_index"])}:
                issues.append(Issue(ep, "task-index-mismatch", WARN,
                                    f"task_index in frames {sorted(tset)} != "
                                    f"metadata {meta_row['task_index']}"))
    if "task" in cols and ds.tasks:
        tvals = set(str(v) for v in data["task"])
        if len(tvals) > 1:
            issues.append(Issue(ep, "task-inconsistent", WARN,
                                f"episode mixes {len(tvals)} different task strings"))
        expected_task = ds.tasks.get(int(data["task_index"][0])) if "task_index" in cols else None
        if expected_task is not None and tvals and next(iter(tvals)) != expected_task:
            issues.append(Issue(ep, "task-text-mismatch", WARN,
                                f"task text {next(iter(tvals))!r} != meta/tasks "
                                f"{expected_task!r}"))

    # ---- timestamps ------------------------------------------------------- #
    nominal = 1.0 / ds.fps
    if "timestamp" in cols:
        ts = np.asarray(data["timestamp"], dtype=np.float64)
        if not np.all(np.isfinite(ts)):
            issues.append(Issue(ep, "ts-nan", ERROR,
                                f"{int(np.count_nonzero(~np.isfinite(ts)))} non-finite timestamps"))
        else:
            absolute = float(np.median(ts)) > 1e8  # unix epoch rather than 0-based
            summary["timestamp_mode"] = "epoch" if absolute else "relative"
            rel = ts - ts[0]
            duration = float(rel[-1])
            summary["duration_s"] = round(duration, 3)

            d = np.diff(ts)
            if n > 1:
                if np.any(d <= 0):
                    k = int(np.count_nonzero(d <= 0))
                    where = int(np.argmax(d <= 0))
                    issues.append(Issue(ep, "ts-not-monotonic", ERROR,
                                        f"timestamps go backwards or repeat at {k} place(s), "
                                        f"first between rows {where} and {where + 1}"))
                gaps = np.nonzero(d > nominal * args.gap_factor)[0]
                if gaps.size:
                    worst = int(gaps[np.argmax(d[gaps])])
                    lost = float(d[worst]) / nominal - 1.0
                    issues.append(Issue(ep, "ts-gap", ERROR,
                                        f"{gaps.size} timestamp gap(s) > "
                                        f"{args.gap_factor:g}x nominal dt; worst at row {worst} "
                                        f"= {d[worst] * 1000:.0f} ms (~{lost:.0f} frames "
                                        f"dropped, nominal {nominal * 1000:.1f} ms)"))
                jitter = np.nonzero(d > nominal * args.jitter_factor)[0]
                soft = int(jitter.size - gaps.size)
                if soft > 0:
                    issues.append(Issue(ep, "ts-jitter", WARN,
                                        f"{soft} frame interval(s) > "
                                        f"{args.jitter_factor:g}x nominal dt "
                                        f"(recording hiccups, no frames lost)"))
                summary["dt_mean_ms"] = round(float(d.mean()) * 1000, 2)
                summary["dt_max_ms"] = round(float(d.max()) * 1000, 2)

            expected_dur = (n - 1) * nominal
            if expected_dur > 0 and abs(duration - expected_dur) > args.duration_tol * expected_dur:
                issues.append(Issue(ep, "duration-mismatch", WARN,
                                    f"spans {duration:.2f}s but {n} frames at {ds.fps:g} fps "
                                    f"should span {expected_dur:.2f}s "
                                    f"({(duration / expected_dur - 1) * 100:+.0f}%)"))

            # Metadata start/end, when the recorder wrote them.
            if meta_row and meta_row.get("timestamp_start") is not None and absolute:
                ms, me = float(meta_row["timestamp_start"]), float(meta_row.get("timestamp_end", 0))
                if abs(ts[0] - ms) > 1.0:
                    issues.append(Issue(ep, "ts-start-mismatch", WARN,
                                        f"first frame {ts[0]:.2f} differs from metadata "
                                        f"timestamp_start {ms:.2f} by {ts[0] - ms:+.2f}s"))
                if me and abs(ts[-1] - me) > 1.0:
                    issues.append(Issue(ep, "ts-end-mismatch", WARN,
                                        f"last frame {ts[-1]:.2f} differs from metadata "
                                        f"timestamp_end {me:.2f} by {ts[-1] - me:+.2f}s"))
    else:
        issues.append(Issue(ep, "ts-absent", WARN, "no timestamp column"))

    # ---- joint / state vectors -------------------------------------------- #
    for key in ds.state_keys:
        if key not in cols:
            issues.append(Issue(ep, "col-missing", ERROR,
                                f"column {key} is declared in info.json but absent here"))

    # Work from the parquet schema so datasets without a features block, or with
    # extra vectors info.json does not mention, are still covered.
    vector_cols = []
    for field_ in table.schema:
        if field_.name in ds.video_keys or field_.name in BOOKKEEPING:
            continue
        t = field_.type
        if pat.is_list(t) or pat.is_large_list(t) or pat.is_fixed_size_list(t):
            if pat.is_floating(t.value_type) or pat.is_integer(t.value_type):
                vector_cols.append(field_.name)

    for key in vector_cols:
        arr, err = _vector_column(data[key], key, ep)
        if err is not None:
            issues.append(err)
            continue

        width = ds.feature_width.get(key)
        if width is not None and arr.shape[1] != width:
            issues.append(Issue(ep, "dim-mismatch", ERROR,
                                f"{key}: {arr.shape[1]} dims, info.json declares {width}"))

        finite = np.isfinite(arr)
        if not finite.all():
            rows = np.unique(np.nonzero(~finite)[0])
            dims = np.unique(np.nonzero(~finite)[1])
            issues.append(Issue(ep, "joint-nan", ERROR,
                                f"{key}: {int((~finite).sum())} NaN/Inf values in "
                                f"{rows.size} row(s), dims {dims.tolist()}"))
            continue

        # Dead channels. Whether a constant joint is a defect depends on the rest
        # of the dataset -- an arm that is idle by design is constant in every
        # episode -- so only the raw fact is recorded here and check_global
        # decides. A column that is constant in *all* dims is always wrong.
        spread = arr.max(axis=0) - arr.min(axis=0)
        dead = np.nonzero(spread < args.dead_eps)[0]
        summary.setdefault("dead_dims", {})[key] = dead.tolist()
        summary.setdefault("width", {})[key] = int(arr.shape[1])
        if dead.size == arr.shape[1]:
            issues.append(Issue(ep, "joint-all-dead", ERROR,
                                f"{key}: all {arr.shape[1]} dims are constant across all "
                                f"{n} frames -- feedback was frozen during recording"))

        # Frozen runs: identical consecutive frames, i.e. the publisher stalled.
        if n > 1:
            same = np.all(np.isclose(arr[1:], arr[:-1], atol=args.dead_eps), axis=1)
            run = best = 0
            for s in same:
                run = run + 1 if s else 0
                best = max(best, run)
            if best >= args.freeze_frames:
                issues.append(Issue(ep, "joint-frozen-run", ERROR,
                                    f"{key}: {best + 1} identical consecutive frames "
                                    f"({(best + 1) / ds.fps:.1f}s) -- stream stalled"))

            # Implausible single-step jumps hint at a splice.
            step = np.abs(np.diff(arr, axis=0)).max(axis=1)
            if step.size:
                med = float(np.median(step))
                thr = max(med * args.jump_factor, args.jump_floor)
                jumps = np.nonzero(step > thr)[0]
                if jumps.size:
                    worst = int(jumps[np.argmax(step[jumps])])
                    issues.append(Issue(ep, "joint-jump", WARN,
                                        f"{key}: {jumps.size} frame-to-frame jump(s) > "
                                        f"{thr:.3f} rad; worst {step[worst]:.3f} rad at row "
                                        f"{worst}->{worst + 1} (median step {med:.4f})"))

        # Values outside the dataset-wide range recorded in meta/stats.json.
        st = ds.stats.get(key)
        if st and "min" in st and "max" in st:
            lo = np.asarray(st["min"], dtype=np.float64)
            hi = np.asarray(st["max"], dtype=np.float64)
            if lo.shape == (arr.shape[1],):
                tol = (hi - lo) * args.range_tol + 1e-9
                below = np.nonzero((arr < lo - tol).any(axis=0))[0]
                above = np.nonzero((arr > hi + tol).any(axis=0))[0]
                out = sorted(set(below.tolist()) | set(above.tolist()))
                if out:
                    issues.append(Issue(ep, "joint-out-of-range", WARN,
                                        f"{key}: dim(s) {out} exceed meta/stats.json "
                                        f"min/max by more than {args.range_tol:.0%}"))

    # ---- videos ----------------------------------------------------------- #
    if not args.no_video:
        for key in ds.video_keys:
            vp = video_path(ds, ep, key)
            if vp is None:
                issues.append(Issue(ep, "video-missing", ERROR, f"{key}: video file missing"))
                continue
            if vp.stat().st_size == 0:
                issues.append(Issue(ep, "video-empty", ERROR, f"{key}: 0-byte video"))
                continue
            frames = video_frame_count(vp, args.deep_video)
            if frames is None:
                issues.append(Issue(ep, "video-unreadable", ERROR,
                                    f"{key}: ffprobe could not read {vp.name} (corrupt/truncated)"))
                continue
            summary.setdefault("video_frames", {})[key] = frames
            if frames != n:
                sev = ERROR if abs(frames - n) > args.video_tol else WARN
                issues.append(Issue(ep, "video-length-mismatch", sev,
                                    f"{key}: {frames} video frames vs {n} parquet rows "
                                    f"({frames - n:+d})"))

    return issues, summary


# --------------------------------------------------------------------------- #
# dataset-wide validation
# --------------------------------------------------------------------------- #


def check_global(ds: Dataset, eps: list[int], summaries: list[dict],
                 args: argparse.Namespace) -> list[Issue]:
    issues: list[Issue] = []
    # With --episodes only part of the dataset was read, so totals, numbering
    # holes and orphan files cannot be judged; the per-episode checks still are.
    whole = not args.episodes

    # Holes in the episode numbering: episodes deleted from the set.
    if eps and whole:
        full = set(range(min(eps), max(eps) + 1))
        missing = sorted(full - set(eps))
        if missing:
            preview = ", ".join(str(m) for m in missing[:20])
            more = f" (+{len(missing) - 20} more)" if len(missing) > 20 else ""
            issues.append(Issue(None, "episode-numbering-hole", WARN,
                                f"{len(missing)} episode number(s) missing between "
                                f"{min(eps)} and {max(eps)} -- episodes were deleted after "
                                f"recording: {preview}{more}"))
        if min(eps) != 0:
            issues.append(Issue(None, "episode-start", WARN,
                                f"episode numbering starts at {min(eps)}, not 0"))

    declared_eps = ds.info.get("total_episodes")
    if whole and declared_eps is not None and int(declared_eps) != len(eps):
        issues.append(Issue(None, "total-episodes-mismatch", ERROR,
                            f"info.json total_episodes={declared_eps} but {len(eps)} "
                            f"episode(s) found"))

    actual_frames = sum(s["rows"] or 0 for s in summaries)
    declared_frames = ds.info.get("total_frames")
    if whole and declared_frames is not None and int(declared_frames) != actual_frames:
        issues.append(Issue(None, "total-frames-mismatch", ERROR,
                            f"info.json total_frames={declared_frames} but parquet files "
                            f"hold {actual_frames} ({actual_frames - int(declared_frames):+d})"))

    # Episodes markedly shorter than the rest of the dataset.
    lengths = [s["rows"] for s in summaries if s.get("rows")]
    if len(lengths) >= 3:
        med = float(np.median(lengths))
        cut = med * args.short_ratio
        for s in summaries:
            r = s.get("rows")
            if r and r < cut:
                issues.append(Issue(s["episode"], "episode-short", ERROR,
                                    f"{r} frames vs dataset median {med:.0f} "
                                    f"({r / med:.0%}) -- truncated recording"))

    # Global index must be contiguous across the whole dataset.
    # Only meaningful between consecutive episode numbers: when episodes have
    # been deleted the global index legitimately skips their rows.
    ordered = [s for s in sorted(summaries, key=lambda x: x["episode"]) if "index_start" in s]
    for prev, cur in zip(ordered, ordered[1:]):
        if cur["episode"] != prev["episode"] + 1:
            continue
        if cur["index_start"] != prev["index_end"] + 1:
            issues.append(Issue(cur["episode"], "global-index-jump", WARN,
                                f"global index starts at {cur['index_start']} but previous "
                                f"episode {prev['episode']} ended at {prev['index_end']}"))

    # A constant channel is only an episode defect when it is frozen here and
    # moves elsewhere; channels constant everywhere are a dataset-wide property.
    vec_keys = sorted({k for s in summaries for k in s.get("dead_dims", {})})
    for key in vec_keys:
        rows = [s for s in summaries if key in s.get("dead_dims", {})]
        width = max(s["width"][key] for s in rows)
        dead_count = np.zeros(width, dtype=int)
        for s in rows:
            for d in s["dead_dims"][key]:
                if d < width:
                    dead_count[d] += 1
        total = len(rows)
        always = [d for d in range(width) if dead_count[d] == total]
        if always:
            issues.append(Issue(None, "channel-constant-dataset-wide", WARN,
                                f"{key}: dim(s) {always} never change in ANY of the "
                                f"{total} episode{'s' if total != 1 else ''} "
                                f"-- no signal to learn on these"))
        rare = {d for d in range(width)
                if d not in always and dead_count[d] <= total * args.dead_ratio}
        for s in rows:
            odd = sorted(d for d in s["dead_dims"][key] if d in rare)
            if odd:
                issues.append(Issue(s["episode"], "joint-dead-dim", WARN,
                                    f"{key}: dim(s) {odd} are frozen in this episode but "
                                    f"move in the other {total - 1}"))

    # Orphan videos with no corresponding episode.
    if whole and not args.no_video:
        known = set(eps)
        vroot = ds.root / "videos"
        if vroot.is_dir():
            orphans: set[int] = set()
            for p in vroot.rglob("episode_*.mp4"):
                stem = p.stem.replace("episode_", "")
                if stem.isdigit() and int(stem) not in known:
                    orphans.add(int(stem))
            if orphans:
                issues.append(Issue(None, "video-orphan", WARN,
                                    f"{len(orphans)} video(s) with no episode: "
                                    f"{sorted(orphans)[:20]}"))
    return issues


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #


def report(ds: Dataset, eps: list[int], issues: list[Issue],
           summaries: list[dict], args: argparse.Namespace) -> int:
    per_ep: dict[int | None, list[Issue]] = {}
    for it in issues:
        per_ep.setdefault(it.episode, []).append(it)

    errors = [i for i in issues if i.severity == ERROR]
    warns = [i for i in issues if i.severity == WARN]
    bad_eps = sorted(e for e, v in per_ep.items()
                     if e is not None and any(i.severity == ERROR for i in v))
    warn_eps = sorted(e for e, v in per_ep.items()
                      if e is not None and e not in bad_eps)

    print()
    print(bold("=" * 78))
    print(bold(f" DATASET  {ds.root}"))
    print(bold("=" * 78))
    frames = sum(s["rows"] or 0 for s in summaries)
    print(f"  robot_type     {ds.info.get('robot_type', '?')}")
    print(f"  codebase       {ds.info.get('codebase_version', '?')}")
    print(f"  fps            {ds.fps:g}")
    print(f"  episodes       {len(eps)}")
    print(f"  frames         {frames}")
    print(f"  tasks          {len(ds.tasks)}")
    print(f"  video keys     {len(ds.video_keys)}"
          + (dim("  (skipped)") if args.no_video else ""))
    print(f"  state columns  {', '.join(k for k in ds.state_keys if k in ds.features) or '-'}")
    if ds.meta_sources:
        print(f"  metadata from  {', '.join(ds.meta_sources)}")

    # Dataset-wide problems.
    if per_ep.get(None):
        print()
        print(bold("-- dataset-level " + "-" * 61))
        for it in per_ep[None]:
            print(f"  [{paint(it.severity)}] {it.code}: {it.message}")

    # Per-episode detail.
    detail = bad_eps + (warn_eps if not args.errors_only else [])
    if detail:
        print()
        print(bold("-- episodes with findings " + "-" * 52))
        for ep in detail:
            eps_issues = per_ep[ep]
            worst = ERROR if any(i.severity == ERROR for i in eps_issues) else WARN
            head = red(f"episode {ep:06d}") if worst == ERROR else yellow(f"episode {ep:06d}")
            summ = next((s for s in summaries if s["episode"] == ep), {})
            extra = []
            if summ.get("rows") is not None:
                extra.append(f"{summ['rows']} frames")
            if summ.get("duration_s") is not None:
                extra.append(f"{summ['duration_s']:.1f}s")
            print(f"\n  {head}  {dim('(' + ', '.join(extra) + ')') if extra else ''}")
            for it in eps_issues:
                print(f"      [{paint(it.severity)}] {it.code}: {it.message}")

    # The bottom line the user actually reads.
    print()
    print(bold("=" * 78))
    if not issues:
        print(green(bold(f" PASS  all {len(eps)} episodes are consistent")))
    else:
        print(bold(f" {len(errors)} error(s), {len(warns)} warning(s)"))
        if bad_eps:
            print()
            print(red(bold(f" BAD EPISODES ({len(bad_eps)}) -- these are wrong:")))
            for line in _wrap_ints(bad_eps):
                print(red(f"   {line}"))
        if warn_eps and not args.errors_only:
            print()
            print(yellow(f" episodes with warnings only ({len(warn_eps)}):"))
            for line in _wrap_ints(warn_eps):
                print(yellow(f"   {line}"))
        clean = len(eps) - len(bad_eps) - len(warn_eps)
        print()
        print(f" clean episodes: {clean}/{len(eps)}")
    print(bold("=" * 78))
    print()

    if args.json:
        out = {
            "dataset": str(ds.root),
            "episodes": len(eps),
            "frames": frames,
            "fps": ds.fps,
            "errors": len(errors),
            "warnings": len(warns),
            "bad_episodes": bad_eps,
            "warn_episodes": warn_eps,
            "issues": [i.as_dict() for i in issues],
            "episode_summaries": summaries,
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"report written to {args.json}\n")

    if errors:
        return 1
    return 1 if (warns and args.strict) else 0


def _wrap_ints(values: list[int], per_line: int = 12) -> list[str]:
    return [", ".join(f"{v:06d}" for v in values[i:i + per_line])
            for i in range(0, len(values), per_line)]


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    """Single source of truth for options and defaults, shared with the web UI."""
    p = argparse.ArgumentParser(
        description="Validate a LeRobot-style dataset and report which episodes are broken.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("dataset", type=Path, help="path to the dataset root (the dir holding meta/)")
    p.add_argument("--json", metavar="FILE", help="also write a machine-readable report here")
    p.add_argument("--no-video", action="store_true", help="skip video probing (much faster)")
    p.add_argument("--deep-video", action="store_true",
                   help="count video packets instead of trusting container metadata (slow, exact)")
    p.add_argument("--errors-only", action="store_true", help="hide warning-only episodes")
    p.add_argument("--strict", action="store_true", help="exit non-zero on warnings too")
    p.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4) * 2),
                   help="parallel episode workers")
    p.add_argument("--episodes", help="only check these, e.g. '0-20,45,90-95'")

    t = p.add_argument_group("thresholds")
    t.add_argument("--gap-factor", type=float, default=2.5,
                   help="dt above this multiple of 1/fps counts as dropped frames")
    t.add_argument("--jitter-factor", type=float, default=1.5,
                   help="dt above this multiple of 1/fps counts as jitter")
    t.add_argument("--duration-tol", type=float, default=0.20,
                   help="allowed relative error between span and frames/fps")
    t.add_argument("--short-ratio", type=float, default=0.9,
                   help="episodes below this fraction of the median length are flagged")
    t.add_argument("--dead-eps", type=float, default=1e-9,
                   help="a channel varying less than this is considered dead")
    t.add_argument("--dead-ratio", type=float, default=0.10,
                   help="a dim frozen in at most this fraction of episodes is normally alive, "
                        "so freezing in one episode is an anomaly")
    t.add_argument("--freeze-frames", type=int, default=30,
                   help="identical consecutive frames before a stall is reported")
    t.add_argument("--jump-factor", type=float, default=25.0,
                   help="frame-to-frame step above this multiple of the median is a jump")
    t.add_argument("--jump-floor", type=float, default=0.05,
                   help="minimum absolute step (rad) before a jump is reported")
    t.add_argument("--range-tol", type=float, default=0.02,
                   help="fraction of the stats.json range values may exceed it by")
    t.add_argument("--video-tol", type=int, default=1,
                   help="video/parquet frame difference tolerated as a warning")

    return p


def main() -> int:
    args = build_parser().parse_args()
    root = args.dataset.expanduser().resolve()
    if not root.is_dir():
        return int(bool(sys.stderr.write(f"no such directory: {root}\n"))) or 2

    ds, issues = load_dataset(root)
    eps = discover_episode_indices(ds)

    if args.episodes:
        wanted: set[int] = set()
        for part in args.episodes.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                wanted.update(range(int(a), int(b) + 1))
            elif part:
                wanted.add(int(part))
        eps = [e for e in eps if e in wanted]

    if not eps:
        print("no episodes found")
        return 2

    print(f"validating {len(eps)} episode(s) in {root} ...", flush=True)

    summaries: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for ep_issues, summ in pool.map(lambda e: check_episode(ds, e, args), eps):
            issues.extend(ep_issues)
            summaries.append(summ)
            done += 1
            if _TTY and done % 5 == 0:
                print(f"\r  {done}/{len(eps)}", end="", flush=True)
    if _TTY:
        print(f"\r  {len(eps)}/{len(eps)}")

    issues.extend(check_global(ds, eps, summaries, args))
    issues.sort(key=lambda i: (i.episode if i.episode is not None else -1,
                               0 if i.severity == ERROR else 1, i.code))
    return report(ds, eps, issues, summaries, args)


if __name__ == "__main__":
    sys.exit(main())
