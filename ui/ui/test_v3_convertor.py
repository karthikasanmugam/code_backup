import json
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import app as ui
import v3_convertor


def test_episode_metadata_writes_le_robot_tasks_list(tmp_path):
    source = tmp_path / "source"
    (source / "meta").mkdir(parents=True)
    (source / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "task": "pick up the box", "task_index": 0}) + "\n"
    )
    output = tmp_path / "converted"
    data_metadata = [{
        "episode_index": 0,
        "_orig_episode_index": 0,
        "data/chunk_index": 0,
        "data/file_index": 0,
        "dataset_from_index": 0,
        "dataset_to_index": 2,
        "length": 2,
    }]

    with patch.object(v3_convertor, "aggregate_stats", return_value={}):
        v3_convertor.convert_episodes_metadata(
            source,
            output,
            data_metadata,
            [{}],
            None,
            episodes_per_meta_file=100,
            chunk_size=1000,
        )

    episode_file = output / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = pd.read_parquet(episode_file)
    assert episodes.loc[0, "tasks"] == ["pick up the box"]
    assert "task" not in episodes.columns


def test_existing_v30_dataset_repairs_legacy_task_column(tmp_path):
    dataset = tmp_path / "legacy_v3"
    meta = dataset / "meta"
    episode_dir = meta / "episodes" / "chunk-000"
    episode_dir.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"codebase_version": "v3.0", "fps": 30}))
    (meta / "tasks.parquet").write_bytes(b"tasks placeholder")
    (meta / "stats.json").write_text(json.dumps({"action": {"mean": [0]}}))
    episode_file = episode_dir / "file-000.parquet"
    pd.DataFrame({"episode_index": [0], "task": ["pick up the box"]}).to_parquet(
        episode_file,
        index=False,
    )
    (tmp_path / "legacy_v3_old").mkdir()

    result = ui.ensure_dataset_v30(str(dataset), str(tmp_path / "conversion.log"))

    episodes = pd.read_parquet(episode_file)
    assert result["repaired"] is True
    assert episodes.loc[0, "tasks"] == ["pick up the box"]
    assert "task" not in episodes.columns
    assert ui._dataset_is_v30_ready(dataset)