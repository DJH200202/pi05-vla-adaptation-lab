"""Convert all 500 original Spatial demonstrations to a self-contained LeRobot dataset."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "local/libero-spatial-500-v1"


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def image_transform(image, size):
    # Match OpenPI's LIBERO inference image convention. PNG preserves the resulting pixels.
    rotated = np.ascontiguousarray(image[::-1, ::-1])
    return np.asarray(Image.fromarray(rotated).resize((size, size), Image.Resampling.BILINEAR))


def read_vectors(demo):
    state = np.concatenate([demo["obs/ee_pos"][:], demo["obs/ee_ori"][:],
                            demo["obs/gripper_states"][:]], axis=1).astype(np.float32)
    actions = demo["actions"][:].astype(np.float32)
    if state.shape != (len(actions), 8) or actions.shape[1:] != (7,):
        raise ValueError("Unexpected state/action dimensions")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Non-finite state/action values")
    return state, actions


def inspect_source(source, seed):
    source_index_path = source.parent / "SOURCE.json"
    source_index = json.loads(source_index_path.read_text())
    expected = {row["path"]: row for row in source_index["files"]}
    files = sorted(source.glob("*.hdf5"))
    if len(files) != 10:
        raise ValueError(f"Expected 10 Spatial task files, found {len(files)}")
    records, sources, frequencies = [], [], set()
    for task_index, path in enumerate(files):
        relative = str(path.relative_to(source.parent))
        actual_sha = digest(path)
        if path.is_symlink() or actual_sha != expected[relative]["sha256"]:
            raise ValueError(f"Original HDF5 checksum mismatch: {relative}")
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            env = json.loads(data.attrs["env_args"])["env_kwargs"]
            frequencies.add(int(env["control_freq"]))
            if env["controller_configs"].get("control_delta") is not True:
                raise ValueError("Expected delta OSC actions")
            if str(data.attrs["macros_image_convention"]) != "opengl":
                raise ValueError("Unexpected image convention")
            prompt = json.loads(data.attrs["problem_info"])["language_instruction"]
            demos = sorted(data, key=lambda name: int(name.rsplit("_", 1)[1]))
            if len(demos) != 50 or {int(name.rsplit("_", 1)[1]) for name in demos} != set(range(50)):
                raise ValueError("Expected unique demo_0 through demo_49 for every task")
            ranked = sorted(demos, key=lambda name: hashlib.sha256(
                f"spatial-500-v1:{seed}:{path.name}:{name}".encode()).hexdigest())
            validation = set(ranked[:5])
            frames = 0
            for name in demos:
                demo = data[name]
                state, actions = read_vectors(demo)
                n = len(actions)
                if n < 2 or int(demo.attrs["num_samples"]) != n:
                    raise ValueError("Invalid demonstration length")
                for key in ["agentview_rgb", "eye_in_hand_rgb"]:
                    image = demo["obs"][key]
                    if image.shape != (n, 128, 128, 3) or image.dtype != np.uint8:
                        raise ValueError(f"Unexpected raw image shape/dtype: {relative}/{name}/{key}")
                records.append({"task_index": task_index, "task": prompt, "source_file": relative,
                                "source_demo": name, "source_demo_index": int(name.rsplit("_", 1)[1]),
                                "source_length": n, "length": n - 1, "source_observation_frame_start": 0, "source_action_frame_start": 1,
                                "split": "validation" if name in validation else "train",
                                "terminal_done": bool(demo["dones"][-1]),
                                "terminal_reward": float(np.asarray(demo["rewards"][-1]).reshape(-1)[0])})
                frames += n
            if frames != expected[relative]["frames"]:
                raise ValueError("Frame total differs from SOURCE.json")
            sources.append({"path": relative, "sha256": actual_sha, "episodes": 50, "frames": frames,
                            "control_freq": env["control_freq"], "image_convention": str(data.attrs["macros_image_convention"])})
        print(f"Source verified {task_index + 1}/10: {path.name}", flush=True)
    if frequencies != {20} or len(records) != 500 or len({row["task"] for row in records}) != 10:
        raise ValueError("Unexpected source FPS, episode count or task count")
    # Contiguous split ranges make metadata and explicit episode-id lists agree.
    records = [row for row in records if row["split"] == "train"] + [row for row in records if row["split"] == "validation"]
    for index, row in enumerate(records):
        row["episode_index"] = index
    provenance = {"source_repo": source_index["source_repo"], "revision": source_index["revision"],
                  "source_index_sha256": digest(source_index_path), "files": sources}
    return records, provenance


def verify_episode(path, record, state, actions, cameras, size, fps, offset, pool):
    table = pq.read_table(path)
    n = record["length"]
    if table.num_rows != n:
        raise ValueError("Saved Parquet length mismatch")
    for key, expected in [("state", state), ("actions", actions)]:
        actual = np.asarray(table[key].to_pylist(), dtype=np.float32)
        np.testing.assert_array_equal(actual, expected)
    for key, expected in [("frame_index", np.arange(n)), ("index", np.arange(offset, offset + n)),
                          ("episode_index", np.full(n, record["episode_index"])),
                          ("task_index", np.full(n, record["task_index"]))]:
        np.testing.assert_array_equal(np.asarray(table[key]), expected)
    np.testing.assert_allclose(np.asarray(table["timestamp"]), np.arange(n) / fps, atol=1e-6, rtol=0)
    images = {key: table[key].to_pylist() for key in cameras}

    def check_frame(index):
        for key, raw in cameras.items():
            item = images[key][index]
            if not item.get("bytes"):
                raise ValueError("Image must be embedded in Parquet, not depend on temporary PNG files")
            with Image.open(io.BytesIO(item["bytes"])) as saved:
                pixels = np.asarray(saved.convert("RGB"))
            np.testing.assert_array_equal(pixels, image_transform(raw[index], size))

    list(pool.map(check_frame, range(n)))


def norm_stats(arrays):
    values = np.concatenate(arrays, axis=0).astype(np.float64)
    return {"mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
            "q01": np.quantile(values, 0.01, axis=0).tolist(), "q99": np.quantile(values, 0.99, axis=0).tolist()}


def reuse_all_training(source, staging, output):
    """复用已验证的500条转换结果，独立复制文件；重写划分并重算全部训练统计。"""
    started = time.monotonic()
    original = json.loads((source / "conversion.json").read_text())
    records = [json.loads(line) for line in (source / "manifests/episodes.jsonl").read_text().splitlines() if line.strip()]
    if len(records) != 500 or sorted(row["episode_index"] for row in records) != list(range(500)):
        raise ValueError("复用源必须完整包含500条Spatial示教")
    counts = {task: sum(row["task_index"] == task for row in records) for task in range(10)}
    if set(counts.values()) != {50} or sum(row["length"] for row in records) != 61750:
        raise ValueError("复用源任务/帧数量不符合Spatial500协议")
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("复用源不能依赖符号链接")
    shutil.copytree(source, staging)
    states, actions = [], []
    for row in records:
        path = staging / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError(f"复用Parquet校验失败: {path}")
        table = pq.read_table(path, columns=["state", "actions", "episode_index", "task_index"])
        if table.num_rows != row["length"]:
            raise ValueError("复用Parquet帧数不一致")
        for key in ("episode_index", "task_index"):
            if not np.all(np.asarray(table[key]) == row[key]):
                raise ValueError(f"复用Parquet {key} 不一致")
        vectors = [np.asarray(table[key].to_pylist(), dtype=np.float32) for key in ("state", "actions")]
        if not all(np.isfinite(x).all() for x in vectors):
            raise ValueError("训练统计含非有限值")
        states.append(vectors[0]); actions.append(vectors[1])
        row["split"] = "train"
    info = json.loads((staging / "meta/info.json").read_text())
    info["splits"] = {"train": "0:500"}
    write_json(staging / "meta/info.json", info)
    write_json(staging / "manifests/train_episode_ids.json", list(range(500)))
    write_json(staging / "manifests/validation_episode_ids.json", [])
    (staging / "manifests/episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    norm_path = original["norm_stats_path"]
    write_json(staging / norm_path, {"norm_stats": {"state": norm_stats(states), "actions": norm_stats(actions)}})
    summary = dict(original)
    summary.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                   protocol="train500-no-heldout", train_episodes=500, validation_episodes=0,
                   train_frames=61750, validation_frames=0, split_seed=None,
                   split_rule="all 500 episodes train; preserve original episode IDs; no held-out validation",
                   reused_processed_root=str(source.relative_to(ROOT)),
                   reused_conversion_sha256=digest(source / "conversion.json"),
                   converter_sha256=digest(Path(__file__)), norm_stats_sha256=digest(staging / norm_path),
                   validation={"all_parquet_sha256": True, "all_vector_indices_finite": True,
                               "original_conversion_checks": original["validation"], "heldout_validation": False},
                   wall_seconds=time.monotonic() - started)
    write_json(staging / "conversion.json", summary)
    staging.rename(output)
    print(json.dumps({"output": str(output), "train_episodes": 500, "train_frames": 61750,
                      "validation_episodes": 0, "norm_stats_sha256": summary["norm_stats_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/raw/libero-original/libero_spatial")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/libero-spatial-500-v1")
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse-processed", type=Path,
                        help="从项目内已有500条转换结果生成全500训练版本，不重复处理图像")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if os.environ.get("MY_PI05_ROOT") != str(ROOT):
        parser.error("Source scripts/env.sh first")
    if not source.is_relative_to(ROOT) or not output.is_relative_to(ROOT):
        parser.error("Source and output must stay inside this project")
    staging = output.with_name(output.name + ".building")
    if output.exists() or staging.exists():
        parser.error("Output or staging directory already exists; inspect it before retrying")
    if args.image_size != 256 or not 1 <= args.workers <= 16:
        parser.error("Version v1 uses 256px images and 1–16 worker threads")
    if args.reuse_processed:
        reuse = args.reuse_processed.resolve()
        if not reuse.is_relative_to(ROOT) or output.is_relative_to(reuse):
            parser.error("复用源必须在项目内，输出不能位于复用源内")
        reuse_all_training(reuse, staging, output)
        return
    os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    started = time.monotonic()
    records, provenance = inspect_source(source, args.split_seed)
    fps = 20
    features = {"image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]}}
    dataset = LeRobotDataset.create(repo_id=REPO_ID, root=staging, fps=fps, robot_type="panda",
                                   features=features, use_videos=False, image_writer_threads=args.workers)
    train_states, train_actions, frame_offset = [], [], 0
    try:
        with ExitStack() as stack, ThreadPoolExecutor(max_workers=args.workers) as pool:
            handles = {}
            for record in records:
                name = record["source_file"]
                if name not in handles:
                    handles[name] = stack.enter_context(h5py.File(source.parent / name, "r"))
                demo = handles[name]["data"][record["source_demo"]]
                state, actions = read_vectors(demo)
                # Original collector records obs AFTER action[t]. Predict action[t+1] from obs[t].
                state, actions = state[:-1], actions[1:]
                cameras = {"image": demo["obs/agentview_rgb"][:-1], "wrist_image": demo["obs/eye_in_hand_rgb"][:-1]}
                for frame in range(record["length"]):
                    dataset.add_frame({"image": image_transform(cameras["image"][frame], args.image_size),
                                       "wrist_image": image_transform(cameras["wrist_image"][frame], args.image_size),
                                       "state": state[frame], "actions": actions[frame], "task": record["task"]})
                dataset.save_episode()
                relative = dataset.meta.get_data_file_path(record["episode_index"])
                verify_episode(staging / relative, record, state, actions, cameras, args.image_size, fps, frame_offset, pool)
                record.update(path=str(relative), sha256=digest(staging / relative),
                              global_frame_start=frame_offset, global_frame_end=frame_offset + record["length"])
                frame_offset += record["length"]
                if record["split"] == "train":
                    train_states.append(state)
                    train_actions.append(actions)
                if (record["episode_index"] + 1) % 10 == 0:
                    print(f"Converted and verified {record['episode_index'] + 1}/500; frames={frame_offset}; seconds={time.monotonic() - started:.1f}", flush=True)
    finally:
        dataset.stop_image_writer()

    train_ids = [row["episode_index"] for row in records if row["split"] == "train"]
    validation_ids = [row["episode_index"] for row in records if row["split"] == "validation"]
    if train_ids != list(range(450)) or validation_ids != list(range(450, 500)) or frame_offset != 61750:
        raise ValueError("Final episode split or preserved frame total mismatch")
    info = dataset.meta.info
    if info["total_episodes"] != 500 or info["total_frames"] != frame_offset or info["total_tasks"] != 10:
        raise ValueError("LeRobot metadata totals mismatch")
    info["splits"] = {"train": "0:450", "validation": "450:500"}
    write_json(staging / "meta/info.json", info)
    norm_path = Path("assets/libero-spatial-500-v1/norm_stats.json")
    write_json(staging / norm_path, {"norm_stats": {"state": norm_stats(train_states), "actions": norm_stats(train_actions)}})
    manifest_dir = staging / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records))
    write_json(manifest_dir / "train_episode_ids.json", train_ids)
    write_json(manifest_dir / "validation_episode_ids.json", validation_ids)
    summary = {"format": "LeRobot", "codebase_version": info["codebase_version"], "repo_id": REPO_ID,
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "source": provenance,
               "converter_sha256": digest(Path(__file__)), "episodes": 500, "tasks": 10, "frames": frame_offset,
               "source_frames": sum(row["source_length"] for row in records), "fps": fps, "train_episodes": 450, "validation_episodes": 50,
               "train_frames": sum(row["length"] for row in records if row["split"] == "train"),
               "validation_frames": sum(row["length"] for row in records if row["split"] == "validation"),
               "split_seed": args.split_seed, "split_rule": "per task SHA256 spatial-500-v1:seed:filename:demo; first five hashes validation; train IDs first",
               "task_index_rule": "lexicographic HDF5 filename; map evaluation tasks by text",
               "processing": {"episode_filter": "none", "frame_filter": "omit final observation of each episode because no next action is stored", "resampling": "none",
                              "alignment": "source obs[t] is post action[t]; output pairs obs[t] with source action[t+1]",
                              "alignment_source": "https://github.com/Lifelong-Robot-Learning/LIBERO/blob/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c/scripts/create_dataset.py",
                              "images": "source uint8 128x128 RGB; rotate180 then Pillow BILINEAR resize to 256x256; lossless embedded PNG",
                              "state": "concat ee_pos[3], ee_ori axis-angle[3], gripper_states[2]; float32",
                              "actions": "original 7D delta OSC actions shifted one frame forward, cast float32; no value scaling or differencing",
                              "normalization": "population mean/std and exact q01/q99 over stored per-frame vectors from train episodes only"},
               "norm_stats_path": str(norm_path), "norm_stats_sha256": digest(staging / norm_path),
               "validation": {"source_files_sha256": True, "all_frames_state_action_roundtrip": True,
                              "all_frames_both_cameras_pixel_roundtrip": True, "all_indices_timestamps": True,
                              "train_validation_overlap": 0}, "wall_seconds": time.monotonic() - started}
    write_json(staging / "conversion.json", summary)
    # Publish only a completed, validated dataset; original HDF5 and the 432 dataset are untouched.
    staging.rename(output)
    print(json.dumps({"output": str(output), **{key: summary[key] for key in ["episodes", "tasks", "source_frames", "frames", "fps", "train_episodes", "validation_episodes", "train_frames", "validation_frames", "wall_seconds"]}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
