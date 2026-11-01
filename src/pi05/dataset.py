"""Spatial-500 数据 pipeline：真实样本、动作 chunk、模型变换和 batch。"""

import argparse
import io
import json
import os
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq


from . import ROOT, OUTPUTS
DEFAULT_DATA_ROOT = ROOT / "data/processed/libero-spatial-500-v1"


def load_episode(episode_index: int, data_root: str | Path | None = None) -> dict:
    """按 metadata 找到轨迹文件，读取整条轨迹；图片此时仍是压缩字节。"""
    root = Path(data_root or os.environ.get("MY_PI05_DATA_ROOT", DEFAULT_DATA_ROOT)).resolve()
    info = json.loads((root / "meta/info.json").read_text())
    if info["total_episodes"] != 500:
        raise ValueError(f"本 pipeline 使用 Spatial-500，请重新 source scripts/env.sh；当前路径：{root}")

    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    metadata = next((item for item in episodes if item["episode_index"] == episode_index), None)
    if metadata is None:
        raise IndexError(f"episode {episode_index} 不存在；可用编号为 0–{info['total_episodes'] - 1}")
    tasks = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]

    # 使用已有的整轨迹清单；不根据随机抽到的帧重新划分训练和验证。
    splits = []
    for split in ("train", "validation"):
        ids = json.loads((root / f"manifests/{split}_episode_ids.json").read_text())
        if episode_index in ids:
            splits.append(split)
    if len(splits) != 1:
        raise ValueError(f"episode {episode_index} 必须且只能属于一个划分，实际为 {splits}")

    relative = info["data_path"].format(
        episode_chunk=episode_index // info["chunks_size"], episode_index=episode_index,
    )
    table = pq.read_table(root / relative)
    if table.num_rows != metadata["length"]:
        raise ValueError(f"episode {episode_index} 的实际长度与 metadata 不一致")
    return {"table": table, "episode_index": episode_index, "length": table.num_rows,
            "split": splits[0], "fps": info["fps"], "info": info,
            "tasks": {item["task_index"]: item["task"] for item in tasks}}


def get_frame(episode: dict, frame_index: int) -> dict:
    """取同一行的观察与动作；转换时已完成对齐，这里不再错开一帧。"""
    if not 0 <= frame_index < episode["length"]:
        raise IndexError(f"frame {frame_index} 越界；本轨迹可用编号为 0–{episode['length'] - 1}")
    row = episode["table"].slice(frame_index, 1).to_pylist()[0]
    if row["episode_index"] != episode["episode_index"] or row["frame_index"] != frame_index:
        raise ValueError("Parquet 行内的轨迹/帧索引与请求不一致")
    sample = {"episode_index": row["episode_index"], "frame_index": row["frame_index"],
              "task_index": row["task_index"], "timestamp": float(row["timestamp"]),
              "prompt": episode["tasks"][row["task_index"]], "split": episode["split"],
              "episode_length": episode["length"], "fps": episode["fps"]}

    for key in ("image", "wrist_image"):
        # LeRobot 的 image 列是 {bytes, path}；本版本 PNG 已嵌入 bytes。
        with Image.open(io.BytesIO(row[key]["bytes"])) as image:
            sample[key] = np.array(image.convert("RGB"), dtype=np.uint8)
    for key in ("state", "actions"):
        sample[key] = np.asarray(row[key], dtype=np.float32)
    for key in ("image", "wrist_image", "state", "actions"):
        expected = tuple(episode["info"]["features"][key]["shape"])
        if sample[key].shape != expected:
            raise ValueError(f"{key} 的 shape 为 {sample[key].shape}，metadata 要求 {expected}")
    return sample


def get_action_chunk(episode: dict, frame_index: int, horizon: int = 10) -> np.ndarray:
    """取同轨迹 actions[i:i+H]；末尾不足时重复最后动作，返回 float32 (H,7)。"""
    if horizon <= 0:
        raise ValueError("horizon 必须是正整数")
    if not 0 <= frame_index < episode["length"]:
        raise IndexError(f"frame {frame_index} 越界；本轨迹可用编号为 0–{episode['length'] - 1}")
    valid_steps = min(horizon, episode["length"] - frame_index)
    # 处理后同一行已是正确监督起点，不再加 1，也不再次差分。
    rows = episode["table"]["actions"].slice(frame_index, valid_steps).to_pylist()
    chunk = np.asarray(rows, dtype=np.float32)
    if valid_steps < horizon:
        padding = np.repeat(chunk[-1:], horizon - valid_steps, axis=0)
        chunk = np.concatenate([chunk, padding], axis=0)
    return chunk


class SpatialDataset:
    """按划分访问所有有效帧，每个样本已完成正式模型所需的数据变换。"""

    def __init__(self, split: str = "train", horizon: int = 10, data_root: str | Path | None = None,
                 *, model_config=None, model_family: str | None = None):
        # 延迟导入，使原始示教预览仍保持轻量。
        from openpi import transforms
        from .model import make_model_config, model_family as config_family, model_spec
        from openpi.training.config import AssetsConfig, LeRobotLiberoDataConfig

        if split not in ("train", "validation") or horizon <= 0:
            raise ValueError("split 须为 train/validation，horizon 须为正整数")
        self.data_root = Path(data_root or os.environ.get("MY_PI05_DATA_ROOT", DEFAULT_DATA_ROOT)).resolve()
        self.split, self.horizon = split, horizon
        self.episode_ids = json.loads((self.data_root / f"manifests/{split}_episode_ids.json").read_text())
        if not self.episode_ids:
            raise ValueError(f"{split} 清单为空；全500训练请使用 --validation-mode none")
        metadata = [json.loads(line) for line in (self.data_root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        lengths = {item["episode_index"]: item["length"] for item in metadata}
        self._ends = np.cumsum([lengths[index] for index in self.episode_ids])
        self._episode = None  # 只缓存最近一条轨迹，不把整个数据集加载到内存。

        # The family selects native OpenPI transforms and training-only normalization.
        self.model_config = model_config or make_model_config(model_family or "pi05", horizon)
        if self.model_config.action_horizon != horizon:
            raise ValueError("dataset horizon 与 model_config 不一致")
        if model_family is not None and config_family(self.model_config) != model_family:
            raise ValueError("dataset model_family 与 model_config 不一致")
        self.model_family = config_family(self.model_config)
        if model_spec(self.model_config) != model_spec(make_model_config(self.model_family, horizon)):
            raise ValueError("dataset 需要本 pipeline 的固定模型输入规格")
        cache = ROOT / "checkpoints/base"
        if Path(os.environ.get("OPENPI_DATA_HOME", "")).resolve() != cache.resolve():
            raise ValueError("请先 source scripts/env.sh，让 tokenizer 使用本项目缓存")
        if not (cache / "big_vision/paligemma_tokenizer.model").is_file():
            raise FileNotFoundError("本项目缺少 paligemma_tokenizer.model")
        self.data_config = LeRobotLiberoDataConfig(
            repo_id="local/libero-spatial-500-v1",
            assets=AssetsConfig(assets_dir=str(self.data_root / "assets"), asset_id="libero-spatial-500-v1"),
            extra_delta_transform=False,
        ).create(self.data_root / "assets", self.model_config)
        if self.data_config.norm_stats is None:
            raise FileNotFoundError("缺少 Spatial-500 的训练集 norm_stats.json")
        self.transform = transforms.compose([
            *self.data_config.repack_transforms.inputs,
            *self.data_config.data_transforms.inputs,
            transforms.Normalize(self.data_config.norm_stats, use_quantiles=self.data_config.use_quantile_norm, strict=True),
            *self.data_config.model_transforms.inputs,
        ])

    def __len__(self) -> int:
        return int(self._ends[-1])

    def locate(self, index: int) -> tuple[int, int]:
        """将划分内的全局样本索引映射成 (episode, frame)。"""
        if not 0 <= index < len(self):
            raise IndexError(f"样本 {index} 越界；{self.split} 共 {len(self)} 帧")
        position = int(np.searchsorted(self._ends, index, side="right"))
        start = int(self._ends[position - 1]) if position else 0
        return self.episode_ids[position], index - start

    def __getitem__(self, index: int) -> dict:
        episode_id, frame_index = self.locate(index)
        if self._episode is None or self._episode["episode_index"] != episode_id:
            self._episode = load_episode(episode_id, self.data_root)
        sample = get_frame(self._episode, frame_index)
        sample["actions"] = get_action_chunk(self._episode, frame_index, self.horizon)
        return self.transform(sample)



class EpochSampler:
    """每个 epoch 轨迹顺序、轨迹内帧顺序分别打乱；每帧恰好一次，可从游标重建。"""

    def __init__(self, dataset: SpatialDataset, seed: int, state: dict | None = None):
        if dataset.split != "train":
            raise ValueError("EpochSampler 只接收训练划分")
        self.dataset, self.seed = dataset, seed
        self.epoch, self.offset = (state["epoch"], state["offset"]) if state else (0, 0)
        if self.epoch < 0 or not 0 <= self.offset < len(dataset):
            raise ValueError("采样 epoch/offset 无效")
        self._order = None

    def _make_order(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        starts = np.concatenate(([0], self.dataset._ends[:-1]))
        return np.concatenate([rng.permutation(np.arange(starts[i], self.dataset._ends[i]))
                               for i in rng.permutation(len(starts))])

    def next_indices(self, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("采样数量必须为正")
        result = []
        while len(result) < count:
            if self._order is None:
                self._order = self._make_order()
            n = min(count - len(result), len(self.dataset) - self.offset)
            result.extend(self._order[self.offset:self.offset + n].tolist())
            self.offset += n
            if self.offset == len(self.dataset):
                self.epoch, self.offset, self._order = self.epoch + 1, 0, None
        return result

    def state_dict(self) -> dict:
        return dict(epoch=self.epoch, offset=self.offset)


def validation_indices(dataset: SpatialDataset, limit: int, seed: int) -> list[int]:
    """0 取完整验证集；小预算按轨迹轮流抽帧，再排序改善读取局部性。"""
    if dataset.split != "validation" or limit < 0:
        raise ValueError("需要验证划分，limit 不能为负")
    if limit == 0 or limit >= len(dataset):
        return list(range(len(dataset)))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    starts = np.concatenate(([0], dataset._ends[:-1]))
    orders = [rng.permutation(np.arange(start, end)) for start, end in zip(starts, dataset._ends)]
    result = []
    for frame in range(max(map(len, orders))):
        for order in orders:
            if frame < len(order):
                result.append(int(order[frame]))
                if len(result) == limit:
                    return sorted(result)
    return sorted(result)


def collate_batch(samples: list[dict]):
    """沿 batch 维堆叠，返回正式模型可接收的 (Observation, actions)。"""
    import jax
    from openpi.models.model import Observation

    if not samples:
        raise ValueError("batch 不能为空")
    batch = jax.tree.map(lambda *values: np.stack(values), *samples)
    # JSON 统计与分词器可能产生 float64/int64，模型输入统一明确 dtype。
    batch["state"] = batch["state"].astype(np.float32)
    batch["actions"] = batch["actions"].astype(np.float32)
    batch["tokenized_prompt"] = batch["tokenized_prompt"].astype(np.int32)
    # 此入口会把 uint8 图像转换成 [-1,1] float32。
    return Observation.from_dict(batch), batch["actions"]


def show_frame(sample: dict, output: str | Path) -> Path:
    """将真实双相机并排保存为一张预览图，适用于没有桌面的服务器。"""
    height, width = sample["image"].shape[:2]
    prompt_lines = textwrap.wrap(sample["prompt"], width=66)
    canvas = Image.new("RGB", (2 * width, height + 80 + 20 * len(prompt_lines)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    title = (f"episode {sample['episode_index']} | frame {sample['frame_index']}/{sample['episode_length'] - 1}"
             f" | {sample['split']} | t={sample['timestamp']:.3f}s")
    draw.text((8, 4), title, fill="black", font=font)
    draw.text((8, 25), "Environment camera", fill="black", font=font)
    draw.text((width + 8, 25), "Wrist camera", fill="black", font=font)
    canvas.paste(Image.fromarray(sample["image"]), (0, 46))
    canvas.paste(Image.fromarray(sample["wrist_image"]), (width, 46))
    for index, line in enumerate(prompt_lines):
        draw.text((8, height + 55 + 20 * index), line, fill="black", font=font)
    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=("pi0", "pi05"), default="pi05")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10, help="未来动作步数，默认 10")
    parser.add_argument("--batch-size", type=int, help="给定后查看模型 batch；默认仍查看单帧")
    parser.add_argument("--split", choices=("train", "validation"), default="train", help="batch 所用划分")
    parser.add_argument("--output", type=Path, help="预览 PNG 的保存位置，默认在 outputs/previews/ 下")
    args = parser.parse_args()
    try:
        if args.batch_size is not None:
            import jax

            dataset = SpatialDataset(args.split, args.horizon, model_family=args.model_family)
            if not 0 < args.batch_size <= len(dataset):
                raise ValueError(f"batch-size 须在 1–{len(dataset)} 之间")
            observation, actions = collate_batch([dataset[index] for index in range(args.batch_size)])
            arrays = {**observation.to_dict(), "actions": actions}
            shapes = jax.tree.map(lambda x: {"shape": list(x.shape), "dtype": str(x.dtype),
                                            "min": float(np.min(x)), "max": float(np.max(x))}, arrays)
            print(json.dumps({"split": args.split, "episodes": len(dataset.episode_ids),
                              "frames": len(dataset), "source": [dataset.locate(i) for i in range(args.batch_size)],
                              "model_family": dataset.model_family,
                              "quantile_normalization": dataset.data_config.use_quantile_norm,
                              "discrete_state_input": dataset.model_config.discrete_state_input,
                              "image_masks": {k: v.tolist() for k, v in observation.image_masks.items()},
                              "prompt_token_counts": observation.tokenized_prompt_mask.sum(axis=1).tolist(),
                              "batch": shapes}, ensure_ascii=False, indent=2))
            return
        episode = load_episode(args.episode)
        sample = get_frame(episode, args.frame)
        chunk = get_action_chunk(episode, args.frame, args.horizon)
        output = args.output or OUTPUTS / f"previews/episode_{args.episode:06d}_frame_{args.frame:06d}.png"
        image_path = show_frame(sample, output)
    except (OSError, ValueError, IndexError, KeyError) as error:
        parser.error(str(error))

    summary = {key: sample[key] for key in (
        "episode_index", "frame_index", "episode_length", "split", "task_index", "timestamp", "fps", "prompt",
    )}
    for key in ("image", "wrist_image", "state", "actions"):
        summary[key] = {"shape": list(sample[key].shape), "dtype": str(sample[key].dtype)}
        if key in ("state", "actions"):
            summary[key]["values"] = sample[key].tolist()
    valid_steps = min(args.horizon, episode["length"] - args.frame)
    summary["action_chunk"] = {
        "shape": list(chunk.shape), "dtype": str(chunk.dtype),
        "source_frames": list(range(args.frame, args.frame + valid_steps))
                         + [episode["length"] - 1] * (args.horizon - valid_steps),
        "valid_steps": valid_steps, "padded_steps": args.horizon - valid_steps,
        "control_duration_seconds": args.horizon / episode["fps"],
        "first_action": chunk[0].tolist(), "last_action": chunk[-1].tolist(),
    }
    summary["preview"] = str(image_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cli():
    """命令行只执行任务，不把模型状态作为进程退出值。"""
    main()


if __name__ == "__main__":
    cli()
