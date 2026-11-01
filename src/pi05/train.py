"""Spatial-500 微调入口：完整采样、梯度累积、验证、日志与可恢复训练。"""

import argparse
import csv
import dataclasses
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

from flax import nnx, struct
import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import optax

from .checkpoints import (
    checkpoint_metadata, checkpoint_path, optimizer_counters, restore_checkpoint, save_checkpoint,
)
from . import ROOT, RUNS
from .dataset import EpochSampler, SpatialDataset, collate_batch, validation_indices
from .model import (
    LORA_FILTER, VISION_FILTER, PROJECTION_FILTER, EXPERT_FILTER, LLM_FILTER, SCOPES,
    adaptation_metadata,
    get_trainable_filter, init_lora_model, lora_config, family_metadata, normalize_config, _checkpoint_path,
)
from openpi.shared import nnx_utils
from openpi.training.optimizer import AdamW, CosineDecaySchedule

# 训练状态与单步更新：沿用阶段 5–7 的梯度累积公式。
@struct.dataclass
class TrainState:
    step: jax.Array
    trainable: nnx.State
    frozen: nnx.State
    opt_state: optax.OptState
    model_def: nnx.GraphDef
    tx: optax.GradientTransformation = struct.field(pytree_node=False)


def create_train_state(model_config, rng, learning_rate: float = 5e-5, scope: str = "strict", *, mesh=None) -> TrainState:
    """按 scope 初始化精度和参数范围，优化器只接收选中的 FP32 参数。"""
    return _state_from_model(init_lora_model(model_config, rng, scope=scope, mesh=mesh), model_config, learning_rate, scope)


def _state_from_model(model, model_config, learning_rate, scope):
    model.train()
    graph, trainable, frozen = nnx.split(model, get_trainable_filter(model_config, scope), ...)
    frozen = nnx_utils.state_map(frozen, nnx.Param, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
    if not jax.tree.leaves(trainable):
        raise ValueError("没有选中可训练参数")
    if any(x.dtype != jnp.float32 for x in jax.tree.leaves(trainable)):
        raise ValueError("选中的可训练参数必须保留 FP32")
    tx = AdamW(clip_gradient_norm=1.0).create(learning_rate)
    return TrainState(step=jnp.asarray(0, dtype=jnp.int32), trainable=trainable, frozen=frozen,
                      opt_state=tx.init(trainable), model_def=graph, tx=tx)


def microbatch_rngs(base_rng, step, count: int):
    """每个更新先 fold_in(step)，再为 microbatch 分配独立 key；K=1 保持阶段5语义。"""
    step_rng = jax.random.fold_in(base_rng, step)
    return step_rng[None] if count == 1 else jax.random.split(step_rng, count)


@jax.jit
def compute_gradients(state: TrainState, rng, batch):
    observation, actions = batch

    def loss_fn(trainable):
        model = nnx.merge(state.model_def, trainable, state.frozen)
        model.train()
        return jnp.mean(model.compute_loss(rng, observation, actions, train=True))

    return jax.value_and_grad(loss_fn)(state.trainable)


@jax.jit
def accumulate_gradients(state: TrainState, rngs, micro_batches):
    """scan 逐个计算并累加梯度，不保留全部 microbatch 的激活，也不更新优化器。"""
    def body(total, inputs):
        rng, batch = inputs
        loss, grads = compute_gradients(state, rng, batch)
        return jax.tree.map(jnp.add, total, grads), loss

    zero = jax.tree.map(jnp.zeros_like, state.trainable)
    total, losses = jax.lax.scan(body, zero, (rngs, micro_batches))
    mean_grads = jax.tree.map(lambda x: x / rngs.shape[0], total)
    return jnp.mean(losses), mean_grads


@jax.jit
def apply_gradients(state: TrainState, loss, grads):
    """传入平均后的原始梯度，在这里统一裁剪并执行一次 AdamW。"""
    updates, opt_state = state.tx.update(grads, state.opt_state, state.trainable)
    trainable = optax.apply_updates(state.trainable, updates)
    delta = jax.tree.map(lambda new, old: new - old, trainable, state.trainable)
    finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves((loss, grads, trainable, opt_state))]))
    metrics = {"loss_before": loss, "grad_norm": optax.global_norm(grads),
               "trainable_delta_norm": optax.global_norm(delta),
               "lora_delta_norm": optax.global_norm(delta.filter(LORA_FILTER)),
               "vision_delta_norm": optax.global_norm(delta.filter(VISION_FILTER)),
               "expert_delta_norm": optax.global_norm(delta.filter(EXPERT_FILTER)),
               "llm_delta_norm": optax.global_norm(delta.filter(LLM_FILTER)),
               "projection_delta_norm": optax.global_norm(delta.filter(PROJECTION_FILTER)), "finite": finite}
    return state.replace(step=state.step + 1, trainable=trainable, opt_state=opt_state), metrics


@jax.jit
def accumulated_train_step(state: TrainState, base_rng, micro_batches):
    count = micro_batches[1].shape[0]
    if count < 1:
        raise ValueError("至少需要一个 microbatch")
    rngs = microbatch_rngs(base_rng, state.step, count)
    loss, grads = accumulate_gradients(state, rngs, micro_batches)
    return apply_gradients(state, loss, grads)


def make_parallel_train_step(mesh):
    """分片逻辑microbatch轴；本地平均后跨卡平均，再统一clip/Adam。"""
    replicated = NamedSharding(mesh, P())
    split_microbatches = NamedSharding(mesh, P("data"))

    def local_gradients(state, rngs, batches):
        loss, grads = accumulate_gradients(state, rngs, batches)
        return jax.lax.pmean(loss, "data"), jax.tree.map(lambda x: jax.lax.pmean(x, "data"), grads)

    gradients = shard_map(local_gradients, mesh=mesh, in_specs=(P(), P("data"), P("data")),
                          out_specs=(P(), P()), check_rep=False)

    def step(state, base_rng, micro_batches):
        # key按全局逻辑microbatch编号生成，不依赖设备编号。
        rngs = microbatch_rngs(base_rng, state.step, micro_batches[1].shape[0])
        loss, grads = gradients(state, rngs, micro_batches)
        return apply_gradients(state, loss, grads)

    return jax.jit(step, in_shardings=(replicated, replicated, split_microbatches),
                   out_shardings=(replicated, replicated))


def make_fsdp_train_step(mesh, state_shardings):
    """E: shard weights/Adam across GPUs; average the same logical microbatch losses.

    Group consecutive logical microbatches across devices, then accumulate groups.
    Each microbatch retains its own RNG and batch size; one global clip/Adam follows.
    """
    replicated = NamedSharding(mesh, P())
    batches_sharding = NamedSharding(mesh, P(None, "fsdp"))
    width = mesh.size

    def step(state, base_rng, groups):
        rngs = microbatch_rngs(base_rng, state.step, groups[1].shape[0] * width)
        rngs = rngs.reshape((groups[1].shape[0], width))

        def group_gradients(keys, batches):
            def loss_fn(trainable):
                def single(key, batch):
                    model = nnx.merge(state.model_def, trainable, state.frozen)
                    model.train()
                    return jnp.mean(model.compute_loss(key, *batch, train=True))
                return jnp.mean(jax.vmap(single)(keys, batches))
            return jax.value_and_grad(loss_fn)(state.trainable)

        def body(total, inputs):
            loss, grads = group_gradients(*inputs)
            return jax.tree.map(jnp.add, total, grads), loss

        zero = jax.tree.map(jnp.zeros_like, state.trainable)
        total, losses = jax.lax.scan(body, zero, (rngs, groups))
        grads = jax.tree.map(lambda x: x / groups[1].shape[0], total)
        return apply_gradients(state, jnp.mean(losses), grads)

    return jax.jit(step, in_shardings=(state_shardings, replicated, batches_sharding),
                   out_shardings=(state_shardings, replicated))


@jax.jit
def train_step(state: TrainState, base_rng, batch):
    """保留阶段5单 batch 接口，等价于 K=1。"""
    return accumulated_train_step(state, base_rng, jax.tree.map(lambda x: x[None], batch))


@jax.jit
def loss_on_batch(state: TrainState, rng, batch):
    """诊断时显式传入同一个 key，固定图像增强、噪声和 Flow 时间。"""
    model = nnx.merge(state.model_def, state.trainable, state.frozen)
    model.train()
    return jnp.mean(model.compute_loss(rng, *batch, train=True))


@jax.jit
def loss_on_microbatches(state: TrainState, rngs, micro_batches):
    def body(carry, inputs):
        rng, batch = inputs
        return carry, loss_on_batch(state, rng, batch)
    _, losses = jax.lax.scan(body, None, (rngs, micro_batches))
    return jnp.mean(losses)


@jax.jit
def frozen_equal(before, after):
    """在设备上逐元素比较冻结状态，仅返回一个 bool，不另存参数或报告。"""
    comparisons = jax.tree.map(jnp.array_equal, before, after)
    leaves = jax.tree.leaves(comparisons)
    return jnp.all(jnp.stack(leaves)) if leaves else jnp.asarray(True)


# 学习率与独立验证。
def make_schedule(config):
    if config["schedule"] == "constant":
        return config["lr"]
    return CosineDecaySchedule(warmup_steps=config["warmup_steps"], peak_lr=config["lr"],
                               decay_steps=config["total_steps"], decay_lr=config["end_lr"]).create()


@jax.jit
def validation_losses(state, rng, batch):
    model = nnx.merge(state.model_def, state.trainable, state.frozen)
    model.eval()
    return jnp.mean(model.compute_loss(rng, *batch, train=False), axis=-1)


def evaluate(state, dataset, config):
    """验证关闭增强，固定样本与噪声；按真实样本数加权，不计最后一批 padding。"""
    batch_size = config["batch_size"]
    indices = validation_indices(dataset, config["val_batches"] * batch_size, config["seed"])
    rng = jax.random.fold_in(jax.random.key(config["seed"]), 104729)
    total = 0.0
    for i in range(0, len(indices), batch_size):
        chosen = indices[i:i + batch_size]
        padded = chosen + [chosen[-1]] * (batch_size - len(chosen))
        batch = jax.tree.map(jnp.asarray, collate_batch([dataset[j] for j in padded]))
        losses = np.asarray(validation_losses(state, jax.random.fold_in(rng, i // batch_size), batch))
        if not np.isfinite(losses).all():
            raise ValueError("验证 loss 出现非有限值")
        total += float(np.sum(losses[:len(chosen)], dtype=np.float64))
    return dict(val_loss=total / len(indices), val_samples=len(indices),
                val_episodes=len({dataset.locate(i)[0] for i in indices}),
                validation_full=len(indices) == len(dataset))


def training_config(args, dataset):
    config = dict(args)
    files = ["conversion.json", "manifests/train_episode_ids.json",
             "assets/libero-spatial-500-v1/norm_stats.json"]
    def digest(names):
        h = hashlib.sha256()
        for name in names:
            h.update(name.encode())
            h.update((dataset.data_root / name).read_bytes())
        return h.hexdigest()
    config.update(data_root=str(dataset.data_root.relative_to(ROOT)), data_signature=digest(files),
                  validation_signature=digest(["manifests/validation_episode_ids.json", "meta/episodes.jsonl", "meta/tasks.jsonl"]),
                  openpi_commit=subprocess.check_output(
                      ["git", "-C", str(ROOT / "external/openpi"), "rev-parse", "HEAD"], text=True).strip(),
                  optimizer=dataclasses.asdict(AdamW(clip_gradient_norm=1.0)))
    config.update(family_metadata(dataset.model_config))
    if config.get("scope") in ("action_lora", "expert_full", "full"):
        config["adaptation"] = adaptation_metadata(dataset.model_config, config["scope"])
    # Resume keeps its original weight provenance; restoration never reloads base arrays.
    config.setdefault("base_checkpoint", str(_checkpoint_path(family=dataset.model_family, require_exists=False).relative_to(ROOT)))
    config.setdefault("base_checkpoint_source", f"gs://openpi-assets/checkpoints/{dataset.model_family}_base")
    _checkpoint_path(ROOT / config["base_checkpoint"], dataset.model_family, require_exists=False)
    return config


# 每次运行的配置、日志与曲线。
def write_event(run_dir, event, step, **values):
    record = dict(event=event, step=int(step), time=datetime.now().astimezone().isoformat(), **values)
    with (run_dir / "metrics.jsonl").open("a") as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def plot_history(run_dir):
    """保留原始逐步 CSV；EMA 只用于显示，完整验证不混入周期验证曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    training, validation, restart_steps = {}, {}, set()
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        step = row["step"]
        if row["event"] == "resume":
            training = {k: v for k, v in training.items() if k <= step}
            validation = {k: v for k, v in validation.items() if k <= step}
            restart_steps.add(step + 1)
        elif row["event"] == "start":
            restart_steps.add(step + 1)
        elif row["event"] == "train":
            training[step] = row
        elif row["event"] == "validation":
            validation[step] = row
    keys = sorted(training)
    fields = ["step", "time", "train_loss", "grad_norm", "lr", "trainable_delta_norm", "lora_delta_norm",
              "llm_delta_norm", "expert_delta_norm", "vision_delta_norm", "projection_delta_norm",
              "data_seconds", "update_seconds", "iteration_seconds", "steps_per_second", "samples_per_second",
              "device_count", "global_batch", "epoch_equivalent", "includes_compile"]
    with (run_dir / "training.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for step in keys:
            row = dict(training[step])
            seconds = row["data_seconds"] + row["update_seconds"]
            row.update(iteration_seconds=seconds, steps_per_second=1 / seconds,
                       includes_compile=step in restart_steps)
            writer.writerow(row)

    def ema(values):
        result = []
        for value in values:
            result.append(value if not result else 0.1 * value + 0.9 * result[-1])
        return result

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    loss_ax, grad_ax, lr_ax, speed_ax = axes.flat
    for axis, field, label, color in ((loss_ax, "train_loss", "train loss", "tab:blue"),
                                       (grad_ax, "grad_norm", "gradient norm (before clip)", "tab:orange")):
        values = [training[k][field] for k in keys]
        axis.plot(keys, values, color=color, alpha=0.25, linewidth=0.7, label=label + " raw")
        axis.plot(keys, ema(values), color=color, linewidth=1.5, label="EMA alpha=0.1")
        axis.legend(fontsize=8)
    val_keys = sorted(validation)
    if val_keys:
        loss_ax.plot(val_keys, [validation[k]["val_loss"] for k in val_keys], color="tab:green",
                     marker=".", label="validation: fixed subset/RNG")
    loss_ax.set_title("Training and sampled validation loss" if val_keys else "Training loss (no held-out validation)")
    loss_ax.set_ylabel("Flow loss")
    loss_ax.legend(fontsize=8)
    grad_ax.set_title("Gradient norm (clip threshold = 1)")
    grad_ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.7)
    lr_ax.plot(keys, [training[k]["lr"] for k in keys], color="tab:green")
    lr_ax.set_title("Learning rate schedule")
    lr_ax.set_ylabel("Learning rate")
    timed = [k for k in keys if k not in restart_steps]
    seconds = [training[k]["data_seconds"] + training[k]["update_seconds"] for k in timed]
    rates = [1 / value for value in seconds]
    speed_ax.plot(timed, rates, color="tab:red", alpha=0.2, linewidth=0.7)
    speed_ax.plot(timed, ema(rates), color="tab:red", label="steps/s EMA")
    speed_ax.set_ylabel("Optimizer steps/s", color="tab:red")
    other = speed_ax.twinx()
    other.plot(timed, [training[k]["update_seconds"] for k in timed], color="gray", alpha=0.4, linewidth=0.7)
    other.set_ylabel("Update seconds", color="gray")
    speed_ax.set_title("Throughput: data + update; compile steps excluded")
    for axis in axes.flat:
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.3)
    config = json.loads((run_dir / "config.json").read_text())
    fig.suptitle(f"{run_dir.name}\n{config.get('model_family', 'pi05')} | scope={config['scope']} | "
                 f"global batch={config['batch_size'] * config['accum_steps']} | seed={config['seed']} | "
                 f"peak LR={config['lr']} | updates={max(keys, default=0)}", fontsize=10)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    fig.text(0.02, 0.012, "CSV keeps every raw update, including compile steps. Timing excludes validation/checkpoint overhead.", fontsize=8)
    fig.savefig(run_dir / "curves.png", dpi=150)
    fig.savefig(run_dir / "curves.svg")
    plt.close(fig)


# 命令行配置与训练主循环。
def parse_config():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train.json")
    parser.add_argument("--devices", type=int, choices=(1, 2, 4, 8), default=1,
                        help="本次参与的数据并行卡数；逻辑accum-steps须能整除卡数")
    parser.add_argument("--model-family", choices=("pi0", "pi05"))
    parser.add_argument("--data-root", type=Path, help="显式选择项目内数据版本；恢复默认读取checkpoint记录")
    parser.add_argument("--validation-mode", choices=("heldout", "none"), help="train500使用none，旧配置默认heldout")
    parser.add_argument("--validate-only", action="store_true", help="恢复checkpoint并对全部验证帧写final_validation_full事件")
    parser.add_argument("--steps", type=int, default=10, help="本次额外更新次数，默认短跑10步")
    parser.add_argument("--run-dir", "--checkpoint-dir", dest="run_dir", type=Path)
    parser.add_argument("--resume", type=Path)
    kinds = dict(scope=str, sampling=str, index=int, batch_size=int, accum_steps=int, horizon=int,
                 lr=float, schedule=str, warmup_steps=int, total_steps=int, end_lr=float,
                 seed=int, save_every=int, eval_every=int, val_batches=int)
    for key, kind in kinds.items():
        parser.add_argument("--" + key.replace("_", "-"), type=kind)
    args = parser.parse_args()
    try:
        profile = json.loads(args.config.read_text())
        if set(profile) - {"model_family", "validation_mode", "data_root"} != set(kinds):
            raise ValueError("配置字段与 configs/train.json 的规范不一致")
        metadata = checkpoint_metadata(args.resume) if args.resume else None
        config = dict(profile)
        config["model_family"] = profile.get("model_family", "pi05")
        if metadata:
            config.update({k: v for k, v in metadata["config"].items() if k in kinds})
            if metadata["format_version"] == 1:
                config.update(sampling="fixed", schedule="constant")
        if metadata:
            config["model_family"] = normalize_config(metadata["config"])["model_family"]
            for key in ("validation_mode", "data_root"):
                if key in metadata["config"]:
                    config[key] = metadata["config"][key]
                else:
                    config.pop(key, None)
        if args.validation_mode is not None:
            if metadata and args.validation_mode != metadata["config"].get("validation_mode", "heldout"):
                raise ValueError("--validation-mode 与checkpoint不一致")
            config["validation_mode"] = args.validation_mode
        if config.get("validation_mode", "heldout") not in ("heldout", "none"):
            raise ValueError("validation_mode 须为 heldout/none")
        if args.validate_only and config.get("validation_mode") == "none":
            raise ValueError("此实验没有留出验证集，不能 --validate-only")
        if args.data_root is not None:
            requested = (ROOT / args.data_root).resolve()
            if metadata and requested != (ROOT / metadata["config"]["data_root"]).resolve():
                raise ValueError("--data-root 与checkpoint不一致")
            config["data_root"] = str(requested.relative_to(ROOT))
        if "data_root" in config:
            config["data_root"] = str((ROOT / config["data_root"]).resolve().relative_to(ROOT))
        if args.model_family is not None:
            if metadata and args.model_family != config["model_family"]:
                raise ValueError("--model-family 与 checkpoint 不一致")
            config["model_family"] = args.model_family
        normalize_config(config)
        if args.validate_only and not metadata:
            raise ValueError("--validate-only 需要 --resume")
        for name in kinds:
            value = getattr(args, name)
            if metadata and value is not None and name in metadata["config"] and value != config[name]:
                raise ValueError(f"--{name.replace('_', '-')} 与 checkpoint 不一致")
            if value is not None:
                config[name] = value
        if metadata and metadata["format_version"] == 1 and (config["sampling"] != "fixed" or config["schedule"] != "constant"):
            raise ValueError("阶段7 checkpoint 继续沿用 fixed 采样和 constant 学习率")
        if config["scope"] not in SCOPES or config["sampling"] not in ("episode_shuffle", "fixed"):
            raise ValueError("scope 或 sampling 无效")
        if config["schedule"] not in ("constant", "cosine"):
            raise ValueError("schedule 须为 constant/cosine")
        if any(config[k] <= 0 for k in ("batch_size", "accum_steps", "horizon", "total_steps", "save_every", "eval_every")):
            raise ValueError("batch、horizon、训练/保存/验证步数须为正")
        if not np.isfinite(config["lr"]) or config["lr"] <= 0 or not 0 <= config["end_lr"] <= config["lr"]:
            raise ValueError("学习率无效")
        if config["seed"] < 0 or config["index"] < 0 or config["val_batches"] < 0 or config["warmup_steps"] < 0:
            raise ValueError("seed/index/val-batches/warmup-steps 不能为负")
        if config["schedule"] == "cosine" and not 0 < config["warmup_steps"] < config["total_steps"]:
            raise ValueError("cosine 需要 0 < warmup-steps < total-steps")
        if config["accum_steps"] % args.devices:
            raise ValueError("全局逻辑accum-steps必须能整除devices")
        step = metadata["step"] if metadata else 0
        if not args.validate_only and (args.steps <= 0 or step + args.steps > config["total_steps"]):
            raise ValueError("本次 steps 须为正且不能超过配置的 total-steps 总预算")
        run_dir = RUNS / (datetime.now().strftime("train-%Y%m%d-%H%M%S-%f-") + config["scope"])
        if args.run_dir:
            run_dir = checkpoint_path(args.run_dir)
        elif args.resume:
            source = Path(args.resume).resolve()
            previous_run = source.parent.parent if source.parent.name == "checkpoints" else source.parent
            if previous_run.is_relative_to(RUNS):
                run_dir = checkpoint_path(previous_run)
        if run_dir.exists() and any(run_dir.iterdir()) and not metadata:
            raise ValueError("新训练 run-dir 非空，请选新目录或 --resume")
        for target in range(step + 1, step + (0 if args.validate_only else args.steps) + 1):
            if target % config["save_every"] == 0 or target == step + args.steps:
                if (run_dir / "checkpoints" / f"step_{target:06d}").exists():
                    raise ValueError("目标 checkpoint 已存在；从旧 step 分支时请指定新的 --run-dir")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    return args, config, metadata, run_dir


def main():
    args, options, metadata, run_dir = parse_config()
    devices = jax.devices()
    if len(devices) != args.devices or any(d.platform != "gpu" for d in devices):
        raise ValueError("先核对占用，显式选择对应数量的授权GPU，设置JAX_PLATFORMS=cuda和--devices")
    fsdp = options["scope"] == "full" and args.devices > 1
    mesh = Mesh(np.asarray(devices), ("fsdp" if fsdp else "data",))
    replicated = (jax.sharding.SingleDeviceSharding(devices[0]) if args.devices == 1
                  else NamedSharding(mesh, P()))
    batch_sharding = (NamedSharding(mesh, P(None, "fsdp")) if fsdp else
                      replicated if args.devices == 1 else NamedSharding(mesh, P("data")))
    update_step = None if fsdp else accumulated_train_step if args.devices == 1 else make_parallel_train_step(mesh)
    dataset = SpatialDataset("train", options["horizon"],
                             ROOT / options["data_root"] if "data_root" in options else None,
                             model_family=options["model_family"])
    validation = (None if options.get("validation_mode") == "none" else
                  SpatialDataset("validation", options["horizon"], dataset.data_root, model_config=dataset.model_config))
    validation_ids = json.loads((dataset.data_root / "manifests/validation_episode_ids.json").read_text())
    if validation is None and validation_ids:
        raise ValueError("validation-mode=none 要求数据版本的验证清单为空，避免误用450训练版本")
    config = training_config(options, dataset)
    if metadata:
        previous = normalize_config(metadata["config"])
        for key in ("base_checkpoint", "base_checkpoint_source"):
            if key in previous:
                config[key] = previous[key]
        if any(config.get(k) != value for k, value in previous.items()):
            raise ValueError("当前配置、数据签名或正式源码与 checkpoint 不一致")
    effective_batch = config["batch_size"] * config["accum_steps"]
    if config["sampling"] == "fixed" and config["index"] + effective_batch > len(dataset):
        raise ValueError("固定 batch 越过训练集末尾")
    schedule = make_schedule(config)
    train_rng, init_rng = jax.random.split(jax.random.key(config["seed"]))
    state_shardings = None
    if metadata or fsdp:
        abstract = jax.eval_shape(lambda key: _state_from_model(lora_config(dataset.model_config, config["scope"]).create(key),
                                 dataset.model_config, schedule, config["scope"]), init_rng)
    if fsdp:
        from openpi.training.sharding import fsdp_sharding
        state_shardings = fsdp_sharding(abstract, mesh)
    if metadata:
        state, train_rng, cursor = restore_checkpoint(args.resume, abstract, sharding=replicated, state_shardings=state_shardings)
    else:
        state = create_train_state(dataset.model_config, init_rng, schedule, config["scope"], mesh=mesh if fsdp else None)
        cursor = None
    if fsdp:
        # Preserve the actual static optimizer/graph objects in the sharding tree.
        state_shardings = fsdp_sharding(state, mesh)
        update_step = make_fsdp_train_step(mesh, state_shardings)
    state = jax.device_put(state, state_shardings if fsdp else replicated)
    train_rng = jax.device_put(train_rng, replicated)
    jax.block_until_ready(state)
    sampler = EpochSampler(dataset, config["seed"], cursor) if config["sampling"] == "episode_shuffle" else None
    if sampler and sampler.epoch * len(dataset) + sampler.offset != int(state.step) * effective_batch:
        raise ValueError("采样游标与已完成更新步数不一致")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if config_path.exists():
        existing = normalize_config(json.loads(config_path.read_text()))
        for key in ("base_checkpoint", "base_checkpoint_source"):
            existing.setdefault(key, config[key])
    if config_path.exists() and existing != config:
        raise ValueError("run-dir 中已有不同的配置，请指定新的目录")
    if not config_path.exists():
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    if args.validate_only:
        started = time.perf_counter()
        evaluation_state = state if args.devices == 1 or fsdp else jax.tree.map(lambda x: x.addressable_shards[0].data, state)
        values = evaluate(evaluation_state, validation, {**config, "val_batches": 0})
        write_event(run_dir, "final_validation_full", state.step, **values, seconds=time.perf_counter() - started)
        return state
    count = lambda tree: sum(x.size for x in jax.tree.leaves(tree))
    parameter_counts = dict(trainable=count(state.trainable), frozen=count(state.frozen),
                            trainable_lora=count(state.trainable.filter(LORA_FILTER)),
                            trainable_llm=count(state.trainable.filter(LLM_FILTER)),
                            trainable_expert=count(state.trainable.filter(EXPERT_FILTER)),
                            trainable_vision=count(state.trainable.filter(VISION_FILTER)),
                            trainable_projection=count(state.trainable.filter(PROJECTION_FILTER)))
    write_event(run_dir, "resume" if metadata else "start", state.step, config=config,
                train_frames=len(dataset), train_episodes=len(dataset.episode_ids),
                validation_frames=0 if validation is None else len(validation),
                device_count=args.devices, global_batch=effective_batch, per_device_microbatch=config["batch_size"],
                local_accum_steps=config["accum_steps"] // args.devices, parameter_counts=parameter_counts,
                adaptation=adaptation_metadata(dataset.model_config, config["scope"]),
                parallelism="fsdp" if fsdp else "data_parallel")
    parameter_manifest = run_dir / "parameters.json"
    if not parameter_manifest.exists():
        rows = []
        for trainable, tree in ((True, state.trainable), (False, state.frozen)):
            for path, value in tree.flat_state().items():
                rows.append(dict(path="/".join(map(str, path)), trainable=trainable,
                                 shape=list(value.value.shape), dtype=str(value.value.dtype), count=value.value.size))
        parameter_manifest.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    initial_frozen = state.frozen
    def record_validation():
        if validation is None:
            plot_history(run_dir)
            return
        started = time.perf_counter()
        # 验证在第一张卡上复用本地副本，不重复计算同一批验证样本。
        evaluation_state = (state if args.devices == 1 or fsdp else
                            jax.tree.map(lambda x: x.addressable_shards[0].data, state))
        values = evaluate(evaluation_state, validation, config)
        write_event(run_dir, "validation", state.step, **values, seconds=time.perf_counter() - started)
        plot_history(run_dir)
    record_validation()
    for iteration in range(args.steps):
        started = time.perf_counter()
        indices = sampler.next_indices(effective_batch) if sampler else list(range(config["index"], config["index"] + effective_batch))
        size = config["batch_size"]
        batches = [collate_batch([dataset[i] for i in indices[j:j + size]]) for j in range(0, len(indices), size)]
        micro_batches = jax.tree.map(lambda *values: jnp.stack(values), *batches)
        if fsdp:
            micro_batches = jax.tree.map(lambda x: x.reshape((-1, args.devices) + x.shape[1:]), micro_batches)
        if args.devices > 1:
            micro_batches = jax.device_put(micro_batches, batch_sharding)
        jax.block_until_ready(micro_batches)
        data_seconds = time.perf_counter() - started
        lr = float(schedule(state.step)) if callable(schedule) else schedule
        started = time.perf_counter()
        state, metrics = jax.block_until_ready(update_step(state, train_rng, micro_batches))
        update_seconds = time.perf_counter() - started
        counters = optimizer_counters(state)
        if not bool(metrics["finite"]) or not counters or any(c != int(state.step) for c in counters):
            raise ValueError("训练出现非有限值或优化器计数不一致")
        if float(metrics["grad_norm"]) <= 0 or float(metrics["trainable_delta_norm"]) <= 0:
            raise ValueError("没有发生有效梯度/参数更新")
        values = {k: float(v) for k, v in metrics.items() if k not in ("finite", "loss_before")}
        write_event(run_dir, "train", state.step, train_loss=float(metrics["loss_before"]), lr=lr,
                    **values, optimizer_counters=counters, sample_indices=indices,
                    source=[dataset.locate(i) for i in indices], sampler=sampler.state_dict() if sampler else None,
                    data_seconds=data_seconds, update_seconds=update_seconds,
                    epoch_equivalent=int(state.step) * effective_batch / len(dataset),
                    device_count=args.devices, global_batch=effective_batch,
                    samples_per_second=effective_batch / (data_seconds + update_seconds))
        last = iteration == args.steps - 1
        if int(state.step) % config["eval_every"] == 0 or last:
            record_validation()
        if int(state.step) % config["save_every"] == 0 or last:
            if not bool(frozen_equal(initial_frozen, state.frozen)):
                raise ValueError("冻结参数发生变化")
            started = time.perf_counter()
            save_checkpoint(run_dir / "checkpoints" / f"step_{int(state.step):06d}", state, train_rng, config,
                            sampler.state_dict() if sampler else None)
            write_event(run_dir, "checkpoint", state.step, seconds=time.perf_counter() - started)
    if int(state.step) == config["total_steps"] and validation is not None:
        started = time.perf_counter()
        evaluation_state = state if args.devices == 1 or fsdp else jax.tree.map(lambda x: x.addressable_shards[0].data, state)
        values = evaluate(evaluation_state, validation, {**config, "val_batches": 0})
        write_event(run_dir, "final_validation_full", state.step, **values, seconds=time.perf_counter() - started)
    plot_history(run_dir)
    return state


def cli():
    """命令行只执行任务，不把模型状态作为进程退出值。"""
    main()


if __name__ == "__main__":
    cli()
