"""正式模型入口：基础权重、Flow 前向与按模块选择的微调范围。"""

import argparse
import dataclasses
import json
import os
from pathlib import Path
import time

from flax import nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from . import ROOT
from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing, nnx_utils
from openpi.training.weight_loaders import CheckpointWeightLoader


# 同时覆盖 Attention 的 lora_a/b 和 MLP 的 gating_einsum_lora_a/b 等。
LORA_FILTER = nnx.All(nnx.Param, nnx_utils.PathRegex(".*lora_[ab]"))
VISION_FILTER = nnx.All(nnx.Param, nnx_utils.PathRegex("PaliGemma/img/.*"))
PROJECTION_FILTER = nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex("PaliGemma/.*")))
EXPERT_FILTER = nnx.All(nnx.Param, nnx_utils.PathRegex("PaliGemma/llm/.*_1.*"))
LLM_FILTER = nnx.All(nnx.Param, nnx_utils.PathRegex("PaliGemma/llm/.*"), nnx.Not(EXPERT_FILTER))
SCOPES = ("strict", "default", "action_lora", "expert_full", "full")


def lora_config(model_config, scope="strict"):
    """旧范围保留双 LoRA；A 仅动作 LoRA，D 仅 LLM LoRA，E 全量无 LoRA。"""
    if scope not in SCOPES:
        raise ValueError(f"scope 须为 {SCOPES}")
    return dataclasses.replace(
        model_config,
        paligemma_variant="gemma_2b" if scope in ("action_lora", "full") else "gemma_2b_lora",
        action_expert_variant="gemma_300m" if scope in ("expert_full", "full") else "gemma_300m_lora",
    )


def adaptation_metadata(model_config, scope):
    from openpi.models.gemma import get_config

    config = lora_config(model_config, scope)
    return dict(
        paligemma_variant=config.paligemma_variant, action_expert_variant=config.action_expert_variant,
        lora={name: {key: dict(rank=value.rank, alpha=value.alpha) for key, value in get_config(variant).lora_configs.items()}
              for name, variant in (("llm", config.paligemma_variant), ("action_expert", config.action_expert_variant))},
        trainable_dtype="float32", frozen_dtype="bfloat16", compute_dtype=config.dtype,
    )


def get_trainable_filter(model_config, scope: str = "strict"):
    """A 只选动作专家适配器；D 用官方单 LLM LoRA 的冻结规则。"""
    if scope == "full":
        return nnx.Param
    if scope == "strict":
        return LORA_FILTER
    if scope == "action_lora":
        return nnx.All(LORA_FILTER, EXPERT_FILTER)
    if scope in ("default", "expert_full"):
        return nnx.All(nnx.Param, nnx.Not(lora_config(model_config, scope).get_freeze_filter()))
    raise ValueError(f"scope 须为 {SCOPES}")


def model_family(config=None):
    """Old checkpoints have no family field and always describe pi05."""
    family = (config.get("model_family", "pi05") if isinstance(config, dict)
              else "pi05" if config is None or config.pi05 else "pi0")
    if family not in ("pi0", "pi05"):
        raise ValueError("model_family 须为 pi0/pi05")
    return family


def make_model_config(family="pi05", horizon=10):
    from openpi.models.pi0_config import Pi0Config

    if family not in ("pi0", "pi05") or horizon <= 0:
        raise ValueError("model_family 须为 pi0/pi05，horizon 须为正整数")
    # Preserve the established pi05 recipe: no discrete state tokens.
    return Pi0Config(pi05=family == "pi05", action_horizon=horizon, discrete_state_input=False)


def model_spec(config):
    return dict(pi05=config.pi05, action_dim=config.action_dim, action_horizon=config.action_horizon,
                max_token_len=config.max_token_len, discrete_state_input=config.discrete_state_input,
                continuous_state_input=not config.pi05, dtype=config.dtype)


def family_metadata(config):
    return dict(model_family=model_family(config), model_spec=model_spec(config),
                normalization="quantile" if config.pi05 else "mean_std")


def normalize_config(config):
    """Fill only new family fields, preserving every legacy recipe/signature."""
    result = dict(config)
    expected = family_metadata(make_model_config(model_family(config), config["horizon"]))
    for key, value in expected.items():
        if key in result and result[key] != value:
            raise ValueError(f"checkpoint/config 的 {key} 与 model_family 不一致")
        result[key] = value
    if "adaptation" in result:
        expected_adaptation = adaptation_metadata(make_model_config(model_family(config), config["horizon"]), config["scope"])
        if result["adaptation"] != expected_adaptation:
            raise ValueError("checkpoint/config 的 adaptation 与 scope 不一致")
    return result


def _checkpoint_path(checkpoint: str | Path | None = None, family="pi05", *, require_exists=True) -> Path:
    if family not in ("pi0", "pi05"):
        raise ValueError("model_family 须为 pi0/pi05")
    # Each family has its own override; MY_PI05_BASE_CHECKPOINT never selects pi0 weights.
    variable = "MY_PI0_BASE_CHECKPOINT" if family == "pi0" else "MY_PI05_BASE_CHECKPOINT"
    path = Path(checkpoint or os.environ.get(variable,
        ROOT / f"checkpoints/base/openpi-assets/checkpoints/{family}_base")).resolve()
    if not path.is_relative_to(ROOT) or path.name != f"{family}_base":
        raise ValueError(f"需要本项目内匹配 {family} 的 {family}_base 基础权重目录：{path}")
    if require_exists and not (path / "params").is_dir():
        raise ValueError(f"基础权重目录缺少 params/：{path}")
    return path


def load_base_model(model_config, checkpoint: str | Path | None = None):
    """用数据端的同一份模型配置加载本项目基础权重，避免输入规格不一致。"""
    path = _checkpoint_path(checkpoint, model_family(model_config))
    params = model_lib.restore_params(path / "params", dtype=jnp.bfloat16)
    model = model_config.load(params)
    model.eval()
    return model


def load_trained_model(checkpoint, model_config):
    """恢复本项目完整训练 checkpoint，仅把合并后的模型交给推理端。"""
    from .checkpoints import checkpoint_metadata, restore_checkpoint
    from .train import _state_from_model, make_schedule

    metadata = checkpoint_metadata(checkpoint)
    config = normalize_config(metadata["config"])
    if family_metadata(model_config) != {k: config[k] for k in family_metadata(model_config)}:
        raise ValueError("推理模型 family/spec/normalization 与训练 checkpoint 不一致")
    if model_config.action_horizon != config["horizon"]:
        raise ValueError("推理 action horizon 必须与训练 checkpoint 一致")
    abstract = jax.eval_shape(
        lambda key: _state_from_model(lora_config(model_config, config["scope"]).create(key), model_config,
                                      make_schedule(config), config["scope"]),
        jax.random.key(config["seed"]),
    )
    state, _, _ = restore_checkpoint(checkpoint, abstract)
    model = nnx.merge(state.model_def, state.trainable, state.frozen)
    model.eval()
    return model


def init_lora_model(model_config, rng, checkpoint: str | Path | None = None, *, scope: str = "strict", mesh=None):
    """按训练范围保留 FP32 参数；其余基础参数冻结为 BF16，LoRA 随机初始化。"""
    config = lora_config(model_config, scope)
    frozen_filter = nnx.All(nnx.Param, nnx.Not(get_trainable_filter(config, scope)))

    def initialize(rng, partial_params=None):
        model = config.create(rng)
        graph, state = nnx.split(model)
        if partial_params is not None:
            state.replace_by_pure_dict(partial_params)
        state = nnx_utils.state_map(state, frozen_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return graph, state

    # 先得到形状，再加载已有参数；缺少的 LoRA 保留形状占位，供初始化时生成。
    _, shape = jax.eval_shape(initialize, rng)
    reference = shape.to_pure_dict()
    loaded = CheckpointWeightLoader(str(_checkpoint_path(checkpoint, model_family(model_config)) / "params")).load(reference)
    array_typing.check_pytree_equality(expected=reference, got=loaded, check_shapes=True, check_dtypes=True)
    partial = traverse_util.unflatten_dict({
        path: value for path, value in traverse_util.flatten_dict(loaded).items()
        if not isinstance(value, jax.ShapeDtypeStruct)
    })
    # JIT 会移除被基础权重覆盖的随机数组计算，只保留新增 LoRA 的实际初始化。
    if mesh is None:
        graph, state = jax.jit(initialize)(rng, partial)
    else:
        from openpi.training.sharding import fsdp_sharding
        shardings = fsdp_sharding(shape, mesh)
        graph, state = jax.jit(initialize, out_shardings=(jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec()), shardings))(rng, partial)
    return nnx.merge(graph, state)


def trace_flow(model, rng, observation, actions):
    """展开固定版 Pi0.compute_loss，返回学习所需中间张量；训练仍用官方方法。"""
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = model_lib.preprocess_observation(preprocess_rng, observation, train=False)
    noise = jax.random.normal(noise_rng, actions.shape)
    flow_time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    t = flow_time[..., None, None]
    noisy_actions = t * noise + (1 - t) * actions
    target_velocity = noise - actions

    prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
    suffix, suffix_mask, suffix_ar, time_condition = model.embed_suffix(observation, noisy_actions, flow_time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
    attention_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [prefix, suffix], mask=attention_mask, positions=positions, adarms_cond=[None, time_condition],
    )
    predicted_velocity = model.action_out_proj(suffix_out[:, -model.action_horizon:])
    loss_per_step = jnp.mean(jnp.square(predicted_velocity - target_velocity), axis=-1)
    return {"noise": noise, "flow_time": flow_time, "noisy_actions": noisy_actions,
            "target_velocity": target_velocity, "prefix_tokens": prefix, "prefix_mask": prefix_mask,
            "suffix_tokens": suffix, "suffix_mask": suffix_mask, "time_condition": time_condition,
            "attention_mask": attention_mask, "positions": positions,
            "predicted_velocity": predicted_velocity, "loss_per_step": loss_per_step}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from .dataset import SpatialDataset, collate_batch

    parser.add_argument("--model-family", choices=("pi0", "pi05"), default="pi05")
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--index", type=int, default=20, help="所选划分内的全局样本起点")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    devices = jax.devices()
    if len(devices) != 1 or devices[0].platform != "gpu":
        parser.error("请先查看占用，再显式设置单张授权 GPU 的 CUDA_VISIBLE_DEVICES 和 JAX_PLATFORMS=cuda")
    if args.batch_size <= 0:
        parser.error("batch-size 须为正整数")
    try:
        dataset = SpatialDataset(args.split, args.horizon, model_family=args.model_family)
        indices = list(range(args.index, args.index + args.batch_size))
        observation, actions = collate_batch([dataset[index] for index in indices])
    except (OSError, ValueError, IndexError) as error:
        parser.error(str(error))
    observation, actions = jax.tree.map(jnp.asarray, (observation, actions))
    print(f"Loading local {args.model_family}_base (bfloat16)...", flush=True)
    started = time.perf_counter()
    model = load_base_model(dataset.model_config)
    parameters = jax.tree.leaves(nnx.state(model, nnx.Param))
    jax.block_until_ready(parameters)
    load_seconds = time.perf_counter() - started
    print(f"Loaded {sum(x.size for x in parameters):,} parameters in {load_seconds:.1f}s", flush=True)

    # NNX 模型拆成静态结构和数组状态；把状态当参数传给 JIT，不把大权重嵌入编译常量。
    graph, state = nnx.split(model)

    @jax.jit
    def traced(state, rng, obs, actions):
        return trace_flow(nnx.merge(graph, state), rng, obs, actions)

    @jax.jit
    def official(state, rng, obs, actions):
        return nnx.merge(graph, state).compute_loss(rng, obs, actions, train=False)

    rng = jax.random.key(args.seed)
    print("Compiling and tracing the real forward pass...", flush=True)
    started = time.perf_counter()
    trace = jax.block_until_ready(traced(state, rng, observation, actions))
    trace_seconds = time.perf_counter() - started
    print("Comparing with official compute_loss...", flush=True)
    reference = jax.block_until_ready(official(state, rng, observation, actions))
    np.testing.assert_allclose(trace["loss_per_step"], reference, rtol=1e-3, atol=1e-4)
    if not all(np.isfinite(np.asarray(value)).all() for value in trace.values() if value is not None):
        raise ValueError("前向中间张量出现非有限值")
    zero_state = observation.replace(state=jnp.zeros_like(observation.state))
    zero_state_loss = jax.block_until_ready(official(state, rng, zero_state, actions))
    summary = {"split": args.split, "source": [dataset.locate(i) for i in indices], "seed": args.seed,
               **family_metadata(dataset.model_config),
               "checkpoint": str(_checkpoint_path(family=args.model_family)), "devices": [str(d) for d in devices],
               "parameter_count": sum(x.size for x in parameters),
               "parameter_dtypes": sorted({str(x.dtype) for x in parameters}),
               "load_seconds": load_seconds, "trace_seconds_including_compile": trace_seconds,
               "input_actions_shape": list(actions.shape),
               "trace": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in trace.items() if value is not None},
               "flow_time": np.asarray(trace["flow_time"]).tolist(),
               "prefix_valid_tokens": np.asarray(trace["prefix_mask"].sum(axis=1)).tolist(),
               "attention_allowed_pairs": np.asarray(trace["attention_mask"].sum(axis=(1, 2))).tolist(),
               "loss_per_step": np.asarray(trace["loss_per_step"]).tolist(),
               "mean_loss": float(jnp.mean(trace["loss_per_step"])),
               "official_mean_loss": float(jnp.mean(reference)),
               "max_official_loss_difference": float(jnp.max(jnp.abs(trace["loss_per_step"] - reference))),
               "state_zero_loss_difference": float(jnp.max(jnp.abs(reference - zero_state_loss))),
               "parameter_updates": 0}
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def cli():
    """命令行只执行任务，不把模型状态作为进程退出值。"""
    main()


if __name__ == "__main__":
    cli()
