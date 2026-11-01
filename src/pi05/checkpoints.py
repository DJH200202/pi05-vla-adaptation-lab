"""训练 checkpoint：完整数组、配置和采样游标；格式 1 保留恢复，新增写格式 2。"""

from pathlib import Path
import tempfile
import mmap

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp

from . import ROOT, RUNS


class _ImmutableCheckpointArray(np.memmap):
    """File-backed save input stays immutable until synchronous commit completes."""
    def __deepcopy__(self, memo):
        # Orbax copies NumPy inputs for async safety. Our files remain alive and
        # unchanged through wait_until_finished, so retaining this view is safe.
        return self


def checkpoint_path(path) -> Path:
    path = Path(path).resolve()
    if not path.is_relative_to(RUNS) or path == RUNS:
        raise ValueError("训练 checkpoint 必须位于本项目 outputs/runs/ 的子目录")
    return path


def checkpoint_metadata(path) -> dict:
    # 旧归档可显式读取；写入仍仅允许当前项目 outputs/runs/。
    path = Path(path).resolve()
    if not path.is_dir() or not ocp.utils.is_checkpoint_finalized(path):
        raise ValueError(f"checkpoint 不存在或尚未完成写入：{path}")
    metadata = ocp.metadata.get_step_metadata(path).custom_metadata
    if not metadata or metadata.get("format_version") not in (1, 2):
        raise ValueError("不是本 pipeline 支持的训练 checkpoint")
    return metadata


def checkpoint_arrays(state, train_rng):
    return {"step": state.step, "trainable": state.trainable, "frozen": state.frozen,
            "opt_state": state.opt_state, "train_rng": jax.random.key_data(train_rng)}


def optimizer_counters(state) -> list[int]:
    return [int(x) for x in jax.tree.leaves(state.opt_state)
            if x.shape == () and jnp.issubdtype(x.dtype, jnp.integer)]


def save_checkpoint(path, state, train_rng, config, sampler_state=None):
    """只在完整 optimizer step 后保存游标，等待 Orbax 原子提交，拒绝覆盖。"""
    path = checkpoint_path(path)
    metadata = {"format_version": 2, "step": int(state.step), "config": config,
                "rng_impl": str(jax.random.key_impl(train_rng)), "sampler": sampler_state}
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = checkpoint_arrays(state, train_rng)
    if config.get("scope") == "full":
        # Bound D2H transfers: avoid enqueueing every shard of the full Adam tree
        # at once. Save logical arrays, so restore can choose any device layout.
        import gc
        import faulthandler
        faulthandler.dump_traceback_later(180, repeat=True)
        from jax.experimental.layout import DeviceLocalLayout, Layout
        staging = tempfile.TemporaryDirectory(prefix="full-checkpoint-", dir=ROOT / ".cache/tmp")
        print(f"Checkpoint staging directory: {staging.name}", flush=True)
        copied = 0
        leaf_index = 0
        def host_array(array):
            nonlocal copied, leaf_index
            value = np.lib.format.open_memmap(Path(staging.name) / f"{leaf_index:04d}.npy", mode="w+", shape=array.shape, dtype=array.dtype)
            leaf_index += 1
            if value.nbytes > 1e9:
                print(f"Checkpoint staging large array: {array.shape}", flush=True)
            for shard in array.addressable_shards:
                if shard.replica_id != 0:
                    continue
                data = shard.data
                layout = Layout(DeviceLocalLayout(tuple(range(data.ndim))),
                                jax.sharding.SingleDeviceSharding(shard.device))
                if data.ndim and data.size * data.dtype.itemsize > 32 * 2**20:
                    # Bound pinned D2H buffers even for multi-GB scanned weights.
                    axis = max(range(data.ndim), key=lambda k: data.shape[k])
                    per_index = data.size * data.dtype.itemsize // data.shape[axis]
                    chunk = max(1, min(data.shape[axis], (32 * 2**20) // per_index))
                    take = jax.jit(lambda x, start: jax.lax.dynamic_slice_in_dim(x, start, chunk, axis=axis),
                                   out_shardings=layout)
                    for start in range(0, data.shape[axis], chunk):
                        start = min(start, data.shape[axis] - chunk)
                        destination = list(shard.index)
                        offset = destination[axis].start or 0
                        destination[axis] = slice(offset + start, offset + start + chunk)
                        block = take(data, start)
                        value[tuple(destination)] = np.asarray(block)
                        del block
                        gc.collect()
                else:
                    local = jax.jit(lambda x: x, out_shardings=layout)(data)
                    value[shard.index] = np.asarray(local)
                    del local
                    gc.collect()
            value.flush()
            value._mmap.madvise(mmap.MADV_DONTNEED)
            copied += value.nbytes
            print(f"Checkpoint host staging: {copied / 1e9:.2f} GB", flush=True)
            value.flags.writeable = False
            return value.view(_ImmutableCheckpointArray)
        arrays = jax.tree.map(host_array, arrays)
        faulthandler.cancel_dump_traceback_later()
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(path, arrays, custom_metadata=metadata)
        checkpointer.wait_until_finished()
    if config.get("scope") == "full":
        del arrays
        gc.collect()
        staging.cleanup()
    print(f"Saved checkpoint: {path}", flush=True)


def restore_checkpoint(path, abstract_state, *, sharding=None, state_shardings=None):
    """模型/优化器模板由 train.py 提供；这里仅负责严格读取数组和游标。"""
    path = Path(path).resolve()
    metadata = checkpoint_metadata(path)
    config = metadata["config"]
    rng = jax.random.key(config["seed"], impl=metadata["rng_impl"])
    sharding = sharding or jax.sharding.SingleDeviceSharding(jax.devices()[0])
    arrays = checkpoint_arrays(abstract_state, rng)
    if state_shardings is None:
        target = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=sharding), arrays)
    else:
        shardings = {k: getattr(state_shardings, k) for k in ("step", "trainable", "frozen", "opt_state")}
        shardings["train_rng"] = sharding
        target = jax.tree.map(lambda x, spec: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=spec), arrays, shardings)
    with ocp.StandardCheckpointer(restore_concurrent_gb=8 if config.get("scope") == "full" else 96) as checkpointer:
        restored = checkpointer.restore(path, target)
    state = abstract_state.replace(**{k: restored[k] for k in ("step", "trainable", "frozen", "opt_state")})
    rng = jax.random.wrap_key_data(restored["train_rng"], impl=metadata["rng_impl"])
    counters = optimizer_counters(state)
    if int(state.step) != metadata["step"] or not counters or any(c != int(state.step) for c in counters):
        raise ValueError("checkpoint 的 step、Adam 或学习率计数不一致")
    sampler = metadata.get("sampler")
    if metadata["format_version"] == 2 and config["sampling"] == "episode_shuffle" and sampler is None:
        raise ValueError("checkpoint 缺少采样游标")
    return state, rng, sampler
