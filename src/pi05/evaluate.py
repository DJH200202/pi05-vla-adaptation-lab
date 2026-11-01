"""LIBERO Spatial 闭环：完整 checkpoint → 动作采样 → 环境执行 → 成功率/视频。"""

import argparse
from collections import deque
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from . import ROOT, RUNS

LIBERO_COMMIT = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"


def text_sha256(value):
    return hashlib.sha256(value.encode()).hexdigest()


def local_demo_xml(xml):
    """只重定位采集机器的 mesh/texture 文件路径，保留场景和相机参数。"""
    tree = ET.fromstring(xml)
    robosuite = Path(importlib.util.find_spec("robosuite").origin).parent
    assets = ROOT / "external/openpi/third_party/libero/libero/libero/assets"
    for element in tree.iter():
        name = element.get("file")
        if not name:
            continue
        parts = Path(name).parts
        if "robosuite" in parts:
            index = max(i for i, value in enumerate(parts) if value == "robosuite")
            path = robosuite.joinpath(*parts[index + 1:]).resolve()
        elif "assets" in parts:
            index = max(i for i, value in enumerate(parts) if value == "assets")
            path = assets.joinpath(*parts[index + 1:]).resolve()
        else:
            raise ValueError(f"示教XML存在未识别的素材路径：{name}")
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"示教XML素材不在本项目或不存在：{path}")
        element.set("file", str(path))
    return ET.tostring(tree, encoding="unicode")


def validation_manifest(config, episode_ids=None):
    """使用已有50条留出清单，记录原始demo、起点及XML身份。"""
    import h5py
    from libero.libero import benchmark

    data = ROOT / config["data_root"]
    ids_path = data / "manifests/validation_episode_ids.json"
    heldout = set(json.loads(ids_path.read_text()))
    selected = heldout if episode_ids is None else set(episode_ids)
    if not selected or not selected <= heldout or (episode_ids is not None and len(selected) != len(episode_ids)):
        raise ValueError("只允许选择既有验证清单中的唯一episode")
    suite = benchmark.get_benchmark_dict()["libero_spatial"](task_order_index=0)
    task_map = {suite.get_task(i).language: i for i in range(suite.n_tasks)}
    records = [json.loads(line) for line in (data / "manifests/episodes.jsonl").read_text().splitlines() if line.strip()]
    result = []
    for record in records:
        if record["episode_index"] not in selected:
            continue
        task_id = task_map[record["task"]]
        if task_id not in config["task_ids"]:
            continue
        if record["split"] != "validation":
            raise ValueError("验证清单与轨迹记录不一致")
        source = (ROOT / config["raw_data_root"] / record["source_file"]).resolve()
        if not source.is_relative_to(ROOT):
            raise ValueError("原始示教必须位于本项目")
        with h5py.File(source, "r") as file:
            demo = file["data"][record["source_demo"]]
            state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
            xml = demo.attrs["model_file"]
            if isinstance(xml, bytes):
                xml = xml.decode()
            if state.ndim != 1 or not np.isfinite(state).all():
                raise ValueError("原始示教起始状态无效")
            language = json.loads(file["data"].attrs["problem_info"])["language_instruction"]
            if language != record["task"]:
                raise ValueError("原始示教任务与验证清单不一致")
            localized = local_demo_xml(xml)
        result.append({"episode_index": record["episode_index"], "task_id": task_id,
                       "prompt": record["task"], "source_file": record["source_file"],
                       "source_demo": record["source_demo"], "source_demo_index": record["source_demo_index"],
                       "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                       "state_dimension": len(state), "model_xml_sha256": text_sha256(xml),
                       "localized_xml_sha256": text_sha256(localized)})
    result.sort(key=lambda row: row["episode_index"])
    if not result:
        raise ValueError("所选任务没有对应验证示教")
    if episode_ids is None:
        counts = {task: sum(row["task_id"] == task for row in result) for task in config["task_ids"]}
        if any(count != 5 for count in counts.values()):
            raise ValueError("每个任务应有5条完整留出示教")
    return result


def benchmark_manifest(config):
    """固定官方任务/初态身份；只读取仓库状态，不读取示教动作。"""
    import torch
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()["libero_spatial"](task_order_index=0)
    result = []
    for task_id in config["task_ids"]:
        task = suite.get_task(task_id)
        path = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        states = torch.load(path, map_location="cpu", weights_only=False)
        if config["episodes_per_task"] > len(states):
            raise ValueError(f"task {task_id}: 请求回合数超过官方初态数 {len(states)}")
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        bddl_hash = hashlib.sha256(Path(suite.get_task_bddl_file_path(task_id)).read_bytes()).hexdigest()
        for initial_id in range(config["episodes_per_task"]):
            state = np.asarray(states[initial_id])
            if state.ndim != 1 or not np.isfinite(state).all():
                raise ValueError("官方初态无效")
            result.append(dict(task_id=task_id, task=task.name, prompt=task.language,
                               initial_state_id=initial_id, state_dimension=len(state),
                               initial_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
                               initial_states_file=str(path.relative_to(ROOT)),
                               initial_states_file_sha256=file_hash, bddl_sha256=bddl_hash))
    return result


def read_validation_start(record, config):
    import h5py

    source = (ROOT / config["raw_data_root"] / record["source_file"]).resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError("原始示教必须位于本项目")
    with h5py.File(source, "r") as file:
        demo = file["data"][record["source_demo"]]
        state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
        xml = demo.attrs["model_file"]
        if isinstance(xml, bytes):
            xml = xml.decode()
        images = [demo[f"obs/{key}"][0] for key in ("agentview_rgb", "eye_in_hand_rgb")]
    localized = local_demo_xml(xml)
    if (hashlib.sha256(state.tobytes()).hexdigest() != record["initial_state_sha256"]
            or text_sha256(xml) != record["model_xml_sha256"]
            or text_sha256(localized) != record["localized_xml_sha256"]):
        raise ValueError("原始示教起点或XML在清单生成后发生变化")
    return state, localized, images


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_libero():
    """导入前设置本项目配置，避免 LIBERO 写 ~/.libero 或询问全局路径。"""
    source = ROOT / "external/openpi/third_party/libero"
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != LIBERO_COMMIT:
        raise ValueError("LIBERO 源码不是本项目固定提交")
    package = source / "libero/libero"
    config_dir = ROOT / ".local-config/libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    expected = {"benchmark_root": str(package), "bddl_files": str(package / "bddl_files"),
                "init_states": str(package / "init_files"), "datasets": str(ROOT / "data/raw"),
                "assets": str(package / "assets")}
    # JSON 也是合法 YAML；父进程写一次，worker 只读。
    if not config_path.exists() or json.loads(config_path.read_text()) != expected:
        write_json(config_path, expected)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    # 本服务器缺系统 NVIDIA EGL 注册；只使用项目内同版本用户态库。
    egl = ROOT / "tools/nvidia-egl"
    if (egl / "lib/libEGL_nvidia.so.0").is_file():
        vendor = egl / "10_nvidia.json"
        if not vendor.exists():
            write_json(vendor, {"file_format_version": "1.0.0",
                                "ICD": {"library_path": str(egl / "lib/libEGL_nvidia.so.0")}})
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor)
        os.environ["LD_LIBRARY_PATH"] = str(egl / "lib") + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def make_policy(checkpoint, denoise_steps, *, model_source="trained", evaluation_config=None):
    import jax
    from openpi import transforms
    from openpi.policies.policy import Policy
    from .checkpoints import checkpoint_metadata
    from .dataset import SpatialDataset
    from .model import load_base_model, load_trained_model, normalize_config, model_family
    from .train import training_config

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise ValueError("每个评估 worker 必须只看到一张 GPU")
    # The reference run supplies rollout protocol only, never the policy family or normalization.
    previous = (dict(evaluation_config) if model_source == "base"
                else normalize_config(checkpoint_metadata(checkpoint)["config"]))
    dataset = SpatialDataset("train", previous["horizon"], ROOT / previous["data_root"],
                             model_family=model_family(previous))
    current = training_config(previous, dataset)
    for key in ("data_signature", "validation_signature", "openpi_commit", "model_family", "model_spec", "normalization"):
        if key in previous and current[key] != previous[key]:
            raise ValueError(f"推理的 {key} 与模型配置不一致")
    data = dataset.data_config
    model = (load_base_model(dataset.model_config, checkpoint) if model_source == "base"
             else load_trained_model(checkpoint, dataset.model_config))
    return Policy(
        model,
        transforms=[*data.data_transforms.inputs,
                    # 推理只有观察，没有训练时的 actions 标签；按官方 Policy 允许缺少该键。
                    transforms.Normalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                    *data.model_transforms.inputs],
        output_transforms=[*data.model_transforms.outputs,
                           transforms.Unnormalize(data.norm_stats, use_quantiles=data.use_quantile_norm),
                           *data.data_transforms.outputs],
        sample_kwargs={"num_steps": denoise_steps},
    ), previous["horizon"]


def camera_images(observation, size=256):
    from PIL import Image
    # 与训练转换及官方 LIBERO 例程一致：原始渲染图各旋转 180°，只做一次。
    return [np.asarray(Image.fromarray(np.ascontiguousarray(observation[key][::-1, ::-1]))
                       .resize((size, size), Image.Resampling.BILINEAR))
            for key in ("agentview_image", "robot0_eye_in_hand_image")]


def policy_input(observation, prompt, image_size=256):
    quaternion = np.asarray(observation["robot0_eef_quat"], dtype=np.float64)
    w = np.clip(quaternion[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - w * w)
    axis_angle = (np.zeros(3) if denominator < 1e-8
                  else quaternion[:3] * (2.0 * np.arccos(w) / denominator))
    image, wrist = camera_images(observation, image_size)
    return {"observation/image": image, "observation/wrist_image": wrist,
            "observation/state": np.concatenate((observation["robot0_eef_pos"], axis_angle,
                                                  observation["robot0_gripper_qpos"])).astype(np.float32),
            "prompt": prompt}


def run_episode(policy, horizon, env, task, task_id, initial_state, initial_id, config, run_dir, demo=None):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    started = time.monotonic()
    env.seed(config["seed"])
    env.reset()
    source_images = None
    restore_error = None
    if demo is not None:
        initial_state, xml, source_images = read_validation_start(demo, config)
        env.reset_from_xml_string(xml)
        env.sim.reset()
    observation = env.set_init_state(initial_state)
    if demo is not None:
        restored = env.get_sim_state()
        if restored.shape != initial_state.shape:
            raise ValueError("恢复后的仿真状态维度与原始示教不一致")
        restore_error = float(np.max(np.abs(restored - initial_state)))
        if restore_error > 1e-8:
            raise ValueError(f"示教起始状态恢复误差过大：{restore_error}")
    for _ in range(config["settle_steps"]):
        observation, _, _, _ = env.step([0.0] * 6 + [-1.0])
    if env.check_success():
        raise ValueError("初始状态在策略执行前已满足成功条件，不能计作模型成功")
    noise_rng = np.random.default_rng(np.random.SeedSequence([config["seed"], task_id, initial_id]))
    action_plan = deque()
    actions_taken, call_seconds = [], []
    predictions, plan_steps, execution_calls, execution_offsets = [], [], [], []
    sim_states = [env.get_sim_state().copy()]
    success, steps = False, 0
    stem = f"episode{initial_id:04d}_task{task_id:02d}" if demo else f"task{task_id:02d}_init{initial_id:02d}"
    temporary = run_dir / "videos" / f"{stem}_running.mp4"
    start_comparison = None
    start_image_mae = None
    if source_images is not None:
        preview = Image.new("RGB", (512, 560), "black")
        draw = ImageDraw.Draw(preview)
        draw.text((6, 3), f"Held-out episode {initial_id}: first recorded demo observation (post action)", fill="white")
        original = [np.asarray(Image.fromarray(np.ascontiguousarray(image[::-1, ::-1]))
                               .resize((256, 256), Image.Resampling.BILINEAR)) for image in source_images]
        current = camera_images(observation)
        preview.paste(Image.fromarray(np.concatenate(original, axis=1)), (0, 24))
        draw.text((6, 283), "Restored demo init_state before any policy action", fill="white")
        preview.paste(Image.fromarray(np.concatenate(current, axis=1)), (0, 304))
        start_comparison = run_dir / "starts" / f"{stem}.png"
        preview.save(start_comparison)
        start_image_mae = [float(np.abs(a.astype(float) - b.astype(float)).mean()) for a, b in zip(original, current)]
    if demo is None:
        start_comparison = run_dir / "starts" / f"{stem}.png"
        Image.fromarray(np.concatenate(camera_images(observation), axis=1)).save(start_comparison)
    # 单帧左右分别为外部/腕部相机，标题写明任务和步数；逐帧写入不积存视频。
    with imageio.get_writer(temporary, fps=20, codec="libx264", pixelformat="yuv420p",
                            output_params=["-movflags", "+faststart"]) as video:
        def record(obs, label):
            canvas = Image.new("RGB", (512, 304), "black")
            canvas.paste(Image.fromarray(np.concatenate(camera_images(obs), axis=1)), (0, 48))
            draw = ImageDraw.Draw(canvas)
            draw.text((6, 3), f"{config['scope']} step{config['checkpoint_step']} | task {task_id} | {'val ep' if demo else 'init'} {initial_id} | {label}", fill="white")
            words = task.language.split()
            first, second = [], []
            for word in words:
                (first if len(" ".join(first + [word])) < 79 and not second else second).append(word)
            draw.text((6, 17), " ".join(first), fill="white")
            draw.text((6, 31), " ".join(second), fill="white")
            video.append_data(np.asarray(canvas))

        record(observation, "start")
        for step in range(config["max_steps"]):
            if not action_plan:
                began = time.monotonic()
                result = policy.infer(policy_input(observation, task.language, config.get("policy_image_size", 256)),
                                      noise=noise_rng.standard_normal((horizon, 32), dtype=np.float32))
                call_seconds.append(time.monotonic() - began)
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.shape != (horizon, 7) or not np.isfinite(actions).all():
                    raise ValueError(f"采样动作无效：{actions.shape}")
                call = len(predictions)
                predictions.append(actions.copy())
                plan_steps.append(step)
                action_plan.extend((action.copy(), call, offset) for offset, action in enumerate(actions[:config["replan_steps"]]))
            action, call, offset = action_plan.popleft()
            observation, _, done, _ = env.step(action.tolist())
            actions_taken.append(action)
            execution_calls.append(call)
            execution_offsets.append(offset)
            sim_states.append(env.get_sim_state().copy())
            steps = step + 1
            success = bool(env.check_success())
            record(observation, f"step {steps}" + (" SUCCESS" if success else ""))
            if success or done:
                break
    outcome = "success" if success else ("terminated" if steps < config["max_steps"] else "timeout")
    video_path = temporary.with_name(f"{stem}_{outcome}.mp4")
    temporary.rename(video_path)
    actions_taken = np.stack(actions_taken)
    predictions = np.stack(predictions)
    if not np.array_equal(actions_taken, predictions[execution_calls, execution_offsets]):
        raise ValueError("执行动作不能追溯到对应预测动作")
    trace = run_dir / "traces" / f"{stem}.npz"
    np.savez_compressed(trace, initial_state=initial_state, sim_states=np.stack(sim_states),
                        predicted_actions=predictions, executed_actions=actions_taken,
                        prediction_control_steps=np.asarray(plan_steps),
                        execution_call=np.asarray(execution_calls), execution_offset=np.asarray(execution_offsets))
    result = dict(task_id=task_id, task=task.name, prompt=task.language, initial_state_id=initial_id,
                initial_state_sha256=hashlib.sha256(np.asarray(initial_state).tobytes()).hexdigest(),
                seed=config["seed"], success=success, outcome=outcome, steps=steps,
                seconds=time.monotonic() - started, policy_calls=len(call_seconds),
                inference_seconds=sum(call_seconds), first_inference_seconds=call_seconds[0],
                action_min=actions_taken.min(axis=0).tolist(), action_max=actions_taken.max(axis=0).tolist(),
                first_action=actions_taken[0].tolist(),
                video=str(video_path.relative_to(run_dir)), trace=str(trace.relative_to(run_dir)),
                time=datetime.now().astimezone().isoformat())
    if demo is not None:
        result.update(episode_index=demo["episode_index"], source_file=demo["source_file"],
                      source_demo=demo["source_demo"], model_xml_sha256=demo["model_xml_sha256"],
                      restore_state_max_abs_error=restore_error, first_recorded_image_mae=start_image_mae,
                      start_comparison=str(start_comparison.relative_to(run_dir)), episode_source="validation")
    else:
        result.update(episode_source="benchmark", start_image=str(start_comparison.relative_to(run_dir)))
    return result


def worker(run_dir, worker_id, workers):
    prepare_libero()
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    torch.set_num_threads(1)
    config = json.loads((run_dir / "config.json").read_text())
    suite = benchmark.get_benchmark_dict()["libero_spatial"](task_order_index=0)
    task_texts = {json.loads(row)["task"] for row in (ROOT / config["data_root"] / "meta/tasks.jsonl").read_text().splitlines()}
    if {suite.get_task(i).language for i in range(suite.n_tasks)} != task_texts:
        raise ValueError("仿真任务文本与 Spatial-500 的10个训练任务不一致")
    results_path = run_dir / f"episodes-{worker_id:02d}.jsonl"
    previous = [json.loads(row) for row in results_path.read_text().splitlines()] if results_path.exists() else []
    completed = {(row["task_id"], row["initial_state_id"]) for row in previous if row["outcome"] != "error"}
    validation = config.get("episode_source") == "validation"
    manifest = None
    if validation or "eval_manifest_sha256" in config:
        manifest_text = (run_dir / "eval_manifest.json").read_text()
        if text_sha256(manifest_text) != config["eval_manifest_sha256"]:
            raise ValueError("固定评测清单发生变化")
        manifest = json.loads(manifest_text)
    if not validation and manifest is not None and benchmark_manifest(config) != manifest:
        raise ValueError("官方初态或任务定义与固定清单不一致")
    np.random.seed(config["seed"])
    print(f"worker {worker_id}: loading checkpoint {config['checkpoint']}", flush=True)
    policy, horizon = make_policy(ROOT / config["checkpoint"], config["denoise_steps"],
                                 model_source=config.get("model_source", "trained"),
                                 evaluation_config=config)
    print(f"worker {worker_id}: model ready; H={horizon}, replan={config['replan_steps']}", flush=True)
    for task_id in config["task_ids"]:
        task = suite.get_task(task_id)
        if validation:
            jobs = {row["episode_index"]: row for index, row in enumerate(manifest)
                    if row["task_id"] == task_id and index % workers == worker_id
                    and (task_id, row["episode_index"]) not in completed}
            initial_ids = list(jobs)
        else:
            jobs = {}
            initial_ids = [i for i in range(config["episodes_per_task"])
                           if (i * len(config["task_ids"]) + config["task_ids"].index(task_id)) % workers == worker_id
                           and (task_id, i) not in completed]
        if not initial_ids:
            continue
        # 官方固定资产中的 NumPy 状态；新 Torch 默认 weights_only=True 不支持此格式。
        initial_states = None
        if not validation:
            states_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
            initial_states = torch.load(states_file, map_location="cpu", weights_only=False)
            if max(initial_ids) >= len(initial_states):
                raise ValueError("评估回合数超过该任务固定初始状态数量")
        env = OffScreenRenderEnv(bddl_file_name=str(suite.get_task_bddl_file_path(task_id)),
                                 camera_heights=config.get("render_resolution", 128), camera_widths=config.get("render_resolution", 128),
                                 render_gpu_device_id=int(os.environ["MY_PI05_RENDER_GPU"]))
        try:
            for initial_id in initial_ids:
                try:
                    result = run_episode(policy, horizon, env, task, task_id,
                                         None if validation else initial_states[initial_id],
                                         initial_id, config, run_dir, jobs.get(initial_id))
                except Exception as error:
                    result = dict(task_id=task_id, initial_state_id=initial_id, outcome="error",
                                  success=None, error=repr(error), traceback=traceback.format_exc())
                    with results_path.open("a") as file:
                        file.write(json.dumps(result, ensure_ascii=False) + "\n")
                        file.flush()
                        os.fsync(file.fileno())
                    raise
                with results_path.open("a") as file:
                    file.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                    file.flush()
                    os.fsync(file.fileno())
                print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            env.close()


def aggregate(run_dir, config, state):
    rows = {}
    for path in sorted(run_dir.glob("episodes-*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # worker 可能刚开始写最后一行，下轮再读。
            rows[row["task_id"], row["initial_state_id"]] = row
    valid = [row for row in rows.values() if row["outcome"] != "error"]
    successes = sum(row["success"] for row in valid)
    per_task = {}
    for task_id in config["task_ids"]:
        subset = [row for row in valid if row["task_id"] == task_id]
        per_task[str(task_id)] = dict(completed=len(subset), successes=sum(row["success"] for row in subset))
    summary = dict(state=state, planned=config.get("planned_episodes", len(config["task_ids"]) * (config["episodes_per_task"] or 0)),
                   completed=len(valid), successes=successes, failures=len(valid) - successes,
                   errors=sum(row["outcome"] == "error" for row in rows.values()),
                   success_rate=successes / len(valid) if valid else None,
                   per_task=per_task, updated=datetime.now().astimezone().isoformat())
    if config.get("episode_source") == "validation":
        manifest = json.loads((run_dir / "eval_manifest.json").read_text())
        expected = {(row["task_id"], row["episode_index"]) for row in manifest}
        if any(key not in expected for key in rows):
            raise ValueError("结果文件包含固定留出清单之外的回合")
        summary["episode_source"] = "validation"
        summary["validation_episode_ids"] = sorted(row["episode_index"] for row in manifest)
        for task, value in per_task.items():
            value["planned"] = sum(row["task_id"] == int(task) for row in manifest)
    elif "eval_manifest_sha256" in config:
        manifest = json.loads((run_dir / "eval_manifest.json").read_text())
        expected = {(row["task_id"], row["initial_state_id"]): row for row in manifest}
        if any(key not in expected for key in rows):
            raise ValueError("结果包含官方固定清单之外的回合")
        for key, row in rows.items():
            if row["outcome"] != "error" and row["initial_state_sha256"] != expected[key]["initial_state_sha256"]:
                raise ValueError("实际运行初态与固定清单不一致")
        summary["episode_source"] = "benchmark"
        summary["eval_manifest_sha256"] = config["eval_manifest_sha256"]
        for task, value in per_task.items():
            value["planned"] = sum(row["task_id"] == int(task) for row in manifest)
            value["success_rate"] = value["successes"] / value["completed"] if value["completed"] else None
    write_json(run_dir / "summary.json", summary)
    return summary


def compare_benchmark_runs(run_dirs, output_dir):
    """仅汇总同清单、同协议且产物完整的Spatial500结果。"""
    import csv

    output_dir = Path(output_dir).resolve()
    if not output_dir.is_relative_to(RUNS):
        raise ValueError("比较产物须位于outputs/runs")
    groups, paired, reference = [], {}, None
    keys = ("eval_manifest_sha256", "task_ids", "episodes_per_task", "seed", "max_steps", "settle_steps",
            "replan_steps", "denoise_steps", "horizon", "render_resolution", "policy_image_size", "resize",
            "libero_commit", "openpi_commit", "data_signature", "normalization", "policy_noise")
    for directory in run_dirs:
        directory = Path(directory).resolve()
        config = json.loads((directory / "config.json").read_text())
        protocol = {key: config[key] for key in keys}
        if config["episode_source"] != "benchmark" or config["planned_episodes"] != 500:
            raise ValueError("比较需要完整官方Spatial500")
        if reference is not None and reference != protocol:
            raise ValueError("模型间协议或清单不一致")
        reference = protocol
        manifest_text = (directory / "eval_manifest.json").read_text()
        if text_sha256(manifest_text) != config["eval_manifest_sha256"]:
            raise ValueError("清单哈希不一致")
        summary = json.loads((directory / "summary.json").read_text())
        if (summary["state"] != "completed" or summary["completed"] != 500 or summary["errors"]
                or (directory / "exit-code.txt").read_text().strip() != "0"):
            raise ValueError(f"评估未完整成功结束：{directory}")
        rows = {}
        for path in directory.glob("episodes-*.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                key = (row["task_id"], row["initial_state_id"])
                if key in rows:
                    raise ValueError("存在重复回合记录，需检查后再汇总")
                for field in ("video", "trace", "start_image"):
                    artifact = directory / row[field]
                    if not artifact.is_file() or artifact.stat().st_size == 0:
                        raise ValueError(f"缺少产物：{artifact}")
                rows[key] = row
        expected = {(r["task_id"], r["initial_state_id"]): r for r in json.loads(manifest_text)}
        if rows.keys() != expected.keys() or sum(r["success"] for r in rows.values()) != summary["successes"]:
            raise ValueError("回合记录与summary/manifest不一致")
        for key, row in rows.items():
            if row["initial_state_sha256"] != expected[key]["initial_state_sha256"]:
                raise ValueError("实际初态与manifest不一致")
            paired.setdefault(key, dict(task_id=key[0], initial_state_id=key[1], prompt=row["prompt"]))[directory.name] = row["success"]
        groups.append(dict(group=directory.name, scope=config["scope"], checkpoint=config["checkpoint"],
                           run_dir=str(directory.relative_to(ROOT)), **summary))
    write_json(output_dir / "comparison.json", dict(state="completed", protocol=reference, groups=groups,
                                                    episodes=[paired[k] for k in sorted(paired)]))
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["group", "scope", "successes", "episodes", "success_rate"] + [f"task_{i:02d}_successes" for i in range(10)])
        for row in groups:
            writer.writerow([row["group"], row["scope"], row["successes"], row["completed"], row["success_rate"]]
                            + [row["per_task"][str(i)]["successes"] for i in range(10)])
    with (output_dir / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["task_id", "initial_state_id", "prompt"] + [r["group"] for r in groups])
        writer.writeheader()
        writer.writerows(paired[k] for k in sorted(paired))
    train_ids = json.loads((ROOT / config["data_root"] / "manifests/train_episode_ids.json").read_text())
    lines = ["# π0.5 官方初态 Spatial500 对比", "",
             f"微调组训练示教{len(train_ids)}条；Base未微调。每组官方10任务×50初态闭环评估。", "",
             "|方案|成功回合|成功率|", "|---|---:|---:|"]
    lines += [f"|{r['group']} ({r['scope']})|{r['successes']}/500|{r['success_rate']:.1%}|" for r in groups]
    lines += ["", "逐任务见summary.csv；逐回合配对见episodes.csv。使用项目固定仿真环境与逐回合噪声流，非官方权重/环境的逐位复现。"]
    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"comparison": str(output_dir), "groups": len(groups), "state": "completed"}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-runs", type=Path, nargs="+", help="只汇总已完成的官方Spatial500评估")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model-source", choices=("trained", "base"), default="trained")
    parser.add_argument("--model-family", choices=("pi0", "pi05"), help="Base默认pi05；训练模型从checkpoint推断")
    parser.add_argument("--data-root", type=Path, help="Base可显式指定训练统计数据版本；trained从checkpoint读取")
    parser.add_argument("--reference-run", type=Path, help="Base/训练模型均可继承此run的完整留出50条清单和闭环协议")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu-uuids", nargs="+")
    parser.add_argument("--episodes-per-task", type=int, default=40)
    parser.add_argument("--episode-source", choices=("benchmark", "validation"), default="benchmark")
    parser.add_argument("--validation-episode-ids", type=int, nargs="+", help="仅用于留出模式的小规模连通性检查；默认完整50条")
    parser.add_argument("--prepare-only", action="store_true", help="只生成固定配置和评测清单")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--render-resolution", type=int, choices=(128, 256),
                        help="新benchmark默认256；validation默认128，旧run恢复保留原配置")
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--settle-steps", type=int, help="benchmark默认10；留出示教默认0")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-id", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--workers", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if args.compare_runs:
        compare_benchmark_runs(args.compare_runs, run_dir)
        return
    if args.worker_id is not None:
        worker(run_dir, args.worker_id, args.workers)
        return
    if not run_dir.is_relative_to(RUNS) or run_dir == RUNS:
        parser.error("评估输出必须位于当前项目 outputs/runs/ 子目录")
    if args.resume:
        config = json.loads((run_dir / "config.json").read_text())
        if args.data_root is not None and (ROOT / args.data_root).resolve() != (ROOT / config["data_root"]).resolve():
            parser.error("--data-root 与已有评估不一致")
        if args.model_family is not None and args.model_family != config.get("model_family", "pi05"):
            parser.error("--model-family 与已有评估不一致")
    else:
        if not args.checkpoint or not args.gpu_uuids:
            parser.error("新评估需要 --checkpoint 和 --gpu-uuids")
        from .checkpoints import checkpoint_metadata
        from .dataset import SpatialDataset
        from .model import _checkpoint_path, family_metadata, model_family, normalize_config
        from .train import training_config

        checkpoint = args.checkpoint.resolve()
        reference, reference_dir = None, None
        if args.reference_run:
            if args.validation_episode_ids:
                parser.error("--reference-run 使用完整固定清单，不能另选episode")
            reference_dir = args.reference_run.resolve()
            if not reference_dir.is_relative_to(RUNS):
                parser.error("参考评估必须位于本项目outputs/runs")
            reference = json.loads((reference_dir / "config.json").read_text())
            if reference.get("episode_source") != "validation" or reference.get("planned_episodes") != 50:
                parser.error("配对参考必须是固定50条留出评估")
        if args.model_source == "base":
            if reference is None and args.episode_source != "benchmark":
                parser.error("Base留出评估需要--reference-run；官方初态评估可独立准备")
            family = args.model_family or "pi05"
            _checkpoint_path(checkpoint, family)
            # No trained checkpoint is needed to run a new family's base control.
            dataset = SpatialDataset("train", 10, ROOT / args.data_root if args.data_root else None, model_family=family)
            previous = training_config({"horizon": 10}, dataset)
            previous["base_checkpoint"] = str(checkpoint.relative_to(ROOT))
            step, scope = 0, "base"
        else:
            metadata = checkpoint_metadata(checkpoint)
            previous = normalize_config(metadata["config"])
            if args.data_root is not None and (ROOT / args.data_root).resolve() != (ROOT / previous["data_root"]).resolve():
                parser.error("--data-root 与训练checkpoint不一致")
            family = model_family(previous)
            if args.model_family is not None and args.model_family != family:
                parser.error("--model-family 与训练 checkpoint 不一致")
            dataset = SpatialDataset("train", previous["horizon"], ROOT / previous["data_root"], model_family=family)
            current = training_config(previous, dataset)
            for key in ("data_signature", "validation_signature", "openpi_commit"):
                if key in previous and current[key] != previous[key]:
                    parser.error(f"当前 {key} 与训练 checkpoint 不一致")
            step, scope = metadata["step"], previous["scope"]
        if args.settle_steps is None:
            args.settle_steps = 0 if args.episode_source == "validation" else 10
        if args.render_resolution is None:
            args.render_resolution = 128 if args.episode_source == "validation" else 256
        if args.validation_episode_ids and args.episode_source != "validation":
            parser.error("--validation-episode-ids 只用于validation模式")
        if not checkpoint.is_relative_to(ROOT):
            parser.error("本次使用当前项目内的正式 checkpoint")
        if reference is not None:
            for key in ("data_root", "data_signature", "validation_signature", "openpi_commit"):
                if key in reference and reference[key] != previous[key]:
                    parser.error(f"参考评估的 {key} 与当前数据/源码不一致")
            if reference.get("horizon", previous["horizon"]) != previous["horizon"]:
                parser.error("参考闭环协议的 horizon 与当前模型不一致")
            for key in ("episode_source", "episodes_per_task", "task_ids", "max_steps", "settle_steps",
                        "replan_steps", "denoise_steps", "seed"):
                setattr(args, key, reference[key])
        if ((args.episodes_per_task is not None and args.episodes_per_task < 1)
                or min(args.max_steps, args.replan_steps, args.denoise_steps) < 1
                or args.settle_steps < 0 or args.replan_steps > previous["horizon"]
                or len(set(args.task_ids)) != len(args.task_ids) or not all(0 <= i < 10 for i in args.task_ids)):
            parser.error("回合/步数/任务编号无效，或重规划长度超过训练 H")
        if (run_dir / "config.json").exists():
            parser.error("已有评估请用 --resume，避免覆盖")
        run_dir.mkdir(parents=True, exist_ok=True)
        config = {key: getattr(args, key) for key in ("episode_source", "episodes_per_task", "task_ids", "max_steps", "settle_steps",
                                                     "replan_steps", "denoise_steps", "seed", "gpu_uuids")}
        config.update(checkpoint=str(checkpoint.relative_to(ROOT)), model_source=args.model_source,
                      checkpoint_step=step, scope=scope, horizon=previous["horizon"],
                      **family_metadata(dataset.model_config),
                      base_checkpoint=previous.get("base_checkpoint", str(_checkpoint_path(family=family, require_exists=False).relative_to(ROOT))),
                      base_checkpoint_source=previous.get("base_checkpoint_source", f"gs://openpi-assets/checkpoints/{family}_base"),
                      data_root=previous["data_root"], data_signature=previous["data_signature"],
                      validation_signature=previous["validation_signature"],
                      suite="libero_spatial", task_order_index=0, render_resolution=args.render_resolution, video_fps=20,
                      policy_image_size=224 if args.episode_source == "benchmark" else 256,
                      resize=(f"{args.render_resolution} render -> 180-degree rotation -> PIL bilinear 224"
                              if args.episode_source == "benchmark" else
                              f"{args.render_resolution} render -> 180-degree rotation -> bilinear 256 -> model 224"),
                      libero_commit=LIBERO_COMMIT, openpi_commit=previous["openpi_commit"],
                      code_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
                      code_dirty=bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()),
                      evaluate_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      created=datetime.now().astimezone().isoformat())
        if reference is not None:
            keys = ("suite", "task_order_index", "render_resolution", "video_fps", "resize",
                    "raw_data_root", "planned_episodes", "eval_manifest_sha256",
                    "initial_state_id_namespace", "initial_state_source", "start_comparison_note")
            config.update({key: reference[key] for key in keys})
            config["policy_image_size"] = reference.get("policy_image_size", 256)
            manifest_text = (reference_dir / "eval_manifest.json").read_text()
            if text_sha256(manifest_text) != reference["eval_manifest_sha256"]:
                raise ValueError("参考评估的固定清单发生变化")
            manifest = json.loads(manifest_text)
            if len(manifest) != 50 or len({row["episode_index"] for row in manifest}) != 50:
                raise ValueError("参考评估必须包含50条唯一留出示教")
            (run_dir / "eval_manifest.json").write_text(manifest_text)
            config.update(paired_reference_run=str(reference_dir.relative_to(ROOT)),
                          reference_config_sha256=hashlib.sha256((reference_dir / "config.json").read_bytes()).hexdigest(),
                          reference_usage="exact heldout manifest and rollout protocol only; independent model family and normalization")
        elif args.episode_source == "validation":
            prepare_libero()
            config["raw_data_root"] = "data/raw/libero-original"
            manifest = validation_manifest(config, args.validation_episode_ids)
            write_json(run_dir / "eval_manifest.json", manifest)
            config.update(episodes_per_task=None, planned_episodes=len(manifest),
                          eval_manifest_sha256=text_sha256((run_dir / "eval_manifest.json").read_text()),
                          initial_state_id_namespace="validation_episode_index",
                          initial_state_source="HDF5 demo.attrs.init_state and model_file; no teacher actions",
                          start_comparison_note="First stored demo observation is post-action; not an exact pixel-identity assertion")
        else:
            prepare_libero()
            manifest = benchmark_manifest(config)
            write_json(run_dir / "eval_manifest.json", manifest)
            config.update(planned_episodes=len(manifest),
                          eval_manifest_sha256=text_sha256((run_dir / "eval_manifest.json").read_text()),
                          initial_state_id_namespace="official_task_id_and_initial_state_id",
                          initial_state_source="LIBERO task .pruned_init; no demonstration replay",
                          policy_noise="NumPy SeedSequence([seed, task_id, initial_state_id]); independent per-episode stream")
        source_dir = run_dir / "source"
        source_dir.mkdir(exist_ok=True)
        (source_dir / "evaluate.py").write_text(Path(__file__).read_text())
        (source_dir / "evaluate.patch").write_bytes(subprocess.check_output(
            ["git", "-C", str(ROOT), "diff", "HEAD", "--", "src/pi05/evaluate.py"]))
        write_json(run_dir / "config.json", config)
    if args.prepare_only:
        print(json.dumps({"prepared": str(run_dir), "planned": config["planned_episodes"],
                          "episode_source": config.get("episode_source", "benchmark")}, ensure_ascii=False))
        return
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    inventory = {uuid.strip(): (int(index), int(memory)) for index, uuid, memory in (line.split(",") for line in inventory.splitlines())}
    if len(set(config["gpu_uuids"])) != len(config["gpu_uuids"]):
        parser.error("GPU UUID 重复")
    for uuid in config["gpu_uuids"]:
        if uuid not in inventory or inventory[uuid][1] > 1000:
            parser.error(f"GPU 不存在或正在占用，请重新核对：{uuid}")
    prepare_libero()
    for folder in ("videos", "starts", "traces"):
        (run_dir / folder).mkdir(exist_ok=True)
    processes, logs = [], []
    try:
        for index, uuid in enumerate(config["gpu_uuids"]):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid, JAX_PLATFORMS="cuda", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MY_PI05_RENDER_GPU=str(inventory[uuid][0]),
                       MUJOCO_EGL_DEVICE_ID=str(inventory[uuid][0]),
                       XLA_PYTHON_CLIENT_PREALLOCATE="false")
            log = (run_dir / f"worker-{index:02d}.log").open("a")
            logs.append(log)
            processes.append(subprocess.Popen([sys.executable, "-u", "-m", "pi05.evaluate", "--run-dir", str(run_dir),
                                                "--worker-id", str(index), "--workers", str(len(config["gpu_uuids"]))],
                                               env=env, stdout=log, stderr=subprocess.STDOUT))
        while any(process.poll() is None for process in processes):
            summary = aggregate(run_dir, config, "running")
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            if any(process.poll() not in (None, 0) for process in processes):
                raise RuntimeError("仿真 worker 出错，检查对应 worker 日志；修复后可 --resume")
            time.sleep(15)
        summary = aggregate(run_dir, config, "completed")
        if any(process.returncode != 0 for process in processes) or summary["completed"] != summary["planned"] or summary["errors"]:
            raise RuntimeError("评估尚未完整完成")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except BaseException:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
        aggregate(run_dir, config, "interrupted")
        raise
    finally:
        for log in logs:
            log.close()


def cli():
    main()


if __name__ == "__main__":
    cli()
