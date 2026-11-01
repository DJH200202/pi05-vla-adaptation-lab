# 查看、重建图表与运行仿真

## 1. 零 GPU 查看已有结果

克隆私有仓库需要账号权限。直接双击 `demo/index.html`，或运行：

```bash
python3 -m http.server 8768 --bind 127.0.0.1
# http://127.0.0.1:8768/demo/
```

网页使用本地JS数据，不需要CDN、登录第三方服务或模型权重。视频均为已完成闭环记录。源码路径和原始日志中的服务器目录只是来源说明，不是观看Demo的运行依赖。

## 2. 从日志重建图表（不训练、不评估）

```bash
python3 -m venv .venv-demo
source .venv-demo/bin/activate
python -m pip install -r requirements-demo.txt
python scripts/build_demo_data.py
python scripts/plot_results.py
python scripts/verify_demo.py
```

`build_demo_data.py`检查每组更新步数1–3000、同一步sample_indices一致、退出码、每组500条唯一回合/每任务50条、汇总与逐回合一致以及manifest哈希。输出`demo/assets/data.{js,json}`与`results/derived.json`。`plot_results.py`只消费这些记录。

loss平滑定义：`EMA[0]=loss[0]`，`EMA[t]=0.05*loss[t]+0.95*EMA[t-1]`；保留全部原始值。计时排除CSV明确标记的编译步1、3。时间为当次运行记录，不作公平硬件速度基准。

## 3. 业务源码与运行资产

`src/pi05/`是实际实验源码快照。运行模型需要Linux NVIDIA GPU环境、固定上游源码、Python依赖、数据/归一化、tokenizer和checkpoint。本仓库未上传这些大型资产，也不声称仅clone即可推理。

- Python 3.11.17；固定依赖见`configs/requirements.lock`。
- OpenPI提交`215abfb217dbac7d5f1273282331b9b1866c0479`，放置于`external/openpi`。
- 其LIBERO子模块提交`f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`。
- 仿真包：MuJoCo2.3.7、robosuite1.4.1，具体以锁文件为准。
- 原始500示教来源见`results/dataset/conversion.json`。
- `checkpoints/base/big_vision/paligemma_tokenizer.model`与基础权重路径见运行配置。
- 完成训练的权重放于本项目内；推理需完整`step_003000/`，不是单个LoRA文件。

保留的`scripts/setup_openpi.sh`是原工程严格固定runtime的环境脚本，要求先准备`tools/bin/uv`和项目内部Python；它不是面向任意电脑的一键安装器。迁移时按其中要求配置，不修改全局Python/驱动。上游初始化示例：

```bash
git clone https://github.com/Physical-Intelligence/openpi.git external/openpi
git -C external/openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
git -C external/openpi submodule update --init third_party/libero
# 准备项目内部runtime、依赖与资产后，运行既有setup脚本。
```

## 4. 数据准备路径

原始HDF5放入`data/raw/libero-original/libero_spatial/`。转换器先创建保留全部500条的旧划分格式，再独立复制到全500训练版本并重算统计：

```bash
source scripts/env.sh
"$MY_PI05_PYTHON" scripts/convert_spatial_500.py
"$MY_PI05_PYTHON" scripts/convert_spatial_500.py \
  --reuse-processed data/processed/libero-spatial-500-v1 \
  --output data/processed/libero-spatial-train500-v1
```

这些命令用于新环境复现，当前Demo制作没有执行它们。转换器拒绝覆盖已有输出。配置`validation_mode=none`，不要将训练数据重新称作验证集。新数据元信息和统计副本见`results/dataset/`。

## 5. 训练命令说明（未在本次展示工作中执行）

在已完成环境准备、明确分配GPU的机器上，A/B/C/D的scope分别为action_lora/strict/default/expert_full。原协议如下：

```bash
source scripts/env.sh
export CUDA_VISIBLE_DEVICES='GPU-uuid0,GPU-uuid1,GPU-uuid2,GPU-uuid3'
export JAX_PLATFORMS=cuda
"$MY_PI05_PYTHON" -m pi05.train --config configs/train500.json \
  --scope default --devices 4 --steps 2 --run-dir outputs/runs/NEW-C
"$MY_PI05_PYTHON" -m pi05.train --config configs/train500.json \
  --devices 4 --steps 2998 --run-dir outputs/runs/NEW-C \
  --resume outputs/runs/NEW-C/checkpoints/step_000002
```

替换GPU UUID和NEW-C为新的目录，禁止覆盖原正式运行。`--steps`是本次追加步数，不能恢复后再写3000。global batch为2×16=32，不再额外乘卡数。

## 6. 可运行仿真 Demo

已有推理环境与完整checkpoint时，使用`src/pi05/evaluate.py`原入口即可。以下示例仅执行任务0的官方初态0；不是网页播放器。

```bash
source scripts/env.sh
"$MY_PI05_PYTHON" -m pi05.evaluate \
  --checkpoint outputs/runs/NEW-C/checkpoints/step_003000 \
  --episode-source benchmark --episodes-per-task 1 --task-ids 0 \
  --gpu-uuids GPU-available-uuid \
  --render-resolution 256 --settle-steps 10 --max-steps 220 \
  --replan-steps 5 --denoise-steps 10 --seed 7 \
  --run-dir outputs/runs/NEW-C-demo
```

将episodes-per-task改50、task-ids显式列出0至9，才是完整500回合协议。推理GPU由`--gpu-uuids`传给worker；不运行旧调度队列。当前整理阶段没有执行上述训练或仿真命令。

## 验证范围

本次完成了记录一致性核对、源文件哈希核对、图表重建、媒体检查及页面交互检查。未在新仓库重新安装全部GPU环境或执行推理，因此GPU复现步骤是基于实际入口与原配置整理的运行说明，而非新机器上的重新实测成绩。
