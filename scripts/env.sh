#!/usr/bin/env bash
# 在 Bash 中 source；配置项目路径，运行时显式使用 "$MY_PI05_PYTHON"。
export MY_PI05_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export MY_PI05_PYTHON="$MY_PI05_ROOT/envs/openpi/bin/python"
export XDG_CACHE_HOME="$MY_PI05_ROOT/.cache"
export TMPDIR="$MY_PI05_ROOT/.cache/tmp"
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export UV_PYTHON_BIN_DIR="$MY_PI05_ROOT/tools/bin"
export UV_TOOL_DIR="$MY_PI05_ROOT/tools/uv-tools"
export UV_TOOL_BIN_DIR="$MY_PI05_ROOT/tools/bin"
export UV_LINK_MODE=copy
export UV_PYTHON_INSTALL_DIR="$MY_PI05_ROOT/tools/python"
export UV_PROJECT_ENVIRONMENT="$MY_PI05_ROOT/envs/openpi"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip"
export PIP_REQUIRE_VIRTUALENV=true
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export HF_HOME="$XDG_CACHE_HOME/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_LEROBOT_HOME="$XDG_CACHE_HOME/lerobot"
export OPENPI_DATA_HOME="$MY_PI05_ROOT/checkpoints/base"
export JAX_COMPILATION_CACHE_DIR="$XDG_CACHE_HOME/jax"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="$MY_PI05_ROOT/outputs/wandb"
export WANDB_MODE=offline
export WANDB_CACHE_DIR="$XDG_CACHE_HOME/wandb"
export WANDB_CONFIG_DIR="$MY_PI05_ROOT/.local-config/wandb"
export WANDB_DATA_DIR="$MY_PI05_ROOT/outputs/wandb-data"
export TORCH_HOME="$XDG_CACHE_HOME/torch"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
# 默认使用 CPU；训练/推理入口再按授权 UUID 显式选择 GPU 后端。
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
unset PYTHONPATH PYTHONHOME LEROBOT_HOME
case ":$PATH:" in
  *":$MY_PI05_ROOT/tools/bin:"*) ;;
  *) export PATH="$MY_PI05_ROOT/tools/bin:$PATH" ;;
esac
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"

# 全部运行资产都实体存放在本项目；原始资产按只读约定使用。
export MY_PI05_DATA_ROOT="$MY_PI05_ROOT/data/processed/libero-spatial-500-v1"
export MY_PI0_BASE_CHECKPOINT="$MY_PI05_ROOT/checkpoints/base/openpi-assets/checkpoints/pi0_base"
export MY_PI05_BASE_CHECKPOINT="$MY_PI05_ROOT/checkpoints/base/openpi-assets/checkpoints/pi05_base"
export MY_PI05_LIBERO_CHECKPOINT="$MY_PI05_ROOT/checkpoints/base/openpi-assets/checkpoints/pi05_libero"
export MY_PI05_TOKENIZER="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
export MY_PI05_FINETUNED_ROOT="$MY_PI05_ROOT/checkpoints/finetuned"
