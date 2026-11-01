#!/usr/bin/env bash
# 仅在需要安装/重建本项目 OpenPI 环境时运行；可传 --offline 使用已填充缓存。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
for option in "$@"; do
  [[ "$option" == --offline ]] || { echo "Only --offline is supported" >&2; exit 2; }
  export UV_OFFLINE=1
done

expected=215abfb217dbac7d5f1273282331b9b1866c0479
[[ "$(git -C external/openpi rev-parse HEAD)" == "$expected" ]]
[[ -z "$(git -C external/openpi status --porcelain)" ]]
# 闭环环境使用 OpenPI 内已固定的 LIBERO 子模块，不跟随上游分支。
git -C external/openpi submodule update --init --depth 1 third_party/libero
[[ "$(git -C external/openpi/third_party/libero rev-parse HEAD)" == f78abd68ee283de9f9be3c8f7e2a9ad60246e95c ]]
uv_bin="$MY_PI05_ROOT/tools/bin/uv"
runtime="$MY_PI05_ROOT/tools/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11"
[[ "$("$uv_bin" --version)" == 'uv 0.13.0 (x86_64-unknown-linux-gnu)' ]]
[[ "$("$runtime" --version)" == 'Python 3.11.17' ]]
if [[ ! -x "$MY_PI05_PYTHON" ]]; then
  "$uv_bin" venv --python "$runtime" "$MY_PI05_ROOT/envs/openpi"
fi
# 拒绝复用指向别处 runtime 的同名环境。
"$MY_PI05_PYTHON" -I -c 'import os,sys; from pathlib import Path; r=Path(os.environ["MY_PI05_ROOT"]); assert Path(sys.prefix)==r/"envs/openpi"; assert Path(sys.base_prefix).is_relative_to(r/"tools/python")'

# LeRobot 固定 Git 对象的可选本地镜像；URL 重写仅限本脚本进程。
mirror="$MY_PI05_ROOT/tools/git-mirrors/lerobot.git"
if [[ -d "$mirror" ]]; then
  git -C "$mirror" cat-file -e '0cf864870cf29f4738d3ade893e6fd13fbd7cdb5^{commit}'
  export GIT_CONFIG_COUNT=1
  export GIT_CONFIG_KEY_0="url.file://$mirror.insteadOf"
  export GIT_CONFIG_VALUE_0=https://github.com/huggingface/lerobot
fi
export GIT_LFS_SKIP_SMUDGE=1
# 为已下载的大包生成临时安装来源；版本、marker 和 hash 仍受依赖锁约束。
"$MY_PI05_PYTHON" -I - <<'PY'
import hashlib
import os
from pathlib import Path
import re

root = Path(os.environ["MY_PI05_ROOT"])
lock = root / "configs/requirements.lock"
lines = lock.read_text().splitlines(keepends=True)
starts = [i for i, line in enumerate(lines)
          if line.strip() and not line[0].isspace() and not line.startswith("#")]
pattern = re.compile(r"^([A-Za-z0-9_.-]+)==([^ ;\\\n]+)(.*)$")
wheel_files = {}
for wheel in sorted((root / ".cache/wheels").glob("*.whl")):
    name, version = wheel.name.split("-")[:2]
    key = (name.replace("_", "-").lower(), version)
    if key in wheel_files:
        raise ValueError(f"多个候选 wheel，需明确平台选择：{key}")
    wheel_files[key] = wheel
verified = []
for position, start in enumerate(starts):
    match = pattern.match(lines[start].rstrip("\n"))
    if not match:
        continue
    name, version, suffix = match.groups()
    wheel = wheel_files.get((name.replace("_", "-").lower(), version))
    if wheel is None:
        continue
    end = starts[position + 1] if position + 1 < len(starts) else len(lines)
    expected = re.findall(r"--hash=sha256:([0-9a-f]{64})", "".join(lines[start:end]))
    digest = hashlib.sha256()
    with wheel.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if value not in expected:
        raise ValueError(f"wheel 与正式锁文件不匹配：{wheel.name}")
    # 原行的环境 marker 保留；只替换下载来源，并留下已经验证的 hash。
    suffix = suffix.rstrip().removesuffix("\\").rstrip()
    lines[start] = f"{name} @ {wheel.as_uri()}{suffix} \\\n    --hash=sha256:{value}\n"
    for index in range(start + 1, end):
        if lines[index].lstrip().startswith("--hash="):
            lines[index] = ""
    verified.append({"package": name, "version": version, "file": str(wheel), "sha256": value})
output = Path(os.environ["TMPDIR"]) / "openpi-install.requirements.txt"
output.write_text("".join(lines))
print(f"Using {len(verified)} local wheels matching the dependency lock")
PY
"$uv_bin" pip sync --python "$MY_PI05_PYTHON" \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple \
  "$@" "$TMPDIR/openpi-install.requirements.txt"
"$uv_bin" pip install --python "$MY_PI05_PYTHON" --no-deps \
  --default-index https://pypi.tuna.tsinghua.edu.cn/simple "$@" \
  -e external/openpi -e external/openpi/packages/openpi-client
# LIBERO 的外层目录为 namespace；compat 模式把固定源码根加入 .pth。
"$uv_bin" pip install --python "$MY_PI05_PYTHON" --no-deps --no-build-isolation "$@" \
  --config-settings editable_mode=compat -e external/openpi/third_party/libero
# 本项目只安装命令入口；运行依赖仍由上面的单一锁文件管理。
"$uv_bin" pip install --python "$MY_PI05_PYTHON" --no-deps --no-build-isolation "$@" -e .
"$uv_bin" pip check --python "$MY_PI05_PYTHON"
