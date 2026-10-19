<!--
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
SPDX-License-Identifier: Apache-2.0
-->

# TensorFlow 2.13.1 HCU/DTK 编译指南

本文档说明如何在 Linux x86_64 HCU 环境中手动编译、打包、安装和验证
TensorFlow 2.13.1 DAS 版本。TensorFlow 2.13.1 的设备代码后端使用 GCVM，
编译前需要准备匹配的 DTK 环境。

以下镜像拉取和容器启动命令在 Linux x86_64 HCU 宿主机执行；进入容器后，
再执行 DTK 安装、源码获取、编译和验证命令。

## 1. 拉取基础镜像

推荐使用光合社区基础镜像。以下镜像均不含 DTK，按目标系统和 Python 版本
选择其中一种。镜像 tag 会更新；拉取后记录实际镜像 digest，以便复现构建环境。
宿主机需已安装并能使用 Docker；先执行 `docker version` 确认客户端和服务端均可用。

| 环境 | 说明 |
| --- | --- |
| Ubuntu 20.04 / Python 3.11 | `dev` 基础镜像，不含 DTK |
| RockyLinux 8.6 | `dev` 基础镜像，tag 中的 Python 版本随 `PYTHON_VERSION` 变化，不含 DTK |

### 1.1 Ubuntu 20.04 / Python 3.11

```bash
export IMAGE=harbor.sourcefind.cn:5443/dcu/admin/base/dev:ubuntu20.04-mpi4.0-gcc9.4-cmake3.19-py3.11-mkl2020.4.304
docker pull "${IMAGE}"
```

### 1.2 RockyLinux 8.6

先选择镜像 tag 中的 Python 版本，例如 `3.11`：

```bash
export PYTHON_VERSION=3.11
export IMAGE="harbor.sourcefind.cn:5443/dcu/admin/base/dev:rockylinux8.6-mpi5.0-gcc10.3-cmake3.29-py${PYTHON_VERSION}-mkl2020.4.304"
docker pull "${IMAGE}"
```

记录所选镜像的 digest：

```bash
docker image inspect --format '{{json .RepoDigests}}' "${IMAGE}"
```

如果输出为空数组，应向镜像提供方确认可固定的 digest；仅记录可变 tag 不能保证
以后拉到相同内容。

## 2. 创建并启动容器

在宿主机上确认 HCU 设备和驱动目录存在。将 `HOST_WORKDIR` 改成实际的宿主机
工作目录；该目录会映射到容器内 `/workspace`，后续在此目录克隆源码。

| 参数 | 说明 |
| --- | --- |
| `CONTAINER_NAME` | 自定义容器名称 |
| `IMAGE` | 第 1 节拉取的镜像名称 |
| `HOST_WORKDIR` | 宿主机已有且可写的绝对路径 |
| `CONTAINER_WORKDIR` | 容器内工作目录，示例为 `/workspace` |

```bash
export CONTAINER_NAME=tensorflow-das-build
export HOST_WORKDIR=/path/to/host/workdir
export CONTAINER_WORKDIR=/workspace

# 部分宿主机没有 /dev/mkfd；存在时才映射。
MKFD_ARGS=()
if [ -e /dev/mkfd ]; then
  MKFD_ARGS=(--device=/dev/mkfd)
fi

test -d "${HOST_WORKDIR}" &&
test -e /dev/kfd &&
test -d /dev/dri &&
test -d /opt/hyhal &&
docker run -it \
  --name "${CONTAINER_NAME}" \
  --network=host \
  --restart=always \
  --privileged \
  --user root \
  --ipc=host \
  --shm-size=16G \
  --group-add video \
  --device=/dev/kfd \
  "${MKFD_ARGS[@]}" \
  --device=/dev/dri \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v "${HOST_WORKDIR}:${CONTAINER_WORKDIR}" \
  -v /opt/hyhal:/opt/hyhal:ro \
  --workdir "${CONTAINER_WORKDIR}" \
  "${IMAGE}" /bin/bash
```

第 3 节及后续构建命令均在上述容器内执行。退出交互 shell 后，
`--restart=always` 可能使容器自动重启；在宿主机执行
`docker exec -it "${CONTAINER_NAME}" /bin/bash` 再次进入。若容器已停止，
先执行 `docker start "${CONTAINER_NAME}"`。

## 3. 安装 DTK（按需）

如果镜像中已安装目标版本 DTK，可以跳过下载，但须按第 7 节检查版本及
`/opt/dtk/dcc/gcvm`。以下安装示例面向新容器；安装前确认 `/opt/dtk` 不存在，
不要直接覆盖已有 DTK。下载地址需要容器能够访问。

### 3.1 RockyLinux 8.6 / DTK 26.10

该系统和 DTK 包的组合使用 `/opt/dtk` 作为默认安装路径。将压缩包下载到 `/tmp`，
再解压到 `/opt/dtk`；`wget -O` 指定输出文件，`tar` 才负责解压。

```bash
DTK_ARCHIVE=/tmp/DTK-26.10-Rocky8.6-x86_64.tar.gz
if [ -e /opt/dtk ]; then
  echo '/opt/dtk 已存在，请先确认其版本或使用新容器。' >&2
else
  wget -O "${DTK_ARCHIVE}" \
    https://download.sourcefind.cn:65024/file/1/DTK-26.10/Rocky8.6/DTK-26.10-Rocky8.6-x86_64.tar.gz &&
  mkdir -p /opt/dtk &&
  tar -xzf "${DTK_ARCHIVE}" -C /opt/dtk --strip-components=1
fi
```

### 3.2 Ubuntu / DTK 26.10

```bash
DTK_ARCHIVE=/tmp/DTK-26.10-Ubuntu22.04-x86_64.tar.gz
if [ -e /opt/dtk ]; then
  echo '/opt/dtk 已存在，请先确认其版本或使用新容器。' >&2
else
  wget -O "${DTK_ARCHIVE}" \
    https://download.sourcefind.cn:65024/file/1/DTK-26.10/Ubuntu22.04/DTK-26.10-Ubuntu22.04-x86_64.tar.gz &&
  mkdir -p /opt/dtk &&
  tar -xzf "${DTK_ARCHIVE}" -C /opt/dtk --strip-components=1
fi
```

安装或确认已有 DTK 后，检查关键文件和实际版本：

```bash
test -f /opt/dtk/env.sh &&
test -x /opt/dtk/bin/hipcc &&
test -d /opt/dtk/dcc/gcvm &&
cat /opt/dtk/.info/version-dev &&
source /opt/dtk/env.sh &&
/opt/dtk/bin/hipcc --version
```

## 4. 支持范围

| 项目 | 支持或推荐值 |
| --- | --- |
| TensorFlow | `2.13.1` |
| DTK | `26.04`、`26.10`，推荐 `26.10` |
| Python | `3.10`、`3.11`，推荐 `3.11` |
| Bazel | `5.3.0` |
| 编译后端 | GCVM |
| 目标平台 | Linux x86_64 HCU |

TensorFlow 2.13.1 的本项目构建范围不包含 Python 3.12。

建议构建资源：

```text
CPU：32 核以上
内存：64 GiB 以上
可用磁盘：200 GiB 以上
```

TensorFlow 完整编译可能持续数小时，建议在 `tmux` 或 `screen` 中执行。

## 5. 前置条件

开始编译前，需要确认：

1. HCU 驱动已经正确安装；
2. `/dev/kfd` 和 `/dev/dri` 可用；
3. `/opt/hyhal` 存在且与当前驱动匹配；
4. DTK 已安装到 `/opt/dtk`，或通过 `DTK_DIR` 指定其他安装目录；
5. 容器内已安装 GCC、G++、Git、curl、wget、tar、zip、unzip 等基础构建工具；
6. 可以访问 Python 软件源、Bazel 下载地址和 Bazel 外部依赖地址。

检查 HCU 驱动和设备：

```bash
ls -l /dev/kfd
ls -ld /dev/dri
ls -ld /opt/hyhal
```

## 6. 获取源码

```bash
git clone --branch r2.13-das \
  https://github.com/HYGON-AI/tensorflow-das.git

cd tensorflow-das
```

构建发布产物前，应确认工作区干净并记录准确 Commit：

```bash
git status --short
git rev-parse HEAD
```

## 7. 准备 DTK 环境

默认以 `/opt/dtk` 为安装目录：

```bash
export DTK_DIR=/opt/dtk

test -f "${DTK_DIR}/env.sh"
test -d "${DTK_DIR}/dcc/gcvm"

source "${DTK_DIR}/env.sh"
export ROCM_PATH="${ROCM_PATH:-${DTK_DIR}}"
```

检查 DTK、编译器和 HCU 设备：

```bash
echo "ROCM_PATH=${ROCM_PATH}"
"${ROCM_PATH}/bin/hipcc" --version
"${ROCM_PATH}/bin/rocminfo"
```

DTK 版本可以通过以下文件确认：

```bash
cat "${ROCM_PATH}/.info/version-dev"
```

## 8. 准备 Python 环境

推荐使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install \
  keras_preprocessing \
  packaging==23.1 \
  requests \
  urllib3==1.25 \
  protobuf==3.20.3 \
  'scipy<1.15' \
  portpicker \
  numpy==1.24.3 \
  patchelf==0.19.1.0 \
  auditwheel
```

检查 Python 和 `patchelf`：

```bash
python --version
hash -r
command -v patchelf
patchelf --version
```

要求 `patchelf >= 0.14.5`。

## 9. 安装 Bazel 5.3.0

仓库根目录的 `.bazelversion` 固定为 Bazel 5.3.0。可以使用 Bazelisk 自动读取该版本：

```bash
export BAZELISK_VERSION=1.28.1

curl -fL \
  -o /usr/local/bin/bazel \
  "https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-amd64"

chmod +x /usr/local/bin/bazel
bazel --version
```

如果当前用户没有 `/usr/local/bin` 写权限，可以安装到用户目录并加入 `PATH`：

```bash
mkdir -p "${HOME}/.local/bin"
curl -fL \
  -o "${HOME}/.local/bin/bazel" \
  "https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-amd64"
chmod +x "${HOME}/.local/bin/bazel"
export PATH="${HOME}/.local/bin:${PATH}"
bazel --version
```

## 10. 配置 HCU 目标架构

`TF_ROCM_AMDGPU_TARGETS` 用于 Bazel 仓库配置和设备代码编译。优先使用 DTK
环境中已有的 `TF_ROCM_AMDGPU_TARGETS`、`AMDGPU_TARGETS` 或 `TARGETS`；如果均未
设置，则从当前 HCU 设备自动检测：

```bash
source "${DTK_DIR}/env.sh"

RAW_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS:-${AMDGPU_TARGETS:-${TARGETS:-}}}"

if [ -z "${RAW_AMDGPU_TARGETS}" ]; then
  RAW_AMDGPU_TARGETS="$(
    "${ROCM_PATH}/bin/rocminfo" 2>/dev/null |
      grep -oE 'gfx[0-9a-f]+' |
      grep -v '^gfx000$' |
      sort -u |
      paste -sd, -
  )"
fi

export TF_ROCM_AMDGPU_TARGETS="$(
  printf '%s' "${RAW_AMDGPU_TARGETS}" |
    tr ';[:space:]' ',' |
    sed -E 's/,+/,/g; s/^,//; s/,$//'
)"

test -n "${TF_ROCM_AMDGPU_TARGETS}"
echo "HCU_TARGETS=${TF_ROCM_AMDGPU_TARGETS}"
```

日常开发建议只编译当前机器实际使用的架构，以缩短编译时间。只有需要生成多架构
发布包时，才应显式配置多个逗号分隔的目标，例如：

```bash
export TF_ROCM_AMDGPU_TARGETS=gfx926,gfx928,gfx936,gfx938
```

## 11. 配置 TensorFlow

在仓库根目录执行：

```bash
export PYTHON_BIN_PATH="$(command -v python)"
source "${DTK_DIR}/env.sh"
export ROCM_PATH="${ROCM_PATH:-${DTK_DIR}}"
export TF_NEED_ROCM=1
export TF_NEED_CUDA=0
export USE_DEFAULT_PYTHON_LIB_PATH=1

yes "" | env \
  TF_NEED_ROCM=1 \
  TF_NEED_CUDA=0 \
  ROCM_PATH="${ROCM_PATH}" \
  TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS}" \
  PYTHON_BIN_PATH="${PYTHON_BIN_PATH}" \
  USE_DEFAULT_PYTHON_LIB_PATH=1 \
  ./configure
```

配置完成后会生成 `.tf_configure.bazelrc`。切换 Python、DTK 或目标架构时，应重新
执行 `./configure`，不要直接复用旧环境生成的配置文件。

## 12. 编译并生成 wheel

执行 TensorFlow HCU 编译：

```bash
bazel build \
  --remote_timeout=300 \
  -c opt \
  --config=rocm \
  --config=gcvm \
  --repo_env=TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS}" \
  --verbose_failures \
  //tensorflow/tools/pip_package:build_pip_package
```

如需限制并行编译任务数，可以增加：

```text
--jobs=<并行任务数>
```

如需复用磁盘缓存，可以增加：

```bash
mkdir -p "${HOME}/.cache/tensorflow-bazel/disk"
mkdir -p "${HOME}/.cache/tensorflow-bazel/repository"
```

并在 Bazel 命令中加入：

```text
--disk_cache=${HOME}/.cache/tensorflow-bazel/disk
--repository_cache=${HOME}/.cache/tensorflow-bazel/repository
```

生成 wheel：

```bash
mkdir -p dist
bazel-bin/tensorflow/tools/pip_package/build_pip_package dist
ls -lh dist/*.whl
```

DTK 26.10 对应的版本示例为：

```text
2.13.1+das.opt1.dtk2610
```

实际 wheel 文件名还会包含 Python ABI 和平台标签。

### 12.1 检测并写入 manylinux 平台标签

TensorFlow 构建脚本生成的原始 wheel 通常使用 `linux_x86_64` 平台标签。正式发布前，
先使用 `auditwheel show` 检测 wheel 实际引用的最高 glibc ABI 下限：

```bash
shopt -s nullglob
wheels=(dist/*.whl)

if [ "${#wheels[@]}" -ne 1 ]; then
  echo "Expected exactly one wheel in dist/, found ${#wheels[@]}" >&2
  exit 1
fi

ORIGINAL_WHEEL="${wheels[0]}"
AUDITWHEEL_REPORT="$(auditwheel show "${ORIGINAL_WHEEL}" 2>&1)"
printf '%s\n' "${AUDITWHEEL_REPORT}"
```

输出中应包含类似结论：

```text
This constrains the platform tag to "manylinux_2_28_x86_64".
```

从检查结果中提取平台标签：

```bash
PLATFORM_TAG="$(
  printf '%s\n' "${AUDITWHEEL_REPORT}" |
    sed -nE 's/.*platform tag to "(manylinux_[0-9]+_[0-9]+_x86_64)".*/\1/p' |
    head -n1
)"

if [ -z "${PLATFORM_TAG}" ]; then
  echo "Unable to determine the manylinux platform tag from auditwheel output." >&2
  exit 1
fi

echo "Detected platform tag: ${PLATFORM_TAG}"
```

确认标签符合发布环境要求后，使用 `wheel tags` 同时更新 wheel 文件名和内部
`WHEEL` 元数据。`--remove` 会在新 wheel 生成成功后删除原始 wheel：

```bash
RETAGGED_WHEEL="$(
  python -m wheel tags \
    --remove \
    --platform-tag="${PLATFORM_TAG}" \
    "${ORIGINAL_WHEEL}"
)"

echo "Retagged wheel: ${RETAGGED_WHEEL}"
ls -lh dist/*.whl
sha256sum dist/*.whl
```

> `auditwheel show` 只负责检查 wheel 的 ABI 和外部动态库依赖；`wheel tags` 只修改
> 文件名及内部标签。上述操作不会消除或打包 DTK 外部运行库，也不能单独证明 wheel
> 满足 PyPA manylinux 策略。如果检查报告仍列出 DTK/HCU 外部动态库，必须按项目的
> 发布规范确认这些库由目标环境提供，并确认目标分发渠道接受这种依赖模型。不要仅凭
> 重打标签宣称 wheel 已完成 manylinux 修复。

## 13. 安装和验证

### 13.1 安装 wheel

建议在新的虚拟环境中验证产物，避免误用源码目录或旧版本 TensorFlow：

```bash
source /opt/dtk/env.sh
python3 -m venv /tmp/tensorflow-das-venv
source /tmp/tensorflow-das-venv/bin/activate

python -m pip install --upgrade pip
python -m pip install dist/*.whl
```

### 13.2 HCU smoke test

离开源码目录后执行验证，确保导入的是已安装 wheel：

```bash
cd /tmp
source /opt/dtk/env.sh
source /tmp/tensorflow-das-venv/bin/activate

python - <<'PY'
import numpy as np
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("TensorFlow path:", tf.__file__)
print("Built with ROCm:", tf.test.is_built_with_rocm())
print("Build info:", tf.sysconfig.get_build_info())

devices = tf.config.list_physical_devices("GPU")
print("GPU devices:", devices)

assert tf.test.is_built_with_rocm(), "TensorFlow was not built with HCU support"
assert devices, "HCU/GPU device not found"

with tf.device("/GPU:0"):
    lhs = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    rhs = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    result = tf.matmul(lhs, rhs)

expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)
print("GPU matmul result:", result.numpy())
PY
```

通过条件：

- `tf.__version__` 包含预期的 DAS/DTK 版本后缀；
- `tf.test.is_built_with_rocm()` 返回 `True`；
- 至少识别到一个 HCU/GPU 设备；
- GPU 矩阵乘法结果正确。

### 13.3 运行 HCU 测试

仓库提供了 HCU 验证脚本，默认加载以下两组完整测试清单：

- `scripts/ci/pytest.sh`：1490 条 TensorFlow Python 测试；
- `scripts/ci/ctest.sh`：1115 条 TensorFlow C++ 测试。

默认执行前会根据 `scripts/ci/hcu_test_skip_targets.txt` 跳过已确认的失败
target；显式传给脚本的 target 不受该清单过滤，因此仍可用于单独复测。

脚本支持串行和多进程两种执行模式。未设置 `HCU_TEST_EXECUTION_MODE` 时默认使用
`serial`，逐条独立执行 `bazel test`，适合开发者在本地手动验证：

```bash
cd /path/to/tensorflow-das
source /opt/dtk/env.sh

HCU_VISIBLE_DEVICES=0,1 \
PYTHON_BIN="$(command -v python)" \
bash scripts/ci/run_tensorflow_hcu_tests.sh
```

如需使用多卡多进程测试，设置 `HCU_TEST_EXECUTION_MODE=parallel`。脚本会按照
`HCU_VISIBLE_DEVICES` 的顺序将设备组成双卡槽，并通过
`HCU_TEST_CONCURRENCY` 控制同时运行的测试数量。以下命令使用 8 张卡同时运行
4 个测试，每个测试只会看到其分配到的两张卡：

```bash
HCU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HCU_TEST_EXECUTION_MODE=parallel \
HCU_TEST_CONCURRENCY=4 \
PYTHON_BIN="$(command -v python)" \
bash scripts/ci/run_tensorflow_hcu_tests.sh
```

串行模式会在每条测试结束后直接记录结果；多进程模式会根据 Bazel Build Event
Protocol 汇总结果。两种模式都会生成 `results.tsv` 和 `failed_targets.txt` 供人工
复核，测试失败不会直接令脚本返回非零状态。

如需执行指定 Bazel 测试，可以把目标直接传给脚本：

```bash
bash scripts/ci/run_tensorflow_hcu_tests.sh \
  //tensorflow/python/kernel_tests/nn_ops:conv_ops_test \
  //tensorflow/python/kernel_tests/math_ops:segment_reduction_ops_test
```

## 14. wheel 检查

检查压缩包完整性：

```bash
unzip -t dist/*.whl
```

查看 wheel 元数据：

```bash
unzip -p dist/*.whl '*/WHEEL'
```

检查 ABI 和外部动态库依赖：

```bash
auditwheel show dist/*.whl
```

如果已按照第 12.1 节重打标签，wheel 文件名和内部 `WHEEL` 元数据中应包含同一个
`manylinux_2_xx_x86_64` 标签。重打标签不会消除外部动态库依赖；正式发布前仍应
根据目标分发环境复核 ABI、DTK 运行库依赖和平台标签。

查看 wheel 中已有的 DTK 相关文件：

```bash
unzip -l dist/*.whl |
  grep -Ei 'lib(galaxyhip|gcvm|hipblaslt|hsa-runtime64|hydrm|origami|rccl|rocm_smi64|rocprofiler-register)[^/ ]*' || true
```

## 15. 常见问题

### 15.1 `@local_config_rocm` 或目标架构检测失败

检查以下变量和命令：

```bash
source /opt/dtk/env.sh
echo "ROCM_PATH=${ROCM_PATH:-}"
/opt/dtk/bin/rocminfo | grep -oE 'gfx[0-9a-f]+' | sort -u
```

确认 `TF_ROCM_AMDGPU_TARGETS` 非空、使用逗号分隔，并重新运行 `./configure`。

### 15.2 `offset_counter` 找不到 `liborigami.so.1`

该问题可能出现在 Bazel action 无法继承 DTK 运行环境时。检查：

```bash
grep -n "use_default_shell_env" tensorflow/tensorflow.bzl
```

同时确认：

- DTK 已正确安装到 `/opt/dtk`；
- 执行 `./configure` 和 Bazel 编译前已经加载 `/opt/dtk/env.sh`；
- 当前源码包含 `use_default_shell_env = True` 的相关修复；
- 没有复用其他 DTK 版本生成的 `.tf_configure.bazelrc` 和 Bazel 输出。

### 15.3 TensorFlow 导入成功但找不到 HCU

依次检查：

```bash
ls -l /dev/kfd
ls -ld /dev/dri
ls -ld /opt/hyhal
source /opt/dtk/env.sh

cd /tmp
python -c 'import tensorflow as tf; print(tf.__file__); print(tf.config.list_physical_devices("GPU"))'
```

同时确认：

1. HCU 驱动和 `/opt/hyhal` 状态正常；
2. wheel 与目标 Python、DTK 版本匹配；
3. 当前 Python 环境没有安装其他 TensorFlow 包；
4. 测试时没有从 TensorFlow 源码目录导入本地包；
5. 更换或安装驱动后已经按要求重启主机。

### 15.4 切换 DTK 或目标架构后出现异常

重新加载 DTK 环境，重新执行第 10 节的目标架构配置以及第 11 节的 TensorFlow 配置，
不要复用旧环境生成的 `.tf_configure.bazelrc`。

如果仍然复用了旧的外部仓库配置，可以清理 Bazel 输出后重新编译：

```bash
bazel clean --expunge
```

该命令会删除当前工作区对应的 Bazel 构建缓存，因此只应在切换工具链、DTK 或目标
架构后确有需要时执行。

## 16. 发布前检查清单

- [ ] 记录源码仓库、分支和准确的 40 位 Commit；
- [ ] 工作区没有未纳入发布范围的修改；
- [ ] Python、DTK 和目标架构与发布计划一致；
- [ ] `/opt/dtk/dcc/gcvm` 存在；
- [ ] Bazel 版本为 5.3.0；
- [ ] wheel 文件名包含正确的 TensorFlow、Python 和 DTK 标识；
- [ ] `auditwheel show` 的 ABI 和外部依赖检查完成；
- [ ] `WHEEL` 元数据与 wheel 文件名一致；
- [ ] `unzip -t` 检查通过；
- [ ] 在目标环境中安装成功；
- [ ] `tf.test.is_built_with_rocm()` 返回 `True`；
- [ ] `tf.config.list_physical_devices("GPU")` 能列出 HCU 设备；
- [ ] GPU 矩阵乘法验证通过；
- [ ] HCU 相关 Bazel 测试通过；
- [ ] 构建日志、wheel、测试日志、SHA-256 和源码 Commit 一并归档。
