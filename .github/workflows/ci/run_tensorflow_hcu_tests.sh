#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DTK_DIR="${DTK_DIR:-/opt/dtk}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
BAZEL_BIN="${BAZEL_BIN:-bazel}"
HCU_VISIBLE_DEVICES="${HCU_VISIBLE_DEVICES:-0}"
TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS:-${HCU_TARGETS:-}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/test_logs/tensorflow-hcu}"

if [[ ! -f "${DTK_DIR}/env.sh" ]]; then
  echo "ERROR: DTK environment not found: ${DTK_DIR}/env.sh" >&2
  exit 1
fi

set +u
source "${DTK_DIR}/env.sh"
[[ ! -f "${DTK_DIR}/env.hipblaslt" ]] || source "${DTK_DIR}/env.hipblaslt"
set -u
unset PYTHONPATH

export ROCM_PATH="${ROCM_PATH:-${DTK_DIR}}"
export HIP_VISIBLE_DEVICES="${HCU_VISIBLE_DEVICES}"
export ROCR_VISIBLE_DEVICES="${HCU_VISIBLE_DEVICES}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-0}"

device_count="$(tr ',' '\n' <<<"${HCU_VISIBLE_DEVICES}" | sed '/^[[:space:]]*$/d' | wc -l)"
if (( device_count == 0 || device_count > 2 )); then
  echo "ERROR: HCU_VISIBLE_DEVICES must expose one or two devices, got: ${HCU_VISIBLE_DEVICES}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/run.log") 2>&1

echo "TensorFlow HCU test environment:"
echo "  Python               $(command -v "${PYTHON_BIN}")"
echo "  DTK                  ${ROCM_PATH}"
echo "  HIP_VISIBLE_DEVICES  ${HIP_VISIBLE_DEVICES}"
echo "  HCU targets          ${TF_ROCM_AMDGPU_TARGETS:-<not set>}"

echo "Running installed-wheel HCU smoke test"
(cd /tmp && "${PYTHON_BIN}" - <<'PY')
import numpy as np
import tensorflow as tf

print("tensorflow version:", tf.__version__)
print("tensorflow path:", tf.__file__)
print("built with ROCm:", tf.test.is_built_with_rocm())

if not tf.test.is_built_with_rocm():
  raise RuntimeError("The installed TensorFlow wheel was not built with ROCm/HCU support.")

gpus = tf.config.list_physical_devices("GPU")
print("visible GPUs:", gpus)
if not gpus:
  raise RuntimeError("TensorFlow did not detect an HCU GPU.")

with tf.device("/GPU:0"):
  lhs = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  rhs = tf.constant([[5.0, 6.0], [7.0, 8.0]])
  result = tf.matmul(lhs, rhs)

expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5, atol=1e-5)
print("GPU matmul result:", result.numpy())
PY

if (( $# > 0 )); then
  test_targets=("$@")
elif [[ -n "${HCU_BAZEL_TEST_TARGETS:-}" ]]; then
  read -r -a test_targets <<<"${HCU_BAZEL_TEST_TARGETS}"
else
  test_targets=(
    //tensorflow/python/kernel_tests/nn_ops:conv_ops_test
    //tensorflow/python/kernel_tests/math_ops:segment_reduction_ops_test
  )
fi

if (( ${#test_targets[@]} == 0 )); then
  echo "ERROR: no Bazel test targets were selected." >&2
  exit 1
fi

bazel_startup_args=()
if [[ -n "${BAZEL_OUTPUT_USER_ROOT:-}" ]]; then
  mkdir -p "${BAZEL_OUTPUT_USER_ROOT}"
  bazel_startup_args+=("--output_user_root=${BAZEL_OUTPUT_USER_ROOT}")
fi

bazel_test_args=(
  --remote_timeout=300
  -c opt
  --config=rocm
  --config=gcvm
  --test_output=errors
  --test_timeout=1800
  --keep_going
  "--repo_env=TF_ROCM_AMDGPU_TARGETS=${TF_ROCM_AMDGPU_TARGETS:-}"
  "--test_env=HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}"
  "--test_env=ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES}"
  "--test_env=ROCM_PATH=${ROCM_PATH}"
  --test_env=LD_LIBRARY_PATH
)
if [[ -n "${HIPBLASLT_TENSILE_LIBPATH:-}" ]]; then
  bazel_test_args+=("--test_env=HIPBLASLT_TENSILE_LIBPATH=${HIPBLASLT_TENSILE_LIBPATH}")
fi
if [[ -n "${BAZEL_DISK_CACHE:-}" ]]; then
  mkdir -p "${BAZEL_DISK_CACHE}"
  bazel_test_args+=("--disk_cache=${BAZEL_DISK_CACHE}")
fi
if [[ -n "${BAZEL_REPOSITORY_CACHE:-}" ]]; then
  mkdir -p "${BAZEL_REPOSITORY_CACHE}"
  bazel_test_args+=("--repository_cache=${BAZEL_REPOSITORY_CACHE}")
fi
if [[ -n "${BAZEL_JOBS:-}" ]]; then
  bazel_test_args+=("--jobs=${BAZEL_JOBS}")
fi

echo "Running Bazel tests:"
printf '  %s\n' "${test_targets[@]}"

cd "${ROOT_DIR}"
"${BAZEL_BIN}" "${bazel_startup_args[@]}" test \
  "${bazel_test_args[@]}" \
  "${test_targets[@]}"

echo "TensorFlow HCU tests completed successfully."
