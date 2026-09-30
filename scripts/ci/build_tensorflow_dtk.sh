#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DTK_DIR="${DTK_DIR:-/opt/dtk}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/dist}"
BAZEL_BIN="${BAZEL_BIN:-bazel}"
TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS:-${HCU_TARGETS:-}}"

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
export TF_NEED_ROCM=1
export TF_NEED_CUDA=0
export PYTHON_BIN_PATH="$(command -v "${PYTHON_BIN}")"
export USE_DEFAULT_PYTHON_LIB_PATH=1

if [[ -z "${TF_ROCM_AMDGPU_TARGETS}" ]]; then
  TF_ROCM_AMDGPU_TARGETS="$(${ROCM_PATH}/bin/rocminfo 2>/dev/null \
    | grep -oE 'gfx[0-9a-f]+' \
    | grep -v '^gfx000$' \
    | sort -u \
    | paste -sd, - || true)"
fi
if [[ -z "${TF_ROCM_AMDGPU_TARGETS}" ]]; then
  echo "ERROR: HCU compile targets are empty and no HCU architecture was detected." >&2
  exit 1
fi
export TF_ROCM_AMDGPU_TARGETS

if [[ ! -x "${ROCM_PATH}/bin/hipcc" ]]; then
  echo "ERROR: hipcc not found: ${ROCM_PATH}/bin/hipcc" >&2
  exit 1
fi

cd "${ROOT_DIR}"
mkdir -p "${OUT_DIR}"

echo "Configuring TensorFlow:"
echo "  source                  ${ROOT_DIR}"
echo "  Python                  ${PYTHON_BIN_PATH}"
echo "  DTK                     ${ROCM_PATH}"
echo "  HCU targets             ${TF_ROCM_AMDGPU_TARGETS}"
echo "  hipcc                   $(${ROCM_PATH}/bin/hipcc --version | head -n1)"

# configure is interactive. Preserve its exit status while avoiding the SIGPIPE
# status produced by `yes` when configure finishes reading its input.
set +o pipefail
if yes "" | env \
    TF_NEED_ROCM=1 \
    TF_NEED_CUDA=0 \
    ROCM_PATH="${ROCM_PATH}" \
    TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS}" \
    PYTHON_BIN_PATH="${PYTHON_BIN_PATH}" \
    USE_DEFAULT_PYTHON_LIB_PATH=1 \
    ./configure; then
  configure_status=0
else
  configure_status="${PIPESTATUS[1]}"
fi
set -o pipefail
if (( configure_status != 0 )); then
  echo "ERROR: TensorFlow configure failed with exit code ${configure_status}." >&2
  exit "${configure_status}"
fi

bazel_startup_args=()
if [[ -n "${BAZEL_OUTPUT_USER_ROOT:-}" ]]; then
  mkdir -p "${BAZEL_OUTPUT_USER_ROOT}"
  bazel_startup_args+=("--output_user_root=${BAZEL_OUTPUT_USER_ROOT}")
fi

bazel_build_args=(
  --remote_timeout=300
  -c opt
  --config=rocm
  --config=gcvm
  "--repo_env=TF_ROCM_AMDGPU_TARGETS=${TF_ROCM_AMDGPU_TARGETS}"
  --verbose_failures
)
if [[ -n "${BAZEL_DISK_CACHE:-}" ]]; then
  mkdir -p "${BAZEL_DISK_CACHE}"
  bazel_build_args+=("--disk_cache=${BAZEL_DISK_CACHE}")
fi
if [[ -n "${BAZEL_REPOSITORY_CACHE:-}" ]]; then
  mkdir -p "${BAZEL_REPOSITORY_CACHE}"
  bazel_build_args+=("--repository_cache=${BAZEL_REPOSITORY_CACHE}")
fi
if [[ -n "${BAZEL_JOBS:-}" ]]; then
  bazel_build_args+=("--jobs=${BAZEL_JOBS}")
fi

"${BAZEL_BIN}" "${bazel_startup_args[@]}" build \
  "${bazel_build_args[@]}" \
  //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package "${OUT_DIR}"

shopt -s nullglob
wheels=("${OUT_DIR}"/*.whl)
if (( ${#wheels[@]} != 1 )); then
  echo "ERROR: expected exactly one TensorFlow wheel in ${OUT_DIR}, found ${#wheels[@]}." >&2
  exit 1
fi

original_wheel="${wheels[0]}"
if ! command -v auditwheel > /dev/null; then
  echo "ERROR: auditwheel is not installed or is not available on PATH." >&2
  exit 1
fi
if ! "${PYTHON_BIN}" -m wheel tags --help > /dev/null; then
  echo "ERROR: the selected Python environment does not provide 'wheel tags'." >&2
  exit 1
fi

auditwheel_report="$(auditwheel show "${original_wheel}" 2>&1)"
printf '%s\n' "${auditwheel_report}" | tee "${OUT_DIR}/auditwheel-show.txt"

platform_tag="$(
  printf '%s\n' "${auditwheel_report}" \
    | sed -nE 's/.*platform tag to "(manylinux_[0-9]+_[0-9]+_x86_64)".*/\1/p' \
    | head -n1
)"
if [[ -z "${platform_tag}" ]]; then
  echo "ERROR: cannot determine the manylinux platform tag from auditwheel output." >&2
  exit 1
fi

echo "Retagging TensorFlow wheel with platform tag ${platform_tag}"
"${PYTHON_BIN}" -m wheel tags \
  --remove \
  "--platform-tag=${platform_tag}" \
  "${original_wheel}"

wheels=("${OUT_DIR}"/*.whl)
if (( ${#wheels[@]} != 1 )); then
  echo "ERROR: expected exactly one retagged wheel in ${OUT_DIR}, found ${#wheels[@]}." >&2
  exit 1
fi

retagged_wheel="${wheels[0]}"
if [[ "$(basename "${retagged_wheel}")" != *"-${platform_tag}.whl" ]]; then
  echo "ERROR: retagged wheel filename does not contain ${platform_tag}: ${retagged_wheel}" >&2
  exit 1
fi
if ! unzip -p "${retagged_wheel}" '*/WHEEL' \
    | grep -F -- "-${platform_tag}" > /dev/null; then
  echo "ERROR: WHEEL metadata does not contain platform tag ${platform_tag}." >&2
  exit 1
fi

unzip -t "${retagged_wheel}"
sha256sum "${retagged_wheel}" | tee "${OUT_DIR}/SHA256SUMS"

echo "TensorFlow wheel build completed:"
echo "  wheel        ${retagged_wheel}"
echo "  platform tag ${platform_tag}"
echo "  audit report ${OUT_DIR}/auditwheel-show.txt"
echo "  checksum     ${OUT_DIR}/SHA256SUMS"
