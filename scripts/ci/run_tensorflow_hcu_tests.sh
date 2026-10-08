#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DTK_DIR="${DTK_DIR:-/opt/dtk}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
BAZEL_BIN="${BAZEL_BIN:-bazel}"
HCU_VISIBLE_DEVICES="${HCU_VISIBLE_DEVICES:-0}"
TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS:-${HCU_TARGETS:-}}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/test_logs/tensorflow-hcu}"
PYTHON_TEST_COMMANDS_FILE="${HCU_PYTHON_TEST_COMMANDS_FILE:-${HCU_TEST_COMMANDS_FILE:-${ROOT_DIR}/scripts/ci/pytest_failed.sh}}"
EXPECTED_PYTHON_TEST_COMMAND_COUNT="${HCU_EXPECTED_PYTHON_TEST_COMMAND_COUNT:-${HCU_EXPECTED_TEST_COMMAND_COUNT:-48}}"
CTEST_COMMANDS_FILE="${HCU_CTEST_COMMANDS_FILE:-${ROOT_DIR}/scripts/ci/ctest.sh}"
EXPECTED_CTEST_COMMAND_COUNT="${HCU_EXPECTED_CTEST_COMMAND_COUNT:-1115}"

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

make_test_command() {
  printf 'bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 %s' "$1"
}

test_commands=()
test_suites=()
test_sources=()
target_source=""

append_test_commands() {
  local suite="$1"
  local commands_file="$2"
  local expected_count="$3"
  local -a file_commands=()

  if [[ ! -f "${commands_file}" ]]; then
    echo "ERROR: Bazel test command file not found: ${commands_file}" >&2
    exit 1
  fi
  if [[ -n "${expected_count}" ]] &&
      [[ ! "${expected_count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: expected command count for ${suite} must be a positive integer or empty." >&2
    exit 1
  fi

  mapfile -t file_commands < <(
    sed \
      -e '/^[[:space:]]*#/d' \
      -e '/^[[:space:]]*$/d' \
      "${commands_file}"
  )
  if [[ -n "${expected_count}" ]] &&
      (( ${#file_commands[@]} != expected_count )); then
    echo "ERROR: expected ${expected_count} ${suite} Bazel commands, found ${#file_commands[@]}." >&2
    exit 1
  fi

  for command in "${file_commands[@]}"; do
    test_commands+=("${command}")
    test_suites+=("${suite}")
  done
  test_sources+=("${suite}: ${commands_file}")
}

if (( $# > 0 )); then
  for target in "$@"; do
    test_commands+=("$(make_test_command "${target}")")
    test_suites+=("custom")
  done
  target_source="command-line arguments"
elif [[ -n "${HCU_BAZEL_TEST_TARGETS:-}" ]]; then
  read -r -a requested_targets <<<"${HCU_BAZEL_TEST_TARGETS}"
  for target in "${requested_targets[@]}"; do
    test_commands+=("$(make_test_command "${target}")")
    test_suites+=("custom")
  done
  target_source="HCU_BAZEL_TEST_TARGETS"
else
  append_test_commands \
    "python-retry" \
    "${PYTHON_TEST_COMMANDS_FILE}" \
    "${EXPECTED_PYTHON_TEST_COMMAND_COUNT}"
  append_test_commands \
    "ctest" \
    "${CTEST_COMMANDS_FILE}" \
    "${EXPECTED_CTEST_COMMAND_COUNT}"
  target_source="${test_sources[*]}"
fi

if (( ${#test_commands[@]} == 0 )); then
  echo "ERROR: no Bazel test targets were selected." >&2
  exit 1
fi
test_targets=()
declare -A seen_targets=()
for index in "${!test_commands[@]}"; do
  command="${test_commands[${index}]}"
  suite="${test_suites[${index}]}"
  read -r -a command_parts <<<"${command}"
  if (( ${#command_parts[@]} < 3 )) ||
      [[ "${command_parts[0]}" != "bazel" ]] ||
      [[ "${command_parts[1]}" != "test" ]]; then
    echo "ERROR: unsupported test command: ${command}" >&2
    exit 1
  fi

  target="${command_parts[${#command_parts[@]} - 1]}"
  if [[ "${target}" != //* ]]; then
    echo "ERROR: test command does not end with a Bazel target: ${command}" >&2
    exit 1
  fi
  if [[ -n "${seen_targets[${target}]+present}" ]]; then
    echo "ERROR: duplicate target found while loading ${suite}: ${target}" >&2
    exit 1
  fi
  seen_targets["${target}"]=1
  test_targets+=("${target}")
done

bazel_startup_args=()
if [[ -n "${BAZEL_OUTPUT_USER_ROOT:-}" ]]; then
  mkdir -p "${BAZEL_OUTPUT_USER_ROOT}"
  bazel_startup_args+=("--output_user_root=${BAZEL_OUTPUT_USER_ROOT}")
fi

selected_targets_file="${LOG_DIR}/selected_targets.txt"
printf 'suite\ttarget\n' > "${selected_targets_file}"
for index in "${!test_targets[@]}"; do
  printf '%s\t%s\n' \
    "${test_suites[${index}]}" \
    "${test_targets[${index}]}" >> "${selected_targets_file}"
done

bazel_test_args=(
  --remote_timeout=300
  --test_output=errors
  "--repo_env=TF_ROCM_AMDGPU_TARGETS=${TF_ROCM_AMDGPU_TARGETS:-}"
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

results_file="${LOG_DIR}/results.tsv"
failed_targets_file="${LOG_DIR}/failed_targets.txt"
printf 'suite\tstatus\telapsed_seconds\ttarget\n' > "${results_file}"
: > "${failed_targets_file}"

echo "Running ${#test_commands[@]} independent Bazel tests from ${target_source}"
echo "Selected target list: ${selected_targets_file}"

cd "${ROOT_DIR}"
failed_count=0
declare -A suite_total_counts=()
declare -A suite_failed_counts=()
for index in "${!test_commands[@]}"; do
  command="${test_commands[${index}]}"
  suite="${test_suites[${index}]}"
  target="${test_targets[${index}]}"
  read -r -a command_parts <<<"${command}"
  suite_total_counts["${suite}"]=$(( ${suite_total_counts["${suite}"]:-0} + 1 ))

  echo
  echo "[$((index + 1))/${#test_commands[@]}][${suite}] Testing ${target}"
  start_seconds="${SECONDS}"
  if "${BAZEL_BIN}" "${bazel_startup_args[@]}" test \
      "${bazel_test_args[@]}" \
      "${command_parts[@]:2}"; then
    status="PASS"
  else
    test_status=$?
    status="FAIL(${test_status})"
    failed_count=$((failed_count + 1))
    suite_failed_counts["${suite}"]=$(( ${suite_failed_counts["${suite}"]:-0} + 1 ))
    printf '%s\t%s\n' "${suite}" "${target}" >> "${failed_targets_file}"
  fi
  elapsed_seconds=$((SECONDS - start_seconds))
  printf '%s\t%s\t%s\t%s\n' \
    "${suite}" "${status}" "${elapsed_seconds}" "${target}" | tee -a "${results_file}"
done

echo "Test suite summary:"
declare -A reported_suites=()
for suite in "${test_suites[@]}"; do
  if [[ -n "${reported_suites[${suite}]+present}" ]]; then
    continue
  fi
  reported_suites["${suite}"]=1
  echo "  ${suite}: ${suite_total_counts[${suite}]} total, ${suite_failed_counts[${suite}]:-0} failed"
done

if (( failed_count > 0 )); then
  echo "WARNING: ${failed_count} of ${#test_commands[@]} Bazel tests failed and require manual review." >&2
  echo "Failed target list: ${failed_targets_file}" >&2
  sed 's/^/  /' "${failed_targets_file}" >&2
  echo "::warning title=TensorFlow HCU test failures::${failed_count} Bazel tests failed; review the uploaded failed_targets.txt artifact."
else
  echo "All ${#test_commands[@]} TensorFlow HCU tests passed."
fi

echo "TensorFlow HCU test execution completed."
exit 0
