#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${HCU_DEVICE_PAIRS:?HCU_DEVICE_PAIRS is required}"
: "${HCU_DEVICE_LOCK_DIR:?HCU_DEVICE_LOCK_DIR is required}"

if (( $# == 0 )); then
  echo "ERROR: no test command was provided to the HCU device wrapper." >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "ERROR: flock is required for HCU device allocation." >&2
  exit 2
fi

mkdir -p "${HCU_DEVICE_LOCK_DIR}"
IFS=';' read -r -a device_pairs <<<"${HCU_DEVICE_PAIRS}"
if (( ${#device_pairs[@]} == 0 )); then
  echo "ERROR: HCU_DEVICE_PAIRS does not contain any device pair." >&2
  exit 2
fi

while true; do
  for pair_index in "${!device_pairs[@]}"; do
    pair="${device_pairs[${pair_index}]}"
    exec 9>"${HCU_DEVICE_LOCK_DIR}/pair-${pair_index}.lock"
    if flock -n 9; then
      export ROCR_VISIBLE_DEVICES="${pair}"
      unset HIP_VISIBLE_DEVICES
      echo "[hcu-device-wrapper] pid=$$ pair=${pair} test=$1"
      if "$@"; then
        test_status=0
      else
        test_status=$?
      fi
      flock -u 9
      exit "${test_status}"
    fi
    exec 9>&-
  done
  sleep 0.1
done
