#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Provision the DTK and Bazel environment used by TensorFlow HCU CI.
# TensorFlow 2.13.1 uses the GCVM backend provided by DTK.

set -euo pipefail

DTK_VERSION="${DTK_VERSION:-26.04.2}"
DTK_DIR="${DTK_DIR:-/opt/dtk}"
DTK_DOWNLOAD_BASE_URL="${DTK_DOWNLOAD_BASE_URL:-https://download.sourcefind.cn:65024/file/1}"
BAZELISK_VERSION="${BAZELISK_VERSION:-1.28.1}"
TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS:-}"

WGET_OPTS=(--timeout=30 --tries=3 -q)

if [[ "$(id -u)" -ne 0 ]]; then
  echo "ERROR: this setup script must run as root." >&2
  exit 1
fi

normalize_targets() {
  printf '%s' "$1" \
    | tr ';[:space:]' ',,' \
    | sed -E 's/,+/,/g; s/^,//; s/,$//'
}

install_dtk() {
  local archive="/tmp/dtk-${DTK_VERSION}.tar.gz"
  local url="${DTK_DOWNLOAD_BASE_URL%/}/DTK-${DTK_VERSION}/Rocky8.6/DTK-${DTK_VERSION}-Rocky8.6-x86_64.tar.gz"

  echo "Installing DTK ${DTK_VERSION} from ${url}"
  wget "${WGET_OPTS[@]}" "${url}" -O "${archive}"
  mkdir -p "${DTK_DIR}"
  tar -xzf "${archive}" -C "${DTK_DIR}" --strip-components=1
  rm -f "${archive}"

  if [[ ! -f "${DTK_DIR}/env.sh" ]]; then
    echo "ERROR: ${DTK_DIR}/env.sh is missing after DTK extraction." >&2
    exit 1
  fi
  if [[ ! -x "${DTK_DIR}/bin/hipcc" ]]; then
    echo "ERROR: ${DTK_DIR}/bin/hipcc is missing after DTK extraction." >&2
    exit 1
  fi
}

install_bazelisk() {
  local url="https://github.com/bazelbuild/bazelisk/releases/download/v${BAZELISK_VERSION}/bazelisk-linux-amd64"

  echo "Installing Bazelisk ${BAZELISK_VERSION}"
  wget "${WGET_OPTS[@]}" "${url}" -O /usr/local/bin/bazel
  chmod +x /usr/local/bin/bazel
}

detect_hcu_targets() {
  "${DTK_DIR}/bin/rocminfo" 2>/dev/null \
    | grep -oE 'gfx[0-9a-f]+' \
    | grep -v '^gfx000$' \
    | sort -u \
    | paste -sd, -
}

list_tensile_libpaths() {
  local dir names
  echo "Tensile library directories under ${DTK_DIR}/lib/hipblaslt:" >&2
  for dir in "${DTK_DIR}"/lib/hipblaslt/library*/ "${DTK_DIR}"/lib/hipblaslt/; do
    [[ -d "${dir}" ]] || continue
    names="$(cd "${dir}" && compgen -G 'TensileLibrary_*.dat' | tr '\n' ' ')"
    echo "  ${dir%/}: ${names:-<no TensileLibrary_*.dat>}" >&2
  done
}

find_tensile_libpath() {
  local arch="$1"
  local dir
  for dir in "${DTK_DIR}"/lib/hipblaslt/library*/ "${DTK_DIR}"/lib/hipblaslt/; do
    [[ -d "${dir}" ]] || continue
    [[ -f "${dir}/TensileLibrary_${arch}.dat" ]] || continue
    printf '%s\n' "${dir%/}"
    return 0
  done
  return 1
}

pin_hipblaslt_tensile_libpath() {
  local hipblaslt_dir="${DTK_DIR}/lib/hipblaslt"
  local first_arch="${TF_ROCM_AMDGPU_TARGETS%%,*}"
  local libpath="${HIPBLASLT_TENSILE_LIBPATH:-}"

  if [[ ! -d "${hipblaslt_dir}" ]]; then
    echo "WARNING: ${hipblaslt_dir} does not exist; skipping hipBLASLt setup." >&2
    return 0
  fi

  if [[ -z "${first_arch}" ]]; then
    echo "ERROR: cannot select a hipBLASLt library without an HCU target." >&2
    exit 1
  fi

  if [[ -n "${libpath}" ]]; then
    if [[ ! -f "${libpath}/TensileLibrary_${first_arch}.dat" ]]; then
      echo "ERROR: ${libpath} has no TensileLibrary_${first_arch}.dat." >&2
      list_tensile_libpaths
      exit 1
    fi
  elif ! libpath="$(find_tensile_libpath "${first_arch}")"; then
    echo "ERROR: no TensileLibrary_${first_arch}.dat was found for DTK ${DTK_VERSION}." >&2
    list_tensile_libpaths
    exit 1
  elif [[ "$(basename "${libpath}")" == "library_${first_arch}" ]]; then
    echo "Using the DTK per-architecture hipBLASLt library: ${libpath}"
    return 0
  fi

  export HIPBLASLT_TENSILE_LIBPATH="${libpath}"
  printf 'export HIPBLASLT_TENSILE_LIBPATH=%q\n' "${HIPBLASLT_TENSILE_LIBPATH}" \
    > "${DTK_DIR}/env.hipblaslt"
}

write_github_environment() {
  [[ -n "${GITHUB_ENV:-}" ]] || return 0
  {
    echo "ROCM_PATH=${DTK_DIR}"
    echo "HCU_TARGETS=${TF_ROCM_AMDGPU_TARGETS}"
    if [[ -n "${HIPBLASLT_TENSILE_LIBPATH:-}" ]]; then
      echo "HIPBLASLT_TENSILE_LIBPATH=${HIPBLASLT_TENSILE_LIBPATH}"
    fi
  } >> "${GITHUB_ENV}"
}

install_dtk
install_bazelisk

set +u
source "${DTK_DIR}/env.sh"
set -u
export ROCM_PATH="${DTK_DIR}"

if [[ -n "${TF_ROCM_AMDGPU_TARGETS}" ]]; then
  TF_ROCM_AMDGPU_TARGETS="$(normalize_targets "${TF_ROCM_AMDGPU_TARGETS}")"
else
  TF_ROCM_AMDGPU_TARGETS="$(detect_hcu_targets || true)"
fi
if [[ -z "${TF_ROCM_AMDGPU_TARGETS}" ]]; then
  echo "ERROR: no HCU architecture was configured or detected." >&2
  exit 1
fi
export TF_ROCM_AMDGPU_TARGETS

pin_hipblaslt_tensile_libpath
write_github_environment

echo "TensorFlow HCU CI environment ready:"
echo "  DTK                    ${DTK_DIR} (${DTK_VERSION})"
echo "  hipcc                  $(command -v hipcc)"
echo "  Bazelisk               /usr/local/bin/bazel (${BAZELISK_VERSION})"
echo "  hipBLASLt library      ${HIPBLASLT_TENSILE_LIBPATH:-DTK default}"
