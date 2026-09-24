# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
import os
UNKNOWN = "Unknown"

def sha_value(tf_root):
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tf_root)
            .decode("ascii")
            .strip()
        )
    except Exception:
        return UNKNOWN

def abi_value():
    try:
        return (
            subprocess.check_output("echo '#include <string>' | gcc -x c++ -E -dM - | fgrep _GLIBCXX_USE_CXX11_ABI", shell=True)
            .decode('ascii')
            .strip()[-1]
        )
    except Exception:
        return UNKNOWN

def dtk_version_value():
    try:
       dtk_path=os.getenv('ROCM_PATH')
       dtk_version_path = os.path.join(dtk_path, '.info', "version-dev")
       with open(dtk_version_path, 'r',encoding='utf-8') as file:
           version = file.readline().strip().split("-", 1)[0]
       return "dtk" + version.replace(".", "")
    except Exception:
       return UNKNOWN

def tfwhl_name():
    try:
        tf_root = Path(__file__).parent
        sha =  sha_value(tf_root)[0:7]
        abi = "abi" + abi_value()
        dtk_version = dtk_version_value()
        whl_name = "das" + "." + "opt1" +"." + dtk_version
        return whl_name
    except Exception:
        return UNKNOWN

def dcu_version():
    try:
        tf_version = '2.13.1'
        dcu_version = tf_version + tfwhl_name()
        return dcu_version
    except Exception:
        return UNKNOWN
def replace_name(file_path, marker, newline):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    matches = [i for i, line in enumerate(lines)
               if line.lstrip().startswith(marker)]
    if len(matches) != 1:
        raise ValueError(f"Expected one occurrence of {marker} in {file_path}")
    lines[matches[0]] = newline + '\n'
    with open(file_path, 'w') as file:
        file.writelines(lines)

if __name__ == "__main__":
    replace_name(file_path="./tensorflow/python/client/pywrap_tf_session.py", marker="__hcu_version__ =", newline=f'__hcu_version__ = "2.13.1+{tfwhl_name()}"')
    replace_name(file_path="./tensorflow/tools/pip_package/setup.py", marker="version=_VERSION.replace(", newline=f"    version=_VERSION.replace('-', '')+str('+{tfwhl_name()}'),")
