# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

# DTK: Add for GCVM support
"""TensorFlow workspace initialization. Consult the WORKSPACE on how to use it."""
def workspace():
    native.new_local_repository(
        name = "gcvm_repo",
        path = "/opt/dtk/dcc/gcvm",
    build_file_content = """
cc_library(
    name = "gcvm_lib",
    hdrs = ["include/gcvm.h"],
    srcs = ["lib/libgcvm.so"],
    visibility = ["//visibility:public"],
)
""",
    )

# Alias so it can be loaded without assigning to a different symbol to prevent
# shadowing previous loads and trigger a buildifier warning.
tf_workspace_gcvm = workspace
