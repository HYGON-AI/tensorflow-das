#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# Temporary diagnostic list for the 18 targets that remained failed in the
# parallel HCU test run. Remove this file after the memory-limit investigation.

bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python:gradients_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python:gradients_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/eager:ops_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/data_structures:stage_op_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/linalg:linear_operator_block_lower_triangular_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/linalg:linear_operator_circulant_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/linalg:matrix_triangular_solve_op_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/linalg:matrix_triangular_solve_op_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/linalg/sparse:csr_sparse_matrix_dense_mat_mul_grad_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/math_ops:batch_matmul_op_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/math_ops:batch_matmul_op_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/linalg/sparse:conjugate_gradient_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/distributions:dirichlet_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/nn_ops:cudnn_deterministic_ops_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/ops/parallel_for:math_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/distributions:multinomial_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/linalg:linear_operator_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/training:momentum_test_cpu
