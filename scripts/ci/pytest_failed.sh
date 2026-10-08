#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
#
# TensorFlow Python Bazel tests that failed during the initial full CI run.
# CI reruns these commands independently and reports all remaining failures
# after the complete run.

bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python:collective_ops_gpu_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python:collective_ops_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python:sparse_ops_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/data/experimental/kernel_tests/service:data_service_ops_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/debug/examples/v2:examples_v2_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:combinations_test_2gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:moving_averages_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:multi_process_runner_test_2gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:multi_process_runner_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:multi_process_runner_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:multi_worker_continuous_run_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:sharded_variable_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:strategy_gather_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute/v1:cross_device_ops_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute/v1:cross_device_ops_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/eager/polymorphic_function:quarantine_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/framework:config_vgpu_test_2gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/framework:config_vgpu_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/framework:config_vgpu_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/framework:extension_type_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/grappler:cluster_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/grappler:cluster_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/grappler:cost_analyzer_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests:collective_ops_test_2gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests:collective_ops_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/array_ops:batch_scatter_ops_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/data_structures:map_ops_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/sparse_ops:sparse_slice_op_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:combinations_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python:collective_ops_gpu_test_2gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python:collective_ops_gpu_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/data/experimental/kernel_tests/service:multi_device_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/debug/lib:session_debug_multi_gpu_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:moving_averages_test_2gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:moving_averages_test_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:strategy_gather_test_2gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:strategy_gather_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute/v1:cross_device_ops_test_2gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests:collective_ops_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/kernel_tests/data_structures:lookup_ops_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/ops/ragged:ragged_tensor_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/autograph/tests:loop_basic_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/data/kernel_tests:sample_from_datasets_test
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/debug/lib:session_debug_multi_gpu_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:multi_worker_continuous_run_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/distribute:strategy_gather_test_xla_gpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/grappler:remapper_test_cpu
bazel test -c opt --config=gcvm --config=rocm --test_sharding_strategy=disabled --test_timeout=1080 //tensorflow/python/grappler:remapper_test_gpu
