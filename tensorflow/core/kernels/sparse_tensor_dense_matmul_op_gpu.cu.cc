/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
SPDX-License-Identifier: Apache-2.0
Modified by Hygon Information Technology Co., Ltd., 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <mutex>
#include <unordered_map>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/sparse_tensor_dense_matmul_op.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
struct OutOfBoundsValue {
  __host__ __device__ static T value() {
    return Eigen::NumTraits<T>::quiet_NaN();
  }
};

template <typename T>
struct OutOfBoundsValue<std::complex<T>> {
  __host__ __device__ static std::complex<T> value() {
    return std::complex<T>(OutOfBoundsValue<T>::value(),
                           OutOfBoundsValue<T>::value());
  }
};

namespace {

template <typename T>
__global__ void SetScalarKernel(T* dst, T value) {
  *dst = value;
}

template <typename Tindices>
__global__ void ValidatePatternedRowKernel(int nnz, int q, int b_rows,
                                           const Tindices* a_indices,
                                           int* ok) {
  GPU_1D_KERNEL_LOOP(idx, nnz) {
    if (ldg(ok) == 0) {
      continue;
    }
    const int row = idx / q;
    const int pos = idx - row * q;
    const int actual_row = static_cast<int>(ldg(a_indices + 2 * idx));
    const int actual_col = static_cast<int>(ldg(a_indices + 2 * idx + 1));
    const int ref_col = static_cast<int>(ldg(a_indices + 2 * pos + 1));
    if (actual_row != row || actual_col != ref_col ||
        !FastBoundsCheck(actual_col, b_rows)) {
      atomicExch(ok, 0);
    }
  }
}

struct PatternedWaveShflCacheKey {
  const void* indices_ptr;
  int nnz;
  int m;
  int b_rows;
};

struct PatternedWaveShflCacheEntry {
  bool valid;
  int q;
};

struct PatternedWaveShflCacheKeyHash {
  size_t operator()(const PatternedWaveShflCacheKey& key) const {
    size_t h = std::hash<const void*>{}(key.indices_ptr);
    h ^= std::hash<int>{}(key.nnz) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.m) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(key.b_rows) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

inline bool operator==(const PatternedWaveShflCacheKey& a,
                       const PatternedWaveShflCacheKey& b) {
  return a.indices_ptr == b.indices_ptr && a.nnz == b.nnz && a.m == b.m &&
         a.b_rows == b.b_rows;
}

inline std::unordered_map<PatternedWaveShflCacheKey,
                          PatternedWaveShflCacheEntry,
                          PatternedWaveShflCacheKeyHash>&
PatternedWaveShflCache() {
  static auto* cache =
      new std::unordered_map<PatternedWaveShflCacheKey,
                             PatternedWaveShflCacheEntry,
                             PatternedWaveShflCacheKeyHash>();
  return *cache;
}

inline std::mutex& PatternedWaveShflCacheMutex() {
  static auto* mu = new std::mutex();
  return *mu;
}

template <typename Tindices, int kRowsPerBlock, int kTileEntries>
__global__ void SparseTensorDenseMatMulPatternedWaveShflKernel(
    int m, int q, const Tindices* __restrict__ a_indices,
    const float* __restrict__ a_values, const float* __restrict__ b,
    float* __restrict__ out) {
  constexpr int kWaveSize = 64;
  const int tid = threadIdx.x;
  const int wave = tid / kWaveSize;
  const int lane = tid & (kWaveSize - 1);
  const int row = blockIdx.x * kRowsPerBlock + wave;
  if (wave >= kRowsPerBlock || row >= m) {
    return;
  }

  float sum0 = 0.0f;
  float sum1 = 0.0f;
  const int row_offset = row * q;

  for (int base = 0; base < q; base += kTileEntries) {
    const int tile_count = (q - base < kTileEntries) ? (q - base) : kTileEntries;
    int local_k = 0;
    float local_a = 0.0f;
    if (lane < tile_count) {
      local_k = static_cast<int>(ldg(a_indices + 2 * (base + lane) + 1));
      local_a = ldg(a_values + row_offset + base + lane);
    }

#pragma unroll
    for (int t = 0; t < kTileEntries; ++t) {
      if (t >= tile_count) {
        break;
      }
      const int k = __shfl(local_k, t, kWaveSize);
      const float scaled_a = __shfl(local_a, t, kWaveSize);
      sum0 = fmaf(scaled_a, ldg(b + k * 128 + lane), sum0);
      sum1 = fmaf(scaled_a, ldg(b + k * 128 + lane + kWaveSize), sum1);
    }
  }

  float* out_row = out + row * 128;
  out_row[lane] = sum0;
  out_row[lane + kWaveSize] = sum1;
}

template <typename Tindices>
Status GetPatternedWaveShflCacheEntry(OpKernelContext* ctx, const GPUDevice& d,
                                      int nnz, int m, int b_rows,
                                      const Tindices* a_indices,
                                      PatternedWaveShflCacheEntry* entry) {
  const PatternedWaveShflCacheKey key{a_indices, nnz, m, b_rows};
  {
    std::lock_guard<std::mutex> lock(PatternedWaveShflCacheMutex());
    auto it = PatternedWaveShflCache().find(key);
    if (it != PatternedWaveShflCache().end()) {
      *entry = it->second;
      return OkStatus();
    }
  }

  PatternedWaveShflCacheEntry computed{false, 0};
  if (m > 0 && nnz > 0 && nnz % m == 0) {
    computed.q = nnz / m;
    if (computed.q > 0) {
      Tensor ok_t;
      TF_RETURN_IF_ERROR(ctx->allocate_temp(DT_INT32, TensorShape({1}), &ok_t));
      int* ok_ptr = ok_t.flat<int32>().data();

      TF_RETURN_IF_ERROR(
          GpuLaunchKernel(SetScalarKernel<int>, 1, 1, 0, d.stream(), ok_ptr, 1));

      GpuLaunchConfig validate_cfg = GetGpuLaunchConfig(nnz, d);
      TF_RETURN_IF_ERROR(GpuLaunchKernel(
          ValidatePatternedRowKernel<Tindices>, validate_cfg.block_count,
          validate_cfg.thread_per_block, 0, d.stream(), nnz, computed.q, b_rows,
          a_indices, ok_ptr));

      auto* device_context = ctx->op_device_context();
      if (device_context == nullptr || device_context->stream() == nullptr) {
        return errors::Internal(
            "No GPU stream available to validate SparseTensorDenseMatMul pattern");
      }
      auto* stream = device_context->stream();
      int host_ok = 0;
      if (!stream->ThenMemcpy(&host_ok, se::DeviceMemoryBase(ok_ptr, sizeof(int)),
                              sizeof(int))
               .ok()) {
        return errors::Internal(
            "Failed to copy SparseTensorDenseMatMul validation flag");
      }
      TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
      computed.valid = (host_ok != 0);
    }
  }

  {
    std::lock_guard<std::mutex> lock(PatternedWaveShflCacheMutex());
    PatternedWaveShflCache()[key] = computed;
  }
  *entry = computed;
  return OkStatus();
}

template <typename T, typename Tindices>
Status MaybeRunPatternedWaveShflFastPath(OpKernelContext* ctx,
                                         const GPUDevice& d, int nnz, int m,
                                         int b_rows, int p,
                                         const Tindices* a_indices,
                                         const T* a_values, const T* b, T* out,
                                         bool* launched) {
  *launched = false;
  return OkStatus();
}

template <typename Tindices>
Status MaybeRunPatternedWaveShflFastPath(OpKernelContext* ctx,
                                         const GPUDevice& d, int nnz, int m,
                                         int b_rows, int p,
                                         const Tindices* a_indices,
                                         const float* a_values,
                                         const float* b, float* out,
                                         bool* launched) {
  *launched = false;
  if (p != 128 || m <= 0 || nnz <= 0) {
    return OkStatus();
  }

  PatternedWaveShflCacheEntry entry;
  TF_RETURN_IF_ERROR(GetPatternedWaveShflCacheEntry(ctx, d, nnz, m, b_rows,
                                                    a_indices, &entry));
  if (!entry.valid || entry.q <= 0) {
    return OkStatus();
  }

  dim3 threads(256);
  dim3 blocks((m + 4 - 1) / 4);
  TF_RETURN_IF_ERROR(GpuLaunchKernel(
      SparseTensorDenseMatMulPatternedWaveShflKernel<Tindices, 4, 16>, blocks,
      threads, 0, d.stream(), m, entry.q, a_indices, a_values, b, out));
  *launched = true;
  return OkStatus();
}

}  // namespace

template <typename T, typename Tsum, typename Tindices, bool ADJ_A, bool ADJ_B>
__global__ void SparseTensorDenseMatMulKernel(
    int nnz, int m, int b_rows, int b_cols, int p,
    const Tindices* __restrict__ a_indices, const T* __restrict__ a_values,
    const T* __restrict__ b, Tsum* __restrict__ out) {
  // out_{ij} = sum_k {a_ik b_kj}
  // out = A * B', out_{ij} = sum_k {a_ik (b')_kj}; b'_{kj} = b_{jk}
  const int n = (ADJ_B) ? b_cols : b_rows;
  GPU_1D_KERNEL_LOOP(index, nnz * p) {
    const int a_ix = index / p;
    const int j = index % p;
    const int i = ldg(a_indices + 2 * a_ix + ((ADJ_A) ? 1 : 0));
    const int k = ldg(a_indices + 2 * a_ix + ((ADJ_A) ? 0 : 1));
    if (!FastBoundsCheck(i, m)) {
      continue;  // Nowhere to signal an error :(
    }
    // out[i, j]
    Tsum* out_location = out + i * p + j;
    if (!FastBoundsCheck(k, n)) {
      GpuAtomicAdd(out_location, OutOfBoundsValue<Tsum>::value());
      continue;
    }

    // a_value == (ADJ_A) ? conj(a[k, i]) : a[i, k]
    const T a_input = ldg(a_values + a_ix);
    if (a_input == T(0)) {
      continue;
    }
    const T a_value = ADJ_A ? Eigen::numext::conj(a_input) : a_input;

    // b_value == (ADJ_B) ? conj(b[j, k]) : b[k, j]
    const T b_input = ldg(b + ((ADJ_B) ? j * b_cols + k : k * b_cols + j));
    const T b_value = ADJ_B ? Eigen::numext::conj(b_input) : b_input;
    GpuAtomicAdd(out_location,
                 static_cast<Tsum>(a_value) * static_cast<Tsum>(b_value));
  }
}

namespace functor {

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B>
struct SparseTensorDenseMatMulFunctor<GPUDevice, T, Tindices, ADJ_A, ADJ_B> {
  static EIGEN_ALWAYS_INLINE Status
  Compute(OpKernelContext* ctx, typename TTypes<T>::Matrix out,
          typename TTypes<Tindices>::ConstMatrix a_indices,
          typename TTypes<T>::ConstVec a_values,
          typename TTypes<T>::ConstMatrix b) {
    int nnz = a_values.size();
    // out = A * B, A is [m x n] and B is [n x p], out is [m x p]
    int m = out.dimension(0);
    int p = out.dimension(1);
    int b_rows = b.dimension(0);
    int b_cols = b.dimension(1);

    const GPUDevice& d = ctx->eigen_device<GPUDevice>();

    if (OpDeterminismRequired()) {
      return errors::Unimplemented(
          "A deterministic GPU implementation of "
          "SparseTensorDenseMatmulOp is not currently available.");
    }

    if (!ADJ_A && !ADJ_B) {
      bool launched = false;
      TF_RETURN_IF_ERROR(MaybeRunPatternedWaveShflFastPath(
          ctx, d, nnz, m, b_rows, p, a_indices.data(), a_values.data(),
          b.data(), out.data(), &launched));
      if (launched) {
        return OkStatus();
      }
    }

    using Tsum = typename SumType<T>::type;
    Tsum* maybe_temp_out_data = nullptr;
    Tensor temp_out_t;
    bool sum_type_is_different = !std::is_same<T, Tsum>::value;
    if (sum_type_is_different) {
      TF_RETURN_IF_ERROR(ctx->allocate_temp(
          DataTypeToEnum<Tsum>::value,
          TensorShape({out.dimension(0), out.dimension(1)}), &temp_out_t));
      auto temp_out = temp_out_t.matrix<Tsum>();
      maybe_temp_out_data = temp_out.data();
      temp_out.device(d) = temp_out.constant(Tsum(0));
    } else {
      // Note: The reinterpret cast is only required to avoid a compilation
      // error; it is only used if Tsum == T.
      maybe_temp_out_data = reinterpret_cast<Tsum*>(out.data());
      out.device(d) = out.constant(T(0));
    }

    // TODO(ebrevdo): Should this be alpha * nnz instead of
    // out.size()?  Perhaps p * nnz ?
    GpuLaunchConfig config = GetGpuLaunchConfig(p * nnz, d);

    TF_CHECK_OK(GpuLaunchKernel(
        SparseTensorDenseMatMulKernel<T, Tsum, Tindices, ADJ_A, ADJ_B>,
        config.block_count, config.thread_per_block, 0, d.stream(), nnz, m,
        b_rows, b_cols, p, a_indices.data(), a_values.data(), b.data(),
        maybe_temp_out_data));

    if (sum_type_is_different) {
      out.device(d) = temp_out_t.matrix<Tsum>().template cast<T>();
    }

    return OkStatus();
  }
};

}  // namespace functor

#define DEFINE(T, Tindices)                                \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, false, false>;               \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, false, true>;                \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, true, false>;                \
  template struct functor::SparseTensorDenseMatMulFunctor< \
      GPUDevice, T, Tindices, true, true>;

#define DEFINE_ALL_INDEX_TYPES(T) \
  DEFINE(T, int32);               \
  DEFINE(T, int64)

DEFINE_ALL_INDEX_TYPES(Eigen::half);
DEFINE_ALL_INDEX_TYPES(float);
DEFINE_ALL_INDEX_TYPES(double);
DEFINE_ALL_INDEX_TYPES(complex64);
DEFINE_ALL_INDEX_TYPES(complex128);

#undef DEFINE_ALL_INDEX_TYPES
#undef DEFINE

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
