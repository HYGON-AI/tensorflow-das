/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/fused_embedding_lookup_reduce_op.h"

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace {

constexpr size_t kMaxSharedIdsBytes = 48 * 1024;
constexpr int kVec4TileWidth = 32;
constexpr int kVec4RowsPerBlock = 4;
#if TENSORFLOW_USE_ROCM
constexpr int kPreferredWaveSize = 64;
#else
constexpr int kPreferredWaveSize = 32;
#endif
constexpr int kFloat2RowsPerBlock = 4;

inline int SelectGpuTileSize(int64_t n) {
  if (n >= 256) return 256;
  if (n >= 128) return 128;
  if (n >= 64) return 64;
  return 32;
}

template <int kStaticWidth, typename AccumT, typename T, typename IdType>
__device__ __forceinline__ AccumT AccumulateLookupReduceScalar(
    const T* __restrict__ params, const IdType* __restrict__ row_ids,
    int64_t width, int64_t inner_size, int64_t col) {
  AccumT acc = AccumT{0};
  if constexpr (kStaticWidth > 0) {
#pragma unroll
    for (int j = 0; j < kStaticWidth; ++j) {
      if (j < width) {
        const int64_t id = static_cast<int64_t>(row_ids[j]);
        acc += static_cast<AccumT>(params[id * inner_size + col]);
      }
    }
  } else {
    for (int64_t j = 0; j < width; ++j) {
      const int64_t id = static_cast<int64_t>(row_ids[j]);
      acc += static_cast<AccumT>(params[id * inner_size + col]);
    }
  }
  return acc;
}

template <int kStaticWidth, typename T, typename IdType, typename AccumT>
__global__ void FusedEmbeddingLookupReduceKernel(const T* __restrict__ params,
                                                 const IdType* __restrict__ ids,
                                                 int64_t batch_size,
                                                 int64_t width,
                                                 int64_t inner_size,
                                                 AccumT scale,
                                                 T* __restrict__ output) {
  const int64_t total = batch_size * inner_size;
  GPU_1D_KERNEL_LOOP(index, total) {
    const int64_t row = index / inner_size;
    const int64_t col = index - row * inner_size;
    const IdType* row_ids = ids + row * width;
    const AccumT acc = AccumulateLookupReduceScalar<kStaticWidth, AccumT>(
        params, row_ids, width, inner_size, col);
    output[index] = static_cast<T>(acc * scale);
  }
}

template <typename T, typename IdType, typename AccumT>
__global__ void FusedEmbeddingLookupReduceTiledKernelNoShared(
    const T* __restrict__ params, const IdType* __restrict__ ids,
    int64_t batch_size, int64_t width, int64_t inner_size, AccumT scale,
    T* __restrict__ output) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t col =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= batch_size || col >= inner_size) {
    return;
  }
  const IdType* row_ids = ids + row * width;
  const AccumT acc = AccumulateLookupReduceScalar<0, AccumT>(
      params, row_ids, width, inner_size, col);
  output[row * inner_size + col] = static_cast<T>(acc * scale);
}

template <int kStaticWidth, typename T, typename IdType, typename AccumT>
__global__ void FusedEmbeddingLookupReduceTiledKernel(
    const T* __restrict__ params, const IdType* __restrict__ ids,
    int64_t batch_size, int64_t width, int64_t inner_size, AccumT scale,
    T* __restrict__ output) {
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(IdType), unsigned char, shared_mem);
  IdType* const shared_ids = reinterpret_cast<IdType*>(shared_mem);

  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size) {
    return;
  }

  const IdType* row_ids = ids + row * width;
  for (int64_t j = threadIdx.x; j < width; j += blockDim.x) {
    shared_ids[j] = row_ids[j];
  }
  __syncthreads();

  const int64_t col =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (col >= inner_size) {
    return;
  }

  const AccumT acc = AccumulateLookupReduceScalar<kStaticWidth, AccumT>(
      params, shared_ids, width, inner_size, col);
  output[row * inner_size + col] = static_cast<T>(acc * scale);
}

template <int kStaticWidth, typename IdType>
__device__ __forceinline__ float4 AccumulateLookupReduceVec4(
    const float4* __restrict__ params_vec, const IdType* __restrict__ row_ids,
    int64_t width, int64_t vec_inner_size, int64_t vec_col) {
  float4 acc = {0.f, 0.f, 0.f, 0.f};
  if constexpr (kStaticWidth > 0) {
#pragma unroll
    for (int j = 0; j < kStaticWidth; ++j) {
      if (j < width) {
        const int64_t id = static_cast<int64_t>(row_ids[j]);
        const float4 value = params_vec[id * vec_inner_size + vec_col];
        acc.x += value.x;
        acc.y += value.y;
        acc.z += value.z;
        acc.w += value.w;
      }
    }
  } else {
    for (int64_t j = 0; j < width; ++j) {
      const int64_t id = static_cast<int64_t>(row_ids[j]);
      const float4 value = params_vec[id * vec_inner_size + vec_col];
      acc.x += value.x;
      acc.y += value.y;
      acc.z += value.z;
      acc.w += value.w;
    }
  }
  return acc;
}

template <int kStaticWidth, typename IdType>
__global__ void FusedEmbeddingLookupReduceVec4KernelNoShared(
    const float* __restrict__ params, const IdType* __restrict__ ids,
    int64_t batch_size, int64_t width, int64_t vec_inner_size, float scale,
    float* __restrict__ output) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  const int64_t vec_col =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (row >= batch_size || vec_col >= vec_inner_size) {
    return;
  }

  const float4* const params_vec = reinterpret_cast<const float4*>(params);
  const IdType* const row_ids_any = ids + row * width;
  const float4 acc = AccumulateLookupReduceVec4<kStaticWidth, IdType>(
      params_vec, row_ids_any, width, vec_inner_size, vec_col);

  float4* const output_vec = reinterpret_cast<float4*>(output);
  float4 out_value = {acc.x * scale, acc.y * scale, acc.z * scale,
                      acc.w * scale};
  output_vec[row * vec_inner_size + vec_col] = out_value;
}

template <int kStaticWidth, typename IdType>
__global__ void FusedEmbeddingLookupReduceVec4Kernel(
    const float* __restrict__ params, const IdType* __restrict__ ids,
    int64_t batch_size, int64_t width, int64_t vec_inner_size, float scale,
    float* __restrict__ output) {
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(IdType), unsigned char, shared_mem);
  IdType* const shared_ids_any = reinterpret_cast<IdType*>(shared_mem);

  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= batch_size) {
    return;
  }

  const IdType* const row_ids_any = ids + row * width;
  for (int64_t j = threadIdx.x; j < width; j += blockDim.x) {
    shared_ids_any[j] = row_ids_any[j];
  }
  __syncthreads();

  const int64_t vec_col =
      static_cast<int64_t>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (vec_col >= vec_inner_size) {
    return;
  }

  const float4* const params_vec = reinterpret_cast<const float4*>(params);
  const float4 acc = AccumulateLookupReduceVec4<kStaticWidth, IdType>(
      params_vec, shared_ids_any, width, vec_inner_size, vec_col);

  float4* const output_vec = reinterpret_cast<float4*>(output);
  float4 out_value = {acc.x * scale, acc.y * scale, acc.z * scale,
                      acc.w * scale};
  output_vec[row * vec_inner_size + vec_col] = out_value;
}

template <int kStaticWidth, typename IdType>
__global__ void FusedEmbeddingLookupReduceVec4RowBlockKernel(
    const float* __restrict__ params, const IdType* __restrict__ ids,
    int64_t batch_size, int64_t width, int64_t vec_inner_size, float scale,
    float* __restrict__ output) {
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(IdType), unsigned char, shared_mem);
  IdType* const shared_ids = reinterpret_cast<IdType*>(shared_mem);

  const int lane = threadIdx.x & (kVec4TileWidth - 1);
  const int local_row = threadIdx.x / kVec4TileWidth;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kVec4RowsPerBlock + local_row;

  if (row < batch_size) {
    IdType* const row_shared_ids = shared_ids + local_row * width;
    const IdType* const row_ids = ids + row * width;
    for (int64_t j = lane; j < width; j += kVec4TileWidth) {
      row_shared_ids[j] = row_ids[j];
    }
  }
  __syncthreads();

  const int64_t vec_col =
      static_cast<int64_t>(blockIdx.y) * kVec4TileWidth + lane;
  if (row >= batch_size || vec_col >= vec_inner_size) {
    return;
  }

  const float4* const params_vec = reinterpret_cast<const float4*>(params);
  const float4 acc = AccumulateLookupReduceVec4<kStaticWidth, IdType>(
      params_vec, shared_ids + local_row * width, width, vec_inner_size,
      vec_col);

  float4* const output_vec = reinterpret_cast<float4*>(output);
  float4 out_value = {acc.x * scale, acc.y * scale, acc.z * scale,
                      acc.w * scale};
  output_vec[row * vec_inner_size + vec_col] = out_value;
}

template <int kStaticWidth, typename IdType>
__global__ void FusedEmbeddingLookupReduceFloat2WaveKernel(
    const float* __restrict__ params, const IdType* __restrict__ ids,
    int64_t batch_size, int64_t width, int64_t pair_inner_size, float scale,
    float* __restrict__ output) {
  GPU_DYNAMIC_SHARED_MEM_DECL(sizeof(IdType), unsigned char, shared_mem);
  IdType* const shared_ids = reinterpret_cast<IdType*>(shared_mem);

  const int lane = threadIdx.x & (kPreferredWaveSize - 1);
  const int local_row = threadIdx.x / kPreferredWaveSize;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * kFloat2RowsPerBlock + local_row;

  if (row < batch_size) {
    IdType* const row_shared_ids = shared_ids + local_row * width;
    const IdType* const row_ids = ids + row * width;
    for (int64_t j = lane; j < width; j += kPreferredWaveSize) {
      row_shared_ids[j] = row_ids[j];
    }
  }
  __syncthreads();

  const int64_t pair_col =
      static_cast<int64_t>(blockIdx.y) * kPreferredWaveSize + lane;
  if (row >= batch_size || pair_col >= pair_inner_size) {
    return;
  }

  const float2* const params_pairs = reinterpret_cast<const float2*>(params);
  float2 acc = {0.f, 0.f};
  const IdType* const row_ids = shared_ids + local_row * width;
  if constexpr (kStaticWidth > 0) {
#pragma unroll
    for (int j = 0; j < kStaticWidth; ++j) {
      if (j < width) {
        const int64_t id = static_cast<int64_t>(row_ids[j]);
        const float2 value = params_pairs[id * pair_inner_size + pair_col];
        acc.x += value.x;
        acc.y += value.y;
      }
    }
  } else {
    for (int64_t j = 0; j < width; ++j) {
      const int64_t id = static_cast<int64_t>(row_ids[j]);
      const float2 value = params_pairs[id * pair_inner_size + pair_col];
      acc.x += value.x;
      acc.y += value.y;
    }
  }

  float2* const output_pairs = reinterpret_cast<float2*>(output);
  float2 out_value = {acc.x * scale, acc.y * scale};
  output_pairs[row * pair_inner_size + pair_col] = out_value;
}

template <int kStaticWidth, typename T, typename IdType, typename AccumT>
Status LaunchScalarKernel(const GPUDevice& device, const T* params_ptr,
                          const IdType* ids_ptr, int64_t batch_size,
                          int64_t width, int64_t inner_size, AccumT scale,
                          T* output_ptr, bool use_shared_ids) {
  const int block_size = SelectGpuTileSize(inner_size);
  const dim3 block_dim(block_size, 1, 1);
  const dim3 grid_dim(static_cast<unsigned int>(batch_size),
                      static_cast<unsigned int>((inner_size + block_size - 1) /
                                                block_size),
                      1);
  if (use_shared_ids) {
    return GpuLaunchKernel(
        FusedEmbeddingLookupReduceTiledKernel<kStaticWidth, T, IdType, AccumT>,
        grid_dim, block_dim, width * sizeof(IdType), device.stream(),
        params_ptr, ids_ptr, batch_size, width, inner_size, scale, output_ptr);
  }
  return GpuLaunchKernel(
      FusedEmbeddingLookupReduceKernel<kStaticWidth, T, IdType, AccumT>,
      grid_dim.x * grid_dim.y, block_dim.x, 0, device.stream(), params_ptr,
      ids_ptr, batch_size, width, inner_size, scale, output_ptr);
}

template <int kStaticWidth, typename T, typename IdType, typename AccumT>
Status DispatchScalarKernel(const GPUDevice& device, const T* params_ptr,
                            const IdType* ids_ptr, int64_t batch_size,
                            int64_t width, int64_t inner_size, AccumT scale,
                            T* output_ptr, bool use_shared_ids) {
  if (use_shared_ids) {
    return LaunchScalarKernel<kStaticWidth, T, IdType, AccumT>(
        device, params_ptr, ids_ptr, batch_size, width, inner_size, scale,
        output_ptr, true);
  }
  const int64_t total = batch_size * inner_size;
  GpuLaunchConfig config = GetGpuLaunchConfig(total, device);
  return GpuLaunchKernel(
      FusedEmbeddingLookupReduceKernel<kStaticWidth, T, IdType, AccumT>,
      config.block_count, config.thread_per_block, 0, device.stream(),
      params_ptr, ids_ptr, batch_size, width, inner_size, scale, output_ptr);
}

template <typename T, typename IdType, typename AccumT>
Status LaunchScalarKernelByWidth(const GPUDevice& device, const T* params_ptr,
                                 const IdType* ids_ptr, int64_t batch_size,
                                 int64_t width, int64_t inner_size,
                                 AccumT scale, T* output_ptr,
                                 bool use_shared_ids) {
  if (width <= 8) {
    return DispatchScalarKernel<8, T, IdType, AccumT>(
        device, params_ptr, ids_ptr, batch_size, width, inner_size, scale,
        output_ptr, use_shared_ids);
  }
  if (width <= 16) {
    return DispatchScalarKernel<16, T, IdType, AccumT>(
        device, params_ptr, ids_ptr, batch_size, width, inner_size, scale,
        output_ptr, use_shared_ids);
  }
  if (width <= 32) {
    return DispatchScalarKernel<32, T, IdType, AccumT>(
        device, params_ptr, ids_ptr, batch_size, width, inner_size, scale,
        output_ptr, use_shared_ids);
  }
  return DispatchScalarKernel<0, T, IdType, AccumT>(
      device, params_ptr, ids_ptr, batch_size, width, inner_size, scale,
      output_ptr, use_shared_ids);
}

template <int kStaticWidth, typename IdType>
Status DispatchVec4Kernel(const GPUDevice& device, const float* params_ptr,
                          const IdType* ids_ptr, int64_t batch_size,
                          int64_t width, int64_t vec_inner_size, float scale,
                          float* output_ptr, bool use_shared_ids) {
  if (use_shared_ids) {
    const dim3 block_dim(kVec4TileWidth * kVec4RowsPerBlock, 1, 1);
    const dim3 grid_dim(
        static_cast<unsigned int>((batch_size + kVec4RowsPerBlock - 1) /
                                  kVec4RowsPerBlock),
        static_cast<unsigned int>((vec_inner_size + kVec4TileWidth - 1) /
                                  kVec4TileWidth),
        1);
    return GpuLaunchKernel(
        FusedEmbeddingLookupReduceVec4RowBlockKernel<kStaticWidth, IdType>,
        grid_dim,
        block_dim, width * sizeof(IdType), device.stream(), params_ptr, ids_ptr,
        batch_size, width, vec_inner_size, scale, output_ptr);
  }
  const int block_size = SelectGpuTileSize(vec_inner_size);
  const dim3 block_dim(block_size, 1, 1);
  const dim3 grid_dim(static_cast<unsigned int>(batch_size),
                      static_cast<unsigned int>((vec_inner_size + block_size -
                                                1) /
                                                block_size),
                      1);
  return GpuLaunchKernel(
      FusedEmbeddingLookupReduceVec4KernelNoShared<kStaticWidth, IdType>,
      grid_dim, block_dim, 0, device.stream(), params_ptr, ids_ptr, batch_size,
      width, vec_inner_size, scale, output_ptr);
}

template <typename IdType>
Status LaunchVec4KernelByWidth(const GPUDevice& device, const float* params_ptr,
                               const IdType* ids_ptr, int64_t batch_size,
                               int64_t width, int64_t vec_inner_size,
                               float scale, float* output_ptr,
                               bool use_shared_ids) {
  if (width <= 8) {
    return DispatchVec4Kernel<8, IdType>(device, params_ptr, ids_ptr,
                                         batch_size, width, vec_inner_size,
                                         scale, output_ptr, use_shared_ids);
  }
  if (width <= 16) {
    return DispatchVec4Kernel<16, IdType>(device, params_ptr, ids_ptr,
                                          batch_size, width, vec_inner_size,
                                          scale, output_ptr, use_shared_ids);
  }
  if (width <= 32) {
    return DispatchVec4Kernel<32, IdType>(device, params_ptr, ids_ptr,
                                          batch_size, width, vec_inner_size,
                                          scale, output_ptr, use_shared_ids);
  }
  return DispatchVec4Kernel<0, IdType>(device, params_ptr, ids_ptr, batch_size,
                                       width, vec_inner_size, scale, output_ptr,
                                       use_shared_ids);
}

template <int kStaticWidth, typename IdType>
Status DispatchFloat2WaveKernel(const GPUDevice& device, const float* params_ptr,
                                const IdType* ids_ptr, int64_t batch_size,
                                int64_t width, int64_t pair_inner_size,
                                float scale, float* output_ptr) {
  const dim3 block_dim(kPreferredWaveSize * kFloat2RowsPerBlock, 1, 1);
  const dim3 grid_dim(
      static_cast<unsigned int>((batch_size + kFloat2RowsPerBlock - 1) /
                                kFloat2RowsPerBlock),
      static_cast<unsigned int>((pair_inner_size + kPreferredWaveSize - 1) /
                                kPreferredWaveSize),
      1);
  return GpuLaunchKernel(
      FusedEmbeddingLookupReduceFloat2WaveKernel<kStaticWidth, IdType>,
      grid_dim, block_dim, width * sizeof(IdType), device.stream(), params_ptr,
      ids_ptr, batch_size, width, pair_inner_size, scale, output_ptr);
}

template <typename IdType>
Status LaunchFloat2WaveKernelByWidth(const GPUDevice& device,
                                     const float* params_ptr,
                                     const IdType* ids_ptr, int64_t batch_size,
                                     int64_t width, int64_t pair_inner_size,
                                     float scale, float* output_ptr) {
  if (width <= 8) {
    return DispatchFloat2WaveKernel<8, IdType>(
        device, params_ptr, ids_ptr, batch_size, width, pair_inner_size, scale,
        output_ptr);
  }
  if (width <= 16) {
    return DispatchFloat2WaveKernel<16, IdType>(
        device, params_ptr, ids_ptr, batch_size, width, pair_inner_size, scale,
        output_ptr);
  }
  if (width <= 32) {
    return DispatchFloat2WaveKernel<32, IdType>(
        device, params_ptr, ids_ptr, batch_size, width, pair_inner_size, scale,
        output_ptr);
  }
  return DispatchFloat2WaveKernel<0, IdType>(
      device, params_ptr, ids_ptr, batch_size, width, pair_inner_size, scale,
      output_ptr);
}

template <typename T, typename IdType>
class GPUFusedEmbeddingLookupReduceOp : public OpKernel {
 public:
  explicit GPUFusedEmbeddingLookupReduceOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ParseFusedEmbeddingLookupReducer(ctx, &reducer_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& params = ctx->input(0);
    const Tensor& ids = ctx->input(1);
    OP_REQUIRES_OK(ctx,
                   ValidateFusedEmbeddingLookupReduceInputs(ctx, params, ids));

    const int64_t vocab_size = params.dim_size(0);
    const int64_t batch_size = ids.dim_size(0);
    const int64_t width = ids.dim_size(1);
    const int64_t inner_size = params.NumElements() / vocab_size;

    TensorShape output_shape = params.shape();
    output_shape.set_dim(0, batch_size);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (batch_size == 0 || inner_size == 0) {
      return;
    }
    if (width == 0) {
      auto out = output->flat<T>();
      out.device(ctx->eigen_gpu_device()) = out.constant(T{0});
      return;
    }

    using AccumT = FusedEmbeddingLookupAccumTypeT<T>;
    AccumT scale = AccumT{1};
    if (reducer_ == FusedEmbeddingLookupReducer::kMean) {
      scale = AccumT{1} / static_cast<AccumT>(width);
    } else if (reducer_ == FusedEmbeddingLookupReducer::kSqrtn) {
      scale = AccumT{1} / static_cast<AccumT>(std::sqrt(width));
    }

    const GPUDevice& device = ctx->eigen_gpu_device();
    const T* const params_ptr = params.flat<T>().data();
    const IdType* const ids_ptr = ids.flat<IdType>().data();
    T* const output_ptr = output->flat<T>().data();
    const bool use_shared_ids = width * sizeof(IdType) <= kMaxSharedIdsBytes;

    if constexpr (std::is_same<T, float>::value) {
      const uintptr_t params_addr = reinterpret_cast<uintptr_t>(params_ptr);
      const uintptr_t output_addr = reinterpret_cast<uintptr_t>(output_ptr);
      if (use_shared_ids && inner_size % 2 == 0 &&
          params_addr % alignof(float2) == 0 &&
          output_addr % alignof(float2) == 0 && inner_size <= 256) {
        OP_REQUIRES_OK(
            ctx, LaunchFloat2WaveKernelByWidth<IdType>(
                     device, params_ptr, ids_ptr, batch_size, width,
                     inner_size / 2, static_cast<float>(scale), output_ptr));
        return;
      }
      if (inner_size % 4 == 0 && params_addr % alignof(float4) == 0 &&
          output_addr % alignof(float4) == 0) {
        OP_REQUIRES_OK(
            ctx, LaunchVec4KernelByWidth<IdType>(
                     device, params_ptr, ids_ptr, batch_size, width,
                     inner_size / 4, static_cast<float>(scale), output_ptr,
                     use_shared_ids));
        return;
      }
    }

    OP_REQUIRES_OK(ctx, LaunchScalarKernelByWidth<T, IdType, AccumT>(
                            device, params_ptr, ids_ptr, batch_size, width,
                            inner_size, scale, output_ptr, use_shared_ids));
  }

 private:
  FusedEmbeddingLookupReducer reducer_;
};

#define REGISTER_GPU_KERNEL(T, IdType)                                      \
  REGISTER_KERNEL_BUILDER(Name("FusedEmbeddingLookupReduce")               \
                              .Device(DEVICE_GPU)                           \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<IdType>("Tids"),             \
                          GPUFusedEmbeddingLookupReduceOp<T, IdType>);

#define REGISTER_GPU_KERNELS_FOR_IDS(T) \
  REGISTER_GPU_KERNEL(T, int32)         \
  REGISTER_GPU_KERNEL(T, int64_t)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_KERNELS_FOR_IDS);

#undef REGISTER_GPU_KERNELS_FOR_IDS
#undef REGISTER_GPU_KERNEL

}  // namespace
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
