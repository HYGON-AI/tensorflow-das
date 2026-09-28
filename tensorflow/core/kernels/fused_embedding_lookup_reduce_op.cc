/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.
Copyright (c) 2026 Hygon Information Technology Co., Ltd.
SPDX-License-Identifier: Apache-2.0

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

#include "tensorflow/core/kernels/fused_embedding_lookup_reduce_op.h"

#include <cmath>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace {

template <typename T, typename IdType, bool kFlatIds>
class FusedEmbeddingLookupReduceOp : public OpKernel {
 public:
  explicit FusedEmbeddingLookupReduceOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ParseFusedEmbeddingLookupReducer(ctx, &reducer_));
    if constexpr (kFlatIds) {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_rows", &num_rows_));
      OP_REQUIRES_OK(ctx, ctx->GetAttr("num_cols", &num_cols_));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& params = ctx->input(0);
    const Tensor& ids = ctx->input(1);

    int64_t batch_size = 0;
    int64_t width = 0;
    if constexpr (kFlatIds) {
      OP_REQUIRES_OK(ctx, ValidateFusedEmbeddingLookupReduceFlatInputs(
                              ctx, params, ids, num_rows_, num_cols_));
      batch_size = num_rows_;
      width = num_cols_;
    } else {
      OP_REQUIRES_OK(ctx,
                     ValidateFusedEmbeddingLookupReduceInputs(ctx, params, ids));
      batch_size = ids.dim_size(0);
      width = ids.dim_size(1);
    }

    const int64_t vocab_size = params.dim_size(0);
    const int64_t inner_size = params.NumElements() / vocab_size;

    auto ids_flat = ids.flat<IdType>();
    for (int64_t i = 0; i < ids.NumElements(); ++i) {
      const IdType id = ids_flat(i);
      OP_REQUIRES(ctx, FastBoundsCheck(id, vocab_size),
                  errors::InvalidArgument("ids[", i, "] = ", id,
                                          " is out of range [0, ", vocab_size,
                                          ")"));
    }

    TensorShape output_shape = params.shape();
    output_shape.set_dim(0, batch_size);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    if (batch_size == 0 || inner_size == 0) {
      return;
    }

    auto params_flat = params.flat_inner_dims<T>();
    auto output_flat = output->flat_inner_dims<T>();

    using AccumT = FusedEmbeddingLookupAccumTypeT<T>;
    AccumT scale = AccumT{1};
    if (width == 0) {
      output_flat.setZero();
      return;
    }
    if (reducer_ == FusedEmbeddingLookupReducer::kMean) {
      scale = AccumT{1} / static_cast<AccumT>(width);
    } else if (reducer_ == FusedEmbeddingLookupReducer::kSqrtn) {
      scale = AccumT{1} / static_cast<AccumT>(std::sqrt(width));
    }

    auto worker_threads = *(ctx->device()->tensorflow_cpu_worker_threads());
    const int64_t cost_per_row = std::max<int64_t>(1, width * inner_size);
    const T* params_base = params_flat.data();
    T* output_base = output_flat.data();
    const IdType* ids_base = ids_flat.data();

    Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
          cost_per_row, [&](int64_t start, int64_t limit) {
            for (int64_t row = start; row < limit; ++row) {
              const IdType* row_ids = ids_base + row * width;
              T* row_out = output_base + row * inner_size;
              for (int64_t k = 0; k < inner_size; ++k) {
                AccumT acc = AccumT{0};
                for (int64_t j = 0; j < width; ++j) {
                  const int64_t id = static_cast<int64_t>(row_ids[j]);
                  acc += static_cast<AccumT>(params_base[id * inner_size + k]);
                }
                if (reducer_ != FusedEmbeddingLookupReducer::kSum) {
                  acc *= scale;
                }
                row_out[k] = static_cast<T>(acc);
              }
            }
          });
  }

 private:
  FusedEmbeddingLookupReducer reducer_;
  int64_t num_rows_ = 0;
  int64_t num_cols_ = 0;
};

#define REGISTER_CPU_KERNEL(OP_NAME, T, IdType, FlatIds)                    \
  REGISTER_KERNEL_BUILDER(Name(OP_NAME)                                     \
                              .Device(DEVICE_CPU)                           \
                              .TypeConstraint<T>("T")                      \
                              .TypeConstraint<IdType>("Tids"),             \
                          FusedEmbeddingLookupReduceOp<T, IdType, FlatIds>);

#define REGISTER_CPU_KERNELS_FOR_IDS(T)                                     \
  REGISTER_CPU_KERNEL("FusedEmbeddingLookupReduce", T, int32, false)       \
  REGISTER_CPU_KERNEL("FusedEmbeddingLookupReduce", T, int64_t, false)     \
  REGISTER_CPU_KERNEL("FusedEmbeddingLookupReduceFlat", T, int32, true)    \
  REGISTER_CPU_KERNEL("FusedEmbeddingLookupReduceFlat", T, int64_t, true)

TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS_FOR_IDS);

#undef REGISTER_CPU_KERNELS_FOR_IDS
#undef REGISTER_CPU_KERNEL

}  // namespace
}  // namespace tensorflow
