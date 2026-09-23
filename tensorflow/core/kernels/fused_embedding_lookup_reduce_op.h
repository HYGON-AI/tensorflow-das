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

#ifndef TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_LOOKUP_REDUCE_OP_H_
#define TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_LOOKUP_REDUCE_OP_H_

#include <string>
#include <type_traits>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

enum class FusedEmbeddingLookupReducer : int32 {
  kSum = 0,
  kMean = 1,
  kSqrtn = 2,
};

inline Status ParseFusedEmbeddingLookupReducer(
    OpKernelConstruction* ctx, FusedEmbeddingLookupReducer* reducer) {
  std::string combiner;
  TF_RETURN_IF_ERROR(ctx->GetAttr("combiner", &combiner));
  if (combiner == "sum") {
    *reducer = FusedEmbeddingLookupReducer::kSum;
  } else if (combiner == "mean") {
    *reducer = FusedEmbeddingLookupReducer::kMean;
  } else if (combiner == "sqrtn") {
    *reducer = FusedEmbeddingLookupReducer::kSqrtn;
  } else {
    return errors::InvalidArgument("Unsupported combiner: ", combiner);
  }
  return OkStatus();
}

inline Status ValidateFusedEmbeddingLookupReduceInputs(OpKernelContext* ctx,
                                                       const Tensor& params,
                                                       const Tensor& ids) {
  if (params.dims() < 2) {
    return errors::InvalidArgument("params must have rank >= 2, got rank ",
                                   params.dims());
  }
  if (!TensorShapeUtils::IsMatrix(ids.shape())) {
    return errors::InvalidArgument("ids must be rank 2, got rank ",
                                   ids.dims());
  }
  if (params.dim_size(0) <= 0) {
    return errors::InvalidArgument("params.shape[0] must be > 0, got ",
                                   params.dim_size(0));
  }
  return OkStatus();
}

template <typename T>
struct FusedEmbeddingLookupAccumType {
  using type = T;
};

template <>
struct FusedEmbeddingLookupAccumType<Eigen::half> {
  using type = float;
};

template <>
struct FusedEmbeddingLookupAccumType<bfloat16> {
  using type = float;
};

template <typename T>
using FusedEmbeddingLookupAccumTypeT =
    typename FusedEmbeddingLookupAccumType<T>::type;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_FUSED_EMBEDDING_LOOKUP_REDUCE_OP_H_
