#pragma once

#include <torch/nn/options/activation.h>
#include <torch/types.h>

namespace torch {
namespace nn{
namespace functional {

inline Tensor elu(Tensor& input, const ELUOptions& options) {
  if (options.inplace()) {
    return torch::elu_(input, options.alpha());
  } else {
    return torch::elu(input, options.alpha());
  }
}

inline Tensor hardshrink(const Tensor& input,
                         const HardshrinkOptions& options) {
  return torch::hardshrink(input, options.lambda());
}

inline Tensor hardtanh(Tensor& input, const HardtanhOptions& options) {
  if (options.inplace()) {
    return torch::hardtanh_(input, options.min_val(), options.max_val());
  } else {
    return torch::hardtanh(input, options.min_val(), options.max_val());
  }
}

inline Tensor leaky_relu(Tensor& input, const LeakyReLUOptions& options) {
  if (options.inplace()) {
    return torch::leaky_relu_(input, options.negative_slope());
  } else {
    return torch::leaky_relu(input, options.negative_slope());
  }
}

inline Tensor logsigmoid(const Tensor& input) {
  return torch::log_sigmoid(input);
}

inline Tensor softmin(const Tensor& input, const SoftminOptions& options) {
  int dim = options.dim();
  torch::Dtype dtype = options.dtype();

  if (dim == -1) {
    int input_dim = input.dim();
    if (input_dim == 0 || input_dim == 1 || input_dim == 3) {
      dim = 0;
    } else {
      dim = 1;
    }
  }
  if (dtype == torch::Dtype::Undefined) {
    return (-input).softmax(dim);
  } else {
    return (-input).softmax(dim, dtype);
  }
}

} // namespace functional
} // namespace nn
} // namespace torch
