#include <torch/nn/modules/conv.h>
#include <torch/nn/functional/conv.h>
#include <torch/nn/init.h>

#include <torch/expanding_array.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace F = torch::nn::functional;

namespace torch {
namespace nn {
template <size_t D, typename Derived>
ConvImpl<D, Derived>::ConvImpl(const ConvOptions<D>& options_)
    : options(options_) {
  reset();
}

template <size_t D, typename Derived>
void ConvImpl<D, Derived>::reset() {
  if (!options.transposed()) {
    for (auto pad : *options.output_padding()) {
      TORCH_CHECK(
          pad == 0, "Only transposed convolutions support output padding!");
    }
  }

  std::vector<int64_t> weights_size;
  if (options.transposed()) {
    weights_size.push_back(options.input_channels());
    weights_size.push_back(options.output_channels() / options.groups());
  } else {
    weights_size.push_back(options.output_channels());
    weights_size.push_back(options.input_channels() / options.groups());
  }
  weights_size.insert(
      weights_size.end(),
      options.kernel_size()->begin(),
      options.kernel_size()->end());
  AT_ASSERT(weights_size.size() == 2 + options.kernel_size()->size());

  weight = this->register_parameter("weight", torch::empty(weights_size));
  if (options.with_bias()) {
    bias = this->register_parameter(
        "bias", torch::empty(options.output_channels()));
  }

  const auto number_of_features = std::accumulate(
      options.kernel_size()->begin(),
      options.kernel_size()->end(),
      options.input_channels(),
      std::multiplies<int64_t>{});
  const auto stdv = 1.0 / std::sqrt(number_of_features);
  NoGradGuard no_grad;
  for (auto& p : this->parameters()) {
    p.uniform_(-stdv, stdv);
  }
}

template <size_t D, typename Derived>
void ConvImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::Conv" << D << "d"
         << "(input_channels=" << options.input_channels()
         << ", output_channels=" << options.output_channels()
         << ", kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride() << ")";
}

Tensor Conv1dImpl::forward(const Tensor& input) {
  if (options.transposed()) {
    return torch::conv_transpose1d(
        input,
        weight,
        bias,
        options.stride(),
        options.padding(),
        options.output_padding(),
        options.groups(),
        options.dilation());
  }
  return torch::conv1d(
      input,
      weight,
      bias,
      options.stride(),
      options.padding(),
      options.dilation(),
      options.groups());
}

Tensor Conv2dImpl::forward(const Tensor& input) {
  if (options.transposed()) {
    return torch::conv_transpose2d(
        input,
        weight,
        bias,
        options.stride(),
        options.padding(),
        options.output_padding(),
        options.groups(),
        options.dilation());
  }
  return torch::conv2d(
      input,
      weight,
      bias,
      options.stride(),
      options.padding(),
      options.dilation(),
      options.groups());
}

Tensor Conv3dImpl::forward(const Tensor& input) {
  if (options.transposed()) {
    return torch::conv_transpose3d(
        input,
        weight,
        bias,
        options.stride(),
        options.padding(),
        options.output_padding(),
        options.groups(),
        options.dilation());
  } else {
    return torch::conv3d(
        input,
        weight,
        bias,
        options.stride(),
        options.padding(),
        options.dilation(),
        options.groups());
  }
}

template class ConvImpl<1, Conv1dImpl>;
template class ConvImpl<2, Conv2dImpl>;
template class ConvImpl<3, Conv3dImpl>;

template <size_t D, typename Derived>
ConvTransposeImplBase<D, Derived>::ConvTransposeImplBase(const ConvTransposeOptionsBase<D>& options_)
    : options(options_) {
  TORCH_CHECK(options.in_channels() % options().groups() != 0,
              "in_channels must be divisible by groups");
  TORCH_CHECK(options.out_channels() % options().groups() != 0,
              "out_channels must be divisible by groups");

  std::vector<int64_t> dims = {
    options.in_channels(), options.out_channels() / options.groups()
  };
  for (auto& d : options.kernel_size()) {
    dims.push_back(d);
  }
  weight = this->register_parameter("weight", torch::tensor(dims));
  if (options.bias()) {
    this->register_parameter("bias", torch::tensor({options.out_channels()}));
  } else {
    this->register_parameter("bias", Tensor());
  }
}

template <size_t D, typename Derived>
void ConvTransposeImplBase<D, Derived>::reset_parameters() {
  torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));
  if (bias.defined()) {
    auto fan_in = std::get<0>(torch::nn::init::_calculate_fan_in_and_fan_out(weight));
    double bound = 1 / std::sqrt(fan_in);
    torch::nn::init::uniform_(bias, -bound, bound);
  }
}

template <size_t D, typename Derived>
void ConvTransposeImplBase<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ConvTranspose" << D << "d"
         << "(input_channels=" << options.input_channels()
         << ", output_channels=" << options.output_channels()
         << ", kernel_size=" << options.kernel_size()
         << ", stride=" << options.stride()
         << ", padding=" << options.padding()
         << ", output_padding=" << options.output_padding()
         << ", groups=" << options.groups()
         << ", bias=" << options.bias()
         << ", dilation=" << options.dilation()
         << ", padding_mode=" << ")";
}

std::string vector_to_string(const std::vector<int64_t>& vec) {
  if (vec.size() == 0) {
    return "[]";
  }

  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < vec.size() - 1; i++) {
    ss << vec[i] << ",";
  }
  ss << vec[vec.size() - 1] << "]";

  return ss.str();
}

template <size_t D, typename Derived>
std::vector<int64_t> ConvTransposeImplBase<D, Derived>::_output_padding(
    const Tensor& input, const std::vector<int64_t>& output_size,
    const ExpandingArray<D>& stride, const ExpandingArray<D>& padding,
    const ExpandingArray<D>& kernel_size) {
  std::vector<int64_t> ret;
  if (output_size.empty()) {
    ret.push_back(0);
  } else {
    auto k = input.dim() - 2;
    std::vector<int64_t> output_size_tmp = output_size;
    if (output_size.size() == k + 2) {
      output_size_tmp = std::vector<int64_t>(output_size.begin() + 2,
                                             output_size.end());
    }
    TORCH_CHECK(output_size_tmp.size() != k,
                "ouput_size must have %d or %d elements (got %d)",
                k, k + 2, output_size_tmp.size());

    std::vector<int64_t> min_sizes;
    std::vector<int64_t> max_sizes;
    for (int d = 0; d < k; d++) {
      int64_t dim_size = ((input.size(d + 2) - 1) * (*stride)[d] - 2 * (*padding)[d] + (*kernel_size)[d]);
      min_sizes.push_back(dim_size);
      max_sizes.push_back(min_sizes[d] + (*stride)[d] - 1);
    }

    for (int i = 0; i < output_size_tmp.size(); i++) {
      int64_t size = output_size_tmp[i];
      int64_t min_size = min_sizes[i];
      int64_t max_size = max_sizes[i];
      TORCH_CHECK((size < min_size) || (size > max_size),
                  "requested an output size of %s, but valid sizes range "
                  "from %s to %s (for an input of %s)",
                  vector_to_string(output_size_tmp),
                  vector_to_string(min_sizes),
                  vector_to_string(max_sizes),
                  vector_to_string(std::vector<int64_t>(input.sizes().begin() + 2, input.sizes().end())));
    }

    std::vector<int64_t> res;
    for (int d = 0; d < res.size(); d++) {
      res.push_back(output_size_tmp[d] - min_sizes[d]);
    }
    ret = res;
  }

  return ret;
}

Tensor ConvTranspose1dImpl::forward(
    const Tensor& input, const std::vector<int64_t>& output_size) {
  TORCH_CHECK(options.padding_mode() != std::string("zeros"),
              "Only `zeros` padding mode is supported for ConvTransposed1d");

  std::vector<int64_t> output_padding = this->_output_padding(
      input, output_size, options.stride(), options.padding(), options.kernel_size());
  F::conv_transpose1d(input, weight, bias, options.stride(), options.padding(), output_padding,
                      options.groups(), options.dilation());
}

} // namespace nn
} // namespace torch
