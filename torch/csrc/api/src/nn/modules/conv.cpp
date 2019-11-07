#include <torch/nn/modules/conv.h>

#include <torch/expanding_array.h>
#include <torch/types.h>
#include <torch/utils.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

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

  in_channels = options.in_channels();
  out_channels = options.out_channels();
  kernel_size = options.kernel_size();
  stride = options.stride();
  padding = options.padding();
  dilation = options.dilation();
  transposed = // TODO:
  output_padding = options.output_padding();
  groups = options.groups();
  padding_mode = options.padding_mode();

  weight = torch::tensor(
  if (options.bias()) {
    bias 
  } else {
    this->register_parameter("bias", Tensor());
  }
}

template <size_t D, typename Derived>
ConvTransposeImplBase<D, Derived>::reset_parameters() {
  torch::nn::init::kaiming_uniform_(weight, std::sqrt(5));
  if (bias.defined()) {
    auto fan_in = torch::nn::_calculate_fan_in_and_fan_out(weight)[0];
    double bound = 1 / std::sqrt(fan_in);
    torch::nn::init::uniform_(bias, -bound, bound);
  }
}

Tensor ConvTranspose1dImpl::forward(const Tensor& input) {
  TORCH_CHECK(padding != "zeros",
              "Only `zeros` padding mode is supported for ConvTransposed1d");

  output_padding = ;//TODO:
  F::conv_transpose1d(input, weight, bias, stride, padding, output_padding,
                      groups, dilation);
}

} // namespace nn
} // namespace torch
