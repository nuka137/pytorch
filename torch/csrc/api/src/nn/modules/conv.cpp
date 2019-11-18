#include <torch/nn/functional/conv.h>
#include <torch/nn/functional/padding.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/functional/conv.h>
#include <torch/nn/init.h>

#include <torch/expanding_array.h>
#include <torch/nn/init.h>
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
Conv1dImpl::Conv1dImpl(
    ConvOptions<1> options_)
    : ConvImpl(options_.transposed(false).output_padding(0)) {}

Tensor Conv1dImpl::forward(const Tensor& input) {
  if (c10::get_if<enumtype::kCircular>(&options.padding_mode())) {
    std::vector<int64_t> expanded_padding = {((*options.padding())[0] + 1) / 2, (*options.padding())[0] / 2};
    return F::detail::conv1d(
      F::detail::pad(input, expanded_padding, torch::kCircular, 0),
      weight, bias,
      options.stride(),
      /*padding=*/0,
      options.dilation(),
      options.groups());
  }
  return F::detail::conv1d(
    input,
    weight,
    bias,
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

Conv2dImpl::Conv2dImpl(
    ConvOptions<2> options_)
    : ConvImpl(options_.transposed(false).output_padding(0)) {}

Tensor Conv2dImpl::forward(const Tensor& input) {
  if (c10::get_if<enumtype::kCircular>(&options.padding_mode())) {
    std::vector<int64_t> expanded_padding = {
      ((*options.padding())[1] + 1) / 2, (*options.padding())[1] / 2,
      ((*options.padding())[0] + 1) / 2, (*options.padding())[0] / 2};
    return F::detail::conv2d(
      F::detail::pad(input, expanded_padding, torch::kCircular, 0),
      weight, bias,
      options.stride(),
      /*padding=*/0,
      options.dilation(),
      options.groups());
  }
  return F::detail::conv2d(
    input,
    weight,
    bias,
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

Conv3dImpl::Conv3dImpl(
    ConvOptions<3> options_)
    : ConvImpl(options_.transposed(false).output_padding(0)) {}

Tensor Conv3dImpl::forward(const Tensor& input) {
  if (c10::get_if<enumtype::kCircular>(&options.padding_mode())) {
    std::vector<int64_t> expanded_padding = {
      ((*options.padding())[2] + 1) / 2, (*options.padding())[2] / 2,
      ((*options.padding())[1] + 1) / 2, (*options.padding())[1] / 2,
      ((*options.padding())[0] + 1) / 2, (*options.padding())[0] / 2};
    return F::detail::conv3d(
      F::detail::pad(input, expanded_padding, torch::kCircular, 0),
      weight, bias,
      options.stride(),
      /*padding=*/0,
      options.dilation(),
      options.groups());
  }
  return F::detail::conv3d(
    input,
    weight,
    bias,
    options.stride(),
    options.padding(),
    options.dilation(),
    options.groups());
}

template class ConvImpl<1, Conv1dImpl>;
template class ConvImpl<2, Conv2dImpl>;
template class ConvImpl<3, Conv3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
ConvTransposeImpl<D, Derived>::ConvTransposeImpl(
    ConvTransposeOptions<D> options_) : ConvImpl<D, Derived>(options_.transposed(true)) {}

template <size_t D, typename Derived>
void ConvTransposeImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ConvTranspose" << D << "d"
         << "(" << this->options.in_channels()
         << ", " << this->options.out_channels()
         << ", kernel_size=" << this->options.kernel_size()
         << ", stride=" << this->options.stride();
  if (*this->options.padding() != *ExpandingArray<D>(0)) {
    stream << ", padding=" << this->options.padding();
  }
  if (*this->options.dilation() != *ExpandingArray<D>(1)) {
    stream << ", dilation=" << this->options.dilation();
  }
  if (*this->options.output_padding() != *ExpandingArray<D>(0)) {
    stream << ", output_padding=" << this->options.output_padding();
  }
  if (this->options.groups() != 1) {
    stream << ", groups=" << this->options.groups();
  }
  if (!this->options.bias()) {
    stream << ", bias=" << std::boolalpha << false;
  }
  if (!c10::get_if<enumtype::kZeros>(&this->options.padding_mode())) {
    stream << ", padding_mode=" << enumtype::get_enum_name(this->options.padding_mode());
  }
  stream << ")";
}

template <size_t D, typename Derived>
std::vector<int64_t> ConvTransposeImpl<D, Derived>::_output_padding(
    const Tensor& input, const c10::optional<at::IntArrayRef>& output_size,
    const ExpandingArray<D>& stride, const ExpandingArray<D>& padding,
    const ExpandingArray<D>& kernel_size) {
  std::vector<int64_t> ret;
  c10::optional<at::IntArrayRef> output_size_ = output_size;

  if (output_size_ == c10::nullopt) {
    ret = at::IntArrayRef(this->options.output_padding()).vec();
  } else {
    auto k = input.dim() - 2;
    if (output_size_.value().size() == k + 2) {
      output_size_ = output_size_.value().slice(2);
    }
    if (output_size_.value().size() != k) {
      TORCH_CHECK(false,
        "output_size must have ", k, " or ", k + 2, " elements (got ", output_size_.value().size(), ")");
    }

    std::vector<int64_t> min_sizes;
    std::vector<int64_t> max_sizes;
    for (int64_t d = 0; d < k; d++) {
      int64_t dim_size = ((input.sizes()[d + 2] - 1) * (*stride)[d] - 2 * (*padding)[d] + (*kernel_size)[d]);
      min_sizes.push_back(dim_size);
      max_sizes.push_back(min_sizes[d] + (*stride)[d] - 1);
    }

    for (size_t i = 0; i < output_size_.value().size(); i++) {
      int64_t size = output_size_.value()[i];
      int64_t min_size = min_sizes[i];
      int64_t max_size = max_sizes[i];
      if (size < min_size || size > max_size) {
        TORCH_CHECK(false,
          "requested an output size of ", output_size_.value(), ", but valid sizes range "
          "from ", min_sizes, " to ", max_sizes, " (for an input of ", input.sizes().slice(2), ")");
      }
    }

    for (int64_t d = 0; d < k; d++) {
      ret.push_back(output_size_.value()[d] - min_sizes[d]);
    }
  }
  return ret;
}

Tensor ConvTranspose1dImpl::forward(
    const Tensor& input, const c10::optional<at::IntArrayRef>& output_size) {
  if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
    TORCH_CHECK(false, "Only `zeros` padding mode is supported for ConvTranspose1d");
  }

  std::vector<int64_t> output_padding = _output_padding(
    input, output_size, options.stride(), options.padding(), options.kernel_size());

  return F::detail::conv_transpose1d(
    input, weight, bias, options.stride(), options.padding(),
    output_padding, options.groups(), options.dilation());
}

Tensor ConvTranspose2dImpl::forward(
    const Tensor& input, const c10::optional<at::IntArrayRef>& output_size) {
  if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
    TORCH_CHECK(false, "Only `zeros` padding mode is supported for ConvTranspose2d");
  }

  std::vector<int64_t> output_padding = _output_padding(
    input, output_size, options.stride(), options.padding(), options.kernel_size());

  return F::detail::conv_transpose2d(
    input, weight, bias, options.stride(), options.padding(),
    output_padding, options.groups(), options.dilation());
}

Tensor ConvTranspose3dImpl::forward(
    const Tensor& input, const c10::optional<at::IntArrayRef>& output_size) {
  if (!c10::get_if<enumtype::kZeros>(&options.padding_mode())) {
    TORCH_CHECK(false, "Only `zeros` padding mode is supported for ConvTranspose3d");
  }

  std::vector<int64_t> output_padding = _output_padding(
    input, output_size, options.stride(), options.padding(), options.kernel_size());

  return F::detail::conv_transpose3d(
    input, weight, bias, options.stride(), options.padding(),
    output_padding, options.groups(), options.dilation());
}

template class ConvTransposeImpl<1, ConvTranspose1dImpl>;
template class ConvTransposeImpl<2, ConvTranspose2dImpl>;
template class ConvTransposeImpl<3, ConvTranspose3dImpl>;

} // namespace nn
} // namespace torch
