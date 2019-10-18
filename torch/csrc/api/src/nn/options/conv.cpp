#include <torch/nn/options/conv.h>

namespace torch {
namespace nn {

template struct ConvOptions<1>;
template struct ConvOptions<2>;
template struct ConvOptions<3>;

template struct ConvTransposeOptions<1>;
template struct ConvTransposeOptions<2>;
template struct ConvTransposeOptions<3>;

} // namespace nn
} // namespace torch
