from __future__ import absolute_import, division, print_function, unicode_literals
from torch.nn.qat import Conv2d as QATConv2d
from torch.nn.modules.conv import conv2d_forwad
from torch.quantization.QConfig import default_qat_qconfig
import torch.nn.functional as F

class ConvReLU2d(QATConv2d):
    r"""
    A Conv2d module attached with FakeQuantize modules for both output
    activation and weight, used for quantization aware training.

    We adopt the same interface as `torch.nn.Conv2d`, please see
    https://pytorch.org/docs/stable/nn.html?highlight=conv2d#torch.nn.Conv2d
    for documentation.

    Similar to `torch.nn.Conv2d`, with FakeQuantize modules initialized to
    default.

    Attributes:
        observer: fake quant module for output activation, it's called observer
            to align with post training flow, TODO: rename?
        weight_fake_quant: fake quant module for weight

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros',
                 activation_fake_quant=default_qat_qconfig.activation(),
                 weight_fake_quant=default_qat_qconfig.weight()):
        super(ConvReLU2d, self).__init__(in_channels, out_channels, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation,
                                         groups=groups, bias=bias, padding_mode=padding_mode)
        self.observer = activation_fake_quant
        self.weight_fake_quant = weight_fake_quant

    def forward(self, input):
        return self.observer(F.relu(conv2d_forwad(input, self.padding_mode,
                             self.padding, self.weight_fake_quant(self.weight),
                             self.bias, self.stride, self.dilation, self.groups),
                             True))
