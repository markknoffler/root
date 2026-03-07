import torch.nn as nn
from .base import BaseLayerParser, make_node


class MaxPool2dParser(BaseLayerParser):
    supported_type = nn.MaxPool2d

    def parse(self, module, name, input_names, output_name):
        pad = self._to_pair(module.padding)
        stride = module.stride if module.stride is not None else module.kernel_size
        return make_node(
            "onnx::MaxPool",
            {
                "kernel_shape": self._to_pair(module.kernel_size),
                "strides":      self._to_pair(stride),
                "pads":         pad + pad,
                "dilations":    self._to_pair(module.dilation),
                "ceil_mode":    int(module.ceil_mode),
            },
            input_names,
            [output_name],
        )
