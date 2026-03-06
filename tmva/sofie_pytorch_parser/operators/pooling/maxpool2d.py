# operators/pooling/maxpool2d.py
#
# MaxPool2D — ONNX op: onnx::MaxPool
#
# Math:
#   output[n,c,h,w] = max over k_h x k_w window of input[n,c, h*stride+i, w*stride+j]
#
# No weight tensors.
# Attributes: kernel_shape, strides, pads (ONNX format: [top, left, bottom, right]),
#             dilations, ceil_mode.
# Maps to ROperator_Pool<float> in SOFIE (MaxPool variant).

import torch.nn as nn
from typing import Dict, List
from ..base import BaseOperatorParser, NodeData, make_node


class MaxPool2dParser(BaseOperatorParser):

    supported_type = nn.MaxPool2d

    def parse(self, module: nn.MaxPool2d, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        """
        Parse nn.MaxPool2d into SOFIE onnx::MaxPool nodeData.

        ONNX pads format is [top, left, bottom, right].
        PyTorch `padding` is symmetric, so we expand: [p, p, p, p].

        Attributes extracted:
          - kernel_shape: [kH, kW]
          - strides:      [sH, sW]  (defaults to kernel_shape if None)
          - pads:         [top, left, bottom, right]
          - dilations:    [dH, dW]
          - ceil_mode:    0 or 1
        """
        pad    = self._to_pair(module.padding)
        stride = module.stride if module.stride is not None else module.kernel_size

        return make_node(
            node_type="onnx::MaxPool",
            attributes={
                "kernel_shape": self._to_pair(module.kernel_size),
                "strides":      self._to_pair(stride),
                "pads":         pad + pad,        # symmetric → ONNX [t,l,b,r]
                "dilations":    self._to_pair(module.dilation),
                "ceil_mode":    int(module.ceil_mode),
            },
            inputs=input_names,
            outputs=[output_name],
        )

