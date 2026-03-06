import torch.nn as nn
from typing import List
from ..base import BaseOperatorParser, NodeData, make_node


class Conv2dParser(BaseOperatorParser):

    supported_type = nn.Conv2d

    def parse(self, module: nn.Conv2d, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        weights = {}

        w_key = self._store_weight(name, "weight", module.weight, weights)
        inputs = input_names + [w_key]

        if module.bias is not None:
            b_key = self._store_weight(name, "bias", module.bias, weights)
            inputs.append(b_key)

        k = self._to_pair(module.kernel_size)
        s = self._to_pair(module.stride)
        p = self._to_pair(module.padding)
        d = self._to_pair(module.dilation)

        return make_node(
            node_type="onnx::Conv",
            attributes={
                "dilations": d,
                "group": module.groups,
                "kernel_shape": k,
                "pads": p + p,
                "strides": s,
            },
            inputs=inputs,
            outputs=[output_name],
            weights=weights,
        )
