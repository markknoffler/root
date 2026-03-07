import torch.nn as nn
from .base import BaseLayerParser, make_node


class Conv2dParser(BaseLayerParser):
    supported_type = nn.Conv2d

    def parse(self, module, name, input_names, output_name):
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
            "onnx::Conv",
            {"dilations": d, "group": module.groups, "kernel_shape": k, "pads": p + p, "strides": s},
            inputs,
            [output_name],
            weights=weights,
        )
