import torch.nn as nn
from .base import BaseLayerParser, make_node


class LinearParser(BaseLayerParser):
    supported_type = nn.Linear

    def parse(self, module, name, input_names, output_name):
        weights = {}
        w_key = self._store_weight(name, "weight", module.weight, weights)
        inputs = input_names + [w_key]
        if module.bias is not None:
            b_key = self._store_weight(name, "bias", module.bias, weights)
            inputs.append(b_key)
        return make_node(
            "onnx::Gemm",
            {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 1},
            inputs,
            [output_name],
            weights=weights,
        )
