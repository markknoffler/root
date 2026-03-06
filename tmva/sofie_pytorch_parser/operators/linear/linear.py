import torch.nn as nn
from typing import List
from ..base import BaseOperatorParser, NodeData, make_node


class LinearParser(BaseOperatorParser):

    supported_type = nn.Linear

    def parse(self, module: nn.Linear, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        weights = {}

        w_key = self._store_weight(name, "weight", module.weight, weights)
        inputs = input_names + [w_key]

        if module.bias is not None:
            b_key = self._store_weight(name, "bias", module.bias, weights)
            inputs.append(b_key)

        return make_node(
            node_type="onnx::Gemm",
            attributes={
                "alpha": 1.0,
                "beta": 1.0,
                "transA": 0,
                "transB": 1,
            },
            inputs=inputs,
            outputs=[output_name],
            weights=weights,
        )
