# operators/activations/relu.py
# ReLU — onnx::Relu — no attributes, no weights.
# f(x) = max(0, x)
# Maps to ROperator_Relu<float> in SOFIE.

import torch.nn as nn
from typing import List
from ..base import BaseOperatorParser, NodeData, make_node


class ReLUParser(BaseOperatorParser):

    supported_type = nn.ReLU

    def parse(self, module: nn.ReLU, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        return make_node("onnx::Relu", {}, input_names, [output_name])

