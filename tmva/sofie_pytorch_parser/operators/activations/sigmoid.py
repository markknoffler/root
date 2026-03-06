# operators/activations/sigmoid.py
# Sigmoid — onnx::Sigmoid — no attributes, no weights.
# f(x) = 1 / (1 + e^(-x))
# Maps to ROperator_Sigmoid<float> in SOFIE.

import torch.nn as nn
from typing import List
from ..base import BaseOperatorParser, NodeData, make_node


class SigmoidParser(BaseOperatorParser):

    supported_type = nn.Sigmoid

    def parse(self, module: nn.Sigmoid, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        return make_node("onnx::Sigmoid", {}, input_names, [output_name])

