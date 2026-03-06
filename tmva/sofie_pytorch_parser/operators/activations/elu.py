# operators/activations/elu.py
#
# ELU (Exponential Linear Unit) — ONNX op: onnx::Elu
#
# Math:
#   f(x) = x              if x > 0
#   f(x) = alpha*(e^x - 1) if x <= 0
#
# No weight tensors. Only attribute: alpha (default 1.0).
# Maps directly to ROperator_Elu<float> in SOFIE.

import torch.nn as nn
from typing import Dict, List
from ..base import BaseOperatorParser, NodeData, make_node


class ELUParser(BaseOperatorParser):

    supported_type = nn.ELU

    def parse(self, module: nn.ELU, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        """
        Parse nn.ELU into SOFIE onnx::Elu nodeData.

        Attributes extracted:
          - alpha: float — scaling factor for negative region (default 1.0)

        Note: nn.ELU has an `inplace` flag — SOFIE inference always copies,
        so inplace is ignored and not stored.
        """
        return make_node(
            node_type="onnx::Elu",
            attributes={"alpha": float(module.alpha)},
            inputs=input_names,
            outputs=[output_name],
        )

