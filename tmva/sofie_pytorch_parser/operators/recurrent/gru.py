import torch.nn as nn
from typing import List
from ..base import BaseOperatorParser, NodeData, make_node
from .base_recurrent import extract_rnn_weights_onnx


class GRUParser(BaseOperatorParser):

    supported_type = nn.GRU

    def parse(self, module: nn.GRU, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        w_key, r_key, b_key, weights = extract_rnn_weights_onnx(module, name, gate_factor=3)

        inputs = input_names + [w_key, r_key]
        if b_key:
            inputs.append(b_key)

        return make_node(
            node_type="onnx::GRU",
            attributes={
                "hidden_size":   int(module.hidden_size),
                "input_size":    int(module.input_size),
                "num_layers":    int(module.num_layers),
                "bidirectional": int(module.bidirectional),
                "has_bias":      int(module.bias),
                "gate_order":    "pytorch_rzn",
            },
            inputs=inputs,
            outputs=[output_name],
            weights=weights,
        )
