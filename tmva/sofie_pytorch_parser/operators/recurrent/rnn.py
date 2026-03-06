import torch.nn as nn
from typing import List
from ..base import BaseOperatorParser, NodeData, make_node
from .base_recurrent import extract_rnn_weights_onnx


class RNNParser(BaseOperatorParser):

    supported_type = nn.RNN

    def parse(self, module: nn.RNN, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        w_key, r_key, b_key, weights = extract_rnn_weights_onnx(module, name, gate_factor=1)

        inputs = input_names + [w_key, r_key]
        if b_key:
            inputs.append(b_key)

        return make_node(
            node_type="onnx::RNN",
            attributes={
                "hidden_size":   int(module.hidden_size),
                "input_size":    int(module.input_size),
                "num_layers":    int(module.num_layers),
                "bidirectional": int(module.bidirectional),
                "nonlinearity":  module.nonlinearity,
                "has_bias":      int(module.bias),
                "activations":   [module.nonlinearity.upper()],
            },
            inputs=inputs,
            outputs=[output_name],
            weights=weights,
        )
