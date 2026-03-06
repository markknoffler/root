# operators/recurrent/rnn.py
#
# Vanilla RNN — ONNX op: onnx::RNN
#
# Math (one layer, one direction, nonlinearity=tanh):
#   h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
#
# Weight shape per layer per direction (gate_factor=1):
#   weight_ih_l{k}: (hidden_size, input_size)   [layer 0]
#                   (hidden_size, hidden_size)   [layer k>0]
#   weight_hh_l{k}: (hidden_size, hidden_size)
#   bias_ih_l{k}:   (hidden_size,)
#   bias_hh_l{k}:   (hidden_size,)
#
# Maps to ROperator_RNN<float> in SOFIE.

import torch.nn as nn
from typing import List
from ..base import BaseOperatorParser, NodeData, make_node
from .base_recurrent import extract_rnn_weights


class RNNParser(BaseOperatorParser):

    supported_type = nn.RNN

    def parse(self, module: nn.RNN, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        """
        Parse nn.RNN into SOFIE onnx::RNN nodeData.

        Attributes:
          - hidden_size:   int
          - input_size:    int
          - num_layers:    int
          - bidirectional: 0 or 1
          - nonlinearity:  'tanh' or 'relu'
          - has_bias:      0 or 1

        gate_factor=1 (single gate — no stacking).
        """
        wnames, weights = extract_rnn_weights(module, name, gate_factor=1)

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
            inputs=input_names + wnames,
            outputs=[output_name],
            weights=weights,
        )

