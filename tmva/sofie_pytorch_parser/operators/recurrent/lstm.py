# operators/recurrent/lstm.py
#
# LSTM — ONNX op: onnx::LSTM
#
# Math (one layer, one direction):
#   i_t = sigmoid(W_ii @ x_t + b_ii + W_hi @ h_{t-1} + b_hi)  [input gate]
#   f_t = sigmoid(W_if @ x_t + b_if + W_hf @ h_{t-1} + b_hf)  [forget gate]
#   g_t = tanh   (W_ig @ x_t + b_ig + W_hg @ h_{t-1} + b_hg)  [cell gate]
#   o_t = sigmoid(W_io @ x_t + b_io + W_ho @ h_{t-1} + b_ho)  [output gate]
#   c_t = f_t * c_{t-1} + i_t * g_t
#   h_t = o_t * tanh(c_t)
#
# Weight shape per layer per direction (gate_factor=4):
#   weight_ih_l{k}: (4*hidden_size, input_size)
#   weight_hh_l{k}: (4*hidden_size, hidden_size)
#   Gate row order: [i=0..H, f=H..2H, g=2H..3H, o=3H..4H]  — PyTorch convention
#
# ONNX LSTM gate order is ALSO [i, o, f, g] — different! Row reorder needed
# if interfacing with strict ONNX consumers. SOFIE's ROperator_LSTM follows
# PyTorch ordering, so no reorder needed here.
#
# Maps to ROperator_LSTM<float> in SOFIE.

import torch.nn as nn
from typing import List
from ..base import BaseOperatorParser, NodeData, make_node
from .base_recurrent import extract_rnn_weights


class LSTMParser(BaseOperatorParser):

    supported_type = nn.LSTM

    def parse(self, module: nn.LSTM, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        """
        Parse nn.LSTM into SOFIE onnx::LSTM nodeData.

        Attributes:
          - hidden_size:   int
          - input_size:    int
          - num_layers:    int
          - bidirectional: 0 or 1
          - has_bias:      0 or 1
          - proj_size:     int (0 means no projection)
          - gate_order:    'pytorch_ifgo' — documents PyTorch gate stacking order

        gate_factor=4 (4 gates: i, f, g, o stacked in row dimension).
        """
        wnames, weights = extract_rnn_weights(module, name, gate_factor=4)

        return make_node(
            node_type="onnx::LSTM",
            attributes={
                "hidden_size":   int(module.hidden_size),
                "input_size":    int(module.input_size),
                "num_layers":    int(module.num_layers),
                "bidirectional": int(module.bidirectional),
                "has_bias":      int(module.bias),
                "proj_size":     int(module.proj_size),
                # Documents PyTorch gate stacking for C++ consumer
                # Rows: [i=0..H, f=H..2H, g=2H..3H, o=3H..4H]
                "gate_order":    "pytorch_ifgo",
            },
            inputs=input_names + wnames,
            outputs=[output_name],
            weights=weights,
        )

