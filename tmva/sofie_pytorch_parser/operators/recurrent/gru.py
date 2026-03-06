# operators/recurrent/gru.py
#
# GRU — ONNX op: onnx::GRU
#
# Math (one layer, one direction):
#   r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)   [reset gate]
#   z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)   [update gate]
#   n_t = tanh   (W_in @ x_t + b_in + r_t * (W_hn @ h_{t-1} + b_hn))  [new gate]
#   h_t = (1 - z_t) * n_t + z_t * h_{t-1}
#
# Weight shape per layer per direction (gate_factor=3):
#   weight_ih_l{k}: (3*hidden_size, input_size)
#   Gate row order: [r=0..H, z=H..2H, n=2H..3H]  — PyTorch convention
#
# CRITICAL — ONNX GRU gate order is [z, r, h] (update, reset, hidden).
# PyTorch uses [r, z, n] (reset, update, new).
# The rows of weight_ih and weight_hh must be reordered if using strict ONNX.
# We document this explicitly via the gate_order attribute.
#
# Maps to ROperator_GRU<float> in SOFIE (uses PyTorch ordering).

import torch.nn as nn
from typing import List
from ..base import BaseOperatorParser, NodeData, make_node
from .base_recurrent import extract_rnn_weights


class GRUParser(BaseOperatorParser):

    supported_type = nn.GRU

    def parse(self, module: nn.GRU, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        """
        Parse nn.GRU into SOFIE onnx::GRU nodeData.

        Attributes:
          - hidden_size:   int
          - input_size:    int
          - num_layers:    int
          - bidirectional: 0 or 1
          - has_bias:      0 or 1
          - gate_order:    'pytorch_rzn' — documents PyTorch vs ONNX row difference

        gate_factor=3 (3 gates: r, z, n stacked in row dimension).

        NOTE on gate_order: If feeding this to a strict ONNX onnx::GRU consumer,
        reorder rows as: [z=rows 1, r=rows 0, h=rows 2] → ONNX [z,r,h] order.
        SOFIE's ROperator_GRU uses PyTorch ordering, so no reorder is needed here.
        """
        wnames, weights = extract_rnn_weights(module, name, gate_factor=3)

        return make_node(
            node_type="onnx::GRU",
            attributes={
                "hidden_size":   int(module.hidden_size),
                "input_size":    int(module.input_size),
                "num_layers":    int(module.num_layers),
                "bidirectional": int(module.bidirectional),
                "has_bias":      int(module.bias),
                # PyTorch=[r,z,n] vs ONNX=[z,r,h] — documented for C++ consumer
                "gate_order":    "pytorch_rzn",
            },
            inputs=input_names + wnames,
            outputs=[output_name],
            weights=weights,
        )

