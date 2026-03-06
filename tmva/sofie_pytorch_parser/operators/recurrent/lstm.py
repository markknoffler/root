import torch.nn as nn
from typing import List
from ..base import BaseOperatorParser, NodeData, make_node
from .base_recurrent import extract_rnn_weights_onnx


class LSTMParser(BaseOperatorParser):

    supported_type = nn.LSTM

    def parse(self, module: nn.LSTM, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        w_key, r_key, b_key, weights = extract_rnn_weights_onnx(module, name, gate_factor=4)

        inputs = input_names + [w_key, r_key]
        if b_key:
            inputs.append(b_key)

        return make_node(
            node_type="onnx::LSTM",
            attributes={
                "hidden_size":   int(module.hidden_size),
                "input_size":    int(module.input_size),
                "num_layers":    int(module.num_layers),
                "bidirectional": int(module.bidirectional),
                "has_bias":      int(module.bias),
                "proj_size":     int(module.proj_size),
                "gate_order":    "pytorch_ifgo",
            },
            inputs=inputs,
            outputs=[output_name],
            weights=weights,
        )
