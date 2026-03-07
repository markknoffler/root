import numpy as np
import torch.nn as nn
from .base import BaseLayerParser, make_node


def _extract_weights(module, name, gate_factor):
    num_directions = 2 if module.bidirectional else 1
    h = module.hidden_size
    weights = {}

    wih_list, whh_list, bih_list, bhh_list = [], [], [], []
    directions = [""] + (["_reverse"] if module.bidirectional else [])

    for layer in range(module.num_layers):
        for sfx in directions:
            wih_list.append(getattr(module, f"weight_ih_l{layer}{sfx}").detach().float().numpy())
            whh_list.append(getattr(module, f"weight_hh_l{layer}{sfx}").detach().float().numpy())
            if module.bias:
                bih_list.append(getattr(module, f"bias_ih_l{layer}{sfx}").detach().float().numpy())
                bhh_list.append(getattr(module, f"bias_hh_l{layer}{sfx}").detach().float().numpy())

    w_key = f"{name}_W"
    r_key = f"{name}_R"
    b_key = f"{name}_B"

    weights[w_key] = np.stack(wih_list).reshape(num_directions, gate_factor * h, -1)
    weights[r_key] = np.stack(whh_list).reshape(num_directions, gate_factor * h, h)

    if module.bias:
        b_combined = np.concatenate([np.stack(bih_list), np.stack(bhh_list)], axis=1)
        weights[b_key] = b_combined.reshape(num_directions, 2 * gate_factor * h)
        return w_key, r_key, b_key, weights

    return w_key, r_key, "", weights


class RNNParser(BaseLayerParser):
    supported_type = nn.RNN

    def parse(self, module, name, input_names, output_name):
        w_key, r_key, b_key, weights = _extract_weights(module, name, 1)
        inputs = input_names + [w_key, r_key]
        if b_key:
            inputs.append(b_key)
        return make_node(
            "onnx::RNN",
            {
                "hidden_size":   int(module.hidden_size),
                "input_size":    int(module.input_size),
                "num_layers":    int(module.num_layers),
                "bidirectional": int(module.bidirectional),
                "nonlinearity":  module.nonlinearity,
                "has_bias":      int(module.bias),
                "activations":   [module.nonlinearity.upper()],
            },
            inputs,
            [output_name],
            weights=weights,
        )


class LSTMParser(BaseLayerParser):
    supported_type = nn.LSTM

    def parse(self, module, name, input_names, output_name):
        w_key, r_key, b_key, weights = _extract_weights(module, name, 4)
        inputs = input_names + [w_key, r_key]
        if b_key:
            inputs.append(b_key)
        return make_node(
            "onnx::LSTM",
            {
                "hidden_size":   int(module.hidden_size),
                "input_size":    int(module.input_size),
                "num_layers":    int(module.num_layers),
                "bidirectional": int(module.bidirectional),
                "has_bias":      int(module.bias),
                "proj_size":     int(module.proj_size),
                "gate_order":    "pytorch_ifgo",
            },
            inputs,
            [output_name],
            weights=weights,
        )


class GRUParser(BaseLayerParser):
    supported_type = nn.GRU

    def parse(self, module, name, input_names, output_name):
        w_key, r_key, b_key, weights = _extract_weights(module, name, 3)
        inputs = input_names + [w_key, r_key]
        if b_key:
            inputs.append(b_key)
        return make_node(
            "onnx::GRU",
            {
                "hidden_size":   int(module.hidden_size),
                "input_size":    int(module.input_size),
                "num_layers":    int(module.num_layers),
                "bidirectional": int(module.bidirectional),
                "has_bias":      int(module.bias),
                "gate_order":    "pytorch_rzn",
            },
            inputs,
            [output_name],
            weights=weights,
        )
