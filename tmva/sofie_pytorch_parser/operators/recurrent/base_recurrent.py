# operators/recurrent/base_recurrent.py
#
# Shared weight extraction logic for RNN / LSTM / GRU.
#
# PyTorch stores all RNN-family weights with a consistent naming pattern:
#   weight_ih_l{k}[_reverse]  — input-to-hidden weights, layer k, direction
#   weight_hh_l{k}[_reverse]  — hidden-to-hidden weights
#   bias_ih_l{k}[_reverse]    — input-to-hidden bias
#   bias_hh_l{k}[_reverse]    — hidden-to-hidden bias
#
# Gate stacking (gates are concatenated in the row dimension):
#   RNN:  1 gate  — weight shape: (    hidden,  input)
#   GRU:  3 gates — weight shape: (3 * hidden,  input), order: [r, z, n]
#   LSTM: 4 gates — weight shape: (4 * hidden,  input), order: [i, f, g, o]
#
# WARNING — ONNX gate ordering differs from PyTorch:
#   PyTorch GRU:  [r, z, n]  (reset, update, new)
#   ONNX GRU:     [z, r, h]  (update, reset, hidden)
#   Rows must be reordered when targeting ONNX-strict consumers.

from typing import Dict, List, Tuple
import numpy as np


def extract_rnn_weights(
    module,
    name: str,
    gate_factor: int,
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """
    Extract all weight tensors for an RNN/LSTM/GRU module.

    Args:
        module:      nn.RNN | nn.LSTM | nn.GRU instance
        name:        prefix for all weight tensor names
        gate_factor: 1 (RNN), 3 (GRU), 4 (LSTM)

    Returns:
        (weight_names, weights_dict)
        weight_names: ordered list of tensor name strings
        weights_dict: name → float32 numpy array
    """
    directions = [""] + (["_reverse"] if module.bidirectional else [])
    weight_names: List[str] = []
    weights_dict: Dict[str, np.ndarray] = {}

    for layer in range(module.num_layers):
        for sfx in directions:
            wih_key = f"{name}_weight_ih_l{layer}{sfx}"
            whh_key = f"{name}_weight_hh_l{layer}{sfx}"

            weights_dict[wih_key] = (
                getattr(module, f"weight_ih_l{layer}{sfx}")
                .detach().float().numpy()
            )
            weights_dict[whh_key] = (
                getattr(module, f"weight_hh_l{layer}{sfx}")
                .detach().float().numpy()
            )
            weight_names += [wih_key, whh_key]

            if module.bias:
                bih_key = f"{name}_bias_ih_l{layer}{sfx}"
                bhh_key = f"{name}_bias_hh_l{layer}{sfx}"
                weights_dict[bih_key] = (
                    getattr(module, f"bias_ih_l{layer}{sfx}")
                    .detach().float().numpy()
                )
                weights_dict[bhh_key] = (
                    getattr(module, f"bias_hh_l{layer}{sfx}")
                    .detach().float().numpy()
                )
                weight_names += [bih_key, bhh_key]

    # Validate stacking — helps catch weight corruption early
    h = module.hidden_size
    for layer in range(module.num_layers):
        in_size = module.input_size if layer == 0 else module.hidden_size
        for sfx in directions:
            wih = weights_dict[f"{name}_weight_ih_l{layer}{sfx}"]
            whh = weights_dict[f"{name}_weight_hh_l{layer}{sfx}"]
            assert wih.shape == (gate_factor * h, in_size), \
                f"weight_ih shape mismatch at layer {layer}: got {wih.shape}"
            assert whh.shape == (gate_factor * h, h), \
                f"weight_hh shape mismatch at layer {layer}: got {whh.shape}"

    return weight_names, weights_dict

