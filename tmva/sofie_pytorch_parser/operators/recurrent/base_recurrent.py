from typing import Dict, List, Tuple
import numpy as np


def extract_rnn_weights_onnx(
    module,
    name: str,
    gate_factor: int,
) -> Tuple[str, str, str, Dict[str, np.ndarray]]:
    num_directions = 2 if module.bidirectional else 1
    h = module.hidden_size
    weights: Dict[str, np.ndarray] = {}

    wih_list = []
    whh_list = []
    bih_list = []
    bhh_list = []

    directions = [""] + (["_reverse"] if module.bidirectional else [])

    for layer in range(module.num_layers):
        for sfx in directions:
            wih = getattr(module, f"weight_ih_l{layer}{sfx}").detach().float().numpy()
            whh = getattr(module, f"weight_hh_l{layer}{sfx}").detach().float().numpy()
            wih_list.append(wih)
            whh_list.append(whh)

            if module.bias:
                bih = getattr(module, f"bias_ih_l{layer}{sfx}").detach().float().numpy()
                bhh = getattr(module, f"bias_hh_l{layer}{sfx}").detach().float().numpy()
                bih_list.append(bih)
                bhh_list.append(bhh)

    w_key = f"{name}_W"
    r_key = f"{name}_R"
    b_key = f"{name}_B"

    w_stack = np.stack(wih_list, axis=0)
    r_stack = np.stack(whh_list, axis=0)

    weights[w_key] = w_stack.reshape(num_directions, gate_factor * h, -1)
    weights[r_key] = r_stack.reshape(num_directions, gate_factor * h, h)

    if module.bias:
        b_combined = np.concatenate([
            np.stack(bih_list, axis=0),
            np.stack(bhh_list, axis=0)
        ], axis=1)
        weights[b_key] = b_combined.reshape(num_directions, 2 * gate_factor * h)
        return w_key, r_key, b_key, weights

    return w_key, r_key, "", weights
