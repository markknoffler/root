from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np

NodeData = Dict[str, Any]


def make_node(node_type, attributes, inputs, outputs, dtype="Float", weights=None):
    return {
        "nodeType":       node_type,
        "nodeAttributes": attributes,
        "nodeInputs":     inputs,
        "nodeOutputs":    outputs,
        "nodeDType":      [dtype],
        "nodeWeights":    weights or {},
    }


class BaseLayerParser(ABC):

    @property
    @abstractmethod
    def supported_type(self):
        ...

    @abstractmethod
    def parse(self, module, name, input_names, output_name):
        ...

    @staticmethod
    def _store_weight(module_name, weight_name, tensor, store):
        import torch
        key = f"{module_name}_{weight_name}"
        if isinstance(tensor, torch.Tensor):
            store[key] = tensor.detach().float().numpy()
        else:
            store[key] = np.array(tensor, dtype=np.float32)
        return key

    @staticmethod
    def _to_pair(v):
        return [v, v] if isinstance(v, int) else list(v)
