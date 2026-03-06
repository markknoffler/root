# operators/base.py
# Defines the canonical SOFIE nodeData structure and abstract base.
# Format matches RModelParser_PyTorch.cxx INTERNAL::MakePyTorchNode() comment block.

from abc import ABC, abstractmethod
from typing import Dict, List, Any
import numpy as np


#: The canonical SOFIE nodeData dict format expected by RModelParser_PyTorch.cxx
#:
#: dict fNode {
#:   'nodeType'       : str         — ONNX op type, e.g. 'onnx::Gemm'
#:   'nodeAttributes' : dict        — op-specific hyperparameters
#:   'nodeInputs'     : List[str]   — ordered tensor names (data + weights)
#:   'nodeOutputs'    : List[str]   — output tensor names
#:   'nodeDType'      : List[str]   — ['Float'] for float32
#:   'nodeWeights'    : dict        — name -> np.ndarray (extension for Python parser)
#: }
NodeData = Dict[str, Any]


def make_node(
    node_type: str,
    attributes: Dict[str, Any],
    inputs: List[str],
    outputs: List[str],
    dtype: str = "Float",
    weights: Dict[str, np.ndarray] = None,
) -> NodeData:
    """
    Construct a SOFIE-compatible nodeData dict.
    This is the single point of truth for the format — mirrors exactly
    the structure documented in RModelParser_PyTorch.cxx MakePyTorchNode().
    """
    return {
        "nodeType":       node_type,
        "nodeAttributes": attributes,
        "nodeInputs":     inputs,
        "nodeOutputs":    outputs,
        "nodeDType":      [dtype],
        "nodeWeights":    weights or {},
    }


class BaseOperatorParser(ABC):
    """
    Abstract base for all layer-specific parsers.
    Each subclass handles exactly one nn.Module type.
    """

    @property
    @abstractmethod
    def supported_type(self):
        """The torch.nn.Module class this parser handles."""
        ...

    @abstractmethod
    def parse(
        self,
        module,
        name: str,
        input_names: List[str],
        output_name: str,
    ) -> NodeData:
        """
        Extract a SOFIE nodeData dict from a live nn.Module.

        Args:
            module:       The nn.Module instance (weights already loaded)
            name:         Unique name prefix for weight tensors
            input_names:  Ordered list of input tensor names
            output_name:  Name for the output tensor

        Returns:
            NodeData dict compatible with RModelParser_PyTorch.cxx format
        """
        ...

    @staticmethod
    def _store_weight(module_name: str, weight_name: str,
                      tensor, store: Dict) -> str:
        """
        Extract a weight tensor to float32 numpy and register it in `store`.
        Returns the canonical key name.
        """
        import torch
        key = f"{module_name}_{weight_name}"
        if isinstance(tensor, torch.Tensor):
            store[key] = tensor.detach().float().numpy()
        else:
            store[key] = np.array(tensor, dtype=np.float32)
        return key

    @staticmethod
    def _to_pair(v):
        """Expand int to [int, int]; leave list/tuple as list."""
        return [v, v] if isinstance(v, int) else list(v)

