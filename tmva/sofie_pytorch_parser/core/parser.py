# core/parser.py
# Main SOFIEPyTorchParser orchestrator.
# Walks an nn.Module tree, dispatches to per-operator parsers,
# collects nodeData dicts + initializer tensors.

import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Type

from ..operators.base import BaseOperatorParser, NodeData
from ..operators.activations.elu import ELUParser
from ..operators.activations.relu import ReLUParser
from ..operators.activations.sigmoid import SigmoidParser
from ..operators.linear.linear import LinearParser
from ..operators.linear.conv2d import Conv2dParser
from ..operators.pooling.maxpool2d import MaxPool2dParser
from ..operators.normalization.batchnorm2d import BatchNorm2dParser
from ..operators.recurrent.rnn import RNNParser
from ..operators.recurrent.lstm import LSTMParser
from ..operators.recurrent.gru import GRUParser


# Registry: maps nn.Module type → parser instance
_OPERATOR_REGISTRY: List[BaseOperatorParser] = [
    ELUParser(),
    ReLUParser(),
    SigmoidParser(),
    LinearParser(),
    Conv2dParser(),
    MaxPool2dParser(),
    BatchNorm2dParser(),
    RNNParser(),
    LSTMParser(),
    GRUParser(),
]

_TYPE_MAP: Dict[Type, BaseOperatorParser] = {
    p.supported_type: p for p in _OPERATOR_REGISTRY
}


class SOFIEPyTorchParser:
    """
    Python-native parser for PyTorch nn.Module → SOFIE RModel format.

    Unlike the existing C++ RModelParser_PyTorch which relies on the deprecated
    private API `torch.onnx.utils._model_to_graph` (broken since PyTorch 2.0+),
    this parser inspects nn.Module attributes directly — stable, version-independent,
    and especially reliable for RNN/LSTM/GRU layers.

    Output format is fully compatible with the nodeData dict structure documented
    in RModelParser_PyTorch.cxx INTERNAL::MakePyTorchNode().
    """

    def __init__(self):
        self._counter = 0

    def _uid(self, prefix: str = "t") -> str:
        self._counter += 1
        return f"{prefix}_{self._counter}"

    def parse(
        self,
        model: nn.Module,
        input_shape: Tuple,
        input_name: str = "input_0",
    ) -> Dict:
        """
        Parse a trained nn.Module into SOFIE-compatible format.

        Args:
            model:        Trained nn.Module — must be in eval() mode
            input_shape:  Full shape with batch dim, e.g. (1, 32) or (1, 3, 224, 224)
            input_name:   Name for the primary input tensor

        Returns:
            {
              'operators':    [nodeData, ...],  # SOFIE-compatible node dicts
              'initializers': {name: np.ndarray},  # all weight tensors
              'inputs':       {name: shape},
              'outputs':      {name: None}
            }
        """
        self._counter = 0
        operators: List[NodeData] = []
        initializers: Dict = {}
        current = input_name

        model.eval()

        for child_name, module in model.named_children():
            out = self._uid(f"out_{child_name}")
            parser = _TYPE_MAP.get(type(module))

            if parser is None:
                print(f"[SOFIE Parser] WARNING: Unsupported layer "
                      f"'{type(module).__name__}' at '{child_name}' — skipping")
                current = out
                continue

            node = parser.parse(module, child_name, [current], out)
            operators.append(node)

            # Merge this operator's weights into the global initializer store
            initializers.update(node.get("nodeWeights", {}))

            current = out

        return {
            "operators":    operators,
            "initializers": initializers,
            "inputs":       {input_name: list(input_shape)},
            "outputs":      {current: None},
        }

