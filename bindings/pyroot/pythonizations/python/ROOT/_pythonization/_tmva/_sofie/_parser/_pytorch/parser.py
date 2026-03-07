import torch.nn as nn

from .layers.relu import ReLUParser
from .layers.elu import ELUParser
from .layers.sigmoid import SigmoidParser
from .layers.linear import LinearParser
from .layers.conv2d import Conv2dParser
from .layers.maxpool2d import MaxPool2dParser
from .layers.batchnorm2d import BatchNorm2dParser
from .layers.recurrent import RNNParser, LSTMParser, GRUParser

_REGISTRY = [
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

_TYPE_MAP = {p.supported_type: p for p in _REGISTRY}


class PyTorch:
    @staticmethod
    def Parse(model, input_shape, input_name="input_0"):
        model.eval()
        counter = [0]

        def uid(prefix):
            counter[0] += 1
            return f"{prefix}_{counter[0]}"

        operators = []
        initializers = {}
        current = input_name

        top = _TYPE_MAP.get(type(model))
        if top is not None:
            out = uid(f"out_{type(model).__name__}")
            node = top.parse(model, type(model).__name__, [current], out)
            operators.append(node)
            initializers.update(node.get("nodeWeights", {}))
            current = out
        else:
            for child_name, module in model.named_children():
                out = uid(f"out_{child_name}")
                parser = _TYPE_MAP.get(type(module))
                if parser is None:
                    print(f"[SOFIE Parser] WARNING: unsupported layer '{type(module).__name__}' at '{child_name}' — skipping")
                    current = out
                    continue
                node = parser.parse(module, child_name, [current], out)
                operators.append(node)
                initializers.update(node.get("nodeWeights", {}))
                current = out

        return {
            "operators":    operators,
            "initializers": initializers,
            "inputs":       {input_name: list(input_shape)},
            "outputs":      {current: None},
        }
