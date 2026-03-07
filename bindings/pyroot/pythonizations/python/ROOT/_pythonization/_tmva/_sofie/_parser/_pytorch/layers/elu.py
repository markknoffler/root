import torch.nn as nn
from .base import BaseLayerParser, make_node


class ELUParser(BaseLayerParser):
    supported_type = nn.ELU

    def parse(self, module, name, input_names, output_name):
        return make_node(
            "onnx::Elu",
            {"alpha": float(module.alpha)},
            input_names,
            [output_name],
        )
