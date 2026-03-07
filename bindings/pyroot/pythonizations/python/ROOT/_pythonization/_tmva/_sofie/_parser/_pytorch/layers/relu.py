import torch.nn as nn
from .base import BaseLayerParser, make_node


class ReLUParser(BaseLayerParser):
    supported_type = nn.ReLU

    def parse(self, module, name, input_names, output_name):
        return make_node("onnx::Relu", {}, input_names, [output_name])
