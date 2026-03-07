import torch.nn as nn
from .base import BaseLayerParser, make_node


class SigmoidParser(BaseLayerParser):
    supported_type = nn.Sigmoid

    def parse(self, module, name, input_names, output_name):
        return make_node("onnx::Sigmoid", {}, input_names, [output_name])
