import torch.nn as nn
from .base import BaseLayerParser, make_node


class BatchNorm2dParser(BaseLayerParser):
    supported_type = nn.BatchNorm2d

    def parse(self, module, name, input_names, output_name):
        weights = {}
        w   = self._store_weight(name, "weight",       module.weight,       weights)
        b   = self._store_weight(name, "bias",         module.bias,         weights)
        mu  = self._store_weight(name, "running_mean", module.running_mean, weights)
        var = self._store_weight(name, "running_var",  module.running_var,  weights)
        return make_node(
            "onnx::BatchNormalization",
            {
                "epsilon":       float(module.eps),
                "momentum":      float(module.momentum) if module.momentum else 0.1,
                "num_features":  int(module.num_features),
                "training_mode": 0,
            },
            input_names + [w, b, mu, var],
            [output_name],
            weights=weights,
        )
