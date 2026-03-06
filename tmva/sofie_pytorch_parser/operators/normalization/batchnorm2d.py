# operators/normalization/batchnorm2d.py
#
# BatchNorm2D — ONNX op: onnx::BatchNormalization
#
# Math (inference mode — running stats used, not batch stats):
#   x_norm = (x - running_mean) / sqrt(running_var + eps)
#   output  = weight * x_norm + bias
#
# Weight tensors (4 total):
#   weight       (gamma / scale):  shape [num_features]
#   bias         (beta / shift):   shape [num_features]
#   running_mean:                  shape [num_features]
#   running_var:                   shape [num_features]
#
# Maps to ROperator_BatchNormalization<float> in SOFIE.
# IMPORTANT: model must be in eval() mode so running stats are frozen.

import torch.nn as nn
from typing import Dict, List
from ..base import BaseOperatorParser, NodeData, make_node


class BatchNorm2dParser(BaseOperatorParser):

    supported_type = nn.BatchNorm2d

    def parse(self, module: nn.BatchNorm2d, name: str,
              input_names: List[str], output_name: str) -> NodeData:
        """
        Parse nn.BatchNorm2d into SOFIE onnx::BatchNormalization nodeData.

        Attributes extracted:
          - epsilon:      float — numerical stability constant (default 1e-5)
          - momentum:     float — running stat update rate (not used at inference)
          - num_features: int   — number of channels C

        Weight tensors stored:
          {name}_weight, {name}_bias, {name}_running_mean, {name}_running_var
        """
        weights: Dict = {}

        w   = self._store_weight(name, "weight",       module.weight,       weights)
        b   = self._store_weight(name, "bias",         module.bias,         weights)
        mu  = self._store_weight(name, "running_mean", module.running_mean, weights)
        var = self._store_weight(name, "running_var",  module.running_var,  weights)

        weight_names = [w, b, mu, var]

        return make_node(
            node_type="onnx::BatchNormalization",
            attributes={
                "epsilon":      float(module.eps),
                "momentum":     float(module.momentum) if module.momentum else 0.1,
                "num_features": int(module.num_features),
                # training_mode=0 signals inference — running stats are used
                "training_mode": 0,
            },
            inputs=input_names + weight_names,
            outputs=[output_name],
            weights=weights,
        )

