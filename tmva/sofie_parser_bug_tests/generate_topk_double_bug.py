from __future__ import annotations

import os

import onnx
from onnx import TensorProto, helper


def write_model(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "topk_double.onnx")

    x = helper.make_tensor_value_info("X", TensorProto.DOUBLE, [4])
    k = helper.make_tensor("K", TensorProto.INT64, [1], [2])
    y_val = helper.make_tensor_value_info("Y_val", TensorProto.DOUBLE, [2])
    y_idx = helper.make_tensor_value_info("Y_idx", TensorProto.INT64, [2])

    node = helper.make_node(
        "TopK",
        inputs=["X", "K"],
        outputs=["Y_val", "Y_idx"],
        axis=0,
        largest=1,
        sorted=1,
    )

    graph = helper.make_graph([node], "topk_double", [x], [y_val, y_idx], initializer=[k])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print(write_model(here))

