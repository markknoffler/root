from __future__ import annotations

import os

import onnx
from onnx import TensorProto, helper


def write_model(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "rnn_clip_runtime_bug.onnx")

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1, 1])
    y_h = helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, [1, 1, 1])

    w = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 1], [1.0])
    r = helper.make_tensor("R", TensorProto.FLOAT, [1, 1, 1], [0.0])
    b = helper.make_tensor("B", TensorProto.FLOAT, [1, 2], [0.0, 0.0])

    node = helper.make_node(
        "RNN",
        inputs=["X", "W", "R", "B"],
        outputs=["Y", "Y_h"],
        hidden_size=1,
        clip=0.5,
    )

    graph = helper.make_graph([node], "rnn_clip_runtime_bug", [x], [y, y_h], initializer=[w, r, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print(write_model(here))
