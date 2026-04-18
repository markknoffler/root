from __future__ import annotations

import os

import onnx
from onnx import TensorProto, helper


def _write(model: onnx.ModelProto, name: str, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    onnx.save(model, path)
    return path


def make_unschedulable_single_add(out_dir: str) -> str:
    add_node = helper.make_node("Add", inputs=["ghost_a", "ghost_b"], outputs=["out"])
    graph = helper.make_graph([add_node], "bad", [], [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return _write(model, "unschedulable_add.onnx", out_dir)


def make_valid_tiny_add(out_dir: str) -> str:
    init = helper.make_tensor("b", TensorProto.FLOAT, [1], [1.0])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    node = helper.make_node("Add", inputs=["x", "b"], outputs=["y"])
    graph = helper.make_graph([node], "ok", [x], [y], initializer=[init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    return _write(model, "valid_tiny_add.onnx", out_dir)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    make_unschedulable_single_add(here)
    make_valid_tiny_add(here)
    print("Wrote ONNX files to", here)
