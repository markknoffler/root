from __future__ import annotations

import os

import onnx
from onnx import TensorProto, helper


def write_model(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "concat_mixed_types.onnx")

    a = helper.make_tensor_value_info("A", TensorProto.FLOAT, [2])
    b = helper.make_tensor_value_info("B", TensorProto.INT64, [2])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [4])  # type here is irrelevant for parsing

    node = helper.make_node("Concat", inputs=["A", "B"], outputs=["Y"], axis=0)
    graph = helper.make_graph([node], "concat_mixed_types", [a, b], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)
    return path


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print(write_model(here))

