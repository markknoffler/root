from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import onnx
from onnx import TensorProto, helper


def make_model(path: str) -> None:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [2, 2])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [2, 2])
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [2, 2])
    d = helper.make_tensor_value_info("d", TensorProto.FLOAT, [2, 2])
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 2])
    y2 = helper.make_tensor_value_info("y2", TensorProto.FLOAT, [2, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 2])

    n0 = helper.make_node("Relu", inputs=["q"], outputs=["r"])
    n1 = helper.make_node("MatMul", inputs=["a", "b"], outputs=["m"])
    n2 = helper.make_node("Add", inputs=["m", "c"], outputs=["y"])
    n3 = helper.make_node("Identity", inputs=["x"], outputs=["q"])
    n4 = helper.make_node("Tanh", inputs=["y"], outputs=["y2"])
    n5 = helper.make_node("Add", inputs=["r", "d"], outputs=["z"])

    graph = helper.make_graph(
        [n0, n1, n2, n3, n4, n5],
        "misindexed_children_bug",
        [a, b, c, d, x],
        [y2, z],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, path)


def run_repro(model_path: str) -> int:
    macro = f"""
{{
   gSystem->Load("libROOTTMVASofieParser");
   TMVA::Experimental::SOFIE::RModelParser_ONNX parser;
   auto model = parser.Parse("{model_path}", false);
}}
"""
    with tempfile.NamedTemporaryFile("w", suffix=".C", delete=False, encoding="utf-8") as tmp:
        tmp.write(macro)
        macro_path = tmp.name

    root_exe = "/home/mark/root_install_outside/bin/root"
    return subprocess.run([root_exe, "-l", "-b", "-q", macro_path], check=False).returncode


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(here, "misindexed_children_fusion_bug.onnx")
    make_model(model_path)
    sys.exit(run_repro(model_path))
