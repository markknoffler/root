from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import onnx
from onnx import TensorProto, helper


def make_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])

    w = helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 3, 3], [1.0] * 9)
    b = helper.make_tensor("b", TensorProto.FLOAT, [1], [0.5])

    conv = helper.make_node(
        "Conv",
        inputs=["x", "w"],
        outputs=["conv_out"],
        kernel_shape=[3, 3],
        strides=[1, 1],
    )
    add = helper.make_node("Add", inputs=["conv_out", "b"], outputs=["add_out"])
    relu = helper.make_node("Relu", inputs=["add_out"], outputs=["y"])

    graph = helper.make_graph([conv, add, relu], "conv_add_fusion_bug", [x], [y], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
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
    model_path = os.path.join(here, "conv_add_fusion_bug.onnx")
    make_model(model_path)
    sys.exit(run_repro(model_path))
