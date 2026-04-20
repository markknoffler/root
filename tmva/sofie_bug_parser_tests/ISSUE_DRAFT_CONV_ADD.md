## Add a title*

[tmva][sofie] Conv+Add fusion path can drop operator and break downstream Relu type registration

## Check duplicate issues.

Checked for duplicates

## Description*

In SOFIE ONNX parser, the `Conv + Add` fusion path calls `ParseFuseConvAdd`.

In the current broken path, `ParseFuseConvAdd` returns a null operator while the `Add` node is already marked as fused in parser flow. Because of that, the expected fused output tensor type is never registered, and a following `Relu` fails with:

`TMVA::SOFIE ONNX Parser relu op has input tensoradd_out but its type is not yet registered`

Expected behavior:

- Either the Conv+Add fused operator should be fully created and registered.
- Or fusion should be skipped cleanly without corrupting graph parse state.

## Reproducer*

1. Build a small ONNX graph: `Conv -> Add -> Relu`.
2. Run it through `TMVA::Experimental::SOFIE::RModelParser_ONNX::Parse(...)`.
3. Observe runtime failure:
   `TMVA::SOFIE ONNX Parser relu op has input tensoradd_out but its type is not yet registered`

Self-contained reproducer (Python):

```python
from __future__ import annotations

import os
import subprocess
import tempfile

import onnx
from onnx import TensorProto, helper


def make_model(path: str) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    w = helper.make_tensor("w", TensorProto.FLOAT, [1, 1, 3, 3], [1.0] * 9)
    b = helper.make_tensor("b", TensorProto.FLOAT, [1], [0.5])

    conv = helper.make_node("Conv", inputs=["x", "w"], outputs=["conv_out"], kernel_shape=[3, 3], strides=[1, 1])
    add = helper.make_node("Add", inputs=["conv_out", "b"], outputs=["add_out"])
    relu = helper.make_node("Relu", inputs=["add_out"], outputs=["y"])

    graph = helper.make_graph([conv, add, relu], "conv_add_fusion_bug", [x], [y], initializer=[w, b])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def run_with_root(model_path: str) -> int:
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
    return subprocess.run(["root", "-l", "-b", "-q", macro_path], check=False).returncode


if __name__ == "__main__":
    model_path = os.path.abspath("conv_add_fusion_bug.onnx")
    make_model(model_path)
    print(run_with_root(model_path))
```

How to run:

`python reproducer.py`

## ROOT version*

`6.39.01`

Built for `linuxx8664gcc` on Apr 15 2026, 21:05:13.

From `heads/bug_hunting@f8c45e358b`.

With `c++ (GCC) 15.2.1 20260209` `std201703`.

## Installation method*

Local source build (CMake + Ninja), tested with dedicated external install/build trees.

## Operating system*

Arch Linux (kernel `6.19.11-arch1-1`)

## Additional context

The fix is implemented by providing a complete `ParseFuseConvAdd` implementation so fused Conv+Add produces a valid operator and registers fused output type.

Respected @lmoneta and @sanjibansg, kindly let me know if you also want me to provide a ROOT-standard regression test placement for this case (`test/` or `roottest`).

With regards,  
Samreedh Bhuyan
