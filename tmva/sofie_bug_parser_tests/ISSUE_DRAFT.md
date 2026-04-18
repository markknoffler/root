## Add a title*

[tmva][sofie] Wrong fusion child lookup after ONNX graph reordering

## Check duplicate issues.

Checked for duplicates

## Description*

In SOFIE ONNX parser (`RModelParser_ONNX`), `nodesChildren` is keyed by original ONNX node index, but during parsing it was being accessed with the parse-loop slot index.

This can cause wrong child selection for fusion when graph node order is non-topological and reordering is applied (`nodesOrder` differs from identity). In this case parser can try to fuse a node with an unrelated child and throw runtime errors such as:

`TMVA::SOFIE ONNX Parser : cannot fuse MatMul and Add since have different inputs`

Important note:

- For standard-compliant ONNX model creation, this is rare because ONNX checker requires topological ordering.
- I verified this with `onnx.checker.check_model`, which returns:
  `ValidationError: Nodes in a graph must be topologically sorted ...`
- Still, ROOT parser has explicit reordering logic, and non-topological protobuf-serialized/manual ONNX inputs can reach this path and trigger the bug.

Expected behavior:

Fusion child lookup should use the current reordered node id (`nodesOrder[i]`), so the child list belongs to the node actually being parsed.

## Reproducer*

1. Create a non-topological ONNX model where:
   - one valid `MatMul -> Add` pair exists
   - another unrelated `Add` exists
   - node list order is intentionally not topological
2. Run `TMVA::Experimental::SOFIE::RModelParser_ONNX::Parse(...)` through ROOT.
3. Observe runtime failure:
   `TMVA::SOFIE ONNX Parser : cannot fuse MatMul and Add since have different inputs`

Self-contained reproducer (Python):

```python
from __future__ import annotations

import os
import subprocess
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
    out = os.path.abspath("misindexed_children_fusion_bug.onnx")
    make_model(out)
    code = run_with_root(out)
    print("root exit code:", code)
```

How to run:

`python reproducer.py`

## ROOT version*

`6.39.01`

Built for `linuxx8664gcc` on Apr 15 2026, 21:05:13.

From `heads/bug_hunting@f8c45e358b`.

With `c++ (GCC) 15.2.1 20260209` `std201703`.

## Installation method*

Local source build (CMake + Ninja), tested with external install/build trees.

## Operating system*

Arch Linux (kernel `6.19.11-arch1-1`)

## Additional context

The fix is a one-line index correction in parse loop child lookup:

- incorrect: `nodesChildren[i]`
- correct: `nodesChildren[nodesOrder[i]]`

Respected @lmoneta and @sanjibansg, kindly let me know if you also want me to provide a ROOT-standard regression test placement for this case (`test/` or `roottest`).

With regards,  
Samreedh Bhuyan
