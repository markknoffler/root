from __future__ import annotations

import os
import unittest

import onnx
from onnx import TensorProto, helper


class TestRNNClipBug(unittest.TestCase):
    def _make_rnn_with_clip(self, out_dir: str, clip_value: float) -> str:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"rnn_clip_{clip_value}.onnx")

        X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 1])
        Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 1])
        W = helper.make_tensor("W", TensorProto.FLOAT, [1, 1, 1], [0.0])
        R = helper.make_tensor("R", TensorProto.FLOAT, [1, 1, 1], [0.0])
        B = helper.make_tensor("B", TensorProto.FLOAT, [1, 2, 1], [0.0, 0.0])

        node = helper.make_node(
            "RNN",
            inputs=["X", "W", "R", "B"],
            outputs=["Y"],
            clip=clip_value,
        )

        graph = helper.make_graph(
            [node],
            "rnn_clip_bug_graph",
            [X],
            [Y],
            initializer=[W, R, B],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.save(model, path)
        return path

    def test_rnn_clip_attribute_float_vs_int_field(self):
        here = os.path.dirname(os.path.abspath(__file__))
        model_path = self._make_rnn_with_clip(here, clip_value=2.5)

        m = onnx.load(model_path)
        self.assertEqual(len(m.graph.node), 1)
        node = m.graph.node[0]
        self.assertEqual(node.op_type, "RNN")

        attrs = [a for a in node.attribute if a.name == "clip"]
        self.assertEqual(len(attrs), 1, "RNN node should have exactly one 'clip' attribute")
        clip_attr = attrs[0]

        self.assertAlmostEqual(clip_attr.f, 2.5, places=6)

        self.assertEqual(
            clip_attr.i,
            0,
            msg=(
                "ONNX 'clip' is stored in the float field (attr.f), "
                "while the integer field (attr.i) is 0. "
                "ParseRNN's use of attribute().i() will therefore drop "
                "the clip value for valid RNN models."
            ),
        )


if __name__ == "__main__":
    unittest.main()

