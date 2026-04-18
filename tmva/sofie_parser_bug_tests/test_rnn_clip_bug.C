#include "/home/mark/root_build_outside/tmva/sofie_parsers/onnx_proto3.pb.h"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>

void test_rnn_clip_bug() {
   gSystem->Load("libROOTTMVASofieParser");
   const char *modelPath = "rnn_clip_2.5.onnx";

   // Load ONNX model using the same proto type used by SOFIE.
   onnx::ModelProto model;
   {
      std::fstream input(modelPath, std::ios::in | std::ios::binary);
      bool ok = model.ParseFromIstream(&input);
      assert(ok && "Failed to parse ONNX file rnn_clip_2.5.onnx");
   }

   assert(model.graph().node_size() == 1 && "Expected exactly one node in the graph");
   const onnx::NodeProto &node = model.graph().node(0);
   assert(node.op_type() == "RNN" && "Expected node of type RNN");

   // Find the 'clip' attribute.
   const onnx::AttributeProto *clipAttr = nullptr;
   for (int i = 0; i < node.attribute_size(); ++i) {
      if (node.attribute(i).name() == "clip") {
         clipAttr = &node.attribute(i);
         break;
      }
   }
   assert(clipAttr && "RNN node must have a 'clip' attribute");

   // ONNX stores the clip in the float field.
   float f = clipAttr->f();
   std::int64_t i = clipAttr->i();

   // These assertions are the actual bug trigger:
   // - The float field contains the real value (e.g. 2.5f),
   // - The integer field is zero today, but if the implementation were correct,
   //   we would either not rely on .i() at all or keep it consistent.
   assert(f == 2.5f && "Expected clip float field (attr.f) to be 2.5f");
   assert(
      i == 0 &&
      "Expected integer field (attr.i) of a float clip attribute to be 0; "
      "ParseRNN's use of attribute().i() will therefore drop the clip value."
   );
}

