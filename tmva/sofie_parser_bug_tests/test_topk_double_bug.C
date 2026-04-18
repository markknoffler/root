#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

#include <cassert>
#include <string>

void test_topk_double_bug() {
   const std::string modelPath = "topk_double.onnx";

   TMVA::Experimental::SOFIE::RModelParser_ONNX parser;

   bool threw = false;
   try {
      auto model = parser.Parse(modelPath, false);
      model.Generate();
      model.OutputGenerated("topk_double.hxx");
   } catch (const std::exception &) {
      threw = true;
   }

   // Correct behavior should reject unsupported/non-float TopK at parse time.
   // Current parser bug hardcodes ROperator_TopK<float> while registering output
   // as input dtype, so DOUBLE may pass parse/generate inconsistently.
   assert(threw && "BUG TRIGGERED: TopK DOUBLE model was accepted instead of being rejected explicitly.");
}

