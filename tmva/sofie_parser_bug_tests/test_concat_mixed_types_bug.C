#include "TMVA/RModelParser_ONNX.hxx"

#include <cassert>
#include <string>

void test_concat_mixed_types_bug() {
   const std::string modelPath = "concat_mixed_types.onnx";

   TMVA::Experimental::SOFIE::RModelParser_ONNX parser;

   bool threw = false;
   try {
      (void)parser.Parse(modelPath, false);
   } catch (const std::exception &) {
      threw = true;
   }

   // Correct behavior: parser must reject Concat with mixed element types.
   // Current bug: ParseConcat only checks types via assert(), which is compiled out
   // in Release builds => parser.Parse(...) succeeds and we hit this assert.
   assert(threw && "BUG TRIGGERED: ParseConcat accepted mixed input dtypes (FLOAT + INT64) in Release build.");
}

