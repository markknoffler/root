#include "TInterpreter.h"
#include "TROOT.h"
#include "TMVA/RModel.hxx"
#include "TMVA/RModelParser_ONNX.hxx"

#include <cassert>
#include <cmath>
#include <vector>

void test_rnn_clip_runtime_bug() {
   const std::string modelName = "rnn_clip_runtime_bug";
   const std::string modelPath = modelName + ".onnx";
   const std::string headerPath = modelName + ".hxx";

   TMVA::Experimental::SOFIE::RModelParser_ONNX parser;
   auto model = parser.Parse(modelPath, false);
   model.Generate();
   model.OutputGenerated(headerPath);

   const std::string includeCode =
      "#include \"" + headerPath + "\"\n"
      "std::vector<std::vector<float>> run_rnn_clip_runtime_bug(float *x) {\n"
      "  TMVA_SOFIE_rnn_clip_runtime_bug::Session s(\"rnn_clip_runtime_bug.dat\");\n"
      "  return s.infer(x);\n"
      "}\n";
   bool ok = gInterpreter->Declare(includeCode.c_str());
   assert(ok && "Failed to declare generated SOFIE RNN session");

   std::vector<float> input = {10.0f};
   const auto cmd = TString::Format(
      "new std::vector<std::vector<float>>(run_rnn_clip_runtime_bug((float*)0x%lx))",
      (ULong_t)input.data());
   auto *output = reinterpret_cast<std::vector<std::vector<float>> *>(gROOT->ProcessLine(cmd));

   assert(output && "Failed to run generated SOFIE RNN inference");
   assert(output->size() == 2 && "Expected Y and Y_h outputs from RNN");
   assert((*output)[1].size() == 1 && "Expected scalar Y_h for this one-step one-hidden RNN");

   const float actual = (*output)[1][0];
   const float expected = std::tanh(0.5f);

   // This is the actual bug trigger:
   // current ParseRNN reads clip with attribute().i(), so clip becomes 0 and
   // the generated model behaves like tanh(10) instead of tanh(0.5).
   assert(std::fabs(actual - expected) < 1e-4f &&
          "RNN clip bug triggered: ParseRNN ignored clip=0.5, so SOFIE inference did not match clipped output.");
}

