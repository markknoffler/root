// Exercise 4: C++ pipeline demo using ParseFromPython()
// Run: root -l -b -q tmva/sofie_pytorch_parser/tests/test_cpp_pipeline.C

void test_cpp_pipeline() {
    using namespace TMVA::Experimental::SOFIE;

    std::string outDir = "tmva/sofie/exercise3_outputs/";

    // GRU model
    std::cout << "\n=== Parsing GRU model ===" << std::endl;
    RModel gru = PyTorch::ParseFromPython(outDir+"gru_model.json", {{1,5,8}});
    gru.Generate();
    gru.OutputGenerated(outDir+"GRUModel_sofie.hxx");
    gru.PrintRequiredInputTensors();

    // LSTM model
    std::cout << "\n=== Parsing LSTM model ===" << std::endl;
    RModel lstm = PyTorch::ParseFromPython(outDir+"lstm_model.json", {{1,5,8}});
    lstm.Generate();
    lstm.OutputGenerated(outDir+"LSTMModel_sofie.hxx");

    // Tutorial dense model via new parser
    std::cout << "\n=== Parsing Tutorial model via new parser ===" << std::endl;
    RModel dense = PyTorch::ParseFromPython(outDir+"TutorialModel_newparser.json", {{2,32}});
    dense.Generate();
    dense.OutputGenerated(outDir+"TutorialModel_newparser.hxx");
    dense.PrintRequiredInputTensors();
    dense.PrintInitializedTensors();

    std::cout << "\n=== All models generated successfully ===" << std::endl;
}
