// test_cpp_pipeline.C
// Demonstrates the full Exercise 4 pipeline:
// Python parser → JSON → ParseFromPython() → RModel → inference .hxx
// Run: root -l -b -q tmva/sofie_pytorch_parser/tests/test_cpp_pipeline.C

void test_cpp_pipeline() {
    using namespace TMVA::Experimental::SOFIE;

    std::string outDir = "tmva/sofie/exercise3_outputs/";

    // ── GRU model (most complex operator) ────────────────────────────────
    std::cout << "\n=== Parsing GRU model from JSON ===" << std::endl;
    std::vector<std::vector<size_t>> gruShape{{1, 5, 8}};
    RModel gru_model = PyTorch::ParseFromPython(outDir + "gru_model.json", gruShape);
    gru_model.Generate();
    gru_model.OutputGenerated(outDir + "GRUModel_sofie.hxx");
    std::cout << "GRU: Generated " << outDir << "GRUModel_sofie.hxx" << std::endl;
    gru_model.PrintRequiredInputTensors();
    gru_model.PrintInitializedTensors();

    // ── LSTM model ────────────────────────────────────────────────────────
    std::cout << "\n=== Parsing LSTM model from JSON ===" << std::endl;
    std::vector<std::vector<size_t>> lstmShape{{1, 5, 8}};
    RModel lstm_model = PyTorch::ParseFromPython(outDir + "lstm_model.json", lstmShape);
    lstm_model.Generate();
    lstm_model.OutputGenerated(outDir + "LSTMModel_sofie.hxx");
    std::cout << "LSTM: Generated " << outDir << "LSTMModel_sofie.hxx" << std::endl;

    // ── Tutorial model via new parser ─────────────────────────────────────
    std::cout << "\n=== Parsing Tutorial model (Linear+ReLU) via new parser ===" << std::endl;
    std::vector<std::vector<size_t>> denseShape{{2, 32}};
    RModel dense_model = PyTorch::ParseFromPython(outDir + "TutorialModel_newparser.json", denseShape);
    dense_model.Generate();
    dense_model.OutputGenerated(outDir + "TutorialModel_newparser.hxx");
    std::cout << "Dense: Generated " << outDir << "TutorialModel_newparser.hxx" << std::endl;
    dense_model.PrintRequiredInputTensors();

    std::cout << "\n=== All models parsed and generated successfully! ===" << std::endl;
}

