
#include "TMVA/RModelParser_ONNX.hxx"
#include <iostream>
void sofie_parse_repro() {
    TMVA::Experimental::SOFIE::RModelParser_ONNX parser;
    try {
        auto model = parser.Parse("/home/mark/Desktop/deep_learning_projects/CERN/root/tmva/sofie_parser_bug_tests/valid_tiny_add.onnx", false);
        std::cout << "PARSE_OK" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "PARSE_EXCEPTION: " << e.what() << std::endl;
        throw;
    }
}
