
{
   gSystem->Load("libROOTTMVASofieParser");
   TMVA::Experimental::SOFIE::RModelParser_ONNX parser;
   auto model = parser.Parse("/home/mark/Desktop/deep_learning_projects/CERN/root/tmva/sofie_parser_bug_tests/conv_add_fusion_bug.onnx", false);
}
