
{
   gSystem->Load("libROOTTMVASofieParser");
   TMVA::Experimental::SOFIE::RModelParser_ONNX p;
   p.CheckModel("/home/mark/Desktop/deep_learning_projects/CERN/root/tmva/sofie_parser_bug_tests/valid_tiny_add.onnx", false);
   p.CheckModel("/home/mark/Desktop/deep_learning_projects/CERN/root/tmva/sofie_parser_bug_tests/valid_tiny_add.onnx", false);
}
