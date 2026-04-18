//Code generated automatically by TMVA for Inference of Model file [topk_double.onnx] at [Thu Apr 16 20:09:46 2026] 

#ifndef ROOT_TMVA_SOFIE_TOPK_DOUBLE
#define ROOT_TMVA_SOFIE_TOPK_DOUBLE

#include <vector>
#include "TMVA/SOFIE_common.hxx"

namespace TMVA_SOFIE_topk_double{
struct Session {
// initialized (weights and constant) tensors

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(32);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor Y_val with size 16 bytes
double* tensor_Y_val = reinterpret_cast<double*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor Y_idx with size 16 bytes
int64_t* tensor_Y_idx = reinterpret_cast<int64_t*>(fIntermediateMemoryPool.data() + 16);


Session(std::string = "") {
}

void doInfer(double const* tensor_X,  std::vector<double> &output_tensor_Y_val, std::vector<int64_t> &output_tensor_Y_idx ){


   //------ TopK
   {
   std::vector<std::pair<float,int64_t>> elements(4);
   size_t xoffset = 0;
   size_t yoffset = 0;
   const size_t j = 0;
      for (size_t l = 0; l < 4; l++) {
         elements[l] = std::make_pair(tensor_X[xoffset + 1*l + j], l);
      }
      std::partial_sort(elements.begin(),elements.begin()+2,elements.end(),[](std::pair<float,int64_t>a,std::pair<float,int64_t>b){return (a.first!=b.first) ? (a.first>b.first) : a.second < b.second;});
      for (size_t l = 0; l < 2; l++) {
         tensor_Y_val[yoffset + 1*l + j] = elements[l].first;
         tensor_Y_idx[yoffset + 1*l + j] = elements[l].second;
      }
   }
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_Y_val, output_tensor_Y_val, 2);
   FillOutput(tensor_Y_idx, output_tensor_Y_idx, 2);
}



std::tuple<std::vector<double>,std::vector<int64_t>> infer(double const* tensor_X){
   std::vector<double > output_tensor_Y_val;
   std::vector<int64_t > output_tensor_Y_idx;
   doInfer(tensor_X, output_tensor_Y_val, output_tensor_Y_idx );
   return {output_tensor_Y_val,output_tensor_Y_idx};
}
};   // end of Session

} //TMVA_SOFIE_topk_double

#endif  // ROOT_TMVA_SOFIE_TOPK_DOUBLE
