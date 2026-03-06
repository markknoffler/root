//Code generated automatically by TMVA for Inference of Model file [tmva/sofie/exercise3_outputs/TutorialModel_newparser.json] at [Fri Mar  6 22:02:55 2026] 

#ifndef ROOT_TMVA_SOFIE_TMVASOFIEEXERCISE3_OUTPUTSTUTORIALMODEL_NEWPARSER
#define ROOT_TMVA_SOFIE_TMVASOFIEEXERCISE3_OUTPUTSTUTORIALMODEL_NEWPARSER

#include <algorithm>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_tmvasofieexercise3_outputsTutorialModel_newparser{
namespace BLAS{
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_2_bias = std::vector<float>(8);
float * tensor_2_bias = fTensor_2_bias.data();
std::vector<float> fTensor_2_weight = std::vector<float>(128);
float * tensor_2_weight = fTensor_2_weight.data();
std::vector<float> fTensor_0_bias = std::vector<float>(16);
float * tensor_0_bias = fTensor_0_bias.data();
std::vector<float> fTensor_0_weight = std::vector<float>(512);
float * tensor_0_weight = fTensor_0_weight.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(256);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor out_0_1 with size 128 bytes
float* tensor_out_0_1 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor out_1_2 with size 128 bytes
float* tensor_out_1_2 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 128);

 // Allocating memory for intermediate tensor out_2_3 with size 64 bytes
float* tensor_out_2_3 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 64);

 // Allocating memory for intermediate tensor out_3_4 with size 64 bytes
float* tensor_out_3_4 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);


Session(std::string filename ="tmvasofieexercise3_outputsTutorialModel_newparser.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_2_bias, "tensor_2_bias", 8);
   ReadTensorFromStream(f, tensor_2_weight, "tensor_2_weight", 128);
   ReadTensorFromStream(f, tensor_0_bias, "tensor_0_bias", 16);
   ReadTensorFromStream(f, tensor_0_weight, "tensor_0_weight", 512);
   f.close();

}

void doInfer(float const* tensor_input_0,  std::vector<float> &output_tensor_out_3_4 ){


//--------- Gemm op_0 { 2 , 32 } * { 16 , 32 } -> { 2 , 16 }
   for (size_t j = 0; j < 2; j++) { 
      size_t y_index = 16 * j;
      for (size_t k = 0; k < 16; k++) { 
         tensor_out_0_1[y_index + k] = tensor_0_bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_out_0_1, true, false, 16, 2, 32, 1, tensor_0_weight, tensor_input_0, 1,nullptr);

//------ RELU
   for (int id = 0; id < 32 ; id++){
      tensor_out_1_2[id] = ((tensor_out_0_1[id] > 0 )? tensor_out_0_1[id] : 0);
   }

//--------- Gemm op_2 { 2 , 16 } * { 8 , 16 } -> { 2 , 8 }
   for (size_t j = 0; j < 2; j++) { 
      size_t y_index = 8 * j;
      for (size_t k = 0; k < 8; k++) { 
         tensor_out_2_3[y_index + k] = tensor_2_bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_out_2_3, true, false, 8, 2, 16, 1, tensor_2_weight, tensor_out_1_2, 1,nullptr);

//------ RELU
   for (int id = 0; id < 16 ; id++){
      tensor_out_3_4[id] = ((tensor_out_2_3[id] > 0 )? tensor_out_2_3[id] : 0);
   }
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_out_3_4, output_tensor_out_3_4, 16);
}



std::vector<float> infer(float const* tensor_input_0){
   std::vector<float > output_tensor_out_3_4;
   doInfer(tensor_input_0, output_tensor_out_3_4 );
   return {output_tensor_out_3_4};
}
};   // end of Session

} //TMVA_SOFIE_tmvasofieexercise3_outputsTutorialModel_newparser

#endif  // ROOT_TMVA_SOFIE_TMVASOFIEEXERCISE3_OUTPUTSTUTORIALMODEL_NEWPARSER
