//Code generated automatically by TMVA for Inference of Model file [PyTorchModel.pt] at [Fri Mar  6 19:43:47 2026] 

#ifndef ROOT_TMVA_SOFIE_PYTORCHMODEL
#define ROOT_TMVA_SOFIE_PYTORCHMODEL

#include <algorithm>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_PyTorchModel{
namespace BLAS{
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_2bias = std::vector<float>(8);
float * tensor_2bias = fTensor_2bias.data();
std::vector<float> fTensor_0weight = std::vector<float>(512);
float * tensor_0weight = fTensor_0weight.data();
std::vector<float> fTensor_2weight = std::vector<float>(128);
float * tensor_2weight = fTensor_2weight.data();
std::vector<float> fTensor_0bias = std::vector<float>(16);
float * tensor_0bias = fTensor_0bias.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(256);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor input0 with size 128 bytes
float* tensor_input0 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor result with size 128 bytes
float* tensor_result = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 128);

 // Allocating memory for intermediate tensor input2 with size 64 bytes
float* tensor_input2 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 64);

 // Allocating memory for intermediate tensor result3 with size 64 bytes
float* tensor_result3 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);


Session(std::string filename ="PyTorchModel.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_2bias, "tensor_2bias", 8);
   ReadTensorFromStream(f, tensor_0weight, "tensor_0weight", 512);
   ReadTensorFromStream(f, tensor_2weight, "tensor_2weight", 128);
   ReadTensorFromStream(f, tensor_0bias, "tensor_0bias", 16);
   f.close();

}

void doInfer(float const* tensor_input1,  std::vector<float> &output_tensor_result3 ){


//--------- Gemm op_0 { 2 , 32 } * { 16 , 32 } -> { 2 , 16 }
   for (size_t j = 0; j < 2; j++) { 
      size_t y_index = 16 * j;
      for (size_t k = 0; k < 16; k++) { 
         tensor_input0[y_index + k] = tensor_0bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_input0, true, false, 16, 2, 32, 1, tensor_0weight, tensor_input1, 1,nullptr);

//------ RELU
   for (int id = 0; id < 32 ; id++){
      tensor_result[id] = ((tensor_input0[id] > 0 )? tensor_input0[id] : 0);
   }

//--------- Gemm op_2 { 2 , 16 } * { 8 , 16 } -> { 2 , 8 }
   for (size_t j = 0; j < 2; j++) { 
      size_t y_index = 8 * j;
      for (size_t k = 0; k < 8; k++) { 
         tensor_input2[y_index + k] = tensor_2bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_input2, true, false, 8, 2, 16, 1, tensor_2weight, tensor_result, 1,nullptr);

//------ RELU
   for (int id = 0; id < 16 ; id++){
      tensor_result3[id] = ((tensor_input2[id] > 0 )? tensor_input2[id] : 0);
   }
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_result3, output_tensor_result3, 16);
}



std::vector<float> infer(float const* tensor_input1){
   std::vector<float > output_tensor_result3;
   doInfer(tensor_input1, output_tensor_result3 );
   return {output_tensor_result3};
}
};   // end of Session

} //TMVA_SOFIE_PyTorchModel

#endif  // ROOT_TMVA_SOFIE_PYTORCHMODEL
