//Code generated automatically by TMVA for Inference of Model file [HiggsModel.keras] at [Fri Mar  6 19:43:44 202] 

#ifndef ROOT_TMVA_SOFIE_HIGGSMODEL
#define ROOT_TMVA_SOFIE_HIGGSMODEL

#include <algorithm>
#include <cmath>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_HiggsModel{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_dense_2bias = std::vector<float>(1);
float * tensor_dense_2bias = fTensor_dense_2bias.data();
std::vector<float> fTensor_dense_2kernel = std::vector<float>(64);
float * tensor_dense_2kernel = fTensor_dense_2kernel.data();
std::vector<float> fTensor_dense_1bias = std::vector<float>(64);
float * tensor_dense_1bias = fTensor_dense_1bias.data();
std::vector<float> fTensor_dense_1kernel = std::vector<float>(4096);
float * tensor_dense_1kernel = fTensor_dense_1kernel.data();
std::vector<float> fTensor_densebias = std::vector<float>(64);
float * tensor_densebias = fTensor_densebias.data();
std::vector<float> fTensor_densekernel = std::vector<float>(448);
float * tensor_densekernel = fTensor_densekernel.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(512);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor denseDense with size 256 bytes
float* tensor_denseDense = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor keras_tensor_5 with size 256 bytes
float* tensor_keras_tensor_5 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 256);

 // Allocating memory for intermediate tensor dense_1Dense with size 256 bytes
float* tensor_dense_1Dense = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor keras_tensor_7 with size 256 bytes
float* tensor_keras_tensor_7 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 256);

 // Allocating memory for intermediate tensor dense_2Dense with size 4 bytes
float* tensor_dense_2Dense = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 252);

 // Allocating memory for intermediate tensor keras_tensor_9 with size 4 bytes
float* tensor_keras_tensor_9 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 248);


Session(std::string filename ="HiggsModel.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_dense_2bias, "tensor_dense_2bias", 1);
   ReadTensorFromStream(f, tensor_dense_2kernel, "tensor_dense_2kernel", 64);
   ReadTensorFromStream(f, tensor_dense_1bias, "tensor_dense_1bias", 64);
   ReadTensorFromStream(f, tensor_dense_1kernel, "tensor_dense_1kernel", 4096);
   ReadTensorFromStream(f, tensor_densebias, "tensor_densebias", 64);
   ReadTensorFromStream(f, tensor_densekernel, "tensor_densekernel", 448);
   f.close();

}

void doInfer(float const* tensor_input_layer,  std::vector<float> &output_tensor_keras_tensor_9 ){


//--------- Gemm op_0 { 1 , 7 } * { 7 , 64 } -> { 1 , 64 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 64 * j;
      for (size_t k = 0; k < 64; k++) { 
         tensor_denseDense[y_index + k] = tensor_densebias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_denseDense, false, false, 64, 1, 7, 1, tensor_densekernel, tensor_input_layer, 1,nullptr);

//------ RELU
   for (int id = 0; id < 64 ; id++){
      tensor_keras_tensor_5[id] = ((tensor_denseDense[id] > 0 )? tensor_denseDense[id] : 0);
   }

//--------- Gemm op_2 { 1 , 64 } * { 64 , 64 } -> { 1 , 64 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 64 * j;
      for (size_t k = 0; k < 64; k++) { 
         tensor_dense_1Dense[y_index + k] = tensor_dense_1bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_dense_1Dense, false, false, 64, 1, 64, 1, tensor_dense_1kernel, tensor_keras_tensor_5, 1,nullptr);

//------ RELU
   for (int id = 0; id < 64 ; id++){
      tensor_keras_tensor_7[id] = ((tensor_dense_1Dense[id] > 0 )? tensor_dense_1Dense[id] : 0);
   }

//--------- Gemm op_4 { 1 , 64 } * { 64 , 1 } -> { 1 , 1 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = j;
      for (size_t k = 0; k < 1; k++) { 
         tensor_dense_2Dense[y_index + k] = tensor_dense_2bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_dense_2Dense, false, false, 1, 1, 64, 1, tensor_dense_2kernel, tensor_keras_tensor_7, 1,nullptr);

//------ Sigmoid -- 5
   for (int id = 0; id < 1 ; id++){
      tensor_keras_tensor_9[id] = 1 / (1 + std::exp( - tensor_dense_2Dense[id]));
   }
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_keras_tensor_9, output_tensor_keras_tensor_9, 1);
}



std::vector<float> infer(float const* tensor_input_layer){
   std::vector<float > output_tensor_keras_tensor_9;
   doInfer(tensor_input_layer, output_tensor_keras_tensor_9 );
   return {output_tensor_keras_tensor_9};
}
};   // end of Session

} //TMVA_SOFIE_HiggsModel

#endif  // ROOT_TMVA_SOFIE_HIGGSMODEL
