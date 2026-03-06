//Code generated automatically by TMVA for Inference of Model file [KerasModel.keras] at [Fri Mar  6 19:43:39 202] 

#ifndef ROOT_TMVA_SOFIE_KERASMODEL
#define ROOT_TMVA_SOFIE_KERASMODEL

#include <algorithm>
#include <cmath>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_KerasModel{
namespace BLAS{
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_dense_3kernel = std::vector<float>(16);
float * tensor_dense_3kernel = fTensor_dense_3kernel.data();
std::vector<float> fTensor_dense_2bias = std::vector<float>(8);
float * tensor_dense_2bias = fTensor_dense_2bias.data();
std::vector<float> fTensor_dense_3bias = std::vector<float>(2);
float * tensor_dense_3bias = fTensor_dense_3bias.data();
std::vector<float> fTensor_dense_2kernel = std::vector<float>(128);
float * tensor_dense_2kernel = fTensor_dense_2kernel.data();
std::vector<float> fTensor_dense_1bias = std::vector<float>(16);
float * tensor_dense_1bias = fTensor_dense_1bias.data();
std::vector<float> fTensor_dense_1kernel = std::vector<float>(512);
float * tensor_dense_1kernel = fTensor_dense_1kernel.data();
std::vector<float> fTensor_densebias = std::vector<float>(32);
float * tensor_densebias = fTensor_densebias.data();
std::vector<float> fTensor_densekernel = std::vector<float>(128);
float * tensor_densekernel = fTensor_densekernel.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(512);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor keras_tensor_8 with size 256 bytes
float* tensor_keras_tensor_8 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor keras_tensor_10 with size 256 bytes
float* tensor_keras_tensor_10 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 256);

 // Allocating memory for intermediate tensor dense_1Dense with size 128 bytes
float* tensor_dense_1Dense = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 128);

 // Allocating memory for intermediate tensor keras_tensor_12 with size 128 bytes
float* tensor_keras_tensor_12 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor dense_2Dense with size 64 bytes
float* tensor_dense_2Dense = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 448);

 // Allocating memory for intermediate tensor keras_tensor_14 with size 64 bytes
float* tensor_keras_tensor_14 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 384);

 // Allocating memory for intermediate tensor keras_tensor_16 with size 16 bytes
float* tensor_keras_tensor_16 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 368);

 // Allocating memory for intermediate tensor keras_tensor_18 with size 16 bytes
float* tensor_keras_tensor_18 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 352);


Session(std::string filename ="KerasModel.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_dense_3kernel, "tensor_dense_3kernel", 16);
   ReadTensorFromStream(f, tensor_dense_2bias, "tensor_dense_2bias", 8);
   ReadTensorFromStream(f, tensor_dense_3bias, "tensor_dense_3bias", 2);
   ReadTensorFromStream(f, tensor_dense_2kernel, "tensor_dense_2kernel", 128);
   ReadTensorFromStream(f, tensor_dense_1bias, "tensor_dense_1bias", 16);
   ReadTensorFromStream(f, tensor_dense_1kernel, "tensor_dense_1kernel", 512);
   ReadTensorFromStream(f, tensor_densebias, "tensor_densebias", 32);
   ReadTensorFromStream(f, tensor_densekernel, "tensor_densekernel", 128);
   f.close();

}

void doInfer(float const* tensor_input_layer,  std::vector<float> &output_tensor_keras_tensor_18 ){


//--------- Gemm op_0 { 2 , 4 } * { 4 , 32 } -> { 2 , 32 }
   for (size_t j = 0; j < 2; j++) { 
      size_t y_index = 32 * j;
      for (size_t k = 0; k < 32; k++) { 
         tensor_keras_tensor_8[y_index + k] = tensor_densebias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_keras_tensor_8, false, false, 32, 2, 4, 1, tensor_densekernel, tensor_input_layer, 1,nullptr);

//------ RELU
   for (int id = 0; id < 64 ; id++){
      tensor_keras_tensor_10[id] = ((tensor_keras_tensor_8[id] > 0 )? tensor_keras_tensor_8[id] : 0);
   }

//--------- Gemm op_2 { 2 , 32 } * { 32 , 16 } -> { 2 , 16 }
   for (size_t j = 0; j < 2; j++) { 
      size_t y_index = 16 * j;
      for (size_t k = 0; k < 16; k++) { 
         tensor_dense_1Dense[y_index + k] = tensor_dense_1bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_dense_1Dense, false, false, 16, 2, 32, 1, tensor_dense_1kernel, tensor_keras_tensor_10, 1,nullptr);

//------ RELU
   for (int id = 0; id < 32 ; id++){
      tensor_keras_tensor_12[id] = ((tensor_dense_1Dense[id] > 0 )? tensor_dense_1Dense[id] : 0);
   }

//--------- Gemm op_4 { 2 , 16 } * { 16 , 8 } -> { 2 , 8 }
   for (size_t j = 0; j < 2; j++) { 
      size_t y_index = 8 * j;
      for (size_t k = 0; k < 8; k++) { 
         tensor_dense_2Dense[y_index + k] = tensor_dense_2bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_dense_2Dense, false, false, 8, 2, 16, 1, tensor_dense_2kernel, tensor_keras_tensor_12, 1,nullptr);

//------ RELU
   for (int id = 0; id < 16 ; id++){
      tensor_keras_tensor_14[id] = ((tensor_dense_2Dense[id] > 0 )? tensor_dense_2Dense[id] : 0);
   }

//--------- Gemm op_6 { 2 , 8 } * { 8 , 2 } -> { 2 , 2 }
   for (size_t j = 0; j < 2; j++) { 
      size_t y_index = 2 * j;
      for (size_t k = 0; k < 2; k++) { 
         tensor_keras_tensor_16[y_index + k] = tensor_dense_3bias[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_keras_tensor_16, false, false, 2, 2, 8, 1, tensor_dense_3kernel, tensor_keras_tensor_14, 1,nullptr);

   //------ SOFTMAX - 2  4  1
   for (int i = 0; i < 2; ++i) {
      size_t offset = i * 2;
      float const * x_ptr = &tensor_keras_tensor_16[offset];
      float * y_ptr = &tensor_keras_tensor_18[offset];
      float vmax = x_ptr[0];
      for (int j = 1; j < 2; ++j) {
         if (x_ptr[j] > vmax) vmax = x_ptr[j];
      }
      float sum = 0.0;
      for (int j = 0; j < 2; ++j) {
         y_ptr[j] = std::exp(x_ptr[j] - vmax);
         sum += y_ptr[j];
      }
      float inv_sum = 1.0f / sum;
      for (int j = 0; j < 2; ++j) {
         y_ptr[j] *= inv_sum;
      }
   }
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_keras_tensor_18, output_tensor_keras_tensor_18, 4);
}



std::vector<float> infer(float const* tensor_input_layer){
   std::vector<float > output_tensor_keras_tensor_18;
   doInfer(tensor_input_layer, output_tensor_keras_tensor_18 );
   return {output_tensor_keras_tensor_18};
}
};   // end of Session

} //TMVA_SOFIE_KerasModel

#endif  // ROOT_TMVA_SOFIE_KERASMODEL
