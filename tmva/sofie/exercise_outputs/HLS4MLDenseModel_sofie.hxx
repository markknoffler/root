//Code generated automatically by TMVA for Inference of Model file [HLS4MLDenseModel] at [Thu Mar 12 16:09:52 202] 

#ifndef ROOT_TMVA_SOFIE_HLS4MLDENSEMODEL
#define ROOT_TMVA_SOFIE_HLS4MLDENSEMODEL

#include <algorithm>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_HLS4MLDenseModel{
namespace BLAS{
	extern "C" void sgemv_(const char * trans, const int * m, const int * n, const float * alpha, const float * A,
	                       const int * lda, const float * X, const int * incx, const float * beta, const float * Y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_dense_2_B = std::vector<float>(4);
float * tensor_dense_2_B = fTensor_dense_2_B.data();
std::vector<float> fTensor_dense_2_W = std::vector<float>(32);
float * tensor_dense_2_W = fTensor_dense_2_W.data();
std::vector<float> fTensor_dense_1_B = std::vector<float>(8);
float * tensor_dense_1_B = fTensor_dense_1_B.data();
std::vector<float> fTensor_dense_1_W = std::vector<float>(128);
float * tensor_dense_1_W = fTensor_dense_1_W.data();
std::vector<float> fTensor_dense_B = std::vector<float>(16);
float * tensor_dense_B = fTensor_dense_B.data();
std::vector<float> fTensor_dense_W = std::vector<float>(128);
float * tensor_dense_W = fTensor_dense_W.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(128);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor dense with size 64 bytes
float* tensor_dense = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor dense_relu with size 64 bytes
float* tensor_dense_relu = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 64);

 // Allocating memory for intermediate tensor dense_1 with size 32 bytes
float* tensor_dense_1 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 32);

 // Allocating memory for intermediate tensor dense_1_elu with size 32 bytes
float* tensor_dense_1_elu = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor dense_2 with size 16 bytes
float* tensor_dense_2 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 112);


Session(std::string filename ="HLS4MLDenseModel.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_dense_2_B, "tensor_dense_2_B", 4);
   ReadTensorFromStream(f, tensor_dense_2_W, "tensor_dense_2_W", 32);
   ReadTensorFromStream(f, tensor_dense_1_B, "tensor_dense_1_B", 8);
   ReadTensorFromStream(f, tensor_dense_1_W, "tensor_dense_1_W", 128);
   ReadTensorFromStream(f, tensor_dense_B, "tensor_dense_B", 16);
   ReadTensorFromStream(f, tensor_dense_W, "tensor_dense_W", 128);
   f.close();

}

void doInfer(float const* tensor_input_layer,  std::vector<float> &output_tensor_dense_2 ){


//--------- Gemm op_0 { 1 , 1 } * { 8 , 16 } -> { 1 , 16 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 16 * j;
      for (size_t k = 0; k < 16; k++) { 
         tensor_dense[y_index + k] = tensor_dense_B[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_dense, false, false, 16, 1, 1, 1, tensor_dense_W, tensor_input_layer, 1,nullptr);

//------ RELU
   for (int id = 0; id < 16 ; id++){
      tensor_dense_relu[id] = ((tensor_dense[id] > 0 )? tensor_dense[id] : 0);
   }

//--------- Gemm op_2 { 1 , 16 } * { 16 , 8 } -> { 1 , 8 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 8 * j;
      for (size_t k = 0; k < 8; k++) { 
         tensor_dense_1[y_index + k] = tensor_dense_1_B[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_dense_1, false, false, 8, 1, 16, 1, tensor_dense_1_W, tensor_dense_relu, 1,nullptr);
   float op_3_alpha = 1;

//------ ELU 
   for (int id = 0; id < 8 ; id++){
      tensor_dense_1_elu[id] = ((tensor_dense_1[id] >= 0 )? tensor_dense_1[id] : op_3_alpha * std::exp(tensor_dense_1[id]) - 1);
   }

//--------- Gemm op_4 { 1 , 8 } * { 8 , 4 } -> { 1 , 4 }
   for (size_t j = 0; j < 1; j++) { 
      size_t y_index = 4 * j;
      for (size_t k = 0; k < 4; k++) { 
         tensor_dense_2[y_index + k] = tensor_dense_2_B[k];
      }
   }
   TMVA::Experimental::SOFIE::Gemm_Call(tensor_dense_2, false, false, 4, 1, 8, 1, tensor_dense_2_W, tensor_dense_1_elu, 1,nullptr);
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_dense_2, output_tensor_dense_2, 4);
}



std::vector<float> infer(float const* tensor_input_layer){
   std::vector<float > output_tensor_dense_2;
   doInfer(tensor_input_layer, output_tensor_dense_2 );
   return {output_tensor_dense_2};
}
};   // end of Session

} //TMVA_SOFIE_HLS4MLDenseModel

#endif  // ROOT_TMVA_SOFIE_HLS4MLDENSEMODEL
