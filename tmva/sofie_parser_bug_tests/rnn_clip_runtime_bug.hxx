//Code generated automatically by TMVA for Inference of Model file [rnn_clip_runtime_bug.onnx] at [Thu Apr 16 20:09:46 2026] 

#ifndef ROOT_TMVA_SOFIE_RNN_CLIP_RUNTIME_BUG
#define ROOT_TMVA_SOFIE_RNN_CLIP_RUNTIME_BUG

#include <cmath>
#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_rnn_clip_runtime_bug{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_B = std::vector<float>(1);
float * tensor_B = fTensor_B.data();
std::vector<float> fTensor_R = std::vector<float>(1);
float * tensor_R = fTensor_R.data();
std::vector<float> fTensor_W = std::vector<float>(1);
float * tensor_W = fTensor_W.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(8);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor Y with size 4 bytes
float* tensor_Y = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

 // Allocating memory for intermediate tensor Y_h with size 4 bytes
float* tensor_Y_h = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 4);

std::vector<float> fVec_op_0_feedforward = std::vector<float>(1);


Session(std::string filename ="rnn_clip_runtime_bug.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_B, "tensor_B", 1);
   ReadTensorFromStream(f, tensor_R, "tensor_R", 1);
   ReadTensorFromStream(f, tensor_W, "tensor_W", 1);
   f.close();

}

void doInfer(float const* tensor_X,  std::vector<float> &output_tensor_Y, std::vector<float> &output_tensor_Y_h ){

   float const*op_0_input = tensor_X;
   float * op_0_feedforward = this->fVec_op_0_feedforward.data();
   float *op_0_hidden_state = tensor_Y;
   char op_0_transA = 'N';
   char op_0_transB = 'T';
   int op_0_m = 1;
   int op_0_n = 1;
   int op_0_k = 1;
   float op_0_alpha = 1.;
   float op_0_beta = .0;
   int op_0_bias_size = 1;
   int op_0_incx = 1;
   int op_0_incy = 1;
   BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_feedforward, &op_0_n);
   BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_B, &op_0_incx, op_0_feedforward, &op_0_incy);
   for (size_t seq = 0; seq < 1; seq++) {
      size_t offset = seq * 1;
      size_t size = 1;
      size_t h_offset = seq * 1 + 0;
      std::copy(op_0_feedforward + offset, op_0_feedforward + offset + size, op_0_hidden_state + h_offset);
   }
   for (size_t seq = 0; seq < 1; seq++) {
      size_t index = seq;
      int m2 = 1;
      size_t offset = index * 1 + 0;
      size_t size = 1;
      if (seq == 0) {
      } else {
         size_t r_offset = 0;
         size_t previous_offset = (seq - 1) * 1 + 0;
         BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_R + r_offset, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_hidden_state + offset, &op_0_n);
      }
      for (size_t i = offset; i < offset + size; i++) {
         float ex = std::exp(-2 * op_0_hidden_state[i]);
            op_0_hidden_state[i] = (1. - ex) / (1. + ex);
      }
   }
   std::copy(op_0_hidden_state + 0, op_0_hidden_state + 0 + 1, tensor_Y_h);
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_Y, output_tensor_Y, 1);
   FillOutput(tensor_Y_h, output_tensor_Y_h, 1);
}



std::vector<std::vector<float>> infer(float const* tensor_X){
   std::vector<float > output_tensor_Y;
   std::vector<float > output_tensor_Y_h;
   doInfer(tensor_X, output_tensor_Y, output_tensor_Y_h );
   return {output_tensor_Y,output_tensor_Y_h};
}
};   // end of Session

} //TMVA_SOFIE_rnn_clip_runtime_bug

#endif  // ROOT_TMVA_SOFIE_RNN_CLIP_RUNTIME_BUG
