//Code generated automatically by TMVA for Inference of Model file [tmva/sofie/exercise3_outputs/lstm_model.json] at [Fri Mar  6 22:02:55 2026] 

#ifndef ROOT_TMVA_SOFIE_TMVASOFIEEXERCISE3_OUTPUTSLSTM_MODEL
#define ROOT_TMVA_SOFIE_TMVASOFIEEXERCISE3_OUTPUTSLSTM_MODEL

#include <vector>
#include "TMVA/SOFIE_common.hxx"
#include <fstream>

namespace TMVA_SOFIE_tmvasofieexercise3_outputslstm_model{
namespace BLAS{
	extern "C" void saxpy_(const int * n, const float * alpha, const float * x,
	                         const int * incx, float * y, const int * incy);
	extern "C" void sgemm_(const char * transa, const char * transb, const int * m, const int * n, const int * k,
	                       const float * alpha, const float * A, const int * lda, const float * B, const int * ldb,
	                       const float * beta, float * C, const int * ldc);
}//BLAS
struct Session {
// initialized (weights and constant) tensors
std::vector<float> fTensor_LSTM_B = std::vector<float>(320);
float * tensor_LSTM_B = fTensor_LSTM_B.data();
std::vector<float> fTensor_LSTM_R = std::vector<float>(1024);
float * tensor_LSTM_R = fTensor_LSTM_R.data();
std::vector<float> fTensor_LSTM_W = std::vector<float>(512);
float * tensor_LSTM_W = fTensor_LSTM_W.data();

//--- Allocating session memory pool to be used for allocating intermediate tensors
std::vector<char> fIntermediateMemoryPool = std::vector<char>(320);


// --- Positioning intermediate tensor memory --
 // Allocating memory for intermediate tensor out_LSTM_1 with size 320 bytes
float* tensor_out_LSTM_1 = reinterpret_cast<float*>(fIntermediateMemoryPool.data() + 0);

std::vector<float> fVec_op_0_ff_input_gate = std::vector<float>(80);
std::vector<float> fVec_op_0_ff_output_gate = std::vector<float>(80);
std::vector<float> fVec_op_0_ff_cell_gate = std::vector<float>(80);
std::vector<float> fVec_op_0_ff_forget_gate = std::vector<float>(80);
std::vector<float> fVec_op_0_input_gate = std::vector<float>(80);
std::vector<float> fVec_op_0_output_gate = std::vector<float>(80);
std::vector<float> fVec_op_0_cell_gate = std::vector<float>(80);
std::vector<float> fVec_op_0_forget_gate = std::vector<float>(80);
std::vector<float> fVec_op_0_cell_state = std::vector<float>(80);
std::vector<float> fVec_op_0_new_cell_state = std::vector<float>(80);


Session(std::string filename ="tmvasofieexercise3_outputslstm_model.dat") {

//--- reading weights from file
   std::ifstream f;
   f.open(filename);
   if (!f.is_open()) {
      throw std::runtime_error("tmva-sofie failed to open file " + filename + " for input weights");
   }
   using TMVA::Experimental::SOFIE::ReadTensorFromStream;
   ReadTensorFromStream(f, tensor_LSTM_B, "tensor_LSTM_B", 320);
   ReadTensorFromStream(f, tensor_LSTM_R, "tensor_LSTM_R", 1024);
   ReadTensorFromStream(f, tensor_LSTM_W, "tensor_LSTM_W", 512);
   f.close();

}

void doInfer(float const* tensor_input_0,  std::vector<float> &output_tensor_out_LSTM_1 ){

   float const *op_0_input = tensor_input_0;
   float * op_0_ff_input_gate = this->fVec_op_0_ff_input_gate.data();
   float * op_0_ff_output_gate = this->fVec_op_0_ff_output_gate.data();
   float * op_0_ff_cell_gate = this->fVec_op_0_ff_cell_gate.data();
   float * op_0_ff_forget_gate = this->fVec_op_0_ff_forget_gate.data();
   float * op_0_input_gate = this->fVec_op_0_input_gate.data();
   float * op_0_output_gate = this->fVec_op_0_output_gate.data();
   float * op_0_cell_gate = this->fVec_op_0_cell_gate.data();
   float * op_0_forget_gate = this->fVec_op_0_forget_gate.data();
   float * op_0_cell_state = this->fVec_op_0_cell_state.data();
   float * op_0_new_cell_state = this->fVec_op_0_new_cell_state.data();
   float *op_0_hidden_state = tensor_out_LSTM_1;
   char op_0_transA = 'N';
   char op_0_transB = 'T';
   int op_0_m = 5;
   int op_0_n = 16;
   int op_0_k = 8;
   float  op_0_alpha = 1.;
   float  op_0_beta = 0.;
   int op_0_bias_size = 80;
   int op_0_incx = 1;
   int op_0_incy = 1;
   BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_LSTM_W, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_input_gate, &op_0_n);
   BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_LSTM_W + 128, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_output_gate, &op_0_n);
   BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_LSTM_W + 384, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_cell_gate, &op_0_n);
   BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &op_0_m, &op_0_k, &op_0_alpha, tensor_LSTM_W + 256, &op_0_k, op_0_input, &op_0_k, &op_0_beta, op_0_ff_forget_gate, &op_0_n);
   BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_LSTM_B, &op_0_incx, op_0_ff_input_gate, &op_0_incy);
   BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_LSTM_B + 80, &op_0_incx, op_0_ff_output_gate, &op_0_incy);
   BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_LSTM_B + 240, &op_0_incx, op_0_ff_cell_gate, &op_0_incy);
   BLAS::saxpy_(&op_0_bias_size, &op_0_alpha, tensor_LSTM_B + 160, &op_0_incx, op_0_ff_forget_gate, &op_0_incy);
   for (size_t seq = 0; seq < 1; seq++) {
      size_t ff_offset = seq * 80;
      size_t gate_offset = seq * 80;
      std::copy(op_0_ff_input_gate + ff_offset, op_0_ff_input_gate + ff_offset + 80, op_0_input_gate + gate_offset);
      std::copy(op_0_ff_output_gate + ff_offset, op_0_ff_output_gate + ff_offset + 80, op_0_output_gate + gate_offset);
      std::copy(op_0_ff_cell_gate + ff_offset, op_0_ff_cell_gate + ff_offset + 80, op_0_cell_gate + gate_offset);
      std::copy(op_0_ff_forget_gate + ff_offset, op_0_ff_forget_gate + ff_offset + 80, op_0_forget_gate + gate_offset);
   }
   for (size_t seq = 0; seq < 1; seq++) {
      size_t index = seq;
      int m2 = 5;
      size_t offset = index * 80;
      if (seq == 0) {
      } else {
         size_t previous_offset = (seq - 1) * 80;
         BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_LSTM_R, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_input_gate + offset, &op_0_n);
         BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_LSTM_R + 256, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_output_gate + offset, &op_0_n);
         BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_LSTM_R + 768, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_cell_gate + offset, &op_0_n);
         BLAS::sgemm_(&op_0_transB, &op_0_transA, &op_0_n, &m2, &op_0_n, &op_0_alpha, tensor_LSTM_R + 512, &op_0_n, op_0_hidden_state + previous_offset, &op_0_n, &op_0_alpha, op_0_forget_gate + offset, &op_0_n);
      }
      for (size_t i = offset; i < offset + 80; i++) {
         float ex = exp(-2 * op_0_cell_gate[i]);
            op_0_cell_gate[i] = (1. - ex) / (1. + ex);
      }
      for (size_t i = offset; i < offset + 80; i++) {
            op_0_input_gate[i] = 1. / (1. + exp(-op_0_input_gate[i]));
      }
      for (size_t i = offset; i < offset + 80; i++) {
            op_0_forget_gate[i] = 1. / (1. + exp(-op_0_forget_gate[i]));
      }
      for (size_t i = offset; i < offset + 80; i++) {
         op_0_cell_state[i] = op_0_input_gate[i] * op_0_cell_gate[i];
      }
      if (seq == 0) {
      } else {
         size_t previous_offset = (seq - 1) * 80;
         for (size_t i = 0; i < 80; i++) {
            op_0_cell_state[i + offset] += op_0_forget_gate[i + offset] * op_0_cell_state[i + previous_offset];
         }
      }
      for (size_t i = offset; i < offset + 80; i++) {
            op_0_output_gate[i] = 1. / (1. + exp(-op_0_output_gate[i]));
      }
      std::copy(op_0_cell_state + offset, op_0_cell_state + offset + 80, op_0_new_cell_state + offset);
      for (size_t i = offset; i < offset + 80; i++) {
         float ex = exp(-2 * op_0_new_cell_state[i]);
            op_0_new_cell_state[i] = (1. - ex) / (1. + ex);
      }
      for (size_t i = offset; i < offset + 80; i++) {
         op_0_hidden_state[i] = op_0_output_gate[i] * op_0_new_cell_state[i];
      }
   }
   using TMVA::Experimental::SOFIE::UTILITY::FillOutput;

   FillOutput(tensor_out_LSTM_1, output_tensor_out_LSTM_1, 80);
}



std::vector<float> infer(float const* tensor_input_0){
   std::vector<float > output_tensor_out_LSTM_1;
   doInfer(tensor_input_0, output_tensor_out_LSTM_1 );
   return {output_tensor_out_LSTM_1};
}
};   // end of Session

} //TMVA_SOFIE_tmvasofieexercise3_outputslstm_model

#endif  // ROOT_TMVA_SOFIE_TMVASOFIEEXERCISE3_OUTPUTSLSTM_MODEL
