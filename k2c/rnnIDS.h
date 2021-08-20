#pragma once 
#include "./include/k2c_tensor_include.h" 
void rnnIDS(k2c_tensor* simple_rnn_input_input, k2c_tensor* dense_output,float* simple_rnn_output_array,float* simple_rnn_kernel_array,float* simple_rnn_recurrent_kernel_array,float* simple_rnn_bias_array,float* dense_kernel_array,float* dense_bias_array); 
void rnnIDS_initialize(float** simple_rnn_output_array 
,float** simple_rnn_kernel_array 
,float** simple_rnn_recurrent_kernel_array 
,float** simple_rnn_bias_array 
,float** dense_kernel_array 
,float** dense_bias_array 
); 
void rnnIDS_terminate(float* simple_rnn_output_array,float* simple_rnn_kernel_array,float* simple_rnn_recurrent_kernel_array,float* simple_rnn_bias_array,float* dense_kernel_array,float* dense_bias_array); 
