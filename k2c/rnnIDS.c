#include <math.h> 
 #include <string.h> 
#include "./include/k2c_include.h" 
#include "./include/k2c_tensor_include.h" 

 


void rnnIDS(k2c_tensor* simple_rnn_input_input, k2c_tensor* dense_output,float* simple_rnn_output_array,float* simple_rnn_kernel_array,float* simple_rnn_recurrent_kernel_array,float* simple_rnn_bias_array,float* dense_kernel_array,float* dense_bias_array) { 

k2c_tensor simple_rnn_output = {simple_rnn_output_array,1,80,{80, 1, 1, 1, 1}}; 
int simple_rnn_go_backwards = 0;
int simple_rnn_return_sequences = 0;
float simple_rnn_fwork[160] = {0}; 
float simple_rnn_state[80] = {0}; 
k2c_tensor simple_rnn_kernel = {simple_rnn_kernel_array,2,9760,{122, 80,  1,  1,  1}}; 
k2c_tensor simple_rnn_recurrent_kernel = {simple_rnn_recurrent_kernel_array,2,6400,{80,80, 1, 1, 1}}; 
k2c_tensor simple_rnn_bias = {simple_rnn_bias_array,1,80,{80, 1, 1, 1, 1}}; 

 
k2c_tensor dense_kernel = {dense_kernel_array,2,400,{80, 5, 1, 1, 1}}; 
k2c_tensor dense_bias = {dense_bias_array,1,5,{5,1,1,1,1}}; 
float dense_fwork[480] = {0}; 

 
k2c_simpleRNN(&simple_rnn_output,simple_rnn_input_input,simple_rnn_state,&simple_rnn_kernel, 
	&simple_rnn_recurrent_kernel,&simple_rnn_bias,simple_rnn_fwork, 
	simple_rnn_go_backwards,simple_rnn_return_sequences,k2c_sigmoid); 
k2c_tensor dropout_output; 
dropout_output.ndim = simple_rnn_output.ndim; // copy data into output struct 
dropout_output.numel = simple_rnn_output.numel; 
memcpy(dropout_output.shape,simple_rnn_output.shape,K2C_MAX_NDIM*sizeof(size_t));  
dropout_output.array = &simple_rnn_output.array[0]; // rename for clarity 
k2c_dense(dense_output,&dropout_output,&dense_kernel, 
	&dense_bias,k2c_softmax,dense_fwork); 

 } 

void rnnIDS_initialize(float** simple_rnn_output_array 
,float** simple_rnn_kernel_array 
,float** simple_rnn_recurrent_kernel_array 
,float** simple_rnn_bias_array 
,float** dense_kernel_array 
,float** dense_bias_array 
) { 

*simple_rnn_output_array = k2c_read_array("rnnIDSsimple_rnn_output_array.csv",80); 
*simple_rnn_kernel_array = k2c_read_array("rnnIDSsimple_rnn_kernel_array.csv",9760); 
*simple_rnn_recurrent_kernel_array = k2c_read_array("rnnIDSsimple_rnn_recurrent_kernel_array.csv",6400); 
*simple_rnn_bias_array = k2c_read_array("rnnIDSsimple_rnn_bias_array.csv",80); 
*dense_kernel_array = k2c_read_array("rnnIDSdense_kernel_array.csv",400); 
*dense_bias_array = k2c_read_array("rnnIDSdense_bias_array.csv",5); 
} 

void rnnIDS_terminate(float* simple_rnn_output_array,float* simple_rnn_kernel_array,float* simple_rnn_recurrent_kernel_array,float* simple_rnn_bias_array,float* dense_kernel_array,float* dense_bias_array) { 

free(simple_rnn_output_array); 
free(simple_rnn_kernel_array); 
free(simple_rnn_recurrent_kernel_array); 
free(simple_rnn_bias_array); 
free(dense_kernel_array); 
free(dense_bias_array); 
} 

