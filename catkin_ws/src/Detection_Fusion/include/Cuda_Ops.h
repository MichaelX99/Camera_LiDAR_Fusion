#ifndef __CUDA_OPS__
#define __CUDA_OPS__

#include <cstdlib>
#include <cublas_v2.h>

#include <vector>

#include <iostream>
#include <stdio.h>

// Output structure of the MLP
struct classification_output
{
  int obj_class;
  float probability;
  float z;
  float l;
  float a;
};

//void gpu_cudnn_softmax(float *d_input);

float* Allocate_MLP(const int num);
void destroy_memory(float *d_input);

float* cublas_Set_Matrix(float *input, const int m, const int n);
float* cublas_Set_Vector(float *input, const int m);
void cublas_Set_Input(float *input, std::vector<float> input_vec, float *output, const int m);

classification_output populate_output(float *_d_class_output, float *_d_length_output, float *_d_z_output, float *_d_rotation_output);
void blas_layer(const float *x, const float *A, const float *b, float *y, const int m, const int n, cublasHandle_t handle);

void RELU(float *input, const int m);
void Softmax(float *input, const int m);
void Sigmoid(float *input, const int m);

void print_layer(const float *input, const int m);

#endif
