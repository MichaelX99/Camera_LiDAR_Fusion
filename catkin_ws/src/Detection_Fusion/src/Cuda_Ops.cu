#include "Cuda_Ops.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

const char* cublasGetErrorString(cublasStatus_t status)
{
/*
  Input(s): CUBLAS Status Output
  Output(s): error code to string conversion
  Function: Convert a CUBLAS output to string

  https://stackoverflow.com/questions/13041399/equivalent-of-cudageterrorstring-for-cublas
*/
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    }
    return "unknown error";
}

static void HandleCUDAError( cudaError_t err,
                             const char *file,
                             int line ) {
/*
  Taken from CUDA by Example An Introduction to General-Purpose GPU Programming by Jason Sanders and Edward Kandort
*/
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_CUDA_ERROR( err ) (HandleCUDAError( err, __FILE__, __LINE__ ))

static void HandleCUBLASError( cublasStatus_t err,
                               const char *file,
                               int line ) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf( "%s in %s at line %d\n", cublasGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_CUBLAS_ERROR( err ) (HandleCUBLASError( err, __FILE__, __LINE__ ))

__global__ void device_RELU(float *input, const int m)
{
/*
  Input(s): Input vector, vector length
  Output(s): N/A
  Function: CUDA kernel to perform the RELU activation
*/
  int x = blockIdx.x;

  if (x < m)
  {
    input[x] = max(0.0, input[x]);
  }
}

void print_layer(const float *input, const int m)
{
/*
  Input(s): Input vector, vector length
  Output(s): N/A
  Function: Debug function to print out the contents of a layer stored in gpu memory
*/
  float *h_input = (float*)malloc(m * sizeof(float));

  HANDLE_CUBLAS_ERROR( cublasGetVector(m, sizeof(float), input, 1, h_input, 1) );

  for (int i = 0; i < m; i++)
  {
    printf("%f\n", h_input[i]);
  }
}

float* Allocate_MLP(const int num)
{
/*
  Input(s): Lenght of vector
  Output(s): Pointer to GPU memory
  Function: GPU memory allocation function
*/
  float *d_MLP;

  HANDLE_CUDA_ERROR( cudaMalloc((void**)&d_MLP, num * sizeof(float)) );

  return d_MLP;
}

void destroy_memory(float *d_input)
{
/*
  Input(s): Input pointer to GPU memory
  Output(s): N/A
  Function: GPU memory destruction function
*/
  HANDLE_CUDA_ERROR( cudaFree(d_input) );
}

classification_output populate_output(float *_d_class_output, float *_d_length_output, float *_d_z_output, float *_d_rotation_output)
{
/*
  Input(s): GPU memory pointers to the class layer output, the length layer output, and z layer output
  Output(s): Output information of cluster/detection
  Function: Populate the output of the LiDAR MLP
*/
  classification_output output;

  // The class layer output
  float *h_class = (float *)malloc(4 * sizeof(float));

  // The lenght layer output
  float *h_length = (float *)malloc(1 * sizeof(float));

  // The z layer output
  float *h_z = (float *)malloc(1 * sizeof(float));

  float *h_rotation = (float *)malloc(1 * sizeof(float));

  HANDLE_CUBLAS_ERROR( cublasGetVector(4, sizeof(float), _d_class_output, 1, h_class, 1) );
  HANDLE_CUBLAS_ERROR( cublasGetVector(1, sizeof(float), _d_z_output, 1, h_z, 1) );
  HANDLE_CUBLAS_ERROR( cublasGetVector(1, sizeof(float), _d_length_output, 1, h_length, 1) );
  HANDLE_CUBLAS_ERROR( cublasGetVector(1, sizeof(float), _d_rotation_output, 1, h_rotation, 1) );

  // Perform the argmax function and max function
  int ind = 0;
  float prob = 0.0;
  for (int i = 0; i < 4; i++)
  {
    if (h_class[i] > prob)
    {
      prob = h_class[i];
      ind = i;
    }
  }

  output.obj_class = ind;
  output.probability = prob;
  output.z = h_z[0];
  output.l = h_length[0];
  output.a = h_rotation[0];

  free(h_class);
  free(h_length);
  free(h_z);
  free(h_rotation);

  return output;
}

float* cublas_Set_Matrix(float *input, const int m, const int n)
{
/*
  Input(s): Host memory pointer, number of rows, number of columns
  Output(s): GPU memory pointer
  Function: Push a matrix into GPU memory using the CUBLAS API
*/
  float *output;
  HANDLE_CUDA_ERROR( cudaMalloc((void**)&output, m*n*sizeof(float)) );

  HANDLE_CUBLAS_ERROR( cublasSetMatrix(m, n, sizeof(float), input, m, output, m) );

  return output;
}

float* cublas_Set_Vector(float *input, const int m)
{
/*
  Input(s): Host memory pointer, length of vector
  Output(s): GPU memory pointer
  Function: Push a vector into GPU memory using the CUBLAS API
*/
  float *output;

  HANDLE_CUDA_ERROR( cudaMalloc((void**)&output, m*sizeof(float)) );

  HANDLE_CUBLAS_ERROR( cublasSetVector(m, sizeof(float), input, 1, output, 1) );

  return output;
}

void cublas_Set_Input(float *input, std::vector<float> input_vec, float *output, const int m)
{
/*
  Input(s): Host memory pointer, vector of elements, GPU memory pointer, lenght of vector
  Output(s): N/A
  Function: Populate an inputted GPU memory pointer with the contents of an inputted vector
*/
  for (int i = 0; i < m; i++)
  {
    input[i] = input_vec[i];
  }

  HANDLE_CUBLAS_ERROR( cublasSetVector(m, sizeof(float), input, 1, output, 1) );
}

void blas_layer(const float *x, const float *A, const float *b, float *y, const int m, const int n, cublasHandle_t handle)
{
/*
  Input(s): GPU memory pointers to the input vector, the layer weight matrix, the layer bias vector, the layer output vector, the number of rows in the weight matrix, the number of columns in the weight matrix, the CUBLAS handle
  Output(s): N/A
  Function: Compute the output of a MLP layer using CUBLAS
*/
  HANDLE_CUBLAS_ERROR( cublasScopy(handle, m, b, 1, y, 1) );

  float al = 1.0;
  float bet = 1.0;

  HANDLE_CUBLAS_ERROR( cublasSgemv(handle, CUBLAS_OP_N, m, n, &al, A, m, x, 1, &bet, y, 1) );
}

void RELU(float *input, const int m)
{
/*
  Input(s): The GPU memory pointer
  Output(s): The number of elements in the vector
  Function: Function call for the RELU kernel
*/
  device_RELU<<<m, 1>>>(input, m);
}

void Softmax(float *input, const int m)
{
/*
  Input(s): The GPU memory pointer
  Output(s): The number of elements in the vector
  Function: Function call for the Softmax function
*/
  float *h_class = (float *)malloc(m * sizeof(float));
  HANDLE_CUBLAS_ERROR( cublasGetVector(m, sizeof(float), input, 1, h_class, 1) );

  double total = 0.0;
  for (int i = 0; i < m; i++)
  {
    total += exp(h_class[i]);
  }
  for (int i = 0; i < m; i++)
  {
    h_class[i] = exp(h_class[i]) / total;
  }

  HANDLE_CUBLAS_ERROR( cublasSetVector(m, sizeof(float), h_class, 1, input, 1) );
}

void Sigmoid(float *input, const int m)
{
/*
  Input(s): The GPU memory pointer
  Output(s): The number of elements in the vector
  Function: Function call for the Sigmoid function
*/
  float *h_input = (float *)malloc(m * sizeof(float));
  HANDLE_CUBLAS_ERROR( cublasGetVector(m, sizeof(float), input, 1, h_input, 1) );

  double temp;
  for (int i = 0; i < m; i++)
  {
    temp = exp(h_input[i]) / (1.0 + exp(h_input[i]));
    h_input[i] = (float)temp;
  }

  HANDLE_CUBLAS_ERROR( cublasSetVector(m, sizeof(float), h_input, 1, input, 1) );
}
