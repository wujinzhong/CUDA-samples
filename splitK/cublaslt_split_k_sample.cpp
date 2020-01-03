/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/* This example demonstrates how to use the CUBLAS library
 * by scaling an array of floating-point values on the device
 * and comparing the result to the same operation performed
 * on the host.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

/* Matrix size */
#define M_ (32)
#define N_ (32)
#define K_ (1024)

#include "ltSgemm.cpp"
#include "ltSgemm_split_k.cpp"

#include <sys/time.h>
#include <unistd.h>

class RunTime{
public:
    struct timeval timeStart, timeEnd;
    void reset(){ gettimeofday(&timeStart, NULL ); }
    float calRunTime(const char* str = "")
    {
        gettimeofday( &timeEnd, NULL );
        double runTime = (timeEnd.tv_sec - timeStart.tv_sec ) + (double)(timeEnd.tv_usec -timeStart.tv_usec)/1000000;
        printf("%s runTime is %lf sec.\n", str, runTime);
    }
};

/* Host implementation of a simple version of sgemm */
static void simple_sgemm(int M, int N, int K, float alpha, const float *A, const float *B,
                         float beta, float *C) {

  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float prod = 0;

      for (int k = 0; k < K; ++k) {
        prod += A[k * M + m] * B[n * K + k];
      }

      C[n * M + m] = alpha * prod + beta * C[n * M + m];
    }
  }
}

void printMatrix( float* mat, int M, int N )
{
	int endM = M>8?8:M;
	int endN = N>4?4:N;
	for( int m=0; m<endM; m++ )
	{
		for( int n=0; n<endN; n++ )
		{
			printf("%.4f, ", mat[n*M+m]);
		}
		printf("\n");
	}
	printf("\n\n");
}

int compareOutput( float* h_C_ref, float* h_C, float* d_C, int M, int N, cudaStream_t stream )
{
  int mn = M*N;
  /* Read the result back */
  cublasStatus_t status = cublasGetVectorAsync(mn, sizeof(h_C[0]), d_C, 1, h_C, 1, stream);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (read C)\n");
    return EXIT_FAILURE;
  }

  /* Check result against reference */
  double error_norm = 0;
  double ref_norm = 0;

  //printf("{M,N,K}={%10d,%10d,%10d}\n", M, N, K_);
  //printMatrix( h_C_ref, M, N );
  //printMatrix( h_C, M, N );
  for (int i = 0; i < mn; ++i) {
    float diff = h_C_ref[i] - h_C[i];
    error_norm += diff * diff;
    ref_norm += h_C_ref[i] * h_C_ref[i];
    //if( i<10 )printf("error_norm=%f, ref_norm=%f\n", error_norm, ref_norm);
  }

  error_norm = sqrt(error_norm);
  ref_norm = sqrt(ref_norm);
  //printf("error_norm=%f, ref_norm=%f\n", error_norm, ref_norm);

  if (fabs(ref_norm) < 1e-7) {
    fprintf(stderr, "!!!! reference norm is 0\n");
    return EXIT_FAILURE;
  }

  if (error_norm / ref_norm < 1e-3f) {
    printf("simpleCUBLAS test passed. error_norm/ref_norm=%f/%f\n", (float)(error_norm), (float)(ref_norm));
    return (EXIT_SUCCESS);
  } else {
    printf("simpleCUBLAS test failed. error_norm/ref_norm=%f/%f\n", (float)(error_norm), (float)(ref_norm));
    return (EXIT_FAILURE);
  }
}

/* Main */
int main(int argc, char **argv) {
  cublasStatus_t status;
  cudaError_t cudaStat;
  float *h_A;
  float *h_B;
  float *h_C;
  float *h_C_ref;
  float *d_A = 0;
  float *d_B = 0;
  float *d_C = 0;
  float* d_workspace = 0;
  int workspacesz = (1<<30);
  float alpha = 1.0f;
  float beta = 0.0f;
  int mn = M_ * N_;
  int mk = M_ * K_;
  int kn = K_ * N_;

  int i;
  float error_norm;
  float ref_norm;
  float diff;
  cublasLtHandle_t cublasLtHandle;
  cublasHandle_t cublasHandle;

  int dev = findCudaDevice(argc, (const char **)argv);

  if (dev == -1) {
    return EXIT_FAILURE;
  }

  /* Initialize CUBLAS */
  printf("simpleCUBLAS test running..\n");

  status = cublasLtCreate(&cublasLtHandle);
  status = cublasCreate(&cublasHandle);
  cudaStream_t stream;
  cudaStat = cudaStreamCreate(&stream);
  status = cublasSetStream(cublasHandle, stream);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS initialization error\n");
    return EXIT_FAILURE;
  }

  /* Allocate host memory for the matrices */
  h_A = reinterpret_cast<float *>(malloc(mk * sizeof(h_A[0])));

  if (h_A == 0) {
    fprintf(stderr, "!!!! host memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  h_B = reinterpret_cast<float *>(malloc(kn * sizeof(h_B[0])));

  if (h_B == 0) {
    fprintf(stderr, "!!!! host memory allocation error (B)\n");
    return EXIT_FAILURE;
  }

  h_C = reinterpret_cast<float *>(malloc(mn * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* Fill the matrices with test data */
  for (i = 0; i < mk; i++) {
    h_A[i] = rand() / static_cast<float>(RAND_MAX);
  }
  for (i = 0; i < kn; i++) {
    h_B[i] = rand() / static_cast<float>(RAND_MAX);
  }
  for (i = 0; i < mn; i++) {
    h_C[i] = rand() / static_cast<float>(RAND_MAX);
  }

  /* Allocate device memory for the matrices */
  if (cudaMalloc(reinterpret_cast<void **>(&d_A), mk * sizeof(d_A[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate A)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_B), kn * sizeof(d_B[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate B)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_C), mn * sizeof(d_C[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return EXIT_FAILURE;
  }

  if (cudaMalloc(reinterpret_cast<void **>(&d_workspace), workspacesz * sizeof(d_workspace[0])) !=
      cudaSuccess) {
    fprintf(stderr, "!!!! device memory allocation error (allocate C)\n");
    return EXIT_FAILURE;
  }

  /* Initialize the device matrices with the host matrices */
  status = cublasSetVectorAsync(mk, sizeof(h_A[0]), h_A, 1, d_A, 1, stream);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write A)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVectorAsync(kn, sizeof(h_B[0]), h_B, 1, d_B, 1, stream);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write B)\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVectorAsync(mn, sizeof(h_C[0]), h_C, 1, d_C, 1, stream);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! device access error (write C)\n");
    return EXIT_FAILURE;
  }

  RunTime runtime;
  runtime.reset();

  printf("{M,N,K}={%10d,%10d,%10d}\n", M_, N_, K_);

  /* Performs operation using plain C code */
  
  simple_sgemm(M_, N_, K_, alpha, h_A, h_B, beta, h_C);
  runtime.calRunTime("simple_sgemm");
  //printMatrix( h_C, M_, N_ );
  
  h_C_ref = h_C;

  /* Allocate host memory for reading back the result from device memory */
  h_C = reinterpret_cast<float *>(malloc(mn * sizeof(h_C[0])));

  if (h_C == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }


  // Performs operation using cublas
  runtime.reset();
  status = cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, M_, N_, K_, &alpha, d_A,
                       M_, d_B, K_, &beta, d_C, M_);
  runtime.calRunTime("cublasSgemm");
  compareOutput( h_C_ref, h_C, d_C, M_, N_, stream );
  
  
  runtime.reset();
  status = LtSgemm(cublasLtHandle, CUBLAS_OP_N, CUBLAS_OP_N, M_, N_, K_, &alpha, d_A,
                         M_, d_B, K_, &beta, d_C, M_, nullptr, 0, stream);
  runtime.calRunTime("LtSgemm");
  compareOutput( h_C_ref, h_C, d_C, M_, N_, stream );
  
  runtime.reset();
  LtSgemmCustomFind(cublasLtHandle,
                  CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  M_,
                  N_,
                  K_,
                  &alpha, //host pointer
                  d_A,
                  M_,
                  d_B,
                  K_,
                  &beta, //host pointer
                  d_C,
                  M_,
                  d_workspace,
                  workspacesz);
  runtime.calRunTime("LtSgemmCustomFind");
  compareOutput( h_C_ref, h_C, d_C, M_, N_, stream );
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }
  //compareOutput( h_C_ref, h_C, d_C, M_, N_, stream );

  /* Memory clean up */
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  if (cudaFree(d_A) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_B) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_C) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

  if (cudaFree(d_workspace) != cudaSuccess) {
    fprintf(stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

  /* Shutdown */
  status = cublasLtDestroy(cublasLtHandle);
  status = cublasDestroy(cublasHandle);
  cudaStreamDestroy(stream);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }
}


