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

cublasStatus_t
LtSgemm(cublasLtHandle_t ltHandle,
       cublasOperation_t transa,
       cublasOperation_t transb,
       int m,
       int n,
       int k,
       const float *alpha, /* host pointer */
       const float *A,
       int lda,
       const float *B,
       int ldb,
       const float *beta, /* host pointer */
       float *C,
       int ldc,
       void *workspace,
       size_t workspaceSize,
       cudaStream_t stream) {
   cublasStatus_t status = CUBLAS_STATUS_SUCCESS;

   cublasLtMatmulDesc_t operationDesc = NULL;
   cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
   cublasLtMatmulPreference_t preference = NULL;

   int returnedResults                             = 0;
   cublasLtMatmulHeuristicResult_t heuristicResult = {};

   // Create operation descriptor; see cublasLtMatmulDescAttributes_t
   // for details about defaults; here we just set the transforms for
   // A and B.
   status = cublasLtMatmulDescCreate(&operationDesc, CUDA_R_32F);
   if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
   status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
   if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
   status = cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa));
   if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

   // Create matrix descriptors. Not setting any extra attributes.
   status = cublasLtMatrixLayoutCreate(
       &Adesc, CUDA_R_32F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda);
   if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
   status = cublasLtMatrixLayoutCreate(
       &Bdesc, CUDA_R_32F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb);
   if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
   status = cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc);
   if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

   // Create preference handle; In general, extra attributes can be
   // used here to disable tensor ops or to make sure algo selected
   // will work with badly aligned A, B, C. However, for simplicity
   // here we assume A,B,C are always well aligned (e.g., directly
   // come from cudaMalloc)
   status = cublasLtMatmulPreferenceCreate(&preference);
   if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;
   status = cublasLtMatmulPreferenceSetAttribute(
       preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
   if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

   // We just need the best available heuristic to try and run matmul.
   // There is no guarantee that this will work. For example, if A is
   // badly aligned, you can request more (e.g. 32) algos and try to
   // run them one by one until something works.
   status = cublasLtMatmulAlgoGetHeuristic(
       ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);
   if (status != CUBLAS_STATUS_SUCCESS) goto CLEANUP;

   if (returnedResults == 0) {
       status = CUBLAS_STATUS_NOT_SUPPORTED;
       goto CLEANUP;
   }

   {
   float msecTotal = 0;
   cudaEvent_t start, stop;
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, stream);

   const int loop = 10;
   for( int i=0; i<loop; i++ )
   status = cublasLtMatmul(ltHandle,
                           operationDesc,
                           alpha,
                           A,
                           Adesc,
                           B,
                           Bdesc,
                           beta,
                           C,
                           Cdesc,
                           C,
                           Cdesc,
                           &heuristicResult.algo,
                           workspace,
                           workspaceSize,
                           stream);
                           
   cudaEventRecord(stop, stream);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&msecTotal, start, stop);
   printf( "cublasLtMatmul mean time: %.2f\n", msecTotal/loop );
   }
CLEANUP:
   // Descriptors are no longer needed as all GPU work was already
   // enqueued.
   if (preference) cublasLtMatmulPreferenceDestroy(preference);
   if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
   if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
   if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
   if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
   return status == CUBLAS_STATUS_SUCCESS ? static_cast<cublasStatus_t>(0) : static_cast<cublasStatus_t>(1);
}

/* Host implementation of a simple version of sgemm */
static void simple_sgemm_n2_backup(int n, float alpha, const float *A, const float *B,
                         float beta, float *C) {
  int i;
  int j;
  int k;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float prod = 0;

      for (k = 0; k < n; ++k) {
        prod += A[k * n + i] * B[j * n + k];
      }

      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}


