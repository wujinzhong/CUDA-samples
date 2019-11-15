#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

__global__ void vecAdd( float* A, float* B, float* C, int N )
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if( i<N )
        C[i] = A[i] + B[i];
}

int main(void)
{
    srand(time(0));
    int N = 1024*1024;
    size_t sz = N*sizeof(float);
    float* hA = (float*)malloc(sz);
    float* hB = (float*)malloc(sz);
    float* hC = (float*)malloc(sz);
    for( int i=0; i<N; i++ )
    {
        hA[i] = rand() & 0xff;
        hB[i] = rand() & 0xff;
        hC[i] = 0;
    }

    float* dA, *dB, *dC;
    cudaMalloc( (void **)&dA, sz );
    cudaMalloc( (void **)&dB, sz );
    cudaMalloc( (void **)&dC, sz );

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync( dA, hA, sz, cudaMemcpyHostToDevice, stream );
    cudaMemcpyAsync( dB, hB, sz, cudaMemcpyHostToDevice, stream );
    vecAdd<<<N/1024, 1024, 0, stream>>>(dA, dB, dC, N);

    cudaMemcpyAsync( hC, dC, sz, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize( stream );

    for( int i=0; i<10; i++ )
    {
        printf("%8f + %8f = %8f\n", hA[i], hB[i], hC[i]);
    }

    free(hA);
    free(hB);
    free(hC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaStreamDestroy(stream);

    return 0;
}
