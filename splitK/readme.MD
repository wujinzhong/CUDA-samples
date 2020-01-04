1. go to current dir ./splitK

2. install CUDA toolkit and select installing cuda samples:
https://docs.nvidia.com/cuda/cuda-samples/index.html

3. compile with:
nvcc -G -g -lcuda -lcublas -lcublasLt -I /ngc-cuda-samples-container/cuda/samples/common/inc -I /current/dir cublaslt_split_k_sample.cpp -o cublaslt_split_k_sample

4. run with:
./cublaslt_split_k_sample

For other matrix size m,n,k for A(m*k) and B(k*n), change the parameter here in cublaslt_split_k_sample.cpp:

#define M_ (32)
#define N_ (32)
#define K_ (1024)
