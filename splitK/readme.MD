go to current dir ./splitK

install CUDA toolkit and select installing cuda samples:
https://docs.nvidia.com/cuda/cuda-samples/index.html

compile with:
nvcc -G -g -lcuda -lcublas -lcublasLt -I /ngc-cuda-samples-container/cuda/samples/common/inc -I /current/dir cublaslt_split_k_sample.cpp -o cublaslt_split_k_sample

run with:
./cublaslt_split_k_sample
