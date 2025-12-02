#include <cuda_runtime.h>
#include <cstdio>
#include "cuda_kernels.h"

// simple vector add kernel
__global__ void vecAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Extern C wrapper so the symbol has C linkage for easy calling from C++ code
extern "C" bool cuda_vecadd(const float* a, const float* b, float* c, int n) {
    if (n <= 0) return true;
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;

    // Allocate device memory
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    if (cudaMalloc((void**)&d_a, sizeof(float) * n) != cudaSuccess) return false;
    if (cudaMalloc((void**)&d_b, sizeof(float) * n) != cudaSuccess) { cudaFree(d_a); return false; }
    if (cudaMalloc((void**)&d_c, sizeof(float) * n) != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return false; }

    if (cudaMemcpy(d_a, a, sizeof(float) * n, cudaMemcpyHostToDevice) != cudaSuccess) goto error;
    if (cudaMemcpy(d_b, b, sizeof(float) * n, cudaMemcpyHostToDevice) != cudaSuccess) goto error;

    vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    if (cudaPeekAtLastError() != cudaSuccess) goto error;
    if (cudaDeviceSynchronize() != cudaSuccess) goto error;

    if (cudaMemcpy(c, d_c, sizeof(float) * n, cudaMemcpyDeviceToHost) != cudaSuccess) goto error;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return true;

error:
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return false;
}
