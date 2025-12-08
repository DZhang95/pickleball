#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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
    // Optional profiling controlled by environment variable
    bool do_profile = false;
    const char* prof_env = std::getenv("PICKLE_CUDA_PROFILE");
    if (prof_env && prof_env[0] != '\0') do_profile = true;

    // Allocate device memory
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    // If profiling, create events to measure timings
    cudaEvent_t e_alloc_start, e_alloc_end, e_h2d_end, e_kernel_end, e_d2h_end;
    if (do_profile) {
        cudaEventCreate(&e_alloc_start);
        cudaEventCreate(&e_alloc_end);
        cudaEventCreate(&e_h2d_end);
        cudaEventCreate(&e_kernel_end);
        cudaEventCreate(&e_d2h_end);
        cudaEventRecord(e_alloc_start, 0);
    }

    if (cudaMalloc((void**)&d_a, sizeof(float) * n) != cudaSuccess) return false;
    if (cudaMalloc((void**)&d_b, sizeof(float) * n) != cudaSuccess) { cudaFree(d_a); return false; }
    if (cudaMalloc((void**)&d_c, sizeof(float) * n) != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return false; }

    if (do_profile) cudaEventRecord(e_alloc_end, 0);

    // Host -> Device copies
    if (cudaMemcpy(d_a, a, sizeof(float) * n, cudaMemcpyHostToDevice) != cudaSuccess) goto error;
    if (cudaMemcpy(d_b, b, sizeof(float) * n, cudaMemcpyHostToDevice) != cudaSuccess) goto error;

    if (do_profile) cudaEventRecord(e_h2d_end, 0);

    // Kernel launch
    vecAddKernel<<<blocks, threads>>>(d_a, d_b, d_c, n);

    if (cudaPeekAtLastError() != cudaSuccess) goto error;
    if (cudaDeviceSynchronize() != cudaSuccess) goto error;

    if (do_profile) cudaEventRecord(e_kernel_end, 0);

    // Device -> Host copy
    if (cudaMemcpy(c, d_c, sizeof(float) * n, cudaMemcpyDeviceToHost) != cudaSuccess) goto error;

    if (do_profile) {
        cudaEventRecord(e_d2h_end, 0);
        // Ensure events are completed
        cudaEventSynchronize(e_d2h_end);

        float t_alloc = 0.0f, t_h2d = 0.0f, t_kernel = 0.0f, t_d2h = 0.0f;
        cudaEventElapsedTime(&t_alloc, e_alloc_start, e_alloc_end);
        cudaEventElapsedTime(&t_h2d, e_alloc_end, e_h2d_end);
        cudaEventElapsedTime(&t_kernel, e_h2d_end, e_kernel_end);
        cudaEventElapsedTime(&t_d2h, e_kernel_end, e_d2h_end);

#ifdef DEBUG
    std::fprintf(stderr, "CUDA PROFILE n=%d alloc=%.3fms H2D=%.3fms kernel=%.3fms D2H=%.3fms total_gpu_path=%.3fms\n",
             n, t_alloc, t_h2d, t_kernel, t_d2h, t_alloc + t_h2d + t_kernel + t_d2h);
#endif

        cudaEventDestroy(e_alloc_start);
        cudaEventDestroy(e_alloc_end);
        cudaEventDestroy(e_h2d_end);
        cudaEventDestroy(e_kernel_end);
        cudaEventDestroy(e_d2h_end);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    return true;

error:
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    if (do_profile) {
        // Best-effort: try to destroy events if they were created
        cudaEventDestroy(e_alloc_start);
        cudaEventDestroy(e_alloc_end);
        cudaEventDestroy(e_h2d_end);
        cudaEventDestroy(e_kernel_end);
        cudaEventDestroy(e_d2h_end);
    }
    return false;
}

// Persistent device buffers implementation
static float *g_d_a = nullptr;
static float *g_d_b = nullptr;
static float *g_d_c = nullptr;
static int g_allocated_n = 0;

extern "C" bool cuda_vecadd_init(int n) {
    if (n <= 0) return false;
    if (g_allocated_n >= n && g_d_a && g_d_b && g_d_c) {
        // already allocated large enough
        return true;
    }

    // free previous if any
    if (g_d_a) cudaFree(g_d_a);
    if (g_d_b) cudaFree(g_d_b);
    if (g_d_c) cudaFree(g_d_c);

    if (cudaMalloc((void**)&g_d_a, sizeof(float) * n) != cudaSuccess) return false;
    if (cudaMalloc((void**)&g_d_b, sizeof(float) * n) != cudaSuccess) { cudaFree(g_d_a); g_d_a = nullptr; return false; }
    if (cudaMalloc((void**)&g_d_c, sizeof(float) * n) != cudaSuccess) { cudaFree(g_d_a); cudaFree(g_d_b); g_d_a = g_d_b = nullptr; return false; }

    g_allocated_n = n;
    return true;
}

extern "C" bool cuda_vecadd_run(const float* a, const float* b, float* c, int n) {
    if (n <= 0) return true;
    if (!g_d_a || !g_d_b || !g_d_c || g_allocated_n < n) return false;

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;

    if (cudaMemcpy(g_d_a, a, sizeof(float) * n, cudaMemcpyHostToDevice) != cudaSuccess) return false;
    if (cudaMemcpy(g_d_b, b, sizeof(float) * n, cudaMemcpyHostToDevice) != cudaSuccess) return false;

    vecAddKernel<<<blocks, threads>>>(g_d_a, g_d_b, g_d_c, n);
    if (cudaPeekAtLastError() != cudaSuccess) return false;
    if (cudaDeviceSynchronize() != cudaSuccess) return false;

    if (cudaMemcpy(c, g_d_c, sizeof(float) * n, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    return true;
}

extern "C" void cuda_vecadd_destroy() {
    if (g_d_a) { cudaFree(g_d_a); g_d_a = nullptr; }
    if (g_d_b) { cudaFree(g_d_b); g_d_b = nullptr; }
    if (g_d_c) { cudaFree(g_d_c); g_d_c = nullptr; }
    g_allocated_n = 0;
}
