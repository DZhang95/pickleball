#ifndef PICKLEBALL_CUDA_KERNELS_H
#define PICKLEBALL_CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Vector add: c[i] = a[i] + b[i]
// Implemented in CUDA when built with CUDA=1 as `cuda_vecadd`.
// A CPU fallback `cpu_vecadd` is always provided.
bool cuda_vecadd(const float* a, const float* b, float* c, int n);

#ifdef __cplusplus
}
#endif

// CPU fallback (C++ linkage)
void cpu_vecadd(const float* a, const float* b, float* c, int n);

#endif // PICKLEBALL_CUDA_KERNELS_H
