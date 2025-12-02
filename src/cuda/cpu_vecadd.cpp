#include "cuda/cuda_kernels.h"

// Simple CPU fallback for vector add
void cpu_vecadd(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}
