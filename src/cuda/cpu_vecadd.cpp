#include "cuda_kernels.h"

// Simple CPU fallback for vector add
void cpu_vecadd(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// Provide CPU-side stubs for the persistent API so the linker succeeds when CUDA
// is not used. These implementations simply perform the work on the CPU.
extern "C" bool cuda_vecadd_init(int n) {
    (void)n;
    return true; // nothing to allocate for CPU fallback
}

extern "C" bool cuda_vecadd_run(const float* a, const float* b, float* c, int n) {
    cpu_vecadd(a, b, c, n);
    return true;
}

extern "C" void cuda_vecadd_destroy() {
    // no-op for CPU fallback
}

// CPU fallback stubs for the physics API. We return false from run so renderer
// will fall back to its CPU implementation when CUDA is not available.
extern "C" bool cuda_physics_init(int n) {
    (void)n;
    return true;
}

extern "C" bool cuda_physics_run(float* px, float* py, float* pvx, float* pvy, int n, float* ballState, float dt) {
    (void)px; (void)py; (void)pvx; (void)pvy; (void)n; (void)ballState; (void)dt;
    return false;
}

extern "C" void cuda_physics_destroy() {
    // no-op
}
