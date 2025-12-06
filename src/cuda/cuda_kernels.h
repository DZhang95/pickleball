#ifndef PICKLEBALL_CUDA_KERNELS_H
#define PICKLEBALL_CUDA_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// Vector add: c[i] = a[i] + b[i]
// Implemented in CUDA when built with CUDA=1 as `cuda_vecadd`.
// A CPU fallback `cpu_vecadd` is always provided.
bool cuda_vecadd(const float* a, const float* b, float* c, int n);

// Persistent-buffer API: allocate device buffers for n elements. Returns true on success.
bool cuda_vecadd_init(int n);
// Run vector-add using pre-allocated device buffers. Returns true on success.
bool cuda_vecadd_run(const float* a, const float* b, float* c, int n);
// Free any persistent device buffers. Safe to call even if init failed or wasn't called.
void cuda_vecadd_destroy();

// GPU physics API: attempt to run the physics step on the GPU.
// init: allocate any persistent buffers for n particles (returns true on success)
bool cuda_physics_init(int n);
// run: runs one physics timestep. Host provides arrays for particle positions and velocities
// in SoA form (px, py, pvx, pvy). ballState is a float[5] buffer: {circleX, circleY, circleVelX, circleVelY, circleSpin}.
// On success this updates the particle arrays in-place and writes back updated ballState. Returns true on success.
bool cuda_physics_run(float* px, float* py, float* pvx, float* pvy, int n, float* ballState, float dt);
// destroy: free any persistent buffers
void cuda_physics_destroy();

#ifdef __cplusplus
}
#endif

// CPU fallback (C++ linkage)
void cpu_vecadd(const float* a, const float* b, float* c, int n);

#endif // PICKLEBALL_CUDA_KERNELS_H
