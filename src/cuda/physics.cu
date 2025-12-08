#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "cuda_kernels.h"

// Helper to print CUDA errors with context
static inline void printCudaError(const char *where, cudaError_t err) {
    if (err == cudaSuccess) return;
    fprintf(stderr, "CUDA ERROR at %s: %s (code=%d)\n", where, cudaGetErrorString(err), (int)err);
}

// Duplicate a few physics constants to keep the CUDA code self-contained.
static const float BALL_MASS_CU = 0.026f;
static const float AIR_PARCEL_MASS_CU = 0.00000001f;
static const float BALL_RADIUS_CU = 0.185f;
static const float BALL_MOMENT_OF_INERTIA_CU = (2.0f / 5.0f) * BALL_MASS_CU * BALL_RADIUS_CU * BALL_RADIUS_CU;
static const float IMPULSE_SCALE_FACTOR_CU = 0.1f;
static const float AIR_AIR_IMPULSE_SCALE_FACTOR_CU = 0.1f;

// Persistent device buffers
static float *g_px = nullptr;
static float *g_py = nullptr;
static float *g_pvx = nullptr;
static float *g_pvy = nullptr;
static int g_n = 0;

// Accumulators on device for ball impulse and spin
static float *g_impulse_x = nullptr;
static float *g_impulse_y = nullptr;
static float *g_spin_accum = nullptr;

// Kernel: integrate particle positions, handle air-air collisions (naive all-pairs where thread i
// processes pairs (i,j) with j>i), handle simple ball-particle collisions, and accumulate impulse
// contributions for the ball using atomics. This is a first-pass implementation (not highly
// optimized) that uses atomicAdd for shared updates. It should work correctly for modest n and
// lets us move air-air work onto the GPU.
__global__ void kernel_physics_step(
    float* px, float* py, float* pvx, float* pvy,
    int n,
    float ballX, float ballY,
    float ballVelX, float ballVelY, float ballSpin,
    float* d_impulse_x, float* d_impulse_y, float* d_spin_accum,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Load current state for particle i
    float xi = px[i];
    float yi = py[i];
    float vxi = pvx[i];
    float vyi = pvy[i];

    // Integrate position tentatively
    float newx = xi + vxi * dt;
    float newy = yi + vyi * dt;

    const float airRadius = 0.005f; // match renderer assumption
    const float minDistance = 2.0f * airRadius;

    // --- Naive all-pairs (each thread handles j>i) to compute air-air collisions ---
    for (int j = i + 1; j < n; ++j) {
        // read peer j (note: these may be concurrently updated by other threads; we accept that)
        float xj = px[j];
        float yj = py[j];
        float vxj = pvx[j];
        float vyj = pvy[j];

        float dx = xj - newx;
        float dy = yj - newy;
        float dist = sqrtf(dx * dx + dy * dy);
        if (dist < minDistance && dist > 1e-6f) {
            float nx = dx / dist;
            float ny = dy / dist;

            // separation: move both particles half the overlap (apply for j atomically, i locally)
            float overlap = minDistance - dist;
            float sepX = nx * overlap * 0.5f;
            float sepY = ny * overlap * 0.5f;

            // Apply separation: j gets atomic add, i we modify local newx/newy
            atomicAdd(&px[j], sepX);
            atomicAdd(&py[j], sepY);
            newx -= sepX;
            newy -= sepY;

            // Compute impulse using the same heuristic as CPU path
            float relVelX = vxj - vxi;
            float relVelY = vyj - vyi;
            float v_rel_dot_n = relVelX * nx + relVelY * ny;

            // Only apply impulse when particles are moving towards each other.
            // Use the two-body impulse formula: J = -(1+e) * v_rel_n / (1/m_i + 1/m_j)
            if (v_rel_dot_n < 0.0f) {
                const float e = 0.0f; // restitution
                float invMi = 1.0f / AIR_PARCEL_MASS_CU;
                float invMj = 1.0f / AIR_PARCEL_MASS_CU;
                float Jn = -(1.0f + e) * v_rel_dot_n / (invMi + invMj);
                Jn *= AIR_AIR_IMPULSE_SCALE_FACTOR_CU;

                float impulseX = Jn * nx;
                float impulseY = Jn * ny;

                // Apply to particle i (local)
                vxi += impulseX * invMi;
                vyi += impulseY * invMi;

                // Apply to particle j (atomic)
                float delta_vx_j = -impulseX * invMj;
                float delta_vy_j = -impulseY * invMj;
                atomicAdd(&pvx[j], delta_vx_j);
                atomicAdd(&pvy[j], delta_vy_j);
            }
        }
    }

    // --- Ball-particle collision (similar to previous implementation) ---
    float dxb = newx - ballX;
    float dyb = newy - ballY;
    float distb = sqrtf(dxb * dxb + dyb * dyb);
    float ballMinDistance = BALL_RADIUS_CU + airRadius;

    if (distb < ballMinDistance && distb > 1e-6f) {
        float nx = dxb / distb;
        float ny = dyb / distb;

        float rx = nx * BALL_RADIUS_CU;
        float ry = ny * BALL_RADIUS_CU;

    // Surface velocity of the ball includes translational velocity and tangential
    // component from spin: surfaceVel = ballVel + omega x r (in 2D simplified)
    float surfaceVelX = ballVelX - ballSpin * ry;
    float surfaceVelY = ballVelY + ballSpin * rx;

    float relVelX = vxi - surfaceVelX;
    float relVelY = vyi - surfaceVelY;

        float v_rel_n = relVelX * nx + relVelY * ny;
        if (v_rel_n < 0.0f) {
            float invMassBall = 1.0f / BALL_MASS_CU;
            float invMassPart = 1.0f / AIR_PARCEL_MASS_CU;

            const float restitution = 0.0f;
            float Jn_mag = -(1.0f + restitution) * v_rel_n / (invMassBall + invMassPart);

            float tx = -ny;
            float ty = nx;
            float v_rel_t = relVelX * tx + relVelY * ty;

            float denom_t = invMassBall + invMassPart + (BALL_RADIUS_CU * BALL_RADIUS_CU) / BALL_MOMENT_OF_INERTIA_CU;
            float Jt_unc = - v_rel_t / denom_t;

            const float mu = 0.05f;
            float Jt = 0.0f;
            if (fabsf(Jt_unc) <= mu * Jn_mag) {
                Jt = Jt_unc;
            } else {
                Jt = (Jt_unc > 0.0f ? 1.0f : -1.0f) * mu * Jn_mag;
            }

            float impulseX = Jn_mag * nx + Jt * tx;
            float impulseY = Jn_mag * ny + Jt * ty;

            impulseX *= IMPULSE_SCALE_FACTOR_CU;
            impulseY *= IMPULSE_SCALE_FACTOR_CU;

            // Update particle velocity: particle should be pushed AWAY from the ball (+J/m)
            vxi += impulseX / AIR_PARCEL_MASS_CU;
            vyi += impulseY / AIR_PARCEL_MASS_CU;

            // small separation
            float overlap = ballMinDistance - distb;
            float sepX = nx * overlap * 0.5f;
            float sepY = ny * overlap * 0.5f;
            newx += sepX;
            newy += sepY;

            // Project to exactly non-penetrating position (small eps) to avoid immediate re-contact
            const float proj_eps = 1e-5f;
            newx = ballX + nx * (ballMinDistance + proj_eps);
            newy = ballY + ny * (ballMinDistance + proj_eps);

            // Clamp the normal component of particle velocity so it's not still moving into the ball
            float surfaceVelX_after = ballVelX - ballSpin * ry;
            float surfaceVelY_after = ballVelY + ballSpin * rx;
            float new_rel_vn = (vxi - surfaceVelX_after) * nx + (vyi - surfaceVelY_after) * ny;
            if (new_rel_vn < 0.0f) {
                vxi -= new_rel_vn * nx;
                vyi -= new_rel_vn * ny;
            }

            float r_cross_J = rx * impulseY - ry * impulseX;
            float spin_change = r_cross_J / BALL_MOMENT_OF_INERTIA_CU;

            // Accumulate negative impulse and spin for the ball (host will add these accumulators)
            // We store -J so that the host-side addition results in applying -J to the ball.
            atomicAdd(d_impulse_x, -impulseX);
            atomicAdd(d_impulse_y, -impulseY);
            atomicAdd(d_spin_accum, -spin_change);
        }
    }

    // Write back updated state for particle i
    px[i] = newx;
    py[i] = newy;
    pvx[i] = vxi;
    pvy[i] = vyi;
}

extern "C" bool cuda_physics_init(int n) {
    // Quick device availability check
    int deviceCount = 0;
    cudaError_t cerr = cudaGetDeviceCount(&deviceCount);
    if (cerr != cudaSuccess) {
        printCudaError("cudaGetDeviceCount", cerr);
        return false;
    }
    if (deviceCount <= 0) {
        fprintf(stderr, "CUDA INFO: no CUDA devices found (deviceCount=%d)\n", deviceCount);
        return false;
    }
    // Use device 0 by default
    cerr = cudaSetDevice(0);
    if (cerr != cudaSuccess) {
        printCudaError("cudaSetDevice(0)", cerr);
        return false;
    }
    if (n <= 0) return false;
    if (g_n >= n && g_px && g_py && g_pvx && g_pvy) {
        return true;
    }

    // free previous
    if (g_px) cudaFree(g_px);
    if (g_py) cudaFree(g_py);
    if (g_pvx) cudaFree(g_pvx);
    if (g_pvy) cudaFree(g_pvy);
    if (g_impulse_x) cudaFree(g_impulse_x);
    if (g_impulse_y) cudaFree(g_impulse_y);
    if (g_spin_accum) cudaFree(g_spin_accum);

    cudaError_t err = cudaMalloc((void**)&g_px, sizeof(float) * n);
    if (err != cudaSuccess) { printCudaError("cudaMalloc g_px", err); return false; }
    err = cudaMalloc((void**)&g_py, sizeof(float) * n);
    if (err != cudaSuccess) { printCudaError("cudaMalloc g_py", err); cudaFree(g_px); g_px = nullptr; return false; }
    err = cudaMalloc((void**)&g_pvx, sizeof(float) * n);
    if (err != cudaSuccess) { printCudaError("cudaMalloc g_pvx", err); cudaFree(g_px); cudaFree(g_py); g_px = g_py = nullptr; return false; }
    err = cudaMalloc((void**)&g_pvy, sizeof(float) * n);
    if (err != cudaSuccess) { printCudaError("cudaMalloc g_pvy", err); cudaFree(g_px); cudaFree(g_py); cudaFree(g_pvx); g_px = g_py = g_pvx = nullptr; return false; }

    err = cudaMalloc((void**)&g_impulse_x, sizeof(float));
    if (err != cudaSuccess) { printCudaError("cudaMalloc g_impulse_x", err); cudaFree(g_px); cudaFree(g_py); cudaFree(g_pvx); cudaFree(g_pvy); return false; }
    err = cudaMalloc((void**)&g_impulse_y, sizeof(float));
    if (err != cudaSuccess) { printCudaError("cudaMalloc g_impulse_y", err); cudaFree(g_impulse_x); cudaFree(g_px); cudaFree(g_py); cudaFree(g_pvx); cudaFree(g_pvy); return false; }
    err = cudaMalloc((void**)&g_spin_accum, sizeof(float));
    if (err != cudaSuccess) { printCudaError("cudaMalloc g_spin_accum", err); cudaFree(g_impulse_y); cudaFree(g_impulse_x); cudaFree(g_px); cudaFree(g_py); cudaFree(g_pvx); cudaFree(g_pvy); return false; }

    g_n = n;
    return true;
}

extern "C" bool cuda_physics_run(float* px, float* py, float* pvx, float* pvy, int n, float* ballState, float dt) {
    if (!g_px || !g_py || !g_pvx || !g_pvy || g_n < n) {
        fprintf(stderr, "CUDA RUN: device buffers not initialized or too small (g_n=%d, n=%d)\n", g_n, n);
        return false;
    }
    if (n <= 0) return true;

    // Copy host arrays to device
    cudaError_t err = cudaMemcpy(g_px, px, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_px", err); return false; }
    err = cudaMemcpy(g_py, py, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_py", err); return false; }
    err = cudaMemcpy(g_pvx, pvx, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_pvx", err); return false; }
    err = cudaMemcpy(g_pvy, pvy, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_pvy", err); return false; }

    // Zero accumulators
    float zero = 0.0f;
    err = cudaMemcpy(g_impulse_x, &zero, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_impulse_x", err); return false; }
    err = cudaMemcpy(g_impulse_y, &zero, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_impulse_y", err); return false; }
    err = cudaMemcpy(g_spin_accum, &zero, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_spin_accum", err); return false; }

    // Launch kernel
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float ballX = ballState[0];
    float ballY = ballState[1];
    float ballVelX = ballState[2];
    float ballVelY = ballState[3];
    float ballSpin = ballState[4];

    kernel_physics_step<<<blocks, threads>>>(g_px, g_py, g_pvx, g_pvy, n, ballX, ballY, ballVelX, ballVelY, ballSpin, g_impulse_x, g_impulse_y, g_spin_accum, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess) { printCudaError("kernel_physics_step launch", err); return false; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printCudaError("cudaDeviceSynchronize after kernel", err); return false; }

    // Copy back particle arrays
    err = cudaMemcpy(px, g_px, sizeof(float) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy D2H g_px", err); return false; }
    err = cudaMemcpy(py, g_py, sizeof(float) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy D2H g_py", err); return false; }
    err = cudaMemcpy(pvx, g_pvx, sizeof(float) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy D2H g_pvx", err); return false; }
    err = cudaMemcpy(pvy, g_pvy, sizeof(float) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy D2H g_pvy", err); return false; }

    // Copy back accumulators
    float impx = 0.0f, impy = 0.0f, spin = 0.0f;
    err = cudaMemcpy(&impx, g_impulse_x, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy D2H g_impulse_x", err); return false; }
    err = cudaMemcpy(&impy, g_impulse_y, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy D2H g_impulse_y", err); return false; }
    err = cudaMemcpy(&spin, g_spin_accum, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy D2H g_spin_accum", err); return false; }

    // Diagnostic print: show accumulated impulse and spin change from kernel
#ifdef DEBUG
    fprintf(stderr, "CUDA accumulators: impx=%f, impy=%f, dspin=%f\n", impx, impy, spin);
#endif

    // Apply accumulators to ball state (host will still add magnus later)
    // ballState layout: [circleX, circleY, circleVelX, circleVelY, circleSpin]
    ballState[2] += impx / BALL_MASS_CU;
    ballState[3] += impy / BALL_MASS_CU;
    ballState[4] += spin;

    return true;
}

// Device-only variant: runs the physics kernel but does not copy particle arrays back to host.
// This is useful when the renderer will consume particle positions directly from device memory.
extern "C" bool cuda_physics_run_device(float* px, float* py, float* pvx, float* pvy, int n, float* ballState, float dt) {
    if (!g_px || !g_py || !g_pvx || !g_pvy || g_n < n) {
        fprintf(stderr, "CUDA RUN DEVICE: device buffers not initialized or too small (g_n=%d, n=%d)\n", g_n, n);
        return false;
    }
    if (n <= 0) return true;

    // Copy host arrays to device (we still need to provide initial state)
    cudaError_t err = cudaMemcpy(g_px, px, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_px", err); return false; }
    err = cudaMemcpy(g_py, py, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_py", err); return false; }
    err = cudaMemcpy(g_pvx, pvx, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_pvx", err); return false; }
    err = cudaMemcpy(g_pvy, pvy, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_pvy", err); return false; }

    // Zero accumulators
    float zero = 0.0f;
    err = cudaMemcpy(g_impulse_x, &zero, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_impulse_x", err); return false; }
    err = cudaMemcpy(g_impulse_y, &zero, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_impulse_y", err); return false; }
    err = cudaMemcpy(g_spin_accum, &zero, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy H2D g_spin_accum", err); return false; }

    // Launch kernel
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    float ballX = ballState[0];
    float ballY = ballState[1];
    float ballVelX = ballState[2];
    float ballVelY = ballState[3];
    float ballSpin = ballState[4];

    kernel_physics_step<<<blocks, threads>>>(g_px, g_py, g_pvx, g_pvy, n, ballX, ballY, ballVelX, ballVelY, ballSpin, g_impulse_x, g_impulse_y, g_spin_accum, dt);
    err = cudaGetLastError();
    if (err != cudaSuccess) { printCudaError("kernel_physics_step launch", err); return false; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printCudaError("cudaDeviceSynchronize after kernel", err); return false; }

    // Copy back accumulators only
    float impx = 0.0f, impy = 0.0f, spin = 0.0f;
    err = cudaMemcpy(&impx, g_impulse_x, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy D2H g_impulse_x", err); return false; }
    err = cudaMemcpy(&impy, g_impulse_y, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy D2H g_impulse_y", err); return false; }
    err = cudaMemcpy(&spin, g_spin_accum, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printCudaError("cudaMemcpy D2H g_spin_accum", err); return false; }

    // Diagnostic print: show accumulated impulse and spin change from kernel
#ifdef DEBUG
    fprintf(stderr, "CUDA accumulators (device-only): impx=%f, impy=%f, dspin=%f\n", impx, impy, spin);
#endif

    // Apply accumulators to ball state
    ballState[2] += impx / BALL_MASS_CU;
    ballState[3] += impy / BALL_MASS_CU;
    ballState[4] += spin;

    return true;
}

// Provide access to device buffers for other translation units (non-exported static pointers)
extern "C" bool cuda_physics_get_device_ptrs(float** out_px, float** out_py, int* out_n) {
    if (!out_px || !out_py || !out_n) return false;
    if (!g_px || !g_py || g_n <= 0) return false;
    *out_px = g_px;
    *out_py = g_py;
    *out_n = g_n;
    return true;
}

extern "C" void cuda_physics_destroy() {
    if (g_px) { cudaFree(g_px); g_px = nullptr; }
    if (g_py) { cudaFree(g_py); g_py = nullptr; }
    if (g_pvx) { cudaFree(g_pvx); g_pvx = nullptr; }
    if (g_pvy) { cudaFree(g_pvy); g_pvy = nullptr; }
    if (g_impulse_x) { cudaFree(g_impulse_x); g_impulse_x = nullptr; }
    if (g_impulse_y) { cudaFree(g_impulse_y); g_impulse_y = nullptr; }
    if (g_spin_accum) { cudaFree(g_spin_accum); g_spin_accum = nullptr; }
    g_n = 0;
}
