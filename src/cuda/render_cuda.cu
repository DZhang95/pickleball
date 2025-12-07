#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include "render_cuda.h"
#include "cuda_kernels.h"
#include <stdint.h>

#if defined(__linux__)
#include <GL/gl.h>
#include <GL/glx.h>
#elif defined(__APPLE__)
#include <OpenGL/gl.h>
#include <OpenGL/CGLCurrent.h>
#elif defined(_WIN32)
#include <Windows.h>
#include <GL/gl.h>
#include <GL/wglew.h>
#endif

static cudaGraphicsResource_t g_cuda_resource = nullptr;
static unsigned int g_vbo = 0;
static int g_maxInstances = 0;

extern "C" {

bool cuda_render_register_instance_vbo(unsigned int vbo, int maxInstances) {
    if (vbo == 0 || maxInstances <= 0) {
        std::cerr << "cuda_render_register_instance_vbo: invalid args" << std::endl;
        return false;
    }

    // If already registered, unregister first
    if (g_cuda_resource) {
        cudaGraphicsUnregisterResource(g_cuda_resource);
        g_cuda_resource = nullptr;
    }
    
    // Basic diagnostic: ensure there is a current GL context. On many remote
    // setups (SSH X-forwarding) the GL context is indirect and CUDA-GL
    // interop will fail. Check platform-specific APIs when available.
#if defined(__linux__)
    GLXContext ctx = glXGetCurrentContext();
    if (!ctx) {
        const char* disp = getenv("DISPLAY");
        std::cerr << "cuda_render_register_instance_vbo: no current GLX context found. DISPLAY=" << (disp?disp:"(null)") << std::endl;
        std::cerr << "This likely means you're using X forwarding or no active GPU-backed GL context; CUDA-OpenGL interop requires a direct GPU-backed GL context." << std::endl;
        return false;
    }
#elif defined(__APPLE__)
    CGLContextObj cgl = CGLGetCurrentContext();
    if (!cgl) {
        std::cerr << "cuda_render_register_instance_vbo: no current CGL context found." << std::endl;
        return false;
    }
#elif defined(_WIN32)
    HGLRC rc = wglGetCurrentContext();
    if (!rc) {
        std::cerr << "cuda_render_register_instance_vbo: no current WGL context found." << std::endl;
        return false;
    }
#endif

    // Try to pick a CUDA device if none selected. This is a best-effort step
    // to make registration more robust on systems with multiple GPUs.
    int devCount = 0;
    cudaError_t cerr = cudaGetDeviceCount(&devCount);
    if (cerr != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(cerr) << std::endl;
        // continue; let cudaGraphicsGLRegisterBuffer report the error
    } else if (devCount == 0) {
        std::cerr << "No CUDA devices found on the system." << std::endl;
        return false;
    } else {
        // If no device is set, try to set device 0
        int curDev = 0;
        cudaError_t e = cudaGetDevice(&curDev);
        if (e != cudaSuccess) {
            // select device 0 as a reasonable default
            cudaError_t se = cudaSetDevice(0);
            if (se != cudaSuccess) {
                std::cerr << "cudaSetDevice(0) failed: " << cudaGetErrorString(se) << std::endl;
                // continue and let register fail with a clearer message
            }
        }
    }
    // Before attempting registration, check for obvious software renderers
    // (llvmpipe, softpipe) which do not support CUDA-OpenGL interop. If
    // detected, return a clear error rather than invoking cudaGraphicsGLRegisterBuffer.
#if defined(__linux__) || defined(_WIN32) || defined(__APPLE__)
    const GLubyte* rendererStr = glGetString(GL_RENDERER);
    const char* rendererC = rendererStr ? (const char*)rendererStr : "(null)";
    // Lowercase copy for simple substring checks
    std::string rendLower;
    for (const char* p = rendererC; *p; ++p) rendLower.push_back((char)tolower(*p));
    if (rendLower.find("llvmpipe") != std::string::npos || rendLower.find("softpipe") != std::string::npos || rendLower.find("software") != std::string::npos) {
        std::cerr << "cuda_render_register_instance_vbo: detected software GL renderer '" << rendererC << "' which does not support CUDA-OpenGL interop." << std::endl;
        std::cerr << "Run on a GPU-backed GL context (local GPU, VirtualGL, or use a machine with a proper GPU driver) to enable CUDA rendering." << std::endl;
        return false;
    }
#endif

    cudaError_t err = cudaGraphicsGLRegisterBuffer(&g_cuda_resource, vbo, cudaGraphicsRegisterFlagsNone);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorString(err) << std::endl;
#if defined(__linux__) || defined(_WIN32) || defined(__APPLE__)
        // Provide extra GL diagnostics to help debug context/driver issues
        const GLubyte* vendor = glGetString(GL_VENDOR);
        const GLubyte* renderer = glGetString(GL_RENDERER);
        const GLubyte* version = glGetString(GL_VERSION);
        std::cerr << "GL_VENDOR=" << (vendor? (const char*)vendor : "(null)") << "\n";
        std::cerr << "GL_RENDERER=" << (renderer? (const char*)renderer : "(null)") << "\n";
        std::cerr << "GL_VERSION=" << (version? (const char*)version : "(null)") << "\n";
#endif
        g_cuda_resource = nullptr;
        return false;
    }

    g_vbo = vbo;
    g_maxInstances = maxInstances;
    return true;
}

bool cuda_render_frame_from_host(const float* px, const float* py, int n) {
    if (!g_cuda_resource) {
        std::cerr << "cuda_render_frame_from_host: no registered resource" << std::endl;
        return false;
    }
    if (n <= 0 || n > g_maxInstances) {
        std::cerr << "cuda_render_frame_from_host: invalid instance count " << n << " (max=" << g_maxInstances << ")" << std::endl;
        return false;
    }

    cudaError_t err = cudaGraphicsMapResources(1, &g_cuda_resource, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsMapResources failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    void* devPtr = nullptr;
    size_t mappedSize = 0;
    err = cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedSize, g_cuda_resource);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsResourceGetMappedPointer failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &g_cuda_resource, 0);
        return false;
    }

    size_t needed = sizeof(float) * 2 * (size_t)n;
    if (mappedSize < needed) {
        std::cerr << "cuda_render_frame_from_host: mapped buffer too small (" << mappedSize << " < " << needed << ")" << std::endl;
        cudaGraphicsUnmapResources(1, &g_cuda_resource, 0);
        return false;
    }

    // Interleave px/py into a temporary host buffer and copy into the mapped VBO
    std::vector<float> interleaved;
    interleaved.resize(2 * n);
    for (int i = 0; i < n; ++i) {
        interleaved[2*i + 0] = px[i];
        interleaved[2*i + 1] = py[i];
    }

    err = cudaMemcpy(devPtr, interleaved.data(), needed, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "cudaMemcpy to mapped VBO failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &g_cuda_resource, 0);
        return false;
    }

    err = cudaGraphicsUnmapResources(1, &g_cuda_resource, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsUnmapResources failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

// Kernel: read device particle arrays and write interleaved NDC positions into mapped VBO
__global__ void kernel_write_instances(float* mapped, const float* d_px, const float* d_py, int n, float world_cx, float world_cy, float ndc_scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = (d_px[i] - world_cx) * ndc_scale;
    float y = (d_py[i] - world_cy) * ndc_scale;
    // write interleaved
    mapped[2 * i + 0] = x;
    mapped[2 * i + 1] = y;
}

bool cuda_render_frame_from_device(int n, float world_cx, float world_cy, float ndc_scale) {
    if (!g_cuda_resource) {
        std::cerr << "cuda_render_frame_from_device: no registered resource" << std::endl;
        return false;
    }
    if (n <= 0 || n > g_maxInstances) {
        std::cerr << "cuda_render_frame_from_device: invalid instance count " << n << " (max=" << g_maxInstances << ")" << std::endl;
        return false;
    }

    // Obtain device pointers for particle arrays from physics module
    float* d_px = nullptr;
    float* d_py = nullptr;
    int dev_n = 0;
    if (!cuda_physics_get_device_ptrs(&d_px, &d_py, &dev_n)) {
        std::cerr << "cuda_render_frame_from_device: failed to get device particle pointers from physics module" << std::endl;
        return false;
    }
    if (!d_px || !d_py || dev_n < n) {
        std::cerr << "cuda_render_frame_from_device: device particle buffers not available or too small (dev_n=" << dev_n << ")" << std::endl;
        return false;
    }

    cudaError_t err = cudaGraphicsMapResources(1, &g_cuda_resource, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsMapResources failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    void* devPtr = nullptr;
    size_t mappedSize = 0;
    err = cudaGraphicsResourceGetMappedPointer(&devPtr, &mappedSize, g_cuda_resource);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsResourceGetMappedPointer failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &g_cuda_resource, 0);
        return false;
    }

    size_t needed = sizeof(float) * 2 * (size_t)n;
    if (mappedSize < needed) {
        std::cerr << "cuda_render_frame_from_device: mapped buffer too small (" << mappedSize << " < " << needed << ")" << std::endl;
        cudaGraphicsUnmapResources(1, &g_cuda_resource, 0);
        return false;
    }

    float* mapped_f = reinterpret_cast<float*>(devPtr);

    // Launch kernel to write instances directly into mapped VBO
    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    kernel_write_instances<<<blocks, threads>>>(mapped_f, d_px, d_py, n, world_cx, world_cy, ndc_scale);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "kernel_write_instances launch failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &g_cuda_resource, 0);
        return false;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize after kernel_write_instances failed: " << cudaGetErrorString(err) << std::endl;
        cudaGraphicsUnmapResources(1, &g_cuda_resource, 0);
        return false;
    }

    err = cudaGraphicsUnmapResources(1, &g_cuda_resource, 0);
    if (err != cudaSuccess) {
        std::cerr << "cudaGraphicsUnmapResources failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }

    return true;
}

// Offscreen image buffer (RGBA8)
static unsigned char* g_img_dev = nullptr;
static int g_img_w = 0;
static int g_img_h = 0;

__global__ void kernel_clear_image(unsigned char* img, int w, int h, unsigned char r, unsigned char g, unsigned char b, unsigned char a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = w * h;
    if (idx >= total) return;
    int off = idx * 4;
    img[off + 0] = r;
    img[off + 1] = g;
    img[off + 2] = b;
    img[off + 3] = a;
}

// Rasterize simple filled circles for each particle into RGBA8 image.
__global__ void kernel_rasterize_particles(unsigned char* img, int w, int h, const float* d_px, const float* d_py, int n, float world_cx, float world_cy, float ndc_scale, float particle_radius_px) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= n) return;
    float ndc_x = (d_px[pid] - world_cx) * ndc_scale;
    float ndc_y = (d_py[pid] - world_cy) * ndc_scale;
    float fx = (ndc_x * 0.5f + 0.5f) * (float)w;
    float fy = (ndc_y * 0.5f + 0.5f) * (float)h;

    int cx = (int)roundf(fx);
    int cy = (int)roundf(fy);

    int rpx = (int)ceilf(particle_radius_px);
    int x0 = max(0, cx - rpx);
    int x1 = min(w-1, cx + rpx);
    int y0 = max(0, cy - rpx);
    int y1 = min(h-1, cy + rpx);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            float dx = (float)x - fx;
            float dy = (float)y - fy;
            float dist2 = dx*dx + dy*dy;
            if (dist2 <= particle_radius_px * particle_radius_px) {
                int idx = (y * w + x) * 4;
                // overwrite pixel with white particle color; no blending for simplicity
                img[idx + 0] = 0xFF;
                img[idx + 1] = 0xFF;
                img[idx + 2] = 0xFF;
                img[idx + 3] = 0xFF;
            }
        }
    }
}

bool cuda_render_alloc_offscreen(int width, int height) {
    if (width <= 0 || height <= 0) return false;
    if (g_img_dev) cudaFree(g_img_dev);
    size_t sz = (size_t)width * (size_t)height * 4;
    cudaError_t err = cudaMalloc((void**)&g_img_dev, sz);
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc offscreen image failed: " << cudaGetErrorString(err) << std::endl;
        g_img_dev = nullptr;
        return false;
    }
    g_img_w = width;
    g_img_h = height;
    return true;
}

bool cuda_render_render_to_host(unsigned char* hostRGBA, int width, int height, int n, float world_cx, float world_cy, float ndc_scale, float particleRadius) {
    if (!g_img_dev || width != g_img_w || height != g_img_h) {
        std::cerr << "cuda_render_render_to_host: offscreen image not allocated or wrong size" << std::endl;
        return false;
    }
    if (!hostRGBA) return false;

    // Clear image to transparent background (so we can blend the particle image
    // over the court/scene and preserve the court color)
    const int total = width * height;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;
    // alpha = 0 => fully transparent background
    kernel_clear_image<<<blocks, threads>>>(g_img_dev, width, height, 25, 25, 25, 0);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { std::cerr << "kernel_clear_image launch failed: " << cudaGetErrorString(err) << std::endl; return false; }

    // Get device particle pointers
    float* d_px = nullptr;
    float* d_py = nullptr;
    int dev_n = 0;
    if (!cuda_physics_get_device_ptrs(&d_px, &d_py, &dev_n)) {
        std::cerr << "cuda_render_render_to_host: failed to get device particle pointers" << std::endl;
        return false;
    }
    if (dev_n < n) {
        std::cerr << "cuda_render_render_to_host: device particle buffer smaller than requested n" << std::endl;
        return false;
    }

    // Compute pixel radius from particleRadius (world units)
    float radius_ndc = particleRadius * ndc_scale;
    float particleRadiusPx = radius_ndc * 0.5f * (float)width; // NDC->pixels: *width/2

    int pthreads = 128;
    int pblocks = (n + pthreads - 1) / pthreads;
    kernel_rasterize_particles<<<pblocks, pthreads>>>(g_img_dev, width, height, d_px, d_py, n, world_cx, world_cy, ndc_scale, particleRadiusPx);
    err = cudaGetLastError();
    if (err != cudaSuccess) { std::cerr << "kernel_rasterize_particles launch failed: " << cudaGetErrorString(err) << std::endl; return false; }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::cerr << "cudaDeviceSynchronize failed after rasterize: " << cudaGetErrorString(err) << std::endl; return false; }

    size_t sz = (size_t)width * (size_t)height * 4;
    err = cudaMemcpy(hostRGBA, g_img_dev, sz, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "cudaMemcpy D2H image failed: " << cudaGetErrorString(err) << std::endl; return false; }

    return true;
}

void cuda_render_free_offscreen() {
    if (g_img_dev) { cudaFree(g_img_dev); g_img_dev = nullptr; }
    g_img_w = g_img_h = 0;
}

void cuda_render_unregister() {
    if (g_cuda_resource) {
        cudaGraphicsUnregisterResource(g_cuda_resource);
        g_cuda_resource = nullptr;
    }
    g_vbo = 0;
    g_maxInstances = 0;
}

} // extern "C"
