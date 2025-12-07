#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Register the instance VBO created by the renderer so the CUDA render helper
// (or stub) can update it. vbo is the GL buffer name. maxInstances is the
// maximum number of particles expected.
bool cuda_render_register_instance_vbo(unsigned int vbo, int maxInstances);

// Update the instance positions from host arrays (px,py) containing n entries.
// Returns true on success. An actual CUDA implementation would copy device
// arrays into the mapped VBO; the stub simply updates via glBufferSubData.
bool cuda_render_frame_from_host(const float* px, const float* py, int n);

// Populate the registered instance VBO directly from device particle arrays.
// world_cx/world_cy and ndc_scale are used to transform world coords to NDC as the CPU path does.
// Returns true on success.
bool cuda_render_frame_from_device(int n, float world_cx, float world_cy, float ndc_scale);

// Offscreen device image path (fallback for systems without CUDA-GL interop).
// Allocate an RGBA8 device image of the given size. Returns true on success.
bool cuda_render_alloc_offscreen(int width, int height);
// Render particles into the device image (using device particle arrays) and copy the
// resulting RGBA8 image into the provided host buffer (size must be width*height*4).
bool cuda_render_render_to_host(unsigned char* hostRGBA, int width, int height, int n, float world_cx, float world_cy, float ndc_scale, float particleRadius);
// Free any offscreen image resources
void cuda_render_free_offscreen();

// Unregister / destroy any CUDA render resources.
void cuda_render_unregister();

#ifdef __cplusplus
}
#endif
