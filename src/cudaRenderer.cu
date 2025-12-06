#include <string>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <vector>
#include <mutex>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "util.h"
#include "CycleTimer.h"

// This stores the global constants
struct GlobalConstants {
    SceneName sceneName;
    int numberOfCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;
    int imageWidth;
    int imageHeight;
    float* imageData;
};

__constant__ GlobalConstants cuConstRendererParams;
__constant__ int cuConstNumCirclesNow;

// Read-only lookup tables used to quickly compute noise (needed by advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// Color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];

#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"

// An approximation of the maximum number of circles we can handle at a time due to memory constraints
// Also experiemented to figure out the best balance between fitting the work set into cache and
// having fewer iterations, coresCount * 3 seems good enough?
int maxCircles = 8832; 
// Number of cores on a 2080
int coresCount = 2944;
// Block size
int blockSize = 256; // CUDA block occupancy API says 1024 is best, 1024 seems to make things slightly worse though


// Uniform Grid Parameters ---
#define GRID_THRESHOLD 1024 // Set to above maxCircles to disable tiled rendering
int gridCellsX = 16; // Default to 16x16 grid, but we actually set this dynamically later
int gridCellsY = 16;
int numGridCells = 16 * 16;
int* hostGridCellCircleCounts = nullptr;
int* hostGridCellCircleOffsets = nullptr;
int* hostGridCellCircleIndices = nullptr;
int* deviceGridCellCircleCounts = nullptr;
int* deviceGridCellCircleOffsets = nullptr;
int* deviceGridCellCircleIndices = nullptr;

// Memory allocation tracking ---
// Track sizes of allocations so we know if we can reuse them
int hostGridCellCircleCountsSize = 0;
int hostGridCellCircleOffsetsSize = 0;
int hostGridCellCircleIndicesSize = 0;
int deviceGridCellCircleCountsSize = 0;
int deviceGridCellCircleOffsetsSize = 0;
int deviceGridCellCircleIndicesSize = 0;

// IMPORTANT: Comment out DEBUG to disable print statements, timing info, and cuda error checking
//#define DEBUG
#ifdef DEBUG
#define CUDACHECK(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define DEBUG_PRINTF(x, ...)      do{fprintf(stderr, "line %u: " x "", __LINE__, ##__VA_ARGS__);}while(0)
#else
#define CUDACHECK(ans) ans
#define DEBUG_PRINTF(x, ...)      /* x */
#endif

////////////////////////////////////////////////////////////////////////////////////////
// All cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
// 
// Update positions of fireworks
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = M_PI;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update 
        return;
    }

    // Determine the firework center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // Update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // Firework sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // Compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // Compute distance from fire-work 
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position 
        // Random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // Travel scaled unit length 
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis   
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() { 
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* radius = cuConstRendererParams.radius; 

    float cutOff = 0.5f;
    // Place circle back in center after reaching threshold radisus 
    if (radius[index] > cutOff) { 
        radius[index] = 0.02f; 
    } else { 
        radius[index] += 0.01f; 
    }   
}   


// kernelAdvanceBouncingBalls
// 
// Update the position of the balls
__global__ void kernelAdvanceBouncingBalls() { 
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x; 
   
    if (index >= cuConstRendererParams.numberOfCircles) 
        return; 

    float* velocity = cuConstRendererParams.velocity; 
    float* position = cuConstRendererParams.position; 

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition 
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball 
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball 
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// Move the snowflake animation forward one time step.  Update circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numberOfCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // Load from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // Hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // Add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // Drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // Update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // Update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // If the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // Restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // Store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}

// shadePixelSnowflak -- (CUDA device code)
//
// Shade a pixel for the snowflake scene. The logic here was originally part of the if statement
// in the original shadePixel function from the starter code, but has been moved to its own function
__device__ __inline__ float4
shadePixelSnowflake(float2 pixelCenter, float3 p, int circleIndex, float4 existingColor) {
    float diffX = p.x - pixelCenter.x;
    float diffY = p.y - pixelCenter.y;
    float pixelDist = diffX * diffX + diffY * diffY;
    float rad = cuConstRendererParams.radius[circleIndex];

    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;
    float normPixelDist = sqrt(pixelDist) / rad;
    float3 rgb = lookupColor(normPixelDist);
    float maxAlpha = .6f + .4f * (1.f-p.z);
    maxAlpha = kCircleMaxAlpha * fmax(fmin(maxAlpha, 1.f), 0.f);
    float alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

    float oneMinusAlpha = 1.f - alpha;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;
    return newColor;
}

// shadePixelDefault -- (CUDA device code)
//
// Shade a pixel for the non-snowflake scenes. The logic here was originally part of the else statement
// in the original shadePixel function from the starter code, but has been moved to its own function
__device__ __inline__ float4
shadePixelDefault(float2 pixelCenter, float3 p, int circleIndex, float4 existingColor) {
    int index3 = 3 * circleIndex;
    float3 rgb = *(float3*)&(cuConstRendererParams.color[index3]);
    float alpha = .5f;

    float oneMinusAlpha = 1.f - alpha;
    float4 newColor;
    newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
    newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
    newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
    newColor.w = alpha + existingColor.w;
    return newColor;
}

// Uniform Grid Render Kernel -- (CUDA device code)
//
// Kernel logic for rendering pixels when a uniform grid is being used
// to accelerate rendering. Each thread processes a single pixel, looking
// up which circles are in the same grid cell as the pixel, and only
// shading the pixel with those circles.
__global__ void kernelRenderPixelsUniformGrid(
    int imageWidth, int imageHeight,
    float* __restrict__ imageData, float* __restrict__ position, float* __restrict__ color, float* __restrict__ radius, SceneName sceneName,
    int gridCellsX, int gridCellsY, int* __restrict__ gridCellCircleOffsets, int* __restrict__ gridCellCircleIndices)
{
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPixels = imageWidth * imageHeight;
    if (pixelIdx >= numPixels) return;

    int pixelY = pixelIdx / imageWidth;
    int pixelX = pixelIdx % imageWidth;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));
    float4* imgPtr = (float4*)(&imageData[4 * pixelIdx]);
    float4 existingColor = *imgPtr;

    // Robust clamping for cellX/cellY to avoid out-of-bounds
    int cellX = min(max(int(pixelCenterNorm.x * gridCellsX), 0), gridCellsX - 1);
    int cellY = min(max(int(pixelCenterNorm.y * gridCellsY), 0), gridCellsY - 1);
    int cellIdx = cellY * gridCellsX + cellX;

    // Defensive: check cellIdx is in bounds
    if (cellIdx < 0 || cellIdx >= gridCellsX * gridCellsY) return;

    // Clamp cellIdx+1 to avoid out-of-bounds on the offsets array
    int safeCellIdx1 = min(cellIdx + 1, gridCellsX * gridCellsY);

    int start = gridCellCircleOffsets[cellIdx];
    int end = gridCellCircleOffsets[safeCellIdx1];
    int totalIndices = gridCellCircleOffsets[gridCellsX * gridCellsY];

    // Clamp start/end to totalIndices to avoid out-of-bounds
    if (start > totalIndices) return;
    if (end > totalIndices) end = totalIndices;
    int cellCount = end - start;
    if (cellCount < 0) return;

    // Prefetch circle data into registers for each iteration
    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        for (int i = 0; i < cellCount; i++) {
            int circleIdx = gridCellCircleIndices[start + i];

            // Prefetch all needed data into registers at the start of the loop
            int index3 = 3 * circleIdx;
            float3 p = *(float3*)(&position[index3]);
            float rad = radius[circleIdx];

            float diffX = p.x - pixelCenterNorm.x;
            float diffY = p.y - pixelCenterNorm.y;
            float pixelDist = diffX * diffX + diffY * diffY;
            if (pixelDist > rad * rad)
                continue;
            existingColor = shadePixelSnowflake(pixelCenterNorm, p, circleIdx, existingColor);
        }
    } else {
        for (int i = 0; i < cellCount; i++) {
            int circleIdx = gridCellCircleIndices[start + i];

            // Prefetch all needed data into registers at the start of the loop
            int index3 = 3 * circleIdx;
            float3 p = *(float3*)(&position[index3]);
            float rad = radius[circleIdx];
            float3 rgb = *(float3*)(&color[index3]);

            float diffX = p.x - pixelCenterNorm.x;
            float diffY = p.y - pixelCenterNorm.y;
            float pixelDist = diffX * diffX + diffY * diffY;
            if (pixelDist > rad * rad)
                continue;
            float alpha = .5f;
            float oneMinusAlpha = 1.f - alpha;
            float4 newColor;
            newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
            newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
            newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
            newColor.w = alpha + existingColor.w;
            existingColor = newColor;
        }
    }
    *imgPtr = existingColor;
}

// kernelRenderPixelsOrdered -- (CUDA device code)
//
// Kernel logic for rendering pixels when NOT using a uniform grid to
// accelerate rendering. Each thread processes a single pixel, iterating
// over all circles to shade the pixel. Better for small numbers of circles
// where the overhead of the grid is not worth it.
__global__ void kernelRenderPixelsOrdered(
    int numCircles, int imageWidth, int imageHeight,
    float* __restrict__ imageData, float* __restrict__ position, float* __restrict__ color, float* __restrict__ radius, SceneName sceneName)
{
    int pixelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int numPixels = imageWidth * imageHeight;
    if (pixelIdx >= numPixels) return;

    int pixelY = pixelIdx / imageWidth;
    int pixelX = pixelIdx % imageWidth;
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                         invHeight * (static_cast<float>(pixelY) + 0.5f));
    float4* imgPtr = (float4*)(&imageData[4 * pixelIdx]);
    float4 existingColor = *imgPtr;    

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        for (int circleIdx = 0; circleIdx < numCircles; ++circleIdx) {
            int index3 = 3 * circleIdx;
            float3 p = *(float3*)(&position[index3]);
            float rad = radius[circleIdx]; // load once
            float diffX = p.x - pixelCenterNorm.x;
            float diffY = p.y - pixelCenterNorm.y;
            float pixelDist = diffX * diffX + diffY * diffY;
            if (pixelDist > rad * rad)
                continue;
            existingColor = shadePixelSnowflake(pixelCenterNorm, p, circleIdx, existingColor);
        }
    } else {
        for (int circleIdx = 0; circleIdx < numCircles; ++circleIdx) {
            int index3 = 3 * circleIdx;
            float3 p = *(float3*)(&position[index3]);
            float rad = radius[circleIdx]; // load once
            float diffX = p.x - pixelCenterNorm.x;
            float diffY = p.y - pixelCenterNorm.y;
            float pixelDist = diffX * diffX + diffY * diffY;
            if (pixelDist > rad * rad)
                continue;
            // Load color once per circleIdx
            float3 rgb = *(float3*)(&color[index3]);
            float alpha = .5f;
            float oneMinusAlpha = 1.f - alpha;
            float4 newColor;
            newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
            newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
            newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
            newColor.w = alpha + existingColor.w;
            existingColor = newColor;
        }
    }
    *imgPtr = existingColor;
}

// kernelRenderPixelsBlockPerCell -- (CUDA device code)
//
// Kernel logic for rendering pixels when using a uniform grid to
// accelerate rendering, with each block processing one cell and
// each thread processing one pixel in that cell. This differs from kernelRenderPixelsUniformGrid
// in that it uses shared memory to load the circle indices for the cell
// once per block, rather than each thread loading them from global memory.
__global__ void kernelRenderPixelsBlockPerCell(
    int imageWidth, int imageHeight,
    float* __restrict__ imageData, float* __restrict__ position, float* __restrict__ color, float* __restrict__ radius, SceneName sceneName,
    int gridCellsX, int gridCellsY, int* __restrict__ gridCellCircleOffsets, int* __restrict__ gridCellCircleIndices)
{
    // Each block is assigned to a single cell
    int cellIdx = blockIdx.x;
    int cellY = cellIdx / gridCellsX;
    int cellX = cellIdx % gridCellsX;

    // Compute pixel bounds for this cell
    int minPixelX = (imageWidth * cellX) / gridCellsX;
    int maxPixelX = (imageWidth * (cellX + 1)) / gridCellsX - 1;
    int minPixelY = (imageHeight * cellY) / gridCellsY;
    int maxPixelY = (imageHeight * (cellY + 1)) / gridCellsY - 1;

    int cellWidth = maxPixelX - minPixelX + 1;
    int cellHeight = maxPixelY - minPixelY + 1;
    int pixelsInCell = cellWidth * cellHeight;

    int threadsPerBlock = blockDim.x;

    // Load circle indices for this cell into shared memory (coalesced)
    extern __shared__ int sharedCircleIndices[];
    int start = gridCellCircleOffsets[cellIdx];
    int end = gridCellCircleOffsets[cellIdx + 1];
    int cellCount = end - start;

    // Cooperative loading of circle indices into shared memory
    for (int i = threadIdx.x; i < cellCount; i += threadsPerBlock) {
        sharedCircleIndices[i] = gridCellCircleIndices[start + i];
    }
    __syncthreads();

    // Each thread processes one pixel in the cell (linear index)
    for (int pixelIdxInCell = threadIdx.x; pixelIdxInCell < pixelsInCell; pixelIdxInCell += threadsPerBlock) {
        int localY = pixelIdxInCell / cellWidth;
        int localX = pixelIdxInCell % cellWidth;
        int pixelX = minPixelX + localX;
        int pixelY = minPixelY + localY;
        if (pixelX >= imageWidth || pixelY >= imageHeight) continue;

        int pixelIdx = pixelY * imageWidth + pixelX;
        float invWidth = 1.f / imageWidth;
        float invHeight = 1.f / imageHeight;
        float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                             invHeight * (static_cast<float>(pixelY) + 0.5f));
        float4* imgPtr = (float4*)(&imageData[4 * pixelIdx]);
        float4 existingColor = *imgPtr;

        if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
            for (int i = 0; i < cellCount; i++) {
                int circleIdx = sharedCircleIndices[i];
                int index3 = 3 * circleIdx;
                float3 p = *(float3*)(&position[index3]);
                float rad = radius[circleIdx];

                float diffX = p.x - pixelCenterNorm.x;
                float diffY = p.y - pixelCenterNorm.y;
                float pixelDist = diffX * diffX + diffY * diffY;
                if (pixelDist > rad * rad)
                    continue;
                existingColor = shadePixelSnowflake(pixelCenterNorm, p, circleIdx, existingColor);
            }
        } else {
            for (int i = 0; i < cellCount; i++) {
                int circleIdx = sharedCircleIndices[i];
                int index3 = 3 * circleIdx;
                float3 p = *(float3*)(&position[index3]);
                float rad = radius[circleIdx];
                float3 rgb = *(float3*)(&color[index3]);

                float diffX = p.x - pixelCenterNorm.x;
                float diffY = p.y - pixelCenterNorm.y;
                float pixelDist = diffX * diffX + diffY * diffY;
                if (pixelDist > rad * rad)
                    continue;
                float alpha = .5f;
                float oneMinusAlpha = 1.f - alpha;
                float4 newColor;
                newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
                newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
                newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
                newColor.w = alpha + existingColor.w;
                existingColor = newColor;
            }
        }
        *imgPtr = existingColor;
    }
}

////////////////////////////////////////////////////////////////////////////////////////

// --- Helper to dynamically choose grid size based on scene ---
int chooseGridSize(int imageWidth, int imageHeight, int numCircles, bool smallScene) {
    // Heuristic: grid size proportional to sqrt(numCircles), but not too small or too large
    int minGrid = 8;
    if (smallScene) minGrid = 16;// Use a finer grid for small scenes to improve performance
    int maxGrid = 128;
    int grid = (int)(sqrtf((float)numCircles) * 0.4f);
    // Clamp to reasonable range
    if (grid < minGrid) grid = minGrid;
    if (grid > maxGrid) grid = maxGrid;
    // Clamp to image size
    if (grid > imageWidth) grid = imageWidth;
    if (grid > imageHeight) grid = imageHeight;
    return grid;
}

// --- CUDA kernel for first pass: count circles per cell (parallel, atomic) ---
__global__ void kernelCountCellCircles(
    int numCircles, float* __restrict__ d_position, float* __restrict__ d_radius,
    int gridCellsX, int gridCellsY, int* __restrict__ d_cellCounts)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= numCircles) return;

    float3 p = *(float3*)(&d_position[3 * c]);
    float rad = d_radius[c];

    int minCellX = max(0, min(gridCellsX - 1, int((p.x - rad) * gridCellsX)));
    int maxCellX = max(0, min(gridCellsX - 1, int((p.x + rad) * gridCellsX)));
    int minCellY = max(0, min(gridCellsY - 1, int((p.y - rad) * gridCellsY)));
    int maxCellY = max(0, min(gridCellsY - 1, int((p.y + rad) * gridCellsY)));

    for (int cy = minCellY; cy <= maxCellY; ++cy) {
        for (int cx = minCellX; cx <= maxCellX; ++cx) {
            int cellIdx = cy * gridCellsX + cx;
            atomicAdd(&d_cellCounts[cellIdx], 1);
        }
    }
}

// buildUniformGrid -- (host code)
//
// Build the uniform grid acceleration structure on the host, using
// a two-pass approach. The first pass counts how many circles are in
// each cell (using a CUDA kernel), and the second pass fills in the
// circle indices for each cell (on the host). The resulting arrays
// are copied to the device for use during rendering. Note that the
// second pass is SEQUENTIAL and on the CPU.
void buildUniformGrid(
    int imageWidth, int imageHeight, int numCircles, float* __restrict__ h_position, float* __restrict__ h_radius, dim3 blockDim, bool smallScene)
{
    // Dynamically choose grid size
    int gridSize = chooseGridSize(imageWidth, imageHeight, numCircles, smallScene);
    gridCellsX = gridSize;
    gridCellsY = gridSize;
    numGridCells = gridCellsX * gridCellsY;

    // --- Allocation reuse ---
    // Track the size so we know if we can reuse this on the next iteration
    if (hostGridCellCircleCountsSize < numGridCells) {
        if (hostGridCellCircleCounts) delete[] hostGridCellCircleCounts;
        hostGridCellCircleCounts = new int[numGridCells]();
        hostGridCellCircleCountsSize = numGridCells;
    } else {
        std::fill(hostGridCellCircleCounts, hostGridCellCircleCounts + numGridCells, 0);
    }

    if (hostGridCellCircleOffsetsSize < numGridCells + 1) {
        if (hostGridCellCircleOffsets) delete[] hostGridCellCircleOffsets;
        hostGridCellCircleOffsets = new int[numGridCells + 1];
        hostGridCellCircleOffsetsSize = numGridCells + 1;
    }

    // --- GPU parallel first pass: count circles per cell ---
    int* d_cellCounts = nullptr;
    float* d_position = nullptr;
    float* d_radius = nullptr;
    cudaMalloc(&d_cellCounts, sizeof(int) * numGridCells);
    cudaMemset(d_cellCounts, 0, sizeof(int) * numGridCells);
    cudaMalloc(&d_position, sizeof(float) * 3 * numCircles);
    cudaMalloc(&d_radius, sizeof(float) * numCircles);
    cudaMemcpy(d_position, h_position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(d_radius, h_radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (numCircles + threadsPerBlock - 1) / threadsPerBlock;
    kernelCountCellCircles<<<blocks, threadsPerBlock>>>(
        numCircles, d_position, d_radius, gridCellsX, gridCellsY, d_cellCounts);
    cudaDeviceSynchronize();

    cudaMemcpy(hostGridCellCircleCounts, d_cellCounts, sizeof(int) * numGridCells, cudaMemcpyDeviceToHost);

    cudaFree(d_cellCounts);
    cudaFree(d_position);
    cudaFree(d_radius);

    // Prefix sum (serial on host)
    hostGridCellCircleOffsets[0] = 0;
    for (int i = 0; i < numGridCells; ++i) {
        hostGridCellCircleOffsets[i + 1] = hostGridCellCircleOffsets[i] + hostGridCellCircleCounts[i];
    }
    int totalRefs = hostGridCellCircleOffsets[numGridCells];

    if (hostGridCellCircleIndicesSize < totalRefs) {
        if (hostGridCellCircleIndices) delete[] hostGridCellCircleIndices;
        hostGridCellCircleIndices = new int[totalRefs];
        hostGridCellCircleIndicesSize = totalRefs;
    }

    // Second pass: fill indices
    std::fill(hostGridCellCircleCounts, hostGridCellCircleCounts + numGridCells, 0);
    for (int c = 0; c < numCircles; ++c) {
        float3 p = *(float3*)(&h_position[3 * c]);
        float rad = h_radius[c];

        int minCellX = std::max(0, std::min(gridCellsX - 1, int((p.x - rad) * gridCellsX)));
        int maxCellX = std::max(0, std::min(gridCellsX - 1, int((p.x + rad) * gridCellsX)));
        int minCellY = std::max(0, std::min(gridCellsY - 1, int((p.y - rad) * gridCellsY)));
        int maxCellY = std::max(0, std::min(gridCellsY - 1, int((p.y + rad) * gridCellsY)));

        for (int cy = minCellY; cy <= maxCellY; ++cy) {
            for (int cx = minCellX; cx <= maxCellX; ++cx) {
                int cellIdx = cy * gridCellsX + cx;
                int offset = hostGridCellCircleOffsets[cellIdx] + hostGridCellCircleCounts[cellIdx];
                hostGridCellCircleIndices[offset] = c;
                hostGridCellCircleCounts[cellIdx]++;
            }
        }
    }

    // --- Allocation reuse ---
    if (deviceGridCellCircleCountsSize < numGridCells) {
        if (deviceGridCellCircleCounts) cudaFree(deviceGridCellCircleCounts);
        cudaMalloc(&deviceGridCellCircleCounts, sizeof(int) * numGridCells);
        deviceGridCellCircleCountsSize = numGridCells;
    }
    if (deviceGridCellCircleOffsetsSize < numGridCells + 1) {
        if (deviceGridCellCircleOffsets) cudaFree(deviceGridCellCircleOffsets);
        cudaMalloc(&deviceGridCellCircleOffsets, sizeof(int) * (numGridCells + 1));
        deviceGridCellCircleOffsetsSize = numGridCells + 1;
    }
    if (deviceGridCellCircleIndicesSize < totalRefs) {
        if (deviceGridCellCircleIndices) cudaFree(deviceGridCellCircleIndices);
        cudaMalloc(&deviceGridCellCircleIndices, sizeof(int) * totalRefs);
        deviceGridCellCircleIndicesSize = totalRefs;
    }

    cudaMemcpy(deviceGridCellCircleCounts, hostGridCellCircleCounts, sizeof(int) * numGridCells, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceGridCellCircleOffsets, hostGridCellCircleOffsets, sizeof(int) * (numGridCells + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceGridCellCircleIndices, hostGridCellCircleIndices, sizeof(int) * totalRefs, cudaMemcpyHostToDevice);
}

//// CudaRenderer class methods

CudaRenderer::CudaRenderer() {
    image = NULL;
    numberOfCircles = 0;
    numberOfIterations = 1;
    numberOfCirclesNow = 0;
    numberOfCirclesPerIteration = NULL;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;
    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

// Clean up device memory, checks for validity before deleting
CudaRenderer::~CudaRenderer() {
    if (image) {
        delete image;
    }
    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }
    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
        free(numberOfCirclesPerIteration);
    }

    if (hostGridCellCircleCounts) delete[] hostGridCellCircleCounts;
    if (hostGridCellCircleOffsets) delete[] hostGridCellCircleOffsets;
    if (hostGridCellCircleIndices) delete[] hostGridCellCircleIndices;
    if (deviceGridCellCircleCounts) cudaFree(deviceGridCellCircleCounts);
    if (deviceGridCellCircleOffsets) cudaFree(deviceGridCellCircleOffsets);
    if (deviceGridCellCircleIndices) cudaFree(deviceGridCellCircleIndices);
}

const Image*
CudaRenderer::getImage() {
    DEBUG_PRINTF("Copying image data from device\n");
    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);
    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numberOfCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {
    double startTime = CycleTimer::currentSeconds();
    int deviceCount = 0;
    bool isFastGPU = false;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    DEBUG_PRINTF("---------------------------------------------------------\n");
    DEBUG_PRINTF("Initializing CUDA for CudaRenderer\n");
    DEBUG_PRINTF("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;
        if (name.compare("GeForce RTX 2080") == 0)
        {
            isFastGPU = true;
        }
        DEBUG_PRINTF("Device %d: %s\n", i, deviceProps.name);
        DEBUG_PRINTF("   SMs:        %d\n", deviceProps.multiProcessorCount);
        DEBUG_PRINTF("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        DEBUG_PRINTF("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    DEBUG_PRINTF("---------------------------------------------------------\n");
    if (!isFastGPU)
    {
        DEBUG_PRINTF("WARNING: "
               "You're not running on a fast GPU, please consider using "
               "NVIDIA RTX 2080.\n");
        DEBUG_PRINTF("---------------------------------------------------------\n");
    }
    
    // Setup for multiple iterations here if there's too many circles (aka more than maxCircles)
    CUDACHECK(cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height));
    numberOfCirclesNow = numberOfCircles;
    if (numberOfCircles > maxCircles) {
        DEBUG_PRINTF("WARNING: Attempting to render %d circles, which is more than the recommended maximum of %d\n", numberOfCircles, maxCircles);
        DEBUG_PRINTF("---------------------------------------------------------\n");
        numberOfIterations = numberOfCircles / maxCircles;
        bool leftovers = (numberOfCircles % maxCircles) != 0;
        if (leftovers) {
            numberOfIterations++;
        }
        DEBUG_PRINTF("We need %d iterations to render all circles\n", numberOfIterations);
        numberOfCirclesPerIteration = (int*)malloc(sizeof(int) * numberOfIterations);
        for (int i=0; i < numberOfIterations; i++) {
            numberOfCirclesPerIteration[i] = maxCircles;
        }
        if (leftovers) {
            numberOfCirclesPerIteration[numberOfIterations - 1] = numberOfCircles % maxCircles;
        }
        numberOfCirclesNow = maxCircles;
    }

    // Allocate device arrays for the first chunk
    CUDACHECK(cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCirclesNow));
    CUDACHECK(cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCirclesNow));
    CUDACHECK(cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCirclesNow));
    CUDACHECK(cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCirclesNow));

    // Only copy to device if data has changed (first setup)
    CUDACHECK(cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numberOfCirclesNow, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numberOfCirclesNow, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numberOfCirclesNow, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numberOfCirclesNow, cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpyToSymbol(cuConstNumCirclesNow, &numberOfCirclesNow, sizeof(int)));

    // Set device-side global constants for the first chunk
    GlobalConstants params;
    params.sceneName = sceneName;
    params.numberOfCircles = numberOfCirclesNow;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;
    CUDACHECK(cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants)));

    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    CUDACHECK(cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256));
    CUDACHECK(cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256));
    CUDACHECK(cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256));

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };
    CUDACHECK(cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE));
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    DEBUG_PRINTF("Initial setup time: %.3f ms\n", 1000.f * overallDuration);
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {
    // 256 threads per block is a healthy number
    dim3 blockDim(16, 16, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(blockSize, 1);
    dim3 gridDim((numberOfCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) { 
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>(); 
    }
    cudaDeviceSynchronize();
}

// Helper function to decide whether to use block-per-cell kernel
// Based on grid statistics. In general, if the grid is sparse or has
// high variance in cell counts, pixel-parallel is better.
bool shouldUseBlockPerCellKernel(int numGridCells, int* hostGridCellCircleCounts) {
    int nonEmptyCells = 0;
    int totalCirclesInCells = 0;
    int maxCirclesPerCell = 0;
    for (int i = 0; i < numGridCells; ++i) {
        int count = hostGridCellCircleCounts[i];
        if (count > 0) nonEmptyCells++;
        totalCirclesInCells += count;
        if (count > maxCirclesPerCell) maxCirclesPerCell = count;
    }
    float avgCirclesPerCell = float(totalCirclesInCells) / numGridCells;
    float fracNonEmpty = float(nonEmptyCells) / numGridCells;

    #ifdef DEBUG
    DEBUG_PRINTF("Grid stats: non-empty cells = %d (%.2f%%), avg circles/cell = %.2f, max circles/cell = %d\n",
                 nonEmptyCells, 100.f * fracNonEmpty, avgCirclesPerCell, maxCirclesPerCell);
    #endif
    // Heuristic: use block-per-cell only if most cells are kind of even and have enough work
    float imbalance = float(maxCirclesPerCell) / (avgCirclesPerCell + 1e-5f); // avoid div by zero
    return (fracNonEmpty > 0.5f && avgCirclesPerCell > 4.0f && imbalance < 3.0f);
}

void
CudaRenderer::render() {
    dim3 blockDim(blockSize, 1);
    int circlesOffset = 0;

    #ifdef DEBUG
    double* renderTimes = (double*)calloc(numberOfIterations + 1, sizeof(double));
    double* binTimes = (double*)calloc(numberOfIterations, sizeof(double));
    renderTimes[0] = CycleTimer::currentSeconds();
    #endif    

    // Only do chunking if numberOfIterations > 1
    if (numberOfIterations == 1) {
        // Use uniform grid for large scenes, naive for small
        if (numberOfCirclesNow > GRID_THRESHOLD) {
            #ifdef DEBUG            
            double gridStart = CycleTimer::currentSeconds();
            #endif
            
            // small scene is true if number of iterations is 1
            buildUniformGrid(image->width, image->height, numberOfCirclesNow, position, radius, blockDim, numberOfIterations == 1);
            
            #ifdef DEBUG
            double gridEnd = CycleTimer::currentSeconds();
            binTimes[0] = gridEnd - gridStart;
            #endif

            // Find maxCellCount for shared memory
            int maxCellCount = 0;
            for (int i = 0; i < numGridCells; ++i) {
                int count = hostGridCellCircleCounts[i];
                if (count > maxCellCount) maxCellCount = count;
            }

            // --- Kernel selection based on grid stats ---
            bool useBlockPerCell = shouldUseBlockPerCellKernel(numGridCells, hostGridCellCircleCounts);

            if (useBlockPerCell) {
                // Launch block-per-cell kernel
                dim3 gridDim(numGridCells, 1, 1);
                int threadsPerBlock = 256; // or tune for your cell size
                kernelRenderPixelsBlockPerCell<<<gridDim, threadsPerBlock, maxCellCount * sizeof(int)>>>(
                    image->width, image->height,
                    cudaDeviceImageData, cudaDevicePosition, cudaDeviceColor, cudaDeviceRadius, sceneName,
                    gridCellsX, gridCellsY, deviceGridCellCircleOffsets, deviceGridCellCircleIndices);
                CUDACHECK(cudaDeviceSynchronize());
                #ifdef DEBUG
                DEBUG_PRINTF("Render time: %.3f ms\n", 1000.f * (CycleTimer::currentSeconds() - renderTimes[0]));
                DEBUG_PRINTF("buildUniformGrid time: %.3f ms\n", 1000.f * binTimes[0]);
                DEBUG_PRINTF("Max cell count for shared memory: %d\n", maxCellCount);
                DEBUG_PRINTF("Kernel: block-per-cell\n");
                #endif
            } else {
                dim3 blockDim(blockSize, 1);
                dim3 gridDim((image->width * image->height + blockDim.x - 1) / blockDim.x);
                kernelRenderPixelsUniformGrid<<<gridDim, blockDim>>>(
                    image->width, image->height,
                    cudaDeviceImageData, cudaDevicePosition, cudaDeviceColor, cudaDeviceRadius, sceneName,
                    gridCellsX, gridCellsY, deviceGridCellCircleOffsets, deviceGridCellCircleIndices);
                CUDACHECK(cudaDeviceSynchronize());
                #ifdef DEBUG
                DEBUG_PRINTF("Render time: %.3f ms\n", 1000.f * (CycleTimer::currentSeconds() - renderTimes[0]));
                DEBUG_PRINTF("Kernel: pixel-parallel (uniform grid)\n");
                #endif
            }
        } else {
            dim3 blockDim(blockSize, 1);
            dim3 gridDim((image->width * image->height + blockDim.x - 1) / blockDim.x);
            kernelRenderPixelsOrdered<<<gridDim, blockDim>>>(
                numberOfCirclesNow, image->width, image->height,
                cudaDeviceImageData, cudaDevicePosition, cudaDeviceColor, cudaDeviceRadius, sceneName);
            CUDACHECK(cudaDeviceSynchronize());
            #ifdef DEBUG
            DEBUG_PRINTF("Render time: %.3f ms\n", 1000.f * (CycleTimer::currentSeconds() - renderTimes[0]));
            DEBUG_PRINTF("Kernel: pixel-parallel (ordered)\n");
            #endif
        }
        #ifdef DEBUG
        free(renderTimes);
        free(binTimes);
        #endif     
        return;
    }

    // Multi-chunk: first chunk already allocated/copied in setup()
    {
        if (numberOfCirclesNow > GRID_THRESHOLD) {
            #ifdef DEBUG            
            double gridStart = CycleTimer::currentSeconds();
            #endif
            
            buildUniformGrid(image->width, image->height, numberOfCirclesNow, position, radius, blockDim, numberOfIterations == 1);
            
            #ifdef DEBUG
            double gridEnd = CycleTimer::currentSeconds();
            binTimes[0] = gridEnd - gridStart;
            #endif

            // --- Begin: Find maxCellCount for shared memory ---
            int maxCellCount = 0;
            for (int i = 0; i < numGridCells; ++i) {
                int count = hostGridCellCircleCounts[i];
                if (count > maxCellCount) maxCellCount = count;
            }
            // --- End: Find maxCellCount for shared memory ---

            // --- Kernel selection based on grid stats ---
            bool useBlockPerCell = shouldUseBlockPerCellKernel(numGridCells, hostGridCellCircleCounts);

            if (useBlockPerCell) {
                // Launch block-per-cell kernel
                dim3 gridDim(numGridCells, 1, 1);
                int threadsPerBlock = 256; // or tune for your cell size
                kernelRenderPixelsBlockPerCell<<<gridDim, threadsPerBlock, maxCellCount * sizeof(int)>>>(
                    image->width, image->height,
                    cudaDeviceImageData, cudaDevicePosition, cudaDeviceColor, cudaDeviceRadius, sceneName,
                    gridCellsX, gridCellsY, deviceGridCellCircleOffsets, deviceGridCellCircleIndices);
                //CUDACHECK(cudaDeviceSynchronize());
                #ifdef DEBUG
                DEBUG_PRINTF("Render time: %.3f ms\n", 1000.f * (CycleTimer::currentSeconds() - renderTimes[0]));
                DEBUG_PRINTF("buildUniformGrid time: %.3f ms\n", 1000.f * binTimes[0]);
                DEBUG_PRINTF("Max cell count for shared memory: %d\n", maxCellCount);
                DEBUG_PRINTF("Kernel: block-per-cell\n");
                #endif
            } else {
                dim3 blockDim(blockSize, 1);
                dim3 gridDim((image->width * image->height + blockDim.x - 1) / blockDim.x);
                kernelRenderPixelsUniformGrid<<<gridDim, blockDim>>>(
                    image->width, image->height,
                    cudaDeviceImageData, cudaDevicePosition, cudaDeviceColor, cudaDeviceRadius, sceneName,
                    gridCellsX, gridCellsY, deviceGridCellCircleOffsets, deviceGridCellCircleIndices);
                //CUDACHECK(cudaDeviceSynchronize());
                #ifdef DEBUG
                DEBUG_PRINTF("Render time: %.3f ms\n", 1000.f * (CycleTimer::currentSeconds() - renderTimes[0]));
                DEBUG_PRINTF("Kernel: pixel-parallel (uniform grid)\n");
                #endif
            }
        } else {
            dim3 blockDim(blockSize, 1);
            dim3 gridDim((image->width * image->height + blockDim.x - 1) / blockDim.x);
            kernelRenderPixelsOrdered<<<gridDim, blockDim>>>(
                numberOfCirclesNow, image->width, image->height,
                cudaDeviceImageData, cudaDevicePosition, cudaDeviceColor, cudaDeviceRadius, sceneName);
            //CUDACHECK(cudaDeviceSynchronize());
            #ifdef DEBUG
            renderTimes[1] = CycleTimer::currentSeconds();
            DEBUG_PRINTF("Kernel: pixel-parallel (ordered)\n");
            #endif        
        }
    }

    // Subsequent iterations
    for (int iter = 1; iter < numberOfIterations; iter++) {
        circlesOffset += numberOfCirclesNow;
        numberOfCirclesNow = numberOfCirclesPerIteration[iter];

        // Allocate device arrays for this chunk
        CUDACHECK(cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numberOfCirclesNow));
        CUDACHECK(cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numberOfCirclesNow));
        CUDACHECK(cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numberOfCirclesNow));
        CUDACHECK(cudaMalloc(&cudaDeviceRadius, sizeof(float) * numberOfCirclesNow));

        // Only copy to device if data has changed (for this chunk)
        CUDACHECK(cudaMemcpy(cudaDevicePosition, &position[3 * circlesOffset], sizeof(float) * 3 * numberOfCirclesNow, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(cudaDeviceVelocity, &velocity[3 * circlesOffset], sizeof(float) * 3 * numberOfCirclesNow, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(cudaDeviceColor, &color[3 * circlesOffset], sizeof(float) * 3 * numberOfCirclesNow, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpy(cudaDeviceRadius, &radius[circlesOffset], sizeof(float) * numberOfCirclesNow, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemcpyToSymbol(cuConstNumCirclesNow, &numberOfCirclesNow, sizeof(int)));

        // Update device-side global constants for this chunk
        GlobalConstants params;
        params.sceneName = sceneName;
        params.numberOfCircles = numberOfCirclesNow;
        params.imageWidth = image->width;
        params.imageHeight = image->height;
        params.position = cudaDevicePosition;
        params.velocity = cudaDeviceVelocity;
        params.color = cudaDeviceColor;
        params.radius = cudaDeviceRadius;
        params.imageData = cudaDeviceImageData;
        CUDACHECK(cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants)));

        if (numberOfCirclesNow > GRID_THRESHOLD) {
            #ifdef DEBUG
            double gridStart = CycleTimer::currentSeconds();
            #endif

            buildUniformGrid(image->width, image->height, numberOfCirclesNow, &position[3 * circlesOffset], &radius[circlesOffset], blockDim, numberOfIterations == 1);

            #ifdef DEBUG
            double gridEnd = CycleTimer::currentSeconds();
            binTimes[iter] = gridEnd - gridStart;
            #endif

            // --- Begin: Find maxCellCount for shared memory ---
            int maxCellCount = 0;
            for (int i = 0; i < numGridCells; ++i) {
                int count = hostGridCellCircleCounts[i];
                if (count > maxCellCount) maxCellCount = count;
            }
            // --- End: Find maxCellCount for shared memory ---

            // --- Kernel selection based on grid stats ---
            bool useBlockPerCell = shouldUseBlockPerCellKernel(numGridCells, hostGridCellCircleCounts);

            if (useBlockPerCell) {
                // Launch block-per-cell kernel
                dim3 gridDim(numGridCells, 1, 1);
                int threadsPerBlock = 256; // or tune for your cell size
                kernelRenderPixelsBlockPerCell<<<gridDim, threadsPerBlock, maxCellCount * sizeof(int)>>>(
                    image->width, image->height,
                    cudaDeviceImageData, cudaDevicePosition, cudaDeviceColor, cudaDeviceRadius, sceneName,
                    gridCellsX, gridCellsY, deviceGridCellCircleOffsets, deviceGridCellCircleIndices);
                //CUDACHECK(cudaDeviceSynchronize());
                #ifdef DEBUG
                double kernelStart = CycleTimer::currentSeconds();
                DEBUG_PRINTF("KERNEL Render time: %.3f ms\n", 1000.f * (CycleTimer::currentSeconds() - kernelStart));
                DEBUG_PRINTF("Kernel: block-per-cell\n");
                #endif

                #ifdef DEBUG
                renderTimes[iter + 1] = CycleTimer::currentSeconds();
                DEBUG_PRINTF("Render time: %.3f ms\n", 1000.f * (renderTimes[iter + 1] - renderTimes[iter]));
                DEBUG_PRINTF("buildUniformGrid time: %.3f ms\n", 1000.f * binTimes[iter]);
                DEBUG_PRINTF("Max cell count for shared memory: %d\n", maxCellCount);
                #endif
            } else {
                dim3 blockDim(blockSize, 1);
                dim3 gridDim((image->width * image->height + blockDim.x - 1) / blockDim.x);
                kernelRenderPixelsUniformGrid<<<gridDim, blockDim>>>(
                    image->width, image->height,
                    cudaDeviceImageData, cudaDevicePosition, cudaDeviceColor, cudaDeviceRadius, sceneName,
                    gridCellsX, gridCellsY, deviceGridCellCircleOffsets, deviceGridCellCircleIndices);
                //CUDACHECK(cudaDeviceSynchronize());
                #ifdef DEBUG
                renderTimes[iter + 1] = CycleTimer::currentSeconds();
                DEBUG_PRINTF("Kernel: pixel-parallel (uniform grid)\n");
                #endif
            }
        } else {
            dim3 blockDim(blockSize, 1);
            dim3 gridDim((image->width * image->height + blockDim.x - 1) / blockDim.x);
            kernelRenderPixelsOrdered<<<gridDim, blockDim>>>(
                numberOfCirclesNow, image->width, image->height,
                cudaDeviceImageData, cudaDevicePosition, cudaDeviceColor, cudaDeviceRadius, sceneName);
            //CUDACHECK(cudaDeviceSynchronize());
            #ifdef DEBUG
            renderTimes[iter + 1] = CycleTimer::currentSeconds();
            DEBUG_PRINTF("Kernel: pixel-parallel (ordered)\n");
            #endif
        }
    }

    #ifdef DEBUG
    for (int i = 1; i <= numberOfIterations; i++) {
        DEBUG_PRINTF("Render time for iteration %d: %.3f ms\n", i, 1000.f * (renderTimes[i] - renderTimes[i - 1]));
        if (i < numberOfIterations)
            DEBUG_PRINTF("buildUniformGrid time: %.3f ms\n", 1000.f * binTimes[i]);
    }
    int minGridSize = 0, blockSizeOpt = 0;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeOpt, kernelRenderPixelsOrdered, 0, 0);
    DEBUG_PRINTF("Optimal block size for kernelRenderPixelsOrdered: %d\n", blockSizeOpt);
    free(renderTimes);
    free(binTimes);
    #endif
}