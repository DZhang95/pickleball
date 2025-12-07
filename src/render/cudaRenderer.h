#ifndef __CUDA_RENDERER_H__
#define __CUDA_RENDERER_H__

#ifndef uint
#define uint unsigned int
#endif

#include "circleRenderer.h"


class CudaRenderer : public CircleRenderer {

private:

    Image* image;
    SceneName sceneName;

    int numberOfCircles;
    int numberOfCirclesNow;
    int numberOfIterations;
    int* numberOfCirclesPerIteration; 
    float* position;
    //float* currentPosition;
    float* velocity;
    float* color;
    float* radius;

    float* cudaDevicePosition;
    float* cudaDeviceVelocity;
    float* cudaDeviceColor;
    float* cudaDeviceRadius;
    float* cudaDeviceImageData;
    uint8_t* cudaDeviceCirclesMask;

public:

    CudaRenderer();
    virtual ~CudaRenderer();

    const Image* getImage();

    void setup();

    void loadScene(SceneName name);

    void allocOutputImage(int width, int height);

    void clearImage();

    void advanceAnimation();

    void render();

    void shadePixel(
        float pixelCenterX, float pixelCenterY,
        float px, float py, float pz,
        float* pixelData, 
        int circleIndex);
};


#endif
